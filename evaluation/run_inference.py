"""
run_inference.py
================
Generate standard-length captions for the test split.

Per Katelyn's instructions, the research question is:
  Does training on enhanced (long) captions produce BETTER STANDARD-LENGTH
  outputs than training on original short captions?

So all models generate short captions (max_new_tokens=64), and we compare:
  - CNN+LSTM trained on jxie/flickr8k (original)       → results/<model>/flickr8k/
  - CNN+LSTM trained on runjiazeng/flickr8k-enhanced    → results/<model>/flickr8k-enhanced/
  - BLIP fine-tuned on flickr8k-enhanced                → results/blip-base/flickr8k-enhanced/
  - ViT-GPT2 fine-tuned on flickr8k-enhanced           → results/vit-gpt2/flickr8k-enhanced/

All outputs evaluated against the ORIGINAL short references for a fair comparison.

Usage
-----
  # Fine-tuned transformer models (enhanced-trained)
  python run_inference.py --model blip-base
  python run_inference.py --model vit-gpt2

  # Enhanced-trained CNN+LSTM models
  python run_inference.py --model simple-cnn-lstm
  python run_inference.py --model sureal01-cnn-lstm

  # Original-trained CNN+LSTM baselines (point to new checkpoints)
  python run_inference.py --model simple-cnn-lstm \
      --checkpoint checkpoints/simple-cnn-lstm-original/best.pt \
      --out_folder flickr8k

  python run_inference.py --model sureal01-cnn-lstm \
      --checkpoint checkpoints/sureal01-cnn-lstm-original/best.pt \
      --out_folder flickr8k
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from torchvision import transforms

# ── make repo-level training/ importable ─────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "training"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = ["blip-base", "vit-gpt2", "simple-cnn-lstm", "sureal01-cnn-lstm"]
VARIANTS = {
    "enhanced": {
        "checkpoint_dir_suffix": "",
        "out_folder": "flickr8k-enhanced",
    },
    "baseline": {
        "checkpoint_dir_suffix": "-baseline",
        "out_folder": "flickr8k",
    },
}

# ── standard caption length — intentionally kept short ───────────────────────
# The research question is whether ENHANCED TRAINING improves STANDARD-LENGTH
# outputs. We do NOT want to generate long captions here.
MAX_NEW_TOKENS = 64
CNN_MAX_LEN    = 64


# ═══════════════════════════════════════════════════════════════════════════════
# Model loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_blip(checkpoint_path: str):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    print("  Loading BLIP base model...")
    model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"  Loaded: {checkpoint_path}")

    model.eval().to(DEVICE)
    return model, processor


def load_vit_gpt2(checkpoint_path: str):
    from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

    print("  Loading ViT-GPT2 base model...")
    model           = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer       = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id         = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    model.config.eos_token_id         = tokenizer.eos_token_id

    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"  Loaded: {checkpoint_path}")

    model.eval().to(DEVICE)
    return model, image_processor, tokenizer


def load_cnn_lstm(checkpoint_path: str, model_type: str):
    from cnn_lstm_common import (
        CaptioningModel, EncoderCNN, SimpleCaptionModel,
        SurealCaptionModel, Vocabulary,
    )

    print(f"  Loading {model_type} checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt.get("config", {})

    vocab   = Vocabulary.from_tokens(ckpt["vocab"])
    encoder = EncoderCNN(
        arch=cfg.get("encoder_arch", "resnet18"),
        trainable_layers=cfg.get("trainable_encoder_layers", 0),
        pooled=True,
    )

    emb_dim    = cfg.get("emb_dim",    256)
    hidden_dim = cfg.get("hidden_dim", 256)
    dropout    = cfg.get("dropout",    0.2)

    if model_type == "sureal01-cnn-lstm":
        decoder = SurealCaptionModel(
            vocab_size=len(vocab), feat_dim=encoder.feat_dim,
            emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout,
        )
    else:
        decoder = SimpleCaptionModel(
            vocab_size=len(vocab), feat_dim=encoder.feat_dim,
            emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout,
        )

    model = CaptioningModel(encoder, decoder)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval().to(DEVICE)

    image_size = cfg.get("image_size", 224)
    print(f"  Loaded: {checkpoint_path}  |  vocab={len(vocab)}  image_size={image_size}")
    return model, vocab, image_size


# ═══════════════════════════════════════════════════════════════════════════════
# Inference functions — all generate standard-length outputs
# ═══════════════════════════════════════════════════════════════════════════════

def infer_blip(model, processor, image):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            early_stopping=True,
        )
    return processor.decode(ids[0], skip_special_tokens=True).strip()


def infer_vit_gpt2(model, image_processor, tokenizer, image):
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            pixel_values,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def infer_cnn_lstm(model, vocab, image_size, pil_image):
    """
    Step-by-step greedy decoding — bypasses the broken generate() method.

    The training code's forward() uses teacher-forcing (captions[:, :-1])
    which returns an empty tensor on the first inference step. We drive
    the LSTM cell-by-cell instead.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = model.encoder(img_tensor)
        dec  = model.decoder

        from cnn_lstm_common import SimpleCaptionModel, SurealCaptionModel

        if isinstance(dec, SimpleCaptionModel):
            image_token   = dec.feature_proj(feat).unsqueeze(1)
            _, (h, c)     = dec.lstm(image_token)
            current       = torch.tensor([[vocab.bos_id]], device=DEVICE)
            tokens        = []
            for _ in range(CNN_MAX_LEN):
                emb         = dec.embedding(current)
                out, (h, c) = dec.lstm(emb, (h, c))
                logit       = dec.output(dec.dropout(out[:, 0, :]))
                next_tok    = logit.argmax(dim=-1).item()
                if next_tok == vocab.eos_id:
                    break
                tokens.append(next_tok)
                current = torch.tensor([[next_tok]], device=DEVICE)

        elif isinstance(dec, SurealCaptionModel):
            h       = dec.init_h(feat)
            c       = dec.init_c(feat)
            current = torch.tensor([[vocab.bos_id]], device=DEVICE)
            tokens  = []
            for _ in range(CNN_MAX_LEN):
                emb      = dec.embedding(current[:, 0])
                lstm_in  = dec.input_proj(torch.cat([feat, emb], dim=1))
                h, c     = dec.lstm(lstm_in, (h, c))
                logit    = dec.output(dec.dropout(h))
                next_tok = logit.argmax(dim=-1).item()
                if next_tok == vocab.eos_id:
                    break
                tokens.append(next_tok)
                current = torch.tensor([[next_tok]], device=DEVICE)

        else:
            raise ValueError(f"Unknown decoder type: {type(dec)}")

    return vocab.decode(tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_model(model_name, checkpoint_path, test_rows, valid_ids):
    print(f"\n{'='*60}\n  Model: {model_name}\n  Checkpoint: {checkpoint_path}\n{'='*60}")
    results = {}

    if model_name == "blip-base":
        model, processor = load_blip(checkpoint_path)
        for row in tqdm(test_rows, desc="  Generating"):
            img_id = str(row["image_id"])
            if img_id not in valid_ids:
                continue
            results[img_id] = infer_blip(model, processor, row["image"].convert("RGB"))

    elif model_name == "vit-gpt2":
        model, image_processor, tokenizer = load_vit_gpt2(checkpoint_path)
        for row in tqdm(test_rows, desc="  Generating"):
            img_id = str(row["image_id"])
            if img_id not in valid_ids:
                continue
            results[img_id] = infer_vit_gpt2(model, image_processor, tokenizer, row["image"].convert("RGB"))

    elif model_name in ("simple-cnn-lstm", "sureal01-cnn-lstm"):
        model, vocab, image_size = load_cnn_lstm(checkpoint_path, model_name)
        for row in tqdm(test_rows, desc="  Generating"):
            img_id = str(row["image_id"])
            if img_id not in valid_ids:
                continue
            results[img_id] = infer_cnn_lstm(model, vocab, image_size, row["image"])

    return results


def save_results(model_name, out_folder, captions):
    out_path = ROOT_DIR / "results" / model_name / out_folder / "captions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"  Saved {len(captions)} captions → {out_path}")


def load_test_split(data_source: str, split: str):
    source_path = ROOT_DIR / data_source
    if source_path.exists():
        print(f"Loading test images from local dataset: {source_path} [{split}]...")
        ds = load_from_disk(str(source_path))
        return ds[split]

    print(f"Loading test images from Hugging Face dataset: {data_source} [{split}]...")
    return load_dataset(data_source, split=split)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    parser.add_argument("--variant", choices=["all", "enhanced", "baseline"], default="all",
                        help="Which checkpoint variant(s) to run")
    parser.add_argument("--checkpoint", default="",
                        help="Path to a single checkpoint file. Use with one model and one variant.")
    parser.add_argument("--data_dir", default="runjiazeng/flickr8k-enhanced",
                        help="Dataset source for test images: local load_from_disk path or HF dataset name")
    parser.add_argument("--split", default="test",
                        help="Dataset split to caption")
    parser.add_argument("--references", default=str(ROOT_DIR / "evaluation" / "flickr8k_references.json"),
                        help="Reference JSON — only image_ids present here will be run")
    parser.add_argument("--checkpoint_dir", default=str(ROOT_DIR / "checkpoints"))
    args = parser.parse_args()

    # load test images
    test_rows = load_test_split(args.data_dir, args.split)
    print(f"  {len(test_rows)} images in {args.split} split")

    # restrict to matched image IDs
    refs      = json.load(open(args.references))
    valid_ids = set(refs.keys())
    print(f"  Using {len(valid_ids)} matched image IDs")

    models_to_run = MODELS if args.model == "all" else [args.model]
    variants_to_run = list(VARIANTS) if args.variant == "all" else [args.variant]

    if args.checkpoint and (len(models_to_run) != 1 or len(variants_to_run) != 1):
        parser.error("--checkpoint can only be used with one model and one variant")

    for model_name in models_to_run:
        for variant in variants_to_run:
            spec = VARIANTS[variant]

            if args.checkpoint:
                ckpt_path = Path(args.checkpoint)
            else:
                ckpt_dir = Path(args.checkpoint_dir) / f"{model_name}{spec['checkpoint_dir_suffix']}"
                ckpt_path = ckpt_dir / "best.pt"

            if not ckpt_path.exists():
                print(f"\n[SKIP] {model_name} ({variant}) — checkpoint not found: {ckpt_path}")
                continue

            captions = run_model(model_name, str(ckpt_path), test_rows, valid_ids)
            save_results(model_name, spec["out_folder"], captions)
            if captions:
                sample = next(iter(captions.items()))
                print(f"  Sample: id={sample[0]}  caption='{sample[1][:80]}'")
            else:
                print("  No captions generated — check dataset image IDs against references")

    print("\n✅ Inference complete.")


if __name__ == "__main__":
    main()
