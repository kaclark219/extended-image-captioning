import argparse
import math
import os
import torch
from huggingface_hub import hf_hub_download
from cnn_lstm_common import (
    CaptioningModel,
    EncoderCNN,
    SurealCaptionModel,
    Vocabulary,
    build_caption_texts,
    make_loaders,
    run_epoch,
    save_checkpoint,
    set_seed,
)


def parse_args():
    """Parse command-line arguments for the Flickr8k baseline model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="jxie/flickr8k")
    parser.add_argument("--hf_repo", default="")
    parser.add_argument("--hf_checkpoint_file", default="")
    parser.add_argument("--output_dir", default="checkpoints/sureal01-cnn-lstm-baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--min_word_freq", type=int, default=2)
    parser.add_argument("--encoder_arch", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--trainable_encoder_layers", type=int, default=1)
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--train_start_pct", type=float, default=0.0)
    parser.add_argument("--train_end_pct", type=float, default=100.0)
    parser.add_argument("--val_start_pct", type=float, default=0.0)
    parser.add_argument("--val_end_pct", type=float, default=100.0)
    parser.add_argument("--train_limit_images", type=int, default=0)
    parser.add_argument("--val_limit_images", type=int, default=0)
    parser.add_argument("--vocab_limit_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def maybe_load_local_checkpoint(model, optimizer, checkpoint_path: str):
    """Try to load a local checkpoint and recover the saved vocabulary."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        return None

    model.load_state_dict(checkpoint["model"], strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    vocab = None
    if checkpoint.get("vocab"):
        vocab = Vocabulary.from_tokens(checkpoint["vocab"])

    return checkpoint, vocab


def maybe_read_local_checkpoint(checkpoint_path: str):
    """Read a local checkpoint before the model exists so we can recover the saved vocabulary."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = None
    if checkpoint.get("vocab"):
        vocab = Vocabulary.from_tokens(checkpoint["vocab"])
    return checkpoint, vocab


def maybe_load_hf_checkpoint(model, optimizer, repo_id: str, filename: str):
    """Try to load a checkpoint from the Hugging Face Hub repo."""
    if not repo_id:
        return ""

    filenames = [filename] if filename else ["best.pt", "last.pt", "model.pt", "checkpoint.pt"]

    for candidate in filenames:
        try:
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=candidate)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            continue

        if "model" not in checkpoint:
            continue

        model.load_state_dict(checkpoint["model"], strict=False)
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return candidate

    return ""


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_epoch = 1
    best_val = math.inf
    local_checkpoint = None
    vocab = None

    local_checkpoint, saved_vocab = maybe_read_local_checkpoint(args.resume_from)
    if local_checkpoint is not None:
        vocab = saved_vocab
        start_epoch = local_checkpoint.get("epoch", 0) + 1
        best_val = local_checkpoint.get("val_loss", math.inf)
        print(f"found local checkpoint at {args.resume_from}")

    if vocab is None:
        vocab_texts = build_caption_texts(
            args.dataset,
            "train",
            limit_images=args.vocab_limit_images,
            start_pct=0.0,
            end_pct=100.0,
        )
        vocab = Vocabulary(min_freq=args.min_word_freq)
        vocab.build(vocab_texts)
        print(f"built vocab from {len(vocab_texts)} captions")

    encoder = EncoderCNN(
        arch=args.encoder_arch,
        trainable_layers=args.trainable_encoder_layers,
        pooled=True,
    )
    decoder = SurealCaptionModel(
        vocab_size=len(vocab),
        feat_dim=encoder.feat_dim,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model = CaptioningModel(encoder, decoder).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if local_checkpoint is not None:
        maybe_load_local_checkpoint(model, optimizer, args.resume_from)
        print(f"resumed local checkpoint from {args.resume_from}")
    else:
        loaded_name = maybe_load_hf_checkpoint(model, optimizer, args.hf_repo, args.hf_checkpoint_file)
        if loaded_name:
            print(f"loaded checkpoint from {args.hf_repo}:{loaded_name}")

    _, _, train_loader, val_loader = make_loaders(
        dataset_name=args.dataset,
        vocab=vocab,
        image_size=args.image_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        train_limit_images=args.train_limit_images,
        val_limit_images=args.val_limit_images,
        num_workers=args.num_workers,
        train_start_pct=args.train_start_pct,
        train_end_pct=args.train_end_pct,
        val_start_pct=args.val_start_pct,
        val_end_pct=args.val_end_pct,
    )
    print(
        f"train slice: {args.train_start_pct:.1f}%..{args.train_end_pct:.1f}% "
        f"({len(train_loader.dataset)} caption pairs) | "
        f"val slice: {args.val_start_pct:.1f}%..{args.val_end_pct:.1f}% "
        f"({len(val_loader.dataset)} caption pairs)"
    )

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, vocab.pad_id, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, vocab.pad_id, train=False)
        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        checkpoint = {
            "dataset": args.dataset,
            "model_variant": "baseline",
            "hf_repo": args.hf_repo,
            "encoder_arch": args.encoder_arch,
            "image_size": args.image_size,
            "max_len": args.max_len,
            "min_word_freq": args.min_word_freq,
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "trainable_encoder_layers": args.trainable_encoder_layers,
            "train_start_pct": args.train_start_pct,
            "train_end_pct": args.train_end_pct,
            "val_start_pct": args.val_start_pct,
            "val_end_pct": args.val_end_pct,
        }
        save_checkpoint(
            os.path.join(args.output_dir, "last.pt"),
            model,
            optimizer,
            epoch,
            val_loss,
            vocab,
            checkpoint,
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                os.path.join(args.output_dir, "best.pt"),
                model,
                optimizer,
                epoch,
                val_loss,
                vocab,
                checkpoint,
            )
            vocab.save(os.path.join(args.output_dir, "vocab.json"))

    print(f"best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
