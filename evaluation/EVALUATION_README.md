# Evaluation Pipeline — Extended Image Captioning

This README covers everything needed to run the full evaluation pipeline:
CIDEr, BLEU-4, METEOR, ROUGE-L, BERTScore, and CLIPScore.

Written by Prakyat — all scripts in this README are ready to run as-is.

---

## Files Included

| File | Purpose |
|---|---|
| `run_inference.py` | Loads checkpoints, generates captions for all 810 test images |
| `evaluate_captions.py` | Computes CIDEr, BLEU-4, METEOR, ROUGE-L (no Java required) |
| `build_references.py` | Builds reference JSONs from HuggingFace datasets (already done) |
| `test_metrics.py` | Sanity check — run this first to verify metrics work |
| `evaluation/flickr8k_references.json` | 810 test image IDs → original short captions (5 per image) |
| `evaluation/flickr8k_enhanced_references.json` | 810 test image IDs → Qwen enhanced captions (5 per image) |

> **Important:** Do NOT rebuild the reference files. The 810 image IDs were
> matched by pixel hash across `jxie/flickr8k` and `runjiazeng/flickr8k-enhanced`.
> Rebuilding will break alignment with existing captions.json files.

---

## Prerequisites

```bash
# Core dependencies (should already be installed)
pip install torch torchvision transformers datasets tqdm numpy

# Evaluation-specific
pip install tabulate bert-score "torchmetrics[multimodal]"
```

Python 3.9+ required.

---

## Directory Structure Expected

Before running the evaluator, your `results/` folder must look like this:

```
results/
  blip-base/
    flickr8k-enhanced/
      captions.json        ← enhanced-trained model output
  vit-gpt2/
    flickr8k-enhanced/
      captions.json
  simple-cnn-lstm/
    flickr8k/
      captions.json        ← original-trained baseline output
    flickr8k-enhanced/
      captions.json        ← enhanced-trained model output
  sureal01-cnn-lstm/
    flickr8k/
      captions.json        ← original-trained baseline output
    flickr8k-enhanced/
      captions.json        ← enhanced-trained model output
```

Each `captions.json` is a flat dict: `{ "image_id": "generated caption", ... }`
where image_id matches the keys in the reference JSON files.

---

## Step 1 — Verify Metrics Work

Run this first to confirm all metric implementations are working:

```bash
python test_metrics.py
# All tests should show ✅
```

---

## Step 2 — Run Inference

### Enhanced-trained models (existing checkpoints from HuggingFace)

Make sure checkpoints are downloaded to `checkpoints/` first.
See `HUGGINGFACE_SETUP.md` in the repo for download instructions.

```bash
python evaluation/run_inference.py
```

This now runs both checkpoint variants automatically:
- enhanced-trained checkpoints from `checkpoints/<model>/best.pt`
- baseline checkpoints from `checkpoints/<model>-baseline/best.pt`

Outputs are saved to:
- `results/<model>/flickr8k-enhanced/captions.json`
- `results/<model>/flickr8k/captions.json`

To run just one model or one checkpoint family:

```bash
python evaluation/run_inference.py --model blip-base
python evaluation/run_inference.py --model blip-base --variant baseline
python evaluation/run_inference.py --model blip-base --variant enhanced
```

---

## Step 3 — CIDEr / BLEU-4 / METEOR / ROUGE-L

```bash
python evaluation/evaluate_captions.py \
    --results_dir results/ \
    --ref_original evaluation/flickr8k_references.json \
    --output_dir eval_output/
```

By default, both result folders are scored against the original short references
for a fair enhanced-vs-baseline comparison. If you also want to score the
enhanced-trained outputs against the long references, add:

```bash
--ref_enhanced evaluation/flickr8k_enhanced_references.json
```

Output files written to `eval_output/`:
- `scores.json` — raw numbers for all models and conditions
- `results_table.txt` — formatted comparison table with delta rows

The evaluator automatically skips any model/condition where `captions.json`
is missing — so you can run it at any point even if not all conditions are done yet.

---

## Step 4 — BERTScore

BERTScore measures semantic similarity using RoBERTa-large embeddings.
It is robust to length differences between hypothesis and reference.

```bash
python - <<'EOF'
import json
from bert_score import score

models = {
    "BLIP-base (enhanced)":          "results/blip-base/flickr8k-enhanced/captions.json",
    "ViT-GPT2 (enhanced)":           "results/vit-gpt2/flickr8k-enhanced/captions.json",
    "CNN+LSTM custom (original)":    "results/simple-cnn-lstm/flickr8k/captions.json",
    "CNN+LSTM custom (enhanced)":    "results/simple-cnn-lstm/flickr8k-enhanced/captions.json",
    "CNN+LSTM sureal01 (original)":  "results/sureal01-cnn-lstm/flickr8k/captions.json",
    "CNN+LSTM sureal01 (enhanced)":  "results/sureal01-cnn-lstm/flickr8k-enhanced/captions.json",
}
refs = json.load(open("references/flickr8k_enhanced_references.json"))

print(f"{'Model':<40} {'P':>8} {'R':>8} {'F1':>8}")
print("-" * 68)
for name, path in models.items():
    import os
    if not os.path.exists(path):
        print(f"{name:<40} {'(missing)':>8}")
        continue
    caps = json.load(open(path))
    ids  = sorted(set(caps) & set(refs))
    P, R, F1 = score([caps[i] for i in ids], [refs[i][0] for i in ids],
                     lang="en", verbose=False)
    print(f"{name:<40} {P.mean():.4f} {R.mean():.4f} {F1.mean():.4f}")
EOF
```

Takes ~10 minutes. Downloads RoBERTa-large on first run (~1.4 GB).

---

## Step 5 — CLIPScore

CLIPScore measures image-caption alignment directly — no reference captions needed.
A score above 20 is decent, above 30 is strong.

```bash
python - <<'EOF'
import json, torch, os
from datasets import load_from_disk
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

ds = load_from_disk("data/flickr8k-enhanced")
test_rows = {str(row["image_id"]): row["image"] for row in ds["test"]}
to_tensor  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

def truncate(t, n=60):
    return " ".join(t.split()[:n])

models = {
    "BLIP-base (enhanced)":          "results/blip-base/flickr8k-enhanced/captions.json",
    "ViT-GPT2 (enhanced)":           "results/vit-gpt2/flickr8k-enhanced/captions.json",
    "CNN+LSTM custom (original)":    "results/simple-cnn-lstm/flickr8k/captions.json",
    "CNN+LSTM custom (enhanced)":    "results/simple-cnn-lstm/flickr8k-enhanced/captions.json",
    "CNN+LSTM sureal01 (original)":  "results/sureal01-cnn-lstm/flickr8k/captions.json",
    "CNN+LSTM sureal01 (enhanced)":  "results/sureal01-cnn-lstm/flickr8k-enhanced/captions.json",
}

print(f"\n{'Model':<40} {'CLIPScore':>10}")
print("-" * 52)
for name, path in models.items():
    if not os.path.exists(path):
        print(f"{name:<40} {'(missing)':>10}")
        continue
    caps = json.load(open(path))
    ids  = sorted(set(caps) & set(test_rows))
    metric.reset()
    for i in range(0, len(ids), 32):
        batch = ids[i:i+32]
        imgs  = (torch.stack([
            to_tensor(test_rows[x].convert("RGB")) for x in batch
        ]) * 255).to(torch.uint8).to(device)
        metric.update(imgs, [truncate(caps[x]) for x in batch])
    print(f"{name:<40} {metric.compute().item():>10.4f}")
print()
EOF
```

Takes ~5 minutes. Downloads CLIP on first run (~600 MB).

---

## What the Metrics Mean

| Metric | What it measures | Length sensitive? |
|---|---|---|
| CIDEr | TF-IDF weighted n-gram overlap | Yes — penalises length mismatch |
| BLEU-4 | 4-gram precision | Yes |
| METEOR | Unigram F-mean with stemming | Somewhat |
| ROUGE-L | Longest common subsequence | Somewhat |
| BERTScore F1 | Semantic similarity via embeddings | No |
| CLIPScore | Direct image-caption alignment | No |

**Key caveat for the report:** all models were trained with `max_len=64` tokens,
so they generate standard-length captions (~15-30 words). The enhanced references
are 100-200 words. This suppresses CIDEr and BLEU-4 scores structurally — not
because captions are wrong, but because n-gram overlap collapses at that length
ratio. BERTScore and CLIPScore are the more honest metrics for this setup.

---

## Prakyat's Results (for reference)

Enhanced-trained models evaluated against enhanced references:

| Model | CIDEr | BLEU-4 | METEOR | ROUGE-L | BERTScore F1 | CLIPScore |
|---|---|---|---|---|---|---|
| ViT-GPT2 | 24.76 | 17.55 | 41.27 | 28.11 | 0.8778 | 22.10 |
| CNN+LSTM sureal01 | 6.19 | 19.83 | 41.69 | 30.17 | 0.8579 | 25.41 |
| BLIP-base | 1.20 | 9.20 | 28.86 | 28.89 | 0.8792 | 32.07 |
| CNN+LSTM custom | 0.06 | 0.00 | 28.64 | 23.27 | 0.7568 | 20.85 |

Note: ViT-GPT2 CIDEr was run with `max_new_tokens=200`. All other scores
use standard `max_new_tokens=64`. The current scripts default to 64.

---

## Common Issues

**`KeyError: 'image_id'`** — the dataset you loaded doesn't have an image_id column.
Make sure you're loading from `data/flickr8k-enhanced` (the saved HuggingFace dataset),
not the raw `jxie/flickr8k`.

**`0 overlapping image IDs`** — your captions.json uses different image IDs than
the reference files. Check that inference is running on `data/flickr8k-enhanced`
and using `str(row["image_id"])` as the key.

**`CLIPScore identical scores for all models`** — captions are exceeding CLIP's
77 token limit and all getting truncated to the same prefix. The `truncate()`
function in Step 5 handles this — make sure you're using the script above exactly.

**`TypeError: unsupported operand type(s) for |`** — you're on Python 3.9.
The scripts are compatible with 3.9 — if you see this error you have an old
version of one of the scripts. Re-download from the repo.
