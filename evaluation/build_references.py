"""
build_references.py
===================
Convert Flickr8k caption files into the reference JSON format expected by
evaluate_captions.py.

Flickr8k provides captions as a text file with one line per entry:
    <filename>#<caption_number>\t<caption text>

This script reads that file, filters to only the test split, and writes:
    {
      "image_filename_or_id": ["ref caption 1", "ref caption 2", ...]
    }

Usage
-----
    # Original Flickr8k references (test split)
    python build_references.py \
        --captions  data/Flickr8k.token.txt \
        --split     data/Flickr_8k.testImages.txt \
        --output    references/flickr8k_references.json

    # Flickr8k Enhanced references (same images, longer captions)
    python build_references.py \
        --captions  data/flickr8k_enhanced_captions.json \
        --split     data/Flickr_8k.testImages.txt \
        --output    references/flickr8k_enhanced_references.json \
        --enhanced

    # HuggingFace dataset variant (runjiazeng/flickr8k-enhanced)
    python build_references.py \
        --hf_dataset runjiazeng/flickr8k-enhanced \
        --split      test \
        --output     references/flickr8k_enhanced_references.json

Notes
-----
- Image IDs are the bare filenames (e.g. "1000268201_693b08cb0e.jpg").
  The generate scripts should use the same convention so IDs align.
- For the enhanced dataset on HuggingFace you need `datasets` installed:
      pip install datasets
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_flickr8k_token_file(captions_path: Path, split_images: set[str]) -> dict:
    """
    Parse Flickr8k.token.txt and filter to test images.
    Each line: <image.jpg>#<N>\t<caption>
    """
    refs: dict[str, list[str]] = defaultdict(list)
    with open(captions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, caption = line.split("\t", 1)
            image_id = key.split("#")[0].strip()
            if not split_images or image_id in split_images:
                refs[image_id].append(caption.strip())
    return dict(refs)


def load_enhanced_json(captions_path: Path, split_images: set[str]) -> dict:
    """
    Load pre-built enhanced captions JSON.
    Expected format (two options):
      Option A: {image_id: "single caption"}
      Option B: {image_id: ["cap1", "cap2", ...]}
    Both are normalised to lists.
    """
    with open(captions_path) as f:
        data = json.load(f)

    refs = {}
    for img_id, cap in data.items():
        if split_images and img_id not in split_images:
            continue
        refs[img_id] = [cap] if isinstance(cap, str) else cap
    return refs


def load_from_hf(dataset_name: str, split: str) -> dict:
    """Download from HuggingFace hub and build reference dict."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[ERROR] Install `datasets` to use --hf_dataset:\n  pip install datasets")

    ds = load_dataset(dataset_name, split=split)
    refs: dict[str, list[str]] = defaultdict(list)
    for example in ds:
        img_id = example.get("image_id") or example.get("filename") or str(example.get("id", ""))
        caption = example.get("caption") or example.get("enhanced_caption") or ""
        if img_id and caption:
            refs[img_id].append(caption)
    return dict(refs)


def load_split_images(split_file: Path) -> set[str]:
    with open(split_file) as f:
        return {line.strip() for line in f if line.strip()}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Build reference JSON from Flickr8k caption files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── input options ──────────────────────────────────────────────────────
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--captions",   type=Path,
                     help="Path to Flickr8k.token.txt (original) or enhanced JSON")
    grp.add_argument("--hf_dataset", type=str,
                     help="HuggingFace dataset name, e.g. runjiazeng/flickr8k-enhanced")

    p.add_argument("--enhanced", action="store_true",
                   help="Treat --captions as a JSON file (enhanced dataset) rather than .token.txt")
    p.add_argument("--split",    type=str, default=None,
                   help="Either a path to a Flickr8k split .txt file OR a HF split name (test/train/val)")
    p.add_argument("--output",   type=Path, required=True,
                   help="Output JSON path")
    args = p.parse_args()

    # ── load ───────────────────────────────────────────────────────────────
    if args.hf_dataset:
        split_name = args.split or "test"
        print(f"Loading from HuggingFace: {args.hf_dataset} ({split_name} split)…")
        refs = load_from_hf(args.hf_dataset, split_name)
    else:
        split_images: set[str] = set()
        if args.split:
            sp = Path(args.split)
            if sp.exists():
                split_images = load_split_images(sp)
                print(f"Using {len(split_images)} images from split file: {sp}")
            else:
                # Might be a HF split name passed by mistake; warn and continue
                print(f"[WARN] --split '{args.split}' is not a file path — loading all images.")

        if args.enhanced:
            print(f"Loading enhanced JSON: {args.captions}")
            refs = load_enhanced_json(args.captions, split_images)
        else:
            print(f"Loading Flickr8k token file: {args.captions}")
            refs = load_flickr8k_token_file(args.captions, split_images)

    # ── write ──────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(refs, f, indent=2)

    total_caps = sum(len(v) for v in refs.values())
    print(f"[✓] Wrote {len(refs)} images / {total_caps} captions → {args.output}")


if __name__ == "__main__":
    main()
