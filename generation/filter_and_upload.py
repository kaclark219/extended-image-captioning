import argparse
import json
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


def has_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text))


def load_and_filter(path: Path) -> tuple[dict, int]:
    """Return ({image_id: captions}, n_dropped)."""
    clean, dropped = {}, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if any(has_chinese(c) for c in rec.get("captions", [])):
                dropped += 1
            else:
                clean[rec["image_id"]] = rec["captions"]
    return clean, dropped


def build_split(split_name: str, caption_map: dict) -> Dataset:
    """Load images from jxie/flickr8k and join with filtered captions."""
    print(f"  loading jxie/flickr8k [{split_name}]...")
    src = load_dataset("jxie/flickr8k", split=split_name)

    rows = []
    for idx in tqdm(range(len(src)), desc=f"  merging {split_name}", leave=False):
        iid = str(idx)
        if iid not in caption_map:
            continue  # dropped due to Chinese
        caps = caption_map[iid]
        rows.append({
            "image_id":  iid,
            "image":     src[idx]["image"].convert("RGB"),
            "caption_0": caps[0],
            "caption_1": caps[1],
            "caption_2": caps[2],
            "caption_3": caps[3],
            "caption_4": caps[4],
        })

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="runjiazeng/flickr8k-enhanced")
    parser.add_argument("--input_dir", default="data")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    split_files = {
        "train":      input_dir / "train.jsonl",
        "validation": input_dir / "validation.jsonl",
        "test":       input_dir / "test.jsonl",
    }

    missing = [str(p) for p in split_files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing files:\n" + "\n".join(missing))

    ds_dict = {}
    for name, path in split_files.items():
        caption_map, dropped = load_and_filter(path)
        total = len(caption_map) + dropped
        print(f"{name}: {len(caption_map)} kept, {dropped} dropped ({100*dropped/total:.1f}%)")
        ds_dict[name] = build_split(name, caption_map)

    dataset = DatasetDict(ds_dict)
    print(f"\n{dataset}")
    dataset.push_to_hub(repo_id=args.repo, private=args.private, token=args.token)
    print(f"\nuploaded → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
