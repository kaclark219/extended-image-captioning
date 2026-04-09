import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_ID = "impactframes/Qwen2-VL-7B-Captioner"
BASE_SEED = 42

SYSTEM_PROMPT = (
    "You are an expert image captioning assistant. "
    "Generate detailed, accurate descriptions of images."
)

# temperature / top_p schedule across the 5 caption slots
PARAMS = [
    {"temperature": 0.70, "top_p": 0.90},
    {"temperature": 0.80, "top_p": 0.90},
    {"temperature": 0.90, "top_p": 0.92},
    {"temperature": 1.00, "top_p": 0.95},
    {"temperature": 1.00, "top_p": 0.95},
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(caption):
    return (
        f'The following short caption describes this image from a particular perspective:\n"{caption}"\n\n'
        "Write a detailed caption of this image in 100 to 200 words (no more than 300). "
        "Cover object identities and attributes, spatial relationships, scene context, "
        "actions or interactions, and background details. "
        "Stay consistent with the perspective in the short caption above. "
        "Write in fluent prose, no bullet points."
    )


def has_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text))


def clip_to_sentence(text, max_words=300):
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    # trim trailing incomplete sentence regardless of length
    if text and text[-1] not in ".!?":
        last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last > len(text) // 2:  # only trim if boundary isn't too early
            text = text[:last + 1]
    return text


def generate_caption(model, processor, image, original, seed, temperature, top_p, max_retries=3):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": build_prompt(original)},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    result = None
    for attempt in range(max_retries):
        # shift seed on retry to get a different sample
        s = seed if attempt == 0 else seed + attempt * 997
        set_seed(s)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
            )
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        result = clip_to_sentence(processor.decode(new_tokens, skip_special_tokens=True).strip())
        if not has_chinese(result):
            return result
    # all retries had Chinese — return last attempt as best effort
    return result


def process_split(rank, world_size, split_name, model, processor, output_dir, base_seed):
    # jxie/flickr8k: one row per image, captions in caption_0 .. caption_4, no image_id field.
    # image_id is the row index, which is stable across loads.
    tmp_path = output_dir / f"{split_name}_rank{rank}.jsonl"

    done = set()
    if tmp_path.exists():
        with open(tmp_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    # only skip if all captions are clean; re-generate if any has Chinese
                    if all(not has_chinese(c) for c in rec["captions"]):
                        done.add(rec["image_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    ds = load_dataset("jxie/flickr8k", split=split_name)
    # interleave work across gpus so each handles every world_size-th image
    my_indices = [i for i in range(len(ds)) if i % world_size == rank]

    with open(tmp_path, "a", encoding="utf-8") as f:
        for image_idx in tqdm(my_indices, desc=f"{split_name}@gpu{rank}", position=rank):
            iid = str(image_idx)
            if iid in done:
                continue

            row = ds[image_idx]
            image = row["image"].convert("RGB")
            originals = [row[f"caption_{i}"] for i in range(5)]
            captions = []

            for cap_idx, original in enumerate(originals):
                seed = base_seed + image_idx * 5 + cap_idx
                p = PARAMS[cap_idx]
                captions.append(generate_caption(
                    model, processor, image, original,
                    seed, p["temperature"], p["top_p"],
                ))

            record = {"image_id": iid, "captions": captions}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def merge_split(split_name, output_dir, world_size):
    # read all rank files; later entries overwrite earlier ones for the same image_id
    # (this handles the case where a bad record was re-generated and appended)
    records = {}
    for rank in range(world_size):
        tmp = output_dir / f"{split_name}_rank{rank}.jsonl"
        if not tmp.exists():
            continue
        with open(tmp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        records[rec["image_id"]] = line
                    except (json.JSONDecodeError, KeyError):
                        pass
        tmp.unlink()

    out_path = output_dir / f"{split_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for line in records.values():
            f.write(line)
    print(f"merged {split_name} -> {out_path} ({len(records)} records)")


def worker(rank, world_size, splits, output_dir, base_seed, model_id):
    torch.cuda.set_device(rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{rank}",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    for split_name in splits:
        process_split(rank, world_size, split_name, model, processor, output_dir, base_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=["train", "validation", "test", "all"])
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--num_gpus", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]

    # pre-download the dataset before spawning to avoid cache races
    print("caching dataset...")
    for s in splits:
        load_dataset("jxie/flickr8k", split=s)

    mp.spawn(
        worker,
        args=(args.num_gpus, splits, output_dir, args.seed, args.model),
        nprocs=args.num_gpus,
        join=True,
    )

    for split_name in splits:
        merge_split(split_name, output_dir, args.num_gpus)

    print("done.")


if __name__ == "__main__":
    main()
