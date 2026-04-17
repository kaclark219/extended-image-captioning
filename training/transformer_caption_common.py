import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cnn_lstm_common import CAPTION_COLUMNS, load_split_rows, select_rows_by_percentage


class CaptionTextDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = row["image"].convert("RGB")

        caption = ""
        for col in CAPTION_COLUMNS:
            value = row.get(col)
            if value:
                caption = value
                break

        return image, caption


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader


def build_image_text_rows(
    dataset_name: str,
    split: str,
    limit_images: int = 0,
    start_pct: float = 0.0,
    end_pct: float = 100.0,
):
    """Build image-text rows for a given dataset split."""
    rows = load_split_rows(dataset_name, split)
    rows = select_rows_by_percentage(rows, start_pct=start_pct, end_pct=end_pct)
    if limit_images:
        rows = rows[:limit_images]
    return rows


def make_image_text_loaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    train_limit_images: int = 0,
    val_limit_images: int = 0,
    train_start_pct: float = 0.0,
    train_end_pct: float = 100.0,
    val_start_pct: float = 0.0,
    val_end_pct: float = 100.0,
):
    """Make image-text data loaders for training and validation."""
    train_rows = build_image_text_rows(
        dataset_name,
        "train",
        limit_images=train_limit_images,
        start_pct=train_start_pct,
        end_pct=train_end_pct,
    )
    val_rows = build_image_text_rows(
        dataset_name,
        "validation",
        limit_images=val_limit_images,
        start_pct=val_start_pct,
        end_pct=val_end_pct,
    )

    train_dataset = CaptionTextDataset(train_rows)
    val_dataset = CaptionTextDataset(val_rows)

    def collate_fn(batch):
        """Collate function to combine images and captions into batches."""
        images, captions = zip(*batch)
        return list(images), list(captions)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    return LoaderBundle(train_loader=train_loader, val_loader=val_loader)


def masked_labels(input_ids: torch.Tensor, pad_token_id: Optional[int]):
    """Mask input IDs for loss calculation by replacing pad token IDs with -100."""
    labels = input_ids.clone()
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    return labels


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    build_batch_inputs: Callable,
    train: bool,
):
    """Run a single epoch of training or validation."""
    model.train(train)
    total_loss = 0.0
    total_items = 0

    for images, captions in tqdm(loader, leave=False):
        if train:
            optimizer.zero_grad(set_to_none=True)

        batch_inputs = build_batch_inputs(images, captions, device)
        outputs = model(**batch_inputs)
        loss = outputs.loss

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        batch_size = len(images)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(1, total_items)


def maybe_read_local_checkpoint(checkpoint_path: str):
    """Read a local checkpoint if it exists, otherwise return None."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    return torch.load(checkpoint_path, map_location="cpu")


def maybe_load_local_checkpoint(model, optimizer, checkpoint_path: str):
    """Try to load a local checkpoint and recover the saved model and optimizer state."""
    checkpoint = maybe_read_local_checkpoint(checkpoint_path)
    if checkpoint is None:
        return None

    model_state = checkpoint.get("model")
    if model_state:
        model.load_state_dict(model_state, strict=False)

    optimizer_state = checkpoint.get("optimizer")
    if optimizer is not None and optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    return checkpoint


def save_checkpoint(path: str, model, optimizer, epoch: int, val_loss: float, config: dict):
    """Save a checkpoint of the model and optimizer state to a local file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )


def save_tokenizer_vocab(output_path: str, tokenizer):
    """Save the tokenizer vocabulary and relevant information to a JSON file."""
    vocab = {}
    if hasattr(tokenizer, "get_vocab"):
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = {}

    fallback = {
        "name_or_path": getattr(tokenizer, "name_or_path", ""),
        "vocab_size": getattr(tokenizer, "vocab_size", 0),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "bos_token": getattr(tokenizer, "bos_token", None),
    }

    payload = {
        "vocab": vocab,
        "tokenizer_info": fallback,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def checkpoint_resume_state(local_checkpoint, default_best=math.inf):
    """Determine the starting epoch and best validation loss from a local checkpoint."""
    if not local_checkpoint:
        return 1, default_best
    start_epoch = local_checkpoint.get("epoch", 0) + 1
    best_val = local_checkpoint.get("val_loss", default_best)
    return start_epoch, best_val
