import argparse
import math
import os

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from cnn_lstm_common import set_seed
from transformer_caption_common import (
    checkpoint_resume_state,
    make_image_text_loaders,
    masked_labels,
    maybe_load_local_checkpoint,
    maybe_read_local_checkpoint,
    run_epoch,
    save_checkpoint,
    save_tokenizer_vocab,
)


def parse_args():
    """Parse command-line arguments for the Flickr8k baseline model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="jxie/flickr8k")
    parser.add_argument("--model_id", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--output_dir", default="checkpoints/blip-base-baseline")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--train_start_pct", type=float, default=0.0)
    parser.add_argument("--train_end_pct", type=float, default=100.0)
    parser.add_argument("--val_start_pct", type=float, default=0.0)
    parser.add_argument("--val_end_pct", type=float, default=100.0)
    parser.add_argument("--train_limit_images", type=int, default=0)
    parser.add_argument("--val_limit_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BlipForConditionalGeneration.from_pretrained(args.model_id)
    processor = BlipProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    local_checkpoint = maybe_read_local_checkpoint(args.resume_from)
    start_epoch, best_val = checkpoint_resume_state(local_checkpoint, default_best=math.inf)

    if local_checkpoint is not None:
        maybe_load_local_checkpoint(model, optimizer, args.resume_from)
        print(f"resumed local checkpoint from {args.resume_from}")

    loaders = make_image_text_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_limit_images=args.train_limit_images,
        val_limit_images=args.val_limit_images,
        train_start_pct=args.train_start_pct,
        train_end_pct=args.train_end_pct,
        val_start_pct=args.val_start_pct,
        val_end_pct=args.val_end_pct,
    )
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader

    print(
        f"train slice: {args.train_start_pct:.1f}%..{args.train_end_pct:.1f}% "
        f"({len(train_loader.dataset)} image-caption pairs) | "
        f"val slice: {args.val_start_pct:.1f}%..{args.val_end_pct:.1f}% "
        f"({len(val_loader.dataset)} image-caption pairs)"
    )

    def build_batch_inputs(images, captions, target_device):
        """Build batch inputs for the BLIP model."""
        tokenized = processor(
            images=images,
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_len,
        )

        input_ids = tokenized.input_ids.to(target_device)
        attention_mask = tokenized.attention_mask.to(target_device)
        pixel_values = tokenized.pixel_values.to(target_device)
        labels = masked_labels(input_ids, tokenizer.pad_token_id)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, build_batch_inputs, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, build_batch_inputs, train=False)
        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        checkpoint_config = {
            "dataset": args.dataset,
            "model_id": args.model_id,
            "model_variant": "baseline",
            "max_len": args.max_len,
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
            checkpoint_config,
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                os.path.join(args.output_dir, "best.pt"),
                model,
                optimizer,
                epoch,
                val_loss,
                checkpoint_config,
            )
            save_tokenizer_vocab(os.path.join(args.output_dir, "vocab.json"), tokenizer)

    print(f"best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
