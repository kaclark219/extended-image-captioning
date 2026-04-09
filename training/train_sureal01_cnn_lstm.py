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
    build_caption_pairs,
    make_loaders,
    run_epoch,
    save_checkpoint,
    set_seed,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="runjiazeng/flickr8k-enhanced")
    parser.add_argument("--hf_repo", default="sureal01/image-captioning-model")
    parser.add_argument("--hf_checkpoint_file", default="")
    parser.add_argument("--output_dir", default="checkpoints/sureal01-cnn-lstm")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--min_word_freq", type=int, default=2)
    parser.add_argument("--encoder_arch", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--trainable_encoder_layers", type=int, default=1)
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--train_limit_images", type=int, default=0)
    parser.add_argument("--val_limit_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def maybe_load_hf_checkpoint(model, optimizer, repo_id: str, filename: str):
    """Try to load a checkpoint from the Hugging Face Hub repo."""
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

    train_pairs = build_caption_pairs(args.dataset, "train", limit_images=args.train_limit_images)
    vocab = Vocabulary(min_freq=args.min_word_freq)
    vocab.build([caption for _, caption in train_pairs])

    # create data loaders
    _, _, train_loader, val_loader = make_loaders(
        dataset_name=args.dataset,
        vocab=vocab,
        image_size=args.image_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        train_limit_images=args.train_limit_images,
        val_limit_images=args.val_limit_images,
        num_workers=args.num_workers,
    )
    # initialize model
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

    # set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # try loading checkpoint from Hugging Face Hub
    loaded_name = maybe_load_hf_checkpoint(model, optimizer, args.hf_repo, args.hf_checkpoint_file)
    if loaded_name:
        print(f"loaded checkpoint from {args.hf_repo}:{loaded_name}")

    # training loop with validation & checkpointing
    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, vocab.pad_id, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, vocab.pad_id, train=False)
        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        checkpoint = {
            "dataset": args.dataset,
            "hf_repo": args.hf_repo,
            "encoder_arch": args.encoder_arch,
            "image_size": args.image_size,
            "max_len": args.max_len,
            "min_word_freq": args.min_word_freq,
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "trainable_encoder_layers": args.trainable_encoder_layers,
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
