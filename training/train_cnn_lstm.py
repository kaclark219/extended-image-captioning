import argparse
import math
import os
import torch
from cnn_lstm_common import (
    CaptioningModel,
    EncoderCNN,
    SimpleCaptionModel,
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
    parser.add_argument("--output_dir", default="checkpoints/simple-cnn-lstm")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--min_word_freq", type=int, default=2)
    parser.add_argument("--encoder_arch", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--trainable_encoder_layers", type=int, default=0)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
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
    decoder = SimpleCaptionModel(
        vocab_size=len(vocab),
        feat_dim=encoder.feat_dim,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model = CaptioningModel(encoder, decoder).to(device)

    # set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop with validation & checkpointing
    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, vocab.pad_id, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, vocab.pad_id, train=False)
        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        checkpoint = {
            "dataset": args.dataset,
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
        # save best checkpoint
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
