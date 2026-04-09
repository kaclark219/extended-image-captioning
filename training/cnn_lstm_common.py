import json
import os
import random
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


CAPTION_COLUMNS = [f"caption_{i}" for i in range(5)]
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str):
    """Tokenize a text string."""
    return re.findall(r"[a-z0-9']+", text.lower())


# simple vocab class for tokenizing text
class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.itos = list(SPECIAL_TOKENS)

    @property
    def pad_id(self):
        """Get ID for the padding token."""
        return self.stoi["<pad>"]

    @property
    def bos_id(self):
        """Get ID for the beginning of sentence token."""
        return self.stoi["<bos>"]

    @property
    def eos_id(self):
        """Get ID for the end of sentence token."""
        return self.stoi["<eos>"]

    @property
    def unk_id(self):
        """Get ID for the unknown token."""
        return self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def build(self, captions):
        """Build the vocabulary from a list of captions."""
        counts = Counter()
        for caption in captions:
            counts.update(tokenize(caption))

        for token, freq in counts.items():
            if freq >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, text: str, max_len: int):
        """Encode a text string into a list of token IDs."""
        ids = [self.bos_id]
        ids.extend(self.stoi.get(tok, self.unk_id) for tok in tokenize(text))
        ids.append(self.eos_id)
        ids = ids[:max_len]
        if ids[-1] != self.eos_id:
            ids[-1] = self.eos_id
        return ids

    def decode(self, token_ids):
        """Decode a list of token IDs into a text string."""
        words = []
        for idx in token_ids:
            if idx == self.eos_id:
                break
            if idx in (self.pad_id, self.bos_id):
                continue
            words.append(self.itos[idx] if 0 <= idx < len(self.itos) else "<unk>")
        return " ".join(words)

    def save(self, path: str):
        """Save the vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos, "min_freq": self.min_freq}, f)

    @classmethod
    def from_tokens(cls, tokens, min_freq: int = 1):
        """Create a vocabulary from a list of tokens."""
        vocab = cls(min_freq=min_freq)
        vocab.itos = list(tokens)
        vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
        return vocab


def build_caption_pairs(dataset_name: str, split: str, limit_images: int = 0):
    """Build caption pairs from a dataset."""
    dataset = load_dataset(dataset_name, split=split)
    pairs = []

    for idx, row in enumerate(dataset):
        if limit_images and idx >= limit_images:
            break
        image = row["image"].convert("RGB")
        for col in CAPTION_COLUMNS:
            caption = row.get(col)
            if caption:
                pairs.append((image, caption))

    return pairs


# datasets & model definitions
class CaptionDataset(Dataset):
    def __init__(self, pairs, vocab: Vocabulary, image_size: int, max_len: int, train: bool):
        """Initialize the caption dataset by storing the caption pairs & setting up image transformations."""
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len

        resize = transforms.Resize((image_size, image_size))
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if train:
            self.transform = transforms.Compose([
                resize,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                resize,
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """Get a caption pair by index."""
        image, caption = self.pairs[idx]
        return (
            self.transform(image),
            torch.tensor(self.vocab.encode(caption, self.max_len), dtype=torch.long),
        )


# collator for padding captions in a batch
class CaptionCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        """Collate a batch of caption pairs by padding the captions to the same length & stacking the images."""
        images, captions = zip(*batch)
        images = torch.stack(images)
        max_len = max(len(caption) for caption in captions)
        padded = torch.full((len(captions), max_len), self.pad_id, dtype=torch.long)

        for idx, caption in enumerate(captions):
            padded[idx, : len(caption)] = caption

        return images, padded


# model definitions
class EncoderCNN(nn.Module):
    def __init__(self, arch: str = "resnet18", trainable_layers: int = 0, pooled: bool = True):
        """Initialize the CNN encoder by loading a pretrained ResNet model & removing the final classification layer."""
        super().__init__()

        if arch == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = 2048
        else:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feat_dim = feat_dim
        self.pooled = pooled

        for param in self.features.parameters():
            param.requires_grad = False

        if trainable_layers > 0:
            for block in list(self.features.children())[-trainable_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, images):
        """Forward pass through the CNN encoder."""
        feats = self.features(images)
        if self.pooled:
            return feats.flatten(1)
        return feats


# simple lstm decoder that takes image features & generates captions
class SimpleCaptionModel(nn.Module):
    def __init__(self, vocab_size: int, feat_dim: int, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.feature_proj = nn.Linear(feat_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_features, captions):
        """Forward pass through the simple caption model."""
        token_embeddings = self.embedding(captions[:, :-1])
        image_token = self.feature_proj(image_features).unsqueeze(1)
        lstm_input = torch.cat([image_token, token_embeddings], dim=1)
        outputs, _ = self.lstm(lstm_input)
        logits = self.output(self.dropout(outputs))
        return logits[:, :-1, :]


# a more complex lstm decoder that uses the image features at every step
class SurealCaptionModel(nn.Module):
    def __init__(self, vocab_size: int, feat_dim: int, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.init_h = nn.Linear(feat_dim, hidden_dim)
        self.init_c = nn.Linear(feat_dim, hidden_dim)
        self.input_proj = nn.Linear(feat_dim + emb_dim, hidden_dim)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_features, captions):
        """Forward pass through the more complex caption model that uses image features at every step."""
        embeddings = self.embedding(captions)
        h = self.init_h(image_features)
        c = self.init_c(image_features)
        steps = []

        for step in range(captions.size(1) - 1):
            decoder_input = torch.cat([image_features, embeddings[:, step, :]], dim=1)
            decoder_input = self.input_proj(decoder_input)
            h, c = self.lstm(decoder_input, (h, c))
            steps.append(self.output(self.dropout(h)).unsqueeze(1))

        return torch.cat(steps, dim=1)


# full captioning model that combines the encoder & decoder
class CaptioningModel(nn.Module):
    def __init__(self, encoder: EncoderCNN, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        """Forward pass through the full captioning model."""
        image_features = self.encoder(images)
        return self.decoder(image_features, captions)

    @torch.no_grad()
    def generate(self, images, bos_id: int, eos_id: int, max_len: int):
        """Generate captions for a batch of images."""
        self.eval()
        image_features = self.encoder(images)
        batch_size = images.size(0)
        captions = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=images.device)

        for _ in range(max_len - 1):
            logits = self.decoder(image_features, captions)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            captions = torch.cat([captions, next_token], dim=1)
            if torch.all(next_token.squeeze(1) == eos_id):
                break

        return captions


def caption_loss(logits, captions, pad_id: int):
    """Calculate the cross-entropy loss for caption generation, ignoring padding tokens."""
    targets = captions[:, 1:]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


def run_epoch(model, loader, optimizer, device, pad_id: int, train: bool):
    """Run a single epoch of training or validation."""
    model.train(train)
    total_loss = 0.0
    total_items = 0

    for images, captions in tqdm(loader, leave=False):
        images = images.to(device)
        captions = captions.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images, captions)
        loss = caption_loss(logits, captions, pad_id)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)

    return total_loss / max(1, total_items)


def make_loaders(dataset_name: str, vocab: Vocabulary, image_size: int, max_len: int, batch_size: int, train_limit_images: int = 0, val_limit_images: int = 0, num_workers: int = 2):
    """Create data loaders for training & validation."""
    train_pairs = build_caption_pairs(dataset_name, "train", limit_images=train_limit_images)
    val_pairs = build_caption_pairs(dataset_name, "validation", limit_images=val_limit_images)

    train_dataset = CaptionDataset(train_pairs, vocab, image_size=image_size, max_len=max_len, train=True)
    val_dataset = CaptionDataset(val_pairs, vocab, image_size=image_size, max_len=max_len, train=False)
    collator = CaptionCollator(vocab.pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    return train_pairs, val_pairs, train_loader, val_loader


def save_checkpoint(path: str, model, optimizer, epoch: int, val_loss: float, vocab: Vocabulary, config: dict):
    """Save a training checkpoint including the model state, optimizer state, epoch, validation loss, vocabulary, & training configuration."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "vocab": vocab.itos,
            "config": config,
        },
        path,
    )