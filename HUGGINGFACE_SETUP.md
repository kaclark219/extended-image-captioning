# Hugging Face Checkpoint Setup

This project includes a helper script at `scripts/hf_checkpoint_hub.py` to can upload the four model checkpoints to one public Hugging Face model repo and download them again later.

## What this script uploads

By default, the upload command sends:

- `simple-cnn-lstm/best.pt`
- `simple-cnn-lstm/last.pt`
- `simple-cnn-lstm/vocab.json`
- `sureal01-cnn-lstm/best.pt`
- `sureal01-cnn-lstm/last.pt`
- `sureal01-cnn-lstm/vocab.json`
- `vit-gpt2/best.pt`
- `vit-gpt2/last.pt`
- `vit-gpt2/vocab.json`
- `blip-base/best.pt`
- `blip-base/last.pt`
- `blip-base/vocab.json`

If you only want the evaluation checkpoints, add `--best-only` and the script will skip each model's `last.pt`.

## One-time setup

1. Activate your virtual environment.

```bash
source .venv/bin/activate
```

2. Make sure dependencies are installed.

```bash
pip install -r requirements.txt
```

3. Create a Hugging Face account if you do not already have one.

`https://huggingface.co/join`

4. Create an access token with write permission.

Open:

`https://huggingface.co/settings/tokens`

Create a token that can write to repositories.

5. Log in from the terminal.

```bash
hf auth login
```

Paste the token when prompted.

6. Confirm the local checkpoint files are present.

```bash
python scripts/hf_checkpoint_hub.py list
```

## Recommended public repo layout

Create one public model repo on Hugging Face, for example:

- `YOUR_USERNAME/extended-image-captioning-checkpoints`

Inside that repo, the helper script will create folders like:

```text
simple-cnn-lstm/
sureal01-cnn-lstm/
vit-gpt2/
blip-base/
```

## Upload the four best checkpoints

Run this from the project root:

```bash
python scripts/hf_checkpoint_hub.py upload \
  --repo-id YOUR_USERNAME/extended-image-captioning-checkpoints \
  --models all
```

This will create the repo if it does not exist yet and upload each model's `best.pt`, `last.pt`, and any available `vocab.json`.


## Download checkpoints later

You or your teammates can download the published files with:

```bash
python scripts/hf_checkpoint_hub.py download \
  --repo-id YOUR_USERNAME/extended-image-captioning-checkpoints \
  --models all \
  --local-dir downloaded-checkpoints
```

To download only `best.pt` plus `vocab.json`:

```bash
python scripts/hf_checkpoint_hub.py download \
  --repo-id YOUR_USERNAME/extended-image-captioning-checkpoints \
  --models all \
  --best-only \
  --local-dir downloaded-checkpoints
```

# Extended Image Captioning Checkpoints

This repository contains trained checkpoints for four image captioning models from the `extended-image-captioning` project:

- `simple-cnn-lstm`
- `sureal01-cnn-lstm`
- `vit-gpt2`
- `blip-base`

## Dataset

- `runjiazeng/flickr8k-enhanced`

## Contents

Each subfolder contains the published checkpoint files for one model.

## Notes

- `best.pt` is the recommended file for evaluation and testing.
- `last.pt` is mainly useful if you want to resume training.
- `vocab.json` is included for models that use a saved vocabulary/tokenizer export.

## Project code

GitHub repository:

`https://github.com/kaclark219/extended-image-captioning`
```

## Useful official docs

- Hugging Face upload guide: `https://huggingface.co/docs/huggingface_hub/guides/upload`
- Hugging Face storage limits: `https://huggingface.co/docs/hub/storage-limits`
- Hugging Face model uploading guide: `https://huggingface.co/docs/hub/en/models-uploading`
