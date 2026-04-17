import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


REPO_TYPE = "model"
CHECKPOINT_ROOT = Path("checkpoints")
MODEL_DIRS = {
    "simple-cnn-lstm": CHECKPOINT_ROOT / "simple-cnn-lstm",
    "sureal01-cnn-lstm": CHECKPOINT_ROOT / "sureal01-cnn-lstm",
    "vit-gpt2": CHECKPOINT_ROOT / "vit-gpt2",
    "blip-base": CHECKPOINT_ROOT / "blip-base",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload or download this project's model checkpoints to a Hugging Face Hub repo."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="Show the local checkpoint folders and files.")

    upload_parser = subparsers.add_parser("upload", help="Upload one or more checkpoint folders to the Hub.")
    upload_parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, for example username/repo.")
    upload_parser.add_argument(
        "--existing-repo",
        action="store_true",
        help="Skip repo creation and upload into an already-created Hugging Face repo.",
    )
    upload_parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to upload. Use 'all' or one/more of: simple-cnn-lstm sureal01-cnn-lstm vit-gpt2 blip-base",
    )
    upload_parser.add_argument(
        "--best-only",
        action="store_true",
        help="Upload only best.pt and vocab.json. By default last.pt is included too.",
    )
    upload_parser.add_argument(
        "--revision",
        default="main",
        help="Hub branch to upload to. Defaults to main.",
    )
    upload_parser.add_argument(
        "--readme-path",
        default="",
        help="Optional local README.md to upload to the root of the Hub repo before the checkpoint files.",
    )

    download_parser = subparsers.add_parser("download", help="Download one or more checkpoint folders from the Hub.")
    download_parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, for example username/repo.")
    download_parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to download. Use 'all' or one/more of: simple-cnn-lstm sureal01-cnn-lstm vit-gpt2 blip-base",
    )
    download_parser.add_argument(
        "--best-only",
        action="store_true",
        help="Download only best.pt and vocab.json. By default last.pt is included too.",
    )
    download_parser.add_argument(
        "--local-dir",
        default="downloaded-checkpoints",
        help="Destination directory for downloaded files.",
    )
    download_parser.add_argument(
        "--revision",
        default="main",
        help="Hub branch to download from. Defaults to main.",
    )
    return parser.parse_args()


def selected_models(requested):
    if requested == ["all"]:
        return list(MODEL_DIRS)

    invalid = [name for name in requested if name not in MODEL_DIRS]
    if invalid:
        raise SystemExit(f"Unknown model name(s): {', '.join(invalid)}")
    return requested


def validate_repo_id(repo_id):
    if "YOUR_USERNAME/" in repo_id:
        raise SystemExit(
            "Replace YOUR_USERNAME with your real Hugging Face namespace, for example "
            "'kaclark219/extended-image-captioning-checkpoints'."
        )


def files_for_model(model_name, include_last):
    model_dir = MODEL_DIRS[model_name]
    files = [model_dir / "best.pt"]
    vocab = model_dir / "vocab.json"
    if vocab.exists():
        files.append(vocab)
    if include_last:
        files.append(model_dir / "last.pt")
    return files


def cmd_list():
    print("Local checkpoint files:")
    for model_name, model_dir in MODEL_DIRS.items():
        print(f"\n{model_name}:")
        if not model_dir.exists():
            print("  missing local folder")
            continue
        for path in sorted(model_dir.iterdir()):
            if path.is_file():
                size_gb = path.stat().st_size / (1024 ** 3)
                print(f"  {path.name:<10} {size_gb:>7.2f} GB")


def upload_readme(api, repo_id, readme_path, revision):
    if not readme_path:
        return

    readme = Path(readme_path)
    if not readme.exists():
        raise SystemExit(f"README file not found: {readme}")

    api.upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        revision=revision,
        commit_message="Upload model card README",
    )
    print("Uploaded README.md")


def cmd_upload(repo_id, models, include_last, revision, readme_path, existing_repo):
    validate_repo_id(repo_id)
    api = HfApi()
    if existing_repo:
        print(f"Uploading into existing repo https://huggingface.co/{repo_id}")
    else:
        try:
            api.create_repo(repo_id=repo_id, repo_type=REPO_TYPE, private=False, exist_ok=True)
        except HfHubHTTPError as exc:
            raise SystemExit(
                "Could not create the Hugging Face repo automatically. "
                "If you already created it in the browser, rerun this command with --existing-repo. "
                f"Original error: {exc}"
            ) from exc
        print(f"Ready to upload to https://huggingface.co/{repo_id}")

    upload_readme(api, repo_id, readme_path, revision)

    for model_name in selected_models(models):
        print(f"\nUploading {model_name}...")
        for path in files_for_model(model_name, include_last):
            if not path.exists():
                raise SystemExit(f"Missing local file: {path}")

            path_in_repo = f"{model_name}/{path.name}"
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                revision=revision,
                commit_message=f"Upload {model_name}/{path.name}",
            )
            print(f"  uploaded {path_in_repo}")

    print("\nUpload complete.")


def cmd_download(repo_id, models, include_last, local_dir, revision):
    validate_repo_id(repo_id)
    output_root = Path(local_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for model_name in selected_models(models):
        print(f"\nDownloading {model_name}...")
        for path in files_for_model(model_name, include_last):
            remote_name = f"{model_name}/{path.name}"
            saved_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_name,
                repo_type=REPO_TYPE,
                revision=revision,
                local_dir=str(output_root),
            )
            print(f"  downloaded {remote_name} -> {saved_path}")

    print("\nDownload complete.")


def main():
    args = parse_args()
    if args.command == "list":
        cmd_list()
    elif args.command == "upload":
        include_last = not args.best_only
        cmd_upload(
            args.repo_id,
            args.models,
            include_last,
            args.revision,
            args.readme_path,
            args.existing_repo,
        )
    elif args.command == "download":
        include_last = not args.best_only
        cmd_download(args.repo_id, args.models, include_last, args.local_dir, args.revision)


if __name__ == "__main__":
    main()
