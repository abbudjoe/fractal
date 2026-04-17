#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload a Stage 0 token-cache artifact to a private Hugging Face dataset repo."
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id, e.g. user/name.")
    parser.add_argument("--artifact", type=Path, required=True, help="Token-cache tarball to upload.")
    parser.add_argument("--checksum", type=Path, help="Optional checksum sidecar to upload.")
    parser.add_argument("--readme", type=Path, help="Optional README/dataset card to upload.")
    parser.add_argument("--skip-create", action="store_true", help="Upload to an existing repo without calling create_repo.")
    parser.add_argument("--public", action="store_true", help="Create the dataset repo as public instead of private.")
    parser.add_argument("--commit-message", default="Upload Stage 0 token cache")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.artifact.exists():
        raise SystemExit(f"artifact does not exist: {args.artifact}")
    if args.checksum is not None and not args.checksum.exists():
        raise SystemExit(f"checksum does not exist: {args.checksum}")
    if args.readme is not None and not args.readme.exists():
        raise SystemExit(f"README does not exist: {args.readme}")

    api = HfApi()
    if not args.skip_create:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=not args.public,
            exist_ok=True,
        )
    api.upload_file(
        path_or_fileobj=args.artifact,
        path_in_repo=args.artifact.name,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
    )
    if args.checksum is not None:
        api.upload_file(
            path_or_fileobj=args.checksum,
            path_in_repo=args.checksum.name,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"{args.commit_message} checksum",
        )
    if args.readme is not None:
        api.upload_file(
            path_or_fileobj=args.readme,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"{args.commit_message} README",
        )
    print(f"https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
