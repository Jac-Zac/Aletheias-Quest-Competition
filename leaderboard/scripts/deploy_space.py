#!/usr/bin/env python3
"""Deploy the runner to the HuggingFace Docker Space.

Uploads the package source + Dockerfile + runner.yaml to the Space, and sets the
Space's root README.md from ``space/README.md`` (which carries the HF Space
front-matter). Tests/scripts/dev cruft are excluded.

    python scripts/deploy_space.py --token hf_xxx
    python scripts/deploy_space.py --token hf_xxx --space NDIF/leaderboard-dev

The token must have WRITE access to the Space's org. After deploying, set the
Space secret ``HF_TOKEN`` (org token) so the runner can read the private
datasets and write to the results bucket.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

UPLOAD_IGNORE = [
    ".git/*", "tests/*", "scripts/*", "space/*",
    "README.md", "DESIGN.md",
    "*.pyc", "__pycache__/*", "*.egg-info/*", ".pytest_cache/*",
    "results.jsonl", "*.zip", ".gitignore", "runner-test.yaml",
    # Fargate migration tooling: MUST NOT ship to the Space — it holds the AWS
    # dispatcher key, a multi-GB venv, and the baked build context. The Space only
    # needs the package source + Dockerfile + runner.yaml.
    "fargate/*", "FARGATE-MIGRATION.md", "admin_token.txt",
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space", default="NDIF/leaderboard-dev")
    p.add_argument("--token", required=True, help="HF token with write access to the org")
    args = p.parse_args(argv)

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)

    print(f"uploading runner source -> {args.space}")
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=args.space,
        repo_type="space",
        ignore_patterns=UPLOAD_IGNORE,
        commit_message="Deploy aletheia_runner",
    )

    print("setting Space README.md from space/README.md")
    api.upload_file(
        path_or_fileobj=str(ROOT / "space" / "README.md"),
        path_in_repo="README.md",
        repo_id=args.space,
        repo_type="space",
        commit_message="Set Space metadata",
    )

    print(f"\ndone -> https://huggingface.co/spaces/{args.space}")
    print("Remember to set the Space secret HF_TOKEN (org token) in Settings.")
    print("Note: /data is a persistent HF bucket mount — to refresh the predownload "
          "cache after re-pushing a dataset, clear the bucket's cache/ prefix.")


if __name__ == "__main__":
    main()
