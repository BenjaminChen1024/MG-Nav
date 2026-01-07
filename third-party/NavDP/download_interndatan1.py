#!/usr/bin/env python3
"""
Download only selected subfolders from the Hugging Face gated dataset
`InternRobotics/InternData-N1` (e.g., v0.1-mini's `vln_n1/traj_data/matterport3d_*`).

It uses `huggingface_hub.snapshot_download`, which shows a progress bar and
downloads only the files that match `allow_patterns`.

Prereqs:
  pip install -U "huggingface_hub"

Auth (choose one):
  1) hf auth login            # interactive, recommended
  2) export HF_TOKEN=hf_xxx   # env var
  3) ~/.netrc with machine huggingface.co ...

Example:
  python download_interndatan1_subset.py \
    --dst /nas_dataset/wangbo/InternData-N1-mini \
    --revision v0.1-mini \
    --ignore "*.mp4" "*.avi" "*.mov"

By default, this script will fetch:
  vln_n1/traj_data/matterport3d_d435i/**
  vln_n1/traj_data/matterport3d_zed/**
"""

import argparse
import sys
from typing import List
from pathlib import Path

def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("[ERROR] huggingface_hub is not installed. Please run:\n"
              "  pip install -U huggingface_hub\n", file=sys.stderr)
        raise

    parser = argparse.ArgumentParser(
        description="Download selected folders from InternRobotics/InternData-N1 with progress."
    )
    parser.add_argument("--repo-id", default="InternRobotics/InternData-N1",
                        help="Hugging Face dataset repo id (default: %(default)s)")
    parser.add_argument("--revision", default="v0.1-mini",
                        help="Repo revision/branch/tag to use (default: %(default)s)")
    parser.add_argument("--dst", type=Path, required=True,
                        help="Destination directory to store downloaded files.")
    parser.add_argument("--no-symlinks", action="store_true",
                        help="Avoid using symlinks inside the cache (copy real files).")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Max parallel downloads (default: %(default)s).")
    parser.add_argument("--ignore", nargs="*", default=[],
                        help="Glob patterns to ignore (e.g., *.mp4 *.zip).")

    # Predefined allow patterns for the two folders requested
    default_allow = [
        "vln_n1/traj_data/matterport3d_d435i/**",
        "vln_n1/traj_data/matterport3d_zed/**",
    ]
    parser.add_argument("--allow", nargs="*", default=default_allow,
                        help="Glob allow patterns (defaults to the two matterport3d_* folders).")

    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    print("=== Download Config ===")
    print(f"Repo ID:          {args.repo_id}")
    print(f"Revision:         {args.revision}")
    print(f"Destination:      {args.dst}")
    print(f"Allow patterns:   {args.allow}")
    print(f"Ignore patterns:  {args.ignore}")
    print(f"Max workers:      {args.max_workers}")
    print(f"No symlinks:      {args.no_symlinks}")
    print("=======================")

    local_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.dst),
        local_dir_use_symlinks=not args.no_symlinks,
        allow_patterns=args.allow,
        ignore_patterns=args.ignore,
        max_workers=args.max_workers,
        resume_download=True,
    )

    print("\nDownloaded to:", local_dir)
    print("Done.")

if __name__ == "__main__":
    main()

"""
python download_interndatan1.py \
  --dst /home/wangbo/codes/NavDP/InternData-N1-mini \
  --revision v0.1-mini

"""