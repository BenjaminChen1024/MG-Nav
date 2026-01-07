#!/usr/bin/env python3
"""
Copy all .npy files from each subfolder under a source root to the corresponding
subfolders under a destination root, preserving the folder structure.

Default paths are set to:
  SRC: /home/wangbo/codes/NavDP/assets/scenes/internscenes_commercial
  DST: /nas_dataset/wangbo/scene_n1/internscenes_commercial/scenes_commercial

Usage:
  python copy_npy_files.py
  python copy_npy_files.py --src /path/to/src --dst /path/to/dst
  python copy_npy_files.py --overwrite      # overwrite existing files
  python copy_npy_files.py --names file1.npy file2.npy  # only copy these names
  python copy_npy_files.py --dry-run        # show what would be copied

Notes:
- By default, existing files are skipped (no overwrite).
- If --names is provided, only files with those exact basenames are copied.
"""

import argparse
from pathlib import Path
import shutil
import sys

def copy_npy_files(src_root: Path, dst_root: Path, overwrite: bool, names: list[str] | None, dry_run: bool) -> tuple[int, int, int]:
    if not src_root.exists():
        print(f"[ERROR] Source root does not exist: {src_root}", file=sys.stderr)
        return (0, 0, 0)
    if not src_root.is_dir():
        print(f"[ERROR] Source root is not a directory: {src_root}", file=sys.stderr)
        return (0, 0, 0)

    total_found = 0
    copied = 0
    skipped = 0

    # Strategy: traverse all .npy files (or restricted names) under src_root
    patterns = ["*.npy"] if not names else names

    # Build a set for quick name filtering when names are provided
    name_filter = set(names) if names else None

    for npy_path in src_root.rglob("*.npy"):
        if name_filter and npy_path.name not in name_filter:
            continue

        total_found += 1
        rel_dir = npy_path.parent.relative_to(src_root)
        dst_dir = dst_root / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_file = dst_dir / npy_path.name

        if dst_file.exists() and not overwrite:
            skipped += 1
            print(f"[SKIP] Exists: {dst_file}")
            continue

        action = "COPY" if not dst_file.exists() else "OVERWRITE"
        print(f"[{action}] {npy_path} -> {dst_file}")
        if not dry_run:
            shutil.copy2(npy_path, dst_file)
        copied += 1

    print("\n=== Summary ===")
    print(f"Source root:      {src_root}")
    print(f"Destination root: {dst_root}")
    print(f"Found .npy files: {total_found}")
    print(f"Copied/Overwrote: {copied}")
    print(f"Skipped:          {skipped}")
    if dry_run:
        print("(Dry run: no files were actually copied.)")

    return (total_found, copied, skipped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy .npy files from src tree to dst tree, preserving structure.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/wangbo/codes/NavDP/assets/scenes/internscenes_commercial_py"),
        help="Source root directory (default: %(default)s)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/home/wangbo/codes/NavDP/assets/scenes/internscenes_commercial/scenes_commercial"),
        help="Destination root directory (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files at destination (default: skip existing).",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Optional list of exact .npy basenames to copy (e.g., grid.npy map.npy). If omitted, copies all .npy files found.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without copying files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_npy_files(args.src, args.dst, overwrite=args.overwrite, names=args.names, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
