"""
Normalizes a Cornell Grasping Dataset download (from whatever mirror,
whatever nesting) into the exact layout the reference loader
(robotic-grasping/utils/data/cornell_data.py) expects:

    <normalized_root>/01/pcd0100cpos.txt
    <normalized_root>/01/pcd0100r.png
    <normalized_root>/01/pcd0100d.tiff
    <normalized_root>/01/pcd0100.txt      (point cloud, if present)
    ...

The loader globs `<dataset-path>/*/pcd*cpos.txt` -- i.e. it needs exactly
one directory level between dataset-path and the files. Kaggle/other
mirrors often add extra nesting (e.g. an extra top-level folder from the
zip) or flatten everything, which breaks that glob even though every file
is present somewhere under the tree. This script finds every
`*cpos.txt` file anywhere under --raw-root, and symlinks it plus its
sibling files (same basename prefix, different suffix) into
--out-root/01/ (a single synthetic subfolder -- the loader doesn't care
about the specific number, only that there's one level of nesting).

Usage (Colab):
    !python scripts/fix_cornell_layout.py --raw-root data/cornell_raw --out-root data/cornell
Then point --dataset-path at data/cornell.
"""

import argparse
import os
from pathlib import Path

SIBLING_SUFFIXES = ["cpos.txt", "cneg.txt", "r.png", "d.tiff", ".txt"]


def normalize(raw_root: str, out_root: str):
    raw_root = Path(raw_root)
    out_dir = Path(out_root) / "01"
    out_dir.mkdir(parents=True, exist_ok=True)

    cpos_files = sorted(raw_root.rglob("*cpos.txt"))
    if not cpos_files:
        raise FileNotFoundError(
            f"No '*cpos.txt' files found anywhere under {raw_root}. "
            "Double-check the raw download actually extracted -- "
            "run `find {raw_root} -type f | head -20` to inspect it."
        )

    print(f"Found {len(cpos_files)} grasp annotation files under {raw_root}")

    n_linked = 0
    for cpos_path in cpos_files:
        # basename prefix, e.g. "pcd0100" from "pcd0100cpos.txt"
        prefix = cpos_path.name[: -len("cpos.txt")]
        parent = cpos_path.parent

        for suffix in SIBLING_SUFFIXES:
            candidate = parent / f"{prefix}{suffix}"
            if candidate.exists():
                link_path = out_dir / candidate.name
                if not link_path.exists():
                    try:
                        os.symlink(candidate.resolve(), link_path)
                    except OSError:
                        # symlinks can fail on some mounted filesystems (e.g.
                        # certain Drive setups) -- fall back to copying.
                        import shutil
                        shutil.copy(candidate, link_path)
                    n_linked += 1

    print(f"Linked {n_linked} files into {out_dir}")
    print(f"Set --dataset-path to: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, help="Wherever the raw download actually extracted")
    parser.add_argument("--out-root", required=True, help="Normalized output, e.g. data/cornell")
    args = parser.parse_args()
    normalize(args.raw_root, args.out_root)
