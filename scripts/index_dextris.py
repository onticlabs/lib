#!/usr/bin/env python3
"""Build dextris_info.csv once so dataset startup does not scan video files."""

import argparse
from pathlib import Path

from ontic_data import DATASETS
from ontic_data.datasets.dextris import META_CSV_NAME, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["dextris", "robot-dextris"], default="dextris")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--refresh", action="store_true", help="rebuild after recordings change")
    args = parser.parse_args()
    path = args.root / META_CSV_NAME
    if path.exists() and not args.refresh:
        print(f"Using existing index: {path}; use --refresh to rebuild")
        return
    cfg = DATASETS[args.dataset](root=str(args.root), tasks=[])
    rows = write_metadata(cfg, path)
    print(f"Saved {len(rows)} recordings to {path}")


if __name__ == "__main__":
    main()
