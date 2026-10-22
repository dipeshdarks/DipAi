#!/usr/bin/env python
"""Stream a manageable subset of TinyStories into data/raw/.

Run from the DIPAI project root:

    python scripts/download_training_data.py                 # 50,000 stories
    python scripts/download_training_data.py --max-stories 5000
    python scripts/download_training_data.py --output data/raw/my.txt

Why streaming: the full TinyStories dataset is ~7.62 GB - far too large for a
machine with ~6 GB RAM and limited disk. This script downloads record-by-record
and never holds more than one story in memory.

Source    : https://huggingface.co/datasets/roneneldan/TinyStories
License   : CDLA-Sharing-1.0 (review terms before redistribution/commercial use)
Format    : output .txt with one story per blank-line-separated record.

Document the download in data/README.md (license log) when you use it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipai.utils import format_bytes, setup_logging

DATASET_ID = "roneneldan/TinyStories"
DATASET_LICENSE = "CDLA-Sharing-1.0"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stream TinyStories subset to data/raw/")
    ap.add_argument("--max-stories", type=int, default=50_000,
                    help="number of stories to save (default: 50000; 0 = all)")
    ap.add_argument("--output", "-o", default="data/raw/tinystories_50k.txt",
                    help="output .txt path (default: data/raw/tinystories_50k.txt)")
    ap.add_argument("--split", default="train",
                    help="dataset split to stream (default: train)")
    ap.add_argument("--progress-every", type=int, default=5000,
                    help="print progress every N stories (default: 5000)")
    return ap


def main() -> int:
    log = setup_logging()
    args = build_arg_parser().parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("the 'datasets' package is required: pip install datasets")
        return 2

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "streaming %s (split=%s, license=%s) -> %s",
        DATASET_ID, args.split, DATASET_LICENSE, out_path,
    )
    t0 = time.perf_counter()

    try:
        dataset = load_dataset(DATASET_ID, split=args.split, streaming=True)
    except Exception as exc:  # noqa: BLE001 - network/auth errors surface here
        log.error("failed to open dataset stream: %s", exc)
        log.error("check your network connection to huggingface.co and retry.")
        return 1

    count = 0
    skipped = 0
    try:
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            for item in dataset:
                story = (item.get("text") or "").strip()
                if not story:
                    skipped += 1
                    continue
                # Strip NUL bytes; blank line = record boundary for prepare_data.
                f.write(story.replace("\x00", "") + "\n\n")
                count += 1
                if args.progress_every and count % args.progress_every == 0:
                    elapsed = time.perf_counter() - t0
                    log.info("saved %s stories (%s) in %ss",
                             f"{count:,}", format_bytes(out_path.stat().st_size),
                             f"{elapsed:.0f}")
                if args.max_stories and count >= args.max_stories:
                    break
    except KeyboardInterrupt:
        log.warning("interrupted - keeping the %s stories saved so far", f"{count:,}")
    except Exception as exc:  # noqa: BLE001
        log.error("download failed after %s stories: %s", f"{count:,}", exc)
        return 1

    elapsed = time.perf_counter() - t0
    print()
    print(f"  stories saved   : {count:,} ({skipped} empty skipped)")
    print(f"  output          : {out_path.resolve()}")
    print(f"  size            : {format_bytes(out_path.stat().st_size)}")
    print(f"  elapsed         : {elapsed:.0f}s")
    print(f"  source / license: {DATASET_ID} / {DATASET_LICENSE}")
    print()
    log.info("next: python scripts/prepare_data.py --input %s --vocab-size 4096", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
