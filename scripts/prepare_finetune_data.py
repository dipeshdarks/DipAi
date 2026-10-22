#!/usr/bin/env python
"""Build fine-tune data (Mode B) from raw text sources or OASST threads.

Run from the DIPAI project root:

    # flat {"text": ...} samples (existing behavior):
    python scripts/prepare_finetune_data.py                       # 2000 stories from tinystories
    python scripts/prepare_finetune_data.py --input data/raw/my.jsonl --limit 500

    # OASST1 conversation threads for chat SFT (requires
    # scripts/download_oasst.py --with-metadata first):
    python scripts/prepare_finetune_data.py --oasst-threads

Outputs:
    flat mode     : one {"text": "..."} object per line (finetune.jsonl),
                    consumed by dipai/finetune.py (JsonlTextDataset) and
                    referenced by configs/finetune.yaml as dataset_path.
    threads mode  : sft_train.jsonl + sft_val.jsonl, one
                    {"messages": [{"role","content"}, ...], "tree_id": ...}
                    per line, split by conversation TREE (leak-free),
                    consumed by configs/sft_oasst.yaml (dipai_checkpoint
                    branch of dipai/finetune.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipai.conversations import (
    ConversationError,
    build_conversations,
    iter_oasst_messages,
    write_splits,
)
from dipai.dataset import InvalidDatasetError, iter_records
from dipai.utils import format_bytes, setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare JSONL fine-tune data")
    ap.add_argument("--input", "-i", nargs="+", default=["data/raw/tinystories_50k.txt"],
                    help="raw source file(s)/dir(s) (default: tinystories subset)")
    ap.add_argument("--output", "-o", default="data/processed/finetune.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="max samples to write (flat default: 2000; "
                         "threads mode default: 0 = all)")
    ap.add_argument("--text-key", default="text", help="field name for text (default: text)")
    ap.add_argument("--min-chars", type=int, default=40,
                    help="drop samples shorter than this (default: 40)")

    # ---- OASST thread mode -------------------------------------------------
    ap.add_argument("--oasst-threads", nargs="?", const="data/raw/oasst1_full.jsonl",
                    default=None, metavar="PATH",
                    help="reconstruct conversations from OASST metadata JSONL "
                         "(default path: data/raw/oasst1_full.jsonl from "
                         "download_oasst.py --with-metadata)")
    ap.add_argument("--train-out", default="data/processed/sft_train.jsonl",
                    help="threads mode: train JSONL (default: data/processed/sft_train.jsonl)")
    ap.add_argument("--val-out", default="data/processed/sft_val.jsonl",
                    help="threads mode: val JSONL (default: data/processed/sft_val.jsonl)")
    ap.add_argument("--val-fraction", type=float, default=0.1,
                    help="threads mode: fraction of trees held out (default: 0.1)")
    ap.add_argument("--seed", type=int, default=1337, help="split seed")
    ap.add_argument("--min-review-count", type=int, default=1,
                    help="threads mode: drop messages with fewer reviews "
                         "(default: 1; OASST1 export mostly has >=3)")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="threads mode: keep messages with review_count < min")
    ap.add_argument("--min-messages", type=int, default=2,
                    help="threads mode: min messages per conversation (default: 2)")
    ap.add_argument("--max-messages", type=int, default=10,
                    help="threads mode: max messages per conversation (default: 10)")
    ap.add_argument("--max-chars", type=int, default=4000,
                    help="threads mode: max chars per conversation (default: 4000)")
    return ap


def run_threads_mode(args, log) -> int:
    src = Path(args.oasst_threads)
    try:
        conversations, stats = build_conversations(
            iter_oasst_messages(src),
            val_fraction=args.val_fraction,
            seed=args.seed,
            lang="en",
            min_review_count=args.min_review_count,
            allow_unreviewed=args.allow_unreviewed,
            min_messages=args.min_messages,
            max_messages=args.max_messages,
            max_chars=args.max_chars,
            limit=args.limit if args.limit is not None else 0,
            input_name=str(src),
        )
        n_train, n_val = write_splits(conversations, args.train_out, args.val_out)
    except ConversationError as exc:
        log.error("thread reconstruction failed: %s", exc)
        return 2

    stats_path = Path(args.train_out).with_name(
        Path(args.train_out).stem + "_stats.json"
    )
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")

    print()
    print(stats.summary())
    print()
    print(f"  train              : {args.train_out} ({n_train:,} conversations)")
    print(f"  val                : {args.val_out} ({n_val:,} conversations)")
    print(f"  stats              : {stats_path}")
    print(f"  train size         : {format_bytes(Path(args.train_out).stat().st_size)}")
    print(f"  source / license   : OpenAssistant/oasst1 / Apache-2.0")
    print()
    if n_val == 0:
        log.warning("val split is empty - lower --val-fraction or add data")
    log.info("next: python -m dipai.finetune --config configs/sft_oasst.yaml --train")
    return 0


def main() -> int:
    log = setup_logging()
    args = build_arg_parser().parse_args()

    if args.oasst_threads is not None:
        return run_threads_mode(args, log)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    limit = 2000 if args.limit is None else args.limit

    written = 0
    skipped_short = 0
    seen = 0
    try:
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            for raw in args.input:
                p = Path(raw)
                if not p.exists():
                    raise InvalidDatasetError(f"input not found: {p}")
                paths = [p] if p.is_file() else sorted(
                    q for ext in ("*.txt", "*.jsonl", "*.csv") for q in p.rglob(ext)
                )
                for path in paths:
                    for record in iter_records(path, text_key=args.text_key):
                        seen += 1
                        text = " ".join(record.split()).replace("\x00", "")
                        if len(text) < args.min_chars:
                            skipped_short += 1
                            continue
                        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        written += 1
                        if limit and written >= limit:
                            break
                    if limit and written >= limit:
                        break
                if limit and written >= limit:
                    break
    except InvalidDatasetError as exc:
        log.error("%s", exc)
        return 2

    print()
    print(f"  samples written : {written:,} (from {seen:,} read, {skipped_short} too short)")
    print(f"  output          : {out.resolve()}")
    print(f"  size            : {format_bytes(out.stat().st_size)}")
    print()
    log.info("next: python -m dipai.finetune --train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
