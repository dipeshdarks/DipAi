#!/usr/bin/env python
"""Stream English OASST1 (OpenAssistant) messages into data/raw/ as JSONL.

Run from the DIPAI project root:

    python scripts/download_oasst.py                  # English messages only
    python scripts/download_oasst.py --lang en --limit 20000

Output records: {"text": "User: ..." | "Assistant: ..."} - one message per
line, so the existing prepare_data.py pipeline can ingest it directly.

With --with-metadata the output instead preserves the thread fields
(message_id / parent_id / message_tree_id / role / lang / deleted /
review_count / review_result / tree_state / created_date), which is what
scripts/prepare_finetune_data.py --oasst-threads needs to reconstruct real
user/assistant conversations with a leak-free train/val split.

Source    : https://huggingface.co/datasets/OpenAssistant/oasst1
License   : Apache 2.0 (per the dataset card)
Scope     : individual messages (role-tagged), NOT reconstructed threads.
            Good as a real-human TEXT CORPUS for scratch training. For
            supervised chat fine-tuning, proper user/assistant PAIRS are
            needed (see scripts/prepare_finetune_data.py --help notes).

Document the download in data/README.md (license log) when you use it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipai.utils import format_bytes, setup_logging

DATASET_ID = "OpenAssistant/oasst1"
DATASET_LICENSE = "Apache-2.0"
VALID_ROLES = ("prompter", "assistant")
# Fields needed to reconstruct conversation threads (kept by --with-metadata).
METADATA_FIELDS = (
    "message_id", "parent_id", "message_tree_id", "role", "lang",
    "deleted", "review_count", "review_result", "tree_state", "created_date",
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stream OASST1 messages to data/raw/")
    ap.add_argument("--output", "-o", default="data/raw/oasst1_english.jsonl",
                    help="output JSONL path (default: data/raw/oasst1_english.jsonl; "
                         "with --with-metadata: data/raw/oasst1_full.jsonl)")
    ap.add_argument("--lang", default="en",
                    help="language filter (default: en; '' = all languages)")
    ap.add_argument("--split", default="train", help="dataset split (default: train)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max messages to save (0 = all)")
    ap.add_argument("--with-metadata", action="store_true",
                    help="keep thread metadata (message_id/parent_id/role/...) "
                         "instead of role-tagged text; required for "
                         "prepare_finetune_data.py --oasst-threads")
    ap.add_argument("--progress-every", type=int, default=5000,
                    help="print progress every N messages (default: 5000)")
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
    if args.with_metadata and args.output == "data/raw/oasst1_english.jsonl":
        out_path = Path("data/raw/oasst1_full.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("streaming %s (split=%s, lang=%r, license=%s, metadata=%s) -> %s",
             DATASET_ID, args.split, args.lang or "*", DATASET_LICENSE,
             args.with_metadata, out_path)
    t0 = time.perf_counter()

    try:
        dataset = load_dataset(DATASET_ID, split=args.split, streaming=True)
    except Exception as exc:  # noqa: BLE001 - network/auth errors surface here
        log.error("failed to open dataset stream: %s", exc)
        log.error("check your network connection to huggingface.co and retry.")
        return 1

    saved = 0
    scanned = 0
    try:
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            for row in dataset:
                scanned += 1
                if args.lang and row.get("lang") != args.lang:
                    continue
                text = (row.get("text") or "").strip()
                role = row.get("role", "")
                if not text or role not in VALID_ROLES:
                    continue
                text = text.replace("\x00", "")
                if args.with_metadata:
                    rec = {k: row.get(k) for k in METADATA_FIELDS}
                    rec["text"] = text
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    label = "User" if role == "prompter" else "Assistant"
                    f.write(json.dumps({"text": f"{label}: {text}"}, ensure_ascii=False) + "\n")
                saved += 1
                if args.progress_every and saved % args.progress_every == 0:
                    log.info("saved %s messages (scanned %s)", f"{saved:,}", f"{scanned:,}")
                if args.limit and saved >= args.limit:
                    break
    except KeyboardInterrupt:
        log.warning("interrupted - keeping the %s messages saved so far", f"{saved:,}")
    except Exception as exc:  # noqa: BLE001
        log.error("download failed after %s messages: %s", f"{saved:,}", exc)
        return 1

    elapsed = time.perf_counter() - t0
    print()
    print(f"  messages saved  : {saved:,} (scanned {scanned:,}, lang={args.lang or '*'})")
    print(f"  output          : {out_path.resolve()}")
    print(f"  size            : {format_bytes(out_path.stat().st_size)}")
    print(f"  elapsed         : {elapsed:.0f}s")
    print(f"  source / license: {DATASET_ID} / {DATASET_LICENSE}")
    print()
    if args.with_metadata:
        log.info("next: python scripts/prepare_finetune_data.py --oasst-threads %s "
                 "(train/val conversation JSONL)", out_path)
    else:
        log.info("next: python scripts/prepare_data.py --input %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
