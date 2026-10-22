#!/usr/bin/env python
"""Verify prepared DIPAI data before spending GPU time on it.

Run from the DIPAI project root:

    # packed pretraining corpus (bins + stats + optional raw re-split check):
    python scripts/verify_dataset.py --data-dir data/processed_50m \
        --tokenizer data/processed_50m/tokenizer.model \
        --raw data/raw/wikipedia_en_20231101.jsonl

    # SFT conversation splits (schema + tree-level leakage check):
    python scripts/verify_dataset.py --sft-train data/processed/sft_train.jsonl \
        --sft-val data/processed/sft_val.jsonl

Checks (each printed as PASS/FAIL; exit 0 only if all pass):
  P1 files exist and are non-empty (train.bin/val.bin/dataset_stats.json)
  P2 stats invariants: records_kept == read - dropped, train+val == kept,
     blocks == tokens // block_size, bin sizes == tokens * 2 bytes
  P3 token id range: every id < vocab (chunked scan, low memory)
  P4 (with --raw) the deterministic hash split over the raw corpus
     reproduces stats.train_records/val_records exactly (split is
     reproducible and the two splits are disjoint BY CONSTRUCTION)
  S1 (with --sft-train) schema: messages list, alternating roles, first
     role=user, non-empty content, >=2 messages
  S2 (with --sft-val) no tree_id appears in BOTH train and val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dipai.dataset import InvalidDatasetError, iter_records
from dipai.utils import format_bytes, setup_logging

FAILURES: list = []


def check(code: str, ok: bool, detail: str) -> None:
    print(f"  [{code}] {'PASS' if ok else 'FAIL'} - {detail}")
    if not ok:
        FAILURES.append(code)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Verify prepared DIPAI datasets")
    ap.add_argument("--data-dir", default=None,
                    help="packed corpus dir with train.bin/val.bin/dataset_stats.json")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer .model used for the bins (id-range check)")
    ap.add_argument("--raw", default=None,
                    help="raw source file to re-derive the split from (P4)")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--sft-train", default=None, help="SFT train JSONL (messages)")
    ap.add_argument("--sft-val", default=None, help="SFT val JSONL (messages)")
    return ap


# ---------------------------------------------------------------------------
# Packed corpus checks
# ---------------------------------------------------------------------------

def verify_corpus(args, log) -> int:
    d = Path(args.data_dir)
    print(f"Packed corpus: {d}")
    train_bin, val_bin, stats_path = d / "train.bin", d / "val.bin", d / "dataset_stats.json"

    missing = [str(p) for p in (train_bin, val_bin, stats_path)
               if not p.exists() or p.stat().st_size == 0]
    check("P1", not missing,
          "train.bin/val.bin/dataset_stats.json exist and are non-empty"
          if not missing else f"missing or empty: {missing}")
    if missing:
        return 1

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    schema = stats.get("schema_version", 1)
    block = int(stats.get("block_size", 0))
    vocab = int(stats.get("vocab_size", 0))
    if args.tokenizer:
        from dipai.tokenizer import DipaiTokenizer
        vocab = DipaiTokenizer.load(args.tokenizer).vocab_size
        check("P1b", vocab == stats.get("vocab_size", vocab),
              f"tokenizer vocab {vocab} == stats vocab {stats.get('vocab_size')}")

    kept = stats.get("records_kept")
    read = stats.get("records_read")
    drops = sum(stats.get(k, 0) for k in (
        "records_empty_dropped", "records_duplicate_dropped",
        "records_too_long_dropped", "records_invalid_json",
        "records_missing_field",
    ))
    inv1 = kept is not None and read is not None and kept == read - drops
    inv2 = (stats.get("train_records", -1) + stats.get("val_records", -2)) == kept
    inv3 = block > 0 and stats.get("train_blocks") == stats.get("train_tokens", -1) // block \
        and stats.get("val_blocks") == stats.get("val_tokens", -2) // block
    # Files hold exactly blocks*block_size tokens: the trailing partial
    # block is discarded by the packer (tokens counts include it).
    size_ok = (
        train_bin.stat().st_size == stats.get("train_blocks", -1) * block * 2
        and val_bin.stat().st_size == stats.get("val_blocks", -1) * block * 2
    )
    check("P2", inv1 and inv2 and inv3 and size_ok,
          f"stats invariants (schema v{schema}; kept={kept}={read}-{drops}, "
          f"records {stats.get('train_records')}+{stats.get('val_records')}, "
          f"blocks=tokens//{block}, bin sizes=blocks*{block}*2)")

    # P3: chunked id-range scan (never materializes the full array)
    max_id = -1
    n_tokens = 0
    for split_bin in (train_bin, val_bin):
        mm = np.memmap(str(split_bin), dtype=np.uint16, mode="r")
        step = 4_000_000
        for i in range(0, mm.shape[0], step):
            chunk = mm[i:i + step]
            cmax = int(chunk.max()) if chunk.shape[0] else -1
            max_id = max(max_id, cmax)
        n_tokens += int(mm.shape[0])
        try:
            mm._mmap.close()
        except Exception:
            pass
    check("P3", max_id < vocab,
          f"max token id {max_id} < vocab {vocab} over {n_tokens:,} tokens")

    if args.raw:
        import hashlib
        from dipai.dataset import MAX_RECORD_CHARS, _split_assign
        n_train = n_val = 0
        seen: set = set()
        for text in iter_records(args.raw, text_key=args.text_key):
            # Replicate the deterministic cleaning that precedes the split
            # (markers, strip, length cap, exact dedup) so the counts are
            # comparable to stats.*_records. Split is hash-of-text, so this
            # also proves the split is reproducible from the raw file.
            if text.startswith("\x00INVALID_JSON\x00:") or \
                    text.startswith("\x00MISSING_FIELD\x00:"):
                continue
            t = text.strip()
            if not t or len(t) > MAX_RECORD_CHARS:
                continue
            h = hashlib.sha1(t.encode("utf-8")).digest()
            if h in seen:
                continue
            seen.add(h)
            if _split_assign(t, stats.get("val_fraction", 0.05),
                             stats.get("seed", 1337)) == "val":
                n_val += 1
            else:
                n_train += 1
        exact = (n_train == stats.get("train_records")
                 and n_val == stats.get("val_records"))
        check("P4", exact,
              f"re-split of raw corpus reproduces stats exactly: "
              f"{n_train}/{n_val} vs {stats.get('train_records')}/"
              f"{stats.get('val_records')} (seed={stats.get('seed')}, "
              f"val_fraction={stats.get('val_fraction')})")
        log.info("P4: deterministic hash split over the raw corpus - each "
                 "record maps to exactly one split (disjoint by construction)")
    return 0


# ---------------------------------------------------------------------------
# SFT conversation checks
# ---------------------------------------------------------------------------

def _load_messages(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def verify_sft(args) -> int:
    trees_by_split: dict = {}
    for label, path_str in (("train", args.sft_train), ("val", args.sft_val)):
        if path_str is None:
            continue
        p = Path(path_str)
        print(f"SFT {label}: {p}")
        code = "S1t" if label == "train" else "S1v"
        check(code, p.exists() and p.stat().st_size > 0,
              "file exists and is non-empty")
        if not p.exists():
            continue
        bad = []
        n = 0
        trees = set()
        for line_no, obj in _load_messages(p):
            n += 1
            msgs = obj.get("messages")
            tree = obj.get("tree_id")
            if not isinstance(msgs, list) or len(msgs) < 2 or tree is None:
                bad.append(f"line {line_no}: structure")
                continue
            roles = [m.get("role") for m in msgs]
            if roles[0] != "user" or any(r not in ("user", "assistant") for r in roles):
                bad.append(f"line {line_no}: roles {roles[:4]}")
                continue
            if any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)):
                bad.append(f"line {line_no}: non-alternating {roles}")
                continue
            if any(not str(m.get("content", "")).strip() for m in msgs):
                bad.append(f"line {line_no}: empty content")
                continue
            trees.add(tree)
        check(code, not bad,
              f"{n:,} conversations valid (alternating user/assistant, >=2 msgs)"
              + ("" if not bad else f"; first issues: {bad[:3]}"))
        trees_by_split[label] = trees

    if "train" in trees_by_split and "val" in trees_by_split:
        overlap = trees_by_split["train"] & trees_by_split["val"]
        check("S2", not overlap,
              f"tree_id disjoint: {len(overlap)} overlapping trees between "
              f"train and val (leak-free split)")
    return 0


def main() -> int:
    log = setup_logging()
    args = build_arg_parser().parse_args()
    if not args.data_dir and not args.sft_train and not args.sft_val:
        log.error("nothing to verify: pass --data-dir and/or --sft-train/--sft-val")
        return 2

    if args.data_dir:
        verify_corpus(args, log)
    if args.sft_train or args.sft_val:
        verify_sft(args)

    print()
    if FAILURES:
        log.error("verification FAILED: %s", ", ".join(sorted(set(FAILURES))))
        return 1
    log.info("verification PASSED (all checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
