#!/usr/bin/env python
"""DIPAI data preparation: validate -> clean -> tokenize -> pack.

Run from the DIPAI project root:

    # Train a tokenizer on raw files and pack train/val .bin files:
    python scripts/prepare_data.py --input data/raw/sample.txt

    # Reuse an existing tokenizer (skip training):
    python scripts/prepare_data.py --input data/raw/my.txt --tokenizer data/processed/tokenizer.model

    # Custom vocab / block size:
    python scripts/prepare_data.py --input data/raw --vocab-size 4096 --block-size 256

Inputs: individual files and/or directories (directories scan for .txt/.jsonl/.csv).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipai.dataset import (
    InvalidDatasetError,
    SUPPORTED_EXTENSIONS,
    iter_records,
    prepare_dataset,
    DatasetStats,
)
from dipai.tokenizer import DipaiTokenizer, TokenizerError, DEFAULT_TRAIN_SAMPLE_CHARS
from dipai.utils import setup_logging, format_duration


def collect_inputs(inputs: list[str]) -> list[Path]:
    """Expand files/directories into a validated list of dataset files."""
    out: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if not p.exists():
            raise InvalidDatasetError(f"input not found: {p}")
        if p.is_file():
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise InvalidDatasetError(
                    f"unsupported file type '{p.suffix}' for {p}. "
                    f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                )
            out.append(p)
        elif p.is_dir():
            for ext in sorted(SUPPORTED_EXTENSIONS):
                out.extend(sorted(p.rglob(f"*{ext}")))
    if not out:
        raise InvalidDatasetError(
            f"no dataset files found under: {inputs} "
            f"(looking for {sorted(SUPPORTED_EXTENSIONS)})"
        )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prepare a dataset for DIPAI scratch training")
    ap.add_argument(
        "--input", "-i", nargs="+", required=True,
        help="raw dataset file(s) and/or directories",
    )
    ap.add_argument(
        "--output-dir", "-o", default="data/processed",
        help="output directory for train.bin/val.bin (default: data/processed)",
    )
    ap.add_argument(
        "--tokenizer", "-t", default="data/processed/tokenizer.model",
        help="tokenizer path; trained here unless --reuse-tokenizer is set",
    )
    ap.add_argument(
        "--reuse-tokenizer", action="store_true",
        help="do not train; load the tokenizer at --tokenizer instead",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="allow training to OVERWRITE an existing tokenizer at --tokenizer "
             "(refused otherwise: one tokenizer path must never be silently "
             "replaced while other .bin corpora may depend on it)",
    )
    ap.add_argument("--vocab-size", type=int, default=8192,
                    help="SentencePiece vocab size when training (default: 8192)")
    ap.add_argument("--block-size", type=int, default=256,
                    help="packed window length in tokens (default: 256)")
    ap.add_argument("--val-fraction", type=float, default=0.05,
                    help="validation split fraction (default: 0.05)")
    ap.add_argument("--seed", type=int, default=1337, help="split seed")
    ap.add_argument("--text-key", default="text",
                    help="JSONL/CSV field containing the text (default: text)")
    ap.add_argument("--no-dedup", action="store_true", help="disable dedupe")
    ap.add_argument("--tokenizer-sample-chars", type=int, default=DEFAULT_TRAIN_SAMPLE_CHARS,
                    help="max characters used for tokenizer training (bounded for low-RAM)")
    # ---- corpus-size limits (streaming-safe; caps are approximate) --------
    ap.add_argument("--max-documents", type=int, default=0,
                    help="global cap on records kept (0 = no cap)")
    ap.add_argument("--max-bytes", type=int, default=0,
                    help="global cap on kept characters, approx bytes (0 = no cap)")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="wall-clock cap for the read phase in seconds (0 = no cap)")
    ap.add_argument("--target-documents", type=int, default=0,
                    help="overall document target; with --source-weights it is "
                         "split into proportional per-file quotas (0 = off)")
    ap.add_argument("--source-weights", type=float, nargs="+", default=None,
                    help="relative weights, one per dataset file AFTER directory "
                         "expansion (e.g. --input wiki.jsonl oasst.jsonl "
                         "--source-weights 0.85 0.15)")
    return ap


def main() -> int:
    log = setup_logging()
    ap = build_arg_parser()
    args = ap.parse_args()
    t0 = time.perf_counter()

    try:
        files = collect_inputs(args.input)
        log.info("found %d dataset file(s): %s", len(files),
                 ", ".join(str(f.name) for f in files))

        # ---- tokenizer ----------------------------------------------------
        if args.reuse_tokenizer:
            tok = DipaiTokenizer.load(args.tokenizer)
            log.info("reusing tokenizer: %s (vocab=%d)", args.tokenizer, tok.vocab_size)
        else:
            if Path(args.tokenizer).exists() and not args.force:
                log.error(
                    "refusing to overwrite existing tokenizer: %s\n"
                    "Other processed .bin corpora may depend on it. Re-run "
                    "with --reuse-tokenizer (keep it), --force (replace it), "
                    "or --tokenizer <new path> (train a separate one).",
                    args.tokenizer,
                )
                return 2
            log.info("training SentencePiece BPE (vocab_size=%d) ...", args.vocab_size)

            def stream():
                for f in files:
                    yield from iter_records(f, text_key=args.text_key)

            tok = DipaiTokenizer.train(
                stream(),
                model_prefix=args.tokenizer,
                vocab_size=args.vocab_size,
                sample_chars=args.tokenizer_sample_chars,
            )
            log.info("tokenizer saved: %s (vocab=%d)", args.tokenizer, tok.vocab_size)

        # ---- validate vocab vs model expectation --------------------------
        if tok.vocab_size < 256:
            raise TokenizerError(
                f"tokenizer vocab_size {tok.vocab_size} is too small; need >= 256"
            )

        # ---- clean + pack -------------------------------------------------
        stats: DatasetStats = prepare_dataset(
            input_paths=files,
            output_dir=args.output_dir,
            tokenizer=tok,
            block_size=args.block_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            text_key=args.text_key,
            dedup=not args.no_dedup,
            max_documents=args.max_documents,
            max_bytes=args.max_bytes,
            max_seconds=args.max_seconds,
            target_documents=args.target_documents,
            source_weights=args.source_weights,
        )
        tok.save_meta(args.tokenizer, extra={"block_size": args.block_size})
        if stats.truncated_by:
            log.warning(
                "corpus limits were reached during reading (%s) - see stats.limits",
                stats.truncated_by,
            )

    except (InvalidDatasetError, TokenizerError) as exc:
        log.error("preparation failed: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface any failure clearly
        log.exception("preparation failed unexpectedly: %s", exc)
        return 1

    print()
    print(stats.summary())
    print(f"  elapsed             : {format_duration(time.perf_counter() - t0)}")
    print(f"  output              : {args.output_dir}")
    print()
    log.info("preparation complete. Next: python scripts/train_scratch (see README)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
