#!/usr/bin/env python
"""Stream a bounded English-Wikipedia subset from Hugging Face parquet shards.

Run from the DIPAI project root:

    # ~60k documents (defaults: 60k docs / 500 MB), spread across ALL
    # parquet shards so the subset spans the whole article-id space:
    python scripts/download_wikipedia.py

    # smaller/larger subset:
    python scripts/download_wikipedia.py --max-documents 80000 --max-bytes 800000000

    # resume an interrupted download (continues the .part file):
    python scripts/download_wikipedia.py --resume

Source        : https://huggingface.co/datasets/wikimedia/wikipedia
                (public mirror of https://dumps.wikimedia.org English dumps;
                this tool NEVER scrapes the live wikipedia.org website)
Snapshot/config: 20231101.en (English, frozen snapshot -> reproducible)
License       : GFDL-1.3 OR CC-BY-SA-3.0 (dual license per the dataset card).
                Never describe Wikipedia text as public domain.
Shards        : resolved live from the HF datasets-server /parquet API.
                Parquet row groups are 1000 rows; only the row groups actually
                needed are fetched (HTTP range reads).

Spread sampling: with --max-documents N (and not --no-spread), each shard
contributes ~ceil(N / shards) documents, so the subset is taken evenly from
the entire dump instead of a prefix (which would be biased toward the oldest
articles). If a byte/time cap stops reading before every shard is visited,
that is recorded in the manifest (spread_complete=false).

Memory/disk behavior:
  * Parquet row batches stream through pyarrow -> one batch in RAM at a time.
  * Output goes to <output>.part (renamed on success); a small .progress.json
    sidecar records the shard being read so --resume can continue without
    re-reading finished shards. Dedup ids are rebuilt from the .part file.
  * RAM and free-disk guards abort BEFORE any download with a real diagnostic.
  * A manifest (JSON) records source URLs, snapshot, retrieval date, counts,
    limits, shard coverage and license next to the data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipai.utils import format_bytes, setup_logging

DATASET_ID = "wikimedia/wikipedia"
CONFIG = "20231101.en"
SPLIT = "train"
LICENSE = "GFDL-1.3 OR CC-BY-SA-3.0"
REPO_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"
DUMPS_URL = "https://dumps.wikimedia.org/other/20231101/"
PARQUET_API = (
    "https://datasets-server.huggingface.co/parquet"
    "?dataset=wikimedia%2Fwikipedia&config={config}&split={split}"
)
MIN_RAM_FREE = 256 * 1024 * 1024         # abort below this available RAM
                                            # (this tool peaks around ~100 MB:
                                            # one 64-row pyarrow batch + IO)
DISK_MARGIN = 500 * 1024 * 1024          # keep this much disk free
BYTES_PER_DOC_ESTIMATE = 15_000          # conservative pre-flight estimate


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Stream a bounded English Wikipedia subset to data/raw/"
    )
    ap.add_argument("--output", "-o", default="data/raw/wikipedia_en_20231101.jsonl",
                    help="output JSONL (default: data/raw/wikipedia_en_20231101.jsonl)")
    ap.add_argument("--config", default=CONFIG,
                    help=f"dataset config/snapshot (default: {CONFIG})")
    ap.add_argument("--max-documents", type=int, default=60000,
                    help="stop after N documents (default 60000; 0 = no cap)")
    ap.add_argument("--max-bytes", type=int, default=500_000_000,
                    help="stop after ~N bytes of JSONL written "
                         "(default 500MB; 0 = off)")
    ap.add_argument("--min-chars", type=int, default=100,
                    help="drop articles shorter than N chars (default 100)")
    ap.add_argument("--no-spread", action="store_true",
                    help="read shards sequentially instead of taking an even "
                         "quota of documents from every shard")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted <output>.part download")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output/partial (never implicit)")
    ap.add_argument("--progress-every", type=int, default=5000,
                    help="log progress every N documents (default 5000)")
    return ap


def _ram_available() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        # Fallback: Windows MEMORYSTATUSEX
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys)
        return -1  # unknown


def resolve_shards(config: str, split: str, log) -> List[str]:
    """Fetch the parquet shard URLs from the HF datasets-server API."""
    import urllib.request
    url = PARQUET_API.format(config=config, split=split)
    log.info("resolving parquet shards: %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    files = data.get("parquet_files", [])
    if not files:
        raise RuntimeError(
            f"datasets-server returned no parquet files for {DATASET_ID} "
            f"config={config} split={split}: {str(data)[:300]}"
        )
    return [f["url"] for f in files]


def _seen_ids_from_part(part_path: Path) -> Set[bytes]:
    """Rebuild the dedup set of article ids from a partial download."""
    import hashlib
    seen: Set[bytes] = set()
    with open(part_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                art_id = json.loads(line).get("id")
            except json.JSONDecodeError:
                continue
            if art_id is not None:
                seen.add(hashlib.sha1(str(art_id).encode("utf-8")).digest())
    return seen


def main() -> int:
    log = setup_logging()
    args = build_arg_parser().parse_args()

    max_docs = args.max_documents

    out_path = Path(args.output)
    part_path = Path(str(out_path) + ".part")
    progress_path = Path(str(out_path) + ".progress.json")
    manifest_path = Path(str(out_path).replace(".jsonl", "") + "_manifest.json")

    # ---- refuse silent overwrite / handle resume ---------------------------
    if out_path.exists() and not args.force:
        log.error(
            "output already exists: %s (refusing to overwrite). "
            "Re-run with --force to replace it, or use a different --output.",
            out_path,
        )
        return 2
    if part_path.exists() and not (args.resume or args.force):
        log.error(
            "partial download found: %s. Re-run with --resume to continue it, "
            "or --force to start over.",
            part_path,
        )
        return 2
    if args.force:
        for p in (out_path, part_path, progress_path):
            if p.exists():
                p.unlink()

    # ---- pre-flight: RAM + disk --------------------------------------------
    ram_free = _ram_available()
    if 0 <= ram_free < MIN_RAM_FREE:
        log.error(
            "ABORT before download: only %s RAM available (need >= %s). "
            "Free memory and retry. Nothing was downloaded.",
            format_bytes(ram_free), format_bytes(MIN_RAM_FREE),
        )
        return 3

    out_path.parent.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(str(out_path.parent)).free
    want = 0
    if max_docs > 0:
        want = int(max_docs * BYTES_PER_DOC_ESTIMATE)
    if args.max_bytes > 0:
        want = max(want, int(args.max_bytes))
    if free_disk < want + DISK_MARGIN:
        log.error(
            "ABORT before download: %s free disk < needed %s (+%s margin). "
            "Lower --max-documents/--max-bytes or free disk space.",
            format_bytes(free_disk), format_bytes(want), format_bytes(DISK_MARGIN),
        )
        return 3
    log.info("pre-flight OK: RAM free %s | disk free %s | cap %s docs / %s bytes",
             format_bytes(max(ram_free, 0)), format_bytes(free_disk),
             f"{max_docs:,}" if max_docs else "unlimited",
             format_bytes(args.max_bytes) if args.max_bytes else "unlimited")

    # ---- resolve shards -----------------------------------------------------
    try:
        shards = resolve_shards(args.config, SPLIT, log)
    except Exception as exc:
        log.error("could not resolve parquet shards: %s", exc)
        log.error("check network access to huggingface.co / datasets-server.huggingface.co")
        return 1
    num_shards = len(shards)
    log.info("%d parquet shard(s) available", num_shards)

    spread = not args.no_spread and max_docs > 0 and num_shards > 0
    # Floor division so sum(quotas) <= max_documents: the global doc cap can
    # never cut the spread short before every shard has had its share.
    per_shard_quota = max(1, max_docs // num_shards) if spread else 0
    if spread:
        log.info("spread sampling: ~%d documents per shard x %d shards",
                 per_shard_quota, num_shards)

    # ---- resume state --------------------------------------------------------
    seen: Set[bytes] = set()
    start_shard = 0
    docs_written = 0
    bytes_written = 0
    resuming = part_path.exists() and args.resume and not args.force
    if resuming:
        seen = _seen_ids_from_part(part_path)
        docs_written = len(seen)             # one id per written document
        bytes_written = part_path.stat().st_size
        if progress_path.exists():
            try:
                start_shard = int(json.loads(progress_path.read_text())["shard"])
            except Exception:
                start_shard = 0
        log.info("resuming: %s docs in %s (%s), starting at shard %d/%d",
                 f"{docs_written:,}", part_path.name, format_bytes(bytes_written),
                 start_shard + 1, num_shards)
    elif part_path.exists():
        part_path.unlink()

    mode = "a" if resuming else "w"
    t0 = time.perf_counter()

    import hashlib
    try:
        import pyarrow.parquet as pq
        import fsspec
    except ImportError as exc:
        log.error("missing dependency for streaming download: %s", exc)
        return 2

    def _write_progress(shard_idx: int) -> None:
        try:
            progress_path.write_text(
                json.dumps({"shard": shard_idx, "docs": docs_written}), encoding="utf-8"
            )
        except OSError:
            pass

    stop = False
    shards_visited = 0
    stop_cause = ""
    try:
        with open(part_path, mode, encoding="utf-8", newline="\n") as out:
            for shard_idx in range(start_shard, num_shards):
                if stop:
                    break
                shard_url = shards[shard_idx]
                _write_progress(shard_idx)
                log.info("shard %d/%d: %s", shard_idx + 1, num_shards,
                         shard_url.rsplit("/", 1)[-1])
                try:
                    fh = fsspec.open(shard_url, "rb", block_size=8 * 1024 * 1024).open()
                except Exception as exc:
                    log.error("cannot open shard %s: %s", shard_url, exc)
                    return 1
                shard_docs = 0
                with fh:
                    pf = pq.ParquetFile(fh)
                    for batch in pf.iter_batches(batch_size=64,
                                                 columns=["id", "url", "title", "text"]):
                        for row in batch.to_pylist():
                            art_id = row.get("id")
                            text = (row.get("text") or "").replace("\x00", "").strip()
                            if len(text) < args.min_chars:
                                continue
                            h = hashlib.sha1(str(art_id).encode("utf-8")).digest()
                            if h in seen:
                                continue
                            seen.add(h)
                            rec = {
                                "id": art_id,
                                "url": row.get("url") or "",
                                "title": (row.get("title") or "").replace("\x00", ""),
                                "text": text,
                            }
                            line = json.dumps(rec, ensure_ascii=False) + "\n"
                            out.write(line)
                            docs_written += 1
                            bytes_written += len(line)
                            shard_docs += 1

                            if args.progress_every and docs_written % args.progress_every == 0:
                                out.flush()
                                _write_progress(shard_idx)
                                elapsed = time.perf_counter() - t0
                                log.info("  %s docs | %s | %.0f docs/s | shard %d/%d",
                                         f"{docs_written:,}",
                                         format_bytes(bytes_written),
                                         docs_written / elapsed if elapsed > 0 else 0,
                                         shard_idx + 1, num_shards)
                                ram_now = _ram_available()
                                if 0 <= ram_now < MIN_RAM_FREE:
                                    log.error(
                                        "ABORT mid-download: RAM free dropped to %s. "
                                        "Partial file kept at %s (resume with --resume).",
                                        format_bytes(ram_now), part_path,
                                    )
                                    return 3

                            if max_docs and docs_written >= max_docs:
                                stop, stop_cause = True, "max_documents"
                                break
                            if args.max_bytes and bytes_written >= args.max_bytes:
                                stop, stop_cause = True, "max_bytes"
                                break
                            if per_shard_quota and shard_docs >= per_shard_quota:
                                break  # this shard's share is done - next shard
                        if stop or (per_shard_quota and shard_docs >= per_shard_quota):
                            break  # leave the batch loop too (quota or cap)
                if stop:
                    break
                shards_visited += 1
                if per_shard_quota and shard_docs >= per_shard_quota:
                    log.info("  shard quota reached (%d docs)", shard_docs)
                elif per_shard_quota:
                    log.info("  shard exhausted at %d docs (quota %d)",
                             shard_docs, per_shard_quota)
    except KeyboardInterrupt:
        _write_progress(min(shard_idx, num_shards - 1))
        log.warning("interrupted - partial file kept: %s (resume with --resume)",
                    part_path)
        return 130
    except Exception as exc:
        if "shard_idx" in dir():
            _write_progress(shard_idx)
        log.error("download failed after %s docs: %s", exc)
        log.error("partial file kept: %s (resume with --resume)", part_path)
        return 1

    if docs_written == 0:
        log.error("no documents written - check limits/filters/network. "
                  "Partial file left at %s", part_path)
        return 1

    # ---- success: rename .part -> final, write manifest ---------------------
    if out_path.exists():
        out_path.unlink()
    part_path.rename(out_path)
    if progress_path.exists():
        progress_path.unlink()
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    spread_complete = bool(spread and shards_visited >= num_shards and not stop_cause)
    manifest = {
        "tool": "scripts/download_wikipedia.py",
        "dataset_id": DATASET_ID,
        "repo_url": REPO_URL,
        "config": args.config,
        "split": SPLIT,
        "snapshot": args.config.split(".")[0],
        "upstream_source": DUMPS_URL,
        "license": LICENSE,
        "license_note": "Wikipedia text is dual-licensed GFDL-1.3 / CC-BY-SA-3.0; "
                        "it is NOT public domain. Attribute Wikipedia contributors.",
        "retrieved_at": retrieved_at,
        "parquet_api": PARQUET_API.format(config=args.config, split=SPLIT),
        "parquet_shards_total": num_shards,
        "shards_visited": shards_visited,
        "spread_sampling": spread,
        "spread_complete": spread_complete,
        "stop_cause": stop_cause or "complete",
        "output": str(out_path),
        "documents": docs_written,
        "bytes": bytes_written,
        "limits": {
            "max_documents": max_docs or None,
            "max_bytes": args.max_bytes or None,
            "min_chars": args.min_chars,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if spread and not spread_complete:
        log.warning(
            "spread sampling incomplete: stopped (%s) after visiting %d/%d "
            "shards - the subset over-represents lower shard ids. Raise the "
            "caps or re-run with --force for full spread.",
            stop_cause, shards_visited, num_shards,
        )

    elapsed = time.perf_counter() - t0
    print()
    print(f"  documents        : {docs_written:,}")
    print(f"  bytes            : {format_bytes(bytes_written)}")
    print(f"  shards visited   : {shards_visited}/{num_shards} "
          f"(spread {'complete' if spread_complete else 'INCOMPLETE' if spread else 'off'})")
    print(f"  output           : {out_path.resolve()}")
    print(f"  manifest         : {manifest_path.resolve()}")
    print(f"  snapshot/license : {args.config} / {LICENSE}")
    print(f"  elapsed          : {elapsed:.0f}s")
    print()
    log.info("next: python scripts/prepare_data.py --input %s --vocab-size 4096 ...",
             out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
