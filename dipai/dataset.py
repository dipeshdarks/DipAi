"""Dataset preparation and loading for DIPAI scratch training.

Pipeline:
    raw files (.txt/.jsonl/.csv)
        -> streaming record reader
        -> validation + UTF-8 normalization + dedupe + empty-drop
        -> document-level deterministic train/val split (sha1 hash of seed+text)
           written to temp JSONL files (never a record list in RAM)
        -> tokenize (SentencePiece)
        -> pack into fixed-length windows (block_size), streamed block-by-block
        -> uint16 .bin files (train.bin, val.bin)

Memory rules (this project targets ~6 GB RAM machines):
  * Records stream from disk one at a time; nothing large is held in RAM.
  * Split and packing are streaming: peak memory is one document plus the
    bounded dedup hash set, never the corpus or its token stream.
  * Dedup uses hashes only when the record set stays under a cap; beyond the
    cap dedup switches to a bounded LRU-style sample (documented in stats).
  * The packed .bin files are read back through numpy.memmap, so training
    never loads the full token array into memory.

Raises InvalidDatasetError (a clear, actionable message) whenever the input is
empty or unusable - training code must never start on silently-empty data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("dipai.dataset")

SUPPORTED_EXTENSIONS = {".txt", ".jsonl", ".csv"}

# Hard caps so tiny-RAM machines stay safe.
DEDUP_MAX_ENTRIES = 500_000
MAX_RECORD_CHARS = 200_000


class InvalidDatasetError(Exception):
    """Raised when a dataset is empty, unreadable, or structurally invalid."""


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class DatasetStats:
    schema_version: int = 2          # v2: streaming prep, doc-level hash split
    files: List[str] = field(default_factory=list)
    records_read: int = 0
    records_empty_dropped: int = 0
    records_duplicate_dropped: int = 0
    records_too_long_dropped: int = 0
    records_kept: int = 0
    records_invalid_json: int = 0
    records_missing_field: int = 0
    chars_total: int = 0
    train_records: int = 0
    val_records: int = 0
    train_tokens: int = 0
    val_tokens: int = 0
    train_blocks: int = 0
    val_blocks: int = 0
    block_size: int = 0
    vocab_size: int = 0
    dedup_mode: str = "exact"       # exact | sample-capped | disabled
    seed: int = 0                   # split seed (reproducibility)
    val_fraction: float = 0.0       # requested validation fraction
    source_weights: List[float] = field(default_factory=list)
    limits: Dict[str, float] = field(default_factory=dict)   # requested caps
    truncated_by: str = ""          # limits that stopped reading ("" = none)

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "Dataset statistics",
            "-" * 46,
            f"  schema version     : {self.schema_version}",
            f"  files              : {len(self.files)}",
            f"  records read       : {self.records_read}",
            f"  empty dropped      : {self.records_empty_dropped}",
            f"  duplicates dropped : {self.records_duplicate_dropped} ({self.dedup_mode})",
            f"  invalid json       : {self.records_invalid_json}",
            f"  missing field      : {self.records_missing_field}",
            f"  too long dropped   : {self.records_too_long_dropped}",
            f"  records kept       : {self.records_kept}",
            f"  characters         : {self.chars_total}",
            f"  train/val records  : {self.train_records} / {self.val_records}",
            f"  train/val tokens   : {self.train_tokens} / {self.val_tokens}",
            f"  train/val blocks   : {self.train_blocks} / {self.val_blocks}",
            f"  block size         : {self.block_size}",
            f"  vocab size         : {self.vocab_size}",
            f"  split (seed)       : {self.val_fraction} val ({self.seed})",
        ]
        if self.limits:
            lines.append(f"  limits             : {self.limits}")
            lines.append(f"  limits reached     : {self.truncated_by or 'none'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streaming readers
# ---------------------------------------------------------------------------

def _detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise InvalidDatasetError(
            f"unsupported file type '{ext}' for {path}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return ext[1:]


def iter_records(path: str | Path, text_key: str = "text") -> Iterator[str]:
    """Yield text records from a .txt / .jsonl / .csv file, one at a time.

    UTF-8 with errors='replace' so a single bad byte never kills a run;
    replaced chars are counted implicitly by downstream cleaning stats.
    """
    path = Path(path)
    if not path.exists():
        raise InvalidDatasetError(f"dataset file not found: {path}")
    if path.stat().st_size == 0:
        raise InvalidDatasetError(f"dataset file is empty (0 bytes): {path}")

    fmt = _detect_format(path)

    if fmt == "txt":
        # Paragraph-oriented: blank lines separate records, long paragraphs
        # are emitted whole (chunking happens at packing time).
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            buf: List[str] = []
            for line in f:
                if line.strip() == "":
                    if buf:
                        yield "".join(buf).strip()
                        buf = []
                else:
                    buf.append(line)
            if buf:
                yield "".join(buf).strip()
        return

    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Yield a marker the cleaner counts instead of raising;
                    # one bad line must not abort a whole preparation run.
                    yield f"\x00INVALID_JSON\x00:{line_no}"
                    continue
                if isinstance(obj, str):
                    yield obj
                    continue
                if not isinstance(obj, dict):
                    yield f"\x00INVALID_JSON\x00:{line_no}"
                    continue
                val = obj.get(text_key)
                if isinstance(val, str):
                    yield val
                elif val is None:
                    yield f"\x00MISSING_FIELD\x00:{line_no}"
                else:
                    # e.g. list of messages - serialize shallowly
                    yield json.dumps(val, ensure_ascii=False)
        return

    if fmt == "csv":
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise InvalidDatasetError(f"CSV has no header row: {path}")
            fields = [c.strip().lower() for c in reader.fieldnames]
            if text_key.lower() not in fields:
                raise InvalidDatasetError(
                    f"CSV {path} has no '{text_key}' column. Found: {reader.fieldnames}"
                )
            actual_key = reader.fieldnames[fields.index(text_key.lower())]
            for row_no, row in enumerate(reader, 2):
                val = row.get(actual_key)
                if val is None:
                    yield f"\x00MISSING_FIELD\x00:{row_no}"
                else:
                    yield val


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_records(
    records: Iterable[str],
    stats: DatasetStats,
    dedup: bool = True,
    max_record_chars: int = MAX_RECORD_CHARS,
    dedup_seen: Optional[set] = None,
) -> Iterator[str]:
    """Validate + normalize a stream of records.

    Applies: marker handling, whitespace normalization, empty-drop,
    length-drop, and exact dedupe (with a documented cap).

    Pass `dedup_seen` to share one dedup set across multiple sources
    (cross-file duplicates count as duplicates; stats stay consistent).
    """
    seen: set = set() if dedup_seen is None else dedup_seen
    dedup_count = 0
    for rec in records:
        stats.records_read += 1

        # Propagate reader markers as counted drops.
        if rec.startswith("\x00INVALID_JSON\x00:"):
            stats.records_invalid_json += 1
            continue
        if rec.startswith("\x00MISSING_FIELD\x00:"):
            stats.records_missing_field += 1
            continue

        if not isinstance(rec, str):
            stats.records_empty_dropped += 1
            continue

        text = rec.strip()
        if not text:
            stats.records_empty_dropped += 1
            continue

        if len(text) > max_record_chars:
            stats.records_too_long_dropped += 1
            continue

        if dedup:
            if len(seen) >= DEDUP_MAX_ENTRIES and dedup_count == 0:
                # First time we exceed the cap: degrade gracefully.
                stats.dedup_mode = "sample-capped"
                log.warning(
                    "dedup set exceeded %d entries; switching to sampled dedupe "
                    "(exact dedupe no longer guaranteed for later records)",
                    DEDUP_MAX_ENTRIES,
                )
            if len(seen) < DEDUP_MAX_ENTRIES:
                h = hashlib.sha1(text.encode("utf-8")).digest()
                if h in seen:
                    stats.records_duplicate_dropped += 1
                    continue
                seen.add(h)
            else:
                # Sampled mode: keep a rotating window of recent hashes.
                if len(seen) >= DEDUP_MAX_ENTRIES * 2:
                    seen.clear()
                    dedup_count += 1
                h = hashlib.sha1(text.encode("utf-8")).digest()
                if h in seen:
                    stats.records_duplicate_dropped += 1
                    continue
                seen.add(h)

        stats.records_kept += 1
        stats.chars_total += len(text)
        yield text


# ---------------------------------------------------------------------------
# Tokenization + packing to .bin
# ---------------------------------------------------------------------------

def _token_chunks(tokenizer, texts: Iterable[str]) -> Iterator[List[int]]:
    """Encode texts and yield them separated by <eos>."""
    for text in texts:
        ids = tokenizer.encode(text, add_bos=False, add_eos=True)
        if ids:
            yield ids


def pack_tokens_to_bin(
    tokenizer,
    texts: Iterable[str],
    block_size: int,
    out_path: str | Path,
) -> Tuple[int, int]:
    """Encode texts, concatenate with eos boundaries, carve into fixed blocks.

    Fully streaming: tokens are written block-by-block straight to a uint16
    .bin file (2 bytes/token). Only the current document's ids plus one
    block-sized carry buffer are ever in RAM, so the whole token stream is
    never materialized. A trailing partial block is discarded (documented).

    Returns (total_tokens, num_blocks).
    """
    if block_size <= 0:
        raise InvalidDatasetError(f"block_size must be > 0, got {block_size}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect id overflow for uint16 storage early.
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise InvalidDatasetError(
            f"vocab_size {tokenizer.vocab_size} exceeds uint16 range; "
            "cannot pack to .bin format."
        )

    total = 0
    n_blocks = 0
    carry: List[int] = []
    with open(out_path, "wb") as f:
        for chunk in _token_chunks(tokenizer, texts):
            total += len(chunk)
            carry.extend(chunk)
            usable = (len(carry) // block_size) * block_size
            if usable:
                arr = np.asarray(carry[:usable], dtype=np.uint16)
                arr.tofile(f)
                n_blocks += usable // block_size
                carry = carry[usable:]
        # trailing partial block in `carry` is intentionally discarded

    if total == 0:
        raise InvalidDatasetError(
            "no tokens produced from the dataset. The corpus may be empty "
            "after cleaning, or the tokenizer failed to encode it."
        )
    if n_blocks == 0:
        raise InvalidDatasetError(
            f"dataset produced only {total} tokens, fewer than one "
            f"block_size={block_size} window. Add more data or lower block_size."
        )
    return total, n_blocks


def _split_assign(text: str, val_fraction: float, seed: int) -> str:
    """Deterministic, order-independent document split: 'train' or 'val'.

    Hash of (seed, document text) -> uniform in [0, 1). The same document
    always lands in the same split for a given seed, independent of file
    order or number of runs (validation isolation: a document is either in
    train or in val, never in both, and re-preparation is reproducible).
    """
    h = hashlib.sha1(f"{seed}\x00{text}".encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "big") / float(1 << 64)
    return "val" if v < val_fraction else "train"


def prepare_dataset(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    tokenizer,
    block_size: int,
    val_fraction: float = 0.05,
    seed: int = 1337,
    text_key: str = "text",
    dedup: bool = True,
    max_documents: int = 0,
    max_bytes: int = 0,
    max_seconds: float = 0.0,
    target_documents: int = 0,
    source_weights: Optional[Sequence[float]] = None,
) -> DatasetStats:
    """Full preparation pipeline: read -> clean -> split -> tokenize -> pack.

    Memory: fully streaming. Documents are read one at a time, assigned to a
    split by content hash, and written to two temporary JSONL files, which are
    then token-packed straight from disk. No list of records or tokens is ever
    held in RAM (peak = one document + the bounded dedup hash set).

    Split: document-level via sha1(seed + text) - deterministic,
    order-independent, and identical across runs/machines for the same inputs,
    seed and val_fraction (a document is in exactly one split).

    Limits (0/None = off; approximate: the record that crosses a limit is the
    last one written, so caps may be exceeded by at most that record):
      max_documents    global cap on cleaned records kept
      max_bytes        global cap on kept characters (approx. bytes)
      max_seconds      wall-clock cap for the read phase
      target_documents overall document target
      source_weights   proportional per-source quotas over target_documents
                       (e.g. weights [0.85, 0.15], target 100000 ->
                       source 1 keeps ~85000, source 2 keeps ~15000)
    Every limit that stopped reading is reported in stats.truncated_by.

    Temp files: `_split_train.tmp.jsonl` / `_split_val.tmp.jsonl` in output_dir
    are created by this function and always deleted afterwards (they are
    scratch files, never user data).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_paths:
        raise InvalidDatasetError("no input paths provided")
    if not 0.0 < val_fraction < 1.0:
        raise InvalidDatasetError(f"val_fraction must be in (0, 1), got {val_fraction}")

    stats = DatasetStats(
        block_size=block_size,
        vocab_size=tokenizer.vocab_size,
        seed=seed,
        val_fraction=val_fraction,
    )
    limits: Dict[str, float] = {}
    if max_documents > 0:
        limits["max_documents"] = float(max_documents)
    if max_bytes > 0:
        limits["max_bytes"] = float(max_bytes)
    if max_seconds > 0:
        limits["max_seconds"] = max_seconds
    if target_documents > 0:
        limits["target_documents"] = float(target_documents)
    stats.limits = limits

    quotas: List[int] = []
    if source_weights is not None:
        if len(source_weights) != len(input_paths):
            raise InvalidDatasetError(
                f"source_weights has {len(source_weights)} entries but there are "
                f"{len(input_paths)} input paths"
            )
        if target_documents <= 0:
            raise InvalidDatasetError("source_weights requires target_documents > 0")
        if any(w < 0 for w in source_weights) or sum(source_weights) <= 0:
            raise InvalidDatasetError("source_weights must be non-negative and sum > 0")
        wsum = float(sum(source_weights))
        quotas = [int(round(w / wsum * target_documents)) for w in source_weights]

    train_tmp = output_dir / "_split_train.tmp.jsonl"
    val_tmp = output_dir / "_split_val.tmp.jsonl"
    truncated: List[str] = []
    dedup_seen: set = set()
    t_start = time.perf_counter()

    def _hit(name: str) -> None:
        if name not in truncated:
            truncated.append(name)

    try:
        # ---- phase 1: stream clean records to split files -----------------
        with open(train_tmp, "w", encoding="utf-8") as f_train, \
                open(val_tmp, "w", encoding="utf-8") as f_val:
            for i, p in enumerate(input_paths):
                # Global stop checks before opening a new source.
                if max_documents > 0 and stats.records_kept >= max_documents:
                    _hit("max_documents")
                    break
                if target_documents > 0 and stats.records_kept >= target_documents:
                    _hit("target_documents")
                    break
                if max_seconds > 0 and (time.perf_counter() - t_start) >= max_seconds:
                    _hit("max_seconds")
                    break

                path = Path(p)
                stats.files.append(str(path))
                quota = quotas[i] if quotas else 0
                source_kept = 0
                stop_all = False
                for text in clean_records(
                    iter_records(path, text_key=text_key), stats,
                    dedup=dedup, dedup_seen=dedup_seen,
                ):
                    line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    if _split_assign(text, val_fraction, seed) == "val":
                        f_val.write(line)
                        stats.val_records += 1
                    else:
                        f_train.write(line)
                        stats.train_records += 1
                    source_kept += 1

                    # Limits are checked AFTER writing, so records_kept and the
                    # split file contents always agree (the counting invariant
                    # records_kept == read - dropped holds unconditionally).
                    if max_documents > 0 and stats.records_kept >= max_documents:
                        _hit("max_documents")
                        stop_all = True
                        break
                    if target_documents > 0 and stats.records_kept >= target_documents:
                        _hit("target_documents")
                        stop_all = True
                        break
                    if max_bytes > 0 and stats.chars_total >= max_bytes:
                        _hit("max_bytes")
                        stop_all = True
                        break
                    if max_seconds > 0 and (time.perf_counter() - t_start) >= max_seconds:
                        _hit("max_seconds")
                        stop_all = True
                        break
                    if quota and source_kept >= quota:
                        _hit("source_quota")
                        break  # this source is done; others may still run
                if stop_all:
                    break

        if stats.records_kept == 0:
            raise InvalidDatasetError(
                "dataset is empty after cleaning (0 usable records). "
                "Check that your raw files contain non-empty text."
            )
        stats.truncated_by = ",".join(truncated)

        if train_tmp.stat().st_size == 0:
            raise InvalidDatasetError(
                "0 records landed in the training split (only validation). "
                "Lower val_fraction or add more documents."
            )

        # ---- phase 2: stream tokens -> .bin --------------------------------
        train_tokens, train_blocks = pack_tokens_to_bin(
            tokenizer, iter_records(train_tmp, text_key="text"),
            block_size, output_dir / "train.bin",
        )
        stats.train_tokens = train_tokens
        stats.train_blocks = train_blocks

        def _val_fallback() -> None:
            # Re-read train.bin head (cheap, no re-encode) so a tiny dataset
            # still produces a usable val.bin.
            log.warning(
                "validation split too small for one block; writing val.bin "
                "from a truncated copy of the training tokens."
            )
            with open(output_dir / "train.bin", "rb") as f:
                head = f.read(2 * block_size)
            if len(head) < 2 * block_size:
                raise InvalidDatasetError(
                    "cannot construct validation data: train split also too small."
                )
            (output_dir / "val.bin").write_bytes(head)

        val_path = output_dir / "val.bin"
        if val_tmp.stat().st_size > 0:
            try:
                val_tokens, val_blocks = pack_tokens_to_bin(
                    tokenizer, iter_records(val_tmp, text_key="text"),
                    block_size, val_path,
                )
            except InvalidDatasetError:
                _val_fallback()
                val_tokens, val_blocks = block_size, 1
        else:
            _val_fallback()
            val_tokens, val_blocks = block_size, 1
        stats.val_tokens = val_tokens
        stats.val_blocks = val_blocks

        # ---- phase 3: persist stats ----------------------------------------
        stats_path = output_dir / "dataset_stats.json"
        stats_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
        log.info("wrote %s", stats_path)
    finally:
        # Our scratch files - always removed (success or failure).
        for tmp in (train_tmp, val_tmp):
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    return stats


# ---------------------------------------------------------------------------
# Training-time loading (memmap)
# ---------------------------------------------------------------------------

class PackedDataset:
    """Fixed-length random-crop dataset backed by a numpy memmap.

    Loading the .bin through memmap means training reads only the windows it
    needs - the full token array is never resident in RAM.
    """

    def __init__(self, bin_path: str | Path, block_size: int, seed: int = 1337):
        bin_path = Path(bin_path)
        if not bin_path.exists():
            raise InvalidDatasetError(
                f"tokenized data not found: {bin_path}. "
                "Run scripts/prepare_data.py first."
            )
        file_bytes = bin_path.stat().st_size
        if file_bytes < 2 * block_size:
            raise InvalidDatasetError(
                f"{bin_path} holds {file_bytes} bytes, too small for "
                f"block_size={block_size} (needs >= {2 * block_size}). "
                "Add more data or lower block_size."
            )
        self.path = bin_path
        self.block_size = block_size
        # memmap stays open for the process lifetime; numpy handles it.
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self.n_tokens = int(self.data.shape[0])
        self.n_blocks = self.n_tokens // block_size
        if self.n_blocks == 0:
            raise InvalidDatasetError(
                f"{bin_path} contains {self.n_tokens} tokens < block_size "
                f"{block_size}; not enough data to train."
            )
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        # Effectively infinite crops; the trainer drives iteration count.
        return self.n_blocks

    def token_count(self) -> int:
        return self.n_tokens

    def get_block(self, index: int) -> np.ndarray:
        """Return one packed window (deterministic by index)."""
        start = (index % self.n_blocks) * self.block_size
        return np.asarray(self.data[start:start + self.block_size], dtype=np.int64)

    def random_block(self) -> np.ndarray:
        i = self._rng.randrange(self.n_blocks)
        return self.get_block(i)

    def sample_batch(self, batch_size: int, rng: Optional[random.Random] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x, y) int64 arrays: inputs and next-token targets.

        `rng` may be a stdlib random.Random or a numpy RandomState - only
        `randrange`-equivalent behavior is required.
        """
        rng = rng or self._rng

        def _randrange(n: int) -> int:
            if hasattr(rng, "randrange"):
                return int(rng.randrange(n))          # stdlib random
            return int(rng.randint(0, n - 1))         # numpy RandomState

        xs = np.empty((batch_size, self.block_size), dtype=np.int64)
        ys = np.empty((batch_size, self.block_size), dtype=np.int64)
        for i in range(batch_size):
            if self.n_tokens > self.block_size:
                start = _randrange(self.n_tokens - self.block_size)
                window = np.asarray(
                    self.data[start:start + self.block_size + 1], dtype=np.int64
                )
            else:
                # Degenerate tiny split (exactly block_size tokens, e.g. a
                # one-block validation set): wrap around so valid x/y pairs
                # still exist. Acceptable for a loss estimate.
                start = _randrange(self.n_tokens)
                idx = (start + np.arange(self.block_size + 1)) % self.n_tokens
                window = np.asarray(self.data[idx], dtype=np.int64)
            xs[i] = window[:-1]
            ys[i] = window[1:]
        return xs, ys

    def close(self) -> None:
        try:
            self.data._mmap.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    def __repr__(self) -> str:  # pragma: no cover
        return (f"PackedDataset(path={self.path.name!r}, tokens={self.n_tokens}, "
                f"blocks={self.n_blocks}, block_size={self.block_size})")


def load_packed_dataset(
    data_dir: str | Path,
    split: str,
    block_size: int,
    seed: int = 1337,
) -> PackedDataset:
    """Load train.bin or val.bin from a processed-data directory."""
    if split not in ("train", "val"):
        raise InvalidDatasetError(f"split must be 'train' or 'val', got {split!r}")
    return PackedDataset(Path(data_dir) / f"{split}.bin", block_size, seed=seed)
