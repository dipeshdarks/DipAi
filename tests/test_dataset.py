"""Tests for dipai.dataset: preparation, stats, packing, memmap sampling."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dipai.dataset import (
    InvalidDatasetError,
    PackedDataset,
    iter_records,
    load_packed_dataset,
    prepare_dataset,
)


def _write_sources(tmp_path):
    txt = tmp_path / "corpus.txt"
    txt.write_text(
        "the quick brown fox jumps over the lazy dog. " * 40
        + "\n\n"
        + "once upon a time there was a little cat named spot. " * 40
        + "\n\n"
        + "machine learning models learn patterns from text data. " * 40,
        encoding="utf-8",
    )
    jsonl = tmp_path / "extra.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for i in range(50):
            f.write(json.dumps({"text": f"user message number {i} about topic {i % 7}"}) + "\n")
        f.write(json.dumps({"text": "duplicate of a line that appears twice here"}) + "\n")
        f.write(json.dumps({"text": "duplicate of a line that appears twice here"}) + "\n")
        f.write("{this is not valid json\n")          # invalid json -> counted
        f.write(json.dumps({"wrong": "field"}) + "\n")  # missing field -> counted
    return [txt, jsonl]


def test_iter_records_txt_and_jsonl(tmp_path):
    files = _write_sources(tmp_path)
    txt_records = list(iter_records(files[0]))
    assert len(txt_records) == 3  # blank-line paragraphs
    json_records = list(iter_records(files[1]))
    assert len(json_records) == 54  # 50 valid + 2 dupes + 1 invalid marker + 1 missing-field marker


def test_prepare_dataset_stats_and_bins(tmp_path, tiny_tokenizer):
    files = _write_sources(tmp_path)
    out = tmp_path / "processed"
    stats = prepare_dataset(
        input_paths=files, output_dir=out, tokenizer=tiny_tokenizer,
        block_size=32, val_fraction=0.2, seed=7,
    )
    assert stats.records_duplicate_dropped >= 1
    assert stats.records_invalid_json >= 1
    assert stats.records_missing_field >= 1
    assert stats.records_kept == (
        stats.records_read
        - stats.records_duplicate_dropped
        - stats.records_empty_dropped
        - stats.records_too_long_dropped
        - stats.records_invalid_json
        - stats.records_missing_field
    )
    assert (out / "train.bin").exists()
    assert (out / "val.bin").exists()
    assert (out / "dataset_stats.json").exists()

    train = np.memmap(out / "train.bin", dtype=np.uint16, mode="r")
    assert len(train) % 32 == 0
    assert train.max(initial=0) < tiny_tokenizer.vocab_size


def test_prepare_rejects_bad_input(tmp_path, tiny_tokenizer):
    with pytest.raises(InvalidDatasetError):
        prepare_dataset(
            input_paths=[tmp_path / "missing.txt"], output_dir=tmp_path / "o",
            tokenizer=tiny_tokenizer, block_size=16, val_fraction=0.1,
        )


def test_packed_dataset_sample_batch(tmp_path, tiny_tokenizer):
    files = _write_sources(tmp_path)
    out = tmp_path / "processed"
    prepare_dataset(
        input_paths=files, output_dir=out, tokenizer=tiny_tokenizer,
        block_size=32, val_fraction=0.2, seed=7,
    )
    ds = load_packed_dataset(out, "train", block_size=32, seed=7)
    try:
        rng = np.random.RandomState(0)
        xs, ys = ds.sample_batch(4, rng=rng)
        assert xs.shape == (4, 32)
        assert ys.shape == (4, 32)
        assert xs.dtype == np.int64
        # Next-token targets are inputs shifted by one within each block.
        assert np.array_equal(xs[:, 1:], ys[:, :-1])
    finally:
        ds.close()


def test_packed_dataset_tiny_split_wraps(tmp_path):
    """A split of exactly one block must still yield batches (wrap-around).

    PackedDataset requires >= 2*block_size BYTES (= block_size uint16 tokens),
    so 32 tokens @ block 32 is the smallest valid file; sample_batch then
    wraps instead of slicing past the end.
    """
    tiny = np.full(32, 5, dtype=np.uint16)
    out = tmp_path / "processed"
    out.mkdir()
    (out / "train.bin").write_bytes(tiny.tobytes())
    ds = load_packed_dataset(out, "train", block_size=32, seed=1)
    try:
        xs, ys = ds.sample_batch(2, rng=np.random.RandomState(0))
        assert xs.shape == (2, 32)
        assert ys.shape == (2, 32)
        assert np.array_equal(xs[:, 1:], ys[:, :-1])
        assert len(ds) == 1
    finally:
        ds.close()
