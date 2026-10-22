"""Focused tests for the prepare orchestration layer (clean -> bins + manifest)."""
import json
from pathlib import Path

import numpy as np
import pytest

from dipai.data.prepare import PrepareConfig, load_prepare_config, prepare
from dipai.dataset import PackedDataset, InvalidDatasetError
from dipai.tokenizer import DipaiTokenizer


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    """Train a tiny real SentencePiece tokenizer once for the module."""
    tmp = tmp_path_factory.mktemp("tok")
    corpus = tmp / "corpus.txt"
    lines = []
    for i in range(200):
        lines.append(
            f"The quick brown fox number {i} jumps over the lazy dog near "
            f"the river bank while the sun sets over the quiet village fields "
            f"and the birds fly home to their nests in the tall green trees."
        )
    corpus.write_text("\n".join(lines), encoding="utf-8")
    prefix = tmp / "tokenizer"
    DipaiTokenizer.train(
        texts=[l for l in lines],
        model_prefix=str(prefix),
        vocab_size=512,
        max_lines=200,
    )
    return prefix


@pytest.fixture()
def raw_jsonl(tmp_path):
    """A small cleaned-style JSONL with a text field."""
    p = tmp_path / "cleaned.jsonl"
    records = []
    for i in range(60):
        records.append(json.dumps({
            "text": (
                f"Document {i} about science and history. The ancient "
                f"civilizations built great cities along the rivers and "
                f"trade routes of the known world many centuries ago."
            ),
            "source": "unit-test",
        }))
    p.write_text("\n".join(records), encoding="utf-8")
    return p


def test_prepare_config_validation():
    with pytest.raises(ValueError):
        PrepareConfig(inputs=[], block_size=512, val_fraction=0.1).validate()
    with pytest.raises(ValueError):
        PrepareConfig(inputs=["x"], block_size=0, val_fraction=0.1).validate()
    with pytest.raises(ValueError):
        PrepareConfig(inputs=["x"], block_size=512, val_fraction=0.0).validate()
    PrepareConfig(inputs=["x"], block_size=512, val_fraction=0.1).validate()


def test_load_prepare_config_rejects_unknown_key(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("inputs: [a]\nnot_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prepare_config(str(p))


def test_prepare_end_to_end(tokenizer, raw_jsonl, tmp_path):
    out = tmp_path / "processed"
    cfg = PrepareConfig(
        inputs=[str(raw_jsonl)],
        output_dir=str(out),
        tokenizer_path=str(tokenizer),
        block_size=64,
        val_fraction=0.1,
        seed=1337,
        max_documents=60,
    )
    manifest = prepare(cfg)

    # artifacts exist
    assert (out / "train.bin").exists()
    assert (out / "val.bin").exists()
    assert (out / "dataset_stats.json").exists()
    assert (out / "manifest.json").exists()

    # stats agree with manifest
    stats = json.loads((out / "dataset_stats.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["train_blocks"] == stats["train_blocks"]
    assert stats["train_blocks"] >= 1
    assert stats["val_blocks"] >= 1

    # train.bin is loadable as a PackedDataset with the same block size
    ds = PackedDataset(out / "train.bin", block_size=64, seed=1)
    assert ds.n_blocks == stats["train_blocks"]
    xs, ys = ds.sample_batch(2)
    assert xs.shape == (2, 64) and ys.shape == (2, 64)


def test_prepare_empty_input_raises(tokenizer, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    cfg = PrepareConfig(
        inputs=[str(empty)],
        output_dir=str(tmp_path / "out"),
        tokenizer_path=str(tokenizer),
        block_size=32,
        val_fraction=0.1,
    )
    with pytest.raises(InvalidDatasetError):
        prepare(cfg)
