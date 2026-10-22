"""Interrupted-training recovery + deterministic validation isolation.

Contract under test (train.py):
  * KeyboardInterrupt mid-run -> exit 130, emergency checkpoint written
  * resuming with the same config completes the remaining steps
  * same seed => identical packed bins (bit-for-bit) across prepare runs
"""

from __future__ import annotations

import numpy as np
import pytest

from dipai.checkpoints import latest_checkpoint, load_checkpoint
from dipai.config import ModelConfig, ScratchConfig, TrainConfig
from dipai.dataset import PackedDataset, prepare_dataset
from dipai.train import train


def _write_corpus(tmp_path):
    txt = tmp_path / "corpus.txt"
    parts = []
    for i in range(40):
        parts.append(f"document number {i} talks about topic {i % 5} with words. " * 8)
    txt.write_text("\n\n".join(parts), encoding="utf-8")
    return txt


def _make_config(tmp_path, tiny_tokenizer, data_dir, ckpt_dir) -> ScratchConfig:
    cfg = ScratchConfig(
        model=ModelConfig(
            vocab_size=tiny_tokenizer.vocab_size, block_size=16, n_layer=2,
            n_head=2, n_embd=32, dropout=0.0, bias=False, tie_weights=True,
        ),
        train=TrainConfig(
            data_dir=str(data_dir),
            tokenizer_path=str(tiny_tokenizer.save(tmp_path / "tok" / "tokenizer")),
            ckpt_dir=str(ckpt_dir),
            batch_size=2,
            gradient_accumulation_steps=1,
            max_iters=4,
            eval_interval=1000,       # eval only fires on the final step
            eval_iters=2,
            save_interval=1000,       # final-step save + emergency save only
            learning_rate=1e-3,
            min_lr=1e-4,
            warmup_iters=0,
            lr_decay_iters=1000,
            seed=7,
            device="cpu",
            dtype="float32",
        ),
    )
    cfg.validate()
    return cfg


def test_interrupt_then_resume_completes(tmp_path, tiny_tokenizer, monkeypatch):
    corpus = _write_corpus(tmp_path)
    data_dir = tmp_path / "processed"
    prepare_dataset(
        input_paths=[corpus], output_dir=data_dir, tokenizer=tiny_tokenizer,
        block_size=16, val_fraction=0.2, seed=7,
    )
    ckpt_dir = tmp_path / "ckpts"
    cfg = _make_config(tmp_path, tiny_tokenizer, data_dir, ckpt_dir)

    # Raise KeyboardInterrupt on the 3rd TRAIN split batch (a val batch must
    # never trigger it - the file name tells the splits apart).
    real_sample = PackedDataset.sample_batch
    calls = {"n": 0}

    def flaky(self, batch_size, rng=None):
        if self.path.name == "train.bin":
            calls["n"] += 1
            if calls["n"] == 3:
                raise KeyboardInterrupt
        return real_sample(self, batch_size, rng=rng)

    monkeypatch.setattr(PackedDataset, "sample_batch", flaky)
    rc = train(cfg)
    assert rc == 130
    assert calls["n"] == 3

    emergency = latest_checkpoint(ckpt_dir)
    assert emergency is not None
    payload = load_checkpoint(emergency)
    assert payload["extra"]["interrupted"] is True
    interrupted_step = int(payload["step"])
    assert 0 < interrupted_step <= cfg.train.max_iters

    # Resume: the patch is restored, the run must finish every step.
    monkeypatch.setattr(PackedDataset, "sample_batch", real_sample)
    cfg.train.resume = True
    rc2 = train(cfg)
    assert rc2 == 0
    final = load_checkpoint(latest_checkpoint(ckpt_dir))
    assert int(final["step"]) == cfg.train.max_iters
    assert not final.get("extra", {}).get("interrupted", False)


def test_prepare_same_seed_identical_bins(tmp_path, tiny_tokenizer):
    corpus = _write_corpus(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    stats_a = prepare_dataset(
        input_paths=[corpus], output_dir=out_a, tokenizer=tiny_tokenizer,
        block_size=16, val_fraction=0.25, seed=7,
    )
    stats_b = prepare_dataset(
        input_paths=[corpus], output_dir=out_b, tokenizer=tiny_tokenizer,
        block_size=16, val_fraction=0.25, seed=7,
    )
    assert (out_a / "train.bin").read_bytes() == (out_b / "train.bin").read_bytes()
    assert (out_a / "val.bin").read_bytes() == (out_b / "val.bin").read_bytes()
    assert stats_a.records_kept == stats_b.records_kept
    assert stats_a.train_tokens == stats_b.train_tokens
    assert stats_a.val_tokens == stats_b.val_tokens

    # Sanity: the bins are real token data shaped in whole blocks.
    train = np.memmap(out_a / "train.bin", dtype=np.uint16, mode="r")
    assert len(train) % 16 == 0
    assert len(train) > 0
