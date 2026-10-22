"""Tests for dipai.config: loading, validation, budgets, seeds."""

from __future__ import annotations

import random

import pytest

from dipai.config import (
    ConfigError,
    FinetuneConfig,
    ModelConfig,
    ScratchConfig,
    load_finetune_config,
    load_scratch_config,
    set_seed,
)

ROOT = "configs"


def test_load_scratch_yaml():
    cfg = load_scratch_config(f"{ROOT}/scratch.yaml")
    assert cfg.model.vocab_size == 4104
    assert cfg.model.n_embd == 512
    assert cfg.model.n_layer == 6
    assert cfg.model.n_head == 8
    assert cfg.train.effective_batch_size == 32


def test_load_finetune_yaml():
    cfg = load_finetune_config(f"{ROOT}/finetune.yaml")
    assert cfg.load_in_4bit is True
    assert cfg.lora.r == 16
    assert cfg.model_name_or_path == "Qwen/Qwen2.5-0.5B-Instruct"


def test_missing_config_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_scratch_config(tmp_path / "nope.yaml")


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  n_embd: 32\n  bogus_key: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_scratch_config(p)


def test_invalid_model_configs():
    with pytest.raises(ConfigError, match="divisible"):
        ModelConfig(n_embd=30, n_head=4).validate()
    with pytest.raises(ConfigError, match="dropout"):
        ModelConfig(dropout=1.5).validate()
    with pytest.raises(ConfigError, match="vocab_size"):
        ModelConfig(vocab_size=0).validate()
    with pytest.raises(ConfigError, match="block_size"):
        ModelConfig(block_size=-1).validate()


def test_finetune_config_validation():
    cfg = FinetuneConfig()
    cfg.validate()
    with pytest.raises(ConfigError, match="model_name_or_path"):
        FinetuneConfig(model_name_or_path="").validate()
    with pytest.raises(ConfigError, match="batch_size"):
        FinetuneConfig(per_device_train_batch_size=0).validate()


def test_param_budget_matches_measured():
    """The 20M-class config must build to its estimated parameter count."""
    from dipai.model import DipaiGPT

    cfg = load_scratch_config(f"{ROOT}/scratch.yaml")
    est = cfg.model.param_budget_estimate()
    model = DipaiGPT(cfg.model)
    assert model.num_params == est
    assert 20_000_000 < model.num_params < 23_000_000


def test_set_seed_reproducible():
    set_seed(123)
    a = [random.random() for _ in range(3)]
    set_seed(123)
    b = [random.random() for _ in range(3)]
    assert a == b


def test_scratch_config_roundtrip(tmp_path):
    cfg = load_scratch_config(f"{ROOT}/scratch.yaml")
    out = tmp_path / "saved.yaml"
    cfg.save(out)
    cfg2 = load_scratch_config(out)
    assert cfg2.to_dict() == cfg.to_dict()
