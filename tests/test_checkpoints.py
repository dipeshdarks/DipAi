"""Tests for dipai.checkpoints: atomic save/load, pointer, retention."""

from __future__ import annotations

import json

import pytest
import torch

from dipai.checkpoints import (
    CheckpointError,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    prune,
    save_checkpoint,
)


def _tiny_setup(tiny_model_cfg):
    from dipai.model import DipaiGPT

    model = DipaiGPT(tiny_model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt


def test_save_and_load_roundtrip(tmp_path, tiny_model_cfg):
    model, opt = _tiny_setup(tiny_model_cfg)
    path = save_checkpoint(
        tmp_path, 7, model, opt,
        config={"model": tiny_model_cfg.__dict__},
        extra={"note": "hello"}, keep_last_n=3,
    )
    assert path.name == "ckpt_00000007.pt"
    payload = load_checkpoint(path)
    assert payload["step"] == 7
    assert payload["extra"]["note"] == "hello"
    assert payload["model_config"]["n_embd"] == tiny_model_cfg.n_embd
    # Weights actually round-trip.
    model2 = type(model)(type(tiny_model_cfg)(**payload["model_config"]))
    model2.load_state_dict(payload["model"])
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b)


def test_latest_pointer(tmp_path, tiny_model_cfg):
    model, opt = _tiny_setup(tiny_model_cfg)
    save_checkpoint(tmp_path, 5, model, opt, keep_last_n=3)
    save_checkpoint(tmp_path, 10, model, opt, keep_last_n=3)
    latest = latest_checkpoint(tmp_path)
    assert latest is not None and "00000010" in latest.name
    info = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert info["step"] == 10


def test_retention_keeps_last_n(tmp_path, tiny_model_cfg):
    model, opt = _tiny_setup(tiny_model_cfg)
    for step in (1, 2, 3, 4, 5):
        save_checkpoint(tmp_path, step, model, opt, keep_last_n=3)
    files = list_checkpoints(tmp_path)
    assert len(files) == 3
    assert [f.name for f in files] == [
        "ckpt_00000003.pt", "ckpt_00000004.pt", "ckpt_00000005.pt"
    ]
    # Pointer must remain valid after pruning.
    assert latest_checkpoint(tmp_path) is not None


def test_no_tmp_leftovers(tmp_path, tiny_model_cfg):
    model, opt = _tiny_setup(tiny_model_cfg)
    save_checkpoint(tmp_path, 1, model, opt)
    assert not list(tmp_path.glob("*.tmp"))


def test_load_from_directory_uses_pointer(tmp_path, tiny_model_cfg):
    model, opt = _tiny_setup(tiny_model_cfg)
    save_checkpoint(tmp_path, 3, model, opt, keep_last_n=3)
    save_checkpoint(tmp_path, 6, model, opt, keep_last_n=3)
    payload = load_checkpoint(tmp_path)  # directory, not file
    assert payload["step"] == 6


def test_load_missing_raises(tmp_path):
    with pytest.raises(CheckpointError, match="no checkpoint"):
        load_checkpoint(tmp_path)


def test_load_garbage_raises(tmp_path):
    bad = tmp_path / "ckpt_00000001.pt"
    bad.write_bytes(b"not a torch file")
    with pytest.raises(CheckpointError, match="failed to read"):
        load_checkpoint(bad)


def test_load_wrong_shape_raises(tmp_path):
    bad = tmp_path / "ckpt_00000001.pt"
    torch.save({"something": 1}, bad)
    with pytest.raises(CheckpointError, match="not a DIPAI checkpoint"):
        load_checkpoint(bad)


def test_prune_public(tmp_path, tiny_model_cfg):
    model, opt = _tiny_setup(tiny_model_cfg)
    for step in (1, 2, 3, 4):
        save_checkpoint(tmp_path, step, model, opt, keep_last_n=10)
    removed = prune(tmp_path, keep_last_n=2)
    assert removed == 2
    assert len(list_checkpoints(tmp_path)) == 2
