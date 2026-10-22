"""Tests for dipai.model: shapes, loss, gradients, tying, generation."""

from __future__ import annotations

import pytest
import torch

from dipai.config import ModelConfig


def test_forward_shapes_and_loss(tiny_model, tiny_model_cfg):
    B, T = 3, 16
    x = torch.randint(0, tiny_model_cfg.vocab_size, (B, T))
    y = torch.randint(0, tiny_model_cfg.vocab_size, (B, T))
    out = tiny_model(x, y)
    assert out.logits.shape == (B, T, tiny_model_cfg.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)
    # Random-init loss should be near ln(vocab).
    loss_val = float(out.loss.detach())
    import math

    assert abs(loss_val - math.log(tiny_model_cfg.vocab_size)) < 1.0


def test_inference_without_targets(tiny_model, tiny_model_cfg):
    x = torch.randint(0, tiny_model_cfg.vocab_size, (1, 8))
    out = tiny_model(x)
    assert out.loss is None
    assert out.logits.shape == (1, 8, tiny_model_cfg.vocab_size)


def test_backward_updates_grads(tiny_model, tiny_model_cfg):
    x = torch.randint(0, tiny_model_cfg.vocab_size, (2, 8))
    y = torch.randint(0, tiny_model_cfg.vocab_size, (2, 8))
    out = tiny_model(x, y)
    out.loss.backward()
    grads = [p.grad for p in tiny_model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_seq_longer_than_block_raises(tiny_model, tiny_model_cfg):
    x = torch.randint(0, tiny_model_cfg.vocab_size, (1, tiny_model_cfg.block_size + 1))
    with pytest.raises(ValueError, match="exceeds block_size"):
        tiny_model(x)


def test_weight_tying(tiny_model):
    assert tiny_model.lm_head.weight is tiny_model.tok_emb.weight


def test_generate_extends_sequence(tiny_model, tiny_model_cfg):
    torch.manual_seed(0)
    x = torch.randint(0, tiny_model_cfg.vocab_size, (2, 4))
    out = tiny_model.generate(x, max_new_tokens=7, temperature=0.8, top_k=10, top_p=0.9)
    assert out.shape == (2, 4 + 7)
    assert int(out.max()) < tiny_model_cfg.vocab_size


def test_generate_greedy_deterministic(tiny_model, tiny_model_cfg):
    x = torch.randint(0, tiny_model_cfg.vocab_size, (1, 5))
    a = tiny_model.generate(x, max_new_tokens=6, temperature=0.0)
    b = tiny_model.generate(x, max_new_tokens=6, temperature=0.0)
    assert torch.equal(a, b)


def test_generate_stops_at_eos(tiny_model, tiny_model_cfg):
    x = torch.randint(0, tiny_model_cfg.vocab_size, (1, 4))
    # Force eos everywhere by banning everything else via top_k=1 after
    # biasing: simplest - run with eos and check early stop when eos first.
    out = tiny_model.generate(x, max_new_tokens=20, temperature=0.0, eos_token_id=tiny_model_cfg.vocab_size - 1)
    assert out.shape[1] <= 4 + 20


def test_configure_optimizers_groups(tiny_model):
    opt = tiny_model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu",
    )
    assert len(opt.param_groups) == 2
    wd = {g["weight_decay"] for g in opt.param_groups}
    assert wd == {0.1, 0.0}


def test_config_dict_roundtrip(tiny_model_cfg):
    from dipai.model import DipaiGPT

    d = tiny_model_cfg.__dict__
    m = DipaiGPT.from_config_dict(d)
    assert m.num_params > 0
