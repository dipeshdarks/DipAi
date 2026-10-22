"""Causal attention, loss masking, and gradient-checkpointing correctness.

These are the model-level invariants the SFT pipeline depends on:
  * logits at position t must not depend on tokens after t
  * F.cross_entropy(ignore_index=-100) must drop masked targets exactly
  * gradient checkpointing must produce the same loss and real gradients
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def test_future_tokens_do_not_affect_past_logits(tiny_model):
    model = tiny_model  # eval mode, dropout=0 (fixture)
    torch.manual_seed(0)
    x1 = torch.randint(4, model.cfg.vocab_size, (1, 16))
    x2 = x1.clone()
    x2[0, -1] = (x2[0, -1] + 1) % model.cfg.vocab_size  # change only the LAST token

    with torch.no_grad():
        l1 = model(x1).logits
        l2 = model(x2).logits

    # Positions before the changed token must be identical ...
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-6)
    # ... and the changed position itself must (virtually always) differ.
    assert not torch.allclose(l1[:, -1], l2[:, -1], atol=1e-6)


def test_earlier_position_unaffected_by_entire_future(tiny_model):
    """Position 0 output must be identical regardless of all later tokens."""
    model = tiny_model
    torch.manual_seed(1)
    x1 = torch.randint(4, model.cfg.vocab_size, (2, 8))
    x2 = x1.clone()
    x2[:, 1:] = torch.randint(4, model.cfg.vocab_size, (2, 7))
    with torch.no_grad():
        l1 = model(x1).logits
        l2 = model(x2).logits
    assert torch.allclose(l1[:, 0], l2[:, 0], atol=1e-6)


def test_masked_targets_drop_out_of_loss_exactly(tiny_model):
    model = tiny_model
    torch.manual_seed(2)
    x = torch.randint(4, model.cfg.vocab_size, (2, 12))
    y = x.roll(-1, dims=1)
    y[0, :5] = -100
    y[1, 3:9] = -100

    with torch.no_grad():
        out = model(x, y)
        mask = y != -100
        manual = F.cross_entropy(
            out.logits[mask], y[mask]
        )
    assert mask.any() and (~mask).any()
    assert torch.allclose(out.loss, manual, atol=1e-6)


def test_all_masked_targets_is_nan(tiny_model):
    """Documents WHY ConversationDataset guarantees >=1 supervised target."""
    model = tiny_model
    x = torch.randint(4, model.cfg.vocab_size, (1, 8))
    y = torch.full((1, 8), -100, dtype=torch.long)
    with torch.no_grad():
        out = model(x, y)
    assert torch.isnan(out.loss)


def test_gradient_checkpointing_backward_matches_plain(tiny_model):
    """Checkpointed forward must yield the same loss and finite gradients."""
    model = tiny_model
    model.train()  # checkpointing only engages in train mode with grad enabled
    torch.manual_seed(3)
    x = torch.randint(4, model.cfg.vocab_size, (2, 10))
    y = x.roll(-1, dims=1)

    out_plain = model(x, y)
    loss_plain = out_plain.loss
    loss_plain.backward()
    grads_plain = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none=True)

    model.set_gradient_checkpointing(True)
    out_ckpt = model(x, y)
    assert torch.allclose(out_plain.loss.detach(), out_ckpt.loss, atol=1e-6)
    out_ckpt.loss.backward()
    grads_ckpt = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.set_gradient_checkpointing(False)

    assert len(grads_plain) == len(grads_ckpt) > 0
    for a, b in zip(grads_plain, grads_ckpt):
        assert torch.isfinite(b).all()
        assert torch.allclose(a, b, atol=1e-5)
