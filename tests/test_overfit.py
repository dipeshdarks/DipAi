"""Controlled small-data overfit test.

Requirement: "Loss decreases on a controlled small-data overfit test."

A tiny model is trained on ONE repeated token block from a real (tiny)
corpus. The only way to drive the loss down is to memorize that data - so a
large, sustained drop proves the full learn loop works: forward loss,
backward, optimizer step, and data are genuinely connected (nothing is
silently detached or zeroed).
"""

from __future__ import annotations

import torch

from dipai.model import DipaiGPT


def _one_block(tokenizer, block_size: int = 32):
    """A single supervised (x, y) block built from real text."""
    text = (
        "the quick brown fox jumps over the lazy dog. "
        "once upon a time there was a little cat named spot. "
        "machine learning models learn patterns from text data."
    )
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    while len(ids) < block_size:
        ids = ids + ids
    ids = ids[:block_size]
    x = torch.tensor([ids], dtype=torch.long)
    y = x.roll(-1, dims=1)  # every position supervised: next-token shift
    return x, y


def test_overfit_single_block_loss_collapses(tiny_model_cfg, tiny_tokenizer):
    assert tiny_model_cfg.vocab_size == tiny_tokenizer.vocab_size
    torch.manual_seed(0)
    model = DipaiGPT(tiny_model_cfg)
    model.train()
    optimizer = model.configure_optimizers(
        weight_decay=0.0, learning_rate=3e-3, betas=(0.9, 0.95), device_type="cpu",
    )
    x, y = _one_block(tiny_tokenizer, block_size=tiny_model_cfg.block_size)

    losses = []
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        out = model(x, y)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(out.loss.detach()))

    initial = sum(losses[:10]) / 10
    final = sum(losses[-10:]) / 10
    # Cross-entropy for a ~400-piece vocab starts near ln(400) ~ 6; memorizing
    # one block must push it far below half of the initial value.
    assert final < initial * 0.5, (
        f"overfit failed: initial={initial:.3f} final={final:.3f}"
    )
