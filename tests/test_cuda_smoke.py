"""CUDA-path smoke test: autocast + gradient checkpointing + optimizer step.

Skipped automatically when no CUDA device is available. This exercises the
exact training configuration used by dipai_50m.yaml (fp16 autocast, fp32
master weights, GradScaler, checkpointed blocks) on a tiny model.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def test_cuda_autocast_ckpt_optimizer_step(tiny_model_cfg):
    from dipai.model import DipaiGPT

    torch.manual_seed(0)
    model = DipaiGPT(tiny_model_cfg).cuda()
    model.train()
    model.set_gradient_checkpointing(True)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cuda",
    )
    scaler = torch.amp.GradScaler(enabled=True)

    x = torch.randint(4, tiny_model_cfg.vocab_size, (2, 16), device="cuda")
    y = x.roll(-1, dims=1)

    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x, y)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))

    assert all(map(torch.isfinite, torch.tensor(losses)))
    peak = torch.cuda.max_memory_allocated()
    assert peak > 0
    torch.cuda.reset_peak_memory_stats()
