"""Tests for dipai.hardware and dipai.utils (no GPU required)."""

from __future__ import annotations

from dipai.hardware import (
    estimate_scratch_vram_bytes,
    fits_in_vram,
    format_report,
    probe_hardware,
    resolve_dtype,
)
from dipai.utils import format_bytes, format_duration


def test_probe_hardware_fields():
    hw = probe_hardware(check_bnb=False)
    assert hw.platform_str
    assert hw.python_version
    assert hw.torch_version
    assert hw.ram_total_gb > 0
    assert hw.device_name is not None
    assert isinstance(hw.cuda_available, bool)


def test_format_report_contains_sections():
    hw = probe_hardware(check_bnb=False)
    text = format_report(hw)
    assert "VRAM" in text or "RAM" in text


def test_estimate_scratch_vram_positive():
    est = estimate_scratch_vram_bytes(
        vocab_size=4104, n_embd=512, n_layer=6, block_size=256,
        batch_size=8, dtype_bytes=2,
    )
    for key in ("param_bytes", "grad_bytes", "optimizer_bytes",
                "activation_bytes", "overhead_bytes", "total_bytes"):
        assert est[key] > 0
    # 21M params fp32-ish + optimizer + activations must exceed raw weights.
    assert est["total_bytes"] > est["param_bytes"]


def test_fits_in_vram_logic():
    hw = probe_hardware(check_bnb=False)
    assert fits_in_vram(1, hw) is True
    assert fits_in_vram(10**15, hw) is False


def test_resolve_dtype_mapping():
    import torch

    assert resolve_dtype("cpu", "float32") == torch.float32
    assert resolve_dtype("cpu", "bfloat16") == torch.bfloat16
    assert resolve_dtype("cpu", "auto") in (torch.float32, torch.float16, torch.bfloat16)


def test_format_helpers():
    assert format_bytes(1024) == "1.00 KiB"
    assert format_bytes(0) == "0 B"
    assert "s" in format_duration(1.5) or "m" in format_duration(1.5)
    assert format_duration(0) != ""
