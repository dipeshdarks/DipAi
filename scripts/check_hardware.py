#!/usr/bin/env python
"""DIPAI hardware compatibility check.

Run:
    python scripts/check_hardware.py

Exits 0 when the machine can run at least CPU-mode training, exits 1 when
PyTorch is missing or fatally broken. Prints a full report either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipai.hardware import probe_hardware, format_report, estimate_scratch_vram_bytes, fits_in_vram


def main() -> int:
    hw = probe_hardware()
    print(format_report(hw))

    # Scratch-mode feasibility probe with the default model config.
    est = estimate_scratch_vram_bytes(
        vocab_size=8192, n_embd=256, n_layer=4, block_size=256, batch_size=8,
        dtype_bytes=2 if hw.cuda_available else 4,
    )
    from dipai.utils import format_bytes

    print("\nScratch-mode estimate (defaults: 256-dim / 4-layer / ctx 256 / bs 8):")
    print(f"  approximate peak VRAM : {format_bytes(est['total_bytes'])}")
    print(f"  fits in VRAM          : {fits_in_vram(est['total_bytes'], hw)}")

    if not hw.cuda_available and hw.ram_available_bytes < 2 * 1024**3:
        print("\nFATAL: no CUDA and less than 2 GB free RAM. Cannot train.")
        return 1

    if hw.cuda_available:
        print("\nOK: CUDA GPU detected. Scratch training is supported.")
        if hw.bitsandbytes_available:
            print("OK: bitsandbytes available. 4-bit QLoRA finetuning is supported.")
        else:
            print(
                "NOTE: bitsandbytes not importable. Finetune mode will fall back "
                "to fp16 (needs more VRAM) or refuse 4-bit."
            )
    else:
        print("\nWARNING: running without CUDA. Training will be very slow on CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
