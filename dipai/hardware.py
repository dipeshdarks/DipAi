"""Hardware probing, compatibility checking, and memory estimation for DIPAI.

Answers three questions the rest of the pipeline needs:
  1. Is CUDA available and what GPU / VRAM do we have?
  2. How much system RAM is free right now?
  3. Will a given scratch model or finetune job plausibly fit?
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil

# Minimum bar for "comfortable" scratch training on this project's target.
MIN_VRAM_BYTES = 2 * 1024**3      # 2 GB
COMFORT_VRAM_BYTES = 4 * 1024**3  # 4 GB
MIN_RAM_BYTES = 4 * 1024**3       # 4 GB system RAM


@dataclass
class HardwareInfo:
    """Snapshot of the machine's relevant capabilities."""

    cuda_available: bool = False
    cuda_version: Optional[str] = None
    device_name: Optional[str] = None
    vram_total_bytes: int = 0
    vram_free_bytes: int = 0
    capability_major: Optional[int] = None
    capability_minor: Optional[int] = None

    ram_total_bytes: int = 0
    ram_available_bytes: int = 0

    cpu_count: int = 0
    platform_str: str = ""
    python_version: str = ""

    torch_version: Optional[str] = None
    bitsandbytes_available: Optional[bool] = None
    bitsandbytes_version: Optional[str] = None

    warnings: List[str] = field(default_factory=list)

    # ---- derived helpers -------------------------------------------------

    @property
    def device(self) -> str:
        return "cuda" if self.cuda_available else "cpu"

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_bytes / 1024**3

    @property
    def vram_free_gb(self) -> float:
        return self.vram_free_bytes / 1024**3

    @property
    def ram_total_gb(self) -> float:
        return self.ram_total_bytes / 1024**3

    @property
    def ram_available_gb(self) -> float:
        return self.ram_available_bytes / 1024**3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "device_name": self.device_name,
            "vram_total_gb": round(self.vram_total_gb, 2),
            "vram_free_gb": round(self.vram_free_gb, 2),
            "compute_capability": (
                f"{self.capability_major}.{self.capability_minor}"
                if self.capability_major is not None
                else None
            ),
            "ram_total_gb": round(self.ram_total_gb, 2),
            "ram_available_gb": round(self.ram_available_gb, 2),
            "cpu_count": self.cpu_count,
            "platform": self.platform_str,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "bitsandbytes_available": self.bitsandbytes_available,
            "bitsandbytes_version": self.bitsandbytes_version,
            "warnings": list(self.warnings),
        }


def probe_hardware(check_bnb: bool = True) -> HardwareInfo:
    """Collect a full hardware snapshot. Never raises on missing torch."""
    info = HardwareInfo()

    # OS / CPU / RAM -------------------------------------------------------
    info.platform_str = f"{platform.system()} {platform.release()} ({platform.machine()})"
    info.python_version = platform.python_version()
    info.cpu_count = os.cpu_count() or 0
    vm = psutil.virtual_memory()
    info.ram_total_bytes = vm.total
    info.ram_available_bytes = vm.available

    # torch / CUDA ---------------------------------------------------------
    try:
        import torch

        info.torch_version = torch.__version__
        info.cuda_available = bool(torch.cuda.is_available())
        if info.cuda_available:
            props = torch.cuda.get_device_properties(0)
            info.device_name = props.name
            info.vram_total_bytes = props.total_memory
            try:
                info.vram_free_bytes = torch.cuda.mem_get_info(0)[0]
            except Exception:
                # Older torch fallback: total minus currently allocated.
                info.vram_free_bytes = props.total_memory - torch.cuda.memory_allocated(0)
            info.capability_major = props.major
            info.capability_minor = props.minor
            try:
                info.cuda_version = getattr(torch.version, "cuda", None)
            except Exception:
                info.cuda_version = None
    except Exception as exc:  # torch missing entirely
        info.warnings.append(f"PyTorch could not be imported: {exc}")

    # bitsandbytes ---------------------------------------------------------
    if check_bnb:
        try:
            import bitsandbytes as bnb  # noqa: F401

            info.bitsandbytes_available = True
            info.bitsandbytes_version = getattr(bnb, "__version__", "unknown")
        except Exception:
            info.bitsandbytes_available = False
            info.bitsandbytes_version = None

    # Warnings -------------------------------------------------------------
    if not info.cuda_available:
        info.warnings.append(
            "CUDA is not available. Training will run on CPU and be very slow. "
            "Check that an NVIDIA driver is installed and that the torch wheel "
            "matches your CUDA version."
        )
    elif info.vram_total_bytes < MIN_VRAM_BYTES:
        info.warnings.append(
            f"VRAM ({info.vram_total_gb:.2f} GB) is below the 2 GB minimum for "
            "comfortable training. Expect out-of-memory errors; use a smaller "
            "model or CPU mode."
        )
    if info.ram_available_bytes < MIN_RAM_BYTES:
        info.warnings.append(
            f"Available system RAM ({info.ram_available_gb:.2f} GB) is low. "
            "Close other applications before training. The data loader uses "
            "num_workers=0 and streaming to minimize footprint."
        )
    if info.ram_total_bytes < 8 * 1024**3:
        info.warnings.append(
            f"Total system RAM is {info.ram_total_gb:.1f} GB (below the 8 GB "
            "design target). Keep batch_size small and avoid loading large "
            "datasets into memory."
        )
    return info


def get_device(preference: str = "auto") -> str:
    """Resolve a device preference string to 'cuda' or 'cpu'."""
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    # auto
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_dtype(device: str, preference: str = "auto"):
    """Map a dtype preference string to a torch.dtype, with sensible defaults."""
    import torch

    if preference == "float32":
        return torch.float32
    if preference == "float16":
        return torch.float16
    if preference == "bfloat16":
        return torch.bfloat16
    # auto
    if device == "cuda":
        # bfloat16 is preferable on Ampere+ but RTX 2050 (Turing/Ampere mobile)
        # supports it; fall back to float16 if bf16 is unsupported.
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def estimate_scratch_vram_bytes(
    vocab_size: int,
    n_embd: int,
    n_layer: int,
    block_size: int,
    batch_size: int,
    dtype_bytes: int = 4,
) -> Dict[str, int]:
    """Rough peak-VRAM estimate for scratch training.

    Accounts for parameters, gradients, AdamW moments (2x fp32), and a
    conservative activation footprint. This is an *estimate*, not a guarantee;
    the trainer still handles real OOM at runtime.
    """
    # parameters
    per_layer_params = 12 * n_embd * n_embd + 4 * n_embd
    params = (
        vocab_size * n_embd          # token embedding
        + block_size * n_embd        # position embedding
        + n_layer * per_layer_params
        + 2 * n_embd                 # final layernorm
        + vocab_size * n_embd        # lm head (untied worst case)
    )
    param_bytes = params * dtype_bytes
    grad_bytes = params * 4           # grads kept in fp32
    optimizer_bytes = params * 8      # AdamW m + v in fp32

    # activations: rough O(batch * seq * n_embd * n_layer * ~16) with SDPA
    seq = block_size
    act_bytes = batch_size * seq * n_embd * n_layer * 16 * dtype_bytes

    # CUDA context + fragmentation overhead
    overhead = 500 * 1024**2

    total = param_bytes + grad_bytes + optimizer_bytes + act_bytes + overhead
    return {
        "params": params,
        "param_bytes": param_bytes,
        "grad_bytes": grad_bytes,
        "optimizer_bytes": optimizer_bytes,
        "activation_bytes": act_bytes,
        "overhead_bytes": overhead,
        "total_bytes": total,
    }


def estimate_finetune_vram_bytes(
    model_params: int,
    lora_params: int,
    load_in_4bit: bool,
    max_seq_length: int,
    batch_size: int,
) -> Dict[str, int]:
    """Rough peak-VRAM estimate for LoRA / QLoRA finetuning."""
    if load_in_4bit:
        weight_bytes = model_params // 2          # 4-bit ~ 0.5 byte/param
    else:
        weight_bytes = model_params * 2           # fp16
    # LoRA weights + grads + AdamW
    lora_bytes = lora_params * (2 + 4 + 8)        # fp16 w + fp32 g + 2x fp32 m/v
    # activations with gradient checkpointing (conservative)
    act_bytes = batch_size * max_seq_length * 1024 * 2 * 2
    overhead = 600 * 1024**2
    total = weight_bytes + lora_bytes + act_bytes + overhead
    return {
        "weight_bytes": weight_bytes,
        "lora_bytes": lora_bytes,
        "activation_bytes": act_bytes,
        "overhead_bytes": overhead,
        "total_bytes": total,
    }


def fits_in_vram(estimate_bytes: int, hw: HardwareInfo, safety_factor: float = 0.9) -> bool:
    """True when the estimate is under safety_factor * total VRAM."""
    if not hw.cuda_available:
        return True  # CPU has no VRAM constraint (RAM is the limit there)
    return estimate_bytes <= hw.vram_total_bytes * safety_factor


# ---------------------------------------------------------------------------
# Pretty report / CLI
# ---------------------------------------------------------------------------

def format_report(hw: HardwareInfo) -> str:
    lines = [
        "=" * 62,
        "  DIPAI Hardware Report",
        "=" * 62,
        f"  Platform          : {hw.platform_str}",
        f"  Python            : {hw.python_version}",
        f"  CPU threads       : {hw.cpu_count}",
        f"  System RAM        : {hw.ram_total_gb:.2f} GB total, "
        f"{hw.ram_available_gb:.2f} GB available",
        "",
        f"  PyTorch           : {hw.torch_version or 'not installed'}",
        f"  CUDA available    : {hw.cuda_available}",
    ]
    if hw.cuda_available:
        lines += [
            f"  CUDA version      : {hw.cuda_version}",
            f"  GPU               : {hw.device_name}",
            f"  Compute capability: {hw.capability_major}.{hw.capability_minor}",
            f"  VRAM              : {hw.vram_total_gb:.2f} GB total, "
            f"{hw.vram_free_gb:.2f} GB free",
        ]
    bnb = (
        f"available ({hw.bitsandbytes_version})"
        if hw.bitsandbytes_available
        else "not available (4-bit QLoRA disabled)"
    )
    lines += [
        f"  bitsandbytes      : {bnb}",
        "-" * 62,
    ]
    if hw.warnings:
        lines.append("  WARNINGS:")
        for w in hw.warnings:
            lines.append(f"    ! {w}")
    else:
        lines.append("  No warnings. Hardware looks good.")
    lines.append("=" * 62)
    return "\n".join(lines)


def main_cli() -> None:
    """Entry point for `python -m dipai.hardware` / the console script."""
    hw = probe_hardware()
    print(format_report(hw))


if __name__ == "__main__":
    main_cli()
