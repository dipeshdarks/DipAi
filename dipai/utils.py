"""Shared utilities for DIPAI: logging, seeding, byte formatting, OOM handling."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO, log_file: Optional[str | Path] = None) -> logging.Logger:
    """Configure root logging once. Returns the 'dipai' logger."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
        root.setLevel(level)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Avoid duplicate file handlers on repeated setup_logging calls.
        for h in root.handlers:
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve():
                break
        else:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
            )
            root.addHandler(fh)
    return logging.getLogger("dipai")


def get_logger(name: str = "dipai") -> logging.Logger:
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Byte / time formatting
# ---------------------------------------------------------------------------

def format_bytes(n: float) -> str:
    """Human-readable byte string, e.g. 1536 -> '1.5 KiB'."""
    if n < 0:
        return "-" + format_bytes(-n)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    value = float(n)
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.2f} {units[idx]}"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


@contextmanager
def timed(label: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    log = logger or get_logger()
    start = time.perf_counter()
    try:
        yield
    finally:
        log.info("%s took %s", label, format_duration(time.perf_counter() - start))


# ---------------------------------------------------------------------------
# Seeding (thin re-export so callers can `from dipai.utils import set_seed`)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    from dipai.config import set_seed as _set_seed

    _set_seed(seed)


# ---------------------------------------------------------------------------
# CUDA OOM handling
# ---------------------------------------------------------------------------

class OutOfMemoryError(RuntimeError):
    """Raised after a CUDA OOM has been caught and cache cleared."""


def is_cuda_oom(exc: BaseException) -> bool:
    """True when the exception looks like a CUDA out-of-memory error."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


@contextmanager
def oom_guard(action: str = "operation", logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """Catch CUDA OOM, clear cache, and re-raise as OutOfMemoryError with advice."""
    log = logger or get_logger()
    try:
        yield
    except Exception as exc:
        if is_cuda_oom(exc):
            clear_cuda_cache()
            advice = (
                f"CUDA out of memory during {action}. Cache cleared. "
                "Try: reduce batch_size, increase gradient_accumulation_steps, "
                "enable gradient_checkpointing (finetune), lower max_seq_length / "
                "block_size, or use a smaller model."
            )
            log.error(advice)
            raise OutOfMemoryError(advice) from exc
        raise


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(model) -> tuple[int, int]:
    """(total, trainable) parameter counts for a torch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
