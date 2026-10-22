"""Checkpoint management for DIPAI scratch training.

Guarantees:
  * Atomic writes: every checkpoint lands as a .tmp file that is fsynced and
    then os.replace()'d into place. A crash mid-save never corrupts an
    existing checkpoint.
  * A `latest.json` pointer (not a symlink - Windows-safe) names the newest
    complete checkpoint for fast resume.
  * Each checkpoint directory carries a snapshot of the config and tokenizer
    files, so a checkpoint is self-describing for reload/export.
  * Retention: keep_last_n numbered checkpoints; older ones are pruned after
    each successful save. The pointer is always valid after pruning.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

log = logging.getLogger("dipai.checkpoints")

POINTER_NAME = "latest.json"
CKPT_PATTERN = re.compile(r"^ckpt_(\d{8})\.pt$")
TMP_SUFFIX = ".tmp"


class CheckpointError(Exception):
    """Raised on missing/corrupt checkpoints or invalid resume requests."""


def _ckpt_path(ckpt_dir: Path, step: int) -> Path:
    return ckpt_dir / f"ckpt_{step:08d}.pt"


def save_checkpoint(
    ckpt_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    tokenizer_src: Optional[str | Path] = None,
    keep_last_n: int = 3,
) -> Path:
    """Atomically save a full training checkpoint. Returns the final path.

    Contents: model state, optimizer/scaler state, step, config dict, and
    whatever the caller puts in `extra` (loss history, rng state, ...).
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "format_version": 1,
        "step": int(step),
        "saved_at": time.time(),
        "model": model.state_dict(),
        "model_config": getattr(model, "cfg", None).__dict__
        if getattr(model, "cfg", None) is not None
        else None,
        "num_params": getattr(model, "num_params", None),
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload["extra"] = extra

    final = _ckpt_path(ckpt_dir, step)
    tmp = final.with_suffix(final.suffix + TMP_SUFFIX)
    try:
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)  # atomic on Windows (same volume)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

    # Update pointer.
    pointer = ckpt_dir / POINTER_NAME
    ptr_tmp = pointer.with_suffix(".json" + TMP_SUFFIX)
    ptr_tmp.write_text(
        json.dumps({"step": int(step), "file": final.name, "saved_at": payload["saved_at"]}, indent=2),
        encoding="utf-8",
    )
    os.replace(ptr_tmp, pointer)

    # Self-describing artifacts.
    if config is not None:
        try:
            import yaml

            cfg_copy = ckpt_dir / "config.yaml"
            cfg_copy.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        except Exception as exc:  # non-fatal
            log.warning("could not snapshot config into checkpoint dir: %s", exc)
    if tokenizer_src is not None:
        _copy_tokenizer(tokenizer_src, ckpt_dir)

    _prune(ckpt_dir, keep_last_n)
    log.info("checkpoint saved: %s", final)
    return final


def _copy_tokenizer(tokenizer_src: str | Path, dest: Path) -> None:
    """Copy the tokenizer model + sidecars next to the checkpoint."""
    src = Path(tokenizer_src)
    candidates = [src]
    if src.suffix == ".model":
        candidates.append(Path(str(src)[: -len(".model")]))  # SP prefix
    for c in candidates:
        for f in (
            Path(str(c) + ".model"),
            Path(str(c) + ".model.meta.json"),
            Path(str(c) + ".vocab"),
            c,
        ):
            if f.is_file() and f.suffix in (".model", ".json", ".vocab"):
                target = dest / f.name
                if f.resolve() != target.resolve():
                    try:
                        shutil.copy2(f, target)
                    except OSError as exc:
                        log.debug("tokenizer copy skipped (%s): %s", f, exc)


def _prune(ckpt_dir: Path, keep_last_n: int) -> None:
    """Delete all but the newest keep_last_n numbered checkpoints."""
    if keep_last_n <= 0:
        return
    ckpts = list_checkpoints(ckpt_dir)
    for old in ckpts[:-keep_last_n]:
        try:
            old.unlink()
            log.debug("pruned old checkpoint %s", old.name)
        except OSError as exc:
            log.warning("could not prune %s: %s", old.name, exc)


def list_checkpoints(ckpt_dir: str | Path) -> List[Path]:
    """All numbered checkpoint files, oldest first."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return []
    out = []
    for f in ckpt_dir.iterdir():
        m = CKPT_PATTERN.match(f.name)
        if m and f.is_file():
            out.append(f)
    out.sort(key=lambda p: int(CKPT_PATTERN.match(p.name).group(1)))  # type: ignore[union-attr]
    return out


def latest_checkpoint(ckpt_dir: str | Path) -> Optional[Path]:
    """Resolve the newest complete checkpoint via the pointer (fallback: scan)."""
    ckpt_dir = Path(ckpt_dir)
    pointer = ckpt_dir / POINTER_NAME
    if pointer.exists():
        try:
            info = json.loads(pointer.read_text(encoding="utf-8"))
            cand = ckpt_dir / info["file"]
            if cand.exists():
                return cand
            log.warning("pointer %s references missing file %s", pointer.name, info.get("file"))
        except (OSError, ValueError, KeyError) as exc:
            log.warning("unreadable checkpoint pointer (%s); falling back to scan", exc)
    ckpts = list_checkpoints(ckpt_dir)
    return ckpts[-1] if ckpts else None


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint file (or a directory containing latest.json).

    Validates the payload shape and raises CheckpointError with a clear
    message on corruption or format mismatch.
    """
    p = Path(path)
    if p.is_dir():
        latest = latest_checkpoint(p)
        if latest is None:
            raise CheckpointError(
                f"no checkpoint found in {p}. Train first, or disable resume."
            )
        p = latest
    if not p.exists():
        raise CheckpointError(f"checkpoint file not found: {p}")

    try:
        # weights_only=False: our checkpoints hold python dicts (config etc.).
        payload = torch.load(p, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise CheckpointError(
            f"failed to read checkpoint {p}: {exc}. "
            "The file may be corrupt (an interrupted save); "
            "delete it or resume from an older step."
        ) from exc

    if not isinstance(payload, dict) or "model" not in payload:
        raise CheckpointError(
            f"{p} is not a DIPAI checkpoint (missing 'model' key)."
        )
    if "step" not in payload:
        raise CheckpointError(f"{p} is missing the 'step' field; cannot resume.")
    return payload


def prune(ckpt_dir: str | Path, keep_last_n: int = 3) -> int:
    """Public prune; returns how many files were removed."""
    ckpt_dir = Path(ckpt_dir)
    before = list_checkpoints(ckpt_dir)
    _prune(ckpt_dir, keep_last_n)
    after = list_checkpoints(ckpt_dir)
    return len(before) - len(after)
