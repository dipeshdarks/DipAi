"""DIPAI scratch training loop (Mode A).

Run:
    python -m dipai.train --config configs/scratch.yaml

What it does:
  * Loads YAML config, probes hardware, prints an estimated peak-memory
    budget BEFORE allocating the model.
  * Syncs model vocab_size to the trained tokenizer (logged explicitly -
    it is a resolution, never silent).
  * Enforces config param_target: refuses to train (exit 5) if the measured
    parameter count is outside the configured range.
  * Trains with gradient accumulation, FP16/BF16 autocast, gradient
    clipping, periodic validation, periodic checkpointing, and resume.
  * Optional gradient checkpointing (train.gradient_checkpointing).
  * Explicit stop conditions: early_stop_val_loss / early_stop_patience
    (0 = disabled; checkpoint saved on stop).
  * On CUDA OOM inside an accumulation cycle: zeroes partial gradients,
    clears cache, applies ONE automatic fallback (enable gradient
    checkpointing, else halve batch with accum compensated), retries once;
    a second OOM aborts with the real failure + proposed settings (exit 4).
  * Logs every interval to <ckpt_dir>/loss_log.csv.

Design constraints (targets ~6 GB RAM / 4 GB VRAM):
  * Data is sampled from numpy memmaps - the corpus is never resident.
  * No DataLoader workers (num_workers=0); batches are built inline.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipai.config import ScratchConfig, load_scratch_config, set_seed
from dipai.checkpoints import (
    CheckpointError,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from dipai.dataset import InvalidDatasetError, PackedDataset, load_packed_dataset
from dipai.hardware import (
    estimate_scratch_vram_bytes,
    fits_in_vram,
    get_device,
    probe_hardware,
    resolve_dtype,
)
from dipai.model import DipaiGPT, build_model
from dipai.tokenizer import DipaiTokenizer, TokenizerError
from dipai.utils import (
    OutOfMemoryError,
    clear_cuda_cache,
    count_parameters,
    format_bytes,
    format_duration,
    get_logger,
    is_cuda_oom,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup -> cosine decay to min_lr
# ---------------------------------------------------------------------------

def get_lr(it: int, cfg_train) -> float:
    t = cfg_train
    if it < t.warmup_iters and t.warmup_iters > 0:
        return t.learning_rate * (it + 1) / t.warmup_iters
    if it >= t.lr_decay_iters:
        return t.min_lr
    decay_ratio = (it - t.warmup_iters) / max(1, t.lr_decay_iters - t.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return t.min_lr + coeff * (t.learning_rate - t.min_lr)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(
    model: DipaiGPT,
    dataset: PackedDataset,
    device: str,
    dtype: torch.dtype,
    eval_iters: int,
    batch_size: int,
    seed: int,
) -> float:
    """Mean cross-entropy over eval_iters fixed batches (reproducible seed)."""
    model.eval()
    rng = np.random.RandomState(seed)
    losses: List[float] = []
    autocast_enabled = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    try:
        for _ in range(eval_iters):
            xs, ys = dataset.sample_batch(batch_size, rng=rng)
            x = torch.from_numpy(xs).to(device, non_blocking=False)
            y = torch.from_numpy(ys).to(device, non_blocking=False)
            with torch.autocast(device_type=device, dtype=dtype, enabled=autocast_enabled):
                out = model(x, y)
            losses.append(float(out.loss))
    finally:
        model.train()
    return sum(losses) / max(1, len(losses))


# ---------------------------------------------------------------------------
# Memory pre-flight
# ---------------------------------------------------------------------------

def print_memory_plan(cfg: ScratchConfig, hw) -> None:
    m, t = cfg.model, cfg.train
    dtype_bytes = 2 if (hw.cuda_available and t.dtype in ("auto", "float16", "bfloat16")) else 4
    est = estimate_scratch_vram_bytes(
        vocab_size=m.vocab_size, n_embd=m.n_embd, n_layer=m.n_layer,
        block_size=m.block_size, batch_size=t.batch_size, dtype_bytes=dtype_bytes,
    )
    log = get_logger()
    log.info("Memory pre-flight (estimate):")
    log.info("  parameters        : ~%s", format_bytes(est["param_bytes"]))
    log.info("  gradients (fp32)  : %s", format_bytes(est["grad_bytes"]))
    log.info("  AdamW moments     : %s", format_bytes(est["optimizer_bytes"]))
    log.info("  activations       : ~%s", format_bytes(est["activation_bytes"]))
    log.info("  overhead reserve  : %s", format_bytes(est["overhead_bytes"]))
    log.info("  estimated peak    : ~%s", format_bytes(est["total_bytes"]))
    if hw.cuda_available:
        fits = fits_in_vram(est["total_bytes"], hw)
        log.info("  VRAM available    : %.2f GB -> %s",
                 hw.vram_free_gb, "FITS" if fits else "DOES NOT FIT")
        if not fits:
            log.warning(
                "Estimated usage exceeds usable VRAM. Training will likely OOM. "
                "Reduce batch_size / block_size / n_layer first."
            )
    log.info("  effective batch   : %d (batch %d x grad-accum %d)",
             t.effective_batch_size, t.batch_size, t.gradient_accumulation_steps)


# ---------------------------------------------------------------------------
# Loss logging
# ---------------------------------------------------------------------------

def append_loss_row(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "train_loss", "val_loss", "lr", "elapsed_s", "tokens_per_s"]
        )
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train(cfg: ScratchConfig, config_path: Optional[str] = None) -> int:
    """Full training run. Returns process exit code (0 = success)."""
    log = setup_logging(log_file=None)
    t = cfg.train
    set_seed(t.seed)

    # ---- hardware -------------------------------------------------------
    hw = probe_hardware(check_bnb=False)
    device = get_device(t.device)
    if t.device == "cuda" and device != "cuda":
        log.error("config requests CUDA but torch.cuda.is_available() is False")
        return 1
    dtype = resolve_dtype(device, t.dtype)
    log.info("device=%s dtype=%s | GPU=%s | VRAM free=%.2f GB | RAM avail=%.2f GB",
             device, dtype, hw.device_name or "-", hw.vram_free_gb, hw.ram_available_gb)

    # ---- tokenizer -------------------------------------------------------
    try:
        tok = DipaiTokenizer.load(t.tokenizer_path)
    except TokenizerError as exc:
        log.error("%s", exc)
        return 2

    # Vocab resolution: the model MUST match the tokenizer. This is logged,
    # and the config snapshot keeps the original value for transparency.
    if cfg.model.vocab_size != tok.vocab_size:
        log.warning(
            "config model.vocab_size=%d but tokenizer vocab=%d -> using %d "
            "(model vocab always follows the tokenizer). "
            "Update configs/scratch.yaml to silence this.",
            cfg.model.vocab_size, tok.vocab_size, tok.vocab_size,
        )
        cfg.model.vocab_size = tok.vocab_size

    # ---- data ------------------------------------------------------------
    data_dir = Path(t.data_dir)
    try:
        train_ds = load_packed_dataset(data_dir, "train", cfg.model.block_size, seed=t.seed)
    except InvalidDatasetError as exc:
        log.error("%s", exc)
        return 2
    val_ds: Optional[PackedDataset] = None
    try:
        val_ds = load_packed_dataset(data_dir, "val", cfg.model.block_size, seed=t.seed + 1)
    except InvalidDatasetError as exc:
        log.warning("no validation split available (%s); training loss only", exc)

    log.info("data: train=%s tokens (%d blocks), val=%s",
             format_bytes(train_ds.token_count() * 2), len(train_ds),
             format_bytes(val_ds.token_count() * 2) if val_ds else "n/a")

    # ---- memory plan ------------------------------------------------------
    print_memory_plan(cfg, hw)

    # ---- model ------------------------------------------------------------
    model = build_model(cfg.model)
    total, trainable = count_parameters(model)
    log.info("model: %.2fM params (%.2fM non-embedding), trainable=%.2fM",
             total / 1e6, model.num_params_nonembed / 1e6, trainable / 1e6)

    # Parameter-count acceptance (config param_target). Checked against the
    # MEASURED count before the model is moved to the GPU, and before any
    # optimizer state is allocated. Absent param_target -> no check.
    pt = cfg.param_target
    if pt is not None and pt.enabled:
        measured = model.num_params
        if not pt.contains(measured):
            log.error(
                "parameter-count check FAILED: built model has %d params "
                "(%.2fM) outside required range %s. Refusing to train. "
                "Fix model vocab_size/n_layer/n_embd/n_head/block_size in the "
                "config (or adjust param_target if the target changed).",
                measured, measured / 1e6, pt.describe(),
            )
            return 5
        log.info("parameter-count check OK: %d params within %s",
                 measured, pt.describe())

    model = model.to(device)
    if t.gradient_checkpointing:
        model.set_gradient_checkpointing(True)
        log.info("gradient checkpointing: ON (activations recomputed in "
                 "backward; memory estimate above is conservative)")

    optimizer = model.configure_optimizers(
        weight_decay=t.weight_decay,
        learning_rate=t.learning_rate,
        betas=(t.beta1, t.beta2),
        device_type=device,
    )
    use_scaler = device == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    autocast_enabled = device == "cuda" and dtype in (torch.float16, torch.bfloat16)

    # ---- resume -----------------------------------------------------------
    start_step = 0
    resume_path: Optional[Path] = None
    if t.resume:
        resume_path = latest_checkpoint(t.ckpt_dir)
    if resume_path is not None:
        try:
            payload = load_checkpoint(resume_path, map_location=device)
        except CheckpointError as exc:
            log.error("%s", exc)
            return 3
        # Architecture must match exactly; refuse rather than corrupt resume.
        ckpt_model_cfg = payload.get("model_config") or {}
        for key in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if ckpt_model_cfg.get(key) != getattr(cfg.model, key):
                log.error(
                    "cannot resume: checkpoint %s has %s=%r but config says %r. "
                    "Use a matching config or start fresh with resume: false.",
                    resume_path.name, key, ckpt_model_cfg.get(key), getattr(cfg.model, key),
                )
                return 3
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if missing:
            log.error("cannot resume: missing keys in checkpoint: %s", missing[:10])
            return 3
        if unexpected:
            log.warning("unexpected keys ignored on resume: %s", unexpected[:10])
        if "optimizer" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except Exception as exc:
                log.warning("optimizer state not restored (%s); weights are restored", exc)
        if "scaler" in payload and use_scaler:
            try:
                scaler.load_state_dict(payload["scaler"])
            except Exception as exc:
                log.warning("grad scaler state not restored: %s", exc)
        start_step = int(payload["step"])
        log.info("resumed from %s at step %d", resume_path.name, start_step)

    if start_step >= t.max_iters:
        log.info("nothing to do: start_step=%d >= max_iters=%d", start_step, t.max_iters)
        return 0

    # ---- training ---------------------------------------------------------
    ckpt_dir = Path(t.ckpt_dir)
    loss_log_path = ckpt_dir / "loss_log.csv"
    eval_rng = np.random.RandomState(t.seed)
    tokens_since_log = 0
    last_log_time = time.perf_counter()
    t_start = time.perf_counter()
    running_loss = 0.0
    micro_steps = 0
    best_val: Optional[float] = None
    history: List[Dict] = []

    model.train()
    optimizer.zero_grad(set_to_none=True)
    log.info("training: steps %d..%d (interval=%d, eval=%d, save=%d)",
             start_step + 1, t.max_iters, t.save_interval, t.eval_interval, t.save_interval)

    stop_reason: Optional[str] = None
    oom_fallback_used = False
    evals_since_best = 0

    try:
        for step in range(start_step + 1, t.max_iters + 1):
            lr = get_lr(step - 1, t)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ---- one accumulation cycle, with OOM recovery -----------------
            # On CUDA OOM: abort the failed cycle cleanly (zero partial
            # gradients, clear cache), auto-apply ONE safe fallback, and retry
            # the same step. A second OOM aborts with the real failure and the
            # proposed settings - success is never fabricated.
            step_loss = 0.0
            step_tokens = 0
            while True:
                try:
                    for _micro in range(t.gradient_accumulation_steps):
                        xs, ys = train_ds.sample_batch(t.batch_size)
                        x = torch.from_numpy(xs).to(device)
                        y = torch.from_numpy(ys).to(device)
                        with torch.autocast(device_type=device, dtype=dtype, enabled=autocast_enabled):
                            out = model(x, y)
                            loss = out.loss / t.gradient_accumulation_steps
                        scaler.scale(loss).backward()
                        step_loss += float(out.loss.detach())
                        step_tokens += t.batch_size * cfg.model.block_size
                    break  # accumulation cycle complete
                except Exception as exc:
                    if not is_cuda_oom(exc):
                        raise
                    optimizer.zero_grad(set_to_none=True)  # abort partial cycle
                    step_loss = 0.0
                    step_tokens = 0
                    clear_cuda_cache()
                    if oom_fallback_used:
                        log.error(
                            "CUDA OOM AGAIN at step %d after automatic fallback "
                            "(batch_size=%d, gradient_accumulation_steps=%d, "
                            "gradient_checkpointing=%s). Real failure: %s. "
                            "Proposed settings: batch_size: 1, "
                            "gradient_accumulation_steps: %d, "
                            "gradient_checkpointing: true; if it still fails, "
                            "reduce block_size (%d) or n_layer (%d). "
                            "Last checkpoint preserved.",
                            step, t.batch_size, t.gradient_accumulation_steps,
                            t.gradient_checkpointing, exc,
                            t.effective_batch_size,
                            cfg.model.block_size, cfg.model.n_layer,
                        )
                        return 4
                    if not t.gradient_checkpointing:
                        oom_fallback_used = True
                        t.gradient_checkpointing = True
                        model.set_gradient_checkpointing(True)
                        log.warning(
                            "CUDA OOM at step %d - automatic fallback 1/1: "
                            "gradient checkpointing ENABLED; partial gradients "
                            "cleared, retrying step %d.",
                            step, step,
                        )
                    elif t.batch_size > 1:
                        oom_fallback_used = True
                        old_bs = t.batch_size
                        new_bs = max(1, old_bs // 2)
                        old_accum = t.gradient_accumulation_steps
                        new_accum = old_accum * (old_bs // new_bs)
                        t.batch_size = new_bs
                        t.gradient_accumulation_steps = new_accum
                        log.warning(
                            "CUDA OOM at step %d - automatic fallback 1/1: "
                            "batch_size %d -> %d, gradient_accumulation_steps "
                            "%d -> %d (effective batch unchanged at %d); "
                            "partial gradients cleared, retrying step %d.",
                            step, old_bs, new_bs, old_accum, new_accum,
                            t.effective_batch_size, step,
                        )
                    else:
                        log.error(
                            "CUDA OOM at step %d with no fallback setting left "
                            "(batch_size=1, gradient checkpointing already on). "
                            "Real failure: %s. Proposed settings: reduce "
                            "block_size (%d) or n_layer (%d), or free VRAM. "
                            "Last checkpoint preserved.",
                            step, exc, cfg.model.block_size, cfg.model.n_layer,
                        )
                        return 4
                    continue  # retry the same accumulation cycle

            # ---- optimizer step --------------------------------------------
            if t.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Successful step -> merge into rolling log stats.
            running_loss += step_loss
            tokens_since_log += step_tokens
            micro_steps += t.gradient_accumulation_steps

            # ---- logging -------------------------------------------------
            if step % 10 == 0 or step == t.max_iters:
                now = time.perf_counter()
                dt = now - last_log_time
                tps = tokens_since_log / dt if dt > 0 else 0.0
                avg_loss = running_loss / max(1, micro_steps)
                log.info("step %5d/%d | train_loss %.4f | lr %.2e | %.0f tok/s | %s",
                         step, t.max_iters, avg_loss, lr, tps,
                         format_duration(now - t_start))
                append_loss_row(loss_log_path, {
                    "step": step, "train_loss": f"{avg_loss:.6f}", "val_loss": "",
                    "lr": f"{lr:.3e}", "elapsed_s": f"{now - t_start:.1f}",
                    "tokens_per_s": f"{tps:.0f}",
                })
                running_loss = 0.0
                micro_steps = 0
                tokens_since_log = 0
                last_log_time = now

            # ---- validation ------------------------------------------------
            if val_ds is not None and (step % t.eval_interval == 0 or step == t.max_iters):
                val_loss = estimate_loss(
                    model, val_ds, device, dtype,
                    eval_iters=t.eval_iters, batch_size=t.batch_size,
                    seed=t.seed + step,
                )
                improved = best_val is None or val_loss < best_val
                if improved:
                    best_val = val_loss
                    evals_since_best = 0
                else:
                    evals_since_best += 1
                log.info("step %5d | val_loss %.4f | best %.4f | no-improve %d/%d",
                         step, val_loss, best_val, evals_since_best,
                         t.early_stop_patience if t.early_stop_patience > 0 else 0)
                append_loss_row(loss_log_path, {
                    "step": step, "train_loss": "", "val_loss": f"{val_loss:.6f}",
                    "lr": f"{lr:.3e}", "elapsed_s": f"{time.perf_counter() - t_start:.1f}",
                    "tokens_per_s": "",
                })
                history.append({"step": step, "val_loss": val_loss})
                eval_rng.seed(step)  # keep deterministic

                # ---- explicit stop conditions (0 = disabled) --------------
                if t.early_stop_val_loss > 0 and val_loss <= t.early_stop_val_loss:
                    stop_reason = (f"val_loss {val_loss:.4f} <= "
                                   f"early_stop_val_loss {t.early_stop_val_loss:g}")
                elif (t.early_stop_patience > 0
                      and evals_since_best >= t.early_stop_patience):
                    stop_reason = (f"no val improvement for {evals_since_best} "
                                   f"evaluations >= early_stop_patience "
                                   f"{t.early_stop_patience}")
                if stop_reason is not None:
                    save_checkpoint(
                        ckpt_dir, step, model, optimizer, scaler,
                        config=cfg.to_dict(),
                        extra={"history": history[-50:], "best_val": best_val,
                               "stop_reason": stop_reason},
                        tokenizer_src=t.tokenizer_path,
                        keep_last_n=t.keep_last_n,
                    )
                    log.info("early stop at step %d: %s (checkpoint saved)",
                             step, stop_reason)
                    break

            # ---- checkpoint ------------------------------------------------
            if step % t.save_interval == 0 or step == t.max_iters:
                save_checkpoint(
                    ckpt_dir, step, model, optimizer, scaler,
                    config=cfg.to_dict(),
                    extra={"history": history[-50:], "best_val": best_val,
                           "torch_rng": torch.get_rng_state()},
                    tokenizer_src=t.tokenizer_path,
                    keep_last_n=t.keep_last_n,
                )

    except KeyboardInterrupt:
        log.warning("interrupted by user - saving emergency checkpoint ...")
        save_checkpoint(
            ckpt_dir, step, model, optimizer, scaler, config=cfg.to_dict(),
            extra={"interrupted": True, "history": history[-50:]},
            tokenizer_src=t.tokenizer_path, keep_last_n=t.keep_last_n,
        )
        log.info("saved at step %d; resume with the same config", step)
        return 130
    except Exception as exc:
        if is_cuda_oom(exc):
            optimizer.zero_grad(set_to_none=True)
            clear_cuda_cache()
            proposed_accum = max(1, t.effective_batch_size)  # with batch_size: 1
            log.error(
                "CUDA out of memory at step %d (real failure: %s). Partial "
                "gradients cleared; last checkpoint preserved. Proposed "
                "settings: batch_size: 1, gradient_accumulation_steps: %d "
                "(effective batch %d), gradient_checkpointing: true; if it "
                "still fails, reduce block_size (%d) or n_layer (%d).",
                step, exc, proposed_accum, t.effective_batch_size,
                cfg.model.block_size, cfg.model.n_layer,
            )
            return 4
        raise
    finally:
        train_ds.close()
        if val_ds is not None:
            val_ds.close()

    if stop_reason is not None:
        log.info("training stopped early at step %d: %s", step, stop_reason)
        log.info("best val_loss=%s | checkpoint dir: %s",
                 f"{best_val:.4f}" if best_val is not None else "n/a", ckpt_dir)
        log.info("next: python -m dipai.chat --ckpt %s", ckpt_dir)
        return 0
    log.info("training complete: %d steps in %s | best val_loss=%s",
             t.max_iters, format_duration(time.perf_counter() - t_start),
             f"{best_val:.4f}" if best_val is not None else "n/a")
    log.info("checkpoint dir: %s", ckpt_dir)
    log.info("next: python -m dipai.chat --ckpt %s", ckpt_dir)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train a DIPAI scratch model (Mode A)")
    ap.add_argument("--config", "-c", default="configs/scratch.yaml",
                    help="path to scratch YAML config")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="override train.max_iters for this run")
    ap.add_argument("--resume", dest="resume", action="store_true", default=None,
                    help="force resume from latest checkpoint")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="start fresh even if checkpoints exist")
    ap.add_argument("--data-dir", default=None, help="override train.data_dir")
    ap.add_argument("--ckpt-dir", default=None, help="override train.ckpt_dir")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    try:
        cfg = load_scratch_config(args.config)
    except Exception as exc:
        setup_logging().error("failed to load config: %s", exc)
        return 2
    if args.max_iters is not None:
        cfg.train.max_iters = args.max_iters
    if args.resume is not None:
        cfg.train.resume = args.resume
    if args.data_dir is not None:
        cfg.train.data_dir = args.data_dir
    if args.ckpt_dir is not None:
        cfg.train.ckpt_dir = args.ckpt_dir
    try:
        cfg.validate()
    except Exception as exc:
        setup_logging().error("invalid config: %s", exc)
        return 2
    return train(cfg, config_path=args.config)


if __name__ == "__main__":
    raise SystemExit(main())
