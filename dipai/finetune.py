"""DIPAI Mode B - fine-tune a model. Two mutually exclusive branches.

Native DIPAI branch (dipai_checkpoint set - e.g. configs/sft_oasst.yaml):
    python -m dipai.finetune --config configs/sft_oasst.yaml           # dry run: VRAM estimate only
    python -m dipai.finetune --config configs/sft_oasst.yaml --train   # actually train
  * Full-parameter SFT of a LOCAL DIPAI scratch checkpoint (Mode A output).
  * Dataset: {"messages": [{"role": "user"|"assistant", "content": ...}]}
    conversations (scripts/prepare_finetune_data.py --oasst-threads).
  * Loss is MASKED on user turns: only assistant tokens (and the turn
    separators the assistant must produce) contribute to training.
  * Prompt layout = dipai.conversations.format_dialog = the SAME layout
    chat inference uses (inference.ConversationSession.as_prompt).
  * Hugging Face model_name_or_path / LoRA / 4-bit settings are ignored in
    this branch (logged explicitly, never silently).
  * Exit codes: 0 ok, 1 runtime/RAM, 2 input/config, 3 resume, 4 CUDA OOM,
    130 interrupted (emergency checkpoint saved).

HF LoRA/QLoRA branch (dipai_checkpoint empty - configs/finetune.yaml):
    python -m dipai.finetune --config configs/finetune.yaml           # dry run: VRAM estimate only
    python -m dipai.finetune --config configs/finetune.yaml --train   # actually train
    python -m dipai.finetune --train --max-steps 10                   # short smoke run
  * 4-bit QLoRA via bitsandbytes (nf4 + double quant + fp16 compute);
    downloads model weights from Hugging Face on first run only.
  * LoRA adapters via PEFT; base weights stay frozen and quantized.

Design (RTX 2050 4 GB / ~6 GB RAM):
  * Lazy per-sample tokenization from JSONL (byte-offset index; the corpus
    is never fully resident - matters with <1 GB free RAM).
  * Safe defaults: memory estimate BEFORE any weights are downloaded or
    loaded, explicit `--train` gate when dry_run_vram_check: true,
    RAM pre-flight before reading a checkpoint payload.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipai.checkpoints import (
    CheckpointError,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from dipai.config import (
    FinetuneConfig,
    ModelConfig,
    load_finetune_config,
    set_seed,
)
from dipai.hardware import (
    estimate_scratch_vram_bytes,
    fits_in_vram,
    get_device,
    probe_hardware,
    resolve_dtype,
)
from dipai.model import DipaiGPT
from dipai.tokenizer import DipaiTokenizer, TokenizerError
from dipai.utils import (
    clear_cuda_cache,
    format_bytes,
    format_duration,
    is_cuda_oom,
    setup_logging,
)

log = None  # set in main()


# ---------------------------------------------------------------------------
# VRAM estimate (before any weights are downloaded/loaded)
# ---------------------------------------------------------------------------

# Param counts for well-known fallbacks (hf_hub model_info is preferred).
_KNOWN_PARAM_COUNTS = {
    "Qwen/Qwen2.5-0.5B-Instruct": 494_000_000,
    "Qwen/Qwen2.5-0.5B": 494_000_000,
    "Qwen/Qwen2.5-1.5B-Instruct": 1_540_000_000,
    "Qwen/Qwen2.5-3B-Instruct": 3_090_000_000,
    "gpt2": 124_000_000,
    "distilgpt2": 82_000_000,
}


def _param_count(model_id: str) -> Optional[int]:
    """Best-effort parameter count without loading weights."""
    if Path(model_id).is_dir():  # local model: read safetensors header only
        try:
            total = 0
            for sf in Path(model_id).glob("*.safetensors"):
                with open(sf, "rb") as f:
                    header_len = int.from_bytes(f.read(8), "little")
                    header = json.loads(f.read(header_len).decode("utf-8"))
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    n = 1
                    for d in info.get("shape", []):
                        n *= int(d)
                    total += n
            if total > 0:
                return total
        except Exception:
            pass
    if model_id in _KNOWN_PARAM_COUNTS:
        return _KNOWN_PARAM_COUNTS[model_id]
    try:  # hub metadata includes safetensors totals (requires network)
        from huggingface_hub import model_info

        info = model_info(model_id)
        st = getattr(info, "safetensors", None)
        if st and st.get("parameters"):
            total = st["parameters"].get("total")
            if total:
                return int(sum(total.values()) if isinstance(total, dict) else total)
    except Exception:
        pass
    return None


def estimate_finetune_vram(cfg: FinetuneConfig, param_count: Optional[int]) -> Dict[str, int]:
    """Rough peak-VRAM budget for a LoRA/QLoRA run. Conservative constants."""
    n = float(param_count or 500_000_000)  # assume 0.5B if unknown (warned by caller)
    if cfg.load_in_4bit:
        weight_bytes = n * 0.56            # nf4 + double-quant + scale overhead
    else:
        weight_bytes = n * 2.0             # fp16/bf16 base weights
    if cfg.use_lora:
        # LoRA trainable: ~2% of params for r=16 on q/k/v/o (empirical).
        trainable = max(1_000_000, n * 0.02)
        # fp32 master + grads + Adam m/v for adapter params.
        adapter_optim = trainable * 16.0
    else:
        # Full-parameter CPT: every param gets fp32 master copy + grad + 2
        # Adam moments (4 fp32 slots on top of the fp16/bf16 weights).
        trainable = n
        adapter_optim = n * 12.0
    # Activations: batch * seq * layers * hidden * bytes * constant / grad-ckpt.
    layers = 24 if n < 800_000_000 else 28
    hidden = 896 if n < 800_000_000 else 1536
    ckpt_div = 6.0 if cfg.gradient_checkpointing else 30.0
    activation = cfg.per_device_train_batch_size * cfg.max_seq_length * layers * hidden * ckpt_div * 2.0
    overhead = 600 * 1024 * 1024           # CUDA context, buffers, fragmentation
    total = weight_bytes + adapter_optim + activation + overhead
    return {
        "param_count": int(n),
        "weight_bytes": int(weight_bytes),
        "adapter_bytes": int(adapter_optim),
        "activation_bytes": int(activation),
        "overhead_bytes": int(overhead),
        "total_bytes": int(total),
    }


# ---------------------------------------------------------------------------
# Dataset: lazy JSONL -> fixed-length token samples
# ---------------------------------------------------------------------------

class JsonlTextDataset(torch.utils.data.Dataset):
    """Reads `{"text": ...}` lines on demand via a byte-offset index.

    Memory profile: one line buffer + one tokenized sample at a time, plus
    the offset index (~8 bytes/line). Safe for <1 GB free RAM.
    """

    def __init__(self, path: str | Path, tokenizer, max_seq_length: int, max_samples: int = -1):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        if not self.path.exists():
            raise FileNotFoundError(
                f"finetune dataset not found: {self.path}. "
                "Create it with: python scripts/prepare_finetune_data.py"
            )
        self.offsets: List[int] = []
        with open(self.path, "rb") as f:
            while True:
                off = f.tell()
                if not f.readline():
                    break
                self.offsets.append(off)
        if max_samples > 0:
            self.offsets = self.offsets[:max_samples]
        if not self.offsets:
            raise ValueError(f"finetune dataset is empty: {self.path}")

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        obj = json.loads(line.decode("utf-8", errors="replace"))
        text = str(obj.get("text", "")).strip()
        ids = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            add_special_tokens=True,
        )["input_ids"]
        # Fixed length: right-pad to max_seq_length.
        attn = [1] * len(ids)
        pad_n = self.max_seq_length - len(ids)
        if pad_n > 0:
            ids = ids + [self.pad_id] * pad_n
            attn = attn + [0] * pad_n
        labels = [t if a == 1 else -100 for t, a in zip(ids, attn)]
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Dataset: conversations JSONL -> (x, y) pairs with masked user loss
# ---------------------------------------------------------------------------

class ConversationDataset(torch.utils.data.Dataset):
    """`{"messages": [...]}` JSONL -> (x, y) next-token pairs for SFT.

    Serialization is exactly the text layout of
    dipai.conversations.format_dialog ("User: ...\\n\\nAssistant: ..."), built
    one turn at a time so the loss mask can follow turn boundaries:
      * user turns (including the separator that PRECEDES them) -> -100
      * assistant turns (including their leading "\\n\\n" separator, and a
        trailing eos when the conversation ends on an assistant) -> supervised
    so the model is trained to emit turn separators, the "Assistant: " header
    and reply content - never to imitate user text.

    Overlong conversations drop the oldest FULL exchange (2 turns) while at
    least two exchanges remain (keeps User-first alternation), then truncate
    at the tail. An extreme user-only overflow rebuilds from the first
    assistant turn; as a last resort the final real transition is supervised
    - a sample can never be 100% masked (that would yield a NaN loss).

    Right-padding to the batch max happens in `_collate_conversations`; pads
    sit AFTER real tokens (causal attention => no effect on them) and their
    targets are -100 (never counted).
    """

    def __init__(self, path: str | Path, tokenizer, max_seq_length: int, max_samples: int = -1):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)
        self.pad_id = tokenizer.pad_id
        self.eos_id = tokenizer.eos_id
        if self.max_seq_length < 2:
            raise ValueError(f"max_seq_length must be >= 2, got {self.max_seq_length}")
        if not self.path.exists():
            raise FileNotFoundError(
                f"SFT dataset not found: {self.path}. Build it with: "
                "python scripts/prepare_finetune_data.py --oasst-threads"
            )
        self.offsets: List[int] = []
        with open(self.path, "rb") as f:
            while True:
                off = f.tell()
                if not f.readline():
                    break
                self.offsets.append(off)
        if max_samples > 0:
            self.offsets = self.offsets[:max_samples]
        if not self.offsets:
            raise ValueError(f"SFT dataset is empty: {self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def _serialize(self, messages: List[dict]) -> Tuple[List[int], List[int]]:
        """Turns -> (ids, labels); labels are -100 on user-side positions."""
        ids: List[int] = []
        labels: List[int] = []
        for i, m in enumerate(messages):
            role = m.get("role")
            text = str(m.get("content", "")).strip()
            seg = f"{'User' if role == 'user' else 'Assistant'}: {text}"
            if i > 0:
                seg = "\n\n" + seg
            seg_ids = self.tokenizer.encode(seg, add_bos=(i == 0), add_eos=False)
            ids.extend(seg_ids)
            labels.extend(seg_ids if role == "assistant" else [-100] * len(seg_ids))
        if messages and messages[-1].get("role") == "assistant":
            ids.append(self.eos_id)
            labels.append(self.eos_id)
        return ids, labels

    def _read(self, idx: int) -> List[dict]:
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        try:
            obj = json.loads(line.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self.path} record {idx + 1}: invalid JSON ({exc})") from exc
        messages = obj.get("messages") if isinstance(obj, dict) else None
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"{self.path} record {idx + 1}: needs a non-empty 'messages' list "
                "(build it with prepare_finetune_data.py --oasst-threads)"
            )
        for j, m in enumerate(messages):
            if (
                not isinstance(m, dict)
                or m.get("role") not in ("user", "assistant")
                or not str(m.get("content", "")).strip()
            ):
                raise ValueError(
                    f"{self.path} record {idx + 1}, message {j + 1}: needs role "
                    "user|assistant and non-empty content"
                )
        return messages

    def _fit(self, messages: List[dict]) -> Tuple[List[int], List[int]]:
        """Serialize, dropping oldest exchanges then tail-truncating to fit."""
        msgs = messages
        ids, labels = self._serialize(msgs)
        # Drop the oldest exchange (2 turns) only while >= 2 exchanges would
        # remain - a shorter remainder must keep its User-first opening.
        while len(ids) > self.max_seq_length and len(msgs) >= 4:
            msgs = msgs[2:]
            ids, labels = self._serialize(msgs)
        if len(ids) > self.max_seq_length:
            ids = ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]
        return ids, labels

    @staticmethod
    def _to_xy(ids: List[int], labels: List[int]) -> Tuple[List[int], List[int]]:
        # Position i predicts ids[i+1]; supervise only when that NEXT token
        # sits in an assistant-side position of `labels`.
        x = ids[:-1]
        y = [ids[i + 1] if labels[i + 1] != -100 else -100 for i in range(len(ids) - 1)]
        return x, y

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        messages = self._read(idx)
        ids, labels = self._fit(messages)
        if len(ids) < 2:
            raise ValueError(f"{self.path} record {idx + 1}: serializes to fewer than 2 tokens")
        x, y = self._to_xy(ids, labels)
        if not any(v != -100 for v in y):
            # User-only prefix overflowed the window: rebuild from the first
            # assistant turn so the sample still carries supervision.
            first = next(
                (i for i, m in enumerate(messages) if m.get("role") == "assistant"), None
            )
            if first is not None and first > 0:
                alt_ids, alt_labels = self._fit(messages[first:])
                if len(alt_ids) >= 2:
                    ids, labels = alt_ids, alt_labels
                    x, y = self._to_xy(ids, labels)
        if not any(v != -100 for v in y):
            y[-1] = ids[-1]  # last resort: supervise the final real transition
        return {"x": x, "y": y}


def _collate_conversations(batch: List[Dict[str, List[int]]], pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad variable-length (x, y) pairs; pad targets are -100."""
    xs = [b["x"] for b in batch]
    ys = [b["y"] for b in batch]
    t_max = max(len(v) for v in xs)
    x = torch.full((len(xs), t_max), pad_id, dtype=torch.long)
    y = torch.full((len(ys), t_max), -100, dtype=torch.long)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        x[i, : len(xi)] = torch.tensor(xi, dtype=torch.long)
        y[i, : len(yi)] = torch.tensor(yi, dtype=torch.long)
    return x, y


# ---------------------------------------------------------------------------
# Native DIPAI SFT helpers
# ---------------------------------------------------------------------------

def _find_tokenizer_file(ckpt_dir: Path) -> Optional[Path]:
    cand = ckpt_dir / "tokenizer.model"
    return cand if cand.is_file() else None


def _read_model_cfg_from_snapshot(ckpt_dir: Path) -> Optional[Dict]:
    """Cheap model config via the config.yaml that train.py snapshots beside
    a checkpoint - lets a dry run estimate VRAM without touching any weights."""
    p = ckpt_dir / "config.yaml"
    if not p.is_file():
        return None
    try:
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    model = (data or {}).get("model")
    return model if isinstance(model, dict) else None


def _load_payload(path: Path) -> Dict:
    """torch.load with mmap (lazy file-backed tensors) - keeps the RAM cost
    of a checkpoint payload (model + optimizer states) far below its size."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError, ValueError, OSError):
        # Legacy/non-zip payloads cannot be mmapped: plain load (the caller
        # ran a RAM pre-flight first).
        return load_checkpoint(path, map_location="cpu")


def _sft_lr(step: int, peak: float, total: int, warmup_steps: int, floor_frac: float = 0.1) -> float:
    """Linear warmup -> cosine decay to floor_frac * peak."""
    if warmup_steps > 0 and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    floor = peak * floor_frac
    if step >= total:
        return floor
    ratio = min(1.0, max(0.0, (step - warmup_steps) / max(1, total - warmup_steps)))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return floor + coeff * (peak - floor)


def _ram_need_gb(cfg: FinetuneConfig, ckpt_bytes: float) -> float:
    """Conservative system-RAM bound for loading a payload.

    A fresh run only materializes the model state: the optimizer/scaler
    entries stay cold in the mmap'd file (torch.load mmap=True) and the new
    optimizer lives on the GPU, so CPU-side need scales with the model share
    of the payload (fp32 weights are ~1/3 of a training checkpoint that also
    carries two AdamW moments), plus a fixed safety margin. Resuming reads
    the whole payload -> full file-size bound.
    """
    if cfg.resume_from_checkpoint:
        return ckpt_bytes / 1024**3 + 0.35
    return ckpt_bytes / 1024**3 * 0.45 + 0.35


def _print_native_plan(
    cfg: FinetuneConfig,
    model_cfg_dict: Optional[Dict],
    hw,
    ckpt_file: Path,
) -> Optional[int]:
    """Memory pre-flight table (before any checkpoint bytes are read).

    Returns the config-estimated parameter count when available.
    """
    print("DIPAI native SFT - memory pre-flight (before loading weights)")
    print("-" * 60)
    ckpt_bytes = ckpt_file.stat().st_size
    print(f"  checkpoint file    : {ckpt_file.name} ({format_bytes(ckpt_bytes)})")
    params: Optional[int] = None
    if model_cfg_dict:
        try:
            params = ModelConfig(**model_cfg_dict).param_budget_estimate()
            dtype_bytes = 2 if (hw.cuda_available and (cfg.fp16 or cfg.bf16)) else 4
            est = estimate_scratch_vram_bytes(
                vocab_size=int(model_cfg_dict.get("vocab_size", 0)),
                n_embd=int(model_cfg_dict.get("n_embd", 0)),
                n_layer=int(model_cfg_dict.get("n_layer", 0)),
                block_size=int(model_cfg_dict.get("block_size", 0)),
                batch_size=cfg.per_device_train_batch_size,
                dtype_bytes=dtype_bytes,
            )
            print(f"  parameters         : {params:,}")
            print(f"  gradients (fp32)   : {format_bytes(est['grad_bytes'])}")
            print(f"  AdamW moments      : {format_bytes(est['optimizer_bytes'])}")
            print(
                f"  activations        : ~{format_bytes(est['activation_bytes'])}"
                f" (batch {cfg.per_device_train_batch_size}, seq {cfg.max_seq_length},"
                f" grad-ckpt {'on' if cfg.gradient_checkpointing else 'off'})"
            )
            print(f"  overhead reserve   : {format_bytes(est['overhead_bytes'])}")
            print(f"  estimated peak     : ~{format_bytes(est['total_bytes'])}")
            if hw.cuda_available:
                fits = fits_in_vram(est["total_bytes"], hw)
                print(f"  VRAM free          : {hw.vram_free_gb:.2f} GB -> {'FITS' if fits else 'DOES NOT FIT'}")
                if not fits:
                    print("  ADVICE: reduce per_device_train_batch_size or enable gradient_checkpointing.")
        except Exception as exc:  # never block a dry run on estimate math
            print(f"  parameters         : estimate failed ({exc})")
    else:
        print("  parameters         : unknown (no config.yaml beside the checkpoint;")
        print("                        the estimate is printed after --train loads it)")
    print(f"  effective batch    : {cfg.per_device_train_batch_size} x {cfg.gradient_accumulation_steps}"
          f" = {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    print(f"  RAM available      : {hw.ram_available_gb:.2f} GB"
          f" (payload load needs ~{_ram_need_gb(cfg, ckpt_bytes):.2f} GB"
          f"{' full-file bound: resume' if cfg.resume_from_checkpoint else '; fresh-run bound'})")
    print()
    return params


def _infinite(loader: DataLoader) -> Iterator:
    """Cycle a DataLoader forever (epochs are implicit in step-driven SFT)."""
    while True:
        for batch in loader:
            yield batch


def _append_sft_row(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "train_loss", "val_loss", "lr", "elapsed_s"]
        )
        if new_file:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def _evaluate_sft(model, loader, device: str, dtype: torch.dtype) -> float:
    """Mean masked cross-entropy over the full val loader (deterministic)."""
    model.eval()
    autocast_enabled = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    losses: List[float] = []
    try:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=autocast_enabled):
                out = model(x, y)
            losses.append(float(out.loss))
    finally:
        model.train()
    return sum(losses) / max(1, len(losses))


# ---------------------------------------------------------------------------
# Native DIPAI SFT (Mode B, local-checkpoint branch)
# ---------------------------------------------------------------------------

def run_native_sft(cfg: FinetuneConfig, do_train: bool) -> int:
    """Full-parameter SFT of a local DIPAI checkpoint. Returns an exit code."""
    log = setup_logging(log_file=None)
    t0 = time.perf_counter()
    ckpt_dir = Path(cfg.dipai_checkpoint)

    print()
    print("DIPAI native SFT - full-parameter fine-tune of a local checkpoint")
    print("-" * 60)
    print(f"  checkpoint         : {ckpt_dir}")
    print(
        f"  HF settings ignored in this branch: model_name_or_path={cfg.model_name_or_path!r}, "
        f"load_in_4bit={cfg.load_in_4bit}, LoRA r={cfg.lora.r}"
    )

    # ---- input validation (fail fast, both modes) ---------------------------
    if not ckpt_dir.is_dir():
        log.error(
            "checkpoint dir not found: %s. Train the base model first "
            "(python -m dipai.train --config configs/dipai_50m.yaml).",
            ckpt_dir,
        )
        return 2
    train_path = Path(cfg.dataset_path)
    if not train_path.is_file():
        log.error(
            "SFT train dataset not found: %s. Build it with: "
            "python scripts/prepare_finetune_data.py --oasst-threads",
            train_path,
        )
        return 2
    val_path = Path(cfg.val_dataset_path) if cfg.val_dataset_path else None
    if val_path is not None and not val_path.is_file():
        log.error("val_dataset_path is set but the file does not exist: %s", val_path)
        return 2

    src_file = latest_checkpoint(ckpt_dir)
    if src_file is None:
        log.error("no checkpoint (ckpt_*.pt) found in %s", ckpt_dir)
        return 2

    # ---- memory pre-flight BEFORE any payload bytes are read ---------------
    hw = probe_hardware(check_bnb=False)
    model_cfg_dict = _read_model_cfg_from_snapshot(ckpt_dir)
    _print_native_plan(cfg, model_cfg_dict, hw, src_file)
    ram_need_gb = _ram_need_gb(cfg, src_file.stat().st_size)
    ram_ok = hw.ram_available_gb >= ram_need_gb
    if not ram_ok:
        log.warning(
            "free RAM (%.2f GB) is below the ~%.2f GB this payload load may need; "
            "close other applications (especially any parallel trainer) before --train.",
            hw.ram_available_gb, ram_need_gb,
        )

    if not do_train:
        print(
            "dry run complete (dry_run_vram_check: true). "
            "Pass --train (or set dry_run_vram_check: false) to start training."
        )
        return 0
    if not ram_ok:
        log.error(
            "not enough free RAM to load %s (%.2f GB file): need ~%.2f GB, "
            "have %.2f GB. Close other applications and retry.",
            src_file.name, src_file.stat().st_size / 1024**3,
            ram_need_gb, hw.ram_available_gb,
        )
        return 1

    # ---- resolve payload: resume output takes precedence --------------------
    resume_path: Optional[Path] = None
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint == "auto":
            resume_path = latest_checkpoint(Path(cfg.output_dir))
        else:
            rp = Path(cfg.resume_from_checkpoint)
            resume_path = latest_checkpoint(rp) if rp.is_dir() else rp
        if resume_path is None or not resume_path.is_file():
            log.error(
                "resume requested but no checkpoint found in %s (looked for ckpt_*.pt)",
                cfg.output_dir,
            )
            return 3
    payload_path = resume_path if resume_path is not None else src_file
    log.info("loading checkpoint: %s", payload_path)
    try:
        payload = _load_payload(payload_path)
    except CheckpointError as exc:
        log.error("%s", exc)
        return 3 if resume_path is not None else 2
    if not isinstance(payload, dict) or "model" not in payload:
        log.error("%s is not a DIPAI checkpoint (missing 'model')", payload_path)
        return 2

    model_cfg_dict = payload.get("model_config") or model_cfg_dict
    if not model_cfg_dict:
        log.error(
            "checkpoint %s has no model_config; cannot rebuild the architecture.",
            payload_path.name,
        )
        return 2
    try:
        model_cfg = ModelConfig(**model_cfg_dict)
    except Exception as exc:
        log.error("checkpoint model_config is unusable: %s", exc)
        return 2

    # ---- tokenizer (self-describing: copied beside the checkpoint) ----------
    tok_file = _find_tokenizer_file(payload_path.parent) or _find_tokenizer_file(ckpt_dir)
    if tok_file is None:
        cfg_tok = ((payload.get("config") or {}).get("train") or {}).get("tokenizer_path")
        if cfg_tok and Path(cfg_tok).exists():
            tok_file = Path(cfg_tok)
    if tok_file is None:
        log.error(
            "tokenizer not found: %s has no tokenizer.model and no usable "
            "tokenizer path in its config snapshot.",
            payload_path.parent,
        )
        return 2
    try:
        tok = DipaiTokenizer.load(tok_file)
    except TokenizerError as exc:
        log.error("%s", exc)
        return 2
    if tok.vocab_size != model_cfg.vocab_size:
        log.error(
            "tokenizer vocab (%d) does not match checkpoint vocab (%d): %s and %s "
            "are from different runs. Use the tokenizer stored with the checkpoint.",
            tok.vocab_size, model_cfg.vocab_size, tok_file, payload_path,
        )
        return 2

    # ---- model ---------------------------------------------------------------
    model = DipaiGPT(model_cfg)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing:
        log.error("checkpoint %s is incomplete; missing keys: %s", payload_path.name, missing[:8])
        del payload
        gc.collect()
        return 2
    if unexpected:
        log.warning("extra keys in checkpoint ignored: %s", unexpected[:8])
    optimizer_state = payload.get("optimizer") if resume_path is not None else None
    scaler_state = payload.get("scaler") if resume_path is not None else None
    start_step = int(payload.get("step", 0)) if resume_path is not None else 0
    del payload
    gc.collect()

    device = get_device("auto")
    dtype = resolve_dtype(device, "float16" if cfg.fp16 else "bfloat16" if cfg.bf16 else "float32")
    log.info("device=%s dtype=%s | GPU=%s | VRAM free=%.2f GB | RAM avail=%.2f GB",
             device, dtype, hw.device_name or "-", hw.vram_free_gb, hw.ram_available_gb)
    model = model.to(device)
    if cfg.gradient_checkpointing:
        model.set_gradient_checkpointing(True)
        log.info("gradient checkpointing: ON (activations recomputed in backward)")
    total_params = model.num_params
    src_step = int(src_file.stem.split("_")[-1]) if src_file.stem.split("_")[-1].isdigit() else 0
    log.info("model: %.2fM params | tokenizer: %s (%d vocab) | base checkpoint step %d",
             total_params / 1e6, tok_file, tok.vocab_size, src_step)

    optimizer = model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
        betas=(0.9, 0.999),
        device_type=device,
    )
    if resume_path is not None and optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            log.info("optimizer state restored from resume checkpoint")
        except Exception as exc:
            log.warning("optimizer state not restored (%s); weights are restored", exc)
    use_scaler = device == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    if resume_path is not None and scaler_state is not None and use_scaler:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception as exc:
            log.warning("grad scaler state not restored: %s", exc)

    max_seq = min(cfg.max_seq_length, model_cfg.block_size)
    if max_seq < cfg.max_seq_length:
        log.warning(
            "max_seq_length %d exceeds model block_size %d -> clamped to %d",
            cfg.max_seq_length, model_cfg.block_size, max_seq,
        )

    # ---- data ----------------------------------------------------------------
    try:
        train_ds = ConversationDataset(train_path, tok, max_seq, cfg.max_samples)
        val_ds = ConversationDataset(val_path, tok, max_seq) if val_path else None
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    log.info("dataset: train=%d conversations%s (%s)",
             len(train_ds), f", val={len(val_ds)}" if val_ds else "", train_path)

    collate = functools.partial(_collate_conversations, pad_id=tok.pad_id)
    loader_gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.per_device_train_batch_size, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=collate, generator=loader_gen,
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=max(2, cfg.per_device_train_batch_size * 2),
            shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate,
        )
        if val_ds is not None
        else None
    )

    if cfg.max_steps > 0:
        total_steps = int(cfg.max_steps)
    else:
        total_steps = math.ceil(len(train_loader) * cfg.num_train_epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    eff_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    log.info(
        "schedule: %d steps | warmup %d -> cosine to 10%% of lr | effective batch %d "
        "(%d x %d) | grad-clip 1.0",
        total_steps, warmup_steps, eff_batch,
        cfg.per_device_train_batch_size, cfg.gradient_accumulation_steps,
    )

    if start_step >= total_steps:
        log.info("nothing to do: start_step=%d >= total_steps=%d", start_step, total_steps)
        return 0

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_log_path = out_dir / "sft_loss_log.csv"
    autocast_enabled = device == "cuda" and dtype in (torch.float16, torch.bfloat16)

    set_seed(cfg.seed)
    batch_iter = _infinite(train_loader)
    running_loss = 0.0
    micro_steps = 0
    best_val: Optional[float] = None
    history: List[Dict] = []
    oom_fallback_used = False
    masked_logged = False
    t_start = time.perf_counter()
    step = start_step

    model.train()
    optimizer.zero_grad(set_to_none=True)
    log.info("training: steps %d..%d (save/eval every %d, log every %d)",
             start_step + 1, total_steps, cfg.save_steps, cfg.logging_steps)

    try:
        for step in range(start_step + 1, total_steps + 1):
            lr = _sft_lr(step - 1, cfg.learning_rate, total_steps, warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ---- one accumulation cycle, with OOM recovery -------------------
            # Same contract as dipai.train: a failed cycle is aborted cleanly
            # (partial grads zeroed, cache cleared), ONE safe fallback is
            # applied, and the same step is retried; a second OOM aborts with
            # the real failure + proposed settings (success is never faked).
            step_loss = 0.0
            while True:
                try:
                    for micro in range(cfg.gradient_accumulation_steps):
                        x, y = next(batch_iter)
                        x = x.to(device)
                        y = y.to(device)
                        with torch.autocast(device_type=device, dtype=dtype, enabled=autocast_enabled):
                            out = model(x, y)
                            loss = out.loss / cfg.gradient_accumulation_steps
                        scaler.scale(loss).backward()
                        step_loss += float(out.loss.detach())
                        if not masked_logged and micro == 0:
                            masked_logged = True
                            log.info("loss mask: %.1f%% of targets supervised "
                                     "(user turns masked out)", 100.0 * float((y != -100).float().mean()))
                    break  # accumulation cycle complete
                except Exception as exc:
                    if not is_cuda_oom(exc):
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    clear_cuda_cache()
                    if oom_fallback_used:
                        log.error(
                            "CUDA OOM AGAIN at SFT step %d after the automatic fallback "
                            "(batch=%d, accum=%d, grad-ckpt=%s). Real failure: %s. "
                            "Proposed settings: per_device_train_batch_size: 1, "
                            "gradient_accumulation_steps: %d, gradient_checkpointing: true; "
                            "if it still fails, lower max_seq_length (%d). "
                            "Last checkpoint preserved.",
                            step, cfg.per_device_train_batch_size,
                            cfg.gradient_accumulation_steps, cfg.gradient_checkpointing,
                            exc, eff_batch, max_seq,
                        )
                        return 4
                    if not cfg.gradient_checkpointing:
                        cfg.gradient_checkpointing = True
                        model.set_gradient_checkpointing(True)
                        log.warning(
                            "CUDA OOM at step %d - fallback 1/1: gradient checkpointing "
                            "ENABLED; retrying step %d.", step, step,
                        )
                    elif cfg.per_device_train_batch_size > 1:
                        old_bs = cfg.per_device_train_batch_size
                        new_bs = max(1, old_bs // 2)
                        old_accum = cfg.gradient_accumulation_steps
                        cfg.gradient_accumulation_steps = old_accum * (old_bs // new_bs)
                        cfg.per_device_train_batch_size = new_bs
                        log.warning(
                            "CUDA OOM at step %d - fallback 1/1: batch %d -> %d, accum "
                            "%d -> %d (effective batch unchanged at %d); retrying step %d.",
                            step, old_bs, new_bs, old_accum,
                            cfg.gradient_accumulation_steps, eff_batch, step,
                        )
                    else:
                        log.error(
                            "CUDA OOM at SFT step %d with no fallback left (batch=1, "
                            "gradient checkpointing already on). Real failure: %s. "
                            "Lower max_seq_length (%d) or free VRAM. Last checkpoint "
                            "preserved.", step, exc, max_seq,
                        )
                        return 4
                    oom_fallback_used = True
                    continue  # retry the same accumulation cycle

            # ---- optimizer step ----------------------------------------------
            if cfg.fp16 and device == "cuda":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += step_loss
            micro_steps += cfg.gradient_accumulation_steps

            # ---- logging ------------------------------------------------------
            if step % cfg.logging_steps == 0 or step == total_steps:
                now = time.perf_counter()
                avg_loss = running_loss / max(1, micro_steps)
                log.info("step %4d/%d | train_loss %.4f | lr %.2e | %s",
                         step, total_steps, avg_loss, lr, format_duration(now - t_start))
                _append_sft_row(loss_log_path, {
                    "step": step, "train_loss": f"{avg_loss:.6f}", "val_loss": "",
                    "lr": f"{lr:.3e}", "elapsed_s": f"{now - t_start:.1f}",
                })
                running_loss = 0.0
                micro_steps = 0

            # ---- validation + checkpoint ----------------------------------------
            if step % cfg.save_steps == 0 or step == total_steps:
                val_loss: Optional[float] = None
                if val_loader is not None:
                    val_loss = _evaluate_sft(model, val_loader, device, dtype)
                    improved = best_val is None or val_loss < best_val
                    if improved:
                        best_val = val_loss
                    history.append({"step": step, "val_loss": val_loss})
                    log.info("step %4d | val_loss %.4f | best %s",
                             step, val_loss, f"{best_val:.4f}" if best_val else "-")
                    _append_sft_row(loss_log_path, {
                        "step": step, "train_loss": "", "val_loss": f"{val_loss:.6f}",
                        "lr": f"{lr:.3e}", "elapsed_s": f"{time.perf_counter() - t_start:.1f}",
                    })
                save_checkpoint(
                    out_dir, step, model, optimizer, scaler,
                    config=cfg.to_dict(),
                    extra={"history": history[-50:], "best_val": best_val,
                           "branch": "native-sft"},
                    tokenizer_src=str(tok_file),
                    keep_last_n=cfg.save_total_limit,
                )

    except KeyboardInterrupt:
        log.warning("interrupted by user - saving emergency checkpoint ...")
        save_checkpoint(
            out_dir, step, model, optimizer, scaler, config=cfg.to_dict(),
            extra={"interrupted": True, "branch": "native-sft",
                   "history": history[-50:]},
            tokenizer_src=str(tok_file), keep_last_n=cfg.save_total_limit,
        )
        log.info("saved at step %d; resume with: --resume (same config)", step)
        return 130
    except Exception as exc:
        if is_cuda_oom(exc):
            optimizer.zero_grad(set_to_none=True)
            clear_cuda_cache()
            log.error(
                "CUDA out of memory at SFT step %d (real failure: %s). Partial "
                "gradients cleared; last checkpoint preserved. Proposed settings: "
                "per_device_train_batch_size: 1, gradient_accumulation_steps: %d, "
                "gradient_checkpointing: true; if it still fails, lower "
                "max_seq_length (%d).", step, exc, eff_batch, max_seq,
            )
            return 4
        raise

    log.info("SFT complete: %d steps in %s | best val_loss=%s",
             total_steps, format_duration(time.perf_counter() - t_start),
             f"{best_val:.4f}" if best_val is not None else "n/a")
    log.info("checkpoint dir: %s", out_dir)
    log.info("next: python -m dipai.chat --ckpt %s", out_dir)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DIPAI Mode B: LoRA/QLoRA fine-tune")
    ap.add_argument("--config", "-c", default="configs/finetune.yaml")
    ap.add_argument("--train", action="store_true",
                    help="actually train even if dry_run_vram_check is true")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override train.max_steps for this run")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="override dataset max_samples for this run")
    ap.add_argument("--output-dir", default=None, help="override output_dir")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the newest checkpoint in output_dir")
    return ap


def run(cfg: FinetuneConfig, do_train: bool) -> int:
    # Native branch first: it must never import/require the HF stack.
    if cfg.dipai_checkpoint:
        return run_native_sft(cfg, do_train)

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from transformers.utils import is_torch_bf16_available  # noqa: F401 (env probe)

    t0 = time.perf_counter()

    # ---- VRAM estimate BEFORE weights -------------------------------------
    n_params = _param_count(cfg.model_name_or_path)
    est = estimate_finetune_vram(cfg, n_params)
    hw_free = None
    try:
        from dipai.hardware import probe_hardware

        hw = probe_hardware(check_bnb=False)
        hw_free = hw.vram_free_gb
    except Exception:
        pass

    print()
    print("DIPAI Mode B - VRAM estimate (before loading weights)")
    print("-" * 56)
    print(f"  model               : {cfg.model_name_or_path}")
    print(f"  quantization        : {'4-bit nf4 (QLoRA)' if cfg.load_in_4bit else 'full-precision (no quant)'}")
    print(f"  parameters          : {est['param_count']:,}" + ("" if n_params else " (assumed)"))
    print(f"  base weights        : {format_bytes(est['weight_bytes'])}")
    print(f"  LoRA + optimizer    : {format_bytes(est['adapter_bytes'])}")
    print(f"  activations         : {format_bytes(est['activation_bytes'])}"
          f" (bs {cfg.per_device_train_batch_size}, seq {cfg.max_seq_length},"
          f" grad-ckpt {'on' if cfg.gradient_checkpointing else 'off'})")
    print(f"  overhead reserve    : {format_bytes(est['overhead_bytes'])}")
    print(f"  estimated peak      : {format_bytes(est['total_bytes'])}")
    if hw_free is not None:
        fits = est["total_bytes"] < hw_free * (1024**3) * 0.95
        print(f"  VRAM free           : {hw_free:.2f} GB -> {'FITS' if fits else 'DOES NOT FIT'}")
        if not fits:
            print("  ADVICE: reduce max_seq_length / batch size, or use load_in_4bit: true.")
    print()

    if not do_train:
        print("dry run complete (dry_run_vram_check: true). "
              "Pass --train (or set dry_run_vram_check: false) to start training.")
        return 0

    # ---- tokenizer / model -------------------------------------------------
    log.info("loading tokenizer: %s", cfg.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=cfg.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("tokenizer has no pad token -> using eos token as pad")

    quant_config = None
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[cfg.bnb_4bit_compute_dtype]
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )

    load_dtype = torch.float16 if cfg.fp16 else (
        torch.bfloat16 if cfg.bf16 else torch.float32
    )
    log.info("loading model: %s (4bit=%s) ...", cfg.model_name_or_path, cfg.load_in_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        quantization_config=quant_config,
        torch_dtype=load_dtype,
        device_map="auto",
    )
    model.config.use_cache = False

    if cfg.use_lora:
        # ---- LoRA (adapter training) -------------------------------------
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.gradient_checkpointing
            )
        lora_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=list(cfg.lora.target_modules),
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        # ---- full-parameter continued-pretraining (no adapter) ------------
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("full-parameter mode (use_lora: false): %s trainable params",
                 f"{n_train:,}")
        if cfg.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

    if cfg.gradient_checkpointing:
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    # ---- data ----------------------------------------------------------------
    train_ds = JsonlTextDataset(
        cfg.dataset_path, tokenizer, cfg.max_seq_length, cfg.max_samples
    )
    log.info("dataset: %s samples x seq %d (%s)",
             len(train_ds), cfg.max_seq_length, cfg.dataset_path)

    # ---- trainer --------------------------------------------------------------
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        dataloader_num_workers=0,      # low-RAM machines
        dataloader_pin_memory=False,
        report_to=[],
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        log_level="info",
        resume_from_checkpoint=cfg.resume_from_checkpoint,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    log.info("starting fine-tune: %s", cfg.model_name_or_path)
    try:
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    except torch.cuda.OutOfMemoryError:
        log.error(
            "CUDA out of memory during fine-tune. Reduce max_seq_length (now %d), "
            "per_device_train_batch_size, or ensure load_in_4bit: true.",
            cfg.max_seq_length,
        )
        return 4

    # ---- save model + tokenizer ---------------------------------------------
    if cfg.use_lora:
        save_dir = out_dir / "adapter"
    else:
        save_dir = out_dir / "model"
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    log.info("%s saved: %s",
             "adapter" if cfg.use_lora else "model", save_dir)
    log.info("peak VRAM during training: %s", format_bytes(peak))
    log.info("fine-tune complete in %s", f"{time.perf_counter() - t0:.0f}s")
    log.info("use it: python -m dipai.chat --model %s (see docs/finetune)", save_dir)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    global log
    log = setup_logging()
    args = build_arg_parser().parse_args(argv)
    try:
        cfg = load_finetune_config(args.config)
    except Exception as exc:
        log.error("failed to load config: %s", exc)
        return 2
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.resume:
        if cfg.dipai_checkpoint:
            # Native branch: newest ckpt_*.pt in output_dir (resolved there).
            cfg.resume_from_checkpoint = "auto"
        else:
            from transformers.trainer_utils import get_last_checkpoint

            cfg.resume_from_checkpoint = None  # stale config value must not leak
            if Path(cfg.output_dir).exists():
                last = get_last_checkpoint(cfg.output_dir)
                if last:
                    cfg.resume_from_checkpoint = str(last)
                    log.info("resuming from %s", last)

    do_train = args.train or not cfg.dry_run_vram_check
    try:
        return run(cfg, do_train)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - surface clearly for CLI users
        log.exception("fine-tune failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
