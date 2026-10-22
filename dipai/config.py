"""Configuration loading, validation, and seed handling for DIPAI.

Configs are YAML files with typed dataclass schemas. Every hyper-parameter is
overridable from the config file; nothing is silently changed at runtime.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed, or invalid."""


# ---------------------------------------------------------------------------
# Scratch-mode schema (Mode A)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Architecture of the from-scratch decoder-only Transformer."""

    vocab_size: int = 8192
    block_size: int = 256          # maximum context length (tokens)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = False             # LayerNorm / Linear bias (GPT-2 style: off)
    tie_weights: bool = True       # tie embedding <-> LM-head weights

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ConfigError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ConfigError(f"block_size must be > 0, got {self.block_size}")
        if self.n_layer <= 0:
            raise ConfigError(f"n_layer must be > 0, got {self.n_layer}")
        if self.n_head <= 0:
            raise ConfigError(f"n_head must be > 0, got {self.n_head}")
        if self.n_embd <= 0:
            raise ConfigError(f"n_embd must be > 0, got {self.n_embd}")
        if self.n_embd % self.n_head != 0:
            raise ConfigError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}")

    def param_budget_estimate(self) -> int:
        """Rough parameter count (embeddings + layers + head)."""
        embed = self.vocab_size * self.n_embd
        pos = self.block_size * self.n_embd
        # per-layer: attn (4 * n_embd^2) + mlp (8 * n_embd^2 with 4x expand)
        # + 2 layernorms (2*n_embd each with bias, n_embd without).
        ln = 2 * self.n_embd if self.bias else self.n_embd
        per_layer = 12 * self.n_embd * self.n_embd + 2 * ln
        layers = self.n_layer * per_layer
        head = self.n_embd * self.vocab_size if not self.tie_weights else 0
        final_ln = ln
        return embed + pos + layers + head + final_ln


@dataclass
class TrainConfig:
    """Optimization / data settings for scratch training."""

    # data
    data_dir: str = "data/processed"
    tokenizer_path: str = "data/processed/tokenizer.model"
    train_file: str = "train.bin"       # tokenized uint16 binary
    val_file: str = "val.bin"

    # optimization
    batch_size: int = 8
    gradient_accumulation_steps: int = 4   # effective batch = batch_size * this
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    lr_decay_iters: int = 2000
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # system
    seed: int = 1337
    device: str = "auto"            # auto | cuda | cpu
    dtype: str = "auto"             # auto | float16 | bfloat16 | float32
    num_workers: int = 0            # 0 recommended on low-RAM machines
    pin_memory: bool = False        # False recommended when RAM < 8 GB

    # checkpointing
    ckpt_dir: str = "data/checkpoints"  # default overridden by configs
    save_interval: int = 500
    keep_last_n: int = 3
    resume: bool = False            # auto-resume if a latest checkpoint exists

    # memory / stopping
    gradient_checkpointing: bool = False   # recompute activations in backward
    early_stop_val_loss: float = 0.0       # stop when val_loss <= this (0 = off)
    early_stop_patience: int = 0           # stop after N evals w/o improvement (0 = off)

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError(f"batch_size must be > 0, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ConfigError("gradient_accumulation_steps must be > 0")
        if self.max_iters <= 0:
            raise ConfigError(f"max_iters must be > 0, got {self.max_iters}")
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ConfigError("beta1 and beta2 must be in [0, 1)")
        if self.dtype not in ("auto", "float16", "bfloat16", "float32"):
            raise ConfigError(f"unsupported dtype: {self.dtype}")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ConfigError(f"unsupported device: {self.device}")
        if self.grad_clip <= 0:
            raise ConfigError(f"grad_clip must be > 0, got {self.grad_clip}")
        if self.early_stop_val_loss < 0:
            raise ConfigError("early_stop_val_loss must be >= 0 (0 disables)")
        if self.early_stop_patience < 0:
            raise ConfigError("early_stop_patience must be >= 0 (0 disables)")
        if self.keep_last_n < 0:
            raise ConfigError("keep_last_n must be >= 0")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class ParamTarget:
    """Acceptance range for a model's total parameter count.

    When present in a config, training refuses to start if the built model's
    measured parameter count falls outside [min_params, max_params].
    """

    min_params: int = 0
    max_params: int = 0            # 0 -> no upper bound

    @property
    def enabled(self) -> bool:
        return self.min_params > 0 or self.max_params > 0

    def validate(self) -> None:
        if self.min_params < 0 or self.max_params < 0:
            raise ConfigError("param_target bounds must be >= 0")
        if self.max_params and self.min_params > self.max_params:
            raise ConfigError(
                f"param_target min ({self.min_params}) > max ({self.max_params})"
            )

    def contains(self, n_params: int) -> bool:
        if self.min_params and n_params < self.min_params:
            return False
        if self.max_params and n_params > self.max_params:
            return False
        return True

    def describe(self) -> str:
        lo = f"{self.min_params / 1e6:.1f}M" if self.min_params else "0"
        hi = f"{self.max_params / 1e6:.1f}M" if self.max_params else "inf"
        return f"[{lo}, {hi}]"


@dataclass
class ScratchConfig:
    """Top-level config for Mode A (train from scratch)."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run_name: str = "dipai-scratch"
    param_target: Optional[ParamTarget] = None

    def validate(self) -> None:
        self.model.validate()
        self.train.validate()
        if self.param_target is not None:
            self.param_target.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Finetune-mode schema (Mode B)
# ---------------------------------------------------------------------------

@dataclass
class LoraConfigData:
    """LoRA adapter hyper-parameters."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def validate(self) -> None:
        if self.r <= 0:
            raise ConfigError(f"LoRA rank r must be > 0, got {self.r}")
        if self.lora_alpha <= 0:
            raise ConfigError(f"lora_alpha must be > 0, got {self.lora_alpha}")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ConfigError("lora_dropout must be in [0, 1)")
        if not self.target_modules:
            raise ConfigError("target_modules must not be empty")


@dataclass
class FinetuneConfig:
    """Top-level config for Mode B (fine-tune a pretrained model)."""

    # model selection - user MUST confirm this before weights download
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    trust_remote_code: bool = False

    # quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"    # float16 | bfloat16
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora: LoraConfigData = field(default_factory=LoraConfigData)

    # training
    output_dir: str = "checkpoints/finetune"
    max_seq_length: int = 512
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1              # -1 -> use epochs
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    seed: int = 1337
    resume_from_checkpoint: Optional[str] = None

    # data
    dataset_path: str = "data/processed/finetune.jsonl"   # jsonl with "text" key
    val_dataset_path: str = ""                # optional held-out jsonl (same schema)
    max_samples: int = -1           # -1 -> all

    # DIPAI scratch-checkpoint SFT (Mode B, native branch). When set, the run
    # fine-tunes a local DIPAI checkpoint with full parameters instead of a
    # Hugging Face model with LoRA; model_name_or_path/load_in_4bit are ignored
    # (logged explicitly, never silently).
    dipai_checkpoint: str = ""

    # safety
    dry_run_vram_check: bool = True  # print estimate and exit if clearly too big

    def validate(self) -> None:
        if not self.model_name_or_path:
            raise ConfigError(
                "model_name_or_path is empty. Set a Hugging Face model id or a "
                "local path in configs/finetune.yaml before fine-tuning."
            )
        if self.max_seq_length <= 0:
            raise ConfigError("max_seq_length must be > 0")
        if self.per_device_train_batch_size <= 0:
            raise ConfigError("per_device_train_batch_size must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ConfigError("gradient_accumulation_steps must be > 0")
        if self.learning_rate <= 0:
            raise ConfigError("learning_rate must be > 0")
        if self.bnb_4bit_compute_dtype not in ("float16", "bfloat16", "float32"):
            raise ConfigError(
                f"unsupported bnb_4bit_compute_dtype: {self.bnb_4bit_compute_dtype}"
            )
        if self.fp16 and self.bf16:
            raise ConfigError("fp16 and bf16 cannot both be true")
        self.lora.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Vision-mode schema (image understanding, separate pretrained VLM)
# ---------------------------------------------------------------------------

@dataclass
class VisionConfig:
    """Settings for the image-understanding module (dipai.vision).

    The vision model is a SEPARATE pretrained vision-language model (VLM).
    It never modifies the scratch model's vocabulary or checkpoints: loading
    or failing to load it cannot affect text-only training or inference.
    """

    # model selection (HF model id or a local directory)
    model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"

    # device / dtype
    device: str = "auto"             # auto | cuda | cpu
    dtype: str = "auto"              # auto | float16 | bfloat16 | float32
    cpu_fallback: bool = True        # fall back to CPU when CUDA loading OOMs

    # loading behaviour
    load_on_startup: bool = False    # False -> load lazily on first image use
    low_cpu_mem_usage: bool = True   # stream weights (needs accelerate)
    local_files_only: bool = False   # True -> never contact the hub

    # upload safety limits
    max_upload_mb: float = 10.0      # maximum image file size (MB)
    max_image_side: int = 4096       # reject if width OR height exceeds this
    max_image_pixels: int = 4_000_000  # reject if width*height exceeds this

    # generation
    max_new_tokens: int = 256        # hard cap on generated tokens
    temperature: float = 0.2         # 0 -> greedy (deterministic)
    top_p: float = 0.95              # nucleus sampling threshold

    def validate(self) -> None:
        if not self.model_id:
            raise ConfigError(
                "vision.model_id is empty. Set a Hugging Face model id or a "
                "local path in configs/vision.yaml."
            )
        if self.device not in ("auto", "cuda", "cpu"):
            raise ConfigError(f"vision.device unsupported: {self.device}")
        if self.dtype not in ("auto", "float16", "bfloat16", "float32"):
            raise ConfigError(f"vision.dtype unsupported: {self.dtype}")
        if self.max_upload_mb <= 0:
            raise ConfigError("vision.max_upload_mb must be > 0")
        if self.max_image_side <= 0:
            raise ConfigError("vision.max_image_side must be > 0")
        if self.max_image_pixels <= 0:
            raise ConfigError("vision.max_image_pixels must be > 0")
        if self.max_new_tokens <= 0:
            raise ConfigError("vision.max_new_tokens must be > 0")
        if self.temperature < 0:
            raise ConfigError("vision.temperature must be >= 0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ConfigError("vision.top_p must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# YAML <-> dataclass helpers
# ---------------------------------------------------------------------------

def _build(cls, data: Dict[str, Any]):
    """Recursively construct a dataclass from a plain dict, raising on unknown keys."""
    import dataclasses as _dc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"expected mapping for {cls.__name__}, got {type(data).__name__}")
    if not _dc.is_dataclass(cls):
        raise ConfigError(f"{cls!r} is not a dataclass")
    field_map = cls.__dataclass_fields__  # type: ignore[attr-defined]
    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        if name not in field_map:
            raise ConfigError(
                f"unknown key '{name}' in {cls.__name__}. "
                f"Valid keys: {sorted(field_map)}"
            )
        ft = field_map[name]
        default_factory = getattr(ft, "default_factory", None)
        # Nested dataclass via default_factory (e.g. LoraConfigData).
        if (
            default_factory is not None
            and default_factory is not _dc.MISSING
            and callable(default_factory)
            and isinstance(value, dict)
        ):
            sample = default_factory()
            if _dc.is_dataclass(sample) and not isinstance(sample, type):
                kwargs[name] = _build(type(sample), value)
                continue
        kwargs[name] = value
    return cls(**kwargs)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__} in {path}")
    return data


def load_scratch_config(path: str | Path) -> ScratchConfig:
    """Load and validate a Mode-A scratch config from YAML."""
    raw = load_yaml(path)
    model_raw = raw.pop("model", {})
    train_raw = raw.pop("train", {})
    param_target_raw = raw.pop("param_target", None)
    top_keys = {"run_name"}
    unknown = set(raw) - top_keys
    if unknown:
        raise ConfigError(f"unknown top-level keys in {path}: {sorted(unknown)}")
    param_target: Optional[ParamTarget] = None
    if param_target_raw is not None:
        if not isinstance(param_target_raw, dict):
            raise ConfigError(
                f"param_target in {path} must be a mapping with min_params/max_params"
            )
        param_target = _build(ParamTarget, param_target_raw)
    cfg = ScratchConfig(
        model=_build(ModelConfig, model_raw),
        train=_build(TrainConfig, train_raw),
        run_name=raw.get("run_name", "dipai-scratch"),
        param_target=param_target,
    )
    cfg.validate()
    return cfg


def load_finetune_config(path: str | Path) -> FinetuneConfig:
    """Load and validate a Mode-B finetune config from YAML."""
    raw = load_yaml(path)
    cfg = _build(FinetuneConfig, raw)  # nested lora handled by _build
    cfg.validate()
    return cfg


def load_vision_config(path: Optional[str | Path] = None, required: bool = False) -> VisionConfig:
    """Load the vision config from YAML.

    Missing file behaviour:
      * required=True  -> ConfigError (strict, like the other loaders)
      * required=False -> validated defaults, so text-only setups and tests
                          work without a configs/vision.yaml present.
    """
    if path is None or not Path(path).exists():
        if required and path is not None:
            raise ConfigError(f"config file not found: {path}")
        cfg = VisionConfig()
        cfg.validate()
        return cfg
    raw = load_yaml(path)
    cfg = _build(VisionConfig, raw)
    cfg.validate()
    return cfg


def default_vision_config(path: Optional[str | Path] = None) -> VisionConfig:
    """Vision config from `path` (default: configs/vision.yaml), or validated
    built-in defaults when the file does not exist."""
    return load_vision_config(
        path if path is not None else "configs/vision.yaml", required=False
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed python, numpy (if present) and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic kernels can be slow; only enable when requested via env.
        if os.environ.get("DIPAI_DETERMINISTIC", "0") == "1":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
