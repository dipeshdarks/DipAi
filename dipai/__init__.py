"""DIPAI - a local, memory-efficient text language model (DIPDARKS).

Two operating modes:
  * Mode A - train a small decoder-only Transformer from scratch.
  * Mode B - fine-tune a compact pretrained causal LM with LoRA / QLoRA.
"""

__version__ = "0.1.0"
__author__ = "DIPDARKS"

from dipai.config import (
    FinetuneConfig,
    ScratchConfig,
    VisionConfig,
    load_finetune_config,
    load_scratch_config,
    load_vision_config,
    set_seed,
)
from dipai.hardware import HardwareInfo, get_device, probe_hardware

__all__ = [
    "FinetuneConfig",
    "HardwareInfo",
    "ScratchConfig",
    "VisionConfig",
    "__version__",
    "get_device",
    "load_finetune_config",
    "load_scratch_config",
    "load_vision_config",
    "probe_hardware",
    "set_seed",
]
