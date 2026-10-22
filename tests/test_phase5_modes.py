"""Verifies the Mode B full-parameter (use_lora: false) save/load wiring.

Uses a mocked transformers Trainer so the test exercises DIPAI's actual
branching and the new DIPAIEngine.from_model loader without a real (slow) train.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from dipai.config import FinetuneConfig


@pytest.fixture()
def tiny_model_id(tmp_path):
    """A tiny local causal LM (built once) so tests never download a big model."""
    transformers = pytest.importorskip("transformers")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tok.save_pretrained(tmp_path / "tiny")
    cfg = AutoConfig.for_model(
        "llama",
        vocab_size=tok.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(tmp_path / "tiny")
    return str(tmp_path / "tiny")


def _base_cfg(**kw):
    base = dict(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        load_in_4bit=True,
        use_lora=True,
        learning_rate=2e-4,
        dipai_checkpoint="",
    )
    base.update(kw)
    return FinetuneConfig(**base)


def test_full_param_with_4bit_rejected():
    from dipai.config import ConfigError
    with pytest.raises(ConfigError, match="load_in_4bit"):
        _base_cfg(use_lora=False, load_in_4bit=True).validate()


def test_full_param_high_lr_rejected():
    from dipai.config import ConfigError
    with pytest.raises(ConfigError, match="learning_rate"):
        _base_cfg(use_lora=False, load_in_4bit=False, learning_rate=2e-4).validate()


def test_full_param_sane_config_accepted():
    _base_cfg(use_lora=False, load_in_4bit=False, learning_rate=1e-5).validate()


def test_estimator_full_param_costs_more_than_lora():
    from dipai.finetune import estimate_finetune_vram

    n = 500_000_000
    lora = estimate_finetune_vram(_base_cfg(use_lora=True, load_in_4bit=False), n)
    full = estimate_finetune_vram(_base_cfg(use_lora=False, load_in_4bit=False), n)
    assert full["adapter_bytes"] > lora["adapter_bytes"] * 5
    assert full["total_bytes"] > lora["total_bytes"]


def _write_text_jsonl(path: Path, n=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        {"text": f"The quick brown fox number {i} jumps over the lazy dog near the river."}
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")


@pytest.fixture()
def qwen_model_id():
    return "Qwen/Qwen2.5-0.5B-Instruct"


def _cfg(model_id, out_dir, dataset):
    return FinetuneConfig(
        model_name_or_path=model_id,
        use_lora=False,
        load_in_4bit=False,
        fp16=False,
        bf16=False,
        learning_rate=1e-5,
        max_seq_length=64,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=2,
        save_steps=2,
        logging_steps=1,
        gradient_checkpointing=False,
        warmup_ratio=0.0,
        weight_decay=0.0,
        output_dir=str(out_dir),
        seed=7,
        dataset_path=str(dataset),
        dipai_checkpoint="",
        dry_run_vram_check=False,
    )


def test_full_param_save_and_load(tiny_model_id, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    dataset = tmp_path / "cpt.jsonl"
    _write_text_jsonl(dataset)
    out_dir = tmp_path / "out"

    import dipai.finetune as ft
    from dipai.utils import setup_logging
    ft.log = setup_logging()

    captured = {}

    class FakeTrainer:
        def __init__(self, model=None, args=None, train_dataset=None):
            captured["model"] = model
            captured["train_dataset"] = train_dataset

        def train(self, resume_from_checkpoint=None):
            return None

    from transformers import Trainer as RealTrainer

    with patch("transformers.Trainer", FakeTrainer):
        with patch.object(RealTrainer, "train", lambda self, **kw: None):
            rc = ft.run(_cfg(tiny_model_id, out_dir, dataset), do_train=True)

    assert rc == 0
    # The model handed to Trainer must be a plain causal LM, not a PEFT wrapper
    model = captured["model"]
    assert not hasattr(model, "peft_config"), "use_lora=False should not wrap in PEFT"
    # Save path: my code writes out_dir/"model"
    model_dir = out_dir / "model"
    assert model_dir.exists(), "full-param save should create out_dir/model"
    assert (model_dir / "config.json").exists()

    # from_model should load it back
    from dipai.inference import DIPAIEngine, GenerationSettings

    eng = DIPAIEngine.from_model(model_dir, device="cpu")
    assert eng.backend == "hf"
    assert eng._status.backend == "hf-full"
    out = eng.generate("The capital of France is",
                       settings=GenerationSettings(max_new_tokens=4))
    assert isinstance(out, str)


def test_lora_path_still_wraps_peft(tiny_model_id, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    dataset = tmp_path / "ft.jsonl"
    _write_text_jsonl(dataset)
    out_dir = tmp_path / "out"

    import dipai.finetune as ft
    from dipai.utils import setup_logging
    ft.log = setup_logging()

    captured = {}

    class FakeTrainer:
        def __init__(self, model=None, args=None, train_dataset=None):
            captured["model"] = model

        def train(self, resume_from_checkpoint=None):
            return None

    from transformers import Trainer as RealTrainer

    cfg = _cfg(tiny_model_id, out_dir, dataset)
    cfg.use_lora = True
    cfg.learning_rate = 2e-4

    with patch("transformers.Trainer", FakeTrainer):
        with patch.object(RealTrainer, "train", lambda self, **kw: None):
            rc = ft.run(cfg, do_train=True)

    assert rc == 0
    model = captured["model"]
    # PEFT wrapper exposes peft_config
    assert hasattr(model, "peft_config"), "use_lora=True should wrap in PEFT"
    assert (out_dir / "adapter").exists(), "LoRA save should create out_dir/adapter"


def test_rng_state_save_restore_roundtrip_is_deterministic():
    """The torch/numpy/python RNG states saved into the checkpoint must, when
    restored, reproduce the exact next draw a continuous run would have made."""
    import random

    import numpy as np

    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # draw a bit to move the state off the seed
    _ = torch.rand(4)
    _ = np.random.rand(3)
    _ = random.random()

    # capture exactly what train.py stores into the checkpoint
    payload = {
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }

    # continuous reference draws (this is what a never-stopped run would emit)
    ref_torch = torch.rand(5)
    ref_np = np.random.rand(5)
    ref_py = [random.random() for _ in range(5)]

    # perturb all three RNGs heavily, then restore from the payload
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    _ = torch.rand(10)
    _ = np.random.rand(10)
    _ = [random.random() for _ in range(10)]

    torch.set_rng_state(
        payload["torch_rng"].cpu().to(torch.uint8)
        if torch.is_tensor(payload["torch_rng"])
        else torch.tensor(payload["torch_rng"], dtype=torch.uint8)
    )
    np.random.set_state(tuple(payload["numpy_rng"]))
    random.setstate(tuple(payload["python_rng"]))

    assert torch.equal(torch.rand(5), ref_torch), "torch RNG restore not deterministic"
    assert np.allclose(np.random.rand(5), ref_np), "numpy RNG restore not deterministic"
    assert [random.random() for _ in range(5)] == ref_py, "python RNG restore not deterministic"
