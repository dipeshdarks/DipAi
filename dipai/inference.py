"""DIPAI inference engine - the single entry point used by terminal chat and
the web UI so both drive the REAL model (no simulation anywhere).

Responsibilities:
  * Load a scratch checkpoint directory (self-describing: model config,
    tokenizer, and training config are snapshotted beside the weights).
  * Load a Mode-B (LoRA/HF) checkpoint via the optional HF backend.
  * Run generation with settings and optional streaming callbacks.
  * Maintain conversation history within a session.
  * Report hardware/model status for the UI badge.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from dipai.checkpoints import CheckpointError, latest_checkpoint, load_checkpoint
from dipai.conversations import format_dialog
from dipai.generate import GenerationError, GenerationSettings, generate_text
from dipai.hardware import get_device, probe_hardware, resolve_dtype
from dipai.model import DipaiGPT
from dipai.tokenizer import DipaiTokenizer, TokenizerError
from dipai.utils import format_bytes

log = logging.getLogger("dipai.inference")


class EngineError(Exception):
    """Raised when the engine cannot load or run a model."""


@dataclass
class EngineStatus:
    """Snapshot shown in chat/webui status areas."""

    backend: str = "scratch"
    checkpoint: str = ""
    model_params: int = 0
    trained_to_step: int = 0
    device: str = "cpu"
    gpu_name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    vocab_size: int = 0
    block_size: int = 0
    ready: bool = False
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "backend": self.backend,
            "checkpoint": self.checkpoint,
            "model_params_m": round(self.model_params / 1e6, 2),
            "trained_to_step": self.trained_to_step,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "vram_total_gb": round(self.vram_total_gb, 2),
            "vram_free_gb": round(self.vram_free_gb, 2),
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "ready": self.ready,
            "notes": self.notes,
        }

    def line(self) -> str:
        vram = (f"VRAM {self.vram_free_gb:.1f}/{self.vram_total_gb:.1f} GB"
                if self.device == "cuda" else "CPU")
        return (f"{self.backend} | {self.model_params / 1e6:.2f}M params | "
                f"step {self.trained_to_step} | {self.device.upper()}"
                f"{' (' + self.gpu_name + ')' if self.gpu_name else ''} | {vram} | "
                f"ctx {self.block_size} | vocab {self.vocab_size}")


# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------

@dataclass
class ConversationSession:
    """In-memory history for the active session only (never persisted)."""

    turns: List[Dict[str, str]] = field(default_factory=list)  # {role, content}
    max_turns: int = 20

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        # Trim oldest turns to bound context length.
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def clear(self) -> None:
        self.turns.clear()

    def as_prompt(self) -> str:
        """Flatten history into the prompt the model was trained on.

        Uses the SAME serialization as SFT training
        (dipai.conversations.format_dialog): "User: ...\\n\\nAssistant: ...".
        When the last turn is the user's - the normal case when called from
        generate() - the trailing "Assistant: " header is appended so the
        model continues as the assistant, exactly the pattern supervised
        during fine-tuning. Roles are therefore part of the model input, not
        just UI metadata.
        """
        if not self.turns:
            return ""
        text = format_dialog(self.turns)
        if self.turns[-1]["role"] == "user":
            text += "\n\nAssistant: "
        return text

    def __len__(self) -> int:
        return len(self.turns)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class _EventStop:
    """transformers StoppingCriteria that trips when stop_event() is true."""

    def __init__(self, event: Callable[[], bool]):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
        try:
            return bool(self.event())
        except Exception:  # noqa: BLE001
            return False


class DIPAIEngine:
    """Loads one model and serves generation requests for chat + webui.

    Backends:
      * "scratch" - DipaiGPT checkpoint (raw next-token LM; plain-text prompts)
      * "hf"      - PEFT adapter on a pretrained HF model (LoRA/QLoRA; uses
                    the model's chat template for proper multi-turn prompts)
    """

    def __init__(self):
        self.model: Optional[DipaiGPT] = None
        self.tokenizer: Optional[DipaiTokenizer] = None
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self.checkpoint_path: str = ""
        self.trained_to_step: int = 0
        self.hf_pipeline = None          # set when backend == 'hf'
        self.hf_tokenizer = None
        self.base_model_name: str = ""
        self.backend: str = "scratch"
        self._status = EngineStatus()

    # ---- loading ---------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str | Path,
        device: str = "auto",
        dtype_pref: str = "auto",
    ) -> "DIPAIEngine":
        """Load the newest checkpoint in a directory and prepare for generation.

        Tokenizer resolution order:
          1. tokenizer.model copied inside the checkpoint dir (self-describing)
          2. tokenizer path stored in the snapshot config
        """
        eng = cls()
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            raise EngineError(f"checkpoint directory not found: {ckpt_dir}")

        path = latest_checkpoint(ckpt_dir)
        if path is None:
            raise EngineError(
                f"no checkpoint in {ckpt_dir}. Train a model first "
                "(python -m dipai.train --config configs/scratch.yaml)."
            )

        eng.device = get_device(device)
        eng.dtype = resolve_dtype(eng.device, dtype_pref)

        try:
            payload = load_checkpoint(path, map_location="cpu")
        except CheckpointError as exc:
            raise EngineError(str(exc)) from exc

        model_cfg_dict = payload.get("model_config")
        if not model_cfg_dict:
            raise EngineError(
                f"checkpoint {path.name} has no model_config; cannot rebuild "
                "the architecture. Use a DIPAI-produced checkpoint."
            )
        from dipai.config import ModelConfig

        model_cfg = ModelConfig(**model_cfg_dict)
        model = DipaiGPT(model_cfg)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if missing:
            raise EngineError(
                f"checkpoint {path.name} is incomplete; missing keys: {missing[:8]}"
            )
        if unexpected:
            log.warning("extra keys in checkpoint ignored: %s", unexpected[:8])
        model.to(eng.device)
        model.eval()
        eng.model = model
        eng.checkpoint_path = str(path)
        eng.trained_to_step = int(payload.get("step", 0))

        # Tokenizer: prefer the copy next to the checkpoint.
        tok: Optional[DipaiTokenizer] = None
        for cand in (ckpt_dir / "tokenizer.model",):
            if cand.exists():
                try:
                    tok = DipaiTokenizer.load(cand)
                    break
                except TokenizerError as exc:
                    log.warning("checkpoint tokenizer unusable (%s); trying config path", exc)
        if tok is None:
            cfg = payload.get("config") or {}
            cfg_tok = (cfg.get("train") or {}).get("tokenizer_path")
            if cfg_tok and Path(cfg_tok).exists():
                tok = DipaiTokenizer.load(cfg_tok)
        if tok is None:
            raise EngineError(
                "tokenizer not found: neither checkpoint-dir tokenizer.model nor "
                "the config path exists. Re-run scripts/prepare_data.py and train."
            )
        if tok.vocab_size != model_cfg.vocab_size:
            raise EngineError(
                f"tokenizer vocab ({tok.vocab_size}) does not match model vocab "
                f"({model_cfg.vocab_size}). The checkpoint and tokenizer are from "
                "different runs - use the tokenizer stored with the checkpoint."
            )
        eng.tokenizer = tok
        eng.backend = "scratch"

        hw = probe_hardware(check_bnb=False)
        eng._status = EngineStatus(
            backend="scratch",
            checkpoint=str(path),
            model_params=model.num_params,
            trained_to_step=eng.trained_to_step,
            device=eng.device,
            gpu_name=hw.device_name or "",
            vram_total_gb=hw.vram_total_gb,
            vram_free_gb=hw.vram_free_gb,
            vocab_size=tok.vocab_size,
            block_size=model_cfg.block_size,
            ready=True,
            notes="scratch-trained LM: continues text, not an instruction assistant",
        )
        log.info("engine ready: %s", eng._status.line())
        return eng

    @classmethod
    def from_adapter(
        cls,
        adapter_dir: str | Path,
        device: str = "auto",
    ) -> "DIPAIEngine":
        """Load a PEFT adapter produced by dipai.finetune (Mode B).

        Loads the base model in 4-bit (CUDA) or full precision (CPU fallback),
        attaches the adapter, and serves chat-template prompts so multi-turn
        dialogue is formatted the way the instruct model expects.
        """
        adapter_dir = Path(adapter_dir)
        if not adapter_dir.exists():
            raise EngineError(f"adapter directory not found: {adapter_dir}")
        if not (adapter_dir / "adapter_config.json").exists():
            raise EngineError(
                f"{adapter_dir} has no adapter_config.json - not a PEFT adapter "
                "directory. Train one with: python -m dipai.finetune --train"
            )

        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        eng = cls()
        eng.device = get_device(device)

        peft_cfg = PeftConfig.from_pretrained(str(adapter_dir))
        base = peft_cfg.base_model_name_or_path
        eng.base_model_name = base

        tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        use_4bit = eng.device == "cuda"
        quant = None
        if use_4bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            log.warning(
                "CUDA not available - loading adapter base in full precision on CPU "
                "(slow; 4-bit QLoRA inference needs a CUDA GPU)."
            )

        log.info("loading base model %s (4bit=%s) ...", base, use_4bit)
        model = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=quant,
            torch_dtype=torch.float16 if use_4bit else torch.float32,
            device_map="auto" if use_4bit else None,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()

        eng.hf_pipeline = model
        eng.hf_tokenizer = tok
        eng.backend = "hf"
        eng.checkpoint_path = str(adapter_dir)

        # Best-effort step count from a Trainer checkpoint inside output_dir.
        trained_to = 0
        for state in sorted(adapter_dir.parent.glob("checkpoint-*/trainer_state.json")):
            try:
                import json as _json

                trained_to = max(trained_to, int(_json.loads(
                    state.read_text(encoding="utf-8")).get("global_step", 0)))
            except Exception:  # noqa: BLE001
                pass

        hw = probe_hardware(check_bnb=False)
        n_params = int(getattr(model, "num_parameters", lambda: 0)())
        try:
            max_ctx = int(getattr(model.config, "max_position_embeddings", 0) or 2048)
        except Exception:  # noqa: BLE001
            max_ctx = 2048
        eng._status = EngineStatus(
            backend="hf-qlora" if use_4bit else "hf-lora",
            checkpoint=str(adapter_dir),
            model_params=n_params,
            trained_to_step=trained_to,
            device=eng.device,
            gpu_name=hw.device_name or "",
            vram_total_gb=hw.vram_total_gb,
            vram_free_gb=hw.vram_free_gb,
            vocab_size=len(tok),
            block_size=max_ctx,
            ready=True,
            notes=f"PEFT adapter on {base} (chat template: "
                  f"{'yes' if tok.chat_template else 'no'})",
        )
        log.info("engine ready: %s", eng._status.line())
        return eng

    @classmethod
    def from_model(
        cls,
        model_dir: str | Path,
        device: str = "auto",
    ) -> "DIPAIEngine":
        """Load a full-parameter Hugging Face model saved by Mode B.

        `model_dir` is a transformers save dir (config.json + weights +
        tokenizer files), e.g. checkpoints/finetune/model produced with
        use_lora: false. Serves chat-template prompts like from_adapter().
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise EngineError(f"model directory not found: {model_dir}")
        if not (model_dir / "config.json").exists():
            raise EngineError(
                f"{model_dir} has no config.json - not a saved HF model dir. "
                "Train one with: python -m dipai.finetune --config ... --train "
                "(use_lora: false)"
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        eng = cls()
        eng.device = get_device(device)

        tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        use_4bit = eng.device == "cuda"
        quant = None
        if use_4bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            log.warning(
                "CUDA not available - loading model in full precision on CPU (slow)."
            )

        log.info("loading full model %s (4bit=%s) ...", model_dir, use_4bit)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            quantization_config=quant,
            torch_dtype=torch.float16 if use_4bit else torch.float32,
            device_map="auto" if use_4bit else None,
        )
        model.eval()

        eng.hf_pipeline = model
        eng.hf_tokenizer = tok
        eng.backend = "hf"
        eng.checkpoint_path = str(model_dir)

        trained_to = 0
        for state in sorted(model_dir.parent.glob("checkpoint-*/trainer_state.json")):
            try:
                import json as _json

                trained_to = max(trained_to, int(_json.loads(
                    state.read_text(encoding="utf-8")).get("global_step", 0)))
            except Exception:  # noqa: BLE001
                pass

        hw = probe_hardware(check_bnb=False)
        n_params = int(getattr(model, "num_parameters", lambda: 0)())
        try:
            max_ctx = int(getattr(model.config, "max_position_embeddings", 0) or 2048)
        except Exception:  # noqa: BLE001
            max_ctx = 2048
        eng._status = EngineStatus(
            backend="hf-full",
            checkpoint=str(model_dir),
            model_params=n_params,
            trained_to_step=trained_to,
            device=eng.device,
            gpu_name=hw.device_name or "",
            vram_total_gb=hw.vram_total_gb,
            vram_free_gb=hw.vram_free_gb,
            vocab_size=len(tok),
            block_size=max_ctx,
            ready=True,
            notes=f"full model (chat template: "
                  f"{'yes' if tok.chat_template else 'no'})",
        )
        log.info("engine ready: %s", eng._status.line())
        return eng

    @classmethod
    def from_named(
        cls,
        name_or_path: str,
        device: str = "auto",
    ) -> "DIPAIEngine":
        """Load by registry name, dispatching on the entry's backend.

        `scratch` -> from_checkpoint, `hf` -> from_adapter,
        `hf-full` -> from_model. An unknown value is treated as a raw
        HF-model directory (legacy `--model <dir>` behaviour), so paths keep
        working unchanged.
        """
        from dipai.model_registry import ModelRegistryError, resolve_for_loading

        try:
            backend, path = resolve_for_loading(name_or_path)
        except ModelRegistryError as exc:
            raise EngineError(str(exc)) from None
        log.info("resolving %r -> backend=%s path=%s", name_or_path, backend, path)
        if backend == "scratch":
            return cls.from_checkpoint(path, device=device)
        if backend == "hf":
            return cls.from_adapter(path, device=device)
        return cls.from_model(path, device=device)

    # ---- generation ------------------------------------------------------

    def generate(
        self,
        prompt: str,
        settings: Optional[GenerationSettings] = None,
        session: Optional[ConversationSession] = None,
        stream_fn: Optional[Callable[[str], None]] = None,
        stop_event: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Generate a continuation for `prompt` (or the session context).

        When `session` is provided, the prompt is flattened with history and
        the new exchange is appended to the session on success.
        """
        if not prompt or not prompt.strip():
            raise GenerationError("prompt is empty")

        settings = settings or GenerationSettings()

        # Dispatch before scratch-field guards: an HF/adapter engine keeps its
        # model in hf_pipeline and may legitimately have self.model is None.
        if self.backend == "hf":
            return self._generate_hf(
                prompt, settings, session=session,
                stream_fn=stream_fn, stop_event=stop_event,
            )

        if self.model is None or self.tokenizer is None:
            raise EngineError("engine not loaded - call from_checkpoint() first")

        if session is not None:
            session.add("user", prompt.strip())
            full_prompt = session.as_prompt()
            response = generate_text(
                self.model, self.tokenizer, full_prompt, settings,
                device=self.device, stream_fn=stream_fn, stop_event=stop_event,
            )
            session.add("assistant", response.strip())
            return response.strip()

        response = generate_text(
            self.model, self.tokenizer, prompt, settings,
            device=self.device, stream_fn=stream_fn, stop_event=stop_event,
        )
        return response.strip()

    def _generate_hf(
        self,
        prompt: str,
        settings: GenerationSettings,
        session: Optional[ConversationSession] = None,
        stream_fn: Optional[Callable[[str], None]] = None,
        stop_event: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Generate with a PEFT/HF model using its chat template."""
        from transformers import StoppingCriteriaList, TextIteratorStreamer

        if self.hf_pipeline is None or self.hf_tokenizer is None:
            raise EngineError("hf engine not loaded - call from_adapter() first")
        if not prompt or not prompt.strip():
            raise GenerationError("prompt is empty")

        tok = self.hf_tokenizer
        prompt = prompt.strip()

        # Build the message list (proper multi-turn formatting for instruct
        # models; falls back to plain text if the tokenizer has no template).
        messages: List[Dict[str, str]] = []
        if session is not None:
            messages.extend(dict(t) for t in session.turns)
        messages.append({"role": "user", "content": prompt})

        if tok.chat_template:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        inputs = tok(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        n_prompt = inputs["input_ids"].shape[1]

        gen_kwargs: Dict = {
            "max_new_tokens": int(settings.max_new_tokens),
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": tok.eos_token_id,
        }
        sample = settings.temperature > 0.0
        gen_kwargs["do_sample"] = sample
        if sample:
            gen_kwargs["temperature"] = float(settings.temperature)
            if settings.top_k and settings.top_k > 0:
                gen_kwargs["top_k"] = int(settings.top_k)
            if 0.0 < settings.top_p < 1.0:
                gen_kwargs["top_p"] = float(settings.top_p)
        if settings.repetition_penalty and settings.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = float(settings.repetition_penalty)

        def _record(full_text: str) -> None:
            if session is not None:
                session.add("user", prompt)
                session.add("assistant", full_text.strip())

        # Non-streaming fast path.
        if stream_fn is None and stop_event is None:
            with torch.no_grad():
                out = self.hf_pipeline.generate(**inputs, **gen_kwargs)
            new_ids = out[0, n_prompt:]
            text_out = tok.decode(new_ids, skip_special_tokens=True).strip()
            _record(text_out)
            return text_out

        # Streaming path: generate in a worker thread, consume chunks here.
        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True,
            timeout=60.0,
        )
        stopping = None
        if stop_event is not None:
            stopping = StoppingCriteriaList([_EventStop(stop_event)])

        def _worker() -> None:
            try:
                with torch.no_grad():
                    self.hf_pipeline.generate(
                        **inputs, streamer=streamer,
                        stopping_criteria=stopping, **gen_kwargs,
                    )
            except Exception as exc:  # noqa: BLE001 - surfaced via streamer end
                log.error("hf generation failed: %s", exc)
                streamer.end()

        th = threading.Thread(target=_worker, daemon=True)
        th.start()

        acc = ""
        try:
            for chunk in streamer:
                acc += chunk
                if stream_fn is not None:
                    stream_fn(acc)
        finally:
            th.join(timeout=max(30.0, float(settings.max_new_tokens)))
        acc = acc.strip()
        _record(acc)
        return acc

    # ---- status ----------------------------------------------------------

    def status(self) -> EngineStatus:
        # Refresh free VRAM on read (it moves between calls).
        if self.device == "cuda":
            hw = probe_hardware(check_bnb=False)
            self._status.vram_free_gb = hw.vram_free_gb
        return self._status

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None


def load_engine(ckpt_dir: str | Path, device: str = "auto") -> DIPAIEngine:
    """Convenience wrapper used by chat.py and webui/app.py."""
    return DIPAIEngine.from_checkpoint(ckpt_dir, device=device)
