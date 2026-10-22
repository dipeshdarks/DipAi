"""DIPAI vision - image understanding with a small pretrained vision-language
model (VLM).

This module is deliberately SEPARATE from the scratch text model:
  * It loads its own pretrained VLM (default: HuggingFaceTB/SmolVLM-256M-Instruct,
    Apache-2.0) with the model's official processor / image preprocessing.
  * The DIPAI scratch checkpoint, tokenizer, and vocabulary are NEVER touched.
  * Nothing is loaded until load_vision_model() or the first image request,
    so text-only chat/training works unchanged - even without Pillow or the
    VLM weights installed (every heavy import is lazy).

Run (vision CLI):
    python -m dipai.vision --image photo.jpg --prompt "What is in this image?"
    # or through the chat CLI (text REPL stays the default):
    python -m dipai.chat --image photo.jpg --prompt "Describe this image"

Interfaces:
    load_vision_model(config=None, *, force=False) -> VisionStatus
    describe_image(image, prompt=None) -> str
    answer_about_image(image, question) -> str
    get_vision_status() -> VisionStatus
    unload_vision_model() -> None
    load_image(source, config=None) -> PIL.Image.Image   # validation helper

Privacy: images and prompts are processed locally. They are never uploaded,
never sent to any external service, and are never written to logs (only
sizes/devices/status are logged). Hugging Face access happens solely to
download model WEIGHTS (and can be disabled with local_files_only: true).
"""

from __future__ import annotations

import argparse
import io
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from dipai.config import (
    ConfigError,
    VisionConfig,
    default_vision_config,
    load_vision_config,
)
from dipai.hardware import get_device, resolve_dtype

log = logging.getLogger("dipai.vision")

# Decoded formats accepted (content-based, not filename-based).
SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP")
DEFAULT_CONFIG_PATH = "configs/vision.yaml"
DEFAULT_DESCRIBE_PROMPT = "Describe this image in detail."


class VisionError(Exception):
    """Raised for invalid image input, model load failures, or inference errors."""


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@dataclass
class VisionStatus:
    """Snapshot of the vision module state (shown in chat/web UI)."""

    loaded: bool = False
    model_id: str = ""
    device: str = ""
    dtype: str = ""
    params: int = 0
    load_seconds: float = 0.0
    inferences: int = 0
    load_error: str = ""
    last_error: str = ""
    notes: str = ""

    def line(self) -> str:
        if not self.loaded:
            base = "vision: not loaded"
            if self.load_error:
                return f"{base} (error: {self.load_error})"
            return f"{base} (loads on first image request)"
        name = self.model_id.rsplit("/", 1)[-1]
        out = (f"vision: {name} | {self.params / 1e6:.1f}M | "
               f"{self.device.upper()} {self.dtype}")
        if self.notes:
            out += f" | {self.notes}"
        if self.last_error:
            out += f" | last error: {self.last_error}"
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loaded": self.loaded,
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
            "params_m": round(self.params / 1e6, 2),
            "load_seconds": round(self.load_seconds, 2),
            "inferences": self.inferences,
            "load_error": self.load_error,
            "last_error": self.last_error,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Module state (single model, thread-safe)
# ---------------------------------------------------------------------------

_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {
    "processor": None,
    "model": None,
    "config": None,          # VisionConfig of the successful load
    "device": "",
    "dtype": "",
    "params": 0,
    "load_seconds": 0.0,
    "inferences": 0,
    "load_error": "",        # last load failure (message)
    "error_config": None,    # VisionConfig that produced the failure
    "last_error": "",        # last inference failure (message)
    "notes": "",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _resolve_device(cfg: VisionConfig) -> str:
    try:
        return get_device(cfg.device)
    except RuntimeError as exc:
        raise VisionError(str(exc)) from exc


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _check_disk_space(cfg: VisionConfig) -> None:
    """Refuse a download that cannot fit (checked BEFORE any fetch)."""
    if cfg.local_files_only:
        return
    if Path(cfg.model_id).exists():
        return  # local directory - no download
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.constants import HF_HUB_CACHE

        info = HfApi().model_info(cfg.model_id, files_metadata=True)
        needed = sum(
            s.size or 0
            for s in info.siblings
            if s.rfilename.endswith(".safetensors")
        )
        needed = int(needed * 1.25)  # margin for configs/tokenizer/cache overhead
        cache_dir = Path(HF_HUB_CACHE)
        anchor = cache_dir if cache_dir.exists() else Path.cwd()
        free = shutil.disk_usage(str(anchor)).free
        if needed > free:
            raise VisionError(
                f"not enough disk space for '{cfg.model_id}': needs ~"
                f"{needed / 1e6:.0f} MB (with margin), only {free / 1e6:.0f} MB "
                "free. Free up space or set model_id to a local path."
            )
    except VisionError:
        raise
    except Exception as exc:
        # from_pretrained report the real error instead of blocking here.
        log.debug("disk pre-check skipped (%s)", exc)


def _load_artifacts(
    cfg: VisionConfig, device: str, dtype: torch.dtype
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Download/load (processor, model). Model stays on CPU here; the caller
    moves it to `device` so CUDA-OOM fallback stays in one place.

    This is the only place transformers is imported (test seam: tests mock
    this function so unit tests never download weights).
    """
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise VisionError(
            "transformers is required for image understanding. Install it: "
            "pip install \"transformers>=4.49\" (or: pip install -r requirements.txt)"
        ) from exc

    common = {
        "trust_remote_code": False,
        "local_files_only": cfg.local_files_only,
    }
    try:
        processor = AutoProcessor.from_pretrained(cfg.model_id, **common)
        model_kwargs: Dict[str, Any] = {"torch_dtype": dtype, **common}
        if cfg.low_cpu_mem_usage:
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    cfg.model_id, low_cpu_mem_usage=True, **model_kwargs
                )
            except (ImportError, ValueError) as exc:
                # accelerate missing or rejected -> retry without streaming
                log.warning("low_cpu_mem_usage unavailable (%s); retrying", exc)
                model = AutoModelForImageTextToText.from_pretrained(
                    cfg.model_id, **model_kwargs
                )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                cfg.model_id, **model_kwargs
            )
    except Exception as exc:
        raise VisionError(
            f"failed to load vision model '{cfg.model_id}': {exc}. "
            "Check the model id/path, your network connection, or set "
            "local_files_only: true after caching the weights."
        ) from exc

    meta = {"params": int(sum(p.numel() for p in model.parameters()))}
    return processor, model, meta


def load_vision_model(
    config: Optional[VisionConfig] = None, *, force: bool = False
) -> VisionStatus:
    """Load the vision model (idempotent).

    * Missing configs/vision.yaml -> validated defaults (text-only setups
      keep working without this feature being configured).
    * A previous load failure for the SAME config is re-raised without
      retrying the download; pass force=True to retry (or change the config).
    * On CUDA out-of-memory during load: retries on CPU when
      config.cpu_fallback is true, and reports that in status.notes.
    """
    cfg = config if config is not None else default_vision_config()
    try:
        cfg.validate()
    except ConfigError as exc:
        raise VisionError(str(exc)) from exc

    with _LOCK:
        if not force and _STATE["model"] is not None and _STATE["config"] == cfg:
            return get_vision_status()
        if not force and _STATE["load_error"] and _STATE["error_config"] == cfg:
            raise VisionError(_STATE["load_error"])

        # Reset before attempting.
        _STATE["processor"] = _STATE["model"] = None
        _STATE["load_error"] = ""
        _STATE["error_config"] = None
        _STATE["notes"] = ""

        t0 = time.perf_counter()
        try:
            device = _resolve_device(cfg)
            dtype = resolve_dtype(device, cfg.dtype)
            _check_disk_space(cfg)
            processor, model, meta = _load_artifacts(cfg, device, dtype)
            try:
                model.to(device)
            except Exception as exc:
                if not _is_cuda_oom(exc) or not cfg.cpu_fallback:
                    raise VisionError(
                        f"failed to place vision model on '{device}': {exc}"
                    ) from exc
                log.warning(
                    "CUDA out of memory loading vision model; falling back to CPU: %s",
                    exc,
                )
                _free_cuda()
                device, dtype = "cpu", torch.float32
                model.to(device)
                _STATE["notes"] = "cpu fallback after cuda OOM"
            model.eval()
        except VisionError as exc:
            _STATE["load_error"] = str(exc)
            _STATE["error_config"] = cfg
            _STATE["model"] = _STATE["processor"] = None
            raise
        except Exception as exc:
            msg = f"failed to load vision model '{cfg.model_id}': {exc}"
            _STATE["load_error"] = msg
            _STATE["error_config"] = cfg
            _STATE["model"] = _STATE["processor"] = None
            raise VisionError(msg) from exc

        _STATE.update(
            processor=processor,
            model=model,
            config=cfg,
            device=device,
            dtype=str(dtype).replace("torch.", ""),
            params=int(meta.get("params", 0)),
            load_seconds=time.perf_counter() - t0,
        )
        log.info("vision ready: %s", get_vision_status().line())
        return get_vision_status()


def unload_vision_model() -> None:
    """Release the vision model and all cached state (frees VRAM/RAM)."""
    with _LOCK:
        _STATE["processor"] = _STATE["model"] = None
        _STATE["config"] = _STATE["error_config"] = None
        _STATE["device"] = _STATE["dtype"] = ""
        _STATE["params"] = _STATE["inferences"] = 0
        _STATE["load_seconds"] = 0.0
        _STATE["load_error"] = _STATE["last_error"] = _STATE["notes"] = ""
        import gc

        gc.collect()
        _free_cuda()


def get_vision_status() -> VisionStatus:
    """Report device, model id, load state, counters, and errors."""
    with _LOCK:
        return VisionStatus(
            loaded=_STATE["model"] is not None,
            model_id=str(_STATE["config"].model_id) if _STATE["config"] else "",
            device=_STATE["device"],
            dtype=_STATE["dtype"],
            params=_STATE["params"],
            load_seconds=_STATE["load_seconds"],
            inferences=_STATE["inferences"],
            load_error=_STATE["load_error"],
            last_error=_STATE["last_error"],
            notes=_STATE["notes"],
        )


def _ensure_loaded() -> VisionStatus:
    """Load on demand; re-raise a cached failure for the same config."""
    status = get_vision_status()
    if status.loaded:
        return status
    return load_vision_model()  # raises VisionError on failure (cached after)


def _free_cuda() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: S110, BLE001 - best effort
        pass


# ---------------------------------------------------------------------------
# Image validation (JPEG / PNG / WebP, size + format + corrupt detection)
# ---------------------------------------------------------------------------

def load_image(source: Any, config: Optional[VisionConfig] = None):
    """Decode and validate an image; return a detached RGB PIL image.

    Accepts: local file path (str/Path), raw bytes, or a PIL.Image.
    Rejects: missing/empty files, files above config.max_upload_mb, decoded
    formats other than JPEG/PNG/WebP, corrupt/truncated data, images above
    config.max_image_side per side or config.max_image_pixels total.

    Remote URLs are intentionally NOT supported (no arbitrary network fetch).
    """
    from PIL import Image, UnidentifiedImageError

    cfg = config if config is not None else (
        _STATE["config"] or default_vision_config()
    )
    max_bytes = int(float(cfg.max_upload_mb) * 1024 * 1024)

    owned = True  # close the decoder when we opened it
    if isinstance(source, Image.Image):
        im = source
        owned = False
    elif isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
        if not data:
            raise VisionError("image data is empty")
        if len(data) > max_bytes:
            raise VisionError(
                f"image is {len(data) / 1e6:.1f} MB, above the "
                f"{cfg.max_upload_mb:g} MB upload limit"
            )
        try:
            im = Image.open(io.BytesIO(data))
        except UnidentifiedImageError:
            raise VisionError(
                "not a recognized image file (supported: JPEG, PNG, WebP)"
            ) from None
        except Exception as exc:
            raise VisionError(f"cannot read image: {exc}") from exc
    elif isinstance(source, (str, Path)):
        path = Path(source)
        # str sources that look like URLs are rejected as missing local files
        # (intentionally: no remote image fetching).
        if not path.is_file():
            raise VisionError(f"image file not found: {path}")
        size = path.stat().st_size
        if size == 0:
            raise VisionError(f"image file is empty: {path}")
        if size > max_bytes:
            raise VisionError(
                f"image is {size / 1e6:.1f} MB, above the "
                f"{cfg.max_upload_mb:g} MB upload limit"
            )
        try:
            im = Image.open(path)
        except UnidentifiedImageError:
            raise VisionError(
                f"not a recognized image file: {path} "
                "(supported: JPEG, PNG, WebP)"
            ) from None
        except Exception as exc:
            raise VisionError(f"cannot read image {path}: {exc}") from exc
    else:
        raise VisionError(
            f"unsupported image input type: {type(source).__name__} "
            "(expected file path, bytes, or PIL.Image)"
        )

    try:
        # Content-based format check (a .jpg containing GIF bytes is rejected).
        fmt = (im.format or "").upper()
        if fmt and fmt not in SUPPORTED_FORMATS:
            raise VisionError(
                f"unsupported image format '{fmt or 'unknown'}' - "
                f"supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        # Size checks from the HEADER, before decoding pixels (cheap + safe).
        w, h = im.size
        if w <= 0 or h <= 0:
            raise VisionError(f"invalid image dimensions: {w}x{h}")
        if w > cfg.max_image_side or h > cfg.max_image_side:
            raise VisionError(
                f"image {w}x{h} exceeds the {cfg.max_image_side}px per-side limit"
            )
        if w * h > cfg.max_image_pixels:
            raise VisionError(
                f"image {w}x{h} = {w * h} pixels exceeds the "
                f"{cfg.max_image_pixels} pixel limit"
            )
        try:
            im.load()  # force full decode; raises on corrupt/truncated data
        except Exception as exc:
            raise VisionError(f"image is corrupt or truncated: {exc}") from exc
        rgb = im.convert("RGB")  # safe RGB conversion (palette/CMYK/RGBA/L)
    finally:
        if owned:
            im.close()

    return rgb


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(image, prompt: str) -> str:
    """Run the VLM on an RGB image + text prompt. Returns the real decoded
    output only - never a fabricated description."""
    with _LOCK:
        _ensure_loaded()
        processor = _STATE["processor"]
        model = _STATE["model"]
        device = _STATE["device"]
        cfg: VisionConfig = _STATE["config"]

        messages: List[Dict[str, Any]] = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(images=image, text=prompt_text, return_tensors="pt")
        inputs = {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        n_prompt = int(inputs["input_ids"].shape[1])

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(cfg.max_new_tokens),  # bounded output
            "do_sample": bool(cfg.temperature > 0.0),
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = float(cfg.temperature)
            if 0.0 < cfg.top_p < 1.0:
                gen_kwargs["top_p"] = float(cfg.top_p)
        tok = getattr(processor, "tokenizer", None)
        eos_id = getattr(tok, "eos_token_id", None)
        pad_id = getattr(tok, "pad_token_id", None) or eos_id
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id

        out = None
        try:
            with torch.inference_mode():  # no grads anywhere
                out = model.generate(**inputs, **gen_kwargs)
            new_ids = out[0, n_prompt:]
            text = str(processor.decode(new_ids, skip_special_tokens=True)).strip()
        except Exception as exc:
            _STATE["last_error"] = str(exc)
            _free_cuda()
            if _is_cuda_oom(exc):
                raise VisionError(
                    "vision inference ran out of GPU memory. Lower "
                    "vision.max_new_tokens in configs/vision.yaml, close other "
                    "GPU apps, or set vision.device: cpu (slower)."
                ) from exc
            raise VisionError(f"vision inference failed: {exc}") from exc
        finally:
            # Release temporary tensors/inputs promptly (4 GB VRAM target).
            del inputs
            if out is not None:
                del out
            if device == "cuda":
                _free_cuda()

        if not text:
            _STATE["last_error"] = "model produced no output"
            raise VisionError(
                "the vision model produced no output for this image/prompt "
                "(empty generation - nothing is fabricated)"
            )
        _STATE["inferences"] += 1
        _STATE["last_error"] = ""
        # Deliberately NOT logging the prompt or image content (privacy).
        log.debug("vision inference ok (%d chars)", len(text))
        return text


def answer_about_image(image: Any, question: str) -> str:
    """Answer a question about an image using real VLM inference."""
    if not question or not str(question).strip():
        raise VisionError(
            "question is empty - ask something about the image "
            "(e.g. 'What is in this picture?')"
        )
    img = load_image(image)  # validates format/size/decode before inference
    return _run_inference(img, str(question).strip())


def describe_image(image: Any, prompt: Optional[str] = None) -> str:
    """Describe an image (uses DEFAULT_DESCRIBE_PROMPT when prompt is None)."""
    return answer_about_image(image, prompt or DEFAULT_DESCRIBE_PROMPT)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m dipai.vision",
        description="Ask a question about a local image using the DIPAI vision "
                    "model (runs fully offline once weights are cached).",
    )
    ap.add_argument("--image", "-i", required=True, metavar="PATH",
                    help="path to a JPEG/PNG/WebP image")
    ap.add_argument("--prompt", "-p", default=DEFAULT_DESCRIBE_PROMPT,
                    help=f"question to ask (default: {DEFAULT_DESCRIBE_PROMPT!r})")
    ap.add_argument("--config", "-c", default=DEFAULT_CONFIG_PATH,
                    help=f"vision YAML config (default: {DEFAULT_CONFIG_PATH}; "
                         "falls back to built-in defaults if missing)")
    ap.add_argument("--model", "-m", default=None,
                    help="override vision.model_id (HF id or local path)")
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"],
                    help="override vision.device")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override vision.max_new_tokens")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    from dipai.utils import setup_logging

    log_ = setup_logging()
    args = build_arg_parser().parse_args(argv)

    try:
        cfg = load_vision_config(args.config, required=False)
        if args.model:
            cfg.model_id = args.model
        if args.device:
            cfg.device = args.device
        if args.max_new_tokens is not None:
            cfg.max_new_tokens = args.max_new_tokens
        cfg.validate()
        status = load_vision_model(cfg, force=True)
        log_.info("loaded %s", status.line())
        answer = answer_about_image(args.image, args.prompt)
    except VisionError as exc:
        log_.error("%s", exc)
        print(f"[vision error: {exc}]")
        return 1
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
