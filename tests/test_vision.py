"""Tests for dipai.vision - image validation, load/inference paths, CLI/UI.

Ordinary tests MOCK the model artifacts (_load_artifacts) so nothing is ever
downloaded. One env-gated GPU smoke test loads the real VLM when explicitly
requested via DIPAI_VISION_SMOKE=1.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dipai import vision
from dipai.config import VisionConfig, load_vision_config
from dipai.vision import DEFAULT_DESCRIBE_PROMPT, VisionError, VisionStatus

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_vision_state():
    """Isolate module state between tests (it is a process-wide singleton)."""
    vision.unload_vision_model()
    yield
    vision.unload_vision_model()


@pytest.fixture()
def cfg() -> VisionConfig:
    """Valid defaults pinned to CPU so assertions are machine-independent."""
    return VisionConfig(device="cpu", local_files_only=True)


def _make_image(
    path: Path, fmt: str = "PNG", size=(32, 32), mode: str = "RGB"
) -> Path:
    from PIL import Image

    colors = {
        "RGB": (255, 0, 0),
        "RGBA": (255, 0, 0, 128),
        "L": 128,
        "P": 1,
    }
    img = Image.new(mode, size, colors.get(mode, 0))
    img.save(path, fmt)
    return path


class FakeProcessor:
    """Stands in for AutoProcessor (no network, no weights)."""

    def __init__(self, decode_text: str = "a red square on a white background"):
        self.decode_text = decode_text
        self.seen_images = []
        self.last_messages = None
        self.last_text = None
        self.tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=0)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        return "<prompt>"

    def __call__(self, images=None, text=None, return_tensors=None, **kwargs):
        self.seen_images.append(images)
        self.last_text = text
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids, skip_special_tokens=True):
        return self.decode_text


class FakeModel:
    """Stands in for AutoModelForImageTextToText."""

    def __init__(self, n_new: int = 4):
        self.n_new = n_new
        self.last_kwargs = None
        self.target_device = None
        self.grad_enabled_during_generate = True

    def to(self, device):
        self.target_device = str(device)
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([])

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        self.grad_enabled_during_generate = torch.is_grad_enabled()
        return torch.tensor([[1, 2, 3] + [9] * self.n_new])


@pytest.fixture()
def fake_vision(monkeypatch):
    """Patch the single transformers seam; returns (proc, model, calls)."""
    calls = {"load": 0}

    def _fake_load(cfg, device, dtype):
        calls["load"] += 1
        proc, model = FakeProcessor(), FakeModel()
        return proc, model, {"params": 123}

    monkeypatch.setattr(vision, "_load_artifacts", _fake_load)
    return SimpleNamespace(proc_holder=[], model_holder=[], calls=calls)


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_valid_formats_accepted(cfg, tmp_path, fmt):
    path = _make_image(tmp_path / f"img.{fmt.lower()}", fmt=fmt)
    img = vision.load_image(path, cfg)
    assert img.mode == "RGB"
    assert img.size == (32, 32)


def test_invalid_file_rejected(cfg, tmp_path):
    bad = tmp_path / "fake.jpg"
    bad.write_bytes(b"this is not an image at all")
    with pytest.raises(VisionError, match="not a recognized image"):
        vision.load_image(bad, cfg)


def test_empty_file_rejected(cfg, tmp_path):
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    with pytest.raises(VisionError, match="empty"):
        vision.load_image(empty, cfg)


def test_corrupt_truncated_image_rejected(cfg, tmp_path):
    path = tmp_path / "trunc.jpg"
    _make_image(path, fmt="JPEG", size=(128, 128))
    data = path.read_bytes()
    path.write_bytes(data[: max(64, len(data) // 10)])
    with pytest.raises(VisionError, match="corrupt|truncated|recognized"):
        vision.load_image(path, cfg)


def test_missing_file_rejected(cfg, tmp_path):
    with pytest.raises(VisionError, match="not found"):
        vision.load_image(tmp_path / "nope.png", cfg)


def test_oversized_upload_rejected(tmp_path):
    small = VisionConfig(device="cpu", max_upload_mb=0.01)  # ~10 KB limit
    big = tmp_path / "big.jpg"
    big.write_bytes(os.urandom(200_000))  # size check fires before decoding
    with pytest.raises(VisionError, match="upload limit"):
        vision.load_image(big, small)


def test_oversized_bytes_input_rejected():
    small = VisionConfig(device="cpu", max_upload_mb=0.01)
    with pytest.raises(VisionError, match="upload limit"):
        vision.load_image(os.urandom(200_000), small)


def test_oversized_dimensions_rejected(cfg, tmp_path):
    tight = VisionConfig(device="cpu", max_image_side=64)
    path = _make_image(tmp_path / "big.png", size=(128, 128))
    with pytest.raises(VisionError, match="per-side limit"):
        vision.load_image(path, tight)


def test_too_many_pixels_rejected(cfg, tmp_path):
    tight = VisionConfig(device="cpu", max_image_pixels=1000)
    path = _make_image(tmp_path / "pix.png", size=(64, 64))  # 4096 px
    with pytest.raises(VisionError, match="pixel limit"):
        vision.load_image(path, tight)


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_rgb_conversion(cfg, tmp_path, mode):
    path = _make_image(tmp_path / f"{mode}.png", mode=mode)
    img = vision.load_image(path, cfg)
    assert img.mode == "RGB"


def test_gif_format_rejected(cfg, tmp_path):
    path = tmp_path / "anim.gif"
    _make_image(path, fmt="GIF")
    with pytest.raises(VisionError, match="unsupported image format 'GIF'"):
        vision.load_image(path, cfg)


def test_remote_url_rejected(cfg):
    # No arbitrary remote fetching: a URL is just a missing local file.
    with pytest.raises(VisionError, match="not found"):
        vision.load_image("https://example.com/photo.jpg", cfg)


def test_empty_bytes_rejected(cfg):
    with pytest.raises(VisionError, match="empty"):
        vision.load_image(b"", cfg)


def test_unsupported_input_type_rejected(cfg):
    with pytest.raises(VisionError, match="unsupported image input"):
        vision.load_image(12345, cfg)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def test_load_and_status_cpu(fake_vision, cfg):
    status = vision.load_vision_model(cfg)
    assert status.loaded is True
    assert status.device == "cpu"
    assert status.model_id == cfg.model_id
    assert status.params == 123
    assert fake_vision.calls["load"] == 1
    # Idempotent: same config does not reload.
    vision.load_vision_model(cfg)
    assert fake_vision.calls["load"] == 1


def test_missing_model_weights_raises(tmp_path):
    cfg = VisionConfig(
        model_id=str(tmp_path / "nonexistent-model"),
        device="cpu",
        local_files_only=True,  # guarantees no network access
    )
    with pytest.raises(VisionError, match="failed to load vision model"):
        vision.load_vision_model(cfg)
    status = vision.get_vision_status()
    assert status.loaded is False
    assert "failed to load" in status.load_error


def test_load_failure_cached_and_retry_with_force(monkeypatch, cfg):
    calls = {"load": 0}

    def _boom(cfg, device, dtype):
        calls["load"] += 1
        raise VisionError("boom: weights unavailable")

    monkeypatch.setattr(vision, "_load_artifacts", _boom)
    with pytest.raises(VisionError, match="boom"):
        vision.load_vision_model(cfg)
    # Cached failure for the same config: no second attempt.
    with pytest.raises(VisionError, match="boom"):
        vision.load_vision_model(cfg)
    assert calls["load"] == 1
    # force=True retries.
    with pytest.raises(VisionError, match="boom"):
        vision.load_vision_model(cfg, force=True)
    assert calls["load"] == 2


def test_unload_resets_status(fake_vision, cfg):
    vision.load_vision_model(cfg)
    assert vision.get_vision_status().loaded is True
    vision.unload_vision_model()
    st = vision.get_vision_status()
    assert st.loaded is False
    assert st.model_id == "" and st.inferences == 0


def test_status_before_load():
    st = vision.get_vision_status()
    assert isinstance(st, VisionStatus)
    assert st.loaded is False
    assert "not loaded" in st.line()
    assert st.to_dict()["loaded"] is False


def test_default_config_from_yaml():
    cfg = load_vision_config("configs/vision.yaml", required=False)
    assert cfg.model_id.startswith("HuggingFaceTB/SmolVLM")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def test_answer_returns_real_model_output(fake_vision, cfg, tmp_path):
    vision.load_vision_model(cfg)
    path = _make_image(tmp_path / "photo.png")
    answer = vision.answer_about_image(path, "What is in this picture?")
    assert isinstance(answer, str)
    assert answer == "a red square on a white background"  # from the fake model
    assert vision.get_vision_status().inferences == 1


def test_describe_uses_default_prompt(fake_vision, cfg, tmp_path):
    vision.load_vision_model(cfg)
    proc = FakeProcessor()
    model = FakeModel()
    import dipai.vision as v

    v._STATE["processor"] = proc
    v._STATE["model"] = model
    path = _make_image(tmp_path / "d.png")
    vision.describe_image(path)
    assert proc.last_messages[0]["content"][1]["text"] == DEFAULT_DESCRIBE_PROMPT


def test_empty_question_rejected(fake_vision, cfg, tmp_path):
    vision.load_vision_model(cfg)
    path = _make_image(tmp_path / "q.png")
    with pytest.raises(VisionError, match="question is empty"):
        vision.answer_about_image(path, "   ")


def test_empty_model_output_raises(fake_vision, cfg, tmp_path):
    v = vision
    v._STATE["processor"] = FakeProcessor(decode_text="")
    v._STATE["model"] = FakeModel()
    v._STATE["config"] = cfg
    v._STATE["device"] = "cpu"
    path = _make_image(tmp_path / "empty-out.png")
    with pytest.raises(VisionError, match="no output"):
        v.answer_about_image(path, "what is this?")


def test_generation_is_bounded_and_configured(fake_vision, cfg, tmp_path):
    cfg = VisionConfig(device="cpu", max_new_tokens=17, temperature=0.0)
    vision.load_vision_model(cfg)
    path = _make_image(tmp_path / "gen.png")
    vision.answer_about_image(path, "question?")
    model = vision._STATE["model"]
    assert model.last_kwargs["max_new_tokens"] == 17
    assert model.last_kwargs["do_sample"] is False
    assert "temperature" not in model.last_kwargs  # greedy: no sampling args
    assert model.grad_enabled_during_generate is False  # inference_mode


def test_gradients_disabled_and_sampling_enabled(fake_vision, cfg, tmp_path):
    cfg = VisionConfig(device="cpu", temperature=0.7, top_p=0.9, max_new_tokens=8)
    vision.load_vision_model(cfg)
    vision.answer_about_image(_make_image(tmp_path / "s.png"), "q?")
    model = vision._STATE["model"]
    assert model.last_kwargs["do_sample"] is True
    assert model.last_kwargs["temperature"] == 0.7
    assert model.last_kwargs["top_p"] == 0.9
    assert model.grad_enabled_during_generate is False


def test_rgb_images_reach_the_model(fake_vision, cfg, tmp_path):
    vision.load_vision_model(cfg)
    path = _make_image(tmp_path / "rgba.png", mode="RGBA")
    vision.answer_about_image(path, "what mode?")
    seen = vision._STATE["processor"].seen_images[-1]
    assert seen.mode == "RGB"


# ---------------------------------------------------------------------------
# Text-only isolation (vision failure must not break text inference)
# ---------------------------------------------------------------------------

def test_text_generation_survives_vision_load_failure(
    monkeypatch, cfg, tiny_model, tiny_tokenizer
):
    def _boom(cfg, device, dtype):
        raise VisionError("vision weights unavailable")

    monkeypatch.setattr(vision, "_load_artifacts", _boom)
    with pytest.raises(VisionError):
        vision.load_vision_model(cfg)

    # Text-only path keeps working afterwards.
    from dipai.generate import GenerationSettings, generate_text

    out = generate_text(
        tiny_model,
        tiny_tokenizer,
        "hello",
        GenerationSettings(max_new_tokens=5, temperature=0.0),
    )
    assert isinstance(out, str)


def test_import_vision_does_not_load_model():
    # Importing the module must leave the model unloaded (lazy by design).
    assert vision.get_vision_status().loaded is False


# ---------------------------------------------------------------------------
# Web UI handlers (upload + error states, no Gradio app construction)
# ---------------------------------------------------------------------------

class _StubVision:
    """Minimal stand-in for the webui's `vision` module reference."""

    def __init__(self, answer="stub answer", error=None):
        self.answer = answer
        self.error = error
        self.loaded = False

    def load_vision_model(self, cfg, **kwargs):
        if self.error:
            raise self.error
        self.loaded = True

    def answer_about_image(self, image, question):
        if self.error:
            raise self.error
        return self.answer

    def get_vision_status(self):
        return VisionStatus(loaded=self.loaded, model_id="stub/model")

    def describe_image(self, image, prompt=None):
        return self.answer


@pytest.fixture()
def webui_app(monkeypatch):
    from webui import app

    return app


def test_webui_ask_without_image(webui_app, monkeypatch):
    stub = _StubVision()
    monkeypatch.setattr(webui_app, "vision", stub)
    answer, _status = webui_app.vision_ask(None, "what is this?")
    assert answer.startswith("[error: upload an image first]")
    assert stub.loaded is False  # nothing loaded for an empty request


def test_webui_ask_without_question(webui_app, monkeypatch, tmp_path):
    stub = _StubVision()
    monkeypatch.setattr(webui_app, "vision", stub)
    path = str(_make_image(tmp_path / "u.png"))
    answer, _ = webui_app.vision_ask(path, "   ")
    assert answer.startswith("[error: enter a question")


def test_webui_ask_model_failure_shown_not_crashed(webui_app, monkeypatch, tmp_path):
    stub = _StubVision(error=VisionError("cuda out of memory (test)"))
    monkeypatch.setattr(webui_app, "vision", stub)
    path = str(_make_image(tmp_path / "f.png"))
    answer, _ = webui_app.vision_ask(path, "what?")
    assert "[error:" in answer and "cuda out of memory" in answer


def test_webui_ask_success_returns_model_answer(webui_app, monkeypatch, tmp_path):
    stub = _StubVision(answer="the real model answer")
    monkeypatch.setattr(webui_app, "vision", stub)
    path = str(_make_image(tmp_path / "ok.png"))
    answer, status = webui_app.vision_ask(path, "describe this")
    assert answer == "the real model answer"
    assert stub.loaded is True
    assert "vision" in status


def test_webui_stream_shows_loading_first(webui_app, monkeypatch, tmp_path):
    stub = _StubVision(answer="final answer")
    monkeypatch.setattr(webui_app, "vision", stub)
    path = str(_make_image(tmp_path / "st.png"))
    gen = webui_app.vision_ask_stream(path, "what?")
    first_answer, first_status = next(gen)
    assert first_answer == ""
    assert "processing image" in first_status
    final_answer, _ = next(gen)
    assert final_answer == "final answer"


def test_webui_clear_resets_fields(webui_app):
    img, q, a, status = webui_app.vision_clear()
    assert img is None and q == "" and a == ""
    assert "vision" in status


# ---------------------------------------------------------------------------
# GPU smoke test with the REAL model (opt-in: downloads ~513 MB once)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required"
)
@pytest.mark.skipif(
    os.environ.get("DIPAI_VISION_SMOKE") != "1",
    reason="set DIPAI_VISION_SMOKE=1 to download and run the real VLM",
)
def test_gpu_smoke_real_model(tmp_path):
    cfg = VisionConfig(device="cuda", max_new_tokens=16, temperature=0.0)
    status = vision.load_vision_model(cfg, force=True)
    assert status.loaded and status.device == "cuda"
    torch.cuda.reset_peak_memory_stats()
    path = _make_image(tmp_path / "gpu.png", size=(128, 128))
    out = vision.describe_image(path)
    assert isinstance(out, str) and out.strip()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    # Sanity: the 256M model in fp16 must not exceed the 4 GB card.
    assert peak_mb < 4000, f"unexpected VRAM use: {peak_mb:.0f} MB"
