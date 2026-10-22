"""Tests for chat.py/webui wiring: --list-models and backend-aware loading."""

from __future__ import annotations

import pytest

from dipai.chat import build_arg_parser


def test_list_models_flag_parses():
    args = build_arg_parser().parse_args(["--list-models"])
    assert args.list_models is True


def test_list_models_cli_runs(capsys):
    from dipai.chat import main

    rc = main(["--list-models"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "scratch" in out
    assert "checkpoints/scratch" in out


def test_from_named_dispatches_by_backend(monkeypatch):
    """A registry name must route to ITS backend's loader, not always from_model."""
    from dipai.inference import DIPAIEngine

    calls = []
    monkeypatch.setattr(
        DIPAIEngine, "from_checkpoint",
        classmethod(lambda cls, p, device="auto": calls.append(("scratch", p)) or "CKPT")
    )
    monkeypatch.setattr(
        DIPAIEngine, "from_adapter",
        classmethod(lambda cls, p, device="auto": calls.append(("hf", p)) or "ADAPTER")
    )
    monkeypatch.setattr(
        DIPAIEngine, "from_model",
        classmethod(lambda cls, p, device="auto": calls.append(("hf-full", p)) or "MODEL")
    )

    # the built-in "scratch" entry -> from_checkpoint with its path
    out = DIPAIEngine.from_named("scratch", device="cpu")
    assert out == "CKPT"
    assert calls[-1] == ("scratch", "checkpoints/scratch")

    # a raw path (not a registry name) -> legacy from_model behaviour
    out = DIPAIEngine.from_named("/some/hf/dir", device="cpu")
    assert out == "MODEL"
    assert calls[-1] == ("hf-full", "/some/hf/dir")


def test_from_named_dispatches_hf_and_hf_full(monkeypatch):
    from dipai.inference import DIPAIEngine

    calls = []
    monkeypatch.setattr(
        DIPAIEngine, "from_checkpoint",
        classmethod(lambda cls, p, device="auto": calls.append(("scratch", p)) or None)
    )
    monkeypatch.setattr(
        DIPAIEngine, "from_adapter",
        classmethod(lambda cls, p, device="auto": calls.append(("hf", p)) or None)
    )
    monkeypatch.setattr(
        DIPAIEngine, "from_model",
        classmethod(lambda cls, p, device="auto": calls.append(("hf-full", p)) or None)
    )

    # from_named resolves through dipai.model_registry at call time, so
    # patching that module's resolver is enough to inject custom entries
    import dipai.model_registry as mr

    monkeypatch.setattr(
        mr, "resolve_for_loading",
        lambda name: {"myadapter": ("hf", "/a"), "mymodel": ("hf-full", "/m")}[name],
    )
    DIPAIEngine.from_named("myadapter")
    DIPAIEngine.from_named("mymodel")
    assert calls == [("hf", "/a"), ("hf-full", "/m")]


def test_from_named_empty_raises_engine_error():
    from dipai.inference import DIPAIEngine, EngineError

    with pytest.raises(EngineError):
        DIPAIEngine.from_named("")


def test_webui_list_models_parses():
    """webui exposes the same --list-models surface as chat."""
    import webui.app as app_mod

    # parse_args lives inside main(); just confirm the module imports and
    # exposes the entry points the CLI needs
    assert hasattr(app_mod, "main")
    assert hasattr(app_mod, "create_app")
