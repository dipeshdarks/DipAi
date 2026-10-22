"""Tests for dipai.model_registry: registration, resolution, backends."""

from __future__ import annotations

import json

import pytest

from dipai.model_registry import (
    BACKENDS,
    ModelRegistry,
    ModelRegistryError,
    RegisteredModel,
    default_registry,
    resolve_for_loading,
)


def test_builtin_has_scratch_default():
    r = default_registry()
    assert "scratch" in r.names()
    assert r.default().name == "scratch"
    assert r.default().backend == "scratch"


def test_resolve_empty_name_returns_default():
    r = default_registry()
    assert r.resolve("") is r.default()
    assert r.resolve().name == "scratch"


def test_get_unknown_raises_with_known_list():
    r = default_registry()
    with pytest.raises(ModelRegistryError) as ei:
        r.get("nope")
    assert "scratch" in str(ei.value)


def test_register_custom_and_resolve(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    r = ModelRegistry()
    r.add(RegisteredModel(name="base-50m", backend="scratch", path=str(d),
                          description="tiny", trained_from="pretrain",
                          tags=["base"]))
    assert "base-50m" in r.names()
    m = r.resolve("base-50m")
    assert m.backend == "scratch"
    assert m.exists() is True
    assert r.default().name == "scratch"  # default unchanged


def test_add_overwrites_same_name(tmp_path):
    d = tmp_path / "a"
    d.mkdir()
    r = ModelRegistry()
    r.add(RegisteredModel(name="m", backend="scratch", path=str(d)))
    r.add(RegisteredModel(name="m", backend="hf-full", path=str(d),
                          description="v2"))
    assert r.get("m").backend == "hf-full"
    assert r.get("m").description == "v2"


def test_remove_and_missing(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    r = ModelRegistry()
    r.add(RegisteredModel(name="gone", backend="scratch", path=str(d)))
    assert r.remove("gone") is True
    assert r.remove("gone") is False
    with pytest.raises(ModelRegistryError):
        r.get("gone")


def test_backend_validation():
    r = ModelRegistry()
    with pytest.raises(ModelRegistryError) as ei:
        r.add(RegisteredModel(name="bad", backend="onnx", path="/tmp"))
    assert "onnx" in str(ei.value)


def test_empty_name_rejected():
    r = ModelRegistry()
    with pytest.raises(ModelRegistryError):
        r.add(RegisteredModel(name="", backend="scratch", path="/tmp"))


def test_empty_path_rejected():
    r = ModelRegistry()
    with pytest.raises(ModelRegistryError):
        r.add(RegisteredModel(name="x", backend="scratch", path=""))


def test_available_filters_by_exists(tmp_path):
    exist = tmp_path / "here"
    exist.mkdir()
    r = ModelRegistry()
    r.add(RegisteredModel(name="here", backend="scratch", path=str(exist)))
    r.add(RegisteredModel(name="away", backend="scratch",
                          path=str(tmp_path / "nothere")))
    avail = {m.name for m in r.available()}
    assert "here" in avail
    assert "away" not in avail


def test_all_backends_accepted(tmp_path):
    d = tmp_path / "any"
    d.mkdir()
    r = ModelRegistry()
    for b in BACKENDS:
        r.add(RegisteredModel(name=f"m-{b}", backend=b, path=str(d)))
        assert r.get(f"m-{b}").backend == b


def test_to_dict_includes_exists(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    r = ModelRegistry()
    r.add(RegisteredModel(name="m", backend="scratch", path=str(d)))
    r.add(RegisteredModel(name="missing", backend="scratch",
                          path=str(tmp_path / "nope")))
    payload = r.to_dict()
    assert payload["m"]["exists"] is True
    assert payload["missing"]["exists"] is False


def test_summary_json_roundtrip(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    r = ModelRegistry()
    r.add(RegisteredModel(name="m", backend="hf", path=str(d),
                          trained_from="sft", params_m=49.4))
    out = tmp_path / "summary.json"
    text = r.summary_json(out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == json.loads(text)
    assert loaded["m"]["params_m"] == 49.4
    assert loaded["m"]["trained_from"] == "sft"


def test_default_flag_reassignment():
    r = ModelRegistry()
    r.add(RegisteredModel(name="newdefault", backend="scratch",
                          path="/tmp", default=True))
    assert r.default().name == "newdefault"
    assert r.resolve("").name == "newdefault"


def test_registered_model_exists_false_on_missing_path():
    m = RegisteredModel(name="m", backend="scratch",
                        path="/definitely/missing/path/xyz")
    assert m.exists() is False
    assert m.to_dict()["exists"] is False


def test_resolve_for_loading_builtin_name():
    assert resolve_for_loading("scratch") == ("scratch", "checkpoints/scratch")


def test_resolve_for_loading_unknown_name_is_raw_hf_path():
    # legacy --model <dir> behaviour: unknown values pass through as hf-full
    assert resolve_for_loading("/models/some_hf_dir") == ("hf-full", "/models/some_hf_dir")


def test_resolve_for_loading_empty_raises():
    with pytest.raises(ModelRegistryError):
        resolve_for_loading("")
