"""Model registry: logical name -> (backend, path) resolution.

Mirrors the shape of dipai.data.registry (dataclass + registry + built-ins)
so the two registries feel like one system. Nothing here loads a model: it
only records where a named model lives and which inference backend should
read it, so `--model <name>` / the webui dropdown work without full paths.

The three backends are the real loaders in dipai.inference:
    scratch  - DIPAI scratch checkpoint dir (dipai.train / checkpoints/*)
    hf       - PEFT adapter dir produced by dipai.finetune with use_lora: true
    hf-full  - full-parameter HF model dir from dipai.finetune (use_lora: false)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelRegistryError(Exception):
    """Unknown model, bad path, or refused registration."""


# The three real inference backends. Must match dipai.inference loaders.
BACKENDS = ("scratch", "hf", "hf-full")

# Default locations relative to the repo root, used to auto-fill `path`
# when a built-in is registered without an explicit one.
_DEFAULT_PATHS = {
    "scratch": "checkpoints/scratch",
}


@dataclass
class RegisteredModel:
    """One named model entry."""
    name: str
    backend: str                      # one of BACKENDS
    path: str                         # directory (or file) the backend reads
    description: str = ""
    # Free-form provenance, kept beside the entry like the data registry does.
    trained_from: str = ""            # e.g. "pretrain" | "cpt" | "sft" | "dpo"
    base_model: str = ""              # upstream base this was derived from
    params_m: float = 0.0             # approx params in millions (0 = unknown)
    tags: List[str] = field(default_factory=list)
    # A registered model may be "available" only once its path exists on disk.
    # Registry operations never require it to exist; resolution does.
    default: bool = False             # resolve as default when name is ""

    def validate(self) -> None:
        if not self.name:
            raise ModelRegistryError("model name must be non-empty")
        if self.backend not in BACKENDS:
            raise ModelRegistryError(
                f"model {self.name!r}: unknown backend {self.backend!r}; "
                f"expected one of {list(BACKENDS)}"
            )
        if not self.path:
            raise ModelRegistryError(f"model {self.name!r}: path must be non-empty")

    def exists(self) -> bool:
        return Path(self.path).exists()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["exists"] = self.exists()
        return d


def _builtin_models() -> Dict[str, RegisteredModel]:
    """Built-in entries. Paths are best-effort defaults; a missing path is not
    an error at registration time (see exists())."""
    scratch = RegisteredModel(
        name="scratch",
        backend="scratch",
        path=_DEFAULT_PATHS["scratch"],
        description="Scratch-pretrained DIPAI LM (dipai.train, base architecture).",
        trained_from="pretrain",
        params_m=0.0,
        tags=["base", "scratch"],
        default=True,
    )
    return {scratch.name: scratch}


_BUILTIN = _builtin_models()


class ModelRegistry:
    """Lookup + validation over named models. Starts from the built-ins."""

    def __init__(self, extra: Optional[List[RegisteredModel]] = None):
        self._models: Dict[str, RegisteredModel] = dict(_BUILTIN)
        for m in extra or []:
            m.validate()
            self._models[m.name] = m

    def get(self, name: str) -> RegisteredModel:
        try:
            return self._models[name]
        except KeyError:
            raise ModelRegistryError(
                f"unknown model {name!r}; known: {sorted(self._models)}"
            ) from None

    def names(self) -> List[str]:
        return sorted(self._models)

    def default(self) -> RegisteredModel:
        """The most recently registered entry flagged default=True, else
        the built-in 'scratch'."""
        for m in reversed(list(self._models.values())):
            if m.default:
                return m
        return self.get("scratch")

    def add(self, model: RegisteredModel) -> None:
        model.validate()
        self._models[model.name] = model

    def remove(self, name: str) -> bool:
        return self._models.pop(name, None) is not None

    def available(self) -> List[RegisteredModel]:
        """Entries whose path currently exists on disk."""
        return [m for m in self._models.values() if m.exists()]

    def resolve(self, name: str = "") -> RegisteredModel:
        """Resolve a name to a concrete model. Empty name -> the default."""
        if not name:
            return self.default()
        return self.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return {name: m.to_dict() for name, m in sorted(self._models.items())}

    def summary_json(self, path: Optional[str | Path] = None) -> str:
        """Dump every entry (with exists()) to a JSON string; optionally write."""
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path:
            Path(path).write_text(payload, encoding="utf-8")
        return payload


def default_registry() -> ModelRegistry:
    return ModelRegistry()


def resolve_for_loading(name_or_path: str) -> "tuple[str, str]":
    """Resolve a CLI --model value to (backend, path).

    A registered name resolves through its entry's backend (scratch / hf /
    hf-full). An unknown value is treated as a raw HF-model directory, which
    preserves the legacy `--model <dir>` behaviour (hf-full).
    """
    if not name_or_path:
        raise ModelRegistryError("--model value must not be empty")
    try:
        m = default_registry().resolve(name_or_path)
    except ModelRegistryError:
        return "hf-full", name_or_path
    return m.backend, m.path
