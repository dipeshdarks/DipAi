"""Provenance manifests for ingestion runs.

A manifest records, per source: licence metadata, the resolved revision,
collection limits, per-source success/failure/skip counts, a content digest,
and timings. It is written next to the collected JSONL so any training run can
prove where its data came from.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SourceResult:
    """Counters + provenance for one collected source."""
    source: str
    kind: str
    status: str = "pending"        # ok | partial | failed | skipped
    docs_collected: int = 0
    docs_skipped_short: int = 0
    docs_skipped_long: int = 0
    docs_skipped_lang: int = 0
    docs_failed: int = 0
    bytes_collected: int = 0
    revision: str = ""             # HF commit hash / dump ETag / "local"
    license_spdx: str = ""
    license_url: str = ""
    attribution_required: bool = False
    share_alike: bool = False
    output_path: str = ""
    content_digest: str = ""       # sha1 over per-doc sha1s (order-dependent)
    started_at: float = 0.0
    ended_at: float = 0.0
    error: str = ""
    # rolling state so a resumed run can continue the digest
    _digest_state: str = field(default="", repr=False)

    def add_doc(self, text: str, out) -> None:
        """Write one JSONL record and fold its sha1 into the running digest."""
        doc = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        rec = {"text": text, "source": self.source, "doc_sha1": doc}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._digest_state = hashlib.sha1(
            (self._digest_state + doc).encode("ascii")
        ).hexdigest()
        self.docs_collected += 1
        self.bytes_collected += len(text.encode("utf-8", "replace"))

    def finish(self, status: str = "ok", error: str = "") -> None:
        self.status = status
        self.error = error
        self.ended_at = time.time()
        self.content_digest = self._digest_state

    def to_dict(self, include_private: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_digest_state", None)
        if not include_private:
            d.pop("error", None) if not self.error else None
        return d


@dataclass
class CollectManifest:
    """Top-level manifest for one collect run across N sources."""
    schema_version: int = 1
    run_id: str = ""
    created_at: float = field(default_factory=time.time)
    software: Dict[str, str] = field(default_factory=dict)
    pilot: bool = False
    limits: Dict[str, Any] = field(default_factory=dict)
    sources: List[SourceResult] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        if not self.software:
            import sys
            self.software = {
                "python": platform.python_version(),
                "platform": platform.platform(),
            }
            try:
                import torch
                self.software["torch"] = torch.__version__
            except Exception:
                pass

    @property
    def total_docs(self) -> int:
        return sum(s.docs_collected for s in self.sources)

    def summary(self) -> str:
        lines = [
            f"collect manifest {self.run_id}  pilot={self.pilot}",
            f"  total docs : {self.total_docs:,}",
        ]
        for s in self.sources:
            lines.append(
                f"  {s.source:24} {s.status:8} docs={s.docs_collected:<6} "
                f"skip(short/long/lang)={s.docs_skipped_short}/"
                f"{s.docs_skipped_long}/{s.docs_skipped_lang} "
                f"fail={s.docs_failed} lic={s.license_spdx}"
            )
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                     encoding="utf-8")
        return p

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "software": self.software,
            "pilot": self.pilot,
            "limits": self.limits,
            "total_docs": self.total_docs,
            "sources": [s.to_dict() for s in self.sources],
            "notes": self.notes,
        }


def load_manifest(path: str | Path) -> CollectManifest:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    m = CollectManifest(
        schema_version=data.get("schema_version", 1),
        run_id=data.get("run_id", ""),
        created_at=data.get("created_at", 0.0),
        software=data.get("software", {}),
        pilot=data.get("pilot", False),
        limits=data.get("limits", {}),
        notes=data.get("notes", ""),
    )
    for s in data.get("sources", []):
        r = SourceResult(**{k: v for k, v in s.items()
                            if k in SourceResult.__dataclass_fields__})
        m.sources.append(r)
    return m


def new_run_id(now: Optional[float] = None) -> str:
    t = time.localtime(now or time.time())
    return time.strftime("collect-%Y%m%d-%H%M%S", t)
