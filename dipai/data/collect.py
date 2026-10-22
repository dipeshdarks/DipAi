"""Bounded, resumable collection of permitted public web data.

Design rules (see the project brief):
  * never download an entire multi-terabyte corpus by default — every source
    is capped by max_docs and max_bytes, and pilot configs use tiny caps;
  * stream, do not materialise — HF datasets are read with `streaming=True`
    and Wikimedia dumps with HTTP range requests + a hard byte ceiling;
  * retry with exponential backoff, but stop after N attempts;
  * record provenance (licence, revision, counts, digest) in a manifest;
  * resume support: a `.progress.json` beside the output lets an interrupted
    run skip docs it already wrote.

CLI:
    python -m dipai.data.collect --config configs/data.yaml
    python -m dipai.data.collect --config configs/data.yaml --sources fineweb-edu
    python -m dipai.data.collect --config configs/data.yaml --dry-run
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import random
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dipai.data.manifest import CollectManifest, SourceResult, new_run_id
from dipai.data.registry import DataSource, RegistryError, SourceRegistry

log = logging.getLogger("dipai.data.collect")

# MediaWiki XML ns (dumps use this default namespace).
_MEDIAWIKI_NS = "{http://www.mediawiki.org/xml/export-0.11/}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CollectConfig:
    """Resolved collection settings (built from YAML + CLI overrides)."""
    output_dir: str = "data/raw/web"
    cache_dir: str = "data/raw/.cache"
    max_docs_per_source: int = 500          # hard per-source document cap
    max_bytes_per_source: int = 20_000_000  # hard per-source byte cap
    max_dump_bytes: int = 50_000_000        # hard byte cap for wikimedia_dump
    pilot: bool = True                      # only pilot_default sources unless --sources
    sources: List[str] = field(default_factory=list)   # explicit allowlist (empty = per pilot)
    retry_attempts: int = 3
    retry_backoff: float = 2.0              # seconds; grows exponentially
    sleep_between_docs: float = 0.0         # politeness delay (seconds)
    seed: int = 1337
    resume: bool = True                     # honour .progress.json if present
    hf_cache: str = ""                      # HF_DATASETS_CACHE override

    def validate(self) -> None:
        if self.max_docs_per_source <= 0:
            raise ValueError("max_docs_per_source must be > 0")
        if self.max_bytes_per_source <= 0:
            raise ValueError("max_bytes_per_source must be > 0")
        if self.retry_attempts < 1:
            raise ValueError("retry_attempts must be >= 1")
        if self.max_dump_bytes <= 0:
            raise ValueError("max_dump_bytes must be > 0")


def load_collect_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> CollectConfig:
    """Load `configs/data.yaml` -> CollectConfig, rejecting unknown keys."""
    from dipai.config import load_yaml
    data = load_yaml(path) if path else {}
    known = {f for f in CollectConfig.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    cfg = CollectConfig(**data)
    for k, v in (overrides or {}).items():
        if v is not None and k in known:
            setattr(cfg, k, v)
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _retry(fn, attempts: int, backoff: float, what: str):
    """Call fn() with exponential backoff. Re-raises the last error."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - any transport error is retryable here
            last = exc
            if i == attempts - 1:
                break
            wait = backoff * (2 ** i)
            log.warning("%s failed (%s); retry %d/%d in %.1fs",
                        what, exc, i + 1, attempts - 1, wait)
            time.sleep(wait)
    raise last


def _progress_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".progress.json")


def _load_progress(out_path: Path) -> Dict[str, Any]:
    p = _progress_path(out_path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt progress -> start over
            return {}
    return {}


def _save_progress(out_path: Path, docs_written: int, digest_state: str) -> None:
    p = _progress_path(out_path)
    p.write_text(json.dumps({"docs_written": docs_written,
                             "digest_state": digest_state}), encoding="utf-8")


def _clear_progress(out_path: Path) -> None:
    p = _progress_path(out_path)
    if p.exists():
        p.unlink()


def _language_ok(doc: Dict[str, Any], src: DataSource) -> bool:
    if not src.languages:
        return True
    if not src.language_field:
        return True
    lang = str(doc.get(src.language_field, "")).lower()
    # accept bare ("en") and region ("en-US") codes
    base = re.split(r"[-_]", lang)[0]
    return base in {l.lower() for l in src.languages}


def _size_ok(text: str, src: DataSource) -> Tuple[bool, str]:
    n = len(text)
    if n < src.min_chars:
        return False, "short"
    if n > src.max_chars:
        return False, "long"
    return True, ""


# ---------------------------------------------------------------------------
# Collectors (one per kind)
# ---------------------------------------------------------------------------

def _collect_hf_dataset(src: DataSource, cfg: CollectConfig,
                        result: SourceResult, out) -> None:
    """Stream a Hugging Face dataset with `datasets`, bounded + resumable."""
    from datasets import load_dataset  # lazy: heavy import

    def _open():
        kwargs: Dict[str, Any] = {"streaming": True, "split": src.hf_split}
        if src.hf_config:
            kwargs["name"] = src.hf_config
        if src.hf_revision:
            kwargs["revision"] = src.hf_revision
        if cfg.hf_cache:
            os.environ["HF_DATASETS_CACHE"] = cfg.hf_cache
        return load_dataset(src.hf_id, **kwargs)

    ds = _retry(lambda: _open(), cfg.retry_attempts, cfg.retry_backoff,
                f"open hf dataset {src.hf_id}")

    # Resolve a concrete revision for provenance when possible.
    try:
        from huggingface_hub import HfApi
        info = _retry(lambda: HfApi().dataset_info(src.hf_id, revision=src.hf_revision or None),
                      cfg.retry_attempts, cfg.retry_backoff, f"info {src.hf_id}")
        result.revision = getattr(info, "sha", "") or ""
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        log.debug("could not resolve HF revision for %s: %s", src.hf_id, exc)
        result.revision = src.hf_revision or "unresolved"

    progress = _load_progress(Path(result.output_path)) if result.output_path else {}
    skip = progress.get("docs_written", 0) if cfg.resume else 0
    if skip:
        log.info("resuming %s: skipping %d already-written docs", src.name, skip)

    import hashlib
    written = 0
    for doc in ds:
        if written >= cfg.max_docs_per_source:
            result.status = "ok"
            log.info("%s hit max_docs_per_source=%d", src.name, cfg.max_docs_per_source)
            return
        if result.bytes_collected >= cfg.max_bytes_per_source:
            result.status = "ok"
            log.info("%s hit max_bytes_per_source=%d", src.name, cfg.max_bytes_per_source)
            return
        try:
            text = doc.get(src.text_field, "")
        except Exception:  # noqa: BLE001
            result.docs_failed += 1
            continue
        if not isinstance(text, str) or not text.strip():
            result.docs_failed += 1
            continue
        if not _language_ok(doc, src):
            result.docs_skipped_lang += 1
            continue
        text = text.strip()
        ok, why = _size_ok(text, src)
        if not ok:
            if why == "short":
                result.docs_skipped_short += 1
            else:
                result.docs_skipped_long += 1
            continue
        if written < skip:
            # already written by a previous run: fold into digest, don't rewrite
            doc_sha = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
            result._digest_state = hashlib.sha1(
                (result._digest_state + doc_sha).encode("ascii")).hexdigest()
            written += 1
            result.docs_collected += 1
            result.bytes_collected += len(text.encode("utf-8", "replace"))
            continue
        result.add_doc(text, out)
        written += 1
        _save_progress(Path(result.output_path), written, result._digest_state)
        if cfg.sleep_between_docs > 0:
            time.sleep(cfg.sleep_between_docs)
    result.status = "ok"


def _collect_wikimedia_dump(src: DataSource, cfg: CollectConfig,
                            result: SourceResult, out) -> None:
    """Download a bounded prefix of a dumps.wikimedia.org XML file and parse
    <page><revision><text> out of it with a streaming pull parser.

    A hard `cfg.max_dump_bytes` ceiling stops us ever pulling a full ~20 GB
    enwiki dump by accident. Supports HTTP Range resume.
    """
    url = src.dump_url
    cache = Path(cfg.cache_dir) / (re.sub(r"[^A-Za-z0-9._-]", "_", url.rsplit("/", 1)[-1]))
    cache.parent.mkdir(parents=True, exist_ok=True)

    have = cache.stat().st_size if cache.exists() else 0
    want = cfg.max_dump_bytes
    headers = {"User-Agent": "DIPAI-collect/0.1 (research; bounded)"}
    if have and have >= want:
        log.info("%s: using cached prefix %s (%d bytes)", src.name, cache, have)
    else:
        if have:
            headers["Range"] = f"bytes={have}-{want - 1}"
        req = urllib.request.Request(url, headers=headers)

        def _fetch():
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(cache, "ab" if have else "wb") as f:
                copied = 0
                while copied < want - have:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    copied += len(chunk)
            return copied

        got = _retry(_fetch, cfg.retry_attempts, cfg.retry_backoff, f"fetch {url}")
        log.info("%s: fetched %d bytes (cap %d)", src.name, got, want)
    result.revision = f"prefix-{cache.stat().st_size}B"

    # Parse whatever prefix we have. A truncated gzip stream raises at the end
    # - that is expected and handled.
    written = 0
    progress = _load_progress(Path(result.output_path))
    skip = progress.get("docs_written", 0) if cfg.resume else 0
    text_key = _MEDIAWIKI_NS + "text"
    page_key = _MEDIAWIKI_NS + "page"
    title_key = _MEDIAWIKI_NS + "title"
    try:
        with gzip.open(cache, "rb") as fh:
            for _, elem in ET.iterparse(fh, events=("end",)):
                if elem.tag != page_key:
                    continue
                title = elem.findtext(title_key) or ""
                text = elem.findtext(text_key) or ""
                elem.clear()
                if not text.strip():
                    result.docs_failed += 1
                    continue
                # strip wiki markup lightly + drop disambiguation/non-articles
                if title.startswith(("Template:", "File:", "Category:", "WP:")):
                    result.docs_skipped_short += 1
                    continue
                flat = _strip_wiki_markup(text)
                ok, why = _size_ok(flat, src)
                if not ok:
                    if why == "short":
                        result.docs_skipped_short += 1
                    else:
                        result.docs_skipped_long += 1
                    continue
                if written < skip:
                    written += 1
                    result.docs_collected += 1
                    continue
                if written >= cfg.max_docs_per_source or \
                        result.bytes_collected >= cfg.max_bytes_per_source:
                    result.status = "ok"
                    return
                result.add_doc(flat, out)
                written += 1
                _save_progress(Path(result.output_path), written, result._digest_state)
    except (EOFError, gzip.BadGzipFile) as exc:
        # truncated prefix is the normal case for a bounded download
        log.info("%s: reached end of bounded prefix (%s)", src.name, exc)
    result.status = "ok"


def _strip_wiki_markup(text: str) -> str:
    """Light, deterministic wiki-markup stripping (not a full parser)."""
    t = re.sub(r"\{\{[^{}]*\}\}", " ", text)            # templates
    t = re.sub(r"<ref[^>]*/>", " ", t)                   # self-closing refs
    t = re.sub(r"<ref[^>]*>.*?</ref>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)                       # remaining html
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", t)  # [[a|b]] -> b
    t = re.sub(r"\[https?://[^\]\s]+\s*([^\]]*)\]", r"\1", t)  # [url label]
    t = re.sub(r"'''?|''", "", t)
    t = re.sub(r"={2,}\s*([^=]+?)\s*={2,}", r"\1", t)   # headings
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _collect_local(src: DataSource, cfg: CollectConfig,
                   result: SourceResult, out) -> None:
    """Collect from already-present local files (no network)."""
    result.revision = "local"
    written = 0
    progress = _load_progress(Path(result.output_path))
    skip = progress.get("docs_written", 0) if cfg.resume else 0
    seen_digests: set = set()
    for pat in src.paths:
        paths = [Path(pat)] if Path(pat).is_file() else sorted(Path().glob(pat))
        for p in paths:
            if not p.exists():
                log.warning("%s: missing path %s", src.name, p)
                result.docs_failed += 1
                continue
            if p.suffix == ".jsonl":
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except Exception:  # noqa: BLE001
                            result.docs_failed += 1
                            continue
                        text = rec.get(src.text_field, "")
                        if not isinstance(text, str) or not text.strip():
                            result.docs_failed += 1
                            continue
                        if not _language_ok(rec, src):
                            result.docs_skipped_lang += 1
                            continue
                        text = text.strip()
                        ok, why = _size_ok(text, src)
                        if not ok:
                            if why == "short":
                                result.docs_skipped_short += 1
                            else:
                                result.docs_skipped_long += 1
                            continue
                        import hashlib
                        d = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
                        if d in seen_digests:
                            result.docs_skipped_short += 1  # counted as dup
                            continue
                        seen_digests.add(d)
                        if written < skip:
                            written += 1
                            result.docs_collected += 1
                            continue
                        if written >= cfg.max_docs_per_source or \
                                result.bytes_collected >= cfg.max_bytes_per_source:
                            result.status = "ok"
                            return
                        result.add_doc(text, out)
                        written += 1
                        _save_progress(Path(result.output_path), written, result._digest_state)
            else:  # .txt - blank-line separated paragraphs
                blob = p.read_text(encoding="utf-8", errors="replace")
                for para in blob.split("\n\n"):
                    para = para.strip()
                    if not para:
                        continue
                    ok, why = _size_ok(para, src)
                    if not ok:
                        if why == "short":
                            result.docs_skipped_short += 1
                        else:
                            result.docs_skipped_long += 1
                        continue
                    if written >= cfg.max_docs_per_source or \
                            result.bytes_collected >= cfg.max_bytes_per_source:
                        result.status = "ok"
                        return
                    result.add_doc(para, out)
                    written += 1
                    _save_progress(Path(result.output_path), written, result._digest_state)
    result.status = "ok"


_COLLECTORS = {
    "hf_dataset": _collect_hf_dataset,
    "wikimedia_dump": _collect_wikimedia_dump,
    "local_files": _collect_local,
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def collect(sources: List[DataSource], cfg: CollectConfig,
            dry_run: bool = False) -> CollectManifest:
    """Collect every source into its own JSONL under cfg.output_dir."""
    cfg.validate()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = CollectManifest(run_id=new_run_id(), pilot=cfg.pilot,
                               limits={
                                   "max_docs_per_source": cfg.max_docs_per_source,
                                   "max_bytes_per_source": cfg.max_bytes_per_source,
                                   "max_dump_bytes": cfg.max_dump_bytes,
                                   "retry_attempts": cfg.retry_attempts,
                                   "seed": cfg.seed,
                               })
    for src in sources:
        src.validate()
        out_path = out_dir / f"{src.name}.jsonl"
        result = SourceResult(
            source=src.name, kind=src.kind,
            license_spdx=src.license.spdx, license_url=src.license.url,
            attribution_required=src.license.attribution_required,
            share_alike=src.license.share_alike,
            output_path=str(out_path), started_at=time.time(),
        )
        if dry_run:
            result.status = "skipped"
            manifest.sources.append(result)
            log.info("[dry-run] would collect %s (%s) -> %s", src.name, src.kind, out_path)
            continue
        log.info("collecting %s (%s) -> %s", src.name, src.kind, out_path)
        collector = _COLLECTORS[src.kind]
        try:
            # open in append mode so resume keeps prior docs
            mode = "a" if (cfg.resume and out_path.exists()) else "w"
            with open(out_path, mode, encoding="utf-8") as out:
                if mode == "w":
                    result._digest_state = ""
                else:
                    prog = _load_progress(out_path)
                    result._digest_state = prog.get("digest_state", "")
                    result.docs_collected = prog.get("docs_written", 0)
                collector(src, cfg, result, out)
            _clear_progress(out_path)
            result.finish(status=result.status or "ok")
        except Exception as exc:  # noqa: BLE001 - one bad source must not kill the run
            log.error("source %s failed: %s", src.name, exc)
            result.finish(status="failed", error=str(exc))
        manifest.sources.append(result)
        log.info("  -> %s docs=%d status=%s", src.name, result.docs_collected, result.status)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dipai.data.collect",
        description="Collect permitted public web data (bounded, resumable, licensed).",
    )
    p.add_argument("--config", "-c", default="configs/data.yaml",
                   help="YAML with CollectConfig keys")
    p.add_argument("--sources", nargs="*", default=None,
                   help="explicit source allowlist (overrides pilot)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-docs", type=int, default=None,
                   help="override max_docs_per_source")
    p.add_argument("--max-bytes", type=int, default=None)
    p.add_argument("--pilot", action="store_true", default=None,
                   help="force pilot mode (only pilot_default sources)")
    p.add_argument("--no-pilot", dest="pilot", action="store_false")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--list", action="store_true", help="list known sources and exit")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    from dipai.utils import setup_logging
    setup_logging()
    args = build_arg_parser().parse_args(argv)

    registry = SourceRegistry()
    if args.list:
        for s in registry.enabled():
            print(f"{s.name:24} {s.kind:16} lic={s.license.spdx:18} pilot={s.pilot_default}")
            print(f"  {s.description}")
        return 0

    overrides = {
        "output_dir": args.output_dir,
        "max_docs_per_source": args.max_docs,
        "max_bytes_per_source": args.max_bytes,
        "pilot": args.pilot,
        "resume": args.resume if not args.resume else None,
    }
    try:
        cfg = load_collect_config(args.config, overrides)
    except FileNotFoundError:
        log.warning("config %s not found; using defaults", args.config)
        cfg = CollectConfig()
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        cfg.validate()
    except (ValueError, RegistryError) as exc:
        log.error("bad config: %s", exc)
        return 2

    try:
        if args.sources:
            chosen = [registry.get(n) for n in args.sources]
        elif cfg.sources:
            chosen = [registry.get(n) for n in cfg.sources]
        elif cfg.pilot:
            chosen = registry.pilot()
            log.info("pilot mode: %s", [s.name for s in chosen])
        else:
            chosen = registry.enabled()
    except RegistryError as exc:
        log.error("%s", exc)
        return 2

    if not chosen:
        log.error("no sources selected")
        return 2

    try:
        manifest = collect(chosen, cfg, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        log.error("collection failed: %s", exc)
        return 1

    if not args.dry_run:
        out = Path(cfg.output_dir) / f"{manifest.run_id}_manifest.json"
        manifest.save(out)
        log.info("manifest written: %s", out)
    print(manifest.summary())
    if any(s.status == "failed" for s in manifest.sources):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
