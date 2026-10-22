"""End-to-end data preparation: collect/clean -> tokenize -> pack bins.

This is the glue that turns cleaned JSONL produced by ``dipai.data.clean``
into the ``train.bin`` / ``val.bin`` that scratch training loads, using the
existing, tested ``dipai.dataset.prepare_dataset``. It is deliberately thin:
all the real streaming/cleaning/tokenization logic lives in modules that are
already unit-tested. This module's job is to (a) wire those stages together
with a single config, and (b) persist a ``manifest.json`` in the output dir
recording exactly what went in (inputs, tokenizer, block_size, seed, counts,
and the cleaning report) so every training run is reproducible and auditable.

Pipeline:
    raw/cleaned JSONL (+ .txt/.csv)
        -> dipai.dataset.prepare_dataset (clean + hash-split + tokenize + pack)
        -> train.bin, val.bin, dataset_stats.json
        -> manifest.json (provenance; links to the cleaning report if present)

CLI:
    python -m dipai.data.prepare --config configs/data.yaml
    python -m dipai.data.prepare --inputs data/cleaned/web.jsonl \
        --output-dir data/processed_smoke --tokenizer data/tokenizer \
        --block-size 512
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("dipai.data.prepare")


@dataclass
class PrepareConfig:
    """Resolved prepare settings (built from YAML + CLI overrides)."""
    inputs: List[str] = field(default_factory=list)   # cleaned JSONL / txt / csv
    output_dir: str = "data/processed"
    tokenizer_path: str = "data/tokenizer"            # dir with tokenizer.model
    block_size: int = 512
    val_fraction: float = 0.05
    seed: int = 1337
    text_key: str = "text"
    # optional caps (0 = off); forwarded to prepare_dataset
    max_documents: int = 0
    max_bytes: int = 0
    max_seconds: float = 0.0

    def validate(self) -> None:
        if not self.inputs:
            raise ValueError("inputs must list at least one file or glob")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")


def load_prepare_config(path: Optional[str] = None,
                        overrides: Optional[Dict[str, Any]] = None) -> PrepareConfig:
    """Load a YAML prepare config, rejecting unknown keys, then apply overrides."""
    data: Dict[str, Any] = {}
    if path:
        from dipai.config import load_yaml
        data = load_yaml(path)
    known = set(PrepareConfig.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    cfg = PrepareConfig(**data)
    for k, v in (overrides or {}).items():
        if v is not None and k in known:
            setattr(cfg, k, v)
    cfg.validate()
    return cfg


def _write_manifest(output_dir: Path, cfg: PrepareConfig, stats_dict: Dict[str, Any],
                    cleaning_report: Optional[Dict[str, Any]]) -> Path:
    manifest = {
        "schema_version": 1,
        "stage": "prepare",
        "config": asdict(cfg),
        "stats": stats_dict,
        "cleaning_report": cleaning_report,
        "artifacts": {
            "train_bin": "train.bin",
            "val_bin": "val.bin",
            "dataset_stats": "dataset_stats.json",
        },
    }
    p = output_dir / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", p)
    return p


def prepare(cfg: PrepareConfig) -> Dict[str, Any]:
    """Run the full prepare pipeline. Returns the manifest dict.

    Raises InvalidDatasetError (from dipai.dataset) on empty/unusable input so
    a training run can never start on silently-empty data.
    """
    from dipai.dataset import prepare_dataset
    from dipai.tokenizer import DipaiTokenizer

    cfg.validate()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading tokenizer from %s", cfg.tokenizer_path)
    tokenizer = DipaiTokenizer.load(cfg.tokenizer_path)

    # Pick up a cleaning report if one sits beside the first input, so the
    # manifest can link raw-cleaning provenance to the packed bins.
    cleaning_report = None
    for pattern in ("clean_report.json", "report.json"):
        cand = output_dir / pattern
        if cand.exists():
            try:
                cleaning_report = json.loads(cand.read_text(encoding="utf-8"))
                break
            except Exception:  # noqa: BLE001 - report is informational only
                pass

    log.info("preparing %d input(s) -> %s (block_size=%d)",
             len(cfg.inputs), output_dir, cfg.block_size)
    stats = prepare_dataset(
        input_paths=cfg.inputs,
        output_dir=output_dir,
        tokenizer=tokenizer,
        block_size=cfg.block_size,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        text_key=cfg.text_key,
        max_documents=cfg.max_documents,
        max_bytes=cfg.max_bytes,
        max_seconds=cfg.max_seconds,
    )

    stats_dict = stats.to_dict()
    manifest_path = _write_manifest(output_dir, cfg, stats_dict, cleaning_report)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log.info("prepare complete: train_blocks=%d val_blocks=%d",
             stats.train_blocks, stats.val_blocks)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dipai.data.prepare",
        description="Clean -> tokenize -> pack into train.bin/val.bin with a manifest.",
    )
    p.add_argument("--config", default=None, help="optional YAML config")
    p.add_argument("--inputs", nargs="*", default=None,
                   help="cleaned JSONL/txt/csv inputs (or globs)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--tokenizer", dest="tokenizer_path", default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-documents", type=int, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    from dipai.utils import setup_logging
    setup_logging()
    args = build_arg_parser().parse_args(argv)
    overrides = {
        "inputs": args.inputs,
        "output_dir": args.output_dir,
        "tokenizer_path": args.tokenizer_path,
        "block_size": args.block_size,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "max_documents": args.max_documents,
    }
    try:
        cfg = load_prepare_config(args.config, overrides)
    except ValueError as exc:
        log.error("bad config: %s", exc)
        return 2
    try:
        manifest = prepare(cfg)
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        log.error("prepare failed: %s", exc)
        return 1
    print(json.dumps({
        "train_blocks": manifest["stats"]["train_blocks"],
        "val_blocks": manifest["stats"]["val_blocks"],
        "train_tokens": manifest["stats"]["train_tokens"],
        "output_dir": cfg.output_dir,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
