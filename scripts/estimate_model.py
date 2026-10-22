#!/usr/bin/env python
"""Measure a DIPAI model's exact parameter count (no training, no data).

Run from the DIPAI project root:

    python scripts/estimate_model.py --config configs/dipai_50m.yaml
    python scripts/estimate_model.py --config configs/smoke_4m.yaml

What it does:
  * loads the scratch config (model section only; data paths are NOT opened)
  * builds the model on the `meta` device (weights are never allocated, so
    even a 10B-param config would run in ~1 MB of RAM)
  * prints a per-component parameter breakdown, the closed-form estimate
    from config.param_budget_estimate, and the measured count
  * enforces config param_target: exit 1 if the measured count is outside
    [min_params, max_params], exit 0 otherwise, exit 2 on config error

This is the script the acceptance report should quote - never estimate
parameter counts by hand or reuse numbers from a different config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dipai.config import load_scratch_config
from dipai.model import build_model
from dipai.utils import setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Measure parameter count of a DIPAI model config (meta device)"
    )
    ap.add_argument("--config", "-c", default="configs/dipai_50m.yaml",
                    help="scratch YAML config (default: configs/dipai_50m.yaml)")
    return ap


def component_breakdown(model: torch.nn.Module) -> dict:
    groups = {
        "token_embedding": 0,
        "position_embedding": 0,
        "transformer_blocks": 0,
        "final_ln": 0,
        "lm_head (untied)": 0,
    }
    for name, p in model.named_parameters():
        if name.startswith("tok_emb"):
            groups["token_embedding"] += p.numel()
        elif name.startswith("pos_emb"):
            groups["position_embedding"] += p.numel()
        elif name.startswith("blocks."):
            groups["transformer_blocks"] += p.numel()
        elif name.startswith("ln_f"):
            groups["final_ln"] += p.numel()
        elif name.startswith("lm_head"):
            groups["lm_head (untied)"] += p.numel()
    return groups


def main() -> int:
    log = setup_logging()
    args = build_arg_parser().parse_args()

    try:
        cfg = load_scratch_config(args.config)
    except Exception as exc:
        log.error("failed to load config %s: %s", args.config, exc)
        return 2

    m = cfg.model
    log.info("config: %s | L=%d H=%d d=%d block=%d vocab=%d tie=%s bias=%s",
             args.config, m.n_layer, m.n_head, m.n_embd, m.block_size,
             m.vocab_size, m.tie_weights, m.bias)

    try:
        with torch.device("meta"):
            model = build_model(m)
    except Exception:  # pragma: no cover - very old torch without device ctx
        log.warning("meta-device build unavailable; building on CPU instead")
        model = build_model(m)

    measured = model.num_params
    estimate = m.param_budget_estimate()
    breakdown = component_breakdown(model)

    print()
    print("Parameter count report (meta device, exact - not an estimate)")
    print("-" * 58)
    for k, v in breakdown.items():
        if v:
            print(f"  {k:<24}: {v:>14,}  ({v / 1e6:.3f}M)")
    print(f"  {'MEASURED total':<24}: {measured:>14,}  ({measured / 1e6:.3f}M)")
    print(f"  {'closed-form estimate':<24}: {estimate:>14,}  "
          f"({estimate / 1e6:.3f}M)")
    print(f"  {'delta (est - measured)':<24}: {estimate - measured:>14,}")
    if m.tie_weights:
        print("  note: embeddings tied with lm_head (counted once)")

    pt = cfg.param_target
    if pt is not None and pt.enabled:
        ok = pt.contains(measured)
        print(f"  param_target {pt.describe()}: "
              f"{'PASS' if ok else 'FAIL'}")
        print("-" * 58)
        print()
        if not ok:
            log.error(
                "measured %d params is OUTSIDE the required range %s - fix the "
                "config before training.",
                measured, pt.describe(),
            )
            return 1
    else:
        print("  param_target: not configured (no range check)")
        print("-" * 58)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
