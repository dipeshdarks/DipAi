"""Parameter-count acceptance for the real configs (smoke_4m + dipai_50m).

Builds each model on the meta device (zero RAM) and checks three things:
  * ModelConfig.param_budget_estimate() matches the measured count exactly
  * the measured count is inside the config's param_target range
Both are the same gates train.py enforces at runtime (exit 5 on failure).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dipai.config import load_scratch_config
from dipai.model import DipaiGPT

ROOT = Path(__file__).resolve().parents[1]


def _measured_params(model_cfg) -> int:
    with torch.device("meta"):
        model = DipaiGPT(model_cfg)
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize(
    "cfg_name", ["configs/smoke_4m.yaml", "configs/dipai_50m.yaml"]
)
def test_param_budget_matches_measured_and_target(cfg_name):
    cfg = load_scratch_config(ROOT / cfg_name)
    estimated = cfg.model.param_budget_estimate()
    measured = _measured_params(cfg.model)
    assert estimated == measured, (
        f"{cfg_name}: param_budget_estimate {estimated} != measured {measured}"
    )
    assert cfg.param_target is not None and cfg.param_target.enabled
    assert cfg.param_target.contains(measured), (
        f"{cfg_name}: {measured} params outside {cfg.param_target.describe()}"
    )
