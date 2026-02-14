from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "simulations" / "active" / "simulation-v15-cuda.py"


def _compact_source() -> str:
    text = SRC_PATH.read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text)


def test_v15_cuda_core_keeps_mass_projection_and_metric_guardrails() -> None:
    src = _compact_source()
    assert "C_new = project_to_invariant(C_new, g_xx, g_xy, g_yy, target_mass)" in src
    assert src.count("C_new = project_to_invariant(C_new, g_xx, g_xy, g_yy, target_mass)") >= 2
    assert "G_ABS_MAX = 10.0" in src
    assert "DETG_MIN = 1e-3" in src
    assert "torch.clamp(g_xx, -G_ABS_MAX, G_ABS_MAX, out=g_xx)" in src
    assert "det_g = torch.where(det_g <= DETG_MIN, DETG_MIN, det_g)" in src


def test_v15_cuda_off_mode_routes_core_without_identity_feedback() -> None:
    src = _compact_source()
    assert "core_I_sum = None if closure_mode == \"off\" else I_sum" in src
    assert "if closure_mode != \"off\" and I_sum is not None:" in src
    assert "spark_mask = spark_mask_soft if events_control_in_core else zero_spark_mask" in src
