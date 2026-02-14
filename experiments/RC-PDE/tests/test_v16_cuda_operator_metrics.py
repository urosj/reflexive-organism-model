from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "experiments" / "papers" / "12-v16-Spec.md"
CHECKLIST_PATH = ROOT / "experiments" / "papers" / "12A-v16-ImplementationChecklist.md"
SIM_PATH = ROOT / "simulations" / "active" / "simulation-v16-cuda.py"


def test_v16_spec_requires_operator_diagnostics() -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert "Operator-carrying" in text
    assert "det(K)" in text
    assert "cond(K)" in text
    assert "operator drift" in text


def test_v16_checklist_requires_operator_metrics_outputs() -> None:
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    assert "Operator-Carrying Formalization" in text
    assert "degeneracy occupancy" in text


def test_v16_sim_persists_operator_metrics_when_sim_exists() -> None:
    if not SIM_PATH.exists():
        pytest.skip("simulation-v16-cuda.py not present yet")
    text = SIM_PATH.read_text(encoding="utf-8")
    assert "operator_detK_mean" in text
    assert "operator_detK_min" in text
    assert "operator_condK_mean" in text
    assert "operator_condK_max" in text
    assert "operator_detg_mean" in text
    assert "operator_detg_min" in text
    assert "operator_g_drift_rms" in text
    assert "operator_degeneracy_detK_frac" in text
    assert "operator_degeneracy_condK_frac" in text
    assert "operator_degeneracy_detg_frac" in text
    assert "OP_CONDK_OCC_THRESH" in text
    assert "OP_DETK_OCC_THRESH" in text
    assert "OP_DETG_OCC_THRESH" in text
    assert "[OPERATOR]" in text
