from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SIM_PATH = ROOT / "simulations" / "active" / "simulation-v16-cuda.py"
CHECKLIST_PATH = ROOT / "experiments" / "papers" / "12A-v16-ImplementationChecklist.md"


def _compact(path: Path) -> str:
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def test_v16_checklist_tracks_invariants_and_mass_accounting() -> None:
    text = CHECKLIST_PATH.read_text(encoding="utf-8").lower()
    assert "mass projection tolerance" in text
    assert "mass accounting" in text


def test_v16_sim_projection_contract_when_sim_exists() -> None:
    if not SIM_PATH.exists():
        pytest.skip("simulation-v16-cuda.py not present yet")
    src = _compact(SIM_PATH)
    assert "project_to_invariant" in src
    assert "step_domain" in src
    assert "domain_mass_rel_error" in src
    assert "domain_mass_target_rel_error" in src
    assert "update_continuous_identity" in src
    assert "compute_closure_observables" in src
    assert "core_I_sum = I_cont" in src
    assert "compute_continuous_collapse_scores" in src
