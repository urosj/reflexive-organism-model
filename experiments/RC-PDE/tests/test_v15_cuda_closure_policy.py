from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "simulations" / "active" / "simulation-v15-cuda.py"


def _compact_source() -> str:
    text = SRC_PATH.read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text)


def test_v15_cuda_closure_policy_exposes_mode_specific_softness() -> None:
    src = _compact_source()
    assert "def update_identities(C, I_tensor, n_active, spark_mask, sqrt_g, dt, step, closure_softness_local):" in src
    assert "hard_prune_mass = id_min_mass * (1.0 - 0.5 * closure_softness_local)" in src
    assert "sparks_factor_t = (1.0 - closure_softness_local) * sparks_hard_t + closure_softness_local * sparks_soft_t" in src
    assert "if closure_mode == \"full\":" in src
    assert "return update_identities(C, I_tensor, n_active, spark_mask, sqrt_g, dt, step, 0.0)" in src


def test_v15_cuda_closure_diagnostics_are_logged() -> None:
    src = _compact_source()
    assert "\"birth_score_t\": birth_score_t" in src
    assert "\"sparks_factor_t\": sparks_factor_t" in src
    assert "\"mass_factor_t\": mass_factor_t" in src
    assert "\"slots_factor_t\": slots_factor_t" in src
    assert "\"interval_factor_t\": interval_factor_t" in src
    assert "closure_birth_score_val=closure_state[\"birth_score_t\"].item()" in src
    assert "closure_sparks_factor_val=closure_state[\"sparks_factor_t\"].item()" in src
    assert "closure_mass_factor_val=closure_state[\"mass_factor_t\"].item()" in src
    assert "closure_slots_factor_val=closure_state[\"slots_factor_t\"].item()" in src
    assert "closure_interval_factor_val=closure_state[\"interval_factor_t\"].item()" in src
    assert "closure_births_val=closure_state[\"n_new_t\"].item()" in src
