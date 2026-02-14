from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "simulations" / "active" / "simulation-v13-cuda.py"
PAPER7_PATH = ROOT / "experiments" / "papers" / "7-IdentitiesAddon.md"
PAPER8_PATH = ROOT / "experiments" / "papers" / "8-Proposal-FieldTheory.md"
PAPER9_PATH = ROOT / "experiments" / "papers" / "9-Proposal-EmergentAgency.md"


def _compact(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text)


def test_v13_cuda_alignment_paper7_identity_budget_and_punctuated_birth() -> None:
    p7 = _compact(PAPER7_PATH)
    src = _compact(SRC_PATH)

    # Paper 7: mass per identity and global gating constraints.
    assert "M_k = \\int I_k(x,t)dxdy" in p7
    assert "global identity-mass budget" in p7
    assert "constraints are satisfied do we place new identities" in p7

    # Code: spark gate + mass budget + interval + per-step birth + max identities.
    assert "spark_birth_sparks_min" in src
    assert "I_global_mass_cap" in src
    assert "id_birth_interval" in src
    assert "id_birth_per_step" in src
    assert "max_identities" in src
    assert "birth_allowed_t = torch.stack([ has_interval_t, enough_sparks_t, has_mass_budget_t, has_slots_t ]).all()" in src


def test_v13_cuda_alignment_paper8_reflexive_triad_and_coherence_invariant() -> None:
    p8 = _compact(PAPER8_PATH).lower()
    src = _compact(SRC_PATH)

    # Paper 8: RC-II triad and coherence invariant with non-conserved identity mass.
    assert "closed triadic feedback system" in p8
    assert "no conservation of identity mass" in p8
    assert "\\frac{d}{dt} \\int_\\omega c\\sqrt{|g|}dx = 0" in p8

    # Code: triad couplings and invariant projection.
    assert "dCdt = dCdt_flux + alpha_id * I_sum" in src
    assert "id_term_xx = eta_id * I_sum * dC_dx * dC_dx" in src
    assert "C_new = project_to_invariant(C_new, g_xx, g_xy, g_yy, target_mass)" in src


def test_v13_cuda_alignment_paper9_discrete_identity_lifecycle() -> None:
    p9 = _compact(PAPER9_PATH)
    src = _compact(SRC_PATH)

    # Paper 9: discrete tracked identities with birth, growth/decay/diffusion, collapse.
    assert "RC-III" in p9
    assert "spark_birth_sparks_min" in p9
    assert "I_global_mass_cap" in p9
    assert "M_k = \\int I_k dx < M_{\\min}" in p9

    # Code: discrete tensor slots, birth insertion, and prune/collapse threshold.
    assert "I_tensor = torch.zeros(max_identities, Nx, Ny, device=device)" in src
    assert "I_tensor[n_active:n_active+n_new] = seed_identities_at_batch(selected_ix, selected_iy)" in src
    assert "alive_mask = masses >= id_min_mass" in src
