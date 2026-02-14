from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "simulations" / "active" / "simulation-v13-cuda.py"


def _source() -> str:
    return SRC_PATH.read_text(encoding="utf-8")


def _compact_source() -> str:
    return re.sub(r"\s+", " ", _source())


def _parse_args_flags() -> set[str]:
    tree = ast.parse(_source())
    parse_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_args":
            parse_fn = node
            break
    assert parse_fn is not None, "Expected _parse_args() in simulation-v13-cuda.py"

    flags: set[str] = set()
    for node in ast.walk(parse_fn):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str) and arg0.value.startswith("--"):
            flags.add(arg0.value)
    return flags


def test_v13_cuda_cli_exposes_identity_budget_controls() -> None:
    flags = _parse_args_flags()
    required = {
        "--headless",
        "--headless-steps",
        "--snapshot-interval",
        "--storage-mode",
        "--animate-interval",
        "--two-pass",
        "--identity-cap-fraction",
        "--coherence-bg-floor",
        "--identity-birth-gate-fraction",
        "--seed",
    }
    missing = required - flags
    assert not missing, f"Missing CLI flags in _parse_args: {sorted(missing)}"


def test_v13_cuda_identity_pde_is_heun_growth_decay_diffusion() -> None:
    src = _compact_source()
    assert "dIdt1 = g_id * C.unsqueeze(0) * I_active - d_id * I_active + D_id * lapI" in src
    assert "I_mid = I_active + 0.5 * dt * dIdt1" in src
    assert "dIdt2 = g_id * C.unsqueeze(0) * I_mid - d_id * I_mid + D_id * lapI_mid" in src
    assert "I_new = I_active + dt * dIdt2" in src


def test_v13_cuda_birth_gate_uses_all_paper7_global_constraints() -> None:
    src = _compact_source()
    assert "enough_sparks_t = num_sparks_t >= spark_birth_sparks_min" in src
    assert "has_mass_budget_t = total_I_mass_t < birth_gate_mass_cap" in src
    assert "has_interval_t = torch.scalar_tensor(step % id_birth_interval == 0, device=device)" in src
    assert "has_slots_t = n_survived_t < max_identities" in src
    assert "birth_allowed_t = torch.stack([ has_interval_t, enough_sparks_t, has_mass_budget_t, has_slots_t ]).all()" in src


def test_v13_cuda_identity_mass_is_metric_weighted() -> None:
    src = _compact_source()
    assert "mass_weight = sqrt_g.unsqueeze(0)" in src
    assert "masses = torch.sum(I_new * mass_weight, dim=(1, 2)) * dx * dy" in src
    assert "surviving_mass = torch.sum(I_new[alive_mask] * mass_weight) * dx * dy" in src
    assert "total_mass = torch.sum(I_tensor[:n_active] * mass_weight) * dx * dy" in src
    assert "I_mass = torch.sum(I_sum * sqrt_g) * dx * dy" in src


def test_v13_cuda_coherence_mass_projection_and_identity_curvature_terms() -> None:
    src = _compact_source()
    assert "current_mass = torch.sum(C * sqrt_g) * dx * dy" in src
    assert "correction = (delta_mass / (weight_sum.item() + eps)) * weight" in src
    assert "grad_term_xx = xi_grad * dC_dx * dC_dx" in src
    assert "id_term_xx = eta_id * I_sum * dC_dx * dC_dx" in src
