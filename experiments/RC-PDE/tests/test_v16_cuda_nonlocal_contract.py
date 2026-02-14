from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ABLA_PATH = ROOT / "experiments" / "scripts" / "run_v16_ablations.sh"
GATE_PATH = ROOT / "experiments" / "scripts" / "run_v16_iteration6_gate.sh"
SIM_PATH = ROOT / "simulations" / "active" / "simulation-v16-cuda.py"


def test_v16_nonlocal_profiles_exist_in_ablation_script() -> None:
    text = ABLA_PATH.read_text(encoding="utf-8")
    assert 'run_case "nonlocal-off"' in text
    assert 'run_case "nonlocal-on"' in text
    assert "--nonlocal-mode off" in text
    assert "--nonlocal-mode on" in text
    assert "NONLOCAL_OFF_EXTRA_ARGS" in text
    assert "NONLOCAL_ON_EXTRA_ARGS" in text


def test_v16_nonlocal_profiles_exist_in_gate_script() -> None:
    text = GATE_PATH.read_text(encoding="utf-8")
    assert '"nonlocal-off"' in text
    assert '"nonlocal-on"' in text
    assert "--nonlocal-mode off" in text
    assert "--nonlocal-mode on" in text
    assert "NONLOCAL_OVERHEAD_EXTRA_ARGS" in text


def test_v16_nonlocal_core_path_uses_explicit_fft_when_sim_exists() -> None:
    if not SIM_PATH.exists():
        pytest.skip("simulation-v16-cuda.py not present yet")
    text = SIM_PATH.read_text(encoding="utf-8")
    assert "torch.fft.rfft2" in text
    assert "torch.fft.irfft2" in text
    assert "dCdt = dCdt + nonlocal_strength * compute_nonlocal_proxy(C)" in text
