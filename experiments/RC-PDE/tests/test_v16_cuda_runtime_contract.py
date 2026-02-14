from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "experiments" / "scripts" / "run_v16_iteration6_gate.sh"
SIM_PATH = ROOT / "simulations" / "active" / "simulation-v16-cuda.py"


def test_v16_gate_script_includes_adaptive_domain_profiles() -> None:
    text = GATE_PATH.read_text(encoding="utf-8")
    assert 'run_case "${OUT_DIR}/throughput" "adaptive-soft"' in text
    assert 'run_case "${OUT_DIR}/snapshot-overhead" "adaptive-soft-int${interval}"' in text
    assert "--domain-mode adaptive" in text
    assert "--domain-adapt-strength" in text
    assert "ADAPTIVE_DOMAIN_STRENGTH" in text
    assert "ADAPTIVE_DOMAIN_INTERVAL" in text


def test_v16_runtime_logs_expose_event_and_domain_markers_when_sim_exists() -> None:
    if not SIM_PATH.exists():
        pytest.skip("simulation-v16-cuda.py not present yet")
    text = SIM_PATH.read_text(encoding="utf-8")
    assert "[EVENT]" in text
    assert "spark_score_mean=" in text
    assert "[DOMAIN]" in text
    assert 'if domain_mode == "adaptive" and domain_adapt_strength > 0.0' in text
