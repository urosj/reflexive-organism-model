from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "experiments" / "scripts" / "run_v15_ablations.sh"
DOC_PATH = ROOT / "experiments" / "papers" / "11C-v15-AblationHarness.md"


def test_v15_cuda_ablation_script_exists_and_defines_three_profiles() -> None:
    assert SCRIPT_PATH.exists(), "Missing ablation harness script"
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'run_case "core-only"' in text
    assert '--closure-mode off' in text

    assert 'run_case "core-events"' in text
    assert '--events-control-in-core' in text

    assert 'run_case "full"' in text
    assert '--closure-mode full' in text


def test_v15_cuda_ablation_doc_references_harness() -> None:
    assert DOC_PATH.exists(), "Missing ablation harness doc"
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "bash experiments/scripts/run_v15_ablations.sh" in text
    assert "core-only" in text
    assert "core-events" in text
    assert "full" in text
