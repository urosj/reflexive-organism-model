from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "experiments" / "scripts" / "run_v16_ablations.sh"
CHECKLIST_PATH = ROOT / "experiments" / "papers" / "12A-v16-ImplementationChecklist.md"
RUNBOOK_PATH = ROOT / "experiments" / "papers" / "12E-v16-CUDARuntimeClosure.md"


def test_v16_ablation_script_exists_and_declares_required_profiles() -> None:
    assert SCRIPT_PATH.exists(), "Missing v16 ablation harness script"
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'run_case "core-only"' in text
    assert 'run_case "core-events"' in text
    assert 'run_case "soft"' in text
    assert 'run_case "full"' in text
    assert 'run_case "nonlocal-off"' in text
    assert 'run_case "nonlocal-on"' in text


def test_v16_docs_reference_runtime_harness() -> None:
    checklist = CHECKLIST_PATH.read_text(encoding="utf-8")
    runbook = RUNBOOK_PATH.read_text(encoding="utf-8")

    assert "run_v16_iteration7_all.sh" in checklist
    assert "run_v16_iteration7_all.sh" in runbook
    assert "v16 ablation matrix" in runbook
