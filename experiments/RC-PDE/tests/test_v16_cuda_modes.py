from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SIM_PATH = ROOT / "simulations" / "active" / "simulation-v16-cuda.py"
ABLA_PATH = ROOT / "experiments" / "scripts" / "run_v16_ablations.sh"
ALL_PATH = ROOT / "experiments" / "scripts" / "run_v16_iteration7_all.sh"


def _parse_args_fn_or_skip() -> ast.FunctionDef:
    if not SIM_PATH.exists():
        pytest.skip("simulation-v16-cuda.py not present yet")
    tree = ast.parse(SIM_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_args":
            return node
    raise AssertionError("Expected _parse_args() in simulation-v16-cuda.py")


def test_v16_cuda_scaffold_scripts_exist() -> None:
    assert ABLA_PATH.exists(), "Missing v16 ablation script"
    assert ALL_PATH.exists(), "Missing v16 iteration-7 aggregate script"


def test_v16_cuda_scaffold_ablation_profiles_are_wired() -> None:
    text = ABLA_PATH.read_text(encoding="utf-8")
    assert 'run_case "core-only"' in text
    assert 'run_case "core-events"' in text
    assert 'run_case "soft"' in text
    assert 'run_case "full"' in text
    assert 'run_case "nonlocal-off"' in text
    assert 'run_case "nonlocal-on"' in text


def test_v16_cuda_cli_exposes_closure_mode_when_sim_exists() -> None:
    parse_fn = _parse_args_fn_or_skip()
    flags: set[str] = set()
    choices: set[str] | None = None

    for node in ast.walk(parse_fn):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        flag = first.value
        if not flag.startswith("--"):
            continue
        flags.add(flag)
        if flag == "--closure-mode":
            for kw in node.keywords:
                if kw.arg == "choices" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    vals = [elt.value for elt in kw.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]
                    choices = set(vals)

    assert "--closure-mode" in flags
    assert "--nonlocal-mode" in flags
    assert "--operator-diagnostics" in flags
    assert "--domain-mode" in flags
    assert "--domain-adapt-strength" in flags
    assert "--domain-adapt-interval" in flags
    assert choices == {"off", "soft", "full"}
