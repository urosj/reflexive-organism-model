from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "simulations" / "active" / "simulation-v15-cuda.py"


def _source() -> str:
    return SRC_PATH.read_text(encoding="utf-8")


def _compact_source() -> str:
    return re.sub(r"\s+", " ", _source())


def _parse_args_fn() -> ast.FunctionDef:
    tree = ast.parse(_source())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_args":
            return node
    raise AssertionError("Expected _parse_args() in simulation-v15-cuda.py")


def test_v15_cuda_cli_exposes_mode_and_ablation_flags() -> None:
    parse_fn = _parse_args_fn()
    flags: set[str] = set()
    closure_mode_choices: set[str] | None = None

    for node in ast.walk(parse_fn):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            continue
        flag = node.args[0].value
        if not flag.startswith("--"):
            continue
        flags.add(flag)
        if flag == "--closure-mode":
            for kw in node.keywords:
                if kw.arg == "choices" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    vals = []
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            vals.append(elt.value)
                    closure_mode_choices = set(vals)

    assert "--closure-mode" in flags
    assert "--events-control-in-core" in flags
    assert closure_mode_choices == {"off", "soft", "full"}


def test_v15_cuda_step_closure_routes_off_soft_full() -> None:
    src = _compact_source()
    assert 'if closure_mode == "off":' in src
    assert 'if closure_mode == "soft":' in src
    assert 'if closure_mode == "full":' in src
    assert "update_identities(C, I_tensor, n_active, spark_mask, sqrt_g, dt, step, closure_softness)" in src
    assert "update_identities(C, I_tensor, n_active, spark_mask, sqrt_g, dt, step, 0.0)" in src
