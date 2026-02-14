from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
V15_PATH = ROOT / "simulations" / "active" / "simulation-v15-cuda.py"
V16_PATH = ROOT / "simulations" / "active" / "simulation-v16-cuda.py"


def _parse_args_fn(path: Path) -> ast.FunctionDef:
    if not path.exists():
        pytest.skip(f"{path.name} not present yet")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_args":
            return node
    raise AssertionError(f"Expected _parse_args() in {path.name}")


def _arg_flags(parse_fn: ast.FunctionDef) -> set[str]:
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


def test_v15_exposes_activity_slider_flag() -> None:
    parse_fn = _parse_args_fn(V15_PATH)
    flags = _arg_flags(parse_fn)
    assert "--activity" in flags


def test_v16_exposes_activity_slider_flag() -> None:
    parse_fn = _parse_args_fn(V16_PATH)
    flags = _arg_flags(parse_fn)
    assert "--activity" in flags


def test_v15_and_v16_use_shared_activity_mapping_module() -> None:
    text_v15 = V15_PATH.read_text(encoding="utf-8")
    text_v16 = V16_PATH.read_text(encoding="utf-8")
    assert "from configs.activity_config import map_activity" in text_v15
    assert "from configs.activity_config import map_activity" in text_v16
    assert "map_activity(float(ARGS.activity))" in text_v15
    assert "map_activity(float(ARGS.activity))" in text_v16
