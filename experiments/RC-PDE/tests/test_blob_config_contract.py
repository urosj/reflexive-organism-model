from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOBS_PATH = ROOT / "configs" / "blobs.json"
SIM_PATHS = [
    ROOT / "simulations" / "legacy" / "simulation-v12.py",
    ROOT / "simulations" / "active" / "simulation-v12-cuda.py",
    ROOT / "simulations" / "active" / "simulation-v13-cuda.py",
    ROOT / "simulations" / "active" / "simulation-v14-cuda.py",
    ROOT / "simulations" / "active" / "simulation-v15-cuda.py",
    ROOT / "simulations" / "active" / "simulation-v16-cuda.py",
]


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_blobs_json_schema_is_present_and_valid() -> None:
    assert BLOBS_PATH.exists(), "Missing blobs.json"
    data = json.loads(BLOBS_PATH.read_text(encoding="utf-8"))
    blobs = data.get("blobs")
    assert isinstance(blobs, list) and blobs, "blobs.json must have a non-empty 'blobs' list"

    required = {"id", "x", "y", "sigma"}
    for idx, blob in enumerate(blobs):
        assert isinstance(blob, dict), f"Blob #{idx} must be an object"
        assert required.issubset(blob.keys()), f"Blob #{idx} missing required keys"
        assert isinstance(blob["id"], str) and blob["id"], f"Blob #{idx} id must be non-empty string"
        assert isinstance(blob["x"], (int, float)), f"Blob #{idx} x must be numeric"
        assert isinstance(blob["y"], (int, float)), f"Blob #{idx} y must be numeric"
        assert isinstance(blob["sigma"], (int, float)), f"Blob #{idx} sigma must be numeric"
        assert 0.0 <= float(blob["x"]) <= 1.0, f"Blob #{idx} x out of range [0, 1]"
        assert 0.0 <= float(blob["y"]) <= 1.0, f"Blob #{idx} y out of range [0, 1]"
        assert float(blob["sigma"]) > 0.0, f"Blob #{idx} sigma must be > 0"


def test_simulations_no_longer_use_hardcoded_blob_seed_literals() -> None:
    legacy_literals = [
        "C += gaussian_blob(Xg, Yg, X[Nx//3], Y[Ny//2], 1.5)",
        "C += gaussian_blob(Xg, Yg, X[2*Nx//3], Y[2*Ny//3], 3.0)",
        "C += gaussian_blob(Xg, Yg, X[Nx//2], Y[2*Ny//3], 5.0)",
        "C += gaussian_blob(Xg, Yg, x[Nx//3], y[Ny//2], 1.5)",
        "C += gaussian_blob(Xg, Yg, x[2*Nx//3], y[2*Ny//3], 2.0)",
    ]

    for sim_path in SIM_PATHS:
        src = _compact(sim_path.read_text(encoding="utf-8"))
        assert 'load_blob_specs("configs/blobs.json")' in src, f"{sim_path.name} must load configs/blobs.json"
        for literal in legacy_literals:
            assert literal not in src, f"{sim_path.name} still contains hardcoded blob seed literal: {literal}"
