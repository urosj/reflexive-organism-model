"""Test configuration to make system pytest see the local venv and package."""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    root = Path(__file__).resolve().parents[1]
    venv_site = (
        root
        / "venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    # Ensure venv site-packages and project root precede system paths.
    for path in (venv_site, root):
        if path.exists():
            sys.path.insert(0, str(path))
