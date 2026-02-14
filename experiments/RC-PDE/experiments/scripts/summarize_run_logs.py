#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


HEADLESS_RE = re.compile(
    r"\[HEADLESS DONE\]\s+steps=(?P<steps>\d+)\s+dt=(?P<dt>[0-9eE+\-.]+)\s+"
    r"mass=(?P<mass>[0-9eE+\-.]+)\s+I_mass=(?P<I_mass>[0-9eE+\-.]+)\s+ids=(?P<ids>\d+)"
)
WALL_RE = re.compile(r"wall_seconds=(?P<wall>[0-9eE+\-.]+)")


def _parse_log(log_path: Path) -> dict[str, float | int | str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rel_profile = str(log_path.parent)

    headless_match = None
    for m in HEADLESS_RE.finditer(text):
        headless_match = m

    wall_match = None
    for m in WALL_RE.finditer(text):
        wall_match = m

    row: dict[str, float | int | str] = {
        "profile": rel_profile,
        "steps": 0,
        "dt": 0.0,
        "mass": 0.0,
        "I_mass": 0.0,
        "ids": 0,
        "wall_seconds": 0.0,
        "steps_per_sec": 0.0,
    }
    if headless_match:
        row["steps"] = int(headless_match.group("steps"))
        row["dt"] = float(headless_match.group("dt"))
        row["mass"] = float(headless_match.group("mass"))
        row["I_mass"] = float(headless_match.group("I_mass"))
        row["ids"] = int(headless_match.group("ids"))
    if wall_match:
        row["wall_seconds"] = float(wall_match.group("wall"))

    steps = float(row["steps"])
    wall = float(row["wall_seconds"])
    if steps > 0.0 and wall > 0.0:
        row["steps_per_sec"] = steps / wall
    return row


def _to_markdown(rows: Iterable[dict[str, float | int | str]], root: Path) -> str:
    label = _infer_label(root)
    lines = []
    lines.append(f"# {label} Log Summary ({root})")
    lines.append("")
    lines.append("| profile | steps | dt | mass | I_mass | ids | wall_seconds | steps_per_sec |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        profile = str(r["profile"])
        lines.append(
            f"| {profile} | {int(r['steps'])} | {float(r['dt']):.3e} | "
            f"{float(r['mass']):.4f} | {float(r['I_mass']):.4f} | {int(r['ids'])} | "
            f"{float(r['wall_seconds']):.3f} | {float(r['steps_per_sec']):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _infer_label(root: Path) -> str:
    version_re = re.compile(r"v(?P<num>\d+)", re.IGNORECASE)
    for part in reversed(root.parts):
        m = version_re.search(part)
        if m:
            return f"v{m.group('num')}"
    return "Run"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize run logs into table/csv.")
    parser.add_argument("--root", type=Path, required=True, help="Root folder containing run.log files")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--md", type=Path, default=None, help="Optional Markdown output path")
    args = parser.parse_args()

    root = args.root.resolve()
    logs = sorted(root.rglob("run.log"))
    if not logs:
        print(f"[WARN] No run.log files found under {root}")
        return 0

    rows = []
    for log in logs:
        row = _parse_log(log)
        row["profile"] = str(log.parent.relative_to(root))
        rows.append(row)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "profile",
                    "steps",
                    "dt",
                    "mass",
                    "I_mass",
                    "ids",
                    "wall_seconds",
                    "steps_per_sec",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] Wrote CSV: {args.csv}")

    md_text = _to_markdown(rows, root)
    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md_text, encoding="utf-8")
        print(f"[OK] Wrote Markdown: {args.md}")
    else:
        print(md_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
