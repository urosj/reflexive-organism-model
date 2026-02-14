#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_METRICS = [
    "OCR_global_median_rb",
    "OCR_active_median_rb",
    "FRG1_median",
    "FRG2_median",
    "FRG_saturation_ratio",
    "LCE_C_max",
    "LCE_C_auc",
    "LCE_J_auc",
    "LCE_g_auc",
]

EPS = 1e-12


def _load_scores(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for score_path in sorted(root.rglob("scores.json")):
        data = json.loads(score_path.read_text(encoding="utf-8"))
        runs.append({"path": str(score_path), "scores": data})
    return runs


def _metric_summary(vals: list[float], max_cv: float, abs_tol: float) -> dict[str, Any]:
    m = mean(vals)
    s = pstdev(vals) if len(vals) > 1 else 0.0
    cv = s / (abs(m) + EPS)
    stable = (cv <= max_cv) or (s <= abs_tol)
    return {
        "count": len(vals),
        "mean": m,
        "std": s,
        "min": min(vals),
        "max": max(vals),
        "cv": cv,
        "stable": bool(stable),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check run-to-run stability from repeated read-back score artifacts.")
    parser.add_argument("--root", required=True, help="Root directory containing repeated run score files.")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS, help="Metrics to evaluate for stability.")
    parser.add_argument("--max-cv", type=float, default=0.10, help="Maximum coefficient-of-variation for stability.")
    parser.add_argument("--abs-tol", type=float, default=1e-5, help="Absolute std tolerance fallback for near-zero metrics.")
    parser.add_argument("--out-json", default="", help="Optional output JSON path.")
    parser.add_argument("--out-md", default="", help="Optional output markdown path.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    runs = _load_scores(root)
    if not runs:
        raise SystemExit(f"No scores.json files found under {root}")

    report: dict[str, Any] = {
        "root": str(root),
        "n_runs": len(runs),
        "max_cv": float(args.max_cv),
        "abs_tol": float(args.abs_tol),
        "metrics": {},
        "decision_consistent": True,
        "decision_values": [],
        "overall_stable": True,
        "run_paths": [r["path"] for r in runs],
    }

    decisions: list[str] = []
    for run in runs:
        dec = run["scores"].get("decision")
        if isinstance(dec, str):
            decisions.append(dec)
    report["decision_values"] = sorted(set(decisions))
    report["decision_consistent"] = len(report["decision_values"]) <= 1

    all_metric_stable = True
    for metric in args.metrics:
        vals: list[float] = []
        for run in runs:
            val = run["scores"].get(metric)
            if val is None:
                continue
            try:
                vals.append(float(val))
            except Exception:
                continue
        if not vals:
            report["metrics"][metric] = {"count": 0, "stable": False, "reason": "missing"}
            all_metric_stable = False
            continue
        summary = _metric_summary(vals, max_cv=float(args.max_cv), abs_tol=float(args.abs_tol))
        report["metrics"][metric] = summary
        if not bool(summary["stable"]):
            all_metric_stable = False

    report["overall_stable"] = bool(all_metric_stable and report["decision_consistent"])

    out_json = Path(args.out_json) if args.out_json else root / "stability.json"
    out_md = Path(args.out_md) if args.out_md else root / "stability.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Readback Stability Report",
        "",
        f"- root: `{report['root']}`",
        f"- runs: `{report['n_runs']}`",
        f"- max_cv: `{report['max_cv']}`",
        f"- abs_tol: `{report['abs_tol']}`",
        f"- decision_consistent: `{report['decision_consistent']}` (`{report['decision_values']}`)",
        f"- overall_stable: `{report['overall_stable']}`",
        "",
        "| metric | count | mean | std | cv | min | max | stable |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for metric in args.metrics:
        row = report["metrics"].get(metric, {})
        if row.get("count", 0) <= 0:
            lines.append(f"| {metric} | 0 | - | - | - | - | - | false |")
            continue
        lines.append(
            f"| {metric} | {int(row['count'])} | {float(row['mean']):.6e} | "
            f"{float(row['std']):.6e} | {float(row['cv']):.6e} | {float(row['min']):.6e} | "
            f"{float(row['max']):.6e} | {bool(row['stable'])} |"
        )
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Wrote stability JSON: {out_json}")
    print(f"[OK] Wrote stability markdown: {out_md}")
    print(f"[OK] overall_stable={report['overall_stable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
