#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_REPORT_SECTIONS = [
    "#### 0) Run metadata",
    "#### 1) Parameter summary",
    "#### 2) Executive summary (1 screen)",
    "#### 3) Tier A: Counterfactual snapshot results",
    "#### 4) OCR: Term dominance over time",
    "#### 5) FRG: Gain and saturation",
    "#### 6) Tier B: Lagged causal effects (experience accumulation)",
    "#### 7) RLI: Return / loop closure",
    "#### 8) CAI: Attenuation bottleneck",
    "#### 9) ALI: Redundancy / alignment",
    "#### 10) Cross-version fingerprints (when running experiment matrix)",
    "#### 11) Notes / anomalies (auto + manual)",
]

REQUIRED_FIGURE_EXACT = [
    "ocr_term_energy_stack.png",
    "ocr_over_time.png",
    "frg_vs_beta.png",
    "frg_saturation_ratio.png",
    "lce_lag_curves.png",
    "rli_curves.png",
    "ali_over_time.png",
    "ali_hist.png",
]

REQUIRED_FIGURE_GLOBS = [
    "tierA_deltaJ_heatmap_t*.png",
    "tierA_deltaG_heatmap_t*.png",
    "tierA_delta_distributions_t*.png",
    "cai_waterfall_t*.png",
]


def _check_run_dir(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "report.md"
    scores_path = run_dir / "scores.json"
    figures_dir = run_dir / "figures"

    missing_files: list[str] = []
    if not scores_path.exists():
        missing_files.append("scores.json")
    if not report_path.exists():
        missing_files.append("report.md")
    if not figures_dir.is_dir():
        missing_files.append("figures/")

    missing_sections: list[str] = []
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        for section in REQUIRED_REPORT_SECTIONS:
            if section not in report_text:
                missing_sections.append(section)

    missing_figures: list[str] = []
    if figures_dir.is_dir():
        for name in REQUIRED_FIGURE_EXACT:
            if not (figures_dir / name).exists():
                missing_figures.append(name)
        for pattern in REQUIRED_FIGURE_GLOBS:
            if not list(figures_dir.glob(pattern)):
                missing_figures.append(pattern)

    ok = not (missing_files or missing_sections or missing_figures)
    return {
        "run_dir": str(run_dir),
        "ok": ok,
        "missing_files": missing_files,
        "missing_sections": missing_sections,
        "missing_figures": missing_figures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Paper-14 audit packet completeness.")
    parser.add_argument("--packets-root", required=True, help="Root containing packets/<sim>/seed-*/")
    parser.add_argument("--matrix-root", default="", help="Optional matrix root to verify version fingerprint artifacts.")
    parser.add_argument("--out-json", default="", help="Output JSON path (default: <packets-root>/verification.json)")
    parser.add_argument("--out-md", default="", help="Output Markdown path (default: <packets-root>/verification.md)")
    args = parser.parse_args()

    packets_root = Path(args.packets_root).resolve()
    run_dirs = sorted([p for p in packets_root.glob("*/seed-*") if p.is_dir()])
    if not run_dirs:
        raise SystemExit(f"No run directories found under {packets_root}")

    run_results = [_check_run_dir(run_dir) for run_dir in run_dirs]
    passed = sum(1 for r in run_results if r["ok"])
    failed = len(run_results) - passed

    matrix_checks: dict[str, Any] = {}
    if args.matrix_root:
        matrix_root = Path(args.matrix_root).resolve()
        fp_grid = matrix_root / "figures" / "version_fingerprint_grid.png"
        matrix_report = matrix_root / "report.md"
        matrix_checks = {
            "matrix_root": str(matrix_root),
            "version_fingerprint_grid_exists": fp_grid.exists(),
            "matrix_report_exists": matrix_report.exists(),
        }
        if matrix_report.exists():
            txt = matrix_report.read_text(encoding="utf-8")
            matrix_checks["has_version_fingerprints_section"] = "## Version Fingerprints" in txt
        else:
            matrix_checks["has_version_fingerprints_section"] = False

    summary = {
        "packets_root": str(packets_root),
        "run_count": len(run_results),
        "passed": passed,
        "failed": failed,
        "matrix_checks": matrix_checks,
        "runs": run_results,
    }

    out_json = Path(args.out_json).resolve() if args.out_json else packets_root / "verification.json"
    out_md = Path(args.out_md).resolve() if args.out_md else packets_root / "verification.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Read-Back Audit Packet Verification",
        "",
        f"- packets_root: `{packets_root}`",
        f"- run_count: `{len(run_results)}`",
        f"- passed: `{passed}`",
        f"- failed: `{failed}`",
        "",
        "| run_dir | ok | missing_files | missing_sections | missing_figures |",
        "|---|---|---|---|---|",
    ]
    for run in run_results:
        lines.append(
            f"| `{run['run_dir']}` | `{run['ok']}` | `{len(run['missing_files'])}` | `{len(run['missing_sections'])}` | `{len(run['missing_figures'])}` |"
        )
    if matrix_checks:
        lines.extend(
            [
                "",
                "## Matrix Checks",
                "",
                f"- version_fingerprint_grid_exists: `{matrix_checks.get('version_fingerprint_grid_exists')}`",
                f"- matrix_report_exists: `{matrix_checks.get('matrix_report_exists')}`",
                f"- has_version_fingerprints_section: `{matrix_checks.get('has_version_fingerprints_section')}`",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
