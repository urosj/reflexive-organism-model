#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _canonicalize_sim_script(sim_script: str, root_dir: Path) -> str:
    """Prefer canonical simulator paths while preserving legacy aliases."""
    sim_path = Path(sim_script)
    sim_name = sim_path.name
    canonical_rel = Path("simulations/active") / sim_name
    canonical_abs = root_dir / canonical_rel
    if not canonical_abs.exists():
        return sim_script

    if sim_script == sim_name or sim_script.startswith(f"./{sim_name}"):
        print(f"[DEPRECATED] simulator path '{sim_script}' is a legacy alias; use '{canonical_rel}'.")
        return str(canonical_rel)
    return sim_script


def _run(cmd: list[str], cwd: Path, log_path: Path) -> None:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee: {log_path}")


def _profile_extra_args(sim_script: str) -> str:
    sim_name = Path(sim_script).name
    if sim_name == "simulation-v14-cuda.py":
        return "--closure-softness 0.6 --spark-softness 0.08 --collapse-softness 0.5"
    if sim_name == "simulation-v15-cuda.py":
        return "--closure-mode soft --closure-softness 0.6 --spark-softness 0.08 --collapse-softness 0.5"
    if sim_name == "simulation-v16-cuda.py":
        return "--closure-mode soft --nonlocal-mode off --domain-mode fixed"
    return ""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_matching(src_dir: Path, dst_dir: Path, patterns: list[str]) -> list[str]:
    copied: list[str] = []
    for pattern in patterns:
        for path in sorted(src_dir.glob(pattern)):
            dst = dst_dir / path.name
            shutil.copy2(path, dst)
            copied.append(path.name)
    return copied


def _find_one(figures_dir: Path, pattern: str) -> str | None:
    matches = sorted(figures_dir.glob(pattern))
    if not matches:
        return None
    return matches[0].name


def _dominant_cai_stage(tiera_scores: dict[str, Any]) -> str:
    cai = tiera_scores.get("CAI") or {}
    stage_keys = [
        "DeltaJ_preClamp",
        "DeltaJ_postClamp",
        "DeltaK_raw",
        "DeltaK_reg",
        "DeltaG_preBlend",
        "DeltaG_postBlend",
    ]
    vals = [(k, float(cai.get(k) or 0.0)) for k in stage_keys]
    vals.sort(key=lambda kv: kv[1], reverse=True)
    return vals[0][0] if vals else "n/a"


def _flag_box(value: bool | None) -> str:
    if value is True:
        return "[x]"
    if value is False:
        return "[ ]"
    return "[-]"


def _write_run_report(
    *,
    packet_dir: Path,
    sim_script: str,
    seed: int,
    args_namespace: argparse.Namespace,
    tiera_scores: dict[str, Any],
    tierb_scores: dict[str, Any],
) -> None:
    figures_dir = packet_dir / "figures"
    target_step = int(tiera_scores.get("target_step", 0))
    cai_stage = _dominant_cai_stage(tiera_scores)
    flags = tierb_scores.get("flags") or {}
    thresholds = tierb_scores.get("thresholds") or {}
    decision = tierb_scores.get("decision")

    tiera_delta_j = tiera_scores.get("DeltaJ")
    tiera_delta_g = tiera_scores.get("DeltaG")
    tiera_delta_l = tiera_scores.get("DeltaL")

    lce_auc = tierb_scores.get("LCE_C_auc")
    frg1 = tierb_scores.get("FRG1_median")
    frg2 = tierb_scores.get("FRG2_median")
    frg_sat = tierb_scores.get("FRG_saturation_ratio")
    ocr_active = tierb_scores.get("OCR_active_median_rb")
    ocr_global = tierb_scores.get("OCR_global_median_rb")
    delta_rli_auc = tierb_scores.get("DeltaRLI_auc")
    delta_rli_mean = tierb_scores.get("DeltaRLI_mean")
    ali_active = tierb_scores.get("ALI_active_median_rb")
    ali_global = tierb_scores.get("ALI_global_median_rb")

    delta_j_map = _find_one(figures_dir, "tierA_deltaJ_heatmap_t*.png")
    delta_g_map = _find_one(figures_dir, "tierA_deltaG_heatmap_t*.png")
    delta_dist = _find_one(figures_dir, "tierA_delta_distributions_t*.png")
    cai_plot = _find_one(figures_dir, "cai_waterfall_t*.png")

    lines = [
        "# Read-Back Audit Report",
        "",
        "#### 0) Run metadata",
        f"- Version: `{sim_script}`",
        "- Git hash / build tag: `<TBD>`",
        "- Date/time: `<auto-from-filesystem>`",
        f"- Grid: `<{args_namespace.nx}, {args_namespace.ny}>`  dt: `<{args_namespace.dx}>`  steps: `<{args_namespace.steps}>`",
        f"- Seeds: `<{seed}>`",
        "- Stimulus / initial condition: `<default blobs.json + simulator defaults>`",
        f"- Modes: closure `<profile>`  other toggles `<{_profile_extra_args(sim_script)}>`",
        "",
        "#### 1) Parameter summary",
        "| Parameter | Value |",
        "|---|---:|",
        f"| beta_rb | `{tierb_scores.get('beta_rb_test')}` |",
        "| zeta_flux | `<sim-default>` |",
        "| blend_base | `<sim-default>` |",
        "| blend_rb (if applicable) | `<sim-default>` |",
        "| J clamp | `<sim-default>` |",
        "| K regularization / clamps | `<sim-default>` |",
        "| metric eigenvalue bounds | `<sim-default>` |",
        "| kappa_grad | `<sim-default>` |",
        f"| thresholds_json | `{args_namespace.thresholds_json}` |",
        "",
        "#### 2) Executive summary (1 screen)",
        f"**Decision:** `{decision}`",
        "**Why (auto-filled bullets):**",
        f"- OCR_active: `{ocr_active}`  OCR_global: `{ocr_global}`",
        f"- FRG (linear regime): `{frg1}`  FRG saturation ratio: `{frg_sat}`",
        f"- LCE_C: max `{tierb_scores.get('LCE_C_max')}`  AUC `{lce_auc}`  (mean ± std across seeds: `<single-seed run>`)",
        f"- DeltaRLI summary: mean `{delta_rli_mean}`, AUC `{delta_rli_auc}`",
        f"- CAI bottleneck stage: `{cai_stage}`",
        f"- ALI median: active `{ali_active}`, global `{ali_global}`",
        "",
        "**Decision rule (required, threshold-based):**",
        "- `Formal-only` if `max(FRG_linear_regime) < FRG_min` AND `LCE_auc < LCE_auc_min`.",
        "- `Weak-effective` if exactly one of `{FRG, LCE_auc}` is above threshold, or both are marginal/unstable.",
        "- `Effective` if `FRG >= FRG_min` AND `LCE_auc >= LCE_auc_min` AND reproducible across seeds (`seed_var <= seed_var_max`) AND not closure-dependent.",
        "",
        "**Flags (auto):**",
        f"- `{_flag_box(flags.get('RB_dominated'))}` RB dominated: OCR_active below threshold",
        f"- `{_flag_box(None if cai_plot is None else False)}` RB attenuated: CAI shows major drop at `<{cai_stage}>`",
        f"- `{_flag_box(flags.get('Saturation_clamp'))}` Saturation/clamp: FRG2/FRG1 below threshold",
        f"- `{_flag_box(flags.get('Redundant_channel'))}` Redundant channel: ALI above threshold",
        f"- `{_flag_box(flags.get('Non_reproducible'))}` Non-reproducible: seed variance above threshold",
        f"- `{_flag_box(flags.get('Closure_dependent'))}` Closure-dependent: efficacy disappears in closure-off mode",
        "",
        "#### 3) Tier A: Counterfactual snapshot results",
        f"- Snapshot times evaluated: `<{target_step}>`",
        "- A1 heatmaps:",
        f"  - `figures/{delta_j_map or 'tierA_deltaJ_heatmap_t{t}.png'}`",
        f"  - `figures/{delta_g_map or 'tierA_deltaG_heatmap_t{t}.png'}`",
        "- A2 distributions:",
        f"  - `figures/{delta_dist or 'tierA_delta_distributions_t{t}.png'}`",
        "",
        "**Tier A interpretation checklist:**",
        "- Does DeltaJ concentrate in the active mask (good) or appear only as rare spikes (suspicious)?",
        "- Is DeltaG nontrivial where DeltaJ is nontrivial?",
        f"- Core deltas: DeltaJ `{tiera_delta_j}` | DeltaG `{tiera_delta_g}` | DeltaL `{tiera_delta_l}`",
        "",
        "#### 4) OCR: Term dominance over time",
        "- `figures/ocr_term_energy_stack.png`",
        "- `figures/ocr_over_time.png`",
        "",
        "**OCR interpretation checklist:**",
        "- Is OCR_active consistently nontrivial or only intermittent spikes?",
        "- Is OCR_global negligible while OCR_active is meaningful (expected)?",
        "- Do changes in OCR correlate with activity bursts?",
        "",
        "#### 5) FRG: Gain and saturation",
        "- `figures/frg_vs_beta.png`",
        "- `figures/frg_saturation_ratio.png`",
        "",
        "**FRG interpretation checklist:**",
        "- Is there a clear linear regime where FRG is stable?",
        "- Does FRG plateau early (suggesting clamp/blend domination)?",
        f"- FRG1/FRG2 observed: `{frg1}` / `{frg2}`",
        "",
        "#### 6) Tier B: Lagged causal effects (experience accumulation)",
        "- `figures/lce_lag_curves.png`",
        "- Optional divergence maps:",
        "- `figures/lce_divergence_map_tau1.png`",
        "- `figures/lce_divergence_map_tau2.png`",
        "- `figures/lce_divergence_map_tau5.png`",
        "",
        "**Tier B interpretation checklist:**",
        "- Do small Tier A deltas accumulate into nontrivial LCE (weak-but-cumulative)?",
        "- Are LCE curves reproducible across seeds?",
        "- Does LCE remain nontrivial when closure is reduced/off?",
        "",
        "#### 7) RLI: Return / loop closure",
        "- `figures/rli_curves.png`",
        "",
        "**RLI interpretation checklist:**",
        "- Does read-back increase return mass into the fixed origin set A_t?",
        f"- DeltaRLI summary: mean `{delta_rli_mean}` | AUC `{delta_rli_auc}`",
        "",
        "#### 8) CAI: Attenuation bottleneck",
        f"- `figures/{cai_plot or 'cai_waterfall_t{t}.png'}`",
        "",
        "**CAI interpretation checklist:**",
        "- At which stage does Delta collapse most (J clamp / K reg / blend / eig clamp)?",
        f"- Dominant stage: `{cai_stage}`",
        "",
        "#### 9) ALI: Redundancy / alignment",
        "- `figures/ali_over_time.png`",
        "- `figures/ali_hist.png`",
        "",
        "**ALI interpretation checklist:**",
        "- Is ALI consistently near 1 (channel likely redundant)?",
        f"- Median ALI active/global: `{ali_active}` / `{ali_global}`",
        "",
        "#### 10) Cross-version fingerprints (when running experiment matrix)",
        "- `../../figures/version_fingerprint_grid.png`",
        "- Summary table (auto-filled) comparing:",
        "- median OCR_active, FRG linear regime, LCE AUC, DeltaRLI summary, CAI bottleneck, median ALI",
        "",
        "#### 11) Notes / anomalies (auto + manual)",
        "- Auto anomalies: `<from flags + thresholds below>`",
        "- Manual notes: `<none>`",
        "",
        "## Thresholds",
    ]
    for key in sorted(thresholds.keys()):
        lines.append(f"- `{key} = {thresholds[key]}`")

    (packet_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _build_version_fingerprint_grid(
    *,
    out_path: Path,
    representative_packets: dict[str, Path],
) -> None:
    rows = [
        ("B2 OCR", "ocr_over_time.png"),
        ("C1 FRG", "frg_vs_beta.png"),
        ("D1 LCE", "lce_lag_curves.png"),
        ("F1 CAI", "cai_waterfall_t*.png"),
        ("G1 ALI", "ali_over_time.png"),
    ]
    versions = sorted(representative_packets.keys())
    if not versions:
        return

    nrows = len(rows)
    ncols = len(versions)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.2 * ncols, 2.6 * nrows), dpi=120)
    axes_arr = np.asarray(axes, dtype=object)
    if axes_arr.ndim == 0:
        axes_arr = axes_arr.reshape(1, 1)
    elif axes_arr.ndim == 1:
        if nrows == 1:
            axes_arr = axes_arr.reshape(1, ncols)
        else:
            axes_arr = axes_arr.reshape(nrows, 1)

    for col, version in enumerate(versions):
        packet_dir = representative_packets[version]
        fig_dir = packet_dir / "figures"
        for row_idx, (row_label, pattern) in enumerate(rows):
            ax = axes_arr[row_idx, col]
            img_path: Path | None
            if "*" in pattern:
                matches = sorted(fig_dir.glob(pattern))
                img_path = matches[0] if matches else None
            else:
                p = fig_dir / pattern
                img_path = p if p.exists() else None

            if img_path is not None:
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.axis("off")
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(True)

            if row_idx == 0:
                ax.set_title(version, fontsize=9)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _collect_summary(packets_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Path]]:
    by_version: dict[str, list[dict[str, Any]]] = {}
    representative_packets: dict[str, Path] = {}
    for packet_dir in sorted(packets_root.glob("*/seed-*")):
        scores_path = packet_dir / "scores.json"
        if not scores_path.exists():
            continue
        scores = _load_json(scores_path)
        version = str(scores.get("sim_script") or packet_dir.parent.name)
        by_version.setdefault(version, []).append(scores)
        representative_packets.setdefault(version, packet_dir)
    return by_version, representative_packets


def _median(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _write_matrix_report(
    *,
    matrix_root: Path,
    by_version: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Read-Back Cross-Version Matrix Report",
        "",
        "## Version Fingerprints",
        "",
        "- Canonical fingerprint grid: `figures/version_fingerprint_grid.png`",
        "",
        "| version | runs | OCR_active_median | FRG1_median | LCE_C_auc_median | DeltaRLI_auc_median | ALI_active_median | decision_mode |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for version in sorted(by_version.keys()):
        rows = by_version[version]
        ocr = _median([r.get("OCR_active_median_rb") for r in rows])
        frg = _median([r.get("FRG1_median") for r in rows])
        lce = _median([r.get("LCE_C_auc") for r in rows])
        drli = _median([r.get("DeltaRLI_auc") for r in rows])
        ali = _median([r.get("ALI_active_median_rb") for r in rows])
        decisions = [str(r.get("decision")) for r in rows if r.get("decision") is not None]
        decision_mode = max(set(decisions), key=decisions.count) if decisions else "n/a"
        lines.append(
            f"| `{version}` | {len(rows)} | {ocr} | {frg} | {lce} | {drli} | {ali} | `{decision_mode}` |"
        )

    lines.extend(
        [
            "",
            "## Artifact Index",
            "",
            "| version | seed | packet_dir |",
            "|---|---:|---|",
        ]
    )
    for version in sorted(by_version.keys()):
        for row in by_version[version]:
            seed = row.get("seed")
            packet_dir = matrix_root / "packets" / Path(version).stem / f"seed-{seed}"
            lines.append(f"| `{version}` | `{seed}` | `{packet_dir}` |")

    (matrix_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Iteration-6 cross-version read-back audit matrix.")
    parser.add_argument("--python-bin", default=str(Path("venv/bin/python")))
    parser.add_argument(
        "--sim-scripts",
        default="simulations/active/simulation-v12-cuda.py,simulations/active/simulation-v13-cuda.py,simulations/active/simulation-v14-cuda.py,simulations/active/simulation-v15-cuda.py,simulations/active/simulation-v16-cuda.py",
        help="Comma-separated simulation scripts.",
    )
    parser.add_argument("--seeds", default="1", help="Comma-separated seeds.")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=12, help="Tier B steps.")
    parser.add_argument("--tiera-steps", type=int, default=2)
    parser.add_argument("--snapshot-interval", type=int, default=100)
    parser.add_argument("--storage-mode", default="memory")
    parser.add_argument("--beta-rb", type=float, default=1.0)
    parser.add_argument("--frg-beta2", type=float, default=2.0)
    parser.add_argument("--run-negative-control", action="store_true")
    parser.add_argument("--negctrl-lam-scale-per-beta", type=float, default=0.10)
    parser.add_argument("--out-dir", default="outputs/readback-baseline/tier6-matrix")
    parser.add_argument("--thresholds-json", default="experiments/scripts/readback_thresholds.json")
    parser.add_argument("--global-sim-extra-args", default="")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent.parent
    matrix_root = (root_dir / args.out_dir).resolve()
    tiera_out = Path(args.out_dir) / "tierA"
    tierb_out = Path(args.out_dir) / "tierB"
    packets_root = matrix_root / "packets"
    packets_root.mkdir(parents=True, exist_ok=True)
    (matrix_root / "logs").mkdir(parents=True, exist_ok=True)

    sim_scripts = [
        _canonicalize_sim_script(s.strip(), root_dir)
        for s in args.sim_scripts.split(",")
        if s.strip()
    ]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for sim_script in sim_scripts:
        sim_stem = Path(sim_script).stem
        profile_args = _profile_extra_args(sim_script)
        merged_extra = " ".join(
            [p for p in [profile_args, args.global_sim_extra_args.strip()] if p]
        ).strip()
        for seed in seeds:
            run_tag = f"{sim_stem}-seed-{seed}"

            tiera_cmd = [
                args.python_bin,
                str(root_dir / "experiments/scripts/run_readback_tierA.py"),
                "--sim-script",
                sim_script,
                "--mode",
                "A2",
                "--seed",
                str(seed),
                "--nx",
                str(args.nx),
                "--ny",
                str(args.ny),
                "--dx",
                str(args.dx),
                "--snapshot-interval",
                str(args.snapshot_interval),
                "--storage-mode",
                args.storage_mode,
                "--steps",
                str(args.tiera_steps),
                "--beta-rb",
                str(args.beta_rb),
                "--out-dir",
                str(tiera_out),
                "--sim-extra-args",
                merged_extra,
            ]
            _run(
                tiera_cmd,
                cwd=root_dir,
                log_path=matrix_root / "logs" / f"{run_tag}-tierA.log",
            )

            tierb_cmd = [
                args.python_bin,
                str(root_dir / "experiments/scripts/run_readback_tierB.py"),
                "--sim-script",
                sim_script,
                "--seed",
                str(seed),
                "--nx",
                str(args.nx),
                "--ny",
                str(args.ny),
                "--dx",
                str(args.dx),
                "--snapshot-interval",
                str(args.snapshot_interval),
                "--storage-mode",
                args.storage_mode,
                "--steps",
                str(args.steps),
                "--beta-rb",
                str(args.beta_rb),
                "--frg-beta2",
                str(args.frg_beta2),
                "--out-dir",
                str(tierb_out),
                "--thresholds-json",
                str(args.thresholds_json),
                "--sim-extra-args",
                merged_extra,
            ]
            if args.run_negative_control:
                tierb_cmd.extend(
                    [
                        "--run-negative-control",
                        "--negctrl-lam-scale-per-beta",
                        str(args.negctrl_lam_scale_per_beta),
                    ]
                )
            _run(
                tierb_cmd,
                cwd=root_dir,
                log_path=matrix_root / "logs" / f"{run_tag}-tierB.log",
            )

            tiera_run_dir = matrix_root / "tierA" / sim_stem / f"a2-seed-{seed}"
            tierb_run_dir = matrix_root / "tierB" / sim_stem / f"seed-{seed}"
            if not tiera_run_dir.exists() or not tierb_run_dir.exists():
                raise RuntimeError(f"Expected run outputs missing for {run_tag}")

            packet_dir = packets_root / sim_stem / f"seed-{seed}"
            figures_dir = packet_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            # Core artifacts from Tier B.
            for name in ("scores.json", "series.npz", "contrast.json", "contrast.md"):
                src = tierb_run_dir / name
                if src.exists():
                    shutil.copy2(src, packet_dir / name)
            shutil.copy2(tierb_run_dir / "report.md", packet_dir / "tierB_report.md")
            shutil.copy2(tiera_run_dir / "scores.json", packet_dir / "tierA_scores.json")
            shutil.copy2(tiera_run_dir / "report.md", packet_dir / "tierA_report.md")

            # Canonical figures: Tier B + Tier A (A/F sets).
            _copy_matching(tierb_run_dir / "figures", figures_dir, ["*.png"])
            _copy_matching(
                tiera_run_dir / "figures",
                figures_dir,
                [
                    "tierA_deltaJ_heatmap_t*.png",
                    "tierA_deltaG_heatmap_t*.png",
                    "tierA_delta_distributions_t*.png",
                    "cai_waterfall_t*.png",
                ],
            )

            tiera_scores = _load_json(packet_dir / "tierA_scores.json")
            tierb_scores = _load_json(packet_dir / "scores.json")
            _write_run_report(
                packet_dir=packet_dir,
                sim_script=sim_script,
                seed=seed,
                args_namespace=args,
                tiera_scores=tiera_scores,
                tierb_scores=tierb_scores,
            )

    by_version, representative_packets = _collect_summary(packets_root)
    _write_matrix_report(matrix_root=matrix_root, by_version=by_version)
    _build_version_fingerprint_grid(
        out_path=matrix_root / "figures" / "version_fingerprint_grid.png",
        representative_packets=representative_packets,
    )

    if not args.skip_verify:
        verify_cmd = [
            args.python_bin,
            str(root_dir / "experiments/scripts/verify_readback_audit_packet.py"),
            "--packets-root",
            str(packets_root),
            "--matrix-root",
            str(matrix_root),
            "--out-json",
            str(matrix_root / "verification.json"),
            "--out-md",
            str(matrix_root / "verification.md"),
        ]
        _run(
            verify_cmd,
            cwd=root_dir,
            log_path=matrix_root / "logs" / "verify.log",
        )

    print(f"[OK] Iteration-6 matrix complete: {matrix_root}")
    print(f"[OK] Packets root: {packets_root}")
    print(f"[OK] Matrix report: {matrix_root / 'report.md'}")
    print(f"[OK] Fingerprint grid: {matrix_root / 'figures/version_fingerprint_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
