#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HOOK_STREAM_RE = re.compile(r"^\[TELEMETRY\]\s+hook_stream=(?P<path>.+)$", re.MULTILINE)
FIELD_DUMP_RE = re.compile(r"^\[TELEMETRY\]\s+field_dump=(?P<path>.+)$", re.MULTILINE)


def _canonicalize_sim_script(sim_script: str, root_dir: Path) -> str:
    """Prefer canonical simulator paths while preserving legacy aliases."""
    sim_path = Path(sim_script)
    sim_name = sim_path.name
    canonical_rel = Path("simulations/active") / sim_name
    canonical_abs = root_dir / canonical_rel
    if not canonical_abs.exists():
        return sim_script

    # Legacy alias usage: root-level simulator path or bare filename.
    if sim_script == sim_name or sim_script.startswith(f"./{sim_name}"):
        print(f"[DEPRECATED] simulator path '{sim_script}' is a legacy alias; use '{canonical_rel}'.")
        return str(canonical_rel)
    return sim_script


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _index_records(rows: list[dict[str, Any]]) -> Dict[Tuple[int, str], dict[str, Any]]:
    """Index by (step, stage), keeping the last record per stage/step."""
    out: Dict[Tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("record_type") == "hook_stream_metadata":
            continue
        if "step" not in row or "stage" not in row:
            continue
        step = int(row["step"])
        stage = str(row["stage"])
        out[(step, stage)] = row
    return out


def _maybe_float(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_paths(run_stdout: str, root_dir: Path) -> tuple[Path, Path | None]:
    hook_match = list(HOOK_STREAM_RE.finditer(run_stdout))
    if not hook_match:
        raise RuntimeError("No hook_stream path found in run output.")
    hook_path = Path(hook_match[-1].group("path").strip())
    if not hook_path.is_absolute():
        hook_path = (root_dir / hook_path).resolve()

    field_match = list(FIELD_DUMP_RE.finditer(run_stdout))
    field_path: Path | None = None
    if field_match:
        field_path = Path(field_match[-1].group("path").strip())
        if not field_path.is_absolute():
            field_path = (root_dir / field_path).resolve()
    return hook_path, field_path


def _run_branch(
    *,
    root_dir: Path,
    python_bin: str,
    sim_script: str,
    sim_args: list[str],
    beta_rb: float,
    branch_out_dir: Path,
) -> dict[str, Path]:
    env = os.environ.copy()
    env["RC_READBACK_HOOKS"] = "1"
    env["RC_READBACK_FIELD_DUMP"] = "1"
    env["RC_READBACK_FIELD_INTERVAL"] = "1"
    env["RC_READBACK_FIELD_DOWNSAMPLE"] = env.get("RC_READBACK_FIELD_DOWNSAMPLE", "2")
    env["RC_READBACK_BETA_RB"] = f"{beta_rb:.12g}"

    cmd = [python_bin, sim_script, *sim_args]
    proc = subprocess.run(
        cmd,
        cwd=root_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    branch_out_dir.mkdir(parents=True, exist_ok=True)
    run_log = branch_out_dir / "run.log"
    run_log.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed for beta_rb={beta_rb}: see {run_log}")

    telemetry_src, fields_src = _extract_paths(proc.stdout, root_dir)
    telemetry_dst = branch_out_dir / "telemetry.jsonl"
    shutil.copy2(telemetry_src, telemetry_dst)

    out_paths = {
        "run_log": run_log,
        "telemetry": telemetry_dst,
    }
    if fields_src is not None and fields_src.exists():
        fields_dst = branch_out_dir / "fields.npz"
        shutil.copy2(fields_src, fields_dst)
        out_paths["fields"] = fields_dst
    return out_paths


def _load_field_at_step(npz_path: Path, key: str, step: int) -> np.ndarray | None:
    if not npz_path.exists():
        return None
    with np.load(npz_path) as data:
        if key not in data:
            return None
        step_key = f"{key}__steps"
        if step_key not in data:
            return None
        steps = data[step_key]
        matches = np.where(steps == step)[0]
        if matches.size == 0:
            return None
        return np.asarray(data[key][int(matches[-1])], dtype=np.float32)


def _compute_active_mask(baseline_j: np.ndarray, top_q_pct: float) -> np.ndarray:
    q = max(0.01, min(99.99, float(top_q_pct)))
    threshold = np.percentile(baseline_j, 100.0 - q)
    return baseline_j >= threshold


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b)


def _save_heatmap(
    array: np.ndarray,
    mask: np.ndarray | None,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    im = ax.imshow(array, cmap="magma", origin="lower")
    if mask is not None:
        ax.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.4)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_delta_distributions(
    delta_j_map: np.ndarray | None,
    delta_g_map: np.ndarray | None,
    mask: np.ndarray | None,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
    if delta_j_map is not None:
        axes[0].hist(delta_j_map.ravel(), bins=40, alpha=0.6, label="global", density=True)
        if mask is not None:
            axes[0].hist(delta_j_map[mask], bins=40, alpha=0.6, label="mask", density=True)
        axes[0].set_title("DeltaJ distribution")
        axes[0].legend()
    if delta_g_map is not None:
        axes[1].hist(delta_g_map.ravel(), bins=40, alpha=0.6, label="global", density=True)
        if mask is not None:
            axes[1].hist(delta_g_map[mask], bins=40, alpha=0.6, label="mask", density=True)
        axes[1].set_title("DeltaG distribution")
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_cai_waterfall(cai: dict[str, float | None], out_path: Path) -> None:
    labels = [
        "DeltaJ_preClamp",
        "DeltaJ_postClamp",
        "DeltaK_raw",
        "DeltaK_reg",
        "DeltaG_preBlend",
        "DeltaG_postBlend",
    ]
    vals = [float(cai.get(k) or 0.0) for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.bar(labels, vals, color="#3b82f6")
    ax.set_ylabel("Counterfactual delta")
    ax.set_title("CAI Stage Deltas")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_dashboard(
    *,
    mode: str,
    step: int,
    delta_j: float | None,
    delta_g: float | None,
    delta_l: float | None,
    cai: dict[str, float | None],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    names = ["DeltaJ", "DeltaG", "DeltaL"]
    vals = [float(delta_j or 0.0), float(delta_g or 0.0), float(delta_l or 0.0)]
    axes[0].bar(names, vals, color=["#2563eb", "#dc2626", "#16a34a"])
    axes[0].set_title(f"{mode} @ step={step}")
    axes[0].set_ylabel("Effect size")

    cai_names = ["preJ", "postJ", "Kraw", "Kreg", "preG", "postG"]
    cai_vals = [
        float(cai.get("DeltaJ_preClamp") or 0.0),
        float(cai.get("DeltaJ_postClamp") or 0.0),
        float(cai.get("DeltaK_raw") or 0.0),
        float(cai.get("DeltaK_reg") or 0.0),
        float(cai.get("DeltaG_preBlend") or 0.0),
        float(cai.get("DeltaG_postBlend") or 0.0),
    ]
    axes[1].plot(cai_names, cai_vals, marker="o", color="#0f766e")
    axes[1].set_title("CAI profile")
    axes[1].set_ylabel("Delta")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _cai_attenuation_ratios(cai: dict[str, float | None]) -> dict[str, float | None]:
    def _ratio(numer_key: str, denom_key: str) -> float | None:
        numer = cai.get(numer_key)
        denom = cai.get(denom_key)
        if numer is None or denom is None:
            return None
        denom_f = float(denom)
        if abs(denom_f) <= 1e-12:
            return None
        return float(numer) / denom_f

    return {
        "J_post_over_pre": _ratio("DeltaJ_postClamp", "DeltaJ_preClamp"),
        "K_reg_over_raw": _ratio("DeltaK_reg", "DeltaK_raw"),
        "G_post_over_pre": _ratio("DeltaG_postBlend", "DeltaG_preBlend"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Paper-14 Tier A counterfactual diagnostics.")
    parser.add_argument("--sim-script", default="simulations/active/simulation-v16-cuda.py")
    parser.add_argument("--python-bin", default=str(Path("venv/bin/python")))
    parser.add_argument("--mode", choices=["A1", "A2"], default="A2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--snapshot-interval", type=int, default=100)
    parser.add_argument("--storage-mode", default="memory")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--target-step", type=int, default=None)
    parser.add_argument("--beta-rb", type=float, default=1.0)
    parser.add_argument("--top-q-pct", type=float, default=5.0)
    parser.add_argument("--out-dir", default="outputs/readback-baseline/tierA-smoke")
    parser.add_argument("--sim-extra-args", default="")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent.parent
    args.sim_script = _canonicalize_sim_script(args.sim_script, root_dir)
    out_root = (root_dir / args.out_dir / Path(args.sim_script).stem / f"{args.mode.lower()}-seed-{args.seed}").resolve()
    baseline_dir = out_root / "baseline"
    rb_dir = out_root / "rb"
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    steps = args.steps
    if steps is None:
        steps = 1 if args.mode == "A1" else 2

    sim_args = [
        "--headless",
        "--headless-steps",
        str(steps),
        "--nx",
        str(args.nx),
        "--ny",
        str(args.ny),
        "--dx",
        str(args.dx),
        "--seed",
        str(args.seed),
        "--snapshot-interval",
        str(args.snapshot_interval),
        "--storage-mode",
        args.storage_mode,
    ]

    extra_args = shlex.split(args.sim_extra_args) if args.sim_extra_args else []
    sim_args.extend(extra_args)

    baseline_paths = _run_branch(
        root_dir=root_dir,
        python_bin=args.python_bin,
        sim_script=args.sim_script,
        sim_args=sim_args,
        beta_rb=0.0,
        branch_out_dir=baseline_dir,
    )
    rb_paths = _run_branch(
        root_dir=root_dir,
        python_bin=args.python_bin,
        sim_script=args.sim_script,
        sim_args=sim_args,
        beta_rb=args.beta_rb,
        branch_out_dir=rb_dir,
    )

    base_rows = _read_jsonl(baseline_paths["telemetry"])
    rb_rows = _read_jsonl(rb_paths["telemetry"])
    base_idx = _index_records(base_rows)
    rb_idx = _index_records(rb_rows)

    if args.target_step is not None:
        step = int(args.target_step)
    else:
        common_steps = sorted(
            set(s for s, _ in base_idx.keys()).intersection(s for s, _ in rb_idx.keys())
        )
        if not common_steps:
            raise RuntimeError("No common steps between baseline and rb branches.")
        step = common_steps[-1]

    def row(branch: Dict[Tuple[int, str], dict[str, Any]], stage: str) -> dict[str, Any] | None:
        return branch.get((step, stage))

    base_post_j_pre = row(base_idx, "post_J_preclamp")
    rb_post_j_pre = row(rb_idx, "post_J_preclamp")
    base_post_j = row(base_idx, "post_J_postclamp")
    rb_post_j = row(rb_idx, "post_J_postclamp")
    base_post_phi = row(base_idx, "post_phi")
    rb_post_phi = row(rb_idx, "post_phi")
    base_post_g_pre = row(base_idx, "post_g_preblend")
    rb_post_g_pre = row(rb_idx, "post_g_preblend")
    base_post_g = row(base_idx, "post_g_postblend")
    rb_post_g = row(rb_idx, "post_g_postblend")
    base_post_k_raw = row(base_idx, "post_K_raw")
    rb_post_k_raw = row(rb_idx, "post_K_raw")
    base_post_k_reg = row(base_idx, "post_K_regularized")
    rb_post_k_reg = row(rb_idx, "post_K_regularized")

    delta_j = _delta(_maybe_float(base_post_j, "J_rms"), _maybe_float(rb_post_j, "J_rms"))
    delta_g = _delta(_maybe_float(base_post_g_pre, "g_new_rms"), _maybe_float(rb_post_g_pre, "g_new_rms"))
    delta_l = _delta(_maybe_float(base_post_phi, "L_proxy_scalar"), _maybe_float(rb_post_phi, "L_proxy_scalar"))

    cai = {
        "DeltaJ_preClamp": _delta(_maybe_float(base_post_j_pre, "J_rms_pre"), _maybe_float(rb_post_j_pre, "J_rms_pre")),
        "DeltaJ_postClamp": _delta(_maybe_float(base_post_j, "J_rms"), _maybe_float(rb_post_j, "J_rms")),
        "DeltaK_raw": _delta(_maybe_float(base_post_k_raw, "T_rb_rms"), _maybe_float(rb_post_k_raw, "T_rb_rms")),
        "DeltaK_reg": _delta(_maybe_float(base_post_k_reg, "detK_mean"), _maybe_float(rb_post_k_reg, "detK_mean")),
        "DeltaG_preBlend": _delta(_maybe_float(base_post_g_pre, "g_new_rms"), _maybe_float(rb_post_g_pre, "g_new_rms")),
        "DeltaG_postBlend": _delta(_maybe_float(base_post_g, "detg_mean"), _maybe_float(rb_post_g, "detg_mean")),
    }

    baseline_fields = baseline_paths.get("fields")
    rb_fields = rb_paths.get("fields")
    delta_j_map = None
    delta_g_map = None
    active_mask = None
    if baseline_fields and rb_fields:
        base_j = _load_field_at_step(baseline_fields, "post_J_postclamp__J_mag_field", step)
        rb_j = _load_field_at_step(rb_fields, "post_J_postclamp__J_mag_field", step)
        if base_j is not None and rb_j is not None:
            delta_j_map = np.abs(rb_j - base_j).astype(np.float32)
            active_mask = _compute_active_mask(base_j, args.top_q_pct)

        base_g = _load_field_at_step(baseline_fields, "post_g_postblend__G_frob_field", step)
        rb_g = _load_field_at_step(rb_fields, "post_g_postblend__G_frob_field", step)
        if base_g is not None and rb_g is not None:
            delta_g_map = np.abs(rb_g - base_g).astype(np.float32)

    if delta_j_map is not None:
        _save_heatmap(
            delta_j_map,
            active_mask,
            f"DeltaJ_map @ step {step}",
            figures_dir / f"tierA_deltaJ_heatmap_t{step}.png",
        )
    if delta_g_map is not None:
        _save_heatmap(
            delta_g_map,
            active_mask,
            f"DeltaG_map @ step {step}",
            figures_dir / f"tierA_deltaG_heatmap_t{step}.png",
        )
    if delta_j_map is not None or delta_g_map is not None:
        _save_delta_distributions(
            delta_j_map,
            delta_g_map,
            active_mask,
            figures_dir / f"tierA_delta_distributions_t{step}.png",
        )

    _save_cai_waterfall(cai, figures_dir / f"cai_waterfall_t{step}.png")
    _save_dashboard(
        mode=args.mode,
        step=step,
        delta_j=delta_j,
        delta_g=delta_g,
        delta_l=delta_l,
        cai=cai,
        out_path=figures_dir / f"tierA_operator_dashboard_t{step}.png",
    )
    cai_ratios = _cai_attenuation_ratios(cai)

    masked_delta_j_mean = None
    masked_delta_g_mean = None
    global_delta_j_mean = None
    global_delta_g_mean = None
    if delta_j_map is not None:
        global_delta_j_mean = float(np.mean(delta_j_map))
        if active_mask is not None:
            masked_delta_j_mean = float(np.mean(delta_j_map[active_mask]))
    if delta_g_map is not None:
        global_delta_g_mean = float(np.mean(delta_g_map))
        if active_mask is not None:
            masked_delta_g_mean = float(np.mean(delta_g_map[active_mask]))

    scores = {
        "mode": args.mode,
        "sim_script": args.sim_script,
        "seed": args.seed,
        "steps": steps,
        "target_step": step,
        "beta_rb_baseline": 0.0,
        "beta_rb_test": float(args.beta_rb),
        "DeltaJ": delta_j,
        "DeltaG": delta_g,
        "DeltaL": delta_l,
        "mask_top_q_pct": float(args.top_q_pct),
        "global_DeltaJ_map_mean": global_delta_j_mean,
        "global_DeltaG_map_mean": global_delta_g_mean,
        "masked_DeltaJ_map_mean": masked_delta_j_mean,
        "masked_DeltaG_map_mean": masked_delta_g_mean,
        "CAI": cai,
        "CAI_ratios": cai_ratios,
        "artifacts": {
            "baseline_telemetry": str(baseline_paths["telemetry"]),
            "rb_telemetry": str(rb_paths["telemetry"]),
            "baseline_fields": str(baseline_paths.get("fields", "")),
            "rb_fields": str(rb_paths.get("fields", "")),
            "figures_dir": str(figures_dir),
        },
    }
    scores_path = out_root / "scores.json"
    scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    report_lines = [
        "# Tier A Operator Sensitivity Report",
        "",
        f"- mode: `{args.mode}`",
        f"- sim: `{args.sim_script}`",
        f"- seed: `{args.seed}`",
        f"- target step: `{step}`",
        f"- beta_rb baseline/test: `0.0 / {args.beta_rb}`",
        "",
        "## Core deltas",
        "",
        f"- DeltaJ: `{delta_j}`",
        f"- DeltaG: `{delta_g}`",
        f"- DeltaL: `{delta_l}`",
        "",
        "## CAI",
        "",
        *(f"- {k}: `{v}`" for k, v in cai.items()),
        "",
        "## CAI ratios",
        "",
        *(f"- {k}: `{v}`" for k, v in cai_ratios.items()),
        "",
        "## Figures",
        "",
        f"- `figures/tierA_operator_dashboard_t{step}.png`",
        f"- `figures/cai_waterfall_t{step}.png`",
        f"- `figures/tierA_deltaJ_heatmap_t{step}.png`",
        f"- `figures/tierA_deltaG_heatmap_t{step}.png`",
        f"- `figures/tierA_delta_distributions_t{step}.png`",
        "",
    ]
    (out_root / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[OK] Tier A run complete: {out_root}")
    print(f"[OK] scores.json: {scores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
