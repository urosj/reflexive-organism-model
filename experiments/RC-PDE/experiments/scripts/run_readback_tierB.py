#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HOOK_STREAM_RE = re.compile(r"^\[TELEMETRY\]\s+hook_stream=(?P<path>.+)$", re.MULTILINE)
FIELD_DUMP_RE = re.compile(r"^\[TELEMETRY\]\s+field_dump=(?P<path>.+)$", re.MULTILINE)
EPS = 1e-12

DEFAULT_THRESHOLDS = {
    "OCR_active_min": 0.05,
    "FRG_min": 0.005,
    "FRG_saturation_ratio_min": 0.6,
    "ALI_redundant_min": 0.90,
    "LCE_auc_min": 0.01,
    "seed_var_max": 0.10,
}


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _run_branch(
    *,
    root_dir: Path,
    python_bin: str,
    sim_script: str,
    sim_args: list[str],
    beta_rb: float,
    negctrl_lam_scale: float,
    branch_out_dir: Path,
    field_interval: int,
    field_downsample: int,
) -> dict[str, Path]:
    env = os.environ.copy()
    env["RC_READBACK_HOOKS"] = "1"
    env["RC_READBACK_FIELD_DUMP"] = "1"
    env["RC_READBACK_FIELD_INTERVAL"] = str(max(1, int(field_interval)))
    env["RC_READBACK_FIELD_DOWNSAMPLE"] = str(max(1, int(field_downsample)))
    env["RC_READBACK_BETA_RB"] = f"{beta_rb:.12g}"
    env["RC_NEGCTRL_LAM_SCALE"] = f"{negctrl_lam_scale:.12g}"

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


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr), dtype=np.float64)))


def _common_steps(*series: dict[int, Any]) -> list[int]:
    if not series:
        return []
    keys = set(series[0].keys())
    for s in series[1:]:
        keys &= set(s.keys())
    return sorted(keys)


def _load_stage_series(npz_path: Path, key: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    with np.load(npz_path) as data:
        if key not in data:
            return out
        step_key = f"{key}__steps"
        if step_key not in data:
            return out
        frames = np.asarray(data[key], dtype=np.float32)
        steps = np.asarray(data[step_key], dtype=np.int64)
        for idx, step in enumerate(steps):
            out[int(step)] = frames[idx]
    return out


def _load_stage_scalar_series(rows: list[dict[str, Any]], stage: str, key: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in rows:
        if row.get("record_type") == "hook_stream_metadata":
            continue
        if row.get("stage") != stage:
            continue
        if "step" not in row or key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        try:
            out[int(row["step"])] = float(value)
        except Exception:
            continue
    return out


def _select_common_step(*series: dict[int, Any]) -> int | None:
    common = _common_steps(*series)
    if not common:
        return None
    return int(common[-1])


def _safe_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(abs(a - b))


def _safe_value(series: dict[int, float], step: int) -> float | None:
    if step not in series:
        return None
    return float(series[step])


def _load_post_k_term_series(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_step: dict[int, dict[str, float]] = {}
    for row in rows:
        if row.get("record_type") == "hook_stream_metadata":
            continue
        if row.get("stage") != "post_K_raw":
            continue
        if "step" not in row:
            continue
        step = int(row["step"])
        cur = by_step.setdefault(step, {})
        for key in ("T_den_rms", "T_grad_rms", "T_id_rms", "T_rb_rms"):
            val = row.get(key)
            if val is None:
                continue
            try:
                cur[key] = float(val)
            except Exception:
                pass
    steps = sorted(by_step.keys())
    if not steps:
        empty_i = np.asarray([], dtype=np.int64)
        empty_f = np.asarray([], dtype=np.float64)
        return empty_i, empty_i, empty_f, empty_f, empty_f, empty_f
    first = steps[0]
    tau = np.asarray([s - first for s in steps], dtype=np.int64)
    t_den = np.asarray([by_step[s].get("T_den_rms", 0.0) for s in steps], dtype=np.float64)
    t_grad = np.asarray([by_step[s].get("T_grad_rms", 0.0) for s in steps], dtype=np.float64)
    t_id = np.asarray([by_step[s].get("T_id_rms", 0.0) for s in steps], dtype=np.float64)
    t_rb = np.asarray([by_step[s].get("T_rb_rms", 0.0) for s in steps], dtype=np.float64)
    return np.asarray(steps, dtype=np.int64), tau, t_den, t_grad, t_id, t_rb


def _compute_lce_series(
    baseline: dict[int, np.ndarray],
    rb: dict[int, np.ndarray],
    *,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_steps = _common_steps(baseline, rb)
    if not common_steps:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    first = int(common_steps[0])
    steps = np.asarray(common_steps, dtype=np.int64)
    tau = steps - first
    vals = []
    for step in steps:
        b = baseline[int(step)]
        r = rb[int(step)]
        diff = _rms(r - b)
        denom = _rms(b) + eps
        vals.append(diff / denom)
    return steps, tau, np.asarray(vals, dtype=np.float64)


def _safe_auc(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size == 0 or y.size == 0:
        return None
    if x.size == 1:
        return float(y[0])
    return float(np.trapezoid(y, x))


def _top_mask(frame: np.ndarray, q_pct: float) -> np.ndarray:
    q = max(0.01, min(99.99, float(q_pct)))
    threshold = np.percentile(frame, 100.0 - q)
    return frame >= threshold


def _compute_ocr_global_series(t_rb: dict[int, float], t_grad: dict[int, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = _common_steps(t_rb, t_grad)
    if not steps:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    first = steps[0]
    vals = [(t_rb[s] / (t_grad[s] + EPS)) for s in steps]
    return np.asarray(steps, dtype=np.int64), np.asarray([s - first for s in steps], dtype=np.int64), np.asarray(vals, dtype=np.float64)


def _compute_ocr_active_series(
    baseline_j: dict[int, np.ndarray],
    branch_rb_trace: dict[int, np.ndarray],
    branch_grad_trace: dict[int, np.ndarray],
    *,
    active_q_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = _common_steps(baseline_j, branch_rb_trace, branch_grad_trace)
    if not steps:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    first = steps[0]
    vals = []
    for step in steps:
        mask = _top_mask(np.abs(baseline_j[step]), active_q_pct)
        rb_val = _rms(branch_rb_trace[step][mask])
        grad_val = _rms(branch_grad_trace[step][mask])
        vals.append(rb_val / (grad_val + EPS))
    return np.asarray(steps, dtype=np.int64), np.asarray([s - first for s in steps], dtype=np.int64), np.asarray(vals, dtype=np.float64)


def _compute_frg_series(
    baseline_j: dict[int, np.ndarray],
    branch_j: dict[int, np.ndarray],
    *,
    beta_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(beta_delta) <= EPS:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    steps = _common_steps(baseline_j, branch_j)
    if not steps:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    first = steps[0]
    vals = []
    for step in steps:
        base = baseline_j[step]
        br = branch_j[step]
        num = _rms(br - base)
        den = abs(beta_delta) * (_rms(base) + EPS)
        vals.append(num / den)
    return np.asarray(steps, dtype=np.int64), np.asarray([s - first for s in steps], dtype=np.int64), np.asarray(vals, dtype=np.float64)


def _compute_rli_series(
    baseline_j: dict[int, np.ndarray],
    rb_j: dict[int, np.ndarray],
    *,
    origin_q_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None, float | None]:
    steps = _common_steps(baseline_j, rb_j)
    if not steps:
        empty = np.asarray([], dtype=np.float64)
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), empty, empty, empty, None, None
    origin_step = steps[0]
    origin_mask = _top_mask(np.abs(baseline_j[origin_step]), origin_q_pct)
    origin_coverage = float(np.mean(origin_mask.astype(np.float32)))
    rli_base = []
    rli_rb = []
    for step in steps:
        b = np.abs(baseline_j[step])
        r = np.abs(rb_j[step])
        b_all = float(np.sum(b))
        r_all = float(np.sum(r))
        rli_base.append(float(np.sum(b[origin_mask])) / (b_all + EPS))
        rli_rb.append(float(np.sum(r[origin_mask])) / (r_all + EPS))
    rli_base_arr = np.asarray(rli_base, dtype=np.float64)
    rli_rb_arr = np.asarray(rli_rb, dtype=np.float64)
    delta = rli_rb_arr - rli_base_arr
    first = steps[0]
    return (
        np.asarray(steps, dtype=np.int64),
        np.asarray([s - first for s in steps], dtype=np.int64),
        rli_base_arr,
        rli_rb_arr,
        delta,
        int(origin_step),
        origin_coverage,
    )


def _compute_ali_series(
    baseline_j_mag: dict[int, np.ndarray],
    branch_jx: dict[int, np.ndarray],
    branch_jy: dict[int, np.ndarray],
    branch_grad_dx: dict[int, np.ndarray],
    branch_grad_dy: dict[int, np.ndarray],
    *,
    active_q_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = _common_steps(baseline_j_mag, branch_jx, branch_jy, branch_grad_dx, branch_grad_dy)
    if not steps:
        empty_i = np.asarray([], dtype=np.int64)
        empty_f = np.asarray([], dtype=np.float64)
        return empty_i, empty_i, empty_f, empty_f, empty_f, empty_f
    first = steps[0]
    ali_global = []
    ali_active = []
    global_samples: list[np.ndarray] = []
    active_samples: list[np.ndarray] = []
    for step in steps:
        jx = branch_jx[step]
        jy = branch_jy[step]
        gdx = branch_grad_dx[step]
        gdy = branch_grad_dy[step]
        jmag = np.sqrt(jx * jx + jy * jy)
        gmag = np.sqrt(gdx * gdx + gdy * gdy)
        ali_map = np.abs(jx * gdx + jy * gdy) / (jmag * gmag + EPS)
        ali_map = np.nan_to_num(ali_map, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mask = _top_mask(np.abs(baseline_j_mag[step]), active_q_pct)
        ali_global.append(float(np.mean(ali_map)))
        ali_active.append(float(np.mean(ali_map[mask])))
        global_samples.append(ali_map.ravel())
        active_samples.append(ali_map[mask].ravel())
    tau = np.asarray([s - first for s in steps], dtype=np.int64)
    return (
        np.asarray(steps, dtype=np.int64),
        tau,
        np.asarray(ali_global, dtype=np.float64),
        np.asarray(ali_active, dtype=np.float64),
        (np.concatenate(global_samples) if global_samples else np.asarray([], dtype=np.float64)),
        (np.concatenate(active_samples) if active_samples else np.asarray([], dtype=np.float64)),
    )


def _save_lce_plot(out_path: Path, tau: np.ndarray, lce_c: np.ndarray, lce_j: np.ndarray | None, lce_g: np.ndarray | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    if tau.size > 0:
        ax.plot(tau, lce_c, label="LCE_C", linewidth=2.0)
    if lce_j is not None and lce_j.size == tau.size:
        ax.plot(tau, lce_j, label="LCE_J", linewidth=1.7)
    if lce_g is not None and lce_g.size == tau.size:
        ax.plot(tau, lce_g, label="LCE_g", linewidth=1.7)
    ax.set_xlabel("tau (steps)")
    ax.set_ylabel("Normalized divergence")
    ax.set_title("Tier B lagged causal effects")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_ocr_plot(
    out_path: Path,
    tau: np.ndarray,
    ocr_global_base: np.ndarray,
    ocr_global_rb: np.ndarray,
    ocr_active_base: np.ndarray,
    ocr_active_rb: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    if tau.size:
        ax.plot(tau, ocr_global_base, label="OCR_global baseline", linewidth=1.6)
        ax.plot(tau, ocr_global_rb, label="OCR_global rb", linewidth=1.8)
        ax.plot(tau, ocr_active_base, label="OCR_active baseline", linewidth=1.6, linestyle="--")
        ax.plot(tau, ocr_active_rb, label="OCR_active rb", linewidth=1.8, linestyle="--")
    ax.set_xlabel("tau (steps)")
    ax.set_ylabel("OCR")
    ax.set_title("OCR over time")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_ocr_term_energy_stack(
    out_path: Path,
    tau: np.ndarray,
    t_den_base: np.ndarray,
    t_grad_base: np.ndarray,
    t_id_base: np.ndarray,
    t_rb_base: np.ndarray,
    t_den_rb: np.ndarray,
    t_grad_rb: np.ndarray,
    t_id_rb: np.ndarray,
    t_rb_rb: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=120, sharey=True)
    labels = ["T_den_rms", "T_grad_rms", "T_id_rms", "T_rb_rms"]
    colors = ["#64748b", "#0ea5e9", "#22c55e", "#f97316"]
    if tau.size:
        axes[0].stackplot(
            tau,
            t_den_base,
            t_grad_base,
            t_id_base,
            t_rb_base,
            labels=labels,
            colors=colors,
            alpha=0.9,
        )
        axes[1].stackplot(
            tau,
            t_den_rb,
            t_grad_rb,
            t_id_rb,
            t_rb_rb,
            labels=labels,
            colors=colors,
            alpha=0.9,
        )
    axes[0].set_title("Baseline term energy")
    axes[1].set_title("Read-back term energy")
    axes[0].set_xlabel("tau (steps)")
    axes[1].set_xlabel("tau (steps)")
    axes[0].set_ylabel("RMS term magnitude")
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_frg_beta_plot(out_path: Path, beta1: float, beta2: float, frg1: float | None, frg2: float | None) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=120)
    xs = [beta1, beta2]
    ys = [float(frg1 or 0.0), float(frg2 or 0.0)]
    ax.plot(xs, ys, marker="o", linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("beta_rb")
    ax.set_ylabel("FRG (median)")
    ax.set_title("FRG(beta) sweep")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_frg_ratio_plot(out_path: Path, ratio: float | None) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=120)
    ax.bar(["FRG2/FRG1"], [float(ratio or 0.0)], color="#2563eb")
    ax.set_title("FRG saturation ratio")
    ax.set_ylabel("ratio")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_rli_plot(out_path: Path, tau: np.ndarray, rli_base: np.ndarray, rli_rb: np.ndarray, delta: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=120)
    if tau.size:
        axes[0].plot(tau, rli_base, label="RLI baseline", linewidth=1.7)
        axes[0].plot(tau, rli_rb, label="RLI rb", linewidth=1.9)
        axes[0].set_title("RLI(tau)")
        axes[0].set_xlabel("tau (steps)")
        axes[0].set_ylabel("RLI")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        axes[1].plot(tau, delta, label="DeltaRLI", linewidth=1.9, color="#0f766e")
        axes[1].set_title("DeltaRLI(tau)")
        axes[1].set_xlabel("tau (steps)")
        axes[1].set_ylabel("DeltaRLI")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_ali_over_time(
    out_path: Path,
    tau: np.ndarray,
    ali_global_base: np.ndarray,
    ali_active_base: np.ndarray,
    ali_global_rb: np.ndarray,
    ali_active_rb: np.ndarray,
    ali_global_neg: np.ndarray | None = None,
    ali_active_neg: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=120)
    if tau.size:
        ax.plot(tau, ali_global_base, label="ALI global baseline", linewidth=1.5)
        ax.plot(tau, ali_active_base, label="ALI active baseline", linewidth=1.5, linestyle="--")
        ax.plot(tau, ali_global_rb, label="ALI global rb", linewidth=1.8)
        ax.plot(tau, ali_active_rb, label="ALI active rb", linewidth=1.8, linestyle="--")
        if ali_global_neg is not None and ali_active_neg is not None:
            ax.plot(tau, ali_global_neg, label="ALI global neg", linewidth=1.6, color="#7c3aed")
            ax.plot(
                tau,
                ali_active_neg,
                label="ALI active neg",
                linewidth=1.6,
                linestyle="--",
                color="#7c3aed",
            )
    ax.set_xlabel("tau (steps)")
    ax.set_ylabel("ALI")
    ax.set_title("ALI over time")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_ali_hist(
    out_path: Path,
    rb_global_samples: np.ndarray,
    rb_active_samples: np.ndarray,
    neg_global_samples: np.ndarray | None = None,
    neg_active_samples: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=120)
    if rb_global_samples.size:
        ax.hist(rb_global_samples, bins=50, density=True, alpha=0.45, label="rb global")
    if rb_active_samples.size:
        ax.hist(rb_active_samples, bins=50, density=True, alpha=0.45, label="rb active")
    if neg_global_samples is not None and neg_global_samples.size:
        ax.hist(neg_global_samples, bins=50, density=True, alpha=0.35, label="neg global")
    if neg_active_samples is not None and neg_active_samples.size:
        ax.hist(neg_active_samples, bins=50, density=True, alpha=0.35, label="neg active")
    ax.set_xlabel("ALI")
    ax.set_ylabel("density")
    ax.set_title("ALI distribution (masked vs global)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_divergence_map(out_path: Path, diff_map: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    im = ax.imshow(diff_map, cmap="magma", origin="lower")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_heatmap_with_mask(
    out_path: Path,
    value_map: np.ndarray,
    title: str,
    mask: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=120)
    im = ax.imshow(value_map, cmap="magma", origin="lower")
    if mask is not None:
        ax.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.4)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_delta_distributions(
    out_path: Path,
    delta_j_map: np.ndarray | None,
    delta_g_map: np.ndarray | None,
    mask: np.ndarray | None,
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


def _save_cai_waterfall(out_path: Path, cai: dict[str, float | None]) -> None:
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


def _save_missing_figure_placeholder(out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=120)
    ax.text(0.5, 0.5, "not available in this run", ha="center", va="center")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _load_thresholds(args: argparse.Namespace) -> dict[str, float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.thresholds_json:
        cfg_path = Path(args.thresholds_json)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for key, val in cfg.items():
            if key in thresholds and val is not None:
                thresholds[key] = float(val)

    overrides = {
        "OCR_active_min": args.ocr_active_min,
        "FRG_min": args.frg_min,
        "FRG_saturation_ratio_min": args.frg_saturation_ratio_min,
        "ALI_redundant_min": args.ali_redundant_min,
        "LCE_auc_min": args.lce_auc_min,
        "seed_var_max": args.seed_var_max,
    }
    for key, val in overrides.items():
        if val is not None:
            thresholds[key] = float(val)
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Paper-14 Tier B paired-trajectory lag + core metric diagnostics.")
    parser.add_argument("--sim-script", default="simulations/active/simulation-v16-cuda.py")
    parser.add_argument("--python-bin", default=str(Path("venv/bin/python")))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--snapshot-interval", type=int, default=100)
    parser.add_argument("--storage-mode", default="memory")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--beta-rb", type=float, default=1.0)
    parser.add_argument("--frg-beta2", type=float, default=None, help="Second beta for FRG2 (default: 2*beta-rb).")
    parser.add_argument("--run-negative-control", action="store_true", help="Run matched-scale lambda negative-control branches.")
    parser.add_argument("--negctrl-lam-scale-per-beta", type=float, default=0.10, help="Lambda scale slope for matched beta sweep.")
    parser.add_argument("--negctrl-lam-scale1", type=float, default=None, help="Explicit lambda scale for control branch 1.")
    parser.add_argument("--negctrl-lam-scale2", type=float, default=None, help="Explicit lambda scale for control branch 2.")
    parser.add_argument("--active-q-pct", type=float, default=5.0)
    parser.add_argument("--rli-origin-q-pct", type=float, default=5.0)
    parser.add_argument("--field-interval", type=int, default=1)
    parser.add_argument("--field-downsample", type=int, default=2)
    parser.add_argument("--out-dir", default="outputs/readback-baseline/tierB-smoke")
    parser.add_argument("--sim-extra-args", default="")

    parser.add_argument("--thresholds-json", default="")
    parser.add_argument("--ocr-active-min", type=float, default=None)
    parser.add_argument("--frg-min", type=float, default=None)
    parser.add_argument("--frg-saturation-ratio-min", type=float, default=None)
    parser.add_argument("--ali-redundant-min", type=float, default=None)
    parser.add_argument("--lce-auc-min", type=float, default=None)
    parser.add_argument("--seed-var-max", type=float, default=None)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent.parent
    args.sim_script = _canonicalize_sim_script(args.sim_script, root_dir)
    out_root = (root_dir / args.out_dir / Path(args.sim_script).stem / f"seed-{args.seed}").resolve()
    baseline_dir = out_root / "baseline"
    rb1_dir = out_root / "rb"
    rb2_dir = out_root / "rb2"
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _load_thresholds(args)
    beta_rb_1 = float(args.beta_rb)
    beta_rb_2 = float(args.frg_beta2) if args.frg_beta2 is not None else (2.0 * beta_rb_1 if beta_rb_1 > 0.0 else 1.0)
    negctrl_scale_1 = (
        float(args.negctrl_lam_scale1)
        if args.negctrl_lam_scale1 is not None
        else (1.0 + float(args.negctrl_lam_scale_per_beta) * beta_rb_1)
    )
    negctrl_scale_2 = (
        float(args.negctrl_lam_scale2)
        if args.negctrl_lam_scale2 is not None
        else (1.0 + float(args.negctrl_lam_scale_per_beta) * beta_rb_2)
    )

    sim_args = [
        "--headless",
        "--headless-steps",
        str(args.steps),
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
        negctrl_lam_scale=1.0,
        branch_out_dir=baseline_dir,
        field_interval=args.field_interval,
        field_downsample=args.field_downsample,
    )
    rb1_paths = _run_branch(
        root_dir=root_dir,
        python_bin=args.python_bin,
        sim_script=args.sim_script,
        sim_args=sim_args,
        beta_rb=beta_rb_1,
        negctrl_lam_scale=1.0,
        branch_out_dir=rb1_dir,
        field_interval=args.field_interval,
        field_downsample=args.field_downsample,
    )
    if abs(beta_rb_2 - beta_rb_1) <= EPS:
        rb2_paths = rb1_paths
    else:
        rb2_paths = _run_branch(
            root_dir=root_dir,
            python_bin=args.python_bin,
            sim_script=args.sim_script,
            sim_args=sim_args,
            beta_rb=beta_rb_2,
            negctrl_lam_scale=1.0,
            branch_out_dir=rb2_dir,
            field_interval=args.field_interval,
            field_downsample=args.field_downsample,
        )

    neg1_dir = out_root / "neg1"
    neg2_dir = out_root / "neg2"
    neg1_paths = None
    neg2_paths = None
    if args.run_negative_control:
        neg1_paths = _run_branch(
            root_dir=root_dir,
            python_bin=args.python_bin,
            sim_script=args.sim_script,
            sim_args=sim_args,
            beta_rb=0.0,
            negctrl_lam_scale=negctrl_scale_1,
            branch_out_dir=neg1_dir,
            field_interval=args.field_interval,
            field_downsample=args.field_downsample,
        )
        if abs(negctrl_scale_2 - negctrl_scale_1) <= EPS:
            neg2_paths = neg1_paths
        else:
            neg2_paths = _run_branch(
                root_dir=root_dir,
                python_bin=args.python_bin,
                sim_script=args.sim_script,
                sim_args=sim_args,
                beta_rb=0.0,
                negctrl_lam_scale=negctrl_scale_2,
                branch_out_dir=neg2_dir,
                field_interval=args.field_interval,
                field_downsample=args.field_downsample,
            )

    if "fields" not in baseline_paths or "fields" not in rb1_paths:
        raise RuntimeError("Tier B requires field dumps from baseline and rb branches.")

    base_rows = _read_jsonl(baseline_paths["telemetry"])
    rb1_rows = _read_jsonl(rb1_paths["telemetry"])
    neg1_rows = _read_jsonl(neg1_paths["telemetry"]) if neg1_paths is not None else []

    base_j_pre_rms = _load_stage_scalar_series(base_rows, "post_J_preclamp", "J_rms_pre")
    rb_j_pre_rms = _load_stage_scalar_series(rb1_rows, "post_J_preclamp", "J_rms_pre")
    base_j_post_rms = _load_stage_scalar_series(base_rows, "post_J_postclamp", "J_rms")
    rb_j_post_rms = _load_stage_scalar_series(rb1_rows, "post_J_postclamp", "J_rms")
    base_k_raw_rb = _load_stage_scalar_series(base_rows, "post_K_raw", "T_rb_rms")
    rb_k_raw_rb = _load_stage_scalar_series(rb1_rows, "post_K_raw", "T_rb_rms")
    base_k_reg_det = _load_stage_scalar_series(base_rows, "post_K_regularized", "detK_mean")
    rb_k_reg_det = _load_stage_scalar_series(rb1_rows, "post_K_regularized", "detK_mean")
    base_g_pre_rms = _load_stage_scalar_series(base_rows, "post_g_preblend", "g_new_rms")
    rb_g_pre_rms = _load_stage_scalar_series(rb1_rows, "post_g_preblend", "g_new_rms")
    base_g_post_det = _load_stage_scalar_series(base_rows, "post_g_postblend", "detg_mean")
    rb_g_post_det = _load_stage_scalar_series(rb1_rows, "post_g_postblend", "detg_mean")
    base_l_proxy = _load_stage_scalar_series(base_rows, "post_phi", "L_proxy_scalar")
    rb_l_proxy = _load_stage_scalar_series(rb1_rows, "post_phi", "L_proxy_scalar")

    base_c = _load_stage_series(Path(baseline_paths["fields"]), "post_core_pre_closure__C_field")
    rb1_c = _load_stage_series(Path(rb1_paths["fields"]), "post_core_pre_closure__C_field")
    rb2_c = _load_stage_series(Path(rb2_paths["fields"]), "post_core_pre_closure__C_field") if "fields" in rb2_paths else {}
    neg1_c = _load_stage_series(Path(neg1_paths["fields"]), "post_core_pre_closure__C_field") if neg1_paths is not None else {}
    neg2_c = _load_stage_series(Path(neg2_paths["fields"]), "post_core_pre_closure__C_field") if neg2_paths is not None else {}
    steps_c, tau_lce, lce_c = _compute_lce_series(base_c, rb1_c)
    if tau_lce.size == 0:
        raise RuntimeError("No overlapping C_field steps found for Tier B.")

    base_j = _load_stage_series(Path(baseline_paths["fields"]), "post_J_postclamp__J_mag_field")
    rb1_j = _load_stage_series(Path(rb1_paths["fields"]), "post_J_postclamp__J_mag_field")
    rb2_j = _load_stage_series(Path(rb2_paths["fields"]), "post_J_postclamp__J_mag_field") if "fields" in rb2_paths else {}
    neg1_j = _load_stage_series(Path(neg1_paths["fields"]), "post_J_postclamp__J_mag_field") if neg1_paths is not None else {}
    neg2_j = _load_stage_series(Path(neg2_paths["fields"]), "post_J_postclamp__J_mag_field") if neg2_paths is not None else {}

    base_jx = _load_stage_series(Path(baseline_paths["fields"]), "post_J_postclamp__Jx_field")
    base_jy = _load_stage_series(Path(baseline_paths["fields"]), "post_J_postclamp__Jy_field")
    rb1_jx = _load_stage_series(Path(rb1_paths["fields"]), "post_J_postclamp__Jx_field")
    rb1_jy = _load_stage_series(Path(rb1_paths["fields"]), "post_J_postclamp__Jy_field")
    neg1_jx = _load_stage_series(Path(neg1_paths["fields"]), "post_J_postclamp__Jx_field") if neg1_paths is not None else {}
    neg1_jy = _load_stage_series(Path(neg1_paths["fields"]), "post_J_postclamp__Jy_field") if neg1_paths is not None else {}

    base_grad_dx = _load_stage_series(Path(baseline_paths["fields"]), "post_gradC__gradC_dx_field")
    base_grad_dy = _load_stage_series(Path(baseline_paths["fields"]), "post_gradC__gradC_dy_field")
    rb1_grad_dx = _load_stage_series(Path(rb1_paths["fields"]), "post_gradC__gradC_dx_field")
    rb1_grad_dy = _load_stage_series(Path(rb1_paths["fields"]), "post_gradC__gradC_dy_field")
    neg1_grad_dx = _load_stage_series(Path(neg1_paths["fields"]), "post_gradC__gradC_dx_field") if neg1_paths is not None else {}
    neg1_grad_dy = _load_stage_series(Path(neg1_paths["fields"]), "post_gradC__gradC_dy_field") if neg1_paths is not None else {}

    _, tau_j, lce_j = _compute_lce_series(base_j, rb1_j)
    lce_j_use = lce_j if tau_j.size == tau_lce.size and np.all(tau_j == tau_lce) else None

    base_g = _load_stage_series(Path(baseline_paths["fields"]), "post_g_postblend__G_frob_field")
    rb1_g = _load_stage_series(Path(rb1_paths["fields"]), "post_g_postblend__G_frob_field")
    neg1_g = _load_stage_series(Path(neg1_paths["fields"]), "post_g_postblend__G_frob_field") if neg1_paths is not None else {}
    _, tau_g, lce_g = _compute_lce_series(base_g, rb1_g)
    lce_g_use = lce_g if tau_g.size == tau_lce.size and np.all(tau_g == tau_lce) else None

    base_rb_trace = _load_stage_series(Path(baseline_paths["fields"]), "post_K_raw__T_rb_trace_field")
    rb1_rb_trace = _load_stage_series(Path(rb1_paths["fields"]), "post_K_raw__T_rb_trace_field")
    neg1_rb_trace = _load_stage_series(Path(neg1_paths["fields"]), "post_K_raw__T_rb_trace_field") if neg1_paths is not None else {}
    base_grad_trace = _load_stage_series(Path(baseline_paths["fields"]), "post_K_raw__T_grad_trace_field")
    rb1_grad_trace = _load_stage_series(Path(rb1_paths["fields"]), "post_K_raw__T_grad_trace_field")
    neg1_grad_trace = _load_stage_series(Path(neg1_paths["fields"]), "post_K_raw__T_grad_trace_field") if neg1_paths is not None else {}

    base_t_rb = _load_stage_scalar_series(base_rows, "post_K_raw", "T_rb_rms")
    base_t_grad = _load_stage_scalar_series(base_rows, "post_K_raw", "T_grad_rms")
    rb1_t_rb = _load_stage_scalar_series(rb1_rows, "post_K_raw", "T_rb_rms")
    rb1_t_grad = _load_stage_scalar_series(rb1_rows, "post_K_raw", "T_grad_rms")
    rb1_t_rb_cv = _load_stage_scalar_series(rb1_rows, "post_K_raw", "T_rb_cv")
    neg1_t_rb = _load_stage_scalar_series(neg1_rows, "post_K_raw", "T_rb_rms") if neg1_rows else {}
    neg1_t_grad = _load_stage_scalar_series(neg1_rows, "post_K_raw", "T_grad_rms") if neg1_rows else {}
    base_k_steps, base_k_tau, base_t_den, base_t_grad_series, base_t_id, base_t_rb_series = _load_post_k_term_series(base_rows)
    rb_k_steps, rb_k_tau, rb_t_den, rb_t_grad_series, rb_t_id, rb_t_rb_series = _load_post_k_term_series(rb1_rows)

    _, tau_ocr_base, ocr_global_base = _compute_ocr_global_series(base_t_rb, base_t_grad)
    _, tau_ocr_rb, ocr_global_rb = _compute_ocr_global_series(rb1_t_rb, rb1_t_grad)
    _, tau_ocr_active_base, ocr_active_base = _compute_ocr_active_series(
        base_j, base_rb_trace, base_grad_trace, active_q_pct=args.active_q_pct
    )
    _, tau_ocr_active_rb, ocr_active_rb = _compute_ocr_active_series(
        base_j, rb1_rb_trace, rb1_grad_trace, active_q_pct=args.active_q_pct
    )
    tau_ocr_neg = np.asarray([], dtype=np.int64)
    ocr_global_neg = np.asarray([], dtype=np.float64)
    ocr_active_neg = np.asarray([], dtype=np.float64)
    if args.run_negative_control and neg1_paths is not None:
        _, tau_ocr_neg, ocr_global_neg = _compute_ocr_global_series(neg1_t_rb, neg1_t_grad)
        _, _, ocr_active_neg = _compute_ocr_active_series(
            base_j, neg1_rb_trace, neg1_grad_trace, active_q_pct=args.active_q_pct
        )

    _, tau_frg1, frg1_series = _compute_frg_series(base_j, rb1_j, beta_delta=beta_rb_1)
    _, tau_frg2, frg2_series = _compute_frg_series(base_j, rb2_j, beta_delta=beta_rb_2)
    neg_frg1_series = np.asarray([], dtype=np.float64)
    neg_frg2_series = np.asarray([], dtype=np.float64)
    if args.run_negative_control and neg1_paths is not None:
        _, _, neg_frg1_series = _compute_frg_series(base_j, neg1_j, beta_delta=abs(negctrl_scale_1 - 1.0))
        if neg2_paths is not None:
            _, _, neg_frg2_series = _compute_frg_series(base_j, neg2_j, beta_delta=abs(negctrl_scale_2 - 1.0))
    frg1_median = float(np.median(frg1_series)) if frg1_series.size else None
    frg2_median = float(np.median(frg2_series)) if frg2_series.size else None
    frg_saturation_ratio = None
    if frg1_median is not None:
        frg_saturation_ratio = float((frg2_median or 0.0) / (frg1_median + EPS))
    neg_frg1_median = float(np.median(neg_frg1_series)) if neg_frg1_series.size else None
    neg_frg2_median = float(np.median(neg_frg2_series)) if neg_frg2_series.size else None
    neg_frg_saturation_ratio = None
    if neg_frg1_median is not None:
        neg_frg_saturation_ratio = float((neg_frg2_median or 0.0) / (neg_frg1_median + EPS))

    _, tau_rli, rli_base, rli_rb, delta_rli, rli_origin_step, rli_origin_coverage = _compute_rli_series(
        base_j, rb1_j, origin_q_pct=args.rli_origin_q_pct
    )
    tau_rli_neg = np.asarray([], dtype=np.int64)
    rli_neg = np.asarray([], dtype=np.float64)
    delta_rli_neg = np.asarray([], dtype=np.float64)
    if args.run_negative_control and neg1_paths is not None:
        _, tau_rli_neg, _, rli_neg, delta_rli_neg, _, _ = _compute_rli_series(
            base_j, neg1_j, origin_q_pct=args.rli_origin_q_pct
        )

    _, tau_ali_base, ali_global_base, ali_active_base, ali_global_samples_base, ali_active_samples_base = _compute_ali_series(
        base_j, base_jx, base_jy, base_grad_dx, base_grad_dy, active_q_pct=args.active_q_pct
    )
    _, tau_ali_rb, ali_global_rb, ali_active_rb, ali_global_samples_rb, ali_active_samples_rb = _compute_ali_series(
        base_j, rb1_jx, rb1_jy, rb1_grad_dx, rb1_grad_dy, active_q_pct=args.active_q_pct
    )
    tau_ali_neg = np.asarray([], dtype=np.int64)
    ali_global_neg = np.asarray([], dtype=np.float64)
    ali_active_neg = np.asarray([], dtype=np.float64)
    ali_global_samples_neg = np.asarray([], dtype=np.float64)
    ali_active_samples_neg = np.asarray([], dtype=np.float64)
    if args.run_negative_control and neg1_paths is not None:
        _, tau_ali_neg, ali_global_neg, ali_active_neg, ali_global_samples_neg, ali_active_samples_neg = _compute_ali_series(
            base_j,
            neg1_jx,
            neg1_jy,
            neg1_grad_dx,
            neg1_grad_dy,
            active_q_pct=args.active_q_pct,
        )

    # Align OCR for plotting
    if (
        tau_ocr_base.size == tau_ocr_rb.size == tau_ocr_active_base.size == tau_ocr_active_rb.size
        and tau_ocr_base.size > 0
        and np.all(tau_ocr_base == tau_ocr_rb)
        and np.all(tau_ocr_base == tau_ocr_active_base)
        and np.all(tau_ocr_base == tau_ocr_active_rb)
    ):
        tau_ocr_plot = tau_ocr_base
    else:
        tau_ocr_plot = np.asarray([], dtype=np.int64)
        ocr_global_base = np.asarray([], dtype=np.float64)
        ocr_global_rb = np.asarray([], dtype=np.float64)
        ocr_active_base = np.asarray([], dtype=np.float64)
        ocr_active_rb = np.asarray([], dtype=np.float64)

    _save_lce_plot(figures_dir / "lce_lag_curves.png", tau_lce, lce_c, lce_j_use, lce_g_use)
    if (
        base_k_steps.size == rb_k_steps.size
        and base_k_steps.size > 0
        and np.all(base_k_steps == rb_k_steps)
    ):
        _save_ocr_term_energy_stack(
            figures_dir / "ocr_term_energy_stack.png",
            base_k_tau,
            base_t_den,
            base_t_grad_series,
            base_t_id,
            base_t_rb_series,
            rb_t_den,
            rb_t_grad_series,
            rb_t_id,
            rb_t_rb_series,
        )
    _save_ocr_plot(
        figures_dir / "ocr_over_time.png",
        tau_ocr_plot,
        ocr_global_base,
        ocr_global_rb,
        ocr_active_base,
        ocr_active_rb,
    )
    _save_frg_beta_plot(figures_dir / "frg_vs_beta.png", beta_rb_1, beta_rb_2, frg1_median, frg2_median)
    _save_frg_ratio_plot(figures_dir / "frg_saturation_ratio.png", frg_saturation_ratio)
    _save_rli_plot(figures_dir / "rli_curves.png", tau_rli, rli_base, rli_rb, delta_rli)
    if (
        tau_ali_base.size == tau_ali_rb.size
        and tau_ali_base.size > 0
        and np.all(tau_ali_base == tau_ali_rb)
    ):
        neg_global_plot = None
        neg_active_plot = None
        if tau_ali_neg.size == tau_ali_base.size and np.all(tau_ali_neg == tau_ali_base):
            neg_global_plot = ali_global_neg
            neg_active_plot = ali_active_neg
        _save_ali_over_time(
            figures_dir / "ali_over_time.png",
            tau_ali_base,
            ali_global_base,
            ali_active_base,
            ali_global_rb,
            ali_active_rb,
            ali_global_neg=neg_global_plot,
            ali_active_neg=neg_active_plot,
        )
    _save_ali_hist(
        figures_dir / "ali_hist.png",
        ali_global_samples_rb,
        ali_active_samples_rb,
        neg_global_samples=ali_global_samples_neg if ali_global_samples_neg.size else None,
        neg_active_samples=ali_active_samples_neg if ali_active_samples_neg.size else None,
    )

    required_exact_figures = [
        "ocr_term_energy_stack.png",
        "ocr_over_time.png",
        "frg_vs_beta.png",
        "frg_saturation_ratio.png",
        "lce_lag_curves.png",
        "rli_curves.png",
        "ali_over_time.png",
        "ali_hist.png",
    ]
    for name in required_exact_figures:
        target = figures_dir / name
        if not target.exists():
            _save_missing_figure_placeholder(target, name)

    for tau_target in (1, 2, 5):
        step_target = int(steps_c[0] + tau_target)
        if step_target in base_c and step_target in rb1_c:
            diff_map = np.abs(rb1_c[step_target] - base_c[step_target]).astype(np.float32)
            _save_divergence_map(
                figures_dir / f"lce_divergence_map_tau{tau_target}.png",
                diff_map,
                f"|C_rb - C_0| at tau={tau_target}",
            )

    tiera_step = _select_common_step(base_j, rb1_j)
    if tiera_step is None and steps_c.size > 0:
        tiera_step = int(steps_c[-1])
    if tiera_step is None:
        tiera_step = 0

    delta_j_map = None
    delta_g_map = None
    active_mask = None
    if tiera_step in base_j and tiera_step in rb1_j:
        delta_j_map = np.abs(rb1_j[tiera_step] - base_j[tiera_step]).astype(np.float32)
        active_mask = _top_mask(np.abs(base_j[tiera_step]), args.active_q_pct)
        _save_heatmap_with_mask(
            figures_dir / f"tierA_deltaJ_heatmap_t{tiera_step}.png",
            delta_j_map,
            f"DeltaJ_map @ step {tiera_step}",
            active_mask,
        )
    else:
        _save_missing_figure_placeholder(
            figures_dir / f"tierA_deltaJ_heatmap_t{tiera_step}.png",
            f"DeltaJ_map @ step {tiera_step}",
        )

    if tiera_step in base_g and tiera_step in rb1_g:
        delta_g_map = np.abs(rb1_g[tiera_step] - base_g[tiera_step]).astype(np.float32)
        _save_heatmap_with_mask(
            figures_dir / f"tierA_deltaG_heatmap_t{tiera_step}.png",
            delta_g_map,
            f"DeltaG_map @ step {tiera_step}",
            active_mask,
        )
    else:
        _save_missing_figure_placeholder(
            figures_dir / f"tierA_deltaG_heatmap_t{tiera_step}.png",
            f"DeltaG_map @ step {tiera_step}",
        )

    _save_delta_distributions(
        figures_dir / f"tierA_delta_distributions_t{tiera_step}.png",
        delta_j_map,
        delta_g_map,
        active_mask,
    )

    delta_j_tiera = _safe_delta(_safe_value(base_j_post_rms, tiera_step), _safe_value(rb_j_post_rms, tiera_step))
    delta_g_tiera = _safe_delta(_safe_value(base_g_pre_rms, tiera_step), _safe_value(rb_g_pre_rms, tiera_step))
    delta_l_tiera = _safe_delta(_safe_value(base_l_proxy, tiera_step), _safe_value(rb_l_proxy, tiera_step))
    cai = {
        "DeltaJ_preClamp": _safe_delta(_safe_value(base_j_pre_rms, tiera_step), _safe_value(rb_j_pre_rms, tiera_step)),
        "DeltaJ_postClamp": _safe_delta(_safe_value(base_j_post_rms, tiera_step), _safe_value(rb_j_post_rms, tiera_step)),
        "DeltaK_raw": _safe_delta(_safe_value(base_k_raw_rb, tiera_step), _safe_value(rb_k_raw_rb, tiera_step)),
        "DeltaK_reg": _safe_delta(_safe_value(base_k_reg_det, tiera_step), _safe_value(rb_k_reg_det, tiera_step)),
        "DeltaG_preBlend": _safe_delta(_safe_value(base_g_pre_rms, tiera_step), _safe_value(rb_g_pre_rms, tiera_step)),
        "DeltaG_postBlend": _safe_delta(_safe_value(base_g_post_det, tiera_step), _safe_value(rb_g_post_det, tiera_step)),
    }
    _save_cai_waterfall(figures_dir / f"cai_waterfall_t{tiera_step}.png", cai)

    ocr_global_median_rb = float(np.median(ocr_global_rb)) if ocr_global_rb.size else None
    ocr_active_median_rb = float(np.median(ocr_active_rb)) if ocr_active_rb.size else None
    ocr_global_median_neg = float(np.median(ocr_global_neg)) if ocr_global_neg.size else None
    ocr_active_median_neg = float(np.median(ocr_active_neg)) if ocr_active_neg.size else None
    delta_rli_mean = float(np.mean(delta_rli)) if delta_rli.size else None
    delta_rli_auc = _safe_auc(tau_rli.astype(np.float64), delta_rli) if tau_rli.size else None
    delta_rli_mean_neg = float(np.mean(delta_rli_neg)) if delta_rli_neg.size else None
    delta_rli_auc_neg = _safe_auc(tau_rli_neg.astype(np.float64), delta_rli_neg) if tau_rli_neg.size else None
    ali_global_median_rb = float(np.median(ali_global_rb)) if ali_global_rb.size else None
    ali_active_median_rb = float(np.median(ali_active_rb)) if ali_active_rb.size else None
    ali_global_median_neg = float(np.median(ali_global_neg)) if ali_global_neg.size else None
    ali_active_median_neg = float(np.median(ali_active_neg)) if ali_active_neg.size else None

    flags = {
        "RB_dominated": (
            None if ocr_active_median_rb is None else bool(ocr_active_median_rb < thresholds["OCR_active_min"])
        ),
        "Saturation_clamp": (
            None if frg_saturation_ratio is None else bool(frg_saturation_ratio < thresholds["FRG_saturation_ratio_min"])
        ),
        "Redundant_channel": (
            None
            if ali_active_median_rb is None or frg1_median is None
            else bool(
                ali_active_median_rb >= thresholds["ALI_redundant_min"]
                and frg1_median < thresholds["FRG_min"]
            )
        ),
        "Non_reproducible": None,    # multi-seed variance gate lands in later iterations
        "Closure_dependent": None,   # closure stratification lands in later iterations
        "Negative_control_fail": None,
    }

    frg_pass = frg1_median is not None and frg1_median >= thresholds["FRG_min"]
    lce_auc = _safe_auc(tau_lce.astype(np.float64), lce_c)
    lce_pass = lce_auc is not None and lce_auc >= thresholds["LCE_auc_min"]
    negative_control_pass = None
    contrast = None
    if args.run_negative_control and neg1_paths is not None:
        frg_adv = (frg1_median or 0.0) - (neg_frg1_median or 0.0)
        loop_adv = abs(delta_rli_auc or 0.0) - abs(delta_rli_auc_neg or 0.0)
        ali_adv = abs((ali_active_median_rb or 0.0) - (ali_global_median_rb or 0.0)) - abs(
            (ali_active_median_neg or 0.0) - (ali_global_median_neg or 0.0)
        )
        contrast_score = int(frg_adv > 0.0) + int(loop_adv > 0.0) + int(ali_adv > 0.0)
        negative_control_pass = contrast_score >= 2
        flags["Negative_control_fail"] = not negative_control_pass
        contrast = {
            "rb": {
                "OCR_active_median": ocr_active_median_rb,
                "FRG1_median": frg1_median,
                "FRG2_median": frg2_median,
                "FRG_saturation_ratio": frg_saturation_ratio,
                "DeltaRLI_auc": delta_rli_auc,
                "ALI_active_median": ali_active_median_rb,
                "ALI_global_median": ali_global_median_rb,
            },
            "negative_control": {
                "lam_scale1": negctrl_scale_1,
                "lam_scale2": negctrl_scale_2,
                "OCR_active_median": ocr_active_median_neg,
                "FRG1_median": neg_frg1_median,
                "FRG2_median": neg_frg2_median,
                "FRG_saturation_ratio": neg_frg_saturation_ratio,
                "DeltaRLI_auc": delta_rli_auc_neg,
                "ALI_active_median": ali_active_median_neg,
                "ALI_global_median": ali_global_median_neg,
            },
            "advantages": {
                "frg_adv": frg_adv,
                "loop_adv": loop_adv,
                "ali_adv": ali_adv,
                "contrast_score": contrast_score,
            },
            "pass": negative_control_pass,
        }

    if not frg_pass and not lce_pass:
        decision = "Formal-only"
    elif frg_pass and lce_pass and not bool(flags["Saturation_clamp"]) and not bool(flags["Redundant_channel"]):
        decision = "Effective"
    else:
        decision = "Weak-effective"
    if negative_control_pass is False and decision == "Effective":
        decision = "Weak-effective"

    # Composite score is a ranking heuristic (not a truth predicate).
    tau_rb = 1e-6
    tau_cv = 0.05
    tau_frg = float(thresholds["FRG_min"])
    tau_lce_min = float(thresholds["LCE_auc_min"])
    ocr_lo = 0.01
    ocr_hi = 5.0

    presence = (
        float(np.mean(rb_t_rb_series > tau_rb))
        if rb_t_rb_series.size
        else 0.0
    )
    rb_cv_vals = np.asarray(
        [float(v) for _, v in sorted(rb1_t_rb_cv.items())],
        dtype=np.float64,
    )
    structure = (
        float(np.mean(rb_cv_vals > tau_cv))
        if rb_cv_vals.size
        else 0.0
    )
    dominance = (
        float(np.mean((ocr_global_rb >= ocr_lo) & (ocr_global_rb <= ocr_hi)))
        if ocr_global_rb.size
        else 0.0
    )
    causal_frg = (
        float(np.mean(frg1_series > tau_frg))
        if frg1_series.size
        else 0.0
    )
    causal_lce = (
        float(np.mean(lce_c > tau_lce_min))
        if lce_c.size
        else 0.0
    )
    causal = causal_frg * causal_lce
    readback_score = float(presence * structure * dominance * causal)

    np.savez_compressed(
        out_root / "series.npz",
        steps=steps_c,
        tau=tau_lce,
        LCE_C=lce_c,
        LCE_J=(lce_j_use if lce_j_use is not None else np.asarray([], dtype=np.float64)),
        LCE_g=(lce_g_use if lce_g_use is not None else np.asarray([], dtype=np.float64)),
        LCE_G=(lce_g_use if lce_g_use is not None else np.asarray([], dtype=np.float64)),
        OCR_global_rb=ocr_global_rb,
        OCR_active_rb=ocr_active_rb,
        T_den_baseline=base_t_den,
        T_grad_baseline=base_t_grad_series,
        T_id_baseline=base_t_id,
        T_rb_baseline=base_t_rb_series,
        T_den_rb=rb_t_den,
        T_grad_rb=rb_t_grad_series,
        T_id_rb=rb_t_id,
        T_rb_rb=rb_t_rb_series,
        FRG1=frg1_series,
        FRG2=frg2_series,
        FRG1_neg=neg_frg1_series,
        FRG2_neg=neg_frg2_series,
        RLI_baseline=rli_base,
        RLI_rb=rli_rb,
        DeltaRLI=delta_rli,
        RLI_neg=rli_neg,
        DeltaRLI_neg=delta_rli_neg,
        ALI_global_baseline=ali_global_base,
        ALI_active_baseline=ali_active_base,
        ALI_global_rb=ali_global_rb,
        ALI_active_rb=ali_active_rb,
        ALI_global_neg=ali_global_neg,
        ALI_active_neg=ali_active_neg,
    )

    scores = {
        "sim_script": args.sim_script,
        "seed": args.seed,
        "steps": args.steps,
        "tierA_snapshot_step": tiera_step,
        "DeltaJ": delta_j_tiera,
        "DeltaG": delta_g_tiera,
        "DeltaL": delta_l_tiera,
        "CAI": cai,
        "beta_rb_baseline": 0.0,
        "beta_rb_test": beta_rb_1,
        "beta_rb_frg2": beta_rb_2,
        "trajectory_length": int(tau_lce.size),
        "OCR_global_median_rb": ocr_global_median_rb,
        "OCR_active_median_rb": ocr_active_median_rb,
        "OCR_global_median_neg": ocr_global_median_neg,
        "OCR_active_median_neg": ocr_active_median_neg,
        "FRG1_median": frg1_median,
        "FRG2_median": frg2_median,
        "FRG_saturation_ratio": frg_saturation_ratio,
        "FRG1_median_neg": neg_frg1_median,
        "FRG2_median_neg": neg_frg2_median,
        "FRG_saturation_ratio_neg": neg_frg_saturation_ratio,
        "RLI_origin_step": rli_origin_step,
        "RLI_origin_coverage": rli_origin_coverage,
        "DeltaRLI_mean": delta_rli_mean,
        "DeltaRLI_auc": delta_rli_auc,
        "DeltaRLI_mean_neg": delta_rli_mean_neg,
        "DeltaRLI_auc_neg": delta_rli_auc_neg,
        "LCE_C_max": float(np.max(lce_c)),
        "LCE_C_auc": lce_auc,
        "LCE_J_max": float(np.max(lce_j_use)) if lce_j_use is not None and lce_j_use.size else None,
        "LCE_J_auc": _safe_auc(tau_lce.astype(np.float64), lce_j_use) if lce_j_use is not None else None,
        "LCE_g_max": float(np.max(lce_g_use)) if lce_g_use is not None and lce_g_use.size else None,
        "LCE_g_auc": _safe_auc(tau_lce.astype(np.float64), lce_g_use) if lce_g_use is not None else None,
        "LCE_G_max": float(np.max(lce_g_use)) if lce_g_use is not None and lce_g_use.size else None,
        "LCE_G_auc": _safe_auc(tau_lce.astype(np.float64), lce_g_use) if lce_g_use is not None else None,
        "ALI_global_median_rb": ali_global_median_rb,
        "ALI_active_median_rb": ali_active_median_rb,
        "ALI_global_median_neg": ali_global_median_neg,
        "ALI_active_median_neg": ali_active_median_neg,
        "negative_control": contrast,
        "negative_control_pass": negative_control_pass,
        "readback_score": readback_score,
        "readback_score_components": {
            "presence": presence,
            "structure": structure,
            "dominance": dominance,
            "causal": causal,
            "causal_frg": causal_frg,
            "causal_lce": causal_lce,
            "thresholds": {
                "tau_rb": tau_rb,
                "tau_cv": tau_cv,
                "tau_frg": tau_frg,
                "tau_lce": tau_lce_min,
                "ocr_lo": ocr_lo,
                "ocr_hi": ocr_hi,
            },
        },
        "readback_score_label": "ranking_heuristic_not_truth_predicate",
        "thresholds": thresholds,
        "flags": flags,
        "decision": decision,
        "artifacts": {
            "baseline_telemetry": str(baseline_paths["telemetry"]),
            "rb_telemetry": str(rb1_paths["telemetry"]),
            "rb2_telemetry": str(rb2_paths["telemetry"]),
            "neg1_telemetry": str(neg1_paths["telemetry"]) if neg1_paths is not None else None,
            "neg2_telemetry": str(neg2_paths["telemetry"]) if neg2_paths is not None else None,
            "baseline_fields": str(baseline_paths["fields"]),
            "rb_fields": str(rb1_paths["fields"]),
            "rb2_fields": str(rb2_paths["fields"]),
            "neg1_fields": str(neg1_paths["fields"]) if neg1_paths is not None else None,
            "neg2_fields": str(neg2_paths["fields"]) if neg2_paths is not None else None,
            "figures_dir": str(figures_dir),
            "series": str(out_root / "series.npz"),
        },
    }
    (out_root / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    if contrast is not None:
        contrast_path = out_root / "contrast.json"
        contrast_path.write_text(json.dumps(contrast, indent=2), encoding="utf-8")
        contrast_lines = [
            "# Read-back vs Negative-Control Contrast",
            "",
            f"- pass: `{contrast['pass']}`",
            f"- contrast_score: `{contrast['advantages']['contrast_score']}`",
            "",
            "| metric | read-back | negative-control |",
            "|---|---:|---:|",
            f"| OCR_active_median | {contrast['rb']['OCR_active_median']} | {contrast['negative_control']['OCR_active_median']} |",
            f"| FRG1_median | {contrast['rb']['FRG1_median']} | {contrast['negative_control']['FRG1_median']} |",
            f"| FRG2_median | {contrast['rb']['FRG2_median']} | {contrast['negative_control']['FRG2_median']} |",
            f"| DeltaRLI_auc | {contrast['rb']['DeltaRLI_auc']} | {contrast['negative_control']['DeltaRLI_auc']} |",
            f"| ALI_active_median | {contrast['rb']['ALI_active_median']} | {contrast['negative_control']['ALI_active_median']} |",
            "",
            "| advantage | value |",
            "|---|---:|",
            f"| frg_adv | {contrast['advantages']['frg_adv']} |",
            f"| loop_adv | {contrast['advantages']['loop_adv']} |",
            f"| ali_adv | {contrast['advantages']['ali_adv']} |",
            "",
        ]
        (out_root / "contrast.md").write_text("\n".join(contrast_lines), encoding="utf-8")

    report_lines = [
        "# Tier B Lagged + Core Metrics Report",
        "",
        "#### 0) Run metadata",
        f"- sim_script: `{args.sim_script}`",
        f"- seed: `{args.seed}`",
        f"- nx,ny,dx: `{args.nx},{args.ny},{args.dx}`",
        f"- steps: `{args.steps}`",
        f"- snapshot_interval: `{args.snapshot_interval}`",
        "",
        "#### 1) Parameter summary",
        f"- beta_rb baseline/test/frg2: `0.0 / {beta_rb_1} / {beta_rb_2}`",
        f"- active_q_pct: `{args.active_q_pct}`",
        f"- rli_origin_q_pct: `{args.rli_origin_q_pct}`",
        f"- field_interval: `{args.field_interval}`",
        f"- field_downsample: `{args.field_downsample}`",
        "",
        "#### 2) Executive summary (1 screen)",
        f"- Decision: `{decision}`",
        f"- OCR_active median: `{ocr_active_median_rb}`",
        f"- FRG1/FRG2: `{frg1_median}` / `{frg2_median}`",
        f"- FRG saturation ratio: `{frg_saturation_ratio}`",
        f"- LCE_C AUC: `{lce_auc}`",
        f"- DeltaRLI mean/AUC: `{delta_rli_mean}` / `{delta_rli_auc}`",
        f"- ALI global/active medians: `{ali_global_median_rb}` / `{ali_active_median_rb}`",
        f"- CAI bottleneck snapshot step: `{tiera_step}`",
        f"- Negative-control pass: `{negative_control_pass}`",
        f"- readback_score (ranking heuristic): `{readback_score}`",
        "",
        "#### 3) Tier A: Counterfactual snapshot results",
        f"- step: `{tiera_step}`",
        f"- DeltaJ: `{delta_j_tiera}`",
        f"- DeltaG: `{delta_g_tiera}`",
        f"- DeltaL: `{delta_l_tiera}`",
        f"- `figures/tierA_deltaJ_heatmap_t{tiera_step}.png`",
        f"- `figures/tierA_deltaG_heatmap_t{tiera_step}.png`",
        f"- `figures/tierA_delta_distributions_t{tiera_step}.png`",
        "",
        "#### 4) OCR: Term dominance over time",
        "- `figures/ocr_term_energy_stack.png`",
        "- `figures/ocr_over_time.png`",
        "",
        "#### 5) FRG: Gain and saturation",
        "- `figures/frg_vs_beta.png`",
        "- `figures/frg_saturation_ratio.png`",
        "",
        "#### 6) Tier B: Lagged causal effects (experience accumulation)",
        "- `figures/lce_lag_curves.png`",
        "- `figures/lce_divergence_map_tau1.png` (if available)",
        "- `figures/lce_divergence_map_tau2.png` (if available)",
        "- `figures/lce_divergence_map_tau5.png` (if available)",
        "",
        "#### 7) RLI: Return / loop closure",
        "- `figures/rli_curves.png`",
        "",
        "#### 8) CAI: Attenuation bottleneck",
        *(f"- {k}: `{v}`" for k, v in cai.items()),
        f"- `figures/cai_waterfall_t{tiera_step}.png`",
        "",
        "#### 9) ALI: Redundancy / alignment",
        "- `figures/ali_over_time.png`",
        "- `figures/ali_hist.png`",
        "",
        "#### 10) Cross-version fingerprints (when running experiment matrix)",
        "- matrix output is produced by `experiments/scripts/run_readback_matrix.py`.",
        "- for single Tier B runs this section is informational only.",
        "",
        "#### 11) Notes / anomalies (auto + manual)",
        f"- RB_dominated: `{flags['RB_dominated']}`",
        f"- Saturation_clamp: `{flags['Saturation_clamp']}`",
        f"- Redundant_channel: `{flags['Redundant_channel']}`",
        f"- Negative_control_fail: `{flags['Negative_control_fail']}`",
        f"- Non_reproducible: `{flags['Non_reproducible']}`",
        f"- Closure_dependent: `{flags['Closure_dependent']}`",
        "",
        "## Thresholds",
        *(f"- {k}: `{v}`" for k, v in thresholds.items()),
        "",
    ]
    (out_root / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[OK] Tier B run complete: {out_root}")
    print(f"[OK] scores.json: {out_root / 'scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
