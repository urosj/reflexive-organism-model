from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


def _list_snapshot_files(snapshot_dir: Path) -> list[Path]:
    files = sorted(snapshot_dir.glob("snap_*.npz"))
    if not files:
        raise FileNotFoundError(f"No snap_*.npz files found in {snapshot_dir}")
    return files


def _step_from_name(path: Path) -> int:
    m = re.search(r"(\d+)\.npz$", path.name)
    return int(m.group(1)) if m else -1


def _load_series(files: list[Path]) -> dict[str, np.ndarray]:
    steps, ids, imass, births, mass = [], [], [], [], []
    for f in files:
        data = np.load(f, allow_pickle=False)
        steps.append(int(data.get("step", _step_from_name(f))))
        ids.append(int(data.get("n_active", 0)))
        imass.append(float(data.get("I_mass", 0.0)))
        births.append(float(data.get("closure_births", 0.0)))
        mass.append(float(data.get("mass", 0.0)))
    return {
        "step": np.asarray(steps, dtype=np.int64),
        "ids": np.asarray(ids, dtype=np.int64),
        "I_mass": np.asarray(imass, dtype=np.float64),
        "closure_births": np.asarray(births, dtype=np.float64),
        "mass": np.asarray(mass, dtype=np.float64),
    }


def _score(
    ids: np.ndarray,
    imass: np.ndarray,
    births: np.ndarray,
    *,
    ids_min: int,
    ids_max: int,
) -> dict[str, float]:
    if ids.size == 0:
        raise ValueError("No snapshots provided")

    alive_frac = float(np.mean(ids > 0))
    stable_frac = float(np.mean((ids >= ids_min) & (ids <= ids_max)))

    if ids.size >= 2:
        turnover = float(np.mean(np.abs(np.diff(ids))))
    else:
        turnover = 0.0
    denom = float(max(1, ids_max - ids_min))
    turnover_norm = float(turnover / denom)

    imass_mean = float(np.mean(imass))
    imass_std = float(np.std(imass))
    imass_cv = float(imass_std / (abs(imass_mean) + 1e-12))

    birth_events_frac = float(np.mean(births > 0.0))

    # Simple "organism-like persistence" score:
    # - wants to stay alive (ids>0)
    # - wants to be in a bounded band (ids within [min,max])
    # - penalizes churn and large identity-mass swings
    # - does NOT require births to keep firing (births are treated as churn proxy)
    score = stable_frac * alive_frac
    score *= math.exp(-1.5 * min(2.0, turnover_norm))
    score *= math.exp(-1.0 * min(2.0, imass_cv))

    return {
        "alive_frac": alive_frac,
        "stable_frac": stable_frac,
        "turnover_mean_abs": float(turnover),
        "turnover_norm": float(turnover_norm),
        "I_mass_mean": imass_mean,
        "I_mass_cv": imass_cv,
        "birth_events_frac": birth_events_frac,
        "score": float(score),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score RC snapshot runs for persistence (v15/v16 compatible).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--snapshot-dir", type=str, required=True, help="Directory containing snap_*.npz (+ optional meta.json)")
    ap.add_argument("--ids-min", type=int, default=2, help="Lower bound for stable identity count band")
    ap.add_argument("--ids-max", type=int, default=20, help="Upper bound for stable identity count band")
    ap.add_argument("--json", action="store_true", help="Print full JSON report (includes meta if present)")
    args = ap.parse_args()

    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    files = _list_snapshot_files(snapshot_dir)
    series = _load_series(files)

    meta_path = snapshot_dir / "meta.json"
    meta = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    metrics = _score(
        series["ids"],
        series["I_mass"],
        series["closure_births"],
        ids_min=args.ids_min,
        ids_max=args.ids_max,
    )

    if args.json:
        out = {
            "snapshot_dir": str(snapshot_dir),
            "n_snapshots": int(series["ids"].size),
            "steps_first": int(series["step"][0]),
            "steps_last": int(series["step"][-1]),
            "ids_first": int(series["ids"][0]),
            "ids_last": int(series["ids"][-1]),
            "metrics": metrics,
            "meta": meta,
        }
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(
            " ".join(
                [
                    f"score={metrics['score']:.6f}",
                    f"stable_frac={metrics['stable_frac']:.3f}",
                    f"alive_frac={metrics['alive_frac']:.3f}",
                    f"turnover={metrics['turnover_mean_abs']:.3f}",
                    f"I_mass_cv={metrics['I_mass_cv']:.3f}",
                    f"birth_events_frac={metrics['birth_events_frac']:.3f}",
                    f"ids_last={int(series['ids'][-1])}",
                ]
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

