"""CLI driver for the Reflexive Coherence simulation."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

import numpy as np

from rc import events, flux, pde
from rc.field import CoherenceField
from rc.geometry import compute_K, metric_from_K
from rc.visualisation import RCAnimator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Reflexive Coherence simulation.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config for parameters.")
    parser.add_argument("--nx", type=int, default=64, help="Grid size in x.")
    parser.add_argument("--ny", type=int, default=64, help="Grid size in y.")
    parser.add_argument("--dx", type=float, default=0.1, help="Grid spacing.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step.")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps to run.")
    parser.add_argument(
        "--step",
        type=str,
        default="euler",
        choices=["euler", "c-n", "rhs"],
        help="Time stepping mode: euler (explicit), c-n (Crank–Nicolson), or rhs (generic step).",
    )
    parser.add_argument("--output", type=str, default="outputs/rc_evolution.gif", help="Path to save animation.")
    parser.add_argument("--sample-every", type=int, default=5, help="Frame sampling stride for animation.")
    parser.add_argument("--no-events", action="store_true", help="Disable topology change events.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initialization noise.")
    parser.add_argument(
        "--basins",
        type=int,
        default=1,
        help="Number of initial basins (Gaussian bumps) to seed randomly across the field.",
    )
    return parser


def _seed_random_basins(
    field: CoherenceField, count: int, rng: np.random.Generator, amplitude: float = 1.0
) -> None:
    """Add `count` Gaussian bumps at random locations to seed multiple basins."""
    if count <= 0:
        return
    ny, nx = field.grid_shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    sigma = 0.08 * min(nx, ny)

    # Small uniform background to prevent zero-field issues
    field.C[:] = 1e-6

    for _ in range(count):
        cx = rng.uniform(2*nx/5, 3*nx/5)  # changed from 0, nx
        cy = rng.uniform(2*ny/5, 3*ny/5)  # changed from 0, ny
        amp = rng.uniform(0.5 * amplitude, 1.0 * amplitude)
        bump = amp * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2)))
        field.C += bump
    field.clip_nonnegative()


def main(args: Optional[argparse.Namespace] = None) -> None:
    parser = build_parser()
    args = args or parser.parse_args()
    config: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    def resolve(name: str):
        cli_val = getattr(args, name)
        default_val = parser.get_default(name)
        if cli_val != default_val:
            return cli_val
        if name in config:
            return config[name]
        return cli_val

    grid_shape = (resolve("ny"), resolve("nx"))

    rng = np.random.default_rng(resolve("seed"))
    field = CoherenceField(
        grid_shape,
        dx=float(resolve("dx")),
        init_func=lambda x, y: np.full_like(x, 0.05),
        rng=rng,
    )
    seed_amp = float(config.get("bump_amplitude", 1.0))
    _seed_random_basins(field, int(resolve("basins")), rng, amplitude=seed_amp)

    animator_kwargs = {
        "dt": float(resolve("dt")),
        "truth_mode": bool(config.get("truth_mode", False)),
        "use_global_clim": bool(config.get("use_global_clim", True)),
        "max_clamp": config.get("max_clamp", None),
        "use_curvature_alpha": bool(config.get("use_curvature_alpha", True)),
        "show_overlay": bool(config.get("show_overlay", True)),
    }
    if animator_kwargs["truth_mode"]:
        animator_kwargs["use_curvature_alpha"] = False
        animator_kwargs["show_overlay"] = False
    animator = RCAnimator(grid_shape, **animator_kwargs)
    target_coherence = field.total_coherence()

    dt = float(resolve("dt"))
    steps = int(resolve("steps"))
    sample_every = int(resolve("sample_every"))
    use_events = not bool(resolve("no_events"))
    step_mode = resolve("step")
    output_path_str = str(resolve("output"))
    renorm_every = int(config.get("coherence_renormalization", 1))

    pde_params = {
        "lam": float(config.get("lam", 1.0)),
        "xi": float(config.get("xi", 0.5)),
        "zeta": float(config.get("zeta", 0.0)),
        "kappa": float(config.get("kappa", 1.0)),
        "potential": config.get("potential", "double_well"),
        "potential_params": config.get("potential_params", None),
        "alpha": float(config.get("alpha", 1.0)),
        "beta": float(config.get("beta", 0.0)),
        "gamma_curv": float(config.get("gamma_curv", 0.0)),
    }

    event_params = {
        "spark_epsilon": float(config.get("spark_epsilon", 1e-4)),
        "bump_amplitude": float(config.get("bump_amplitude", 0.1)),
        "apply_every": int(config.get("events_every", 1)),
        "spark_enabled": bool(config.get("spark_enabled", True)),
        "events_start": int(config.get("events_start", 0)),
        "debug_events": bool(config.get("debug_events", False)),
    }

    print(f"Using potential: {pde_params['potential']} with params={pde_params['potential_params']}")
    progress_interval = max(1, steps // 10)

    for step_idx in range(steps):
        if step_mode == "c-n":
            pde.step_crank_nicolson(
                field,
                dt=dt,
                target_coherence=target_coherence,
                renormalize_every=renorm_every,
                step_index=step_idx,
                **pde_params,
            )
        elif step_mode == "rhs":
            pde.step(
                field,
                dt=dt,
                renormalize_every=renorm_every,
                step_index=step_idx,
                **pde_params,
            )
        else:  # explicit Euler
            pde.step_euler(
                field,
                dt=dt,
                renormalize_every=renorm_every,
                step_index=step_idx,
                target_coherence=target_coherence,
                **pde_params,
            )

        if (
            use_events
            and (step_idx + 1) >= event_params["events_start"]
            and (step_idx + 1) % max(1, event_params["apply_every"]) == 0
        ):
            change_info = events.apply_topology_change(
                field,
                spark_epsilon=event_params["spark_epsilon"],
                bump_amplitude=event_params["bump_amplitude"],
                spark_enabled=event_params["spark_enabled"],
                return_details=event_params["debug_events"],
            )
            if event_params["debug_events"]:
                print(f"[events] step {step_idx+1}: spark={change_info.get('spark')}")

        # Recompute flux for diagnostics/visualization
        K = compute_K(field, lam=pde_params["lam"], xi=pde_params["xi"], zeta=pde_params["zeta"])
        g, g_inv, sqrt_det = metric_from_K(field, K, lam=pde_params["lam"], xi=pde_params["xi"])
        vx, vy = flux.velocity(field, g_inv, K)
        Jx, Jy = flux.flux(field, (vx, vy))

        #_C_min = 1e-6
        #field.C = np.maximum(field.C, _C_min)

        if step_idx % sample_every == 0:
            animator.update(step_idx, field.C, Jx, Jy, K=K.as_array())

        if (step_idx + 1) % progress_interval == 0 or step_idx == steps - 1:
            percent = int((step_idx + 1) / steps * 100)
            print(f"Step {step_idx + 1}/{steps} ({percent}%)", end="\r", flush=True)
            # Temporary diagnostics for coherence statistics
            print(
                "DEBUG C stats:",
                "min=",
                float(field.C.min()),
                "max=",
                float(field.C.max()),
                "mean=",
                float(field.C.mean()),
                "sum=",
                float(field.C.sum()),
            )

    print()  # newline after progress
    output_path = animator.save(output_path_str)

    # Diagnostics
    total = field.total_coherence()
    hess = field.hessian()
    curvature = hess[0, 0] + hess[1, 1]
    min_curv, max_curv = float(np.min(curvature)), float(np.max(curvature))
    print(f"Finished {steps} steps; saved animation to {output_path}")
    print(f"Total coherence: {total:.6f}")
    print(f"Curvature min/max: {min_curv:.4e} / {max_curv:.4e}")


if __name__ == "__main__":
    main()
