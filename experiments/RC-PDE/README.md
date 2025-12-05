# Reflexive Coherence (RC) Simulation

This project implements a 2-D simulation of the Reflexive Coherence (RC) loop. It models a scalar coherence field on a Cartesian grid, evolves it via metric-aware flux rules, detects collapse/spark events, and visualises system dynamics with `matplotlib.animation.FuncAnimation`. Core dependencies: `numpy` for array math, `scipy` for numerical methods, and `matplotlib` for plotting/animation. Modules separate field geometry, flux rules, PDE stepping, events, and visualisation.

## Requirements
- numpy>=1.24
- scipy>=1.11
- matplotlib>=3.8

## Structure
- `rc/field.py`: grid-based coherence field with gradients, Laplacian/Laplace–Beltrami, Hessian, integration helpers.
- `rc/geometry.py`: coherence tensor construction, metric/inverse determinants, Laplace–Beltrami operator.
- `rc/flux.py`: velocity laws, flux computation, divergence (Euclidean or covariant).
- `rc/pde.py`: variational derivative, RHS assembly, Euler and semi-implicit Crank–Nicolson steppers with coherence renormalisation cadence.
- `rc/events.py`: spark detection and topology-change hooks.
- `rc/visualisation.py`: RCAnimator for heatmaps, flux quiver overlay, curvature fade, and GIF saving.
- `run_simulation.py`: CLI entry point (supports `--step` mode selection, `--basins N` to seed multiple bumps, frame sampling, and progress printing).
- `tests/`: unit and integration tests covering field ops, geometry, flux, PDE conservation, events, and visualisation.
- `docs/quickstart.md`: usage guide; `examples/basic_demo.ipynb`: short inline demo; `LICENSE`: CC BY-SA 4.0.

Install deps with `python -m pip install -r requirements.txt`.

## Quick start
- Run a simulation (GIF output by default):  
  `python run_simulation.py --nx 128 --ny 128 --dt 0.01 --steps 500 --basins 3 --output outputs/rc.gif`

### CLI parameters (run_simulation.py)
- `--nx`, `--ny` (int): grid dimensions. Larger grids capture finer structure but cost more CPU (64–256 typical).
- `--dx` (float): grid spacing. Smaller `dx` refines the mesh; reduce `dt` accordingly for stability.
- `--dt` (float): time step. Start around `0.01`; lower to improve stability if fields oscillate or blow up.
- `--steps` (int): number of iterations to run. More steps show longer evolution; try 200–1000.
- `--basins` (int): number of initial Gaussian bumps placed randomly. Exact count is seeded; higher values add competing attractors (e.g., 1–5).
- `--step` (str): choose stepping scheme: `euler` (explicit), `c-n` (Crank–Nicolson), or `rhs` (generic step wrapper).
- `--sample-every` (int): record every Nth frame for animation. Increase to shrink output size (default 5).
- `--output` (path): where to save animation (GIF by default).
- `--no-events` (flag): disable collapse/spark topology changes if you want pure PDE evolution.
- `--seed` (int): RNG seed for reproducible initialization.
- `--config` (path): JSON file to set parameters (see below). CLI flags override config only when they differ from defaults.

### Parameters in code/config (pde.rhs/step_*, events, viz)
- `lam` (λ): scales base metric coupling to C; larger λ strengthens the identity component of K.
- `xi` (ξ): weights gradient outer-product in K; higher ξ sharpens metric response to edges and drives anisotropy.
- `zeta` (ζ): reserved for extra flux coupling (currently unused).
- `kappa` (κ): mobility/curvature weight in Φ; higher κ increases smoothing via Laplace–Beltrami.
- `alpha`, `beta`: weights for dissipative vs. gradient-following velocity components (v = -α g⁻¹∇Φ + β g⁻¹∇C).
- `gamma_curv`: curvature/anisotropy tension injected into Φ (via trace(K)).
- Potentials: choose `double_well` (`a`, `b`), `quadratic` (`k`, `C0`), or `plateau` (flat between `C_low`, `C_high` with quadratic walls `k_low`, `k_high`). Plateau reduces the drive to coalesce and supports multi-basins.
- Event knobs: `spark_epsilon`, `bump_amplitude`, `events_every`, `events_start`, `spark_enabled`.
- Visualisation: `truth_mode` (hide overlay/quiver), `show_overlay`, `use_curvature_alpha`, `use_global_clim` (fixed clim across frames), `max_clamp` (clip display), `sample_every` (frame capture stride).
- Coherence renormalisation: `coherence_renormalization` (apply renormalisation every N steps; default 1).

### Visualization defaults
- RCAnimator uses `cmap=viridis`, quiver downsampling (default every 8th point), GIF output at ~10 fps. Curvature strength is blended into alpha to highlight sparks; adjust in `rc/visualisation.py` if desired. Set `use_global_clim=false` for per-frame scaling.

### Config file example
You can place PDE weights and overrides in a JSON file:

```json
{
  "nx": 128,
  "ny": 128,
  "dt": 0.005,
  "steps": 800,
  "basins": 4,
  "lam": 0.8,
  "xi": 1.0,
  "kappa": 0.5,
  "potential": "plateau",
  "potential_params": { "C_low": 0.1, "C_high": 0.5, "k_low": 1.0, "k_high": 1.0 },
  "spark_epsilon": 1e-5,
  "bump_amplitude": 0.2,
  "events_every": 3,
  "spark_enabled": true,
  "events_start": 100,
  "alpha": 1.0,
  "beta": 0.3,
  "gamma_curv": 0.1,
  "truth_mode": true,
  "use_global_clim": true,
  "max_clamp": null,
  "coherence_renormalization": 10,
  "step": "euler"
}
```

Run with `python run_simulation.py --config config.json`. Any CLI flag you pass that differs from its default will override the config. See `docs/quickstart.md` for more details and options.


### Potentials
- `plateau` with params `C_low`, `C_high`, `k_low`, `k_high`
- `double_well` with params `a`, `b`
- `quadratic` with params `k`, `C0`

#### Examples

  "potential": "plateau",
  "potential_params": {
    "C_low": 0.05,
    "C_high": 0.6,
    "k_low": 0.2,
    "k_high": 0.2
  },


  "potential": "double_well",
  "potential_params": {
    "a": 1.0,
    "b": 0.5
  },

  

  ## Simulation V2

  The simulation was built on the outcomes of the observations. It is more stable, however, it is still more or less about diffusion.

  ## Observations

  See section the papers in the `experiments` directory. The content of the papers is as follows:

  - 1A is about theoretical explanations of the observed
  - 1B is same content, but from the engineering/applied view
  - 2 deals with observations themselves
  - 3 provides a proof that RC can not be modeled through PDEs alone

  