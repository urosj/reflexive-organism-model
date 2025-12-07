# Reflexive Coherence (RC) — Purpose, Vision, and Use

This repository is the evolving research workspace for *Reflexive Coherence* (RC):
a framework for understanding how structure, identity, agency, and geometry can
emerge from continuous fields and their feedback loops. The code here does not aim
to be a production engine or a polished agent framework. Instead, it serves as a
laboratory for exploring three central ideas:

1. **Identity as a field**, not an object.  
   In RC, persistent structures (“identities”) appear as localized concentrations
   of a coherence field, maintained by feedback rather than hard-coded rules.

2. **Geometry as something agents create**, not something they merely move in.  
   The dynamics of the coherence field induce an effective metric that shapes
   motion, information flow, and the stability of structures.

3. **Reflexivity as the mechanism of agency.**  
   Agents (identities) alter the fields that guide them. Fields alter the agents
   that maintain them. This circularity—rather than discrete logic or fixed
   policies—is what gives RC its expressive power.

The simulations here (v1 ... v12, and the RC-Ant example) illustrate how relatively
simple PDE rules can support nontrivial phenomena: attractor basins, collapse
and spark events, identity lifecycles, resource-driven population change,
and field-mediated communication. These experiments highlight *how agency can
emerge*, not because it is pre-programmed, but because the system is allowed to
shape—and be shaped by—its own geometry.

This makes RC useful for researchers interested in:

- collective behavior and distributed systems  
- artificial ecologies and swarm intelligence  
- adaptive topologies and emergent communication  
- conceptual foundations of agentic architectures  
- field-based models of cognition, coordination, or resource flow  

While RC is not yet a complete agent architecture, it offers a substrate where
identity, action, memory, and environment are continuous processes rather than
discrete modules. The ant-colony simulation (Paper X) provides a concrete example
of how RC dynamics can support multi-agent phenomena without abandoning the
mathematical core of RC-II and RC-III.

This repository should be understood as a *seed*: a space for experimentation,
iteration, and theoretical development. The code is deliberately modular and
transparent, enabling you to adjust the dynamics, introduce new fields, or explore
structural questions—without committing to any fixed “agent model”.

For details on implementation, parameters, and project structure, see the `experiments`
directory.

# Reflexive Coherence (RC) Simulation - V1

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

- Or simply run whatever version of simulation directly, no params.

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

## Simulation V6

... is the one that follows pure RC equation implementation.

## Simulation v12

... is the most feature complete, however, diverges from pure RC equations. See [paper 7](./experiments/7-IdentitiesAddon.md).

## Observations

See section the papers in the `experiments` directory.

