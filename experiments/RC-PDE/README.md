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

The simulations here (v1 ... v16, and the RC-Ant example) illustrate how relatively
simple PDE rules can support nontrivial phenomena: attractor basins, collapse
and spark events, identity lifecycles, resource-driven population change,
and field-mediated communication. These experiments highlight *how agency can
emerge*, not because it is pre-programmed, but because the system is allowed to
shape—and be shaped by—its own geometry.  
The newer v15/v16 line also serves a second purpose: testing what remains of RC
behavior when we move away from strict RC-II/RC-III closure assumptions and push
the PDE-only approximation as far as possible.

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
mathematical core of RC-II and RC-III, while the v15/v16 line explicitly tests
how far PDE-only behavior can be pushed as those strict closure assumptions are relaxed.

This repository should be understood as a *seed*: a space for experimentation,
iteration, and theoretical development. The code is deliberately modular and
transparent, enabling you to adjust the dynamics, introduce new fields, or explore
structural questions—without committing to any fixed “agent model”.

## Repository map (quick)

- `papers/`: conceptual and formal RC papers.
- `simulations/active/`: active CUDA simulator lanes (`v12-cuda`..`v16-cuda`).
- `simulations/legacy/`: older/archival simulator lanes.
- `simulations/helpers/`: shared simulation helpers (for example the VisPy viewer wrapper).
- `configs/`: shared config and blob definitions.
- `experiments/`: papers and runbooks across implementation and evaluation.
- `experiments/papers/`: paper/checklist/guide markdown sources.
- `experiments/scripts/`: experiment runners, scoring, schema, and verification tooling.
- `scripts/`: repository utility scripts (for example artifact cleanup).
- `run_simulation.py` + `rc/`: earlier V1 package and utilities.
- `QuickStart.md`: concise runbook for v16 telemetry and read-back audit actions.

## Path Policy

- Canonical paths are now:
  - simulators: `simulations/active/...`
  - experiment scripts: `experiments/scripts/...`
  - experiment papers: `experiments/papers/...`
- Legacy root/`experiments/<name>` symlink aliases have been removed.
- New copy/paste examples should always use canonical paths.
- Canonical references:
  - simulator runtime: `simulations/active/simulation-v16-cuda.py`
  - read-back tier runner: `experiments/scripts/run_readback_tierB.py`
  - read-back user guide: `experiments/papers/14B-ExperienceReadback-UsersGuide.md`

## Recommended entry points

- All commands below assume you are in the repository root.
- If you use the repo virtualenv, replace `python` with `./venv/bin/python`.
- `simulations/active/simulation-v12-cuda.py` through `simulations/active/simulation-v16-cuda.py` are CUDA-preferred and can fall back to CPU when CUDA is unavailable (with lower throughput).
- Non-CUDA fallback entry point: `python run_simulation.py --help`.
- Fast start guide: `QuickStart.md`.
- Read the project arc first: `experiments/README.md`.
- Run the current simulator: `python simulations/active/simulation-v16-cuda.py --help`.
- Start quickly on GPU:
  `python simulations/active/simulation-v16-cuda.py --headless --storage-mode disk --snapshot-interval 100 --nx 1024 --ny 1024`.
- Full v16 (best feature coverage, interactive):
  `python simulations/active/simulation-v16-cuda.py --closure-mode soft --nonlocal-mode on --operator-diagnostics --operator-diagnostics-interval 10 --domain-mode adaptive --domain-adapt-strength 0.30 --domain-adapt-interval 50`
- Full v16 (best feature coverage, headless artifacts):
  `python simulations/active/simulation-v16-cuda.py --headless --headless-steps 2000 --nx 1024 --ny 1024 --dx 0.1 --seed 1 --storage-mode disk --snapshot-interval 50 --closure-mode soft --nonlocal-mode on --operator-diagnostics --operator-diagnostics-interval 10 --domain-mode adaptive --domain-adapt-strength 0.30 --domain-adapt-interval 50`
- Full closure pipeline (baseline + ablations + gate + summaries):
  `bash experiments/scripts/run_v16_iteration7_all.sh`
- Clean cache artifacts:
  `bash scripts/clean_artifacts.sh`
- Full cleanup (cache + snapshots + outputs + venv):
  `bash scripts/clean_artifacts.sh --full`

## Scope and caveats

- The current simulators are still fixed-grid approximations; they do not yet implement the full evolving-manifold RC ideal described in Papers 5–6.
- v12/v13 are the stronger RC-II/RC-III-aligned closure baselines; v14 introduces soft-closure continuation; v15 enforces core/events/closure separation; v16 explores PDE-boundary behavior when relaxing strict RC-II/RC-III closure dependence.
- `simulations/active/simulation-v16-cuda.py` remains PDE-only and fixed-grid, but adds explicit nonlocal terms, operator diagnostics, continuous-identity intrinsic paths, and an adaptive-domain proxy to push this boundary.
- v16 `closure-mode off` is the intrinsic claim path; `soft/full` are retained as explicit dependence comparators, not hidden defaults.
- None of v12–v16 should be interpreted as full structural-operator/topology-transition completion; they are staged approximations and diagnostics.
- v13/v14/v15/v16 use geometry-consistent (`sqrt(|g|)`) identity-mass accounting to avoid mixed-measure interpretation errors.
- Papers 13 and 14 are analysis/observation layers (persistence search + read-back audit protocol) applied to existing simulators; they are not additional RC simulator implementations.

## Legacy RC V1 simulation package (`run_simulation.py` + `rc/`)

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
- `experiments/papers/README.md`: paper/runbook project arc and execution references.
- `LICENSE`: GPLv2.

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

Run with `python run_simulation.py --config configs/config.json`. Any CLI flag you pass that differs from its default will override the config. For execution packs and runbooks, see `experiments/papers/README.md`.

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

... is the most feature complete, however, diverges from pure RC equations. See [paper 7](./experiments/papers/7-IdentitiesAddon.md).

## Simulation v12 CUDA (`simulations/active/simulation-v12-cuda.py`)

`simulations/active/simulation-v12-cuda.py` is the GPU-oriented implementation of v12. It keeps the
same RC-PDE model direction, but focuses on large-grid performance and long runs.
The code uses PyTorch tensors end-to-end and runs on CUDA when available (falls
back to CPU when CUDA is not available).

### What this version adds

- Torch/CUDA backend for field, metric, flux, and identity updates.
- Vectorized identity storage: a single tensor `[max_identities, Nx, Ny]`.
- Vectorized identity diffusion/evolution (`laplacian_3d`) and batched identity seeding.
- Headless snapshot system for long runs (`memory` or `disk` storage modes).
- Offline animation export from snapshots (`.mp4` with ffmpeg, fallback `.gif`).
- Optional fast live viewer with VisPy, plus Matplotlib fallback.
- Global identity mass cap tied to initial total coherence mass budget.

### Identity mass note

Compared to `simulations/legacy/simulation-v12.py`, the CUDA variant changes identity mass handling:

- The identity mass cap is set from the initial coherence mass at runtime (instead of a fixed constant).
- A post-update rescaling step enforces the global identity mass cap each step.

This means identity-mass dynamics are intentionally stricter than the original CPU v12 implementation.

### CUDA requirements

- Install standard deps: `python -m pip install -r requirements.txt`
- Install CUDA/VisPy deps: `python -m pip install -r requirements-cuda.txt`

`requirements-cuda.txt` includes `torch`, `vispy`, and Qt backend support.

### Runtime modes

- Live VisPy (default): `python simulations/active/simulation-v12-cuda.py`
- Live Matplotlib fallback: `python simulations/active/simulation-v12-cuda.py --no-vispy`
- Headless long run:  
  `python simulations/active/simulation-v12-cuda.py --headless --headless-steps 5000 --storage-mode disk --snapshot-interval 100`

### Useful flags

- `--nx`, `--ny`, `--dx`: grid resolution.
- `--headless`, `--headless-steps`: non-interactive long-run control.
- `--storage-mode {memory,disk}`, `--snapshot-interval`, `--snapshot-dir`: snapshot pipeline.
- `--use-vispy`, `--no-vispy`: force viewer choice.
- `--fps`, `--animate-interval`, `--two-pass`: offline animation output behavior.
- `--seed`: reproducible initialization.

## Simulation v13 CUDA (`simulations/active/simulation-v13-cuda.py`) — paper-aligned baseline

`simulations/active/simulation-v13-cuda.py` is the stricter baseline used for paper alignment and
regression checks.

### Position of v13 in the Papers 1–9 arc

- **Papers 1A–6:** v13 is not the full coherence-only evolving-manifold realization; it remains a fixed-grid approximation with explicit closures.
- **Paper 7 (RC-v12):** v13 captures the discrete identity lifecycle machinery (birth gates, collapse thresholding, global identity budget, Heun identity updates).
- **Paper 8 (RC-II):** v13 operationalizes the coherence–identity–geometry triad and coherence-mass invariant projection, but still with discrete RC-III closures.
- **Paper 9 (RC-III):** v13 is strongly aligned as a computational closure for emergent, tracked identities (proto-agent style dynamics).

### v13 mass-accounting improvements vs v12-cuda

- Identity mass uses geometry-consistent weighting:
  `M_I = ∫ I_sum * sqrt(|g|) dxdy`.
- Cap uses structured coherence baseline:
  `I_cap = identity_cap_fraction * ∫ max(C0 - C_bg, 0) * sqrt(|g|) dxdy`.
- Birth hysteresis + hard cap enforcement stabilize long-run identity budgets.

### Baseline tests (v13 semantics)

These tests validate paper-aligned baseline semantics and are useful as regression
checks when changing v14 soft-closure behavior:

- `tests/test_v13_cuda_structure.py`
- `tests/test_v13_cuda_paper_alignment.py`

Run:

`./venv/bin/python -m pytest -q tests/test_v13_cuda_structure.py tests/test_v13_cuda_paper_alignment.py`

## Simulation v14 CUDA (`simulations/active/simulation-v14-cuda.py`)

`simulations/active/simulation-v14-cuda.py` is the soft-closure continuation branch. It keeps the v13 reflexive
core (coherence invariance projection, C↔g coupling, identity feedback), but adds a
controlled way to reduce hard RC-III gating dependence.

### What v14 is trying to achieve

- Preserve RC-III computational closure while softening binary event machinery.
- Blend hard spark/birth/collapse thresholds with continuous field-driven scores.
- Support parameter sweeps from strict RC-III behavior to softer closure behavior,
  without removing reflexive geometry feedback.
- Theoretical context is documented in
  `experiments/papers/10-SoftClosureContinuation.md`.

### v14-specific knobs

- `--closure-softness` (0 = hard RC-III gates, 1 = softest blending)
- `--spark-softness` (spark intensity transition width)
- `--collapse-softness` (soft collapse damping near `id_min_mass`)

### v14 quick examples

- Hard gating baseline:
  `python simulations/active/simulation-v14-cuda.py --headless --closure-softness 0 --seed 1`
- Softer closure behavior:
  `python simulations/active/simulation-v14-cuda.py --headless --closure-softness 0.8 --spark-softness 0.08 --collapse-softness 0.5 --seed 1`

## Simulation v15 CUDA (`simulations/active/simulation-v15-cuda.py`) — core-first architecture baseline

`simulations/active/simulation-v15-cuda.py` introduced the explicit layer split used by later versions:
L0 reflexive PDE core, L1 intrinsic event diagnostics, and L2 closure fallback.
It is the reference architecture for closure-dependence ablations.

### What is unique in v15

- Strict mode routing via `--closure-mode off|soft|full`.
- Core-only and core+events intrinsic paths without mandatory closure edits.
- Built-in closure budget diagnostics (`M_I/M_C` tracking and closure factors).
- Reproducible ablation harness designed around aligned seeds/horizons.

### v15 companion docs and scripts

- Spec/handoff: `experiments/papers/11F-v15-Codex-Handoff.md`
- Implementation checklist: `experiments/papers/11A-v15-ImplementationChecklist.md`
- Baseline freeze: `experiments/papers/11B-v15-BaselineFreeze.md`
- Ablation harness note: `experiments/papers/11C-v15-AblationHarness.md`
- Performance/evaluation note: `experiments/papers/11D-v15-Iteration6-PerformanceAndEvaluation.md`
- Runtime closure runbook: `experiments/papers/11E-v15-Iteration7-CUDARuntimeClosure.md`
- Ablation script: `experiments/scripts/run_v15_ablations.sh`
- Performance gate script: `experiments/scripts/run_v15_iteration6_gate.sh`
- Full closure pipeline: `experiments/scripts/run_v15_iteration7_all.sh`

## Simulation v16 CUDA (`simulations/active/simulation-v16-cuda.py`) — current PDE-boundary target

`simulations/active/simulation-v16-cuda.py` is the current runtime target for PDE-only RC continuation.
It keeps the v15 architectural split (core/events/closure) and adds explicit v16
upgrades aimed at maximizing fidelity within PDE limits.

### What is unique in v16

- Explicit nonlocal PDE path (`--nonlocal-mode on`) using FFT-based kernel coupling.
- First-class operator diagnostics (`det(K)`, `cond(K)`, drift, degeneracy occupancy).
- Adaptive-domain proxy (`--domain-mode adaptive`) with conservative remap accounting.
- Continuous identity substrate in intrinsic paths (`closure-mode off`) with event readout.
- Closure comparators (`soft`, `full`) retained for explicit dependence analysis.

### v16 companion docs and scripts

- Spec: `experiments/papers/12-v16-Spec.md`
- Checklist: `experiments/papers/12A-v16-ImplementationChecklist.md`
- Performance/evaluation note: `experiments/papers/12D-v16-PerformanceAndEvaluation.md`
- Runtime closure runbook: `experiments/papers/12E-v16-CUDARuntimeClosure.md`
- Release manifest: `experiments/papers/12F-v16-ReleaseManifest.md`
- Ablation harness: `experiments/scripts/run_v16_ablations.sh`
- Performance gate: `experiments/scripts/run_v16_iteration6_gate.sh`
- Full closure pipeline: `experiments/scripts/run_v16_iteration7_all.sh`

## Observations

See the papers in the `experiments` directory.
