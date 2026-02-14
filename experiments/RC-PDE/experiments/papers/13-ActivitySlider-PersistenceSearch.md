# Paper 13 — Activity Slider: Persistence Search & Evaluation (v15/v16)

This paper is a practical runbook for discovering “organism-like” persistence regimes in the RC PDE sims using a **single knob** (`--activity`) and a **single scalar score** (from snapshots) so an experimenter or agent can run structured searches instead of eyeballing transients.

## Goal

Find configurations where identities:

- persist over long horizons (not just interesting transients),
- stay within a bounded band (not collapse to 0, not coalesce to 1, not explode until capped),
- exhibit low churn (few merges/birth-death thrash),
- remain “alive” without relying on constant births.

Because fixed-domain PDEs strongly bias toward attractors, this paper is about **finding the best regime that exists in the current equations**, and recognizing when stability is structurally unavailable without changing the model.

## What `--activity` Does

`--activity` is a convenience slider in `[0, 1]` that overrides a small set of already-existing CLI knobs:

- identity budget: `--identity-cap-fraction`, `--identity-birth-gate-fraction`
- soft closure family: `--closure-softness`, `--spark-softness`, `--collapse-softness`

The mapping lives in `configs/activity_config.py` and is intentionally conservative: it provides a searchable 1D slice through the regime space, not a claim of optimal physics.

Notes:

- Default behavior is unchanged if `--activity` is not set.
- `meta.json` in snapshot outputs records `"activity": ...` when used.

## Required Outputs (Disk Snapshots)

This paper assumes **disk snapshots** (`--storage-mode disk`) so runs can be scored without rerunning the sim.

Scoring reads:

- `snap_*.npz` fields: `n_active`, `I_mass`, `closure_births`, `mass`
- optional `meta.json` (for provenance, including activity)

## Scoring: `score_persistence.py`

`experiments/scripts/score_persistence.py` produces:

- `score` (higher is better)
- `stable_frac`: fraction of snapshots with identity count within `[ids_min, ids_max]`
- `alive_frac`: fraction with `ids > 0`
- `turnover`: mean absolute step-to-step change in `ids`
- `I_mass_cv`: coefficient of variation of identity mass
- `birth_events_frac`: fraction of snapshots with `closure_births > 0`

Interpretation (high level):

- You generally want **high** `stable_frac`, **high** `alive_frac`
- You generally want **low** `turnover`, **low** `I_mass_cv`
- You do **not** necessarily want constant births; frequent births often correlate with churn

## Recommended “First Search” Protocol (v15)

Keep everything fixed except `--activity`:

- same seed(s)
- same grid size
- same step count
- same snapshot interval

### Single Run (one activity)

```bash
out=outputs/v15-activity/a0.65-s1
mkdir -p "$out"
./venv/bin/python simulations/active/simulation-v15-cuda.py \
  --headless --headless-steps 8000 \
  --nx 1024 --ny 1024 --dx 0.1 --seed 1 \
  --storage-mode disk --snapshot-interval 50 --snapshot-dir "$out" \
  --closure-mode soft \
  --activity 0.65

./venv/bin/python experiments/scripts/score_persistence.py --snapshot-dir "$out"
```

### Sweep Run (rank activities)

```bash
for a in 0.30 0.40 0.50 0.60 0.65 0.70 0.80; do
  out=outputs/v15-activity/a${a}-s1
  mkdir -p "$out"
  ./venv/bin/python simulations/active/simulation-v15-cuda.py \
    --headless --headless-steps 8000 \
    --nx 1024 --ny 1024 --dx 0.1 --seed 1 \
    --storage-mode disk --snapshot-interval 50 --snapshot-dir "$out" \
    --closure-mode soft \
    --activity "$a"
  ./venv/bin/python experiments/scripts/score_persistence.py --snapshot-dir "$out"
done
```

### Seed Robustness Check

Take the top 1–3 activity values and repeat across multiple seeds:

```bash
for s in 1 2 3; do
  for a in 0.60 0.65 0.70; do
    out=outputs/v15-activity/a${a}-s${s}
    mkdir -p "$out"
    ./venv/bin/python simulations/active/simulation-v15-cuda.py \
      --headless --headless-steps 8000 \
      --nx 1024 --ny 1024 --dx 0.1 --seed "$s" \
      --storage-mode disk --snapshot-interval 50 --snapshot-dir "$out" \
      --closure-mode soft \
      --activity "$a"
    ./venv/bin/python experiments/scripts/score_persistence.py --snapshot-dir "$out"
  done
done
```

## Recommended “First Search” Protocol (v16)

Start with v16 in parity-like mode, then optionally turn on scaffold features once you have a baseline:

### Baseline v16 (parity-like closures)

```bash
out=outputs/v16-activity/a0.65-s1
mkdir -p "$out"
./venv/bin/python simulations/active/simulation-v16-cuda.py \
  --headless --headless-steps 8000 \
  --nx 1024 --ny 1024 --dx 0.1 --seed 1 \
  --storage-mode disk --snapshot-interval 50 --snapshot-dir "$out" \
  --closure-mode soft \
  --activity 0.65

./venv/bin/python experiments/scripts/score_persistence.py --snapshot-dir "$out"
```

### v16 With Scaffolds Enabled (optional)

Use this only after you have a baseline, because it increases degrees of freedom:

```bash
out=outputs/v16-activity-nonlocal/a0.65-s1
mkdir -p "$out"
./venv/bin/python simulations/active/simulation-v16-cuda.py \
  --headless --headless-steps 8000 \
  --nx 1024 --ny 1024 --dx 0.1 --seed 1 \
  --storage-mode disk --snapshot-interval 50 --snapshot-dir "$out" \
  --closure-mode soft \
  --nonlocal-mode on \
  --operator-diagnostics --operator-diagnostics-interval 10 \
  --domain-mode adaptive --domain-adapt-strength 0.30 --domain-adapt-interval 50 \
  --activity 0.65

./venv/bin/python experiments/scripts/score_persistence.py --snapshot-dir "$out"
```

## Practical Tips (Avoid False Positives)

- Run long enough to clear transients (8k–20k steps beats 1k–2k for persistence claims).
- Keep snapshot interval fixed when comparing scores.
- Always rerun top configs across multiple seeds.
- Don’t tune by visuals: tune by score first, then inspect top runs.
- If you get high scores only at tiny grids (e.g. 128/256) but not at 1024, that’s usually a discretization artifact.

## What to Do If No Regime Scores Well

If all activity values produce:

- rapid collapse (`alive_frac` low),
- coalescence to 1 identity (`stable_frac` low, `ids_last` near 1),
- high churn (`turnover` high),

then the limitation is likely structural (missing inhibition/saturation/repulsion terms), not “wrong knob”. At that point:

- use this paper’s results as a reproducible negative finding, and
- move to model changes (e.g. v16 nonlocal/operator/domain paths, or explicit competition/saturation terms).
