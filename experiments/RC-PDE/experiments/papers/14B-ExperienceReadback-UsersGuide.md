# 14B — Experience Read-Back User's Guide

## 1. Purpose and scope
This guide is the operator manual for the Paper 14 telemetry and audit pipeline used in `v12-cuda` through `v16-cuda`. It explains how to run Tier A / Tier B counterfactuals, generate audit artifacts, run cross-version matrix comparisons, and validate packet completeness.

Path policy:
- This guide uses canonical paths (`simulations/active/...`, `experiments/scripts/...`).
- Legacy symlink paths were removed; use canonical paths only.

Operational definition used here:
- Effective read-back means intervention on read-back coupling (`beta_rb`) causes measurable deltas in immediate operator/flux behavior (Tier A) and lagged trajectory divergence (Tier B), quantified by `OCR`, `FRG`, `LCE`, `RLI`, `CAI`, `ALI`.

## 2. Quickstart (minimal working run)
Minimal single-version audit (v16 lane, Tier A + Tier B):

```bash
# Tier A snapshot counterfactual (A2 default + A1 optional via wrapper)
bash experiments/scripts/run_readback_iteration3_tierA.sh

# Tier B paired trajectory scoring + plots
bash experiments/scripts/run_readback_iteration4_tierB.sh

# Open key outputs
ls outputs/readback-baseline/tierB-smoke/simulation-v16-cuda/seed-1
ls outputs/readback-baseline/tierB-smoke/simulation-v16-cuda/seed-1/figures

# Verify packet completeness on the quickstart output
./venv/bin/python experiments/scripts/verify_readback_audit_packet.py \
  --packets-root outputs/readback-baseline/tierB-smoke \
  --out-json outputs/readback-baseline/tierB-smoke/verification.json \
  --out-md outputs/readback-baseline/tierB-smoke/verification.md
```

Primary files to inspect:
- `outputs/readback-baseline/tierB-smoke/simulation-v16-cuda/seed-1/report.md`
- `outputs/readback-baseline/tierB-smoke/simulation-v16-cuda/seed-1/scores.json`
- `outputs/readback-baseline/tierB-smoke/simulation-v16-cuda/seed-1/figures/frg_vs_beta.png`
- `outputs/readback-baseline/tierB-smoke/simulation-v16-cuda/seed-1/figures/lce_lag_curves.png`

Fast-start references:
- `QuickStart.md` for an end-to-end, copy/paste entry path.
- `experiments/papers/14E-ExperienceReadback-InterpretationGuide.md` for standardized interpretation logic.

## Runbook
| Goal | Command | Output to check |
|---|---|---|
| CI smoke (schema + Tier B + verify) | `bash experiments/scripts/run_readback_ci_smoke.sh` | `outputs/readback-baseline/ci-smoke/verification.md` |
| CI smoke (hard gate) | `STRICT_VERIFY=1 bash experiments/scripts/run_readback_ci_smoke.sh` | shell exit code + `verification.md` |
| single run audit (Tier B) | `bash experiments/scripts/run_readback_iteration4_tierB.sh` | `outputs/readback-baseline/tierB-smoke/.../report.md` |
| Tier A snapshot deltas | `bash experiments/scripts/run_readback_iteration3_tierA.sh` | `figures/tierA_deltaJ_heatmap_t*.png`, `cai_waterfall_t*.png` |
| stability repeatability | `bash experiments/scripts/run_readback_iteration4_stability.sh` | `stability.md`, `stability.json` |
| negative-control contrast | `bash experiments/scripts/run_readback_iteration5_negative_control.sh` | `contrast.md`, `scores.json` flags |
| cross-version matrix | `bash experiments/scripts/run_readback_iteration6_matrix.sh` | `figures/version_fingerprint_grid.png`, `verification.md` |

## 3. Concept map: what gets produced
Pipeline:
- Simulator (`simulations/active/simulation-v12-cuda.py` ... `simulations/active/simulation-v16-cuda.py`)
- In-loop hooks (`experiments/scripts/readback_hooks.py`) emit telemetry stream and optional field dumps
- Tier runners (`run_readback_tierA.py`, `run_readback_tierB.py`) run counterfactual branches and compute metrics
- Matrix runner (`run_readback_matrix.py`) assembles per-run packets and cross-version fingerprints
- Verifier (`verify_readback_audit_packet.py`) checks required sections/figures

Artifact flow:
- Raw telemetry stream: `outputs/readback-telemetry/<sim>-seed-<seed>-<ts>-<pid>.jsonl`
- Optional sparse fields: same path with `.fields.npz`
- Tier A packet: `.../tierA-smoke/<sim>/<mode>-seed-<seed>/`
- Tier B packet: `.../tierB-smoke/<sim>/seed-<seed>/`
- Matrix packet: `.../tier6-matrix/packets/<sim>/seed-<seed>/`

Plot sets A-H mapping:
- A (Tier A heatmaps/distributions): `tierA_deltaJ_heatmap_t*.png`, `tierA_deltaG_heatmap_t*.png`, `tierA_delta_distributions_t*.png`
- B (OCR): `ocr_term_energy_stack.png`, `ocr_over_time.png`
- C (FRG): `frg_vs_beta.png`, `frg_saturation_ratio.png`
- D (LCE): `lce_lag_curves.png`, optional `lce_divergence_map_tau{1,2,5}.png`
- E (RLI): `rli_curves.png`
- F (CAI): `cai_waterfall_t*.png`
- G (ALI): `ali_over_time.png`, `ali_hist.png`
- H (cross-version): `version_fingerprint_grid.png`

## 4. Prerequisites and environment
- Python: use repo virtualenv if available: `./venv/bin/python`
- CUDA: required for `*-cuda` scripts on GPU host; CPU fallback is possible but slower
- Working directory: repository root
- Output roots:
  - live telemetry: `outputs/readback-telemetry/`
  - run artifacts: `outputs/readback-baseline/...`

Storage/performance notes:
- Telemetry size scales with `steps`, `nx*ny`, field dump cadence, and downsample factor.
- Heavy knobs:
  - `--steps`
  - `--nx`, `--ny`
  - `--field-interval`
  - `--field-downsample`
  - `--snapshot-interval`

### Before you run anything (preflight)
```bash
# 1) confirm repo root (should show simulation and experiments files)
pwd
ls simulations/active/simulation-v16-cuda.py experiments/scripts/run_readback_tierB.py

# 2) confirm virtualenv python exists (create venv only if missing)
test -x ./venv/bin/python && echo "venv ok" || python -m venv venv

# 3) optional CUDA sanity for cuda lanes
nvidia-smi || true
./venv/bin/python -c "import torch; print('cuda_available=', torch.cuda.is_available())"

# 4) validate schema once
./venv/bin/python experiments/scripts/validate_readback_schema.py
```

### Output path convention
Use a stable output convention for comparability:
- `outputs/<project>/<tier>/<sim>/<seed>/...`
- Keep `<project>` stable across comparable runs (examples: `readback-baseline`, `readback-sweep-2026-02-14`).
- Avoid changing `--out-dir` casually when you intend cross-run comparisons.

Verification output convention:
- Matrix runs: verification artifacts are matrix-level:
- `outputs/readback-baseline/tier6-matrix/verification.json`
- `outputs/readback-baseline/tier6-matrix/verification.md`
- Ad-hoc packet checks: pass explicit `--out-json` and `--out-md` to avoid ambiguous default locations.

## 5. Telemetry: how to enable and configure
Telemetry is emitted by in-loop hooks in each CUDA simulator:
- `simulations/active/simulation-v12-cuda.py`
- `simulations/active/simulation-v13-cuda.py`
- `simulations/active/simulation-v14-cuda.py`
- `simulations/active/simulation-v15-cuda.py`
- `simulations/active/simulation-v16-cuda.py`

Core controls (environment variables):
- `RC_READBACK_HOOKS` (`1` default in sims): enable/disable hook emission
- `RC_READBACK_FIELD_DUMP` (`0` default in sims): enable sparse field dump
- `RC_READBACK_FIELD_DOWNSAMPLE` (`4` sim default; runners often set `2`)
- `RC_READBACK_FIELD_INTERVAL` (`10` sim default; runners often set `1`)
- `RC_READBACK_BETA_RB` (`1.0` default): scales read-back channel
- `RC_NEGCTRL_LAM_SCALE` (`1.0` default): unrelated coefficient perturbation for negative control

Canonical hook points (`experiments/scripts/readback_hooks.py`):
- `post_gradC`
- `post_phi`
- `post_J_preclamp`
- `post_J_postclamp`
- `post_K_raw`
- `post_K_regularized`
- `post_g_preblend`
- `post_g_postblend`
- `post_divergence`
- `post_core_pre_closure`
- `post_closure`

Schema contract:
- `experiments/scripts/readback_schema.py` (`SCHEMA_NAME=ExperienceSample`, `SCHEMA_VERSION=14.1.0`)
- Validator: `experiments/scripts/validate_readback_schema.py`
- Fixtures: `experiments/scripts/readback_schema_fixtures.json`

Mask provenance rule (implemented):
- Active masks are baseline-derived (`beta_rb=0`) and reused in counterfactual comparisons (e.g., OCR_active, ALI_active computations use baseline `|J|` masks).

Common misuse warnings:
- Do not recompute active masks per branch. That invalidates OCR/ALI counterfactual comparisons.
- Do not set `frg_beta2 == beta_rb`; FRG saturation ratio (`FRG2/FRG1`) becomes non-informative.
- Avoid very short Tier B runs for efficacy judgment. If LCE is flat and horizon is short, increase `--steps` before changing physics knobs.

## 6. Scripts reference

### Script: `experiments/scripts/readback_hooks.py`
Purpose: shared in-loop hook recorder and telemetry writer used by simulators.

Inputs:
- Stage payloads from simulators.

Outputs:
- JSONL stream via `enable_jsonl(...)`
- Optional NPZ field dump via `enable_field_dump(...)`

Key parameters/functions:
| Name | Type | Default | Meaning | Common failure mode if mis-set |
|---|---|---:|---|---|
| `InMemoryHookRecorder(enabled=...)` | bool | `True` | master enable | no telemetry emitted |
| `enable_field_dump(..., downsample, every_n_steps)` | int | `4`, `10` | sparse field capture control | huge files / OOM if too dense |
| `CANONICAL_HOOK_POINTS` | list | fixed | completeness contract | missing-stage diagnostics fail |

Examples:
```bash
./venv/bin/python -c "from experiments.scripts.readback_hooks import CANONICAL_HOOK_POINTS; print(CANONICAL_HOOK_POINTS)"
```

Common errors and fixes:
- Missing hook coverage: verify stage emits in simulation loop and check `[HOOKS] missing_steps` output.

### Script: `experiments/scripts/readback_schema.py`
Purpose: canonical `ExperienceSample` schema and alias normalization (`E_* -> T_*_rms`).

Inputs:
- telemetry dict records.

Outputs:
- validation error lists.

Key parameters:
| Name | Type | Default | Meaning | Failure mode |
|---|---|---:|---|---|
| `SCHEMA_VERSION` | str | `14.1.0` | schema lock | mismatch errors |
| `KEY_SPECS` | dict | fixed | required/nullable/type contract | downstream parse drift |
| `ALIASES` | dict | fixed | legacy compatibility map | missing OCR inputs if not normalized |

### Script: `experiments/scripts/validate_readback_schema.py`
Purpose: validate schema fixtures or adapted records.

Inputs:
- `--fixtures` JSON list of records.

Outputs:
- stdout pass/fail; exit code.

Key parameters:
| Flag | Type | Default | Meaning | Failure mode |
|---|---|---:|---|---|
| `--fixtures` | path | `experiments/scripts/readback_schema_fixtures.json` | fixture file | file not found |
| `--non-strict-version` | bool | off | relax version check | can hide version drift |

Examples:
```bash
./venv/bin/python experiments/scripts/validate_readback_schema.py
./venv/bin/python experiments/scripts/validate_readback_schema.py --fixtures experiments/scripts/readback_schema_fixtures.json
```

### Script: `experiments/scripts/run_readback_tierA.py`
Purpose: Tier A counterfactual snapshot engine (A2 default, A1 diagnostic), produces `DeltaJ/DeltaG/DeltaL`, CAI, CAI ratios, plots.

Inputs:
- simulator script (`--sim-script`)
- runtime envelope (`--nx --ny --dx --steps`)
- read-back coupling (`--beta-rb`)

Outputs:
- `<out-dir>/<sim>/a{1|2}-seed-<seed>/`
  - `scores.json`
  - `report.md`
  - `figures/` (A/F plots)
  - `baseline/telemetry.jsonl`, `rb/telemetry.jsonl`, optional fields.

Key parameters:
| Flag | Type | Default | Meaning | Common failure mode |
|---|---|---:|---|---|
| `--sim-script` | str | `simulations/active/simulation-v16-cuda.py` | target simulator | wrong script path |
| `--mode` | `A1|A2` | `A2` | geometry-only vs one-step recompute | confusing interpretation if mislabeled |
| `--steps` | int/None | None | headless steps (`A1->1`, `A2->2` if omitted) | no common snapshot step |
| `--target-step` | int/None | None | force scoring step | out-of-range step gives missing map data |
| `--beta-rb` | float | `1.0` | read-back intervention branch value | tiny deltas if too small |
| `--top-q-pct` | float | `5.0` | active mask top percentile | noisy mask if too high/low |
| `--out-dir` | str | `outputs/readback-baseline/tierA-smoke` | artifact root | hard to compare runs if changed often |
| `--sim-extra-args` | str | `""` | pass-through simulator args | invalid quoted args |

Examples:
```bash
# Default A2 run
./venv/bin/python experiments/scripts/run_readback_tierA.py --sim-script simulations/active/simulation-v16-cuda.py

# A1 diagnostic
./venv/bin/python experiments/scripts/run_readback_tierA.py --sim-script simulations/active/simulation-v16-cuda.py --mode A1 --steps 1

# v15 with soft closure profile
./venv/bin/python experiments/scripts/run_readback_tierA.py \
  --sim-script simulations/active/simulation-v15-cuda.py \
  --sim-extra-args "--closure-mode soft --closure-softness 0.6 --spark-softness 0.08 --collapse-softness 0.5"
```

Common errors and fixes:
- `No hook_stream path found`: run must be headless; check simulator stdout and env.
- Missing `fields.npz`: ensure field dump is enabled (runner sets it automatically).

Success indicators:
- `scores.json` exists with `DeltaJ`, `DeltaG`, `DeltaL`, `CAI`, `CAI_ratios`.
- `report.md` exists.
- `figures/tierA_deltaJ_heatmap_t*.png` and `figures/cai_waterfall_t*.png` exist.

Failure indicators:
- Missing `telemetry.jsonl` branch files: simulator did not emit hook stream.
- Missing Tier A heatmaps: field dump not produced or no common scored step.

### Script: `experiments/scripts/run_readback_tierB.py`
Purpose: Tier B paired trajectories + scoring engine. Generates OCR/FRG/LCE/RLI/ALI, optional negative control, thresholds, readback score, plots, and reports.

Inputs:
- simulator + envelope
- `beta_rb` and optional `frg_beta2`
- optional negative-control settings
- thresholds json/overrides.

Outputs:
- `<out-dir>/<sim>/seed-<seed>/`
  - `scores.json`
  - `series.npz`
  - `report.md`
  - `figures/` (B/C/D/E/G + optional maps)
  - optional `contrast.json`, `contrast.md`
  - branch subdirs `baseline/`, `rb/`, `rb2/`, optional `neg1/`, `neg2/`.

Key parameters:
| Flag | Type | Default | Meaning | Common failure mode |
|---|---|---:|---|---|
| `--sim-script` | str | `simulations/active/simulation-v16-cuda.py` | target simulator | wrong version profile |
| `--steps` | int | `20` | trajectory length | too short to see LCE |
| `--beta-rb` | float | `1.0` | intervention strength | near-zero FRG/LCE |
| `--frg-beta2` | float/None | `None` | second beta for FRG2 (`2*beta_rb` default) | bad FRG ratio if equal to beta1 |
| `--run-negative-control` | bool | off | enable matched lambda perturbation branches | extra runtime |
| `--negctrl-lam-scale-per-beta` | float | `0.10` | lambda scale slope for control branch | unfair control if too large |
| `--active-q-pct` | float | `5.0` | OCR/ALI active mask percentile | unstable active metrics |
| `--rli-origin-q-pct` | float | `5.0` | RLI origin set percentile | noisy `DeltaRLI` |
| `--field-interval` | int | `1` | field cadence for analysis | high storage if too low |
| `--field-downsample` | int | `2` | spatial downsample factor | large files if 1 |
| `--thresholds-json` | path | `""` | load threshold defaults | missing flags if omitted and no overrides |
| `--ocr-active-min` etc. | float | None | threshold overrides | inconsistent decisions if mixed ad hoc |
| `--out-dir` | str | `outputs/readback-baseline/tierB-smoke` | artifact root | fragmented comparison |
| `--sim-extra-args` | str | `""` | simulator options | quoting mistakes |

Examples:
```bash
# Minimal Tier B
./venv/bin/python experiments/scripts/run_readback_tierB.py \
  --sim-script simulations/active/simulation-v16-cuda.py --seed 1 --steps 20

# Tier B with thresholds config
./venv/bin/python experiments/scripts/run_readback_tierB.py \
  --sim-script simulations/active/simulation-v16-cuda.py \
  --thresholds-json experiments/scripts/readback_thresholds.json

# Negative-control contrast
./venv/bin/python experiments/scripts/run_readback_tierB.py \
  --sim-script simulations/active/simulation-v16-cuda.py \
  --run-negative-control \
  --negctrl-lam-scale-per-beta 0.10
```

Common errors and fixes:
- `Tier B requires field dumps`: ensure field dump enabled (runner does this).
- `No overlapping C_field steps`: raise `--steps` or reduce aggressive filters.
- Flat FRG (`FRG2/FRG1` very low): likely clamp/blend-limited regime.

Success indicators:
- `scores.json`, `series.npz`, `report.md` exist.
- `figures/lce_lag_curves.png`, `figures/frg_vs_beta.png`, `figures/ocr_term_energy_stack.png` exist.
- `scores.json` includes `decision`, `flags`, `readback_score`.

Failure indicators:
- Missing `series.npz`: scoring path failed before serialization.
- Missing `lce_lag_curves.png`: branch alignment or field extraction failed.
- `Negative_control_fail=true` with weak FRG/LCE: likely non-read-back-specific response.

### Script: `experiments/scripts/check_readback_stability.py`
Purpose: repeatability check across repeated Tier B runs.

Inputs:
- root containing repeated run `scores.json`.

Outputs:
- `stability.json`, `stability.md`.

Key parameters:
| Flag | Type | Default | Meaning | Failure mode |
|---|---|---:|---|---|
| `--root` | path | required | run root | no scores found |
| `--metrics` | list | built-in metric list | stability panel | missing key counts=0 |
| `--max-cv` | float | `0.10` | CV threshold | too strict false instability |
| `--abs-tol` | float | `1e-5` | near-zero fallback | false fails for tiny metrics |

### Script: `experiments/scripts/run_readback_matrix.py`
Purpose: Iteration-6 matrix orchestrator; runs Tier A + Tier B per version/seed, assembles packet reports, builds cross-version fingerprint grid, runs packet verifier.

Inputs:
- sim script list, seeds, runtime envelope, thresholds, optional negative control.

Outputs:
- `<out-dir>/`
  - `tierA/`, `tierB/`, `packets/`, `logs/`
  - `figures/version_fingerprint_grid.png`
  - `report.md` (matrix summary)
  - `verification.json`, `verification.md` (unless `--skip-verify`)

Key parameters:
| Flag | Type | Default | Meaning | Common failure mode |
|---|---|---:|---|---|
| `--sim-scripts` | csv str | `simulations/active/simulation-v12-cuda.py,...,simulations/active/simulation-v16-cuda.py` | version lanes | includes non-instrumented script |
| `--seeds` | csv str | `1` | multi-seed matrix | long runtime |
| `--steps` | int | `12` | Tier B horizon | weak LCE if too short |
| `--tiera-steps` | int | `2` | Tier A horizon | missing target step |
| `--beta-rb` | float | `1.0` | read-back sweep baseline | tiny effects |
| `--frg-beta2` | float | `2.0` | second beta | poor FRG saturation estimate |
| `--run-negative-control` | bool | off | matrix-level control branches | runtime/storage overhead |
| `--out-dir` | str | `outputs/readback-baseline/tier6-matrix` | output root | compare drift across roots |
| `--thresholds-json` | path | `experiments/scripts/readback_thresholds.json` | decision thresholds | inconsistent flags if omitted |
| `--global-sim-extra-args` | str | `""` | extra simulator args for all lanes | invalid combos per version |
| `--skip-verify` | bool | off | skip packet validation | may miss missing artifacts |

Examples:
```bash
# Default matrix (v12-cuda..v16-cuda, seed 1)
./venv/bin/python experiments/scripts/run_readback_matrix.py

# Multi-seed matrix with negative control
./venv/bin/python experiments/scripts/run_readback_matrix.py \
  --seeds 1,2,3 \
  --run-negative-control \
  --negctrl-lam-scale-per-beta 0.10
```

Success indicators:
- Matrix `report.md` exists.
- `figures/version_fingerprint_grid.png` exists.
- `packets/<sim>/seed-*/report.md` present for each run.
- `verification.json` reports `failed=0` (unless `--skip-verify` used).

Failure indicators:
- Missing packet reports: Tier A/Tier B subrun failed for one lane.
- Missing fingerprint grid: packet figure set incomplete or build stage interrupted.
- Verification failures: missing required report sections or canonical plot filenames.

### Script: `experiments/scripts/verify_readback_audit_packet.py`
Purpose: enforce required packet files, report sections, and canonical figures.

Inputs:
- `--packets-root` and optional `--matrix-root`.

Outputs:
- JSON + markdown verification summaries.

Key parameters:
| Flag | Type | Default | Meaning | Failure mode |
|---|---|---:|---|---|
| `--packets-root` | path | required | packets tree | no runs discovered |
| `--matrix-root` | path | `""` | verify matrix report/grid | false matrix check if omitted |
| `--out-json` | path | `<packets-root>/verification.json` | result json (set explicitly for reproducibility) | write path issues |
| `--out-md` | path | `<packets-root>/verification.md` | result markdown (set explicitly for reproducibility) | write path issues |

Recommended explicit usage:
```bash
./venv/bin/python experiments/scripts/verify_readback_audit_packet.py \
  --packets-root outputs/readback-baseline/tier6-matrix/packets \
  --matrix-root outputs/readback-baseline/tier6-matrix \
  --out-json outputs/readback-baseline/tier6-matrix/verification.json \
  --out-md outputs/readback-baseline/tier6-matrix/verification.md
```

### Wrapper scripts (shell)

#### `experiments/scripts/run_readback_iteration3_tierA.sh`
Purpose: run Tier A `A1` and `A2` in sequence.

Main env-configurable vars:
- `SIM_SCRIPT` (default `simulations/active/simulation-v16-cuda.py`)
- `SEED` (1), `NX` (128), `NY` (128), `DX` (0.1)
- `BETA_RB` (1.0), `OUT_DIR` (`outputs/readback-baseline/tierA-smoke`)
- `SIM_EXTRA_ARGS` (empty)

#### `experiments/scripts/run_readback_iteration4_tierB.sh`
Purpose: one Tier B run with optional `FRG_BETA2` and thresholds config.

Main vars:
- `SIM_SCRIPT` (`simulations/active/simulation-v16-cuda.py`)
- `SEED`, `NX`, `NY`, `DX`, `STEPS` (`20`)
- `BETA_RB` (`1.0`), `FRG_BETA2` (empty uses Tier B default logic)
- `THRESHOLDS_JSON` (`experiments/scripts/readback_thresholds.json`)

#### `experiments/scripts/run_readback_iteration4_stability.sh`
Purpose: repeated Tier B runs + stability aggregation.

Main vars:
- `REPEATS` (`3`), `MAX_CV` (`0.10`), `ABS_TOL` (`1e-5`)
- Default `SIM_EXTRA_ARGS`: `--closure-mode soft --nonlocal-mode off --domain-mode fixed`

#### `experiments/scripts/run_readback_iteration5_negative_control.sh`
Purpose: Tier B with `--run-negative-control` enabled.

Main vars:
- `NEGCTRL_LAM_SCALE_PER_BETA` (`0.10`)
- Default `SIM_EXTRA_ARGS`: `--closure-mode soft --nonlocal-mode off --domain-mode fixed`

#### `experiments/scripts/run_readback_iteration6_matrix.sh`
Purpose: matrix wrapper for `run_readback_matrix.py`.

Main vars:
- `SIM_SCRIPTS` default includes `v12-cuda..v16-cuda`
- `RUN_NEGATIVE_CONTROL` (`1`)
- `OUT_DIR` (`outputs/readback-baseline/tier6-matrix`)

## 7. How to read the results
Use this order to reduce misdiagnosis:
1. Open `report.md`.
2. Check `decision` and `flags`.
3. Check CAI first (`cai_waterfall_t*.png`) to locate attenuation bottleneck.
4. Check FRG (`frg_vs_beta.png`, `frg_saturation_ratio.png`) to confirm immediate sensitivity.
5. Check lag accumulation (`lce_lag_curves.png`).
6. Check ALI (`ali_over_time.png`, `ali_hist.png`) for redundancy vs causal influence.

If you only have 3 minutes: diagnose read-back
- Look at CAI waterfall: where does the signal die?
- Look at FRG sweep + saturation ratio: is the operator sensitive or clamp-limited?
- Look at LCE lag curves: does effect accumulate?
- Look at ALI: is read-back redundant with gradient alignment?

Cheat sheet:
- High `OCR` + low `FRG`: term exists but operator insensitive.
- Low Tier A deltas + nontrivial `LCE`: weak per-step effect that accumulates.
- High `FRG` + high `LCE`: effective read-back regime.
- High `ALI` + low `FRG`: likely redundant read-back channel.
- Low `FRG2/FRG1`: saturation/clamp-limited regime.
- CAI drop at `J` stages: flux clamp bottleneck.
- CAI drop at `K`/`g` stages: regularization/blend bottleneck.

Common patterns:
- Formal-only/suppressed: `FRG` and `LCE` both below thresholds.
- Weak-but-cumulative: moderate/small Tier A effect with nontrivial lag divergence.
- Effective read-back: both local sensitivity and lagged effect are stable.
- Redundant channel: ALI high, FRG weak.
- Clamp-limited: FRG saturation ratio low and CAI attenuation strong.

## 8. Experiment workflows

Recommended minimum settings:
| Workflow | nx x ny | steps | field_interval | field_downsample | seeds |
|---|---:|---:|---:|---:|---:|
| smoke Tier A | 128 x 128 | 2 | 1 | 2 | 1 |
| smoke Tier B | 128 x 128 | 20 | 1 | 2 | 1 |
| efficacy check | 256 x 256 | 50 | 2 | 2-4 | 3 |
| matrix compare | 128 x 128 | 12-20 | 2 | 2 | 1-3 |

### Single run audit
```bash
bash experiments/scripts/run_readback_iteration4_tierB.sh
```
Check:
- `.../scores.json`
- `.../report.md`
- `.../figures/{frg_vs_beta,lce_lag_curves,ocr_over_time}.png`

### Counterfactual snapshot workflow (Tier A)
```bash
bash experiments/scripts/run_readback_iteration3_tierA.sh
```
Check:
- `.../tierA-smoke/<sim>/a2-seed-<seed>/scores.json`
- `figures/tierA_deltaJ_heatmap_t*.png`
- `figures/cai_waterfall_t*.png`

### Twin-trajectory workflow (Tier B)
```bash
./venv/bin/python experiments/scripts/run_readback_tierB.py \
  --sim-script simulations/active/simulation-v16-cuda.py \
  --steps 20 --beta-rb 1.0 --frg-beta2 2.0
```
Check:
- baseline/rb branch subdirs
- `LCE_*` values in `scores.json`
- `lce_lag_curves.png`

### Beta sweep workflow
```bash
./venv/bin/python experiments/scripts/run_readback_tierB.py \
  --sim-script simulations/active/simulation-v16-cuda.py \
  --beta-rb 0.5 --frg-beta2 1.0
```
Sweep by rerunning with different `--beta-rb` values and comparing:
- `FRG1_median`, `FRG2_median`
- `frg_vs_beta.png`

### Cross-version fingerprint workflow
```bash
bash experiments/scripts/run_readback_iteration6_matrix.sh
```
Check:
- `outputs/readback-baseline/tier6-matrix/figures/version_fingerprint_grid.png`
- `outputs/readback-baseline/tier6-matrix/report.md`
- `outputs/readback-baseline/tier6-matrix/verification.md`

## 9. Parameter tuning guidance (audit-first)
Primary knobs:
- `beta_rb`: interventional scaling of the read-back channel. In practice, FRG should respond first; sustained increases then appear in LCE if feedback accumulates.
- `frg_beta2`: second point for saturation check. If too close to `beta_rb`, `FRG2/FRG1` is low-information; if equal, it is effectively meaningless.
- `field_interval` / `field_downsample`: change confidence/resolution of spatial diagnostics (Tier A heatmaps, divergence maps), not model physics.
- `steps`: controls how much accumulation window you observe. Too small a horizon can hide true lag effects.
- simulator clamp/blend args via `--sim-extra-args` (version-dependent): impacts CAI stage attenuation and FRG saturation.

Audit-first interpretation:
- If `FRG` weak and CAI strong attenuation: change clamp/blend regime before increasing `beta_rb`.
- If `ALI` very high: boosting `beta_rb` may only amplify redundant directionality.
- If `LCE` noisy: increase steps or seeds before changing physics.

Warning:
- Do not optimize only `readback_score`; it is labeled as a ranking heuristic, not a truth predicate.

## 10. Troubleshooting
- Missing telemetry files:
  - Symptom: `No hook_stream path found`.
  - Fix: run headless simulator path through tier runners; ensure hooks enabled.
- Missing required figures/sections:
  - Run verifier:
  ```bash
  ./venv/bin/python experiments/scripts/verify_readback_audit_packet.py \
    --packets-root outputs/readback-baseline/tier6-matrix/packets \
    --matrix-root outputs/readback-baseline/tier6-matrix \
    --out-json outputs/readback-baseline/tier6-matrix/verification.json \
    --out-md outputs/readback-baseline/tier6-matrix/verification.md
  ```
- OOM or huge files:
  - reduce `--nx/--ny`, `--steps`
  - increase `--field-downsample`
  - increase `--field-interval`
- Non-reproducible decision:
  - run stability harness and inspect `stability.md`.
- Version mismatch behavior:
  - apply lane profile args:
    - v14: `--closure-softness 0.6 --spark-softness 0.08 --collapse-softness 0.5`
    - v15: `--closure-mode soft --closure-softness 0.6 --spark-softness 0.08 --collapse-softness 0.5`
    - v16: `--closure-mode soft --nonlocal-mode off --domain-mode fixed`

## 11. Appendix

### Glossary
- Tier A: one-step/interventional local sensitivity (`DeltaJ`, `DeltaG`, `DeltaL`).
- Tier B: paired-trajectory lag divergence (`LCE`).
- OCR: read-back term dominance relative to gradient terms.
- FRG: flux response gain to read-back perturbation.
- LCE: lagged causal effect (trajectory divergence).
- RLI: return/loop index trend.
- CAI: attenuation localization across stages.
- ALI: alignment/redundancy indicator.

### Artifact inventory
- Core run files:
  - `scores.json`
  - `series.npz` (Tier B / matrix packets)
  - `report.md`
  - `figures/*.png`
- Branch internals:
  - `baseline/run.log`, `baseline/telemetry.jsonl`, optional `baseline/fields.npz`
  - `rb/...`, optional `rb2/...`, optional `neg1/...`, `neg2/...`
- Matrix:
  - `report.md`
  - `figures/version_fingerprint_grid.png`
  - `verification.json`, `verification.md`

### Default thresholds
From `experiments/scripts/readback_thresholds.json`:
- `OCR_active_min = 0.05`
- `FRG_min = 0.005`
- `FRG_saturation_ratio_min = 0.6`
- `ALI_redundant_min = 0.9`
- `LCE_auc_min = 0.01`
- `seed_var_max = 0.1`

### Planned / Not Implemented Yet
- `experiments/scripts/score_readback.py`: referenced in checklist intent, not present as a separate file.
  - Use instead: `experiments/scripts/run_readback_tierB.py` (`scores.json`, `series.npz`, figures).
- `plot_readback.py`: no standalone script present.
  - Use instead: built-in plotting from `run_readback_tierA.py`, `run_readback_tierB.py`, and `run_readback_matrix.py`.
- Optional correlation-drop secondary metric: checklist item remains open.
- Closure-mode stratified matrix gate: checklist item remains open.
- Legacy `simulations/legacy/simulation-v12.py` is not instrumented with read-back hooks; operational audit runs use `simulations/active/simulation-v12-cuda.py` through `simulations/active/simulation-v16-cuda.py`.
