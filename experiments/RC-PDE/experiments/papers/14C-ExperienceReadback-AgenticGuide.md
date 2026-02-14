# 14C — Experience Read-Back Agentic Guide

## Purpose
This guide is the execution protocol for agents running Paper 14 tooling. It defines how to:
- validate that read-back code paths are operational,
- run Tier A/Tier B/matrix simulations,
- evaluate artifacts and report conclusions consistently.

Scope is `simulations/active/simulation-v12-cuda.py` through `simulations/active/simulation-v16-cuda.py` plus the `experiments/scripts/readback_*` toolchain.

Interpretation contract companion:
- `experiments/papers/14E-ExperienceReadback-InterpretationGuide.md`

Path policy:
- Use canonical paths only (`simulations/active/...`, `experiments/scripts/...`).
- Legacy symlink paths were removed; do not use non-canonical paths in agent outputs.

## Agent rules
- Run from repo root.
- Do not invent script names/flags; use only code-defined CLI/env vars.
- Do not edit unrelated files.
- Do not mutate the system under test during evaluation runs:
  - do not change thresholds/defaults/wrapper scripts mid-run,
  - if changes are required, use a new run label and record diffs explicitly in notes.
- Prefer wrapper scripts for reproducibility, then per-script CLI for targeted diagnostics.
- For CUDA-host runs in restricted environments, run outside sandbox when required.

## Canonical scripts
- Schema/contract:
  - `experiments/scripts/readback_schema.py`
  - `experiments/scripts/validate_readback_schema.py`
- Tier engines:
  - `experiments/scripts/run_readback_tierA.py`
  - `experiments/scripts/run_readback_tierB.py`
- Wrappers:
  - `experiments/scripts/run_readback_ci_smoke.sh`
  - `experiments/scripts/run_readback_iteration3_tierA.sh`
  - `experiments/scripts/run_readback_iteration4_tierB.sh`
  - `experiments/scripts/run_readback_iteration4_stability.sh`
  - `experiments/scripts/run_readback_iteration5_negative_control.sh`
  - `experiments/scripts/run_readback_iteration6_matrix.sh`
- Matrix/report/verification:
  - `experiments/scripts/run_readback_matrix.py`
- `experiments/scripts/verify_readback_audit_packet.py`
  - `experiments/scripts/check_readback_stability.py`

## Single source of truth (defaults)
- Threshold defaults: `experiments/scripts/readback_thresholds.json`.
- Baseline output roots: `outputs/readback-baseline/...`.
- Default sim lane in wrappers: `simulations/active/simulation-v16-cuda.py` unless overridden via `SIM_SCRIPT`.
- Matrix wrapper defaults:
  - `SIM_SCRIPTS=simulations/active/simulation-v12-cuda.py,simulations/active/simulation-v13-cuda.py,simulations/active/simulation-v14-cuda.py,simulations/active/simulation-v15-cuda.py,simulations/active/simulation-v16-cuda.py`
  - `OUT_DIR=outputs/readback-baseline/tier6-matrix`

Verification output convention:
- Matrix runs: verification artifacts live at matrix root:
  - `outputs/readback-baseline/tier6-matrix/verification.json`
  - `outputs/readback-baseline/tier6-matrix/verification.md`
- Ad-hoc packet checks (for smoke runs): set explicit `--out-json/--out-md` paths (recommended), do not rely on implicit defaults.

## Phase 0: Preflight
```bash
pwd
ls simulations/active/simulation-v16-cuda.py experiments/scripts/run_readback_tierB.py
test -x ./venv/bin/python && echo "venv ok" || python -m venv venv
./venv/bin/python -c "import torch; print('cuda_available=', torch.cuda.is_available())"
./venv/bin/python experiments/scripts/validate_readback_schema.py
```

Hard fail conditions:
- Missing `./venv/bin/python` and no fallback python.
- Schema validator non-zero exit.

Soft fail (proceed with warning):
- CUDA unavailable: run CPU-capable lane only if supported and mark results as non-comparable to CUDA baselines.
- Missing optional divergence maps (`lce_divergence_map_tau*.png`): continue.
- Missing matrix grid/report checks: continue only if matrix phase was intentionally skipped.

## Phase 1: Code-path validation (fast)
Run static checks for the audit scripts before runtime:
```bash
./venv/bin/python -m py_compile experiments/scripts/readback_hooks.py
./venv/bin/python -m py_compile experiments/scripts/run_readback_tierA.py
./venv/bin/python -m py_compile experiments/scripts/run_readback_tierB.py
./venv/bin/python -m py_compile experiments/scripts/run_readback_matrix.py
./venv/bin/python -m py_compile experiments/scripts/verify_readback_audit_packet.py
```

Pass criteria:
- All commands exit `0`.

## Phase 2: Tier A smoke (local counterfactual)
```bash
bash experiments/scripts/run_readback_iteration3_tierA.sh
```

Expected outputs:
- `outputs/readback-baseline/tierA-smoke/<sim>/a1-seed-<seed>/scores.json`
- `outputs/readback-baseline/tierA-smoke/<sim>/a2-seed-<seed>/scores.json`
- Tier A figures (`tierA_delta*.png`, `cai_waterfall_t*.png`)

Minimum validation:
- `scores.json` contains `DeltaJ`, `DeltaG`, `DeltaL`, `CAI`, `CAI_ratios`.

## Phase 3: Tier B smoke (paired trajectories + scoring)
```bash
bash experiments/scripts/run_readback_iteration4_tierB.sh
```

Expected outputs:
- `outputs/readback-baseline/tierB-smoke/<sim>/seed-<seed>/scores.json`
- `outputs/readback-baseline/tierB-smoke/<sim>/seed-<seed>/series.npz`
- `outputs/readback-baseline/tierB-smoke/<sim>/seed-<seed>/report.md`
- figures including:
  - `ocr_term_energy_stack.png`
  - `ocr_over_time.png`
  - `frg_vs_beta.png`
  - `frg_saturation_ratio.png`
  - `lce_lag_curves.png`
  - `rli_curves.png`
  - `ali_over_time.png`
  - `ali_hist.png`

Minimum validation:
- `scores.json` contains `decision`, `flags`, `readback_score`.

Smoke packet verification (recommended immediately after Tier B smoke):
```bash
./venv/bin/python experiments/scripts/verify_readback_audit_packet.py \
  --packets-root outputs/readback-baseline/tierB-smoke \
  --out-json outputs/readback-baseline/tierB-smoke/verification.json \
  --out-md outputs/readback-baseline/tierB-smoke/verification.md
```

## Phase 4: Stability and negative-control gates
Stability:
```bash
bash experiments/scripts/run_readback_iteration4_stability.sh
```
Check:
- `.../tierB-stability/<sim>/seed-<seed>/stability.json`
- `overall_stable=true` preferred.

Negative control:
```bash
bash experiments/scripts/run_readback_iteration5_negative_control.sh
```
Check:
- `contrast.json`, `contrast.md` present.
- `scores.json` includes `negative_control_pass` and `flags.Negative_control_fail`.

## Phase 5: Cross-version matrix and packet verification
```bash
bash experiments/scripts/run_readback_iteration6_matrix.sh
```

Expected outputs:
- `outputs/readback-baseline/tier6-matrix/report.md`
- `outputs/readback-baseline/tier6-matrix/figures/version_fingerprint_grid.png`
- `outputs/readback-baseline/tier6-matrix/verification.json`
- `outputs/readback-baseline/tier6-matrix/verification.md`
- per-run packets:
  - `outputs/readback-baseline/tier6-matrix/packets/<sim>/seed-<seed>/scores.json`
  - `outputs/readback-baseline/tier6-matrix/packets/<sim>/seed-<seed>/report.md`
  - `outputs/readback-baseline/tier6-matrix/packets/<sim>/seed-<seed>/figures/...`

Minimum matrix pass:
- `verification.json` has `failed=0`.

## Evaluation protocol (agent decision logic)
Use this ordering:
1. `report.md` decision/flags.
2. CAI bottleneck (`cai_waterfall_t*.png`) for suppression localization.
3. FRG sensitivity (`frg_vs_beta.png`, `frg_saturation_ratio.png`).
4. LCE accumulation (`lce_lag_curves.png`).
5. ALI redundancy (`ali_over_time.png`, `ali_hist.png`).

Interpretation patterns:
- `Formal-only/suppressed`: FRG and LCE below thresholds.
- `Weak-effective`: one of FRG/LCE is nontrivial, or both marginal.
- `Effective`: FRG and LCE pass thresholds, no dominant suppression/redundancy flags.
- `Redundant channel`: ALI high + FRG low.
- `Clamp-limited`: FRG saturation ratio low + strong CAI attenuation.

## Metrics and flags to report
Always report:
- `decision`
- `OCR_active_median_rb`, `OCR_global_median_rb`
- `FRG1_median`, `FRG2_median`, `FRG_saturation_ratio`
- `LCE_C_auc`, `LCE_C_max`
- `DeltaRLI_auc`, `DeltaRLI_mean`
- `ALI_active_median_rb`, `ALI_global_median_rb`
- `readback_score` (label as ranking heuristic)
- `flags.*` including `Negative_control_fail`

Threshold source:
- `experiments/scripts/readback_thresholds.json`

## Failure triage
- `No hook_stream path found`:
  - check headless run path and hook env wiring.
- Missing Tier B `series.npz`:
  - scoring path failed before serialization; inspect `run.log`.
- Missing required packet plots/sections:
  - rerun verifier:
  ```bash
  ./venv/bin/python experiments/scripts/verify_readback_audit_packet.py \
    --packets-root outputs/readback-baseline/tier6-matrix/packets \
    --matrix-root outputs/readback-baseline/tier6-matrix
  ```
- Flat LCE with short runs:
  - increase `--steps` before changing physics.
- FRG ratio meaningless:
  - avoid `frg_beta2 == beta_rb`.

## Agent output template
When reporting completion, include:
1. Commands executed (or wrapper names).
2. Artifact roots produced.
3. Verification status (`failed=0` or exact failures).
4. Key metrics summary table for main run(s).
5. Classification (`Formal-only` / `Weak-effective` / `Effective`) and why.
6. Any blocked/pending items (for example: closure-stratified gate not yet implemented).

Minimum reporting bundle table:
| Item | Value |
|---|---|
| decision | |
| CAI bottleneck | |
| FRG1/FRG2/ratio | |
| LCE_C AUC/max | |
| OCR_active/global | |
| ALI_active/global | |
| negative_control_pass | |

## Planned / not implemented (must not be fabricated)
- No standalone `experiments/scripts/score_readback.py`; scoring lives in `experiments/scripts/run_readback_tierB.py`.
- No standalone `plot_readback.py`; plotting lives inside Tier A/Tier B/matrix scripts.
- Optional correlation-drop secondary metric is still pending.
- Closure-mode stratified matrix gate is still pending.
