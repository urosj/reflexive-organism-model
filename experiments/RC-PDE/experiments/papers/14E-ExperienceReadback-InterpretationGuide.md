# 14E — Experience Read-Back Results Interpretation Guide (Agent Protocol)

## Purpose
This document standardizes how agents interpret Paper-14 outputs when a user points to a results folder.

Primary goal:
- produce consistent, evidence-based interpretation from existing artifacts (`scores.json`, `report.md`, figures, optional matrix packet).

Non-goals:
- do not re-derive physics from scratch.
- do not replace the scorer with ad-hoc metrics.
- do not claim confidence levels that the artifacts do not support.

## Scope
Applies to outputs produced by:
- `experiments/scripts/run_readback_tierA.py`
- `experiments/scripts/run_readback_tierB.py`
- `experiments/scripts/run_readback_matrix.py`

Canonical output roots:
- single-run Tier A: `outputs/readback-baseline/tierA-*`
- single-run Tier B: `outputs/readback-baseline/tierB-*`
- matrix: `outputs/readback-baseline/tier6-matrix`

## Input Modes
An agent may be given one of:
- Tier A run directory (contains `scores.json` + Tier A figures)
- Tier B run directory (contains `scores.json`, `series.npz`, `report.md`, figures)
- matrix root (`report.md`, `version_fingerprint_grid.png`, packets subtree, verification files)

If input type is ambiguous:
- inspect file presence first, then classify mode from artifact set.

## Artifact Completeness Checks
Before interpretation, verify files.

Tier A minimum:
- `scores.json`
- `figures/tierA_deltaJ_heatmap_t*.png`
- `figures/tierA_deltaG_heatmap_t*.png`
- `figures/tierA_delta_distributions_t*.png`
- `figures/cai_waterfall_t*.png`

Tier B minimum:
- `scores.json`
- `series.npz`
- `report.md`
- `figures/ocr_term_energy_stack.png`
- `figures/ocr_over_time.png`
- `figures/frg_vs_beta.png`
- `figures/frg_saturation_ratio.png`
- `figures/lce_lag_curves.png`
- `figures/rli_curves.png`
- `figures/ali_over_time.png`
- `figures/ali_hist.png`

Matrix minimum:
- `report.md`
- `figures/version_fingerprint_grid.png`
- `verification.json`
- `verification.md`
- `packets/<sim>/seed-<seed>/scores.json` for each compared lane

If required artifacts are missing:
- report interpretation as partial.
- do not produce final categorical judgment unless missing pieces are non-critical and clearly stated.

### Completeness Pass/Fail Rules
Use this classification before issuing a categorical judgment.

Hard fail (no categorical judgment):
- missing `scores.json` in the target packet.
- missing both core causal plots in Tier B packet: `figures/frg_vs_beta.png` and `figures/lce_lag_curves.png`.
- missing `report.md` and no equivalent structured summary source.
- for Tier A-based attenuation claims, missing `figures/cai_waterfall_t*.png`.

Soft fail (judgment allowed with explicit caveat):
- missing optional divergence maps (`lce_divergence_map_tau*.png`).
- missing matrix grid figure when not in matrix interpretation mode.
- missing optional negative-control artifacts when run was not configured with negative control.

## Canonical Interpretation Order
Always interpret in this order:
1. Artifact completeness + mode detection.
2. Decision and flags from `scores.json` / `report.md`.
3. CAI/attenuation (if Tier A or available via packet figures).
4. FRG sensitivity and saturation.
5. LCE accumulation.
6. RLI loop/return behavior.
7. ALI redundancy/alignment context.
8. Negative control status (if available).
9. Reproducibility and scope caveats (single seed, closure mode, run envelope).

Reason:
- this order localizes whether read-back exists, whether it survives numerics, and whether it accumulates.

## Tier B Decision Logic (Exact)
Use scorer outputs as authoritative. The implemented logic in `run_readback_tierB.py` is:

Definitions:
- `frg_pass = FRG1_median >= FRG_min`
- `lce_pass = LCE_C_auc >= LCE_auc_min`
- `RB_dominated = OCR_active_median_rb < OCR_active_min`
- `Saturation_clamp = FRG_saturation_ratio < FRG_saturation_ratio_min`
- `Redundant_channel = (ALI_active_median_rb >= ALI_redundant_min) AND (FRG1_median < FRG_min)`

Decision mapping:
- `Formal-only` if `not frg_pass AND not lce_pass`
- `Effective` if `frg_pass AND lce_pass AND not Saturation_clamp AND not Redundant_channel`
- `Weak-effective` otherwise
- downgrade rule:
  - if negative control ran and `negative_control_pass == false`, then `Effective` is downgraded to `Weak-effective`

Important:
- source of truth precedence:
  - prefer `scores.json.decision`, `scores.json.flags`, and `scores.json.thresholds` over manual re-evaluation from this text.
  - use textual logic only for explanation and consistency checks.
- thresholds come from `scores.json.thresholds` (or scorer defaults if absent).
- do not substitute custom thresholds unless explicitly requested.

## How to Read Core Metrics
Use these interpretations:

- `OCR_active_median_rb`:
  - read-back structural presence in active regions.
  - very low values with nonzero FRG/LCE indicate causal effect despite low dominance.

- `FRG1_median`, `FRG2_median`, `FRG_saturation_ratio`:
  - read-back sensitivity and saturation behavior.
  - low ratio suggests clamp/blend/regularization saturation.

- `LCE_C_auc`, `LCE_C_max`:
  - lagged divergence accumulation.
  - AUC is integrated over horizon; interpret together with max/mean behavior from series.

- `DeltaRLI_mean`, `DeltaRLI_auc`:
  - return-loop shift sign and magnitude under read-back intervention.
  - negative means less return to origin-set compared to baseline.

- `ALI_active_median_rb`, `ALI_global_median_rb`:
  - alignment/redundancy context.
  - high ALI alone is not failure; pair with FRG.

Mask provenance requirement:
- active-mask metrics (`OCR_active`, `ALI_active`) must be based on baseline-derived `M_t` and reused across counterfactual branches.
- if packet provenance indicates branch-specific masks, mark interpretation low-confidence.

## Interpretation Signatures (Fast Pattern Map)
Use these signatures to write concise narrative judgments:

- Suppressed by clamp/blend:
  - FRG low or saturation ratio low, plus CAI drop at clamp/regularization/blend stage.
- Weak-but-cumulative:
  - Tier A deltas small, but `LCE_C_auc`/`LCE_C_max` nontrivial and stable.
- Redundant channel:
  - ALI high and FRG low (OCR may be nontrivial but causal response remains weak).
- Closure-dependent efficacy:
  - strong efficacy in closure-on profile that collapses in closure-soft/off profile under matched envelope.
- Formal-only:
  - both FRG and LCE below thresholds.
- Effective:
  - FRG and LCE pass, without dominant saturation/redundancy flags, and preferably with negative-control pass.

## Tier A Interpretation Rules
Tier A is local counterfactual evidence ("right now"):
- emphasize `DeltaJ`, `DeltaG`, `DeltaL` first.
- then use CAI stage deltas/ratios to localize suppression:
  - J clamp loss: `DeltaJ_postClamp << DeltaJ_preClamp`
  - K regularization loss: `DeltaK_reg << DeltaK_raw`
  - blend loss: `DeltaG_postBlend << DeltaG_preBlend`

Tier A alone:
- sufficient for diagnosing immediate efficacy and attenuation location.
- insufficient for final accumulation claim (needs Tier B/LCE).

## Matrix Interpretation Rules
For matrix outputs:
- summarize per-version medians from packet `scores.json`.
- compare mechanism fingerprints, not just visual similarity.
- include:
  - OCR active/global
  - FRG1/FRG2/ratio
  - LCE_C AUC/max
  - DeltaRLI AUC
  - ALI active/global
  - decision + key flags

Do not claim cross-version superiority if:
- seeds differ, or
- run envelope differs materially (`nx/ny/steps/closure-mode/nonlocal/domain settings`).

Run envelope comparability checklist (required before `vX > vY` claims):
- same `nx`, `ny`, `dx`, and step horizon.
- same closure configuration (`closure-mode` and key closure softness knobs if used).
- same nonlocal/domain toggles and strengths.
- same beta sweep settings (`beta_rb`, `frg_beta2`, negative-control settings).
- seed sets overlap or are explicitly normalized.
- if any item fails, label comparison as `not controlled`.

## Confidence / Caveat Rules
Always declare caveats explicitly.

High-confidence interpretation requires:
- complete Tier B packet
- negative control executed
- at least multi-seed comparison or matrix verification

Medium-confidence:
- complete Tier B packet, single seed, no negative control

Low-confidence:
- missing required artifacts, or Tier A-only evidence used for trajectory claims

Always mark these as unknown when absent:
- `Negative_control_fail`
- `Non_reproducible`
- `Closure_dependent`

## Standard Agent Output Template
When user asks "interpret these results", respond in this structure:

1. Run identity
- path, simulator, seed(s), steps, major mode flags (closure/nonlocal/domain)

2. Completeness
- packet completeness pass/fail
- missing artifacts (if any)

3. Key metrics
- compact table:
  - decision
  - OCR_active/global medians
  - FRG1/FRG2/ratio
  - LCE_C AUC/max
  - DeltaRLI mean/AUC
  - ALI active/global medians
  - negative-control status

4. Interpretation
- one paragraph on efficacy class (`Formal-only`, `Weak-effective`, `Effective`)
- one paragraph on suppression mechanism (CAI/saturation/redundancy)
- one paragraph on loop behavior (`DeltaRLI` sign and implication)

5. Confidence + caveats
- confidence level (high/medium/low)
- specific missing evidence

6. Next actions
- 1-3 concrete runs (for example add negative control, run multi-seed matrix, rerun with adjusted damping)

## Recommended Next-Action Recipes (Copy/Paste)
Negative-control rerun for current Tier B packet style:

```bash
./venv/bin/python experiments/scripts/run_readback_tierB.py \
  --sim-script simulations/active/simulation-v16-cuda.py \
  --seed 1 --nx 512 --ny 512 --dx 0.1 --steps 2000 \
  --snapshot-interval 50 --storage-mode memory \
  --beta-rb 1.0 --frg-beta2 2.0 \
  --run-negative-control \
  --sim-extra-args "--closure-mode soft --nonlocal-mode on --operator-diagnostics --operator-diagnostics-interval 10 --domain-mode adaptive --domain-adapt-strength 0.30 --domain-adapt-interval 50" \
  --out-dir outputs/readback-baseline/tierB-v16-negctrl
```

Multi-seed controlled matrix (v14-v16):

```bash
./venv/bin/python experiments/scripts/run_readback_matrix.py \
  --sim-scripts simulations/active/simulation-v14-cuda.py,simulations/active/simulation-v15-cuda.py,simulations/active/simulation-v16-cuda.py \
  --seeds 1,2,3 --nx 512 --ny 512 --dx 0.1 \
  --steps 2000 --tiera-steps 2 --snapshot-interval 50 --storage-mode memory \
  --beta-rb 1.0 --frg-beta2 2.0 \
  --global-sim-extra-args "--closure-mode soft --nonlocal-mode on --operator-diagnostics --operator-diagnostics-interval 10 --domain-mode adaptive --domain-adapt-strength 0.30 --domain-adapt-interval 50" \
  --out-dir outputs/readback-baseline/tier6-v14-v16-seeds123
```

Longer-horizon Tier B (for accumulation confirmation):

```bash
./venv/bin/python experiments/scripts/run_readback_tierB.py \
  --sim-script simulations/active/simulation-v16-cuda.py \
  --seed 1 --nx 512 --ny 512 --dx 0.1 --steps 4000 \
  --snapshot-interval 100 --storage-mode memory \
  --beta-rb 1.0 --frg-beta2 2.0 \
  --sim-extra-args "--closure-mode soft --nonlocal-mode on --operator-diagnostics --operator-diagnostics-interval 10 --domain-mode adaptive --domain-adapt-strength 0.30 --domain-adapt-interval 50" \
  --out-dir outputs/readback-baseline/tierB-v16-long
```

## Anti-Misread Rules
- Do not treat `readback_score` as a truth label; it is explicitly a ranking heuristic.
- Do not call negative control pass/fail if it is `null`.
- Do not over-interpret single-seed outcomes as stable regime claims.
- Do not compare runs with different envelopes as if they were controlled.
- Do not infer Tier B accumulation from Tier A-only packet.

## Quick Commands for Interpretation Sessions
Given `RUN_DIR=<tierB packet dir>`:

```bash
RUN_DIR="outputs/readback-baseline/tierB-smoke/simulation-v16-cuda/seed-1"
ls "$RUN_DIR"
sed -n '1,200p' "$RUN_DIR/report.md"
cat "$RUN_DIR/scores.json"
```

Matrix check:

```bash
MATRIX_DIR="outputs/readback-baseline/tier6-matrix"
ls "$MATRIX_DIR"
sed -n '1,200p' "$MATRIX_DIR/report.md"
cat "$MATRIX_DIR/verification.json"
```

Use this guide as the interpretation contract across sessions and agents.
