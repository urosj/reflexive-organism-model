# **RC Read-Back Audit Implementation Checklist (v12-v16)**

Companion to `experiments/papers/14-ExperienceReadback-GapAndPlan.md`.

Path policy:
- This checklist uses canonical paths (`simulations/active/...`, `experiments/scripts/...`).
- Legacy symlink aliases were removed; non-canonical paths should be treated as invalid.

---

## **0) Baseline Freeze and Artifact Contract**

- [x] Freeze reference runs for v12, v12-cuda, v13-cuda, v14-cuda, v15-cuda, v16-cuda on aligned seeds/grids.
- [x] Record current diagnostics availability per version (which read-back proxies already exist).
- [x] Define canonical output layout for audit artifacts (run metadata, telemetry stream, optional field products).
- [x] Lock Audit Packet deliverables per run: `scores.json`, optional `series.npz`, `figures/`, and required deterministic `report.md`.
- [x] Archive baseline summaries for direct post-instrumentation deltas.

**Done when:** aligned baseline pack is reproducible and version-tagged.

### **Frozen Baseline Matrix (14A-1.1)**

This matrix is frozen for Paper-14 baseline comparisons.

Global aligned envelope (for comparable runs):

- seeds: `1,2,3`
- grid: `nx=512`, `ny=512`, `dx=0.1`
- horizon: `--headless --headless-steps 2000`
- snapshots: `--snapshot-interval 50`
- storage: `--storage-mode memory`

Version matrix:

| Version lane | Script | Frozen command profile |
|---|---|---|
| `v12` (legacy anchor) | `simulations/legacy/simulation-v12.py` | `python simulations/legacy/simulation-v12.py` (non-CLI legacy; excluded from strict aligned seed/grid comparisons) |
| `v12-cuda` | `simulations/active/simulation-v12-cuda.py` | `python simulations/active/simulation-v12-cuda.py --headless --headless-steps 2000 --nx 512 --ny 512 --dx 0.1 --seed <seed> --snapshot-interval 50 --storage-mode memory` |
| `v13-cuda` | `simulations/active/simulation-v13-cuda.py` | `python simulations/active/simulation-v13-cuda.py --headless --headless-steps 2000 --nx 512 --ny 512 --dx 0.1 --seed <seed> --snapshot-interval 50 --storage-mode memory` |
| `v14-cuda` | `simulations/active/simulation-v14-cuda.py` | `python simulations/active/simulation-v14-cuda.py --headless --headless-steps 2000 --nx 512 --ny 512 --dx 0.1 --seed <seed> --snapshot-interval 50 --storage-mode memory --closure-softness 0.6 --spark-softness 0.08 --collapse-softness 0.5` |
| `v15-cuda` | `simulations/active/simulation-v15-cuda.py` | `python simulations/active/simulation-v15-cuda.py --headless --headless-steps 2000 --nx 512 --ny 512 --dx 0.1 --seed <seed> --snapshot-interval 50 --storage-mode memory --closure-mode soft --closure-softness 0.6 --spark-softness 0.08 --collapse-softness 0.5` |
| `v16-cuda` | `simulations/active/simulation-v16-cuda.py` | `python simulations/active/simulation-v16-cuda.py --headless --headless-steps 2000 --nx 512 --ny 512 --dx 0.1 --seed <seed> --snapshot-interval 50 --storage-mode memory --closure-mode soft --nonlocal-mode off --domain-mode fixed` |

Frozen artifact layout:

`outputs/readback-baseline/<version>/seed-<seed>/`

- `run.log`
- `metadata.json` (run envelope, script hash, seed, version lane)
- `snapshots/` (existing simulator snapshot pipeline)
- `telemetry.jsonl` or `telemetry.npz` (introduced in Iteration 2)
- `scores.json`, optional `series.npz`, `figures/`, and `report.md` (introduced in Iteration 3+)

---

## **1) Canonical Telemetry Schema (Shared Contract)**

- Contract file: `experiments/scripts/readback_schema.py` (`SCHEMA_NAME=ExperienceSample`, `SCHEMA_VERSION=14.1.0`).
- Validator: `experiments/scripts/validate_readback_schema.py`.
- Fixture set: `experiments/scripts/readback_schema_fixtures.json` (v12, v12-cuda, v13-cuda, v14-cuda, v15-cuda, v16-cuda samples).
- [x] Add schema contract module/docs for `ExperienceSample` (keys, units, fallback policy, schema version).
- [x] Lock mandatory keys: `J_*`, `T_rb_*`, `T_grad_rms`, `OCR inputs`, context metrics, run metadata.
- [x] Define optional keys (`T_den_rms`, field products) and nullability rules.
- [x] Lock term-naming compatibility rule: canonical telemetry uses `T_*_rms`; if legacy `E_*` names appear, scorer maps `E_* := T_*_rms` (no duplicate semantics).
- [x] Lock active-mask provenance rule: `M_t` is computed from baseline branch and reused for all counterfactual comparisons.
- [x] Add schema validator used by offline tools.

**Acceptance**
- Every version emits schema-valid records.
- Missing quantities are explicit (`null`/fallback), never implicit.

---

## **2) In-Loop Hook Points and Telemetry Capture**

- [x] Instrument canonical hook points:
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
  - `post_closure` (when applicable)
- [x] Compute scalar telemetry in-loop with minimal additional sync.
- [x] Keep simulation logic unchanged (audit-only instrumentation path).

**Acceptance**
- Hook completeness passes for all instrumented versions.
- Runtime overhead remains within agreed budget.

---

## **3) Dedicated Telemetry Storage Stream**

- [x] Persist scalar telemetry to dedicated stream (`telemetry.jsonl` or `telemetry.npz`).
- [x] Store optional downsampled fields (`|J|`, `T_rb_trace`) on sparse cadence.
- [x] Keep existing snapshot output pipeline intact and decoupled.
- [x] Add stable run metadata record (`schema_version`, sim version, seed, grid, mode).

**Acceptance**
- Offline tools can run from telemetry stream alone.
- Storage overhead is measured and bounded.

---

## **4) Tier A Counterfactual Engine (Interventional Local Test)**

- [x] Implement Tier A mode `A2` as default:
  - branch `beta_rb=0` vs `beta_rb>0`
  - rebuild one-step operator stack (including `phi`) without full trajectory advance.
- [x] Lock read-back coupling placement across versions:
  - `K_rb <- beta_rb * zeta_flux * (J ⊗ J)` (equivalently `zeta_eff = beta_rb * zeta_flux`).
- [x] Implement Tier A mode `A1` as diagnostic:
  - hold `J_t` fixed, propagate through geometry/operator branch only.
- [x] Emit `DeltaJ`, `DeltaG`, `DeltaL` from same fixed state snapshot.
- [x] Define `L` proxy telemetry contract for `DeltaL`:
  - `L_proxy_scalar` (RMS summaries of transport operator coefficients),
  - optional `L_proxy_field` (downsampled coefficient field),
  - compute `DeltaL` on whichever proxy is emitted.
- [x] Add baseline-mask rule for active-region metrics (`M_t` from baseline branch, reused in both branches).

**Acceptance**
- Tier A outputs are reproducible for fixed state/seed.
- `A1` and `A2` are explicitly labeled in artifacts.

---

## **5) Tier B Counterfactual Engine (Lagged Experience Test)**

- [x] Run paired trajectories (`beta_rb=0` vs `beta_rb>0`) with identical initial state and RNG path.
- [x] Emit lagged divergence curves (`LCE_C(tau)`, optional `LCE_J/LCE_g`).
- [ ] Emit optional correlation-drop secondary metric.
- [x] Align Tier B sampling cadence with telemetry intervals.

**Acceptance**
- Lagged curves are reproducible across reruns.
- Tier B artifacts are comparable across versions.

---

## **6) Read-Back Metrics and Offline Scoring**

- [x] Implement offline scoring engine using canonical schema (`run_readback_tierB.py` scorer path).
- [x] Compute and report: `OCR_global`, `OCR_active`, `FRG1`, `FRG2`, saturation ratio, `LCE`, `RLI`, `ALI`, `CAI`.
- [x] Enforce FRG two-delta default policy (`delta_abs_min`, `delta1`, `delta2`).
- [x] Implement threshold-configurable anomaly flags (`OCR_active_min`, `FRG_min`, `FRG_saturation_ratio_min`, `ALI_redundant_min`, `LCE_auc_min`, `seed_var_max`).
- [x] Emit composite `readback_score` with explicit "ranking heuristic" label.
- [x] Emit canonical Audit Packet plots with stable filenames for sets A-H.
- [x] Generate required deterministic `report.md` from the template (metadata, parameter table, executive summary, flags, canonical plot links, interpretation checklists).

**Acceptance**
- One command scores any schema-valid run.
- Metric panel is stable across versions and seeds.
- Audit Packet outputs are complete and filename-stable across reruns.

---

## **7) Failure-Mode Localization (CAI / ALI)**

- [x] Compute CAI stage deltas as counterfactual deltas at same Tier A snapshot.
- [x] Report attenuation ratios between stages (`K_raw -> K_clamped -> g_preblend -> g_postblend`).
- [x] Compute `ALI` and flag likely redundancy regimes (`ALI high`, `FRG low`).
- [x] Add "redundancy escape hatch" annotations (history-channel or orthogonalized read-back candidate).

**Acceptance**
- Reports identify where signal dies when efficacy is weak.
- Redundancy vs suppression cases are distinguishable.

---

## **8) Negative Control and RC-Likeness Decision Gate**

- [x] Add negative-control sweep (perturb unrelated coefficient, e.g. `lambda`) at matched scale.
- [x] Compare signatures vs read-back sweep:
  - anisotropy/loop metrics (`OCR_active`, `RLI`, `ALI`)
  - sensitivity metrics (`FRG`, `LCE`)
- [ ] Apply RC-like criterion:
  - nontrivial Tier A deltas
  - reproducible nontrivial Tier B lagged effects
  - non-degenerate OCR/FRG/RLI behavior
  - not solely closure-driven.

**Acceptance**
- Read-back efficacy claims survive negative control discrimination.
- Decision output is explicit: effective / suppressed / redundant regime.

---

## **9) Cross-Version Matrix and Report Pack**

- [x] Run aligned matrix across v12-v16 variants (same stimuli/seeds/horizons where possible).
- [x] Produce merged comparison tables and plots.
- [x] Produce cross-version fingerprint grid (`figures/version_fingerprint_grid.png`) and populate "Version Fingerprints" section in `report.md`.
- [ ] Include closure-mode stratification for versions that support it.
- [x] Publish concise audit note with mechanism-based conclusions.

**Acceptance**
- Cross-version statements are mechanism-based, not movie-based.
- Artifacts are sufficient for repeatable review.

**Validation notes (remaining unchecked)**
- `5) Emit optional correlation-drop secondary metric`: optional; not yet emitted by scorer.
- `8) Apply RC-like criterion` remains open only for full "not solely closure-driven" enforcement.
- `9) Include closure-mode stratification`: pending matrix extension to run closure-off/soft/full lanes where supported.

---

## **Suggested Delivery Order**

1. Baseline Freeze and Artifact Contract
2. Canonical Telemetry Schema
3. In-Loop Hook Points and Capture
4. Dedicated Telemetry Stream
5. Tier A Counterfactual Engine
6. Tier B Counterfactual Engine
7. Offline Scoring
8. Failure-Mode Localization
9. Negative Control + RC-Likeness Gate
10. Cross-Version Matrix + Report Pack

---

## **Implementation Iterations (Execution Plan)**

### **Iteration 1 — Schema + Baseline**

- [x] 1.1 Freeze baseline matrix and artifact layout (versions, seeds, grids, modes).
- [x] 1.2 Finalize `ExperienceSample` schema keys, nullability, and schema versioning.
- [x] 1.3 Implement schema validator and fixture samples for each version family.
- [x] 1.4 Validate schema on at least one baseline artifact per version family.
  - Evidence: `outputs/readback-baseline/schema_validation_report.md`
  - Artifact roots: `outputs/readback-baseline/<version>/seed-*/{metadata.json,telemetry.jsonl}`

**Exit criteria**
- Baseline pack and schema contract are frozen.

### **Iteration 2 — Instrumentation + Storage**

- [x] 2.1 Add canonical hook-point instrumentation without persistent writes (in-memory checks).
  - Evidence: `outputs/readback-baseline/hook-smoke/README.md`
  - Runtime smoke logs: `outputs/readback-baseline/hook-smoke/{v12-cuda.log,v13-cuda.log,v14-cuda.log,v15-cuda.log,v16-cuda.log}`
- [x] 2.2 Enable scalar telemetry persistence (`telemetry.jsonl` or `telemetry.npz`).
  - Evidence: `outputs/readback-baseline/telemetry-smoke/README.md`
  - Stream helper: `experiments/scripts/readback_hooks.py::default_telemetry_jsonl_path`
  - Runtime streams: `outputs/readback-telemetry/simulation-v*-seed-*.jsonl`
- [x] 2.3 Enable optional downsampled field persistence on sparse cadence.
  - Evidence: `outputs/readback-baseline/field-smoke/README.md`
  - Controls: `RC_READBACK_FIELD_DUMP`, `RC_READBACK_FIELD_INTERVAL`, `RC_READBACK_FIELD_DOWNSAMPLE`
  - Runtime fields: `outputs/readback-telemetry/simulation-v*-seed-*.fields.npz`
- [x] 2.4 Validate hook completeness and telemetry overhead on CUDA and CPU variants.
  - Evidence: `outputs/readback-baseline/overhead-smoke/README.md`
  - Hook toggle: `RC_READBACK_HOOKS=0|1`
  - Result: hooks-on completeness `missing_steps=0`; measured overhead recorded for CUDA lane and CPU fallback lane.

**Exit criteria**
- All target versions emit canonical telemetry stream with bounded overhead.

### **Iteration 3 — Tier A + CAI**

- [x] 3.1 Implement Tier A2 default runner (`beta_rb=0` vs `beta_rb>0`, one-step recompute).
  - Runner: `experiments/scripts/run_readback_tierA.py --mode A2`
  - Evidence: `outputs/readback-baseline/tierA-smoke/simulation-v16-cuda/a2-seed-1/`
- [x] 3.2 Implement Tier A1 diagnostic runner (geometry-only propagation).
  - Runner: `experiments/scripts/run_readback_tierA.py --mode A1`
  - Evidence: `outputs/readback-baseline/tierA-smoke/simulation-v16-cuda/a1-seed-1/`
- [x] 3.3 Emit `DeltaJ`, `DeltaG`, `DeltaL` at fixed snapshots with baseline-derived `M_t`.
  - Score outputs: `scores.json` now includes `DeltaJ`, `DeltaG`, `DeltaL`, and mask-derived global/masked map summaries.
  - Evidence: `outputs/readback-baseline/tierA-smoke/simulation-v16-cuda/{a1-seed-1,a2-seed-1}/scores.json`
- [x] 3.4 Implement CAI stage deltas at the same Tier A snapshot reference.
  - Score outputs: `CAI.{DeltaJ_preClamp,DeltaJ_postClamp,DeltaK_raw,DeltaK_reg,DeltaG_preBlend,DeltaG_postBlend}`
  - Figure: `figures/cai_waterfall_t*.png`
- [x] 3.5 Produce first operator-sensitivity dashboard from Tier A outputs.
  - Figure: `figures/tierA_operator_dashboard_t*.png`
  - Entry-point wrapper: `experiments/scripts/run_readback_iteration3_tierA.sh`

**Exit criteria**
- Tier A deltas and attenuation stages are reproducible.

### **Iteration 4 — Tier B + Core Metrics**

- [x] 4.1 Implement paired-trajectory Tier B runner with seed-locked counterfactual branches.
  - Runner: `experiments/scripts/run_readback_tierB.py`
  - Wrapper: `experiments/scripts/run_readback_iteration4_tierB.sh`
  - Evidence: `outputs/readback-baseline/tierB-smoke/simulation-v16-cuda/seed-1/`
  - Cross-version smoke: `outputs/readback-baseline/tierB-smoke/simulation-v15-cuda/seed-1/`
- [x] 4.2 Implement lag metrics (`LCE_C`, optional `LCE_J/LCE_g`) and lag-curve generation.
  - Outputs: `series.npz` (`LCE_C`, optional `LCE_J`, `LCE_g`), `scores.json` (`LCE_*_max`, `LCE_*_auc`)
  - Figure: `figures/lce_lag_curves.png`
  - Optional maps: `figures/lce_divergence_map_tau{1,2,5}.png`
- [x] 4.3 Implement core metric panel (`OCR`, `FRG`, `RLI`) with shared norm conventions.
  - Outputs in `scores.json`: `OCR_global_median_rb`, `OCR_active_median_rb`, `FRG1_median`, `FRG2_median`, `FRG_saturation_ratio`, `DeltaRLI_mean`, `DeltaRLI_auc`.
  - Figures: `figures/ocr_over_time.png`, `figures/frg_vs_beta.png`, `figures/frg_saturation_ratio.png`, `figures/rli_curves.png`.
  - Evidence: `outputs/readback-baseline/tierB-smoke/simulation-v16-cuda/seed-1/`
- [x] 4.4 Add threshold-based anomaly flags and default-threshold configuration wiring.
  - Default config: `experiments/scripts/readback_thresholds.json`
  - Runner wiring: `--thresholds-json` and per-threshold CLI overrides in `experiments/scripts/run_readback_tierB.py`
  - Flags/decision in `scores.json`: `flags.*` + `decision` (`Formal-only|Weak-effective|Effective`)
- [x] 4.5 Validate run-to-run metric stability on repeated artifacts.
  - Harness: `experiments/scripts/run_readback_iteration4_stability.sh` (repeat runs + aggregate)
  - Aggregator: `experiments/scripts/check_readback_stability.py`
  - Evidence: `outputs/readback-baseline/tierB-stability/simulation-v16-cuda/seed-1/stability.{json,md}`
  - Result: `overall_stable=true` on 3 repeated runs (`decision_consistent=true`).

**Exit criteria**
- Paired-trajectory lag metrics available and stable.
- Thresholded flags are computed reproducibly from run artifacts.

### **Iteration 5 — ALI + Negative Control**

- [x] 5.1 Implement ALI time-series and masked/global ALI distribution outputs.
  - Outputs in `scores.json`: `ALI_global_median_rb`, `ALI_active_median_rb` (plus negative-control ALI medians when enabled)
  - Figures: `figures/ali_over_time.png`, `figures/ali_hist.png`
  - Evidence: `outputs/readback-baseline/tier5-negctrl-smoke/simulation-v16-cuda/seed-1/`
- [x] 5.2 Implement negative-control sweep path (matched scale against read-back sweep).
  - Control knob: `RC_NEGCTRL_LAM_SCALE` (applied in all v12-cuda..v16-cuda lanes)
  - Runner support: `--run-negative-control`, `--negctrl-lam-scale-per-beta`, optional explicit `--negctrl-lam-scale{1,2}`
  - Wrapper: `experiments/scripts/run_readback_iteration5_negative_control.sh`
- [x] 5.3 Run first read-back vs negative-control contrast and collect comparison tables.
  - Outputs: `contrast.json`, `contrast.md`
  - Evidence: `outputs/readback-baseline/tier5-negctrl-smoke/simulation-v16-cuda/seed-1/contrast.md`
- [x] 5.4 Apply RC-likeness decision gate (`Formal-only` / `Weak-effective` / `Effective`).
  - Decision now includes negative-control discrimination (`flags.Negative_control_fail`, `negative_control_pass`)
  - v16 smoke result: `decision=Weak-effective` (negative-control contrast did not pass)
  - Cross-version smoke: `outputs/readback-baseline/tier5-negctrl-smoke/simulation-v15-cuda/seed-1/`

**Exit criteria**
- Suppression vs redundancy vs efficacy classification works on at least one seed set.

### **Iteration 6 — Cross-Version Audit Pack**

- [x] 6.1 Run aligned cross-version matrix (v12-v16 family) with matched seeds/stimuli/horizons where possible.
  - Orchestrator: `experiments/scripts/run_readback_matrix.py`
  - Wrapper: `experiments/scripts/run_readback_iteration6_matrix.sh`
  - Matrix layout: `outputs/readback-baseline/tier6-matrix/{tierA,tierB,packets,logs}/`
- [x] 6.2 Generate canonical Audit Packet plots A-H with stable filenames for each run.
  - Added B1 output: `figures/ocr_term_energy_stack.png` in `experiments/scripts/run_readback_tierB.py`
  - Packet assembly copies Tier A sets A/F + Tier B sets B/C/D/E/G into per-run `figures/`
  - Matrix output H: `outputs/readback-baseline/tier6-matrix/figures/version_fingerprint_grid.png`
- [x] 6.3 Generate deterministic `report.md` from required template for each run.
  - Implemented in `experiments/scripts/run_readback_matrix.py` (`_write_run_report`)
  - Per-run report path: `outputs/readback-baseline/tier6-matrix/packets/<sim>/seed-<seed>/report.md`
- [x] 6.4 Publish merged comparison report, version fingerprint grid, and artifact index.
  - Matrix report: `outputs/readback-baseline/tier6-matrix/report.md`
  - Fingerprint grid: `outputs/readback-baseline/tier6-matrix/figures/version_fingerprint_grid.png`
  - Packet index section embedded in matrix `report.md`
- [x] 6.5 Verify required `report.md` sections and canonical plots A-H are present per run.
  - Verifier: `experiments/scripts/verify_readback_audit_packet.py`
  - Verification outputs: `outputs/readback-baseline/tier6-matrix/{verification.json,verification.md}`

**Exit criteria**
- Paper-14 audit protocol is executable end-to-end across v12-v16 family.
- Each run emits a complete deterministic Audit Packet (`scores.json`, `figures/`, `report.md`) with comparable cross-version fingerprints.
