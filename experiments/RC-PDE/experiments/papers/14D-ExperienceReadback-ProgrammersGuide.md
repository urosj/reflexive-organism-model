# 14D — Experience Read-Back Programmer's Guide

## 0. Audience and scope
This guide is for developers changing Paper-14 read-back instrumentation/scoring code.

Use this when you need to:
- modify telemetry emitted by simulators,
- add or change metrics,
- add/modify canonical plots,
- extend matrix workflows or verification rules,
- keep outputs backward-compatible across `v12-cuda` … `v16-cuda`.

Companion docs:
- theory/spec: `experiments/papers/14-ExperienceReadback-GapAndPlan.md`
- implementation checklist: `experiments/papers/14A-ExperienceReadback-ImplementationChecklist.md`
- operator runbook: `experiments/papers/14B-ExperienceReadback-UsersGuide.md`
- agent runbook: `experiments/papers/14C-ExperienceReadback-AgenticGuide.md`
- interpretation protocol: `experiments/papers/14E-ExperienceReadback-InterpretationGuide.md`

Path policy:
- Canonical paths are required for all new code/documentation changes.
- Legacy symlink paths were removed and must not be referenced.

## 1. Architecture map (single-screen)
System path:
1. Simulator loop computes fields and emits hook samples:
   - `simulations/active/simulation-v12-cuda.py`
   - `simulations/active/simulation-v13-cuda.py`
   - `simulations/active/simulation-v14-cuda.py`
   - `simulations/active/simulation-v15-cuda.py`
   - `simulations/active/simulation-v16-cuda.py`
2. Hook recorder writes telemetry/fields:
   - `experiments/scripts/readback_hooks.py`
3. Tier runners run counterfactual branches and score:
   - Tier A: `experiments/scripts/run_readback_tierA.py`
   - Tier B: `experiments/scripts/run_readback_tierB.py`
4. Matrix orchestrator assembles run packets and version fingerprint:
   - `experiments/scripts/run_readback_matrix.py`
5. Packet verifier enforces required sections/figures:
   - `experiments/scripts/verify_readback_audit_packet.py`
6. Schema utilities validate canonical record shape:
   - `experiments/scripts/readback_schema.py`
   - `experiments/scripts/validate_readback_schema.py`

Outputs:
- per-run: `scores.json`, `series.npz` (Tier B), `report.md`, `figures/`
- matrix: `figures/version_fingerprint_grid.png`, `report.md`, `verification.json`, `verification.md`

### 1.1 Single source of truth (defaults and contracts)
Use these files as authoritative:
- Threshold defaults: `experiments/scripts/readback_thresholds.json`.
- Required report sections and required figure filenames:
  - `experiments/scripts/verify_readback_audit_packet.py`.
- Schema keys/types/nullability/version:
  - `experiments/scripts/readback_schema.py`.

Policy:
- Code contracts are source of truth; docs must follow code.
- If docs and code differ, fix code or docs in the same PR and note the change.

## 2. Canonical contracts

### 2.1 Hook stage contract
Canonical hook names are defined in `experiments/scripts/readback_hooks.py`:
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

If you change stage names, you must update:
- simulator emit sites,
- Tier A/B readers (`_load_stage_series`, `_load_stage_scalar_series`),
- checklist/docs,
- verifier expectations if plot/report content changes.

### 2.2 Telemetry schema contract
Defined in `experiments/scripts/readback_schema.py`:
- `SCHEMA_NAME = "ExperienceSample"`
- `SCHEMA_VERSION = "14.1.0"`
- required vs nullable fields in `KEY_SPECS`
- alias map `E_* -> T_*_rms` in `ALIASES`

Validation entry point:
```bash
./venv/bin/python experiments/scripts/validate_readback_schema.py
```

Important rule:
- never rely on implicit missing values; use explicit `null`/fallback behavior per schema.

### 2.3 Packet contract (artifact-level)
Per run (Tier B or matrix packet), ownership is:

| Artifact | Required | Producer | Verifier rule |
|---|---|---|---|
| `scores.json` | yes | Tier B runner / matrix packet builder | file existence |
| `series.npz` | Tier B path | Tier B runner | file existence when Tier B used |
| `report.md` | yes | Tier B runner or matrix packet builder | `REQUIRED_REPORT_SECTIONS` |
| `figures/ocr_term_energy_stack.png` | yes | Tier B runner | `REQUIRED_FIGURE_EXACT` |
| `figures/ocr_over_time.png` | yes | Tier B runner | `REQUIRED_FIGURE_EXACT` |
| `figures/frg_vs_beta.png` | yes | Tier B runner | `REQUIRED_FIGURE_EXACT` |
| `figures/frg_saturation_ratio.png` | yes | Tier B runner | `REQUIRED_FIGURE_EXACT` |
| `figures/lce_lag_curves.png` | yes | Tier B runner | `REQUIRED_FIGURE_EXACT` |
| `figures/rli_curves.png` | yes | Tier B runner | `REQUIRED_FIGURE_EXACT` |
| `figures/ali_over_time.png` | yes | Tier B runner | `REQUIRED_FIGURE_EXACT` |
| `figures/ali_hist.png` | yes | Tier B runner | `REQUIRED_FIGURE_EXACT` |
| `figures/tierA_deltaJ_heatmap_t*.png` | yes | Tier A runner (copied by matrix builder) | `REQUIRED_FIGURE_GLOBS` |
| `figures/tierA_deltaG_heatmap_t*.png` | yes | Tier A runner (copied by matrix builder) | `REQUIRED_FIGURE_GLOBS` |
| `figures/tierA_delta_distributions_t*.png` | yes | Tier A runner (copied by matrix builder) | `REQUIRED_FIGURE_GLOBS` |
| `figures/cai_waterfall_t*.png` | yes | Tier A runner (copied by matrix builder) | `REQUIRED_FIGURE_GLOBS` |

Verifier authority:
- `experiments/scripts/verify_readback_audit_packet.py`:
  - `REQUIRED_REPORT_SECTIONS`
  - `REQUIRED_FIGURE_EXACT`
  - `REQUIRED_FIGURE_GLOBS`

### 2.4 Norm conventions (comparability contract)
Maintain these conventions unless schema/version policy is updated:
- Field scalarization: use RMS over domain unless explicitly mask-scoped.
- Tensor terms: per-cell Frobenius-style magnitude, then RMS aggregation.
- Vector fields: magnitude first (e.g., `sqrt(x^2+y^2)`), then RMS.
- Active-mask metrics (`OCR_active`, `ALI_active`) use baseline-derived mask `M_t` only.
- Ratio stabilizer uses small epsilon (`EPS = 1e-12` in Tier B runner) consistently.

### 2.5 Backwards compatibility policy
- Breaking field/schema changes require:
  - schema version bump in `readback_schema.py`,
  - compatibility handling (alias/fallback),
  - migration note in docs/changelog.
- Prefer additive changes:
  - add nullable fields first,
  - avoid renaming/removing existing keys in one step.
- Required plot/report contract changes require:
  - verifier update,
  - producer update,
  - docs update (`14B/14C/14D`) in same PR.

## 3. Simulator instrumentation internals

### 3.1 How simulators enable telemetry
All CUDA lanes instantiate:
- `InMemoryHookRecorder(enabled=(os.environ.get("RC_READBACK_HOOKS", "1") != "0"))`

At first hook emit in headless mode, they configure:
- JSONL stream via `default_telemetry_jsonl_path(...)`
- optional field dump when `RC_READBACK_FIELD_DUMP=1`

Environment controls:
- `RC_READBACK_HOOKS`
- `RC_READBACK_FIELD_DUMP`
- `RC_READBACK_FIELD_DOWNSAMPLE`
- `RC_READBACK_FIELD_INTERVAL`
- `RC_READBACK_BETA_RB`
- `RC_NEGCTRL_LAM_SCALE`

### 3.2 `L_proxy_scalar` and DeltaL contract
`DeltaL` in Tier A depends on `post_phi` field `L_proxy_scalar`.
If you alter how transport operator proxy is computed in simulators, keep:
- key name stable (`L_proxy_scalar`),
- stage stable (`post_phi`),
- scalar numeric semantics monotonic with operator magnitude.

Recommended interpretation contract:
- `L_proxy_scalar` should reflect transport-operator strength, not raw field amplitude.
- Typical shape: blended magnitude of gradient/laplacian operator terms (as in current code path).

Optional extension (non-breaking):
- emit `L_proxy_components` (structured detail) while keeping `L_proxy_scalar` required.
- use components for debugging only; `DeltaL` remains scalar for cross-version comparability.

### 3.3 Field dump compatibility keys
Tier A/B currently read these NPZ keys:
- `post_core_pre_closure__C_field`
- `post_J_postclamp__J_mag_field`
- `post_J_postclamp__Jx_field`
- `post_J_postclamp__Jy_field`
- `post_gradC__gradC_dx_field`
- `post_gradC__gradC_dy_field`
- `post_g_postblend__G_frob_field`
- `post_K_raw__T_rb_trace_field`
- `post_K_raw__T_grad_trace_field`

Do not rename without patching both runners.

## 4. Tier engines: mechanics and extension points

### 4.1 Tier A (`experiments/scripts/run_readback_tierA.py`)
Branching:
- baseline branch: `RC_READBACK_BETA_RB=0`
- intervention branch: `RC_READBACK_BETA_RB=<beta_rb>`

Modes:
- `A2` default: one-step recompute pathway
- `A1` diagnostic: geometry-branch focused short run

Core outputs:
- scalars: `DeltaJ`, `DeltaG`, `DeltaL`
- maps: `DeltaJ_map`, `DeltaG_map`
- CAI stage deltas:
  - `DeltaJ_preClamp`
  - `DeltaJ_postClamp`
  - `DeltaK_raw`
  - `DeltaK_reg`
  - `DeltaG_preBlend`
  - `DeltaG_postBlend`
- CAI ratios:
  - `J_post_over_pre`
  - `K_reg_over_raw`
  - `G_post_over_pre`

Extension example: add new Tier A scalar
1. emit key in simulator hook stage
2. parse via `_maybe_float(...)`
3. add to `scores.json`
4. document in `report.md`
5. if mandatory, update verifier report section expectations

### 4.2 Tier B (`experiments/scripts/run_readback_tierB.py`)
Branch topology:
- baseline (`beta_rb=0`)
- rb1 (`beta_rb`)
- rb2 (`frg_beta2` or auto `2*beta_rb`)
- optional negative control branches (`neg1`, `neg2`) using `RC_NEGCTRL_LAM_SCALE`

Metrics:
- OCR (`OCR_global`, `OCR_active`)
- FRG (`FRG1`, `FRG2`, saturation ratio)
- LCE (`LCE_C`, optional `LCE_J`, `LCE_g`)
- RLI (`RLI_baseline`, `RLI_rb`, `DeltaRLI`)
- ALI (global/active + hist samples)
- flags and decision
- `readback_score` + component breakdown (explicit ranking heuristic label)

Threshold loading:
- defaults in `DEFAULT_THRESHOLDS`
- overridden by `--thresholds-json`
- then overridden by CLI per-threshold flags

Extension example: add a new derived metric
1. compute series from loaded branches
2. serialize in `series.npz` and `scores.json`
3. optionally add plot file (stable filename)
4. include summary line in generated `report.md`
5. if required artifact, update verifier
6. update docs (`14B`, `14C`, this guide)

## 5. Matrix orchestrator internals

### 5.1 `experiments/scripts/run_readback_matrix.py`
Responsibilities:
- executes Tier A and Tier B for each `<sim,seed>`
- applies per-version profile args:
  - v14: closure softness profile
  - v15: closure mode + softness profile
  - v16: closure + nonlocal/domain fixed baseline
- assembles packet directory:
  - copies Tier B core artifacts
  - copies Tier A A/F plot set
  - writes deterministic packet `report.md`
- builds version fingerprint grid
- writes matrix summary report
- optionally runs verifier

### 5.2 Verification path convention
Matrix-level verification outputs should be written at matrix root:
- `outputs/readback-baseline/tier6-matrix/verification.json`
- `outputs/readback-baseline/tier6-matrix/verification.md`

Ad-hoc packet checks should set explicit `--out-json`/`--out-md`.

## 6. Non-negotiable invariants
- Baseline-mask provenance:
  - active masks for counterfactual comparisons come from baseline branch.
- FRG two-delta policy:
  - do not collapse FRG to a single-delta metric.
- Stable canonical filenames:
  - avoid run-specific filename drift for required plots.
- Deterministic report structure:
  - required sections must remain discoverable by verifier.
- Comparability discipline:
  - do not change thresholds/wrapper defaults mid-evaluation run.
  - if changed, use new project label and note diffs.

### 6.1 Change impact matrix (quick triage)
| If you change... | You must also change... |
|---|---|
| Hook stage name or stage field key | simulator emits, Tier reader loaders, schema/docs, possibly verifier |
| Threshold logic | `readback_thresholds.json`, Tier B parser/flags, report interpretation docs |
| Canonical plot filename | plot producer, verifier required list, docs references |
| Report section structure | report generator + verifier required sections |
| Decision/flag semantics | Tier B scoring + docs + matrix summary interpretation |

### 6.2 How to add a new hook field safely
1. Emit the field in simulator at a stable stage.
2. Add schema entry in `readback_schema.py`:
   - start as nullable/additive unless truly required.
3. Add runner reader path:
   - Tier A and/or Tier B loader functions.
4. Serialize into `scores.json` and/or `series.npz` as needed.
5. Add report summary lines if the field is decision-relevant.
6. Run:
   - schema validator,
   - Tier smoke runs,
   - packet verifier.
7. Only then consider making the field required or binding it to decisions.

## 7. Debugging playbook

### 7.1 Fast health checks
```bash
./venv/bin/python -m py_compile experiments/scripts/run_readback_tierA.py
./venv/bin/python -m py_compile experiments/scripts/run_readback_tierB.py
./venv/bin/python -m py_compile experiments/scripts/run_readback_matrix.py
./venv/bin/python experiments/scripts/validate_readback_schema.py
```

### 7.2 Runtime smoke
```bash
bash experiments/scripts/run_readback_iteration3_tierA.sh
bash experiments/scripts/run_readback_iteration4_tierB.sh
```

### 7.3 Packet verification
```bash
./venv/bin/python experiments/scripts/verify_readback_audit_packet.py \
  --packets-root outputs/readback-baseline/tierB-smoke \
  --out-json outputs/readback-baseline/tierB-smoke/verification.json \
  --out-md outputs/readback-baseline/tierB-smoke/verification.md
```

### 7.4 Common failures
- `No hook_stream path found in run output`:
  - simulator did not emit telemetry line; check hook enable path and headless config.
- `Tier B requires field dumps`:
  - missing field dump branches, confirm `RC_READBACK_FIELD_DUMP=1`.
- verifier missing sections/figures:
  - packet builder or report template drifted.
- flat FRG and flat LCE with tiny runs:
  - increase `--steps` before changing physics knobs.

### 7.5 Recommended golden packet smoke test
For regression-proofing (local CI-style check), keep a tiny smoke profile:
```bash
./venv/bin/python experiments/scripts/run_readback_tierB.py \
  --sim-script simulations/active/simulation-v16-cuda.py \
  --nx 32 --ny 32 --steps 5 --seed 1 \
  --out-dir outputs/readback-baseline/golden-smoke

./venv/bin/python experiments/scripts/verify_readback_audit_packet.py \
  --packets-root outputs/readback-baseline/golden-smoke \
  --out-json outputs/readback-baseline/golden-smoke/verification.json \
  --out-md outputs/readback-baseline/golden-smoke/verification.md
```

Minimum pass:
- verifier reports no missing required files/sections/figures.
- `scores.json` includes decision, flags, and primary metric keys.

## 8. Performance and scaling notes
- Main cost drivers:
  - grid size (`nx * ny`)
  - number of steps
  - number of branches (Tier B + negative control + matrix lanes)
  - field dump cadence and downsample
- Practical tuning:
  - smoke: `128x128`, short horizons
  - decision-quality: larger horizon and multi-seed
  - matrix: keep envelope fixed across versions

## 9. Contributor workflow (recommended)
1. Implement code change.
2. Run py_compile + schema validation.
3. Run Tier A smoke.
4. Run Tier B smoke.
5. Run verifier.
6. If scoring/report/plots changed: run matrix smoke.
7. Update docs (`14A`, `14B`, `14C`, `14D`) in same PR.

## 10. PR checklist (copy/paste)
| Change type | Must update |
|---|---|
| new required plot | `verify_readback_audit_packet.py`, Tier runner writer, docs |
| new metric in scores | Tier runner, report template lines, docs |
| new hook stage/key | simulator emits, runners readers, schema/docs |
| threshold logic change | `readback_thresholds.json`, runner flags, docs |
| matrix packet structure | matrix writer, verifier expectations, docs |
| decision/flags logic | Tier B scorer, report summary, docs |

## 11. Planned gaps (current state)
- No standalone `experiments/scripts/score_readback.py`; scoring is in `experiments/scripts/run_readback_tierB.py`.
- No standalone `plot_readback.py`; plotting is integrated in tier/matrix scripts.
- Optional correlation-drop metric remains pending.
- Closure-stratified matrix gate remains pending.

## 12. Concrete examples

### 12.1 Add a mandatory new figure safely
Suppose you add `figures/new_metric_curve.png` as required:
1. generate it in Tier B or matrix packet builder.
2. add filename to `REQUIRED_FIGURE_EXACT` in verifier.
3. add reference in packet `report.md` generator.
4. run smoke + verifier.
5. update `14B/14C/14D`.

### 12.2 Add a new threshold
Suppose `NEW_metric_min`:
1. add default in `DEFAULT_THRESHOLDS` and `experiments/scripts/readback_thresholds.json`.
2. add CLI override flag in Tier B parser.
3. include in `scores.json` thresholds dump.
4. integrate into flags/decision as needed.
5. update docs and checklist references.

### 12.3 Add a new simulator lane
Suppose `simulation-v17-cuda.py`:
1. implement same hook contract and key names.
2. include lane in matrix defaults/wrapper only when stable.
3. run schema validation on sample telemetry.
4. run matrix smoke and ensure verifier passes.
