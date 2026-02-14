# Paper 14 — Experience Read-Back Gap (v12–v16) and Implementation Plan

## Summary

In the identity choice/abundance framing, RC “experience” is not primarily a memory trace of what happened. It is **read-back**: the system’s own transport output (flux) contributes back into the tensor/operator that shapes the next step of the loop.

In other words, if the coherence tensor/operator is composed from:

- a **density** term (from `C`),
- a **gradient/tension** term (from `∇C`),
- a **read-back flux** term (from `J`),

then “experience” is the existence, magnitude, and structure of that **flux term** and its influence on subsequent evolution.

This paper defines:

- the experience/read-back gap in the current v12–v16 sims,
- a version-compatible telemetry schema to record read-back signals in-loop and store them,
- an offline observer toolchain for scoring and visualization (processing does not live inside the sim loop),
- an implementation plan to add comparable read-back telemetry to v12–v16.

## The Gap: What We Have vs What the Papers Mean

What the current sims do (v12–v16):

- compute transport quantities (flux `J`, velocities, divergence) to update `C`,
- compute geometry/metric updates from `C` and gradient-derived terms (e.g. `∂C ⊗ ∂C`),
- compute intrinsic event indicators (spark/collapse),
- optionally apply closure policies (identity birth/collapse) in some versions.
- include a formal read-back contribution in the operator path (`K_rb = zeta * (J ⊗ J)`).

What the papers’ effective read-back implies:

- formal read-back may already be present (e.g. `K_rb = zeta * (J ⊗ J)`) depending on version,
- the papers’ claim requires **effective read-back**: intervening on the read-back channel must measurably change next-step operator/flux behavior and produce lagged trajectory differences,
- the remaining problem is causal efficacy and observability: whether the read-back signal survives clamps/blends/regularizers and is not redundant with gradient structure.

Telemetry of gradients, masses, counts, etc. is useful, but it is not itself the read-back term. The read-back term is specifically **flux-derived** and lives in the operator/tensor composition.

Interventional definition (audit-grade):

- **Effective read-back exists if and only if intervening on the read-back channel** (`do(beta_rb = value)`)
  **changes future dynamics while all else is held fixed** (same `C_t`, same seed/noise path, same clamps/blends/regularizers).
- This separates ontology ("term exists in equations") from causal reality ("term changes outcomes").

## Definitions (Version-Compatible)

### Coherence Flux `J`

`J` is whatever flux the simulator already computes to advance `C` (vector field with components `Jx, Jy` in 2D).

### Read-Back Flux Tensor `T_rb`

Define a flux-derived symmetric tensor proxy that can be formed from `J`:

- simplest: `T_rb = J ⊗ J` (outer product), components:
  - `T_rb_xx = Jx*Jx`
  - `T_rb_xy = Jx*Jy`
  - `T_rb_yy = Jy*Jy`

Alternative (if desired later):

- power-like read-back: `P = |J · ∇C|` (scalar), used as a weight on an existing tensor.

The exact form can be varied per paper, but the core requirement is invariant: it must be derived from the produced flux `J`.

### Operator/Tensor Composition (Telemetry Targets)

The paper-level idea is that the coherence tensor/operator is composed from at least:

- density contribution `T_den(C)`
- gradient/tension contribution `T_grad(∇C)` (often proportional to `∂C ⊗ ∂C`)
- read-back contribution `T_rb(J)` (flux-derived)

Even before adding a new physical read-back coupling into the dynamics, we can already compute and store these pieces (or proxies) for comparison across versions.

## Telemetry: Computed In-Loop, Stored, Processed Offline

Telemetry is computed inside the simulation loop (because it depends on intermediate fields like `J`), but:

- scoring/analysis is done by separate offline tools that load stored artifacts,
- visualization is done offline (or in a separate viewer reading stored snapshots),
- the simulation should not contain analysis logic beyond recording telemetry.

### Canonical Architecture (v12–v16)

To avoid version drift, telemetry should not be implemented ad hoc per simulator.
Use a single architecture:

1. **Canonical schema** (`ExperienceSample`)
   - one shared set of keys and units for all versions.
2. **Per-version adapters**
   - each simulator maps local variables into the canonical schema.
3. **Fixed hook points**
   - sampling happens at consistent phase boundaries and operator stages:
     - `post_gradC`
     - `post_phi`
     - `post_J_preclamp`
     - `post_J_postclamp`
     - `post_K_raw`
     - `post_K_regularized`
     - `post_g_preblend`
     - `post_g_postblend`
     - `post_divergence` (coherence RHS assembled)
     - `post_core_pre_closure`
     - `post_closure` (when closure exists)
4. **Separate telemetry stream**
   - telemetry records are written to a dedicated artifact (e.g. `telemetry.jsonl` or `telemetry.npz`),
     not mixed into bulky field snapshots.

This keeps cross-version comparison stable while preserving each version’s internal implementation details.

### Telemetry Keys (Per Snapshot)

At minimum, store scalar summaries for each snapshot step:

- flux magnitude:
  - `J_rms`, `J_max`
- read-back tensor magnitude:
  - `T_rb_rms`, `T_rb_trace_mean` (where `trace(T_rb)=Jx^2+Jy^2`)
- gradient/tension magnitude (already used in curvature updates in many versions):
  - `gradC_rms`, `gradC_max` or `gradC2_mean` (field-level)
  - `T_grad_rms` (tensor-term norm for OCR denominator)
- operator contribution ratios (telemetry-only proxies):
  - `rb_vs_grad = T_rb_rms / (T_grad_rms + eps)`
  - `rb_vs_den  = T_rb_rms / (T_den_rms + eps)` (optional; requires explicit density proxy)
  - recommended density proxy when used: `T_den = C*g` term norm (RMS of tensor contribution).

Optional but useful for interpretation:

- localization/heterogeneity indices:
  - `J_cv = std(|J|) / (mean(|J|)+eps)`
  - `T_rb_cv` analogously
- intrinsic event summaries (already present in v14+):
  - `spark_score_mean`, `spark_score_max`, collapse scores
- closure intervention summaries (v14–v16):
  - `closure_births`, gating factors, etc.

### Canonical `ExperienceSample` Keys

Every version should emit the same keys (with fallbacks set to `0` or `null` when unavailable):

- identity/meta:
  - `schema_version`, `sim_version`, `step`, `phase`, `dt`
  - `nx`, `ny`, `dx`, `seed`
- read-back core:
  - `J_rms`, `J_max`, `J_cv`
  - `T_rb_rms`, `T_rb_trace_mean`, `T_rb_cv`
  - `gradC_rms`, `gradC_max` (field-level)
  - `T_grad_rms` (tensor-term norm)
  - `T_den_rms` (optional, only if `T_den = C*g` proxy is emitted)
  - `rb_vs_grad`, `rb_vs_den` (if `T_den` proxy exists; else `null`)
- context:
  - `closure_mode`, `n_active`, `I_mass`
  - `spark_score_mean`, `spark_score_max`
  - `closure_births`

The schema must be versioned (`schema_version`) so offline tools can remain backward-compatible.
`M_t` must be computed from the baseline branch (`beta_rb` reference) and reused across counterfactual comparisons, so the mask is not intervention-sensitive.

### Telemetry Fields (Optional, Downsampled)

For offline visualization, optionally store downsampled fields:

- `|J|`, `T_rb_trace`, maybe a single principal-direction indicator

Store sparingly to avoid I/O overhead. Scalars are enough to score; fields are for interpretability.

Recommended policy:

- write scalar telemetry every snapshot step,
- write downsampled fields every `N` telemetry steps (sparser cadence),
- keep field products in separate files from scalar telemetry.

## Offline Tooling (Observer / Agent Workflow)

The offline toolchain should:

- load telemetry stream + run metadata + optional downsampled field products,
- compute experience/read-back scores and plots,
- be version-agnostic (operate on stored telemetry keys).

Minimal tools:

1. `score_readback.py`
   - computes a scalar `readback_score` and component metrics from canonical telemetry.

2. `plot_readback.py` (or a notebook)
   - plots time series: `J_rms`, `T_rb_rms`, `rb_vs_grad`, event metrics, identity counts.

3. (Optional) a viewer that can render downsampled `|J|` and `T_rb_trace` fields side-by-side with `C`.

## Counterfactual Test Protocol

The audit uses two mandatory tiers because they reveal different failure modes.

### Tier A: One-Step Operator Sensitivity (Local)

At a fixed state snapshot, run one of two explicit Tier A modes:

- **A1 (geometry-only propagation)**
  - keep `J_t` fixed,
  - rebuild `K`/`g` with `beta_rb=0` and `beta_rb>0`,
  - recompute predicted transport output under each resulting `g`.
- **A2 (full one-step recompute)**
  - rebuild the one-step operator stack under each branch (including `phi`),
  - recompute predicted `J` without advancing the full trajectory.

Default Tier A mode for reports is **A2**.
Use A1 as a diagnostic to isolate geometry-only propagation effects.

Primary deltas:

- `DeltaJ = ||J^(rb) - J^(0)||`,
- `DeltaG = ||g^(rb) - g^(0)||`,
- `DeltaL = ||L^(rb) - L^(0)||` where `L` is the discrete transport operator proxy used in code.

`L` proxy contract (required):

- `L` MUST be represented by at least one compatible proxy from the transport path:
  - `L_proxy_scalar`: scalar summaries of discrete operator coefficients used by transport (for example RMS diag/offdiag magnitudes),
  - `L_proxy_field`: optional downsampled coefficient field.
- `DeltaL` is computed on whichever proxy is emitted (scalar or field), with the same norm convention as below.

This is the cheapest "does read-back do work right now?" test.

### Tier B: Lagged Trajectory Divergence (Experience Proper)

Run two trajectories from identical initial state and random seed:

- run A: `beta_rb=0`
- run B: `beta_rb>0`

Measure divergence at lag `tau = 1,2,...`:

- `LCE_C(tau) = ||C^(rb)_(t+tau) - C^(0)_(t+tau)|| / (||C^(0)_(t+tau)|| + eps)`
- optionally same for `J` and `g`.

Tier B catches delayed accumulation effects that Tier A can miss.

#### Interpretation

- Tier A answers: "does read-back do work right now at this state?"
- Tier B answers: "does read-back accumulate into experience over time (trajectory divergence)?"
- If Tier A is small but Tier B is nontrivial, read-back is weak per-step but accumulative.
- If Tier A is large but Tier B is small, read-back may be canceled by projection/clamps or re-regularized away.
- Use CAI to localize suppression when deltas vanish.

## Read-Back Metrics and Score (Evaluation)

The purpose of the score is to rank runs and compare versions on read-back strength and structure, not to prove “life”.

Define each metric with one role and one normalization:

Norm conventions (applies to all `||.||` uses in this paper):

- field norms: RMS over domain, or RMS over active mask `M_t` when masked metrics are requested,
- tensor norms: Frobenius norm per cell, then RMS across domain/mask,
- operator proxy norms: RMS of emitted `L` proxy coefficients (scalar summaries or coefficient fields).

- `OCR` (Operator Contribution Ratio)
  - global: `OCR_global = T_rb_rms / (T_grad_rms + eps)`
  - active-region: `OCR_active = T_rb_rms(M_t) / (T_grad_rms(M_t) + eps)`
  - mask rule: `M_t` is top `q%` by `|J|` post-clamp (fixed `q`, recommended `q=5%`)
  - counterfactual mask rule: compute `M_t` from the baseline branch (`beta`) and reuse the same mask in both branches.
  - norm rule: use RMS in both numerator and denominator, globally and on mask.
- `FRG` (Flux Response Gain, finite difference)
  - `FRG = ||J_(beta+delta) - J_beta|| / (|delta| * (||J_beta|| + eps))`
  - mandatory two-delta evaluation:
    - recommended `delta_abs_min = 1e-4`
    - `delta1 = max(beta*0.1, delta_abs_min)`
    - `delta2 = max(beta*1.0, 10*delta_abs_min)`
  - report `FRG1`, `FRG2`, and saturation indicator `FRG2 / (FRG1 + eps)`.
- `LCE` (Lagged Causal Effect)
  - `LCE_C(tau)` as defined in Tier B.
  - optional secondary: `1 - corr(C^(rb)_(t+tau), C^(0)_(t+tau))`.
- `RLI` (Return / Loop Index)
  - define origin set `A_t` once at intervention time `t` as top-`k%` active cells (from `|J|` or `|dC/dt|`)
  - keep `A_t` fixed for all lags `tau`
  - `RLI(tau) = sum_(x in A_t) |J_(t+tau)(x)| / (sum_x |J_(t+tau)(x)| + eps)`
  - compare `RLI^(rb) - RLI^(0)` under intervention.

Composite score (minimal version):

- `presence = mean(T_rb_rms > tau_rb)`
- `structure = mean(T_rb_cv > tau_cv)`
- `dominance = mean(OCR_global in [lo, hi])`
- `causal = mean(FRG > tau_frg) * mean(LCE_C(tau) > tau_lce for tau in TauSet)`

`readback_score = presence * structure * dominance * causal`

Closure intensity should be reported alongside score to separate intrinsic and closure-assisted effects.
`readback_score` is a ranking heuristic, not a truth predicate.

## Scoring Outputs and Standard Plots (Offline Audit Packet)

This paper specifies not only metrics, but the canonical analysis products the scorer must produce so that results are comparable across versions and parameter sweeps.

### Output artifact format
The offline scorer MUST produce:
- `scores.json` (scalar summaries per run + per snapshot)
- `series.npz` (time series arrays per run; optional if using parquet)
- `figures/` directory with standardized plots described below
- `report.md` auto-generated summary that links plots and lists key anomalies

`report.md` MUST be deterministic given the run artifacts (no manual edits required).
`report.md` MUST include: run metadata, parameter summary, seed info, metric summary table, canonical plot embeds/links, anomaly flags (threshold-based), and a short Interpretation checklist.

All plots MUST support:
- baseline vs read-back intervention (`beta_rb=0` vs `beta_rb>0`)
- global vs active-mask (`M_t`) variants where applicable
- aggregation across seeds (mean ± std) where applicable

### Plot set A — Tier A (counterfactual snapshot): “does read-back do work now?”
**A1. Delta heatmaps (required)**
- Plot: `DeltaJ_map(x) = |J_rb - J_0|` heatmap
- Plot: `DeltaG_map(x) = ||g_rb - g_0||_F` heatmap
- Overlay: active mask `M_t` contours (mask computed from baseline branch and reused for both branches)
- Output: `figures/tierA_deltaJ_heatmap_t{t}.png`, `figures/tierA_deltaG_heatmap_t{t}.png`
- Inputs: `J`, `g` (both branches), `M_t`

**A2. Delta distributions (required)**
- Plot: histogram or CDF of per-cell `DeltaJ`, `DeltaG`
- Two curves: masked (`M_t`) vs global
- Output: `figures/tierA_delta_distributions_t{t}.png`
- Inputs: same as A1

### Plot set B — OCR (term dominance): “is rb present relative to other terms?”
**B1. Term-energy stack over time (required)**
- Plot: time-series stacked area (or lines) for `T_den_rms`, `T_grad_rms`, `T_id_rms`, `T_rb_rms` (RMS norms of each K-term contribution)
- Compatibility note: if an implementation emits `E_*`, define `E_* := T_*_rms` in the scorer (no extra telemetry keys required).
- Output: `figures/ocr_term_energy_stack.png`
- Inputs: per-step term norms from telemetry (`T_*_rms` or `E_*` alias), or reconstructable term tensors

**B2. OCR_global and OCR_active over time (required)**
- Plot: `OCR_global(t)` and `OCR_active(t)` lines
- Output: `figures/ocr_over_time.png`
- Inputs: `T_*_rms` (or `E_*` alias) and `M_t` definition (mask computed from baseline branch)

### Plot set C — FRG (gain): “how sensitive is transport to beta_rb?”
**C1. FRG(beta) sweep curve (required)**
- Plot: `FRG(beta)` with log-scale beta on x-axis
- Output: `figures/frg_vs_beta.png`
- Inputs: beta sweep results at fixed snapshot/time window; `J(beta)` pairs

**C2. Saturation indicator (required)**
- Plot: `FRG2/FRG1` vs beta (two-delta requirement)
- Output: `figures/frg_saturation_ratio.png`
- Inputs: `FRG` computed at two delta scales (e.g., `delta1`, `delta2`)

### Plot set D — Tier B (lagged effects): “does it accumulate into experience?”
**D1. LCE lag curves (required)**
- Plot: `LCE_C(tau)` vs tau (mean ± std across seeds), and optionally `LCE_J(tau)`, `LCE_g(tau)`
- Output: `figures/lce_lag_curves.png`
- Inputs: paired trajectories (baseline vs rb), identical initial state/seed

**D2. Divergence maps at selected lags (optional but recommended)**
- Plot: `|C_rb - C_0|` heatmap at tau in `{1,2,5}` (or configurable)
- Output: `figures/lce_divergence_map_tau{tau}.png`
- Inputs: `C` fields from both trajectories at matching times

### Plot set E — RLI (loop/return): “does it come back?”
**E1. RLI(tau) curves baseline vs rb (required)**
- Plot: `RLI(tau)` baseline and rb, plus `DeltaRLI(tau)` line
- Output: `figures/rli_curves.png`
- Inputs: origin set `A_t` defined once at intervention time `t` (fixed for all tau), plus `J` fields over time

### Plot set F — CAI (attenuation): “where is rb being destroyed?”
**F1. Attenuation waterfall per snapshot (required)**
- Plot: waterfall (or step-down bars) showing counterfactual delta magnitude at each stage:
- `DeltaJ_preClamp -> DeltaJ_postClamp`
- `DeltaK_raw -> DeltaK_reg`
- `DeltaG_preBlend -> DeltaG_postBlend`
- Output: `figures/cai_waterfall_t{t}.png`
- Inputs: telemetry captured at hook points pre/post clamp, raw/reg, pre/post blend

### Plot set G — ALI (redundancy): “is rb colinear with gradient structure?”
**G1. ALI over time + distribution (required)**
- Plot: `ALI(t)` line
- Plot: histogram of ALI values, masked vs global
- Output: `figures/ali_over_time.png`, `figures/ali_hist.png`
- Inputs: `J`, `gradC` (and/or `gradphi` if that is the chosen alignment reference), plus `M_t`

### Plot set H — Cross-version comparison packet (required for experiment matrix)
For each version/run group, produce a standardized comparison page:
- Table: median `OCR_active`, median FRG (linear regime), max/AUC `LCE_C`, `DeltaRLI(tau)` summary, dominant CAI bottleneck stage, median ALI
- Small multiples: (B2, C1, D1, F1, G1) aligned across versions
- Output: `figures/version_fingerprint_grid.png` and `report.md` section “Version Fingerprints”

### report.md (auto-generated) — Required Template

The offline scorer MUST generate `report.md` following this structure so every run produces a comparable audit packet.

#### 0) Run metadata
- Version: `<v12|v12-cuda|v13|v14|v15|v16|...>`
- Git hash / build tag: `<...>`
- Date/time: `<...>`
- Grid: `<Nx, Ny>`  dt: `<...>`  steps: `<...>`
- Seeds: `<list>`
- Stimulus / initial condition: `<descriptor>`
- Modes: closure `<off|soft|full>`  other toggles `<...>`

#### 1) Parameter summary
| Parameter | Value |
|---|---:|
| beta_rb | `<...>` |
| zeta_flux | `<...>` |
| blend_base | `<...>` |
| blend_rb (if applicable) | `<...>` |
| J clamp | `<...>` |
| K regularization / clamps | `<...>` |
| metric eigenvalue bounds | `<...>` |
| kappa_grad | `<...>` |
| (others critical to operator) | `<...>` |

#### 2) Executive summary (1 screen)
**Decision:** `<Formal-only | Weak-effective | Effective>`  
**Why (auto-filled bullets):**
- OCR_active: `<...>`  OCR_global: `<...>`
- FRG (linear regime): `<...>`  FRG saturation ratio: `<...>`
- LCE_C: max `<...>`  AUC `<...>`  (± std across seeds)
- DeltaRLI summary: `<...>`
- CAI bottleneck stage: `<...>`
- ALI median: `<...>`

**Decision rule (required, threshold-based):**
- `Formal-only` if `max(FRG_linear_regime) < FRG_min` AND `LCE_auc < LCE_auc_min`.
- `Weak-effective` if exactly one of `{FRG, LCE_auc}` is above threshold, or both are marginal/unstable.
- `Effective` if `FRG >= FRG_min` AND `LCE_auc >= LCE_auc_min` AND reproducible across seeds (`seed_var <= seed_var_max`) AND not closure-dependent.

**Flags (auto):**
- `[ ]` RB dominated: OCR_active below threshold
- `[ ]` RB attenuated: CAI shows major drop at `<stage>`
- `[ ]` Saturation/clamp: FRG2/FRG1 below threshold
- `[ ]` Redundant channel: ALI above threshold
- `[ ]` Non-reproducible: seed variance above threshold
- `[ ]` Closure-dependent: efficacy disappears in closure-off mode

#### 3) Tier A: Counterfactual snapshot results
- Snapshot times evaluated: `<t list>`
- A1 heatmaps:
  - `figures/tierA_deltaJ_heatmap_t{t}.png`
  - `figures/tierA_deltaG_heatmap_t{t}.png`
- A2 distributions:
  - `figures/tierA_delta_distributions_t{t}.png`

**Tier A interpretation checklist:**
- Does DeltaJ concentrate in the active mask (good) or appear only as rare spikes (suspicious)?
- Is DeltaG nontrivial where DeltaJ is nontrivial?
- Do deltas persist across snapshots?

#### 4) OCR: Term dominance over time
- `figures/ocr_term_energy_stack.png`
- `figures/ocr_over_time.png`

**OCR interpretation checklist:**
- Is OCR_active consistently nontrivial or only intermittent spikes?
- Is OCR_global negligible while OCR_active is meaningful (expected)?
- Do changes in OCR correlate with activity bursts?

#### 5) FRG: Gain and saturation
- `figures/frg_vs_beta.png`
- `figures/frg_saturation_ratio.png`

**FRG interpretation checklist:**
- Is there a clear linear regime where FRG is stable?
- Does FRG plateau early (suggesting clamp/blend domination)?
- Do FRG results match the Tier A delta distributions?

#### 6) Tier B: Lagged causal effects (experience accumulation)
- `figures/lce_lag_curves.png`
- Optional divergence maps:
  - `figures/lce_divergence_map_tau{tau}.png`

**Tier B interpretation checklist:**
- Do small Tier A deltas accumulate into nontrivial LCE (weak-but-cumulative)?
- Are LCE curves reproducible across seeds?
- Does LCE remain nontrivial when closure is reduced/off?

#### 7) RLI: Return / loop closure
- `figures/rli_curves.png`

**RLI interpretation checklist:**
- Does read-back increase return mass into the fixed origin set A_t?
- Is DeltaRLI consistent across seeds and snapshots?

#### 8) CAI: Attenuation bottleneck
- `figures/cai_waterfall_t{t}.png`

**CAI interpretation checklist:**
- At which stage does Delta collapse most (J clamp / K reg / blend / eig clamp)?
- Is the dominant bottleneck stable across snapshots?

#### 9) ALI: Redundancy / alignment
- `figures/ali_over_time.png`
- `figures/ali_hist.png`

**ALI interpretation checklist:**
- Is ALI consistently near 1 (channel likely redundant)?
- Is redundancy concentrated inside or outside the active mask?

#### 10) Cross-version fingerprints (when running experiment matrix)
- `figures/version_fingerprint_grid.png`
- Summary table (auto-filled) comparing:
  - median OCR_active, FRG linear regime, LCE AUC, DeltaRLI summary, CAI bottleneck, median ALI

#### 11) Notes / anomalies (auto + manual)
- Auto anomalies: `<list>`
- Manual notes: `<free text>`

### Default thresholds (tunable)

- `OCR_active_min = <TBD>`
- `FRG_min = <TBD>`
- `FRG_saturation_ratio_min = <TBD>`
- `ALI_redundant_min = <TBD>`
- `LCE_auc_min = <TBD>`
- `seed_var_max = <TBD>`

## Failure-Mode Discriminators

Two diagnostics are mandatory to avoid false conclusions.

### CAI: Clamp/Blend Attenuation Index

Track attenuation of read-back effect through the numerical pipeline:

- `DeltaK_raw`
- `DeltaK_clamped`
- `DeltaG_preblend`
- `DeltaG_postblend`

Compute attenuation ratios between consecutive stages. High attenuation localizes where read-back is being numerically suppressed.
All stage deltas are counterfactual deltas:
`DeltaX_stage = ||X_stage(beta+delta) - X_stage(beta)||` under identical state/noise.
CAI is evaluated at the same snapshot used for Tier A so attenuation and `DeltaJ/DeltaG` share one reference point.

### ALI: Alignment / Redundancy Index

`ALI = E[ |J·gradC| / (|J|*|gradC| + eps) ]`

If `ALI` is high while `FRG` is low, read-back may be present but largely redundant with gradient structure.

Redundancy escape hatch (if ALI remains high):

- history-channel option: `M_(t+1) = (1-gamma)*M_t + gamma*T_rb_hat(t)`,
- orthogonalized read-back option: build `T_rb` from the component of `J` orthogonal to `gradC`.

## Implementation Plan (v12–v16 Compatible)

The plan is staged so you can compare versions early, before changing core dynamics.

### Iteration 1: Canonical Schema + Adapters (No Dynamics Change)

- define `ExperienceSample` schema in one shared module (`experiments/scripts/readback_schema.py` or equivalent docs-first contract),
- add per-version adapters (v12, v12-cuda, v13, v14, v15, v16) mapping local variable names to schema keys,
- emit telemetry at the canonical fixed hook-point set defined in the architecture section.

Acceptance:

- all versions emit the same schema keys,
- missing quantities use explicit fallback policy,
- schema validation passes offline.
- hook-point completeness is validated (required hooks present per version).

### Iteration 2: Dedicated Telemetry Storage + Minimal Fields

- write scalar telemetry to a dedicated stream (`telemetry.jsonl` or `telemetry.npz`),
- optionally write downsampled field products (`|J|`, `T_rb_trace`) on sparse cadence,
- keep existing snapshot pipeline unchanged for simulation products.

Acceptance:

- telemetry I/O overhead is bounded and measurable,
- offline tools no longer depend on simulator-specific snapshot internals,
- viewer can render optional read-back fields when present.

### Iteration 3: Offline Observer Toolchain

- implement `score_readback.py` against canonical schema,
- implement `plot_readback.py` / notebook templates for standard read-back panels,
- add simple cross-version comparison runner (same seeds/stimuli, merged report),
- include Tier A + Tier B counterfactual test runners.
- offline scorer must emit the Audit Packet plot set (A-H) with stable filenames and per-run `scores.json`.
- generate `report.md` using the required template, embedding/linking canonical plots and printing the executive summary + flags.

Acceptance:

- one command can score any version output with no code edits,
- cross-version reports are reproducible.
- negative control passes: perturbing an unrelated coefficient (e.g. `lambda`) does not replicate read-back-specific signatures (`OCR/FRG/RLI` profile).
- expected negative-control contrast: read-back perturbations preferentially alter anisotropy/loop signatures (`OCR_active`, `RLI`, `ALI`), while unrelated coefficient perturbations mostly rescale baseline curvature/intensity.

### Iteration 4: Introduce Explicit Flux Read-Back Coupling (Controlled, Ablated)

Only after telemetry is stable:

- add an audit coefficient `β_rb` that uniformly scales the already-existing read-back channel across versions,
- explicit coupling rule: `K_rb <- beta_rb * zeta_flux * (J ⊗ J)` (equivalently `zeta_eff = beta_rb * zeta_flux`),
- optionally add a normalized `J⊗J` variant to improve observability and scale comparability,
- make it ablatable and default-off for parity where needed.

Acceptance:

- measured `T_rb` contribution changes in predictable ways,
- outcomes change measurably without breaking invariants (mass projection, stability),
- comparisons across versions remain interpretable.

## Why This Matters

Without flux read-back, the loop is effectively:

`C -> operator(from C, ∇C) -> flux J -> C`

With read-back, it becomes:

`C -> operator(from C, ∇C, J) -> flux J -> C`

The second form is what the papers call “experience”: the loop’s own action (flux) contributes back into what governs the next action.

Telemetry and offline tooling then let you quantify:

- how much read-back exists,
- whether it is structured/localized,
- whether it changes across versions and closure modes,
- whether it correlates with intrinsic events and identity dynamics.

## What You Gain From This Audit (and Why It’s Needed)

### 1) A causal microscope: term existence vs term efficacy

- The simulator can include formal `T_rb` (`J⊗J`) and still show negligible practical read-back effect.
- The audit measures intervention sensitivity: whether `do(beta_rb)` changes `g`, discrete operator proxy `L`, and predicted `J` at fixed state (Tier A).
- The audit also checks whether intervention produces lagged divergence in `C/J/g` (Tier B).
- Without this protocol, "experience" is inferred from visuals; with it, experience is quantified by effect sizes (`DeltaJ`, `DeltaG`, `DeltaL`, `LCE`).
- This separates mathematical inclusion from causal contribution.

### 2) Localization of failure: where the read-back signal dies

- CAI is a localization tool, not only a summary metric.
- It identifies suppression stage: `J` clamp, `K` regularization, metric blending, eigenvalue clamps, or downstream projection.
- This turns debugging from broad coefficient tuning into stage-targeted fixes.
- It prevents false negatives where read-back exists upstream but is numerically attenuated downstream.
- It also supports stable remediation plans by pinpointing one stage at a time.

### 3) Version comparison becomes mechanism-based, not outcome-based

- Different versions can produce similar `C` movies while implementing different operator mechanics.
- Cross-version comparison is performed on standardized axes:
- magnitude/dominance: `OCR_global`, `OCR_active`
- sensitivity: `FRG1`, `FRG2`, saturation ratio
- persistence: `LCE_C(tau)` curves
- loop-closure tendency: `RLI(tau)` differences
- redundancy check: `ALI`
- This enables mechanism statements such as: "v15 has high OCR but low FRG" (read-back present, operator response damped).
- It also supports cases like: "v16 has modest OCR but nontrivial LCE" (weak per-step coupling with cumulative effect).
- The result is version ranking by causal mechanism, not by visual similarity.

### 4) Separating read-back from closure (identity policies) explicitly

- Closure policies can generate apparent feedback even when transport read-back is weak.
- The audit therefore requires closure intensity reporting and mode-stratified comparisons.
- Read-back claims are stronger when efficacy metrics persist in closure-off and soft settings.
- This avoids attributing closure-driven effects to transport read-back.
- It provides a clean intrinsic-vs-policy decomposition for interpretation.

### 5) A practical criterion for "Does this PDE behave like RC?"

- RC-like read-back means nontrivial Tier A deltas and stable, reproducible Tier B lagged effects (`LCE`) under intervention.
- It also requires nontrivial `OCR/FRG` and a non-degenerate `RLI` pattern.
- If `DeltaJ ≈ 0` across beta sweeps and `LCE` stays near zero, the regime is formal-only or numerically suppressed.
- If `ALI` is high while FRG is low, the read-back channel is likely redundant rather than absent.

### 6) Engineering payoff: regression suite for future versions

- Once telemetry and scoring are in place, v17+ changes can be evaluated objectively.
- The protocol becomes a regression suite: same seeds/stimuli, comparable metric panels, immediate detection of efficacy changes.
- It distinguishes "larger magnitude" from "greater causal efficacy."
- It also detects regressions where stability is preserved but read-back influence collapses.
- This makes future development measurable, auditable, and reproducible.

## Abstract (Proposed)

We show that flux read-back is present formally in v12-v16 via a `J⊗J` contribution to the coherence tensor `K`. We then operationalize experience as **effective read-back**: intervention on the read-back channel produces measurable changes in the next-step transport operator and in lagged trajectory dynamics. We define a canonical telemetry schema with per-version adapters, a two-tier counterfactual test suite, and efficacy metrics (`OCR`, `FRG`, `LCE`, `RLI`) with failure-mode discriminators (`CAI`, `ALI`). This yields a reproducible audit protocol for determining when read-back is dynamically consequential versus numerically attenuated or redundant.
