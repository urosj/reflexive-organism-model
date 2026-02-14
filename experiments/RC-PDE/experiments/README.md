# **Purpose of the Repository**

* A complete narrative of **theory → engineering → phenomenology → formalization → emergence**.
* A living record of the development of Reflexive Coherence from first principles through RC-III and current PDE-boundary extensions (v15/v16), plus explicit analysis/observation layers over those implementations.
* A bridge between mathematical formulation, computational implementation, and emergent behavior.
* A guide for future RC research into evolving manifolds, identity ecologies, and higher-order reflexive systems.

# **Reflexive Coherence — Experiments & Observations**

*A structured exploration of coherence dynamics, stability engineering, phenomenology, and theoretical limits.*

This repository contains a connected sequence of papers documenting the development,
analysis, and theoretical grounding of **Reflexive Coherence (RC)** simulators and their evaluation protocols.
Together, they form a coherent narrative:

Layout note:
- Canonical paper files live under `experiments/papers/`.
- Canonical run/scoring scripts live under `experiments/scripts/`.
- Legacy `experiments/<name>` compatibility symlink paths have been removed.
- Use canonical `experiments/papers/...` and `experiments/scripts/...` paths in all commands and docs.

1. **Paper 1A — Theoretical Foundations of Reflexive Coherence**
2. **Paper 1B — The Quest for Stable PDEs in RC Dynamics**
3. **Paper 2 — Observations From RC-Inspired PDE Experiments**
4. **Paper 3 — Why PDEs Cannot Capture RC Dynamics**
5. **Paper 4 — PDEs *Can* Represent RC (But Only in a Generalized, Multi-Field Form)**
6. **Paper 5 — What Iterative PDE Experiments Revealed About RC**
7. **Paper 6 — The Reflexive Gap Between Simulators and RC**
8. **Paper 7 — RC-v12: Punctuated Identity Emergence**
9. **Paper 8 — RC-II: Field-Theoretic Formalization**
10. **Paper 9 — RC-III: Emergent Agency**
11. **Paper 10 — Soft-Closure Continuation in RC-v14**
12. **Paper 11 — RC-v15 Spec: Core-First Reflexive Geometry**
13. **Paper 12 — RC-v16 Spec: PDE-Boundary Approximation with Nonlocal Operator Reflexivity**
14. **Paper 13 — Activity Slider: Persistence Search & Evaluation (v15/v16)**
15. **Paper 14 — Experience Read-Back Gap (v12–v16) and Implementation Plan**
16. **Paper 14E — Experience Read-Back Results Interpretation Guide (Agent Protocol)**
17. **Paper X — Reflexive Ecologies (Ant Colony Case Study)**

These papers cover the full arc: theory → implementation → phenomenology →
limitations → formalization → emergence → PDE-boundary refinement → analysis/observation protocols → ecological application.

Clarification:
- Papers 7–12 track simulator/spec evolution (v12→v16 line).
- Papers 13–14E are analysis/observation/evaluation overlays on existing simulators; they do not introduce new RC simulator versions.

## **Paper 1A — Theoretical Foundations of Reflexive Coherence**

**Summary:**
This paper presents a quick outline of the mathematical core of Reflexive Coherence (RC). Identity is defined as a basin in a **coherence-induced geometry**, not as a static region of a scalar field. Coherence $C$ induces a metric $K_{\mu\nu}[C]$; coherence flow evolves *inside* this geometry; and the geometry itself updates in response to coherence dynamics.

Key contributions:

* Definition of coherence, induced geometry, coherence flux, and RC attractors.
* Formalization of **sparks** (geometry degeneracies) and **collapse** (trajectory-level decisions).
* Topological interpretation of identity formation, reproduction, pruning, and experience.
* RC as a self-referential dynamical system:
  $$
  C \rightarrow K[C] \rightarrow \text{flow of } C \rightarrow C.
  $$

Paper 1A establishes *what RC is*, independently of implementation.

## **Paper 1B — The Quest for Stable PDEs in RC Dynamics**

*(Engineering layer: how to build a numerically stable playground)*

**Summary:**
This paper documents the step-by-step construction of a **numerically stable PDE simulator** inspired by RC—but not yet faithful to its geometry. The goal is *engineering*, not correctness: how to make something run that does not blow up, collapse to nothing, or behave as a numerical artifact.

Key content:

* Grid setup, discretization, timestep control, and invariants (positivity, boundedness).
* Experiments with diffusion, nonlinear potentials, curvature terms, and RC-mode coupling.
* Recognition and removal of problematic event operators (collapse & spark surgery).
* Introduction of diagnostics: basin detection, spark tracing, gradient/Hessian monitoring.
* Lessons on separating **numerical engineering** from **phenomenological interpretation**.

Paper 1B teaches **how to construct a stable system you can trust** before interpreting results.

## **Paper 2 — Observations From RC-Inspired PDE Experiments**

*(Phenomenology layer: what the stable system actually does)*

**Summary:**
Once the PDE simulator is stable, we observe its dynamics. The result is a surprisingly rich set of behaviors—coarsening, filament formation, spark-induced fragmentation, basin motion—yet all taking place in a fixed Euclidean geometry.

Key observations:

* Early coarsening toward one or a few dominant basins.
* **Spark-driven fragmentation cycles**: sparks repeatedly split basins, relocate maxima, and repopulate identities.
* Sparked coherence **flows into denser basins** along filamentary “soft curvature” channels.
* Basins move, merge, and reorganize across the plane.
* Identity multiplicity is maintained **only** due to extrinsic spark triggering.
* Filaments act like curvature-induced conduits, but the geometry remains fixed.

Paper 2 documents the actual **empirical phenomenology** of the simulator.

## **Paper 3 — Why PDEs Cannot Capture RC Dynamics**

*(Mathematical layer: why the PDE approach must eventually be abandoned)*

**Summary:**
This paper proves that no local PDE on a fixed background geometry can reproduce RC dynamics. The spark-driven fragmentation and moving identities seen in Paper 2 are **observable but extrinsic**—they arise from additional Hessian-triggered code, not from the PDE itself.

Key arguments:

* PDEs evaluate derivatives in a **fixed** geometry; RC requires derivatives defined in the **induced geometry** $K[C]$.
* RC collapse is a **trajectory-level decision**, not a geometric merging of maxima.
* RC sparks are **metric degeneracies**, not Euclidean Hessian zero-crossings.
* Local PDEs are inherently smoothing/coarsening systems; RC requires **reflexive updating of geometry**.
* Extrinsic spark operators cannot substitute for intrinsic geometric bifurcations.

Paper 3 shows that PDEs are useful *prototypes* but fundamentally the **wrong class** for realizing RC.

## **Paper 4 — PDEs *Can* Represent RC (But Only in a Generalized, Multi-Field Form)**

*(Reconciliation layer: resolving the false impossibility proof)*

**Summary:**
Paper 4 revisits the impossibility claim made in Paper 3 and demonstrates that the conclusion arose from an overly narrow definition of “PDE”. The argument in Paper 3 correctly shows that *a single scalar PDE on a fixed Euclidean domain* cannot reproduce RC—but RC itself is not that kind of system. RC requires multiple interacting fields and an induced geometry. Once these are accounted for, **RC has a natural formulation as a system of coupled geometric PDEs**.

Key insights:

* The metric $g_{\mu\nu}$ need not be fixed; it can be an **unknown field** solved together with coherence.
* The coherence tensor $K_{\mu\nu}(C,\nabla C,J_C)$ is **local** and can be computed pointwise; no global operator is needed.
* Collapse and sparks correspond to **bifurcations** (loss of convexity) and **metric degeneracies**, both phenomena known in geometric flows.
* A closed reflexive loop $C \rightarrow K[C] \rightarrow g[K] \rightarrow v(C,g) \rightarrow C$ is fully representable as PDE dynamics if all fields evolve together.
* The “no intrinsic collapse” result of Paper 3 applies only to PDEs **with fixed operators**; RC requires **operator-carrying PDEs** whose geometry depends on the solution.

Paper 4 resolves the logical gap: RC does not contradict PDE theory—**it simply requires a broader PDE framework (coupled, geometric, reflexive).**
This establishes that PDEs are not disqualified for RC; the earlier impossibility was a misinterpretation of what “PDE” encompasses.

## **Paper 5 — What Iterative PDE Experiments Revealed About RC**

*(Synthesis layer: what we learned by trying to simulate RC)*

**Summary:**
Paper 5 documents the full sequence of iterative PDE models constructed in this repository, each adding more components of RC: geometry feedback, sparks as Hessian degeneracies, dynamic metrics, identity richness, and non-gradient coherence circulation. Despite this increasing sophistication, every fixed-domain PDE formulation eventually settled into a stable configuration.

This empirical stabilization is **not** a contradiction of RC. It is a diagnostic signal that:

> All our PDE formulations implicitly assumed a **fixed spatial domain** $\Omega\subset\mathbb{R}^2$,
> whereas RC requires an **evolving coherence manifold** whose geometry, topology, and identity structure are part of the dynamics.

Key conclusions:

* Each iteration captured part of RC:

  * basic coarsening,
  * spark-like events,
  * local geometry adaptation,
  * identity enrichment,
  * circulation of coherence.

* But all retained hidden assumptions foreign to RC:

  * static domain $\Omega$,
  * dissipative operators ensuring bounded attractors,
  * finite identity space,
  * geometry treated as a field *on* the domain instead of a field *that defines* the domain.
* These assumptions enforce convergence to a steady state—even when RC ingredients are present.
* In RC itself, identity proliferation and geometry rewriting continually update the effective operators; the system does not admit a classical fixed point.
* Therefore:

  * PDEs remain viable for RC **only** when formulated on **evolving manifolds** with **expanding identity space** and **reflexive operator updates**.
  * The models constructed so far should be viewed as **relaxation layers**, not complete RC embodiments.

Paper 5 clarifies the current status of the project: we now assume **why fixed-domain PDEs cannot show full RC behavior**,
what structural features RC actually demands, and what the next mathematical step must be—**geometric PDEs on evolving internal manifolds with identity-linked coordinates**.

## **Paper 6 — The Reflexive Gap Between Simulators and RC**

*(Conceptual layer: where the code and the theory still diverge)*

**Summary:**
Paper 6 introduces the idea of the **reflexive gap**: the structural mismatch between what the current simulators actually implement and what “true RC” requires.

Key points:

* The simulators operate on a fixed Euclidean grid with explicitly coded operators; RC operates on an evolving, coherence-defined manifold with operator structure derived from $K[C]$ and identity geometry.
* Many behaviors that *look* RC-like (spark fragmentation, identity multiplicity, geometry wells) are still produced by **engineered mechanisms** (Hessian triggers, thresholds, caps), not intrinsically by the field equations.
* The reflexive gap is defined as the distance between:

  * the **observable phenomenology** in the code, and
  * the **minimal reflexive structure** demanded by the theoretical RC definition.
* Paper 6 lays out concrete criteria for closing this gap:

  * moving from extrinsic event operators to intrinsic bifurcations,
  * replacing hard-coded birth/collapse by field-derived thresholds,
  * lifting identity from “things in the code” to “modes of the coherence–geometry system”.

Paper 6 sets the conceptual stage for RC-II (Paper 8) and RC-III (Paper 9) by making the remaining discrepancies explicit.

## **Paper 7 — Reflexive Coherence VII (RC-v12): Punctuated Identity Emergence**

**Summary:**
Introduces discrete identity fields $I_k$, spark-triggered birth, collapse thresholds, and reflexive feedback via α-source (identity→coherence) and η-curvature (identity→geometry), producing lifecycles and burst-like emergence.

Key contributions of RC-v12:

* Identity PDEs with growth, decay, diffusion.
* Spark birth gating with `spark_birth_sparks_min`, `I_global_mass_cap`, `id_birth_interval`.
* Heun integration for identity fields.
* Separating ξ-curvature from η-curvature.
* Redistributive coherence mass projection respecting admissible C range.
* Under-relaxed geometry blending.
* Toggleable optional potential terms (`USE_SPARK_DEEPENING`, `USE_IDENTITY_TILT`).

Paper 7 is the **complete empirical description** of RC-v12.

## **Paper 8 — Reflexive Coherence VIII: RC-II Field Theory**

**Summary:**
A formalization of RC-v12’s continuous substrate:
the **coherence–identity–geometry PDE triad**.

RC-II introduces:

* A continuous identity density field $I(x,t)$.
* Identity PDE:
  
$$
\partial_t I = g_{id} C I - d_{id} I + D_{id}\Delta I.
$$

* Coherence PDE with redistribution source preserving global invariance:

$$
S_C = \alpha_{id} \left(I - I_0\right).
$$

* Separated curvature tensor:

$$
K_{\mu\nu} =
\lambda C g_{\mu\nu}  + \xi \partial_\mu C \partial_\nu C + \eta I \partial_\mu C \partial_\nu C + \zeta J_\mu J_\nu.
$$

* Metric update via inverse curvature + under-relaxation.
* Spark manifolds as natural instability loci.
* Conservation of coherence mass under the redistributive source.
* Continuous pattern formation without discrete births.

Paper 8 defines RC-II as a **proper field theory**, supplying the mathematical backbone beneath RC-v12.

## **Paper 9 — Reflexive Coherence IX: RC-III (Emergent Agency)**

**Summary:**
RC-III interprets discrete identities as **proto-agents** emerging from RC-II.
Key concepts:

* Sparks as proto-perceptual structures.
* Identity birth as a triggered event when spark density, resource budgets, and timing align.
* Growth, diffusion, collapse = identity life cycle.
* Identities interacting indirectly through curvature modification and coherence redistribution.
* Competition, cooperation, territoriality, and niche formation.
* The reflexive triad

$$
I_k \leftrightarrow C \leftrightarrow g_{\mu\nu}
$$

becomes an action–perception loop.

Paper 9 positions RC as a **field-theoretic ecology** capable of expressing emergent, agent-like behavior.

## **Paper 10 — Soft-Closure Continuation in RC-v14**

**Summary:**
This note introduces RC-v14 as a methodological continuation layer over RC-III closures.
It keeps the reflexive core (C↔g coupling, coherence-mass projection, identity feedback),
while blending hard birth/collapse/spark gates into continuous field-driven scores via
closure-softness controls.

Key concepts:

* Hard-to-soft interpolation (`closure_softness`) instead of binary replacement.
* Soft spark intensity and soft birth factors for sparks, budget, interval, and slots.
* Soft collapse damping near `id_min_mass` with retained hard hygiene floor.
* Parameter sweeps to test which RC-III behaviors survive reduced threshold dependence.
* Explicit failure signatures to detect regression toward Paper-3 fixed-PDE artifacts.

Paper 10 positions v14 as a **methodological instrument** for probing the reflexive gap,
not as a new foundational RC theory layer.

## **Paper 11 — RC-v15 Spec: Core-First Reflexive Geometry**

**Summary:**
This document specifies v15 as a model-structure upgrade guided by the limits identified in Paper 3.
It enforces a strict three-layer architecture:

* **L0 core reflexive PDE loop** (coherence ↔ induced geometry),
* **L1 intrinsic event extraction** (spark/collapse indicators from induced invariants),
* **L2 minimal closure fallback** (soft, budgeted interventions only when needed).

Key requirements:

* ablation matrix (`core-only`, `core+events`, `full`) as a mandatory acceptance gate,
* explicit identity/coherence budget tracking (\(M_I/M_C\)),
* clear separation between intrinsic dynamics and closure policy effects,
* CUDA/headless workflow continuity for large-grid runs.

Paper 11 positions v15 as a **validation-and-architecture step** toward mathematically closer RC,
rather than another parameter retune.

Companion implementation checklist:
`experiments/papers/11A-v15-ImplementationChecklist.md`.

Companion baseline freeze profile:
`experiments/papers/11B-v15-BaselineFreeze.md`.

Companion ablation harness:
`experiments/papers/11C-v15-AblationHarness.md` and `experiments/scripts/run_v15_ablations.sh`.

Iteration-6 execution pack (performance + evaluation):
`experiments/papers/11D-v15-Iteration6-PerformanceAndEvaluation.md`,
`experiments/scripts/run_v15_iteration6_gate.sh`,
`experiments/scripts/summarize_run_logs.py`.

Iteration-7 CUDA runtime closure:
`experiments/papers/11E-v15-Iteration7-CUDARuntimeClosure.md`,
`experiments/scripts/run_v15_iteration7_all.sh`.

Codex resume handoff:
`experiments/papers/11F-v15-Codex-Handoff.md`.

## **Paper 12 — RC-v16 Spec: PDE-Boundary Approximation with Nonlocal Operator Reflexivity**

**Summary:**
This document specifies v16 as a PDE-boundary maximization step following v15.
It keeps the L0/L1/L2 architecture but strengthens PDE-side expressivity and auditability.

Key requirements:

* explicit nonlocal PDE term path (`nonlocal-off|on`) with stable CUDA execution,
* operator-carrying diagnostics as first-class outputs (`det/cond/drift` + degeneracy occupancy),
* adaptive-domain proxy with conservative remap and mass accounting diagnostics,
* intrinsic identity/readout emphasis in `closure-mode off`, with closure retained as comparator,
* theorem-aware reporting of intrinsic-event persistence vs closure dependence.

Paper 12 positions v16 as a **PDE-limit validation step**, not a claim of structural operator completion.

Companion implementation checklist:
`experiments/papers/12A-v16-ImplementationChecklist.md`.

Performance/evaluation note:
`experiments/papers/12D-v16-PerformanceAndEvaluation.md`.

Runtime closure runbook:
`experiments/papers/12E-v16-CUDARuntimeClosure.md`.

Release manifest:
`experiments/papers/12F-v16-ReleaseManifest.md`.

Execution scripts:
`experiments/scripts/run_v16_ablations.sh`,
`experiments/scripts/run_v16_iteration6_gate.sh`,
`experiments/scripts/run_v16_iteration7_all.sh`.

## **Paper 13 — Activity Slider: Persistence Search & Evaluation (v15/v16)**

**Summary:**
This document is a practical analysis runbook for discovering “organism-like” persistence regimes in the fixed-domain PDE sims.
It introduces a single convenience knob (`--activity`) to explore closure/budget regimes and a simple snapshot scorer to rank results,
so experimenters and agents can search systematically rather than tuning by visuals.
It does not define a new simulator line; it standardizes how to evaluate existing v15/v16 runs.

Key requirements:

* sweep `--activity` with fixed seeds/grid/steps and disk snapshots for comparable artifacts,
* score runs from `snap_*.npz` + `meta.json` (no reruns required) using a single scalar `score` with sub-metrics (stability, churn, mass variance),
* rerun top candidates across multiple seeds to reject transient false positives,
* treat “no good regime found” as a valid reproducible outcome (often a structural limitation, not a missed parameter).

Companion runbook:
`experiments/papers/13-ActivitySlider-PersistenceSearch.md`.

Companion evaluator:
`experiments/scripts/score_persistence.py`.

## **Paper 14 — Experience Read-Back Gap (v12–v16) and Implementation Plan**

**Summary:**
This document defines the “experience/read-back” gap: the sims compute flux/tension/curvature and event indicators, but most of that content is ephemeral or reduced to thin scalars.
Paper 14 reframes “experience” as flux read-back: telemetry derived from the produced coherence flux `J` (a read-back term) recorded in-loop and stored, then scored/visualized offline so versions can be compared by read-back strength and structure, not only by visible identity dynamics.
It is an audit/evaluation architecture over v12–v16, not a separate simulator implementation.

Key requirements:

* define flux read-back telemetry targets (e.g. `T_rb = J ⊗ J`) and record their summaries per snapshot,
* compute telemetry in-loop (where `J` exists) but do all processing/visualization offline,
* persist read-back scalars (and optionally downsampled fields) in snapshots for all v12–v16,
* provide a `readback_score` evaluator and contract-style fixtures to compare versions under identical stimuli.

Companion plan paper:
`experiments/papers/14-ExperienceReadback-GapAndPlan.md`.

Companion implementation checklist:
`experiments/papers/14A-ExperienceReadback-ImplementationChecklist.md`.

## **Paper X — Reflexive Ecologies: A Case Study Using Ant Colonies in Reflexive Coherence Simulations**

This paper demonstrates how RC-II and RC-III can be applied to ecological systems.
Using an ant colony as a worked example, it shows how identity fields (ants),
environmental gradients (food, nest, pheromones), and the coherence field C interact
to generate emergent foraging, trail formation, population regulation, and adaptive
collective behavior.

The RC-Ant simulation serves as a practical tutorial on how to use Reflexive
Coherence as a scientific modelling tool. It provides a methodological bridge between
the formal RC equations and empirical ecological dynamics, and concludes with a short
outlook toward a possible RC-IV (ecological-scale reflexivity), without attempting to
formalize it as a new theoretical layer.

This paper is intended for readers who want a concrete example of how RC simulations
can be used to study distributed systems, multi-agent environments, or ecological
self-organization.

---

## **Appendix A — Symbol → Code Crosswalk (v12 vs v14/v15/v16)**

This appendix is a navigation aid: it maps common paper symbols/constructs to the
concrete identifiers used in the simulators.

**Code targets:**
`simulations/legacy/simulation-v12.py`, `simulations/active/simulation-v14-cuda.py`, `simulations/active/simulation-v15-cuda.py`, `simulations/active/simulation-v16-cuda.py`.

| Paper quantity / construct | Meaning in papers (rough) | v12 (CPU) | v14 (CUDA) | v15 (CUDA) | v16 (CUDA) |
|---|---|---|---|---|---|
| `C(x,t)` | Coherence density field | `C` | `C` | `C` | `C` |
| `g_{μν}` | Induced metric | `g_xx, g_xy, g_yy` | `g_xx, g_xy, g_yy` | `g_xx, g_xy, g_yy` | `g_xx, g_xy, g_yy` |
| `K_{μν}` | Coherence tensor / curvature proxy | `K_xx, K_xy, K_yy` in `update_metric_from_K()` | same | same | same + `operator_state` in `update_metric_from_K()` |
| `Φ = δP/δC` | Functional derivative driving transport | `delta_P_over_delta_C()` (stored as `phi` in `compute_flux()`) | same | same | same |
| `v_C` | Coherence “velocity” (index-lifted gradient flow) | `v_up, v_vp` from `compute_flux()` | same | same | same |
| `J_C` | Coherence flux | `Jx, Jy` from `compute_flux()` | same | same | same |
| `∂t C = -∇·J + …` | Coherence RHS / continuity update | `rhs_C()` + `covariant_divergence()` | same | same | same + nonlocal addend (below) |
| `Π_C` (projection) | Global coherence invariance enforcement | `project_to_invariant()` | same | `rk2_step()` projects **before + after** metric update | `rk2_step()` projects **before + after** metric update |
| Spark condition | Degeneracy + strong gradient heuristic | `compute_spark_mask()` (binary mask) | `compute_spark_mask()` (hard/soft blend into continuous intensity) | `compute_intrinsic_spark_fields()` + `step_events()` (route hard/soft mask by mode) | same (`spark_soft` is also fed into continuous identity) |
| Spark score `S(x)` | Intrinsic instability readout in `[0,1]` | `spark_mask` (binary) | `spark_mask` (continuous) | `spark_soft` (from `compute_intrinsic_spark_fields()`) | `spark_soft` (from `compute_intrinsic_spark_fields()`) |
| Discrete identities `I_k` | Tracked identity fields (RC-III-style closure) | `I_fields` list; `update_identities()` | `I_tensor` + `n_active`; `update_identities()` | same (`step_closure()` routes by `--closure-mode`) | same (`step_closure()` routes by `--closure-mode`) |
| Continuous identity `I(x,t)` | RC-II substrate (no discrete tracking) | (not present) | (not present) | (not present in `closure-mode off`) | `I_cont`; `update_continuous_identity()` |
| Identity PDE | Growth/decay/diffusion update | `g_id, d_id, D_id` in `update_identities()` | `g_id, d_id, D_id` in `update_identities()` | same | `update_continuous_identity()` (continuous) + `update_identities()` (discrete comparator) |
| Identity → coherence | Source coupling `S_C(C,I)` | `rhs_C(): dCdt += alpha_id * I_sum` | same | same, but core uses `core_I_sum=None` when `--closure-mode off` | core uses `I_cont` always (`core_I_sum = I_cont` or `I_cont + I_sum`) |
| Identity → geometry | Identity-curvature coupling `η` | `eta_id * I_sum * (∂C⊗∂C)` in `update_metric_from_K()` | same | same | same |
| Identity mass `M_I` | Identity budget observable | Euclidean: `∑ I * dxdy` (no `sqrt_g`) | Metric-weighted: `∑ I * sqrt_g * dxdy` | same | same (continuous + discrete) |
| Global cap / budget | Limits total identity mass | `I_global_mass_cap` (fixed) | `I_global_mass_cap = I_cap_fraction * structured_mass0` | same | same |
| Closure continuation `χ` | Hard↔soft closure interpolation | (not present) | `closure_softness`, `spark_softness`, `collapse_softness` | same + per-mode `closure_softness_local` inside `update_identities()` | same |
| L0/L1/L2 routing | Architecture separation + ablations | (not explicit) | (not explicit) | `step_core()` / `step_events()` / `step_closure()` + `--closure-mode off\|soft\|full` | same |
| Nonlocal term `N_C` | Explicit nonlocal PDE contribution | (not present) | (not present) | (not present) | `compute_nonlocal_proxy()` + `rhs_C(): dCdt += nonlocal_strength * …` |
| Operator diagnostics `O(t)` | `det(K)`, `cond(K)`, drift, occupancy | (not present) | (not present) | (not present) | `operator_state` (e.g. `detK_mean`, `condK_max`, `g_drift_rms`, degeneracy fractions) |
| Adaptive-domain proxy | PDE-only “evolving domain” approximation | (not present) | (not present) | (not present) | `step_domain()` + `remap_field_bilinear()` + conservative remap accounting |
