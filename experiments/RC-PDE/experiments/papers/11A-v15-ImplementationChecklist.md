# **RC-v15 Implementation Checklist**

Companion to `experiments/papers/11-RC-v15-Spec.md`.

---

## **0) Baseline Freeze (Before Refactor)**

- [x] Create frozen benchmark configs for v13 and v14 (same seed/grid/steps).
- [x] Record baseline metrics: coherence mass drift, `M_I/M_C`, identity count, spark stats.
- [x] Save baseline snapshots and diagnostics for side-by-side comparisons.

**Done when:** baseline outputs are reproducible and archived.

---

## **1) Core Extraction (L0)**

- [x] Split simulation step into explicit phases: `step_core`, `step_events`, `step_closure`.
- [x] Ensure `step_core` runs without birth/prune logic.
- [x] Keep coherence-mass projection and geometry regularization inside core.
- [x] Add runtime mode flag: `--closure-mode off|soft|full`.

**Acceptance**
- `--closure-mode off` runs stably with L0 only.
- No identity birth/prune side effects in L0 path.

---

## **2) Intrinsic Event Metrics (L1)**

- [x] Implement intrinsic spark score from induced-geometry degeneracy indicators.
- [x] Implement intrinsic collapse score from identity support instability.
- [x] Keep L1 compute on device; only sync at diagnostics/snapshot interval.
- [x] Persist L1 diagnostics in snapshot metadata/time series.

**Acceptance**
- L1 metrics are available in headless outputs.
- In `off` mode, L1 produces signals but performs no field edits.

---

## **3) Policy-Gated Minimal Closure (L2)**

- [x] Implement policy wrapper for births/collapse tied to L1 scores.
- [x] Enforce budget/slot constraints via smooth factors + hard safety limits.
- [x] Keep hard cleanup only for numerical hygiene.
- [x] Log closure decisions (counts + gating factors) per diagnostics interval.

**Acceptance**
- `soft` mode shows reduced threshold brittleness vs v13.
- `full` mode reproduces expected RC-III-like intervention intensity.

---

## **4) Ablation Harness (Required)**

- [x] Add reproducible run presets for:
  - `core-only` (`off`),
  - `core+events` (L0+L1, no L2 edits),
  - `full` (L0+L1+L2).
- [x] Same seed and comparable horizon across presets.
- [x] Export comparable metrics + snapshots for all three.

**Acceptance**
- One command set produces all three ablations with aligned outputs.

---

## **5) Tests (v15-specific)**

Add tests with prefix `test_v15_cuda_*`.

- [x] `test_v15_cuda_modes.py`: verifies `off|soft|full` mode routing.
- [x] `test_v15_cuda_core_invariants.py`: mass projection and geometry guardrails in `off`.
- [x] `test_v15_cuda_event_metrics.py`: L1 metrics are emitted and bounded.
- [x] `test_v15_cuda_closure_policy.py`: L2 gating respects budget and slots.
- [x] `test_v15_cuda_ablation_contract.py`: all ablation presets run and produce expected artifacts.

**Acceptance**
- v15 test suite passes in CI/local.

---

## **6) Performance/Scaling Gate**

- [x] Prepare reproducible gate harness and summary tooling (`run_v15_iteration6_gate.sh`, `summarize_run_logs.py`).
- [x] Confirm no new hot-path host syncs.
- [x] Validate 1024x1024 headless run with disk snapshots for long horizon.
- [x] Track per-step time and snapshot overhead by interval.

**Acceptance**
- Throughput regression vs v14 is within agreed budget.

---

## **7) Evaluation Note (Paper Companion)**

- [x] Write a short experiment note comparing v13/v14/v15 on same seeds.
- [x] Include ablation plots/tables: invariants, `M_I/M_C`, births, lifetimes, spark stats.
- [x] Explicitly call out where behavior is intrinsic (L0/L1) vs closure-supported (L2).

**Acceptance**
- Note is sufficient to justify v15 design claims from the spec.

---

## **Suggested Delivery Order**

1. Baseline Freeze  
2. Core Extraction  
3. Event Metrics  
4. Closure Policy  
5. Ablation Harness  
6. Tests  
7. Performance Gate  
8. Evaluation Note  
9. CUDA Runtime Closure

---

## **Implementation Iterations (Execution Plan)**

This is the iteration sequence we will follow during implementation.

### **Iteration 1 — Baseline + Scaffolding**

- [x] Complete section 0 (baseline freeze).
- [x] Add structural scaffolding for `step_core`, `step_events`, `step_closure`.
- [x] Add `--closure-mode off|soft|full` plumbing with no behavior change yet.

**Exit criteria**
- Baseline artifacts captured.
- Refactored step structure compiles/runs with parity to pre-refactor defaults.

### **Iteration 2 — L0 Core Isolation**

- [x] Complete section 1 (core extraction).
- [x] Verify `off` mode executes only L0 dynamics (static routing check).
- [x] Re-validate invariants under `off` mode.

**Exit criteria**
- No birth/prune side effects in L0 path.
- Core invariants remain stable in short and medium runs.

Runtime note: invariant re-validation completed via runtime closure + v15 invariant tests.

### **Iteration 3 — L1 Event Metrics**

- [x] Complete section 2 (intrinsic event metrics).
- [x] Persist L1 diagnostics to snapshots/metadata.
- [x] Ensure L1 remains non-intervening in `off` mode.

**Exit criteria**
- L1 signals visible in outputs and bounded.
- No L1-triggered field edits unless closure policy requests them.

Runtime note: visibility/boundedness checks completed in runtime closure artifacts.

### **Iteration 4 — L2 Minimal Closure**

- [x] Complete section 3 (policy-gated closure).
- [x] Wire soft/full closure behavior to L1 scores.
- [x] Add closure diagnostics for auditability.

**Exit criteria**
- `soft` and `full` behave distinctly and reproducibly.
- Budget/slot constraints always enforced.

Runtime note: distinctness/reproducibility checks completed in runtime closure artifacts.

### **Iteration 5 — Ablations + Tests**

- [x] Complete section 4 (ablation harness).
- [x] Implement section 5 (`test_v15_cuda_*` suite).
- [x] Run ablations with same seed/horizon and collect artifacts.

**Exit criteria**
- All v15 tests pass.
- Ablation artifacts are generated from one reproducible command set.

Runtime note: ablation artifact generation completed in runtime closure artifacts.

### **Iteration 6 — Performance + Evaluation**

- [x] Complete section 6 (performance/scaling gate).
- [x] Prepare evaluation/runtime execution pack (`11D-v15-Iteration6-PerformanceAndEvaluation.md`).
- [x] Complete section 7 (evaluation note).
- [x] Compare v13/v14/v15 on aligned metrics.

**Exit criteria**
- Throughput/regression budget accepted.
- Companion note is publishable in `/experiments`.

Runtime note: Iteration 6 runtime execution completed on CUDA host.

### **Iteration 7 — CUDA Runtime Closure (Execution)**

- [x] Run full runtime closure pipeline:
  - `bash experiments/scripts/run_v15_iteration7_all.sh`
- [x] Confirm baseline artifacts exist and are summarized:
  - `outputs/v15-iter1-baseline/summary.md`
- [x] Confirm ablation artifacts exist and are summarized:
  - `outputs/v15-ablations/summary.md`
- [x] Confirm performance gate artifacts exist and are summarized:
  - `outputs/v15-iter6-gate/summary.md`
- [x] Fill and finalize:
  - `experiments/papers/11D-v15-Iteration6-PerformanceAndEvaluation.md`
- [x] Close all remaining pending checklist items in sections 0, 6, and 7.

**Exit criteria**
- All runtime-pending items are completed with recorded artifacts.
- v15 implementation checklist is fully checked off.

Runtime companion:
`experiments/papers/11E-v15-Iteration7-CUDARuntimeClosure.md`.
