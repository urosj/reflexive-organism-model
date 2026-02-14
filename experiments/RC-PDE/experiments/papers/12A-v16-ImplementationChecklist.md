# **RC-v16 Implementation Checklist**

Companion to `experiments/papers/12-v16-Spec.md`.

---

## **0) Baseline Freeze (v15 Reference Before v16 Changes)**

- [x] Freeze v15 baseline artifacts for selected seeds/grids.
- [x] Record baseline operator/event metrics available in v15 outputs.
- [x] Archive v15 summaries for direct v16 delta comparisons.

**Done when:** v15 reference pack is reproducible and archived.

---

## **1) Nonlocal PDE Upgrade (L0)**

- [x] Add explicit nonlocal term path in coherence update (`nonlocal-off|on`).
- [x] Keep local-only mode available for ablation parity.
- [x] Ensure nonlocal path is CUDA-vectorized and bounded.

**Acceptance**
- Nonlocal path runs stably on 512 and 1024 grids.
- Nonlocal contribution is measurable in diagnostics.

---

## **2) Operator-Carrying Formalization (L0 Diagnostics)**

- [x] Promote operator diagnostics (`det(K)`, `cond(K)`, drift norms) to first-class outputs.
- [x] Persist operator diagnostics in snapshot metadata/time series.
- [x] Add degeneracy occupancy metrics (spatial fraction near thresholds).

**Acceptance**
- Operator diagnostics are available in all v16 profiles.
- Metrics correlate with morphology changes in at least one controlled seed.

---

## **3) Evolving-Domain Approximation (Still PDE)**

- [x] Implement adaptive mesh/domain-growth proxy with conservative remap.
- [x] Keep fixed-domain mode switch for regression comparability.
- [x] Verify mass accounting through domain adaptation steps.

**Acceptance**
- Domain proxy reduces fixed-grid artifacts on long runs.
- Mass projection tolerance remains satisfied.

---

## **4) Continuous Identity Substrate First (Readout Priority)**

- [x] Keep identity dynamics continuous in core path.
- [x] Treat births/collapse as observables by default in intrinsic runs.
- [x] Preserve optional closure fallback only for dependence comparison.

**Acceptance**
- `closure-mode off` remains the primary intrinsic claim path.
- L1 diagnostics are informative without L2 edits.

---

## **5) Ablation Harness (Required)**

- [x] Add reproducible presets for:
  - `core-only` (`off`),
  - `core+events` (L0+L1, no L2),
  - `soft`,
  - `full`,
  - `nonlocal-off` vs `nonlocal-on` within intrinsic paths.
- [x] Use same seed/horizon for aligned comparisons.
- [x] Export comparable metrics + snapshots for all profiles.

**Acceptance**
- One command set produces the full v16 matrix with aligned outputs.

---

## **6) Tests (v16-specific)**

Add tests with prefix `test_v16_cuda_*`.

- [x] `test_v16_cuda_modes.py`: mode routing including nonlocal toggles.
- [x] `test_v16_cuda_invariants.py`: mass projection + geometry/operator guardrails.
- [x] `test_v16_cuda_operator_metrics.py`: operator diagnostics emitted and bounded.
- [x] `test_v16_cuda_nonlocal_contract.py`: nonlocal path active/inactive as configured.
- [x] `test_v16_cuda_ablation_contract.py`: required matrix runs and artifacts exist.

**Acceptance**
- v16 test suite passes in CI/local.

---

## **7) Performance/Scaling Gate**

- [x] Prepare reproducible gate harness + summary tooling for v16.
- [x] Validate 1024x1024 headless run with disk snapshots.
- [x] Track throughput and nonlocal overhead vs local baseline.
- [x] Track adaptive-domain overhead separately.

**Acceptance**
- Throughput/regression budget is explicitly accepted with nonlocal/domain features enabled.

---

## **8) Evaluation Note (Theorem-Aware Companion)**

- [x] Write v15-v16 aligned comparison on same seeds.
- [x] Include intrinsic-event vs closure-dependence report.
- [x] Include operator diagnostics table and nonlocal contribution summary.
- [x] State theorem-aware claim boundaries explicitly for PDE-only class.

**Acceptance**
- Note is sufficient to justify v16 as PDE-boundary upgrade, not full structural-operator completion.

---

## **Suggested Delivery Order**

1. v15 Baseline Freeze
2. Nonlocal PDE Upgrade
3. Operator Diagnostics
4. Evolving-Domain Proxy
5. Continuous Identity Readout Path
6. Ablation Harness
7. Tests
8. Performance Gate
9. Evaluation Note
10. CUDA Runtime Closure

---

## **Implementation Iterations (Execution Plan)**

This is the iteration sequence for v16.

### **Iteration 1 — Baseline + Scaffolding**

- [x] Complete section 0 (v15 baseline freeze).
- [x] Add v16 runtime/scaffold flags (nonlocal/operator/domain toggles).

**Exit criteria**
- Baseline artifacts captured.
- v16 scaffold compiles/runs with local-equivalent behavior in strict baseline mode.

### **Iteration 2 — Nonlocal Core**

- [x] Complete section 1 (nonlocal PDE upgrade).
- [x] Validate stability in short/medium runs.

**Exit criteria**
- Nonlocal-on/off profiles run reproducibly.
- Mass and geometry invariants remain bounded.

Runtime note: validated on CUDA with explicit nonlocal diagnostics in logs (`outputs/v16-iter2-validation/*.log`), including stable 512 and 1024 runs.

### **Iteration 3 — Operator Diagnostics**

- [x] Complete section 2 (operator formalization diagnostics).
- [x] Persist diagnostics to snapshots/metadata.

**Exit criteria**
- Operator metrics are present and interpretable in outputs.

Runtime note: static verification passes (`13 passed` in `tests/test_v16_cuda_*`), and v16 now emits/persists occupancy diagnostics (`operator_degeneracy_detK_frac`, `operator_degeneracy_condK_frac`, `operator_degeneracy_detg_frac`). CUDA runtime validation now passes when run outside sandbox (`outputs/v16-iter3-validation/operator-128.log`, with `[OPERATOR]` including occupancy fractions).

### **Iteration 4 — Evolving-Domain Approximation**

- [x] Complete section 3 (adaptive domain proxy).
- [x] Verify conservative remap and bounded drift.

**Exit criteria**
- Domain proxy improves artifact profile without violating invariants.

Runtime note: adaptive-domain CUDA smoke passes outside sandbox (`outputs/v16-iter4-validation/adaptive-128.log`) with explicit `[DOMAIN]` diagnostics and zero reported mass accounting error (`mass_rel_error=0`, `mass_target_rel_error=0`).

### **Iteration 5 — Ablations + Tests**

- [x] Complete sections 4, 5, and 6.
- [x] Generate full ablation artifacts.

**Exit criteria**
- v16 tests pass.
- Full matrix artifacts generated from one command set.

Runtime note: intrinsic runs now carry continuous identity substrate in core and emit readout-only birth/collapse observables in `closure-mode off`; optional closure fallback remains in `soft/full`. Ablation matrix generated with one command (`bash experiments/scripts/run_v16_ablations.sh`, aligned env overrides) and summarized at `outputs/v16-ablations/summary.md`. Test suite status: `13 passed`.

### **Iteration 6 — Performance + Evaluation**

- [x] Complete sections 7 and 8.
- [x] Produce v15-v16 theorem-aware comparison note.

**Exit criteria**
- Throughput/regression budget accepted.
- Intrinsic-vs-closure dependence report completed.

Runtime note: iteration-6 gate executed on CUDA at 1024 with adaptive-domain overhead profiles (`outputs/v16-iter6-gate/summary.md`), then evaluated in `experiments/papers/12D-v16-PerformanceAndEvaluation.md` including throughput/nonlocal/adaptive overhead, intrinsic-event vs closure-dependence, operator table, and theorem-aware claim boundary.

### **Iteration 7 — CUDA Runtime Closure (Execution)**

- [x] Run full runtime closure pipeline:
  - `bash experiments/scripts/run_v16_iteration7_all.sh`
- [x] Confirm baseline summaries:
  - `outputs/v16-iter1-baseline/summary.md`
- [x] Confirm ablation summaries:
  - `outputs/v16-ablations/summary.md`
- [x] Confirm performance gate summaries:
  - `outputs/v16-iter6-gate/summary.md`
- [x] Fill and finalize:
  - `experiments/papers/12D-v16-PerformanceAndEvaluation.md`
- [x] Close all pending checklist items.

**Exit criteria**
- All runtime-pending items are completed with recorded artifacts.
- v16 checklist is fully checked off.

Runtime companion:
`experiments/papers/12E-v16-CUDARuntimeClosure.md`.

Runtime note: full iteration-7 runtime closure completed on CUDA (`bash experiments/scripts/run_v16_iteration7_all.sh`) with regenerated summaries in `outputs/v16-iter1-baseline/summary.md`, `outputs/v16-ablations/summary.md`, and `outputs/v16-iter6-gate/summary.md`; `experiments/papers/12D-v16-PerformanceAndEvaluation.md` finalized against these artifacts.
