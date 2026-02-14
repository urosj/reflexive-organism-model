# **RC-v16 Release Manifest (Finalization Snapshot)**

Date: `2026-02-13`

This manifest captures the final v16 release state and the exact runtime closure
artifacts used for signoff.

---

## **1) Canonical Runtime Command**

```bash
bash experiments/scripts/run_v16_iteration7_all.sh
```

This executes:
1. v16 baseline freeze,
2. v16 ablation matrix,
3. v16 performance gate,
4. summary generation for all outputs.

---

## **2) Required Artifacts**

All present after final run:

- `outputs/v16-iter1-baseline/summary.md`
- `outputs/v16-ablations/summary.md`
- `outputs/v16-iter6-gate/summary.md`
- `outputs/v16-iter6-gate/summary.csv`

Per-profile subfolders include:
- `run.log`
- `tail.txt`
- animation artifact (`simulation_output.mp4` or `.gif`)

---

## **3) Final Runtime Notes**

- Throughput and overhead acceptance details are recorded in:
  - `experiments/papers/12D-v16-PerformanceAndEvaluation.md`
- Implementation completion checklist is recorded in:
  - `experiments/papers/12A-v16-ImplementationChecklist.md`
- Runtime closure procedure is documented in:
  - `experiments/papers/12E-v16-CUDARuntimeClosure.md`

---

## **4) Quick Metrics Snapshot (from final `outputs/v16-iter6-gate/summary.md`)**

- `throughput/nonlocal-off`: `71.762` steps/s
- `throughput/nonlocal-on`: `69.444` steps/s
- `throughput/core-only`: `71.023` steps/s
- `throughput/soft`: `41.841` steps/s
- `throughput/adaptive-soft`: `32.680` steps/s

---

## **5) Regression Contract**

Runtime contract checks are covered by:

- `tests/test_v16_cuda_runtime_contract.py`

This guards:
1. adaptive-domain profiles present in `experiments/scripts/run_v16_iteration6_gate.sh`,
2. runtime log markers `[EVENT]` and `[DOMAIN]` present in `simulations/active/simulation-v16-cuda.py`.
