# **RC-v16 Iteration 6 — Performance Gate & Evaluation Note (Execution Pack)**

This document records completed Iteration 6 runtime/evaluation results for v16 CUDA.

---

## **1) Performance Gate Run**

Executed:

```bash
bash experiments/scripts/run_v16_iteration7_all.sh
```

Implemented gate matrix:
- throughput: `core-only`, `core-events`, `soft`, `full`, `nonlocal-off`, `nonlocal-on`, `adaptive-soft`
- snapshot-overhead sweep: `soft-int{10,25,50,100}` and `adaptive-soft-int{10,25,50,100}`

---

## **2) Summarize Logs**

Executed:

```bash
python experiments/scripts/summarize_run_logs.py \
  --root outputs/v16-iter6-gate \
  --csv outputs/v16-iter6-gate/summary.csv \
  --md outputs/v16-iter6-gate/summary.md
```

This produces:
- `outputs/v16-iter6-gate/summary.csv`
- `outputs/v16-iter6-gate/summary.md`

---

## **3) Throughput Acceptance (fill after run)**

- Throughput regression vs v15 accepted: `yes`
  - reference comparison (`core-only`, 1024 gate): v15 `81.070` vs v16 `71.023` steps/s (about `-12.4%`)
  - soft path comparison: v15 `46.200` vs v16 `41.841` steps/s (about `-9.4%`)
- Worst-case `steps_per_sec` at 1024: `32.680` (`throughput/adaptive-soft`)
- Best-case `steps_per_sec` at 1024: `71.762` (`throughput/nonlocal-off`)
- Nonlocal overhead vs local baseline accepted: `yes`
  - `nonlocal-on` vs `nonlocal-off`: `69.444` vs `71.762` steps/s (about `-3.23%`)

## **4) Snapshot + Domain Overhead Acceptance**

Compare selected profiles across intervals (`10,25,50,100`):
- wall time monotonic with snapshot frequency: `yes`
  - soft/nonlocal-on wall seconds: `109.12 > 72.95 > 59.87 > 53.93`
  - adaptive-soft/nonlocal-on wall seconds: `120.47 > 85.20 > 71.97 > 66.05`
- adaptive-domain overhead acceptable: `yes`
  - at interval `100`: `30.280` (adaptive) vs `37.085` (fixed) steps/s in overhead sweep basis (about `-18.4%`)
- recommended production interval for long runs: `50`
  - interval `50` keeps materially better observability than `100` while retaining strong throughput.

---

## **5) Theorem-Aware Evaluation (v15 vs v16, PDE-only)**

Artifacts used:
- `outputs/v15-iter1-baseline/`
- `outputs/v16-iter1-baseline/`
- `outputs/v16-ablations/`
- `outputs/v16-iter6-gate/`
- `outputs/v16-iter6-gate/throughput/*/run.log` (final-frame `[EVENT]`, `[OPERATOR]`, `[NONLOCAL]`, `[DOMAIN]`)

### Final State Metrics (1024 gate)
- Mass remains invariant across profiles (`mass=419.9802`).
- Identity/readout separation remains visible:
  - intrinsic profiles (`core-only`, `core-events`, `nonlocal-off`, `nonlocal-on`): `ids=0`, `I_mass=80.6993`
  - closure fallback profiles: `soft` gives `ids=8`, `I_mass=122.5095`; `full` gives `ids=0`, `I_mass=80.6993`
  - adaptive closure profile: `adaptive-soft` gives `ids=16`, `I_mass=157.4735`

### Intrinsic Event vs Closure Dependence
- Intrinsic spark scores are nontrivial in closure-off runs:
  - `core-only`: `spark_score_mean=1.873811e-02`, `spark_score_max=6.217437e-01`
  - `core-events`: `spark_score_mean=1.878127e-02`, `spark_score_max=6.711231e-01`
- In closure-off runs, birth observables remain readout-only:
  - `closure_birth_score=0`, `closure_births=0`, `ids=0`
- With closure fallback enabled, dependence appears explicitly:
  - `soft`: `closure_birth_score=1.271786e-02`, `ids=8`
  - `full`: hard gating suppresses births in this seed/horizon (`closure_birth_score=0`, `ids=0`)

### Operator Diagnostics Table (throughput logs, final frame)
| profile | detK_mean | condK_mean | condK_max | g_drift_rms | deg_detK_frac | deg_condK_frac | deg_detg_frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| core-only | 1.263054e-03 | 1.019270e+00 | 4.022290e+03 | 4.583265e-01 | 2.406120e-03 | 1.907349e-06 | 0.000000e+00 |
| core-events | 1.263072e-03 | 1.013143e+00 | 1.927238e+02 | 4.457898e-01 | 2.431870e-03 | 0.000000e+00 | 9.536743e-07 |
| soft | 1.258511e-03 | 1.009411e+00 | 3.320574e+02 | 3.725580e-01 | 2.849579e-03 | 0.000000e+00 | 0.000000e+00 |
| nonlocal-on | 1.196955e-03 | 1.001432e+00 | 1.067349e+00 | 8.927794e-05 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| adaptive-soft | 1.535263e-03 | 1.000065e+00 | 1.006682e+00 | 4.056502e-04 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |

### Nonlocal Contribution Summary
- `nonlocal-off`: `term_mean=0`, `term_rms=0`
- `nonlocal-on`: `term_mean=4.498189e-05`, `term_rms=5.787411e-05`
- overhead impact remains low (about `-3.23%` on throughput).

### Theorem-Aware Boundary Claim
v16 materially improves PDE-boundary behavior (nonlocal effects, operator diagnostics, adaptive-domain accounting, continuous identity substrate), but these results do **not** claim structural operator transition or full intrinsic identity-selection completion beyond PDE class. Closure dependence remains explicitly measurable in `soft/full` comparators.

---

## **6) Publishable Checklist**

- [x] Performance gate result recorded.
- [x] Snapshot/domain overhead recommendation recorded.
- [x] v15-v16 aligned comparison written.
- [x] Intrinsic event vs closure dependence report written.
- [x] Operator diagnostics table included.
- [x] Theorem-aware claim boundary explicitly stated.
- [x] Iteration 6 marked complete in `experiments/papers/12A-v16-ImplementationChecklist.md`.
