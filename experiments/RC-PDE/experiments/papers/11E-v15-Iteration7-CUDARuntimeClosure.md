# **RC-v15 Iteration 7 — CUDA Runtime Closure (Tomorrow Runbook)**

This iteration closes all runtime-pending items once a torch/CUDA environment is available.

---

## **Single Command**

```bash
bash experiments/scripts/run_v15_iteration7_all.sh
```

This runs:
1. v13/v14 baselines,
2. v15 ablations,
3. v15 performance gate,
4. summary generation for all logs.

---

## **Expected Output Folders**

- `outputs/v15-iter1-baseline/`
- `outputs/v15-ablations/`
- `outputs/v15-iter6-gate/`

Each root should contain:
- per-profile `run.log`,
- `tail.txt`,
- animation artifact (`.mp4` or `.gif`),
- `summary.csv`,
- `summary.md`.

---

## **Post-Run Completion Steps**

1. Open and fill:
   - `experiments/papers/11D-v15-Iteration6-PerformanceAndEvaluation.md`
2. Mark pending items complete in:
   - `experiments/papers/11A-v15-ImplementationChecklist.md`
3. Confirm final acceptance:
   - throughput/regression budget accepted,
   - 1024x1024 long-run gate accepted,
   - v13/v14/v15 evaluation note complete.

---

## **Fallback (if long runs are too heavy)**

Use environment overrides:

```bash
NX=512 NY=512 THROUGHPUT_STEPS=1000 OVERHEAD_STEPS=1000 \
bash experiments/scripts/run_v15_iteration7_all.sh
```

Then repeat with 1024x1024 for final gate.
