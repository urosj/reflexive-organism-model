# **RC-v16 Iteration 7 — CUDA Runtime Closure (Runbook)**

This iteration closes runtime-pending items for v16 once a torch/CUDA environment is available.

---

## **Single Command**

```bash
bash experiments/scripts/run_v16_iteration7_all.sh
```

This should run:
1. v16 baseline freeze,
2. v16 ablation matrix,
3. v16 performance gate,
4. summary generation for all logs.

---

## **Expected Output Folders**

- `outputs/v16-iter1-baseline/`
- `outputs/v16-ablations/`
- `outputs/v16-iter6-gate/`

Each root should contain:
- per-profile `run.log`,
- `tail.txt`,
- animation artifact (`.mp4` or `.gif`),
- `summary.csv`,
- `summary.md`.

---

## **Post-Run Completion Steps**

1. Open and fill:
   - `experiments/papers/12D-v16-PerformanceAndEvaluation.md`
2. Mark pending items complete in:
   - `experiments/papers/12A-v16-ImplementationChecklist.md`
3. Confirm final acceptance:
   - throughput/nonlocal/domain overhead budgets accepted,
   - 1024x1024 long-run gate accepted,
   - theorem-aware v15-v16 note complete.

---

## **Fallback (if long runs are too heavy initially)**

Use environment overrides:

```bash
NX=512 NY=512 THROUGHPUT_STEPS=1000 OVERHEAD_STEPS=1000 \
bash experiments/scripts/run_v16_iteration7_all.sh
```

Then repeat full 1024x1024 gate before final signoff.
