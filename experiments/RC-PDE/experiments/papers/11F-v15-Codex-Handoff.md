# RC-v15 Codex Handoff (Resume Point)

Use this document to continue exactly where the previous Codex session stopped.

## 1) Mission

Complete **Iteration 7 (CUDA Runtime Closure)** for v15:

- execute all prepared runtime pipelines,
- collect summaries/artifacts,
- resolve any runtime regressions found,
- finalize docs/checklists with actual measured results.

Primary references:

- `experiments/papers/11A-v15-ImplementationChecklist.md`
- `experiments/papers/11D-v15-Iteration6-PerformanceAndEvaluation.md`
- `experiments/papers/11E-v15-Iteration7-CUDARuntimeClosure.md`

## 2) Current State (Already Implemented)

Implemented and statically validated:

- `simulations/active/simulation-v15-cuda.py` with L0/L1/L2 mode routing:
  - `--closure-mode off|soft|full`
  - `--events-control-in-core` for `core-events` ablation.
- v15 ablation/baseline/performance scripts:
  - `experiments/scripts/run_v15_iter1_baselines.sh`
  - `experiments/scripts/run_v15_ablations.sh`
  - `experiments/scripts/run_v15_iteration6_gate.sh`
  - `experiments/scripts/run_v15_iteration7_all.sh`
- log summarizer:
  - `experiments/scripts/summarize_run_logs.py`
- v15 structural test suite:
  - `tests/test_v15_cuda_*.py` (5 files)

Known: previous environment had no torch/CUDA, so runtime execution is pending.

## 3) First Commands To Run (In CUDA Environment)

From repo root:

```bash
bash experiments/scripts/run_v15_iteration7_all.sh
```

If runs are too heavy initially:

```bash
NX=512 NY=512 THROUGHPUT_STEPS=1000 OVERHEAD_STEPS=1000 \
bash experiments/scripts/run_v15_iteration7_all.sh
```

Skip this one: Then repeat full 1024x1024 gate before final signoff.

## 4) Expected Artifacts

After successful run, verify:

- `outputs/v15-iter1-baseline/summary.md`
- `outputs/v15-ablations/summary.md`
- `outputs/v15-iter6-gate/summary.md`

and per-profile `run.log` + `tail.txt` + animation output.

## 5) If Runtime Issues Appear

1. Diagnose from `run.log` in failing profile folder.
2. Patch minimally in `simulations/active/simulation-v15-cuda.py` (do not rewrite architecture).
3. Re-run only failing profile first, then re-run full iteration-7 pipeline.
4. Keep ablation contract intact:
   - `core-only`: `--closure-mode off`
   - `core-events`: `--closure-mode off --events-control-in-core`
   - `full`: `--closure-mode full`

## 6) Finalization Tasks

After runtime pass:

1. Fill `experiments/papers/11D-v15-Iteration6-PerformanceAndEvaluation.md` with measured values.
2. Mark remaining unchecked items in `experiments/papers/11A-v15-ImplementationChecklist.md` as done.
3. Run v15 tests:

```bash
python -m pytest -q tests/test_v15_cuda_modes.py tests/test_v15_cuda_core_invariants.py tests/test_v15_cuda_event_metrics.py tests/test_v15_cuda_closure_policy.py tests/test_v15_cuda_ablation_contract.py
```

4. Add short final note (what passed, what changed) in commit/PR message.

## 7) Important Workspace Note

There is an unrelated modified file outside this project tree:

- `../../../../Obsidian/SentientLife/Self/26-02/2026-02-12.md`

Do not revert or touch it.
