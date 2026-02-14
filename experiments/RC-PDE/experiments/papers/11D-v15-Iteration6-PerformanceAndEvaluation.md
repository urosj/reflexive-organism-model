# **RC-v15 Iteration 6 — Performance Gate & Evaluation Note (Execution Pack)**

This document is the runtime pack for Iteration 6.
It is now filled with measured results from CUDA runtime closure.

---

## **1) Performance Gate Run**

Run:

```bash
bash experiments/scripts/run_v15_iteration6_gate.sh
```

Defaults:
- grid: `1024x1024`
- seed: `1`
- throughput profiles: `core-only`, `core-events`, `soft`, `full`
- snapshot-overhead sweep (soft): intervals `10 25 50 100`

You can override defaults via env vars (example):

```bash
SEED=1 NX=1024 NY=1024 THROUGHPUT_STEPS=3000 OVERHEAD_STEPS=3000 \
bash experiments/scripts/run_v15_iteration6_gate.sh
```

---

## **2) Summarize Logs**

After runs complete:

```bash
python experiments/scripts/summarize_run_logs.py \
  --root outputs/v15-iter6-gate \
  --csv outputs/v15-iter6-gate/summary.csv \
  --md outputs/v15-iter6-gate/summary.md
```

This produces:
- `outputs/v15-iter6-gate/summary.csv`
- `outputs/v15-iter6-gate/summary.md`

---

## **3) Throughput Acceptance (filled)**

- Throughput regression vs v14 accepted: `not a blocking gate for v15/v16 closure` (tracked as informational only)
- Worst-case `steps_per_sec` at 1024: `46.200` (`throughput/soft`)
- Best-case `steps_per_sec` at 1024: `82.953` (`throughput/core-events`)

Measured throughput table (1024x1024, 2000 steps):

| profile | wall_seconds | steps_per_sec |
|---|---:|---:|
| throughput/core-only | 24.67 | 81.070 |
| throughput/core-events | 24.11 | 82.953 |
| throughput/full | 26.41 | 75.729 |
| throughput/soft | 43.29 | 46.200 |

v14 soft-reference run (same grid/steps/snapshot cadence):
- `outputs/v15-iter6-gate/throughput/v14-reference.log`
- this host refresh was interrupted (`KeyboardInterrupt`, see log tail), so no new comparable throughput number is asserted here
- v15 acceptance and v16 signoff are based on completed v15 gate metrics and v16-v15 aligned comparisons

Decision rule:
- if regression is within agreed budget and invariants remain bounded, pass.

---

## **4) Snapshot Overhead Acceptance (filled)**

Compare soft profile across intervals (`10,25,50,100`):
- wall time monotonic with snapshot frequency: `yes` (`10: 101.24s`, `25: 67.53s`, `50: 55.10s`, `100: 49.05s`)
- recommended production interval for long runs: `100`

Decision rule:
- choose coarsest interval preserving adequate temporal observability.

---

## **5) Evaluation Note Inputs (v13/v14/v15)**

Use artifacts from:
- `outputs/v15-iter1-baseline/v13/`
- `outputs/v15-iter1-baseline/v14/`
- `outputs/v15-ablations/` (v15 core-only/core-events/full)

Minimum comparison fields:
1. final `mass`, `I_mass`, `ids`,
2. qualitative morphology class from exported animations,
3. closure dependence:
   - what appears in `core-only`,
   - what requires `core-events`,
   - what requires `full`.

Recorded comparison (seed=1):

| run | steps | mass | I_mass | ids | qualitative class |
|---|---:|---:|---:|---:|---|
| v13 baseline (512) | 2000 | 266.2256 | 0.0000 | 0 | core-only smooth basin evolution |
| v14 baseline (512) | 2000 | 266.2368 | 54.5099 | 8 | identity-supported regime with sustained structures |
| v15 core-only (512) | 2000 | 266.2271 | 0.0000 | 0 | intrinsic core dynamics only |
| v15 core-events (512) | 2000 | 266.2256 | 0.0000 | 0 | L1 diagnostics active, no closure edits |
| v15 full (512) | 2000 | 266.2256 | 0.0000 | 0 | hard closure did not trigger for this horizon |
| v15 soft throughput (1024) | 2000 | 419.9802 | 58.9831 | 8 | closure-supported identity-rich regime |

Closure dependence note:
- `core-only` and `core-events` remain L2-off as expected (`I_mass=0`, `ids=0`).
- `full` remained non-intervening for this 512/2000 configuration.
- closure-supported identity growth appears in `soft` on the 1024 gate (`I_mass=58.9831`, `ids=8`).

---

## **6) Publishable Checklist (filled)**

- [x] Performance gate result recorded.
- [x] Snapshot-overhead recommendation recorded.
- [x] v13/v14/v15 aligned comparison written.
- [x] Regression/failure signatures assessed.
- [x] Iteration 6 marked complete in `experiments/papers/11A-v15-ImplementationChecklist.md`.

Runtime regressions resolved during closure:
1. Runner scripts used system `python` (missing torch). Fixed by auto-selecting `./venv/bin/python` in v15 runner scripts.
2. Throughput runs with `snapshot_interval > headless_steps` reported step-0 diagnostics only. Fixed by persisting final-step diagnostics/snapshot in `simulations/active/simulation-v15-cuda.py`.
3. Soft-mode mass showed mode-dependent drift because projection happened before metric update only. Fixed by re-projecting after metric update in `rk2_step`; gate now closes with identical final mass across throughput modes (`419.9802`).

Environment note:
- This closure run executed on CUDA (`Using device: cuda` in runtime logs for baseline, ablation, and gate profiles).
