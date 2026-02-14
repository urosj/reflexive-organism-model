# **RC-v15 Iteration 5 — Ablation Harness**

Companion to:
- `experiments/papers/11-RC-v15-Spec.md`
- `experiments/papers/11A-v15-ImplementationChecklist.md`

Run all required v15 ablations with one command:

```bash
bash experiments/scripts/run_v15_ablations.sh
```

## Profiles

1. **core-only**
   - `--closure-mode off`
   - L0 only, no L1 control signal injected into core, no L2 closure identities.

2. **core-events**
   - `--closure-mode off --events-control-in-core`
   - L0 + L1 spark signal in core, still no L2 closure identities.

3. **full**
   - `--closure-mode full`
   - L0 + L1 + L2 with hard closure policy.

## Output layout

- `outputs/v15-ablations/core-only/`
- `outputs/v15-ablations/core-events/`
- `outputs/v15-ablations/full/`

Each folder contains:
- `run.log`,
- `tail.txt`,
- `simulation_output.mp4` or `.gif`.

## Environment overrides

Set run parameters without editing the script:

```bash
SEED=1 NX=512 NY=512 HEADLESS_STEPS=2000 SNAPSHOT_INTERVAL=50 \
bash experiments/scripts/run_v15_ablations.sh
```
