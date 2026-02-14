# **RC-v15 Iteration 1 — Baseline Freeze**

Companion to:
- `experiments/papers/11-RC-v15-Spec.md`
- `experiments/papers/11A-v15-ImplementationChecklist.md`

This freeze defines reproducible v13/v14 reference runs used before v15 refactoring.

---

## **Frozen Baseline Profile**

Use the same runtime profile for both simulators:

- `seed=1`
- `nx=512`, `ny=512`, `dx=0.1`
- `headless_steps=2000`
- `snapshot_interval=50`
- `storage_mode=memory`
- `fps=10`, `animate_interval=100`

Rationale:
- same geometry and horizon across versions,
- enough snapshots for offline morphology comparison,
- memory mode avoids auto-cleanup of disk snapshot folders in current scripts.

---

## **Commands**

Automated runner:

```bash
bash experiments/scripts/run_v15_iter1_baselines.sh
```

### **v13 baseline**

```bash
python simulations/active/simulation-v13-cuda.py \
  --headless \
  --headless-steps 2000 \
  --nx 512 --ny 512 --dx 0.1 \
  --seed 1 \
  --snapshot-interval 50 \
  --storage-mode memory \
  --fps 10 \
  --animate-interval 100
```

### **v14 baseline**

```bash
python simulations/active/simulation-v14-cuda.py \
  --headless \
  --headless-steps 2000 \
  --nx 512 --ny 512 --dx 0.1 \
  --seed 1 \
  --snapshot-interval 50 \
  --storage-mode memory \
  --fps 10 \
  --animate-interval 100 \
  --closure-softness 0.6 \
  --spark-softness 0.08 \
  --collapse-softness 0.5
```

---

## **Artifacts to Preserve**

For each run, preserve:

1. stdout/stderr log,
2. offline animation file (`simulation_output.mp4` or `.gif`),
3. final summary line with:
   - steps,
   - dt,
   - mass,
   - I_mass,
   - ids.

Recommended destination:
- `outputs/v15-iter1-baseline/v13/`
- `outputs/v15-iter1-baseline/v14/`

---

## **Notes**

- Run baselines before introducing behavioral changes in v15.
- Keep seed and profile fixed for all future ablation comparisons.
