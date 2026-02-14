## Suggested run envelope (for controlled comparisons)
- `nx=512`, `ny=512`, `dx=0.1`
- `steps=2000`
- `seed=1` (use `1,2,3` for stronger confidence)
- `closure-mode=soft`, `nonlocal-mode=on`, `domain-mode=adaptive`

If you compare versions/runs, keep this envelope aligned; otherwise mark conclusions as **not controlled**.

Use this from repo root:

```bash
python simulations/active/simulation-v16-cuda.py \
  --headless \
  --headless-steps 2000 \
  --nx 512 --ny 512 --dx 0.1 \
  --seed 1 \
  --storage-mode disk \
  --snapshot-interval 50 \
  --closure-mode soft \
  --nonlocal-mode on \
  --operator-diagnostics \
  --operator-diagnostics-interval 10 \
  --domain-mode adaptive \
  --domain-adapt-strength 0.30 \
  --domain-adapt-interval 50
```

This gives you:
- simulator snapshots/artifacts
- full runtime telemetry stream at `outputs/readback-telemetry/*.jsonl`
- operator/domain/event diagnostics in logs and metadata

If you want the full Paper-14 audit packet (`scores.json`, `series.npz`, `figures/`, `report.md`), run:
```bash
bash experiments/scripts/run_readback_iteration4_tierB.sh
```

For a fast pipeline smoke check (schema + short Tier B + packet verifier), run:
```bash
bash experiments/scripts/run_readback_ci_smoke.sh
```

Then verify packet completeness:
```bash
./venv/bin/python experiments/scripts/verify_readback_audit_packet.py \
  --packets-root outputs/readback-baseline/tierB-smoke \
  --out-json outputs/readback-baseline/tierB-smoke/verification.json \
  --out-md outputs/readback-baseline/tierB-smoke/verification.md
```

For standardized interpretation of packet outputs, use:
- `experiments/papers/14E-ExperienceReadback-InterpretationGuide.md`

## Runtime and expected artifacts
- Single v16 run:
  - fastest path; produces sim artifacts + hook stream under `outputs/readback-telemetry/`.
- Tier A:
  - local counterfactual diagnostics (`DeltaJ/DeltaG/DeltaL`, CAI) in `outputs/readback-baseline/tierA-*`.
- Tier B:
  - slower (multi-branch) and produces full audit packet (`scores.json`, `series.npz`, `report.md`, canonical figures).
- Matrix:
  - slowest (multi-version, optional multi-seed), produces fingerprint grid + verification outputs.

Or use the telemetry in 3 practical ways:

1. **Quick inspect the hook stream**
```bash
HOOK_LOG="$(ls -t outputs/readback-telemetry/simulation-v16-cuda-seed-1-*.jsonl | head -n 1)"
echo "$HOOK_LOG"
tail -n 20 "$HOOK_LOG"
```
This confirms stages are present (`post_*` hooks, operator/domain/event signals).

2. **Build a full audit packet (recommended)**
```bash
./venv/bin/python experiments/scripts/run_readback_tierB.py \
  --sim-script simulations/active/simulation-v16-cuda.py \
  --seed 1 --nx 512 --ny 512 --dx 0.1 --steps 2000 \
  --snapshot-interval 50 --storage-mode memory \
  --beta-rb 1.0 --frg-beta2 2.0 \
  --sim-extra-args "--closure-mode soft --nonlocal-mode on --operator-diagnostics --operator-diagnostics-interval 10 --domain-mode adaptive --domain-adapt-strength 0.30 --domain-adapt-interval 50" \
  --out-dir outputs/readback-baseline/tierB-v16-manual
```
Then open:
- `outputs/readback-baseline/tierB-v16-manual/simulation-v16-cuda/seed-1/report.md`
- `.../scores.json`
- `.../figures/frg_vs_beta.png`
- `.../figures/lce_lag_curves.png`
- `.../figures/cai_waterfall_t*.png`

3. **Run matrix comparison (v14–v16)**
```bash
./venv/bin/python experiments/scripts/run_readback_matrix.py \
  --sim-scripts simulations/active/simulation-v14-cuda.py,simulations/active/simulation-v15-cuda.py,simulations/active/simulation-v16-cuda.py \
  --seeds 1 --nx 512 --ny 512 --dx 0.1 --steps 2000 --tiera-steps 2 \
  --snapshot-interval 50 --storage-mode memory \
  --beta-rb 1.0 --frg-beta2 2.0 \
  --global-sim-extra-args "--closure-mode soft --nonlocal-mode on --operator-diagnostics --operator-diagnostics-interval 10 --domain-mode adaptive --domain-adapt-strength 0.30 --domain-adapt-interval 50" \
  --out-dir outputs/readback-baseline/tier6-v14-v16-512
```
Then inspect:
- `.../report.md`
- `.../figures/version_fingerprint_grid.png`

## Troubleshooting
- CUDA not available:
  - run still works on CPU, but throughput is much lower.
- Missing `scores.json` or `report.md`:
  - treat as incomplete packet; check branch `run.log` and rerun verifier.
- No `hook_stream` / telemetry path in logs:
  - ensure `RC_READBACK_HOOKS=1` and headless mode are active.
- Matrix comparison disagreement:
  - re-check envelope alignment (grid/steps/modes/seeds); otherwise classify as not controlled.

## Tier A vs Tier B (quick distinction)
- **Tier A**: single-state counterfactual check ("does read-back do work now?").
  - Outputs immediate deltas like `DeltaJ`, `DeltaG`, `DeltaL` and CAI attenuation.
  - Fast, local diagnostic.
- **Tier B**: paired-trajectory audit ("does read-back accumulate over time?").
  - Outputs `scores.json`, `series.npz`, standard figures, and `report.md`.
  - Usually much slower than one plain v16 run because it runs multiple branches.
