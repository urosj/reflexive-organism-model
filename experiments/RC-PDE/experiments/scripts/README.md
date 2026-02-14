# Experiments Scripts

Canonical location for experiment runners, scoring, schema, and verification scripts.

Key entrypoints:
- `bash experiments/scripts/run_readback_ci_smoke.sh`
  - Fast smoke: schema validate + short Tier B packet + packet verifier.
  - Use `STRICT_VERIFY=1` to make verifier gaps fail the script.
- `bash experiments/scripts/run_readback_iteration3_tierA.sh`
  - Tier A counterfactual snapshot diagnostics.
- `bash experiments/scripts/run_readback_iteration4_tierB.sh`
  - Tier B paired-trajectory audit packet.
- `bash experiments/scripts/run_readback_iteration6_matrix.sh`
  - Cross-version matrix packet + fingerprint report.
