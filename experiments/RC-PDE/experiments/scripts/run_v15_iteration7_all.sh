#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

echo "[ITER-7] Step 1/6: v13/v14 baseline freeze"
bash "${ROOT_DIR}/experiments/scripts/run_v15_iter1_baselines.sh"

echo "[ITER-7] Step 2/6: v15 ablation set"
bash "${ROOT_DIR}/experiments/scripts/run_v15_ablations.sh"

echo "[ITER-7] Step 3/6: v15 iteration-6 performance gate"
bash "${ROOT_DIR}/experiments/scripts/run_v15_iteration6_gate.sh"

echo "[ITER-7] Step 4/6: summarize baseline logs"
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/summarize_run_logs.py" \
  --root "${ROOT_DIR}/outputs/v15-iter1-baseline" \
  --csv "${ROOT_DIR}/outputs/v15-iter1-baseline/summary.csv" \
  --md "${ROOT_DIR}/outputs/v15-iter1-baseline/summary.md"

echo "[ITER-7] Step 5/6: summarize ablation logs"
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/summarize_run_logs.py" \
  --root "${ROOT_DIR}/outputs/v15-ablations" \
  --csv "${ROOT_DIR}/outputs/v15-ablations/summary.csv" \
  --md "${ROOT_DIR}/outputs/v15-ablations/summary.md"

echo "[ITER-7] Step 6/6: summarize performance gate logs"
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/summarize_run_logs.py" \
  --root "${ROOT_DIR}/outputs/v15-iter6-gate" \
  --csv "${ROOT_DIR}/outputs/v15-iter6-gate/summary.csv" \
  --md "${ROOT_DIR}/outputs/v15-iter6-gate/summary.md"

echo "[ITER-7 DONE] Runtime artifacts are ready."
echo "Next: complete experiments/papers/11D-v15-Iteration6-PerformanceAndEvaluation.md and mark pending checkboxes in experiments/papers/11A-v15-ImplementationChecklist.md."
