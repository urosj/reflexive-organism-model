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

SIM_SCRIPT="${SIM_SCRIPT:-simulations/active/simulation-v16-cuda.py}"
SEED="${SEED:-1}"
NX="${NX:-64}"
NY="${NY:-64}"
DX="${DX:-0.1}"
STEPS="${STEPS:-8}"
BETA_RB="${BETA_RB:-1.0}"
FRG_BETA2="${FRG_BETA2:-2.0}"
REPEATS="${REPEATS:-3}"
OUT_DIR="${OUT_DIR:-outputs/readback-baseline/tierB-stability}"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-${ROOT_DIR}/experiments/scripts/readback_thresholds.json}"
SIM_EXTRA_ARGS="${SIM_EXTRA_ARGS:---closure-mode soft --nonlocal-mode off --domain-mode fixed}"
MAX_CV="${MAX_CV:-0.10}"
ABS_TOL="${ABS_TOL:-1e-5}"

SIM_STEM="$(basename "${SIM_SCRIPT}" .py)"
RUN_ROOT="${ROOT_DIR}/${OUT_DIR}/${SIM_STEM}/seed-${SEED}"
mkdir -p "${RUN_ROOT}"

echo "[ITER-4.5] Running ${REPEATS} repeated Tier B runs"
for i in $(seq 1 "${REPEATS}"); do
  RUN_IDX=$(printf "%02d" "${i}")
  RUN_OUT="${OUT_DIR}/${SIM_STEM}/seed-${SEED}/repeat-${RUN_IDX}"
  echo "[ITER-4.5] repeat ${RUN_IDX}/${REPEATS} -> ${RUN_OUT}"
  "${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/run_readback_tierB.py" \
    --sim-script "${SIM_SCRIPT}" \
    --seed "${SEED}" \
    --nx "${NX}" \
    --ny "${NY}" \
    --dx "${DX}" \
    --steps "${STEPS}" \
    --beta-rb "${BETA_RB}" \
    --frg-beta2 "${FRG_BETA2}" \
    --out-dir "${RUN_OUT}" \
    --thresholds-json "${THRESHOLDS_JSON}" \
    --sim-extra-args "${SIM_EXTRA_ARGS}"
done

echo "[ITER-4.5] Aggregating stability"
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/check_readback_stability.py" \
  --root "${RUN_ROOT}" \
  --max-cv "${MAX_CV}" \
  --abs-tol "${ABS_TOL}" \
  --out-json "${RUN_ROOT}/stability.json" \
  --out-md "${RUN_ROOT}/stability.md"

echo "[ITER-4.5 DONE] Stability artifacts in ${RUN_ROOT}"
