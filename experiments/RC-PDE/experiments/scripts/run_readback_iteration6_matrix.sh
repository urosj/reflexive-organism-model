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

SIM_SCRIPTS="${SIM_SCRIPTS:-simulations/active/simulation-v12-cuda.py,simulations/active/simulation-v13-cuda.py,simulations/active/simulation-v14-cuda.py,simulations/active/simulation-v15-cuda.py,simulations/active/simulation-v16-cuda.py}"
SEEDS="${SEEDS:-1}"
NX="${NX:-64}"
NY="${NY:-64}"
DX="${DX:-0.1}"
STEPS="${STEPS:-12}"
TIERA_STEPS="${TIERA_STEPS:-2}"
BETA_RB="${BETA_RB:-1.0}"
FRG_BETA2="${FRG_BETA2:-2.0}"
RUN_NEGATIVE_CONTROL="${RUN_NEGATIVE_CONTROL:-1}"
NEGCTRL_LAM_SCALE_PER_BETA="${NEGCTRL_LAM_SCALE_PER_BETA:-0.10}"
OUT_DIR="${OUT_DIR:-outputs/readback-baseline/tier6-matrix}"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-experiments/scripts/readback_thresholds.json}"
GLOBAL_SIM_EXTRA_ARGS="${GLOBAL_SIM_EXTRA_ARGS:-}"

EXTRA_FLAGS=()
if [[ "${RUN_NEGATIVE_CONTROL}" == "1" ]]; then
  EXTRA_FLAGS+=(--run-negative-control --negctrl-lam-scale-per-beta "${NEGCTRL_LAM_SCALE_PER_BETA}")
fi

echo "[ITER-6] Cross-version matrix run"
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/run_readback_matrix.py" \
  --python-bin "${PYTHON_BIN}" \
  --sim-scripts "${SIM_SCRIPTS}" \
  --seeds "${SEEDS}" \
  --nx "${NX}" \
  --ny "${NY}" \
  --dx "${DX}" \
  --steps "${STEPS}" \
  --tiera-steps "${TIERA_STEPS}" \
  --beta-rb "${BETA_RB}" \
  --frg-beta2 "${FRG_BETA2}" \
  --out-dir "${OUT_DIR}" \
  --thresholds-json "${THRESHOLDS_JSON}" \
  --global-sim-extra-args "${GLOBAL_SIM_EXTRA_ARGS}" \
  "${EXTRA_FLAGS[@]}"

echo "[ITER-6 DONE] Matrix artifacts in ${ROOT_DIR}/${OUT_DIR}"
