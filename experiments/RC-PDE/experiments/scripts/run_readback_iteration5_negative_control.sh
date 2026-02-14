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
NEGCTRL_LAM_SCALE_PER_BETA="${NEGCTRL_LAM_SCALE_PER_BETA:-0.10}"
OUT_DIR="${OUT_DIR:-outputs/readback-baseline/tier5-negctrl-smoke}"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-${ROOT_DIR}/experiments/scripts/readback_thresholds.json}"
SIM_EXTRA_ARGS="${SIM_EXTRA_ARGS:---closure-mode soft --nonlocal-mode off --domain-mode fixed}"

echo "[ITER-5] ALI + negative-control contrast run"
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/run_readback_tierB.py" \
  --sim-script "${SIM_SCRIPT}" \
  --seed "${SEED}" \
  --nx "${NX}" \
  --ny "${NY}" \
  --dx "${DX}" \
  --steps "${STEPS}" \
  --beta-rb "${BETA_RB}" \
  --frg-beta2 "${FRG_BETA2}" \
  --run-negative-control \
  --negctrl-lam-scale-per-beta "${NEGCTRL_LAM_SCALE_PER_BETA}" \
  --out-dir "${OUT_DIR}" \
  --thresholds-json "${THRESHOLDS_JSON}" \
  --sim-extra-args "${SIM_EXTRA_ARGS}"

echo "[ITER-5 DONE] Artifacts in ${ROOT_DIR}/${OUT_DIR}"
