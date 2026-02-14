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
NX="${NX:-128}"
NY="${NY:-128}"
DX="${DX:-0.1}"
STEPS="${STEPS:-20}"
BETA_RB="${BETA_RB:-1.0}"
FRG_BETA2="${FRG_BETA2:-}"
OUT_DIR="${OUT_DIR:-outputs/readback-baseline/tierB-smoke}"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-${ROOT_DIR}/experiments/scripts/readback_thresholds.json}"
SIM_EXTRA_ARGS="${SIM_EXTRA_ARGS:-}"

EXTRA_ARGS=()
if [[ -n "${FRG_BETA2}" ]]; then
  EXTRA_ARGS+=(--frg-beta2 "${FRG_BETA2}")
fi
if [[ -n "${THRESHOLDS_JSON}" ]]; then
  EXTRA_ARGS+=(--thresholds-json "${THRESHOLDS_JSON}")
fi

echo "[ITER-4] Tier B paired-trajectory lag run"
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/run_readback_tierB.py" \
  --sim-script "${SIM_SCRIPT}" \
  --seed "${SEED}" \
  --nx "${NX}" \
  --ny "${NY}" \
  --dx "${DX}" \
  --steps "${STEPS}" \
  --beta-rb "${BETA_RB}" \
  --out-dir "${OUT_DIR}" \
  --sim-extra-args "${SIM_EXTRA_ARGS}" \
  "${EXTRA_ARGS[@]}"

echo "[ITER-4 DONE] Tier B artifacts available under ${ROOT_DIR}/${OUT_DIR}"
