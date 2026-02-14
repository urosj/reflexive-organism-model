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
BETA_RB="${BETA_RB:-1.0}"
OUT_DIR="${OUT_DIR:-outputs/readback-baseline/tierA-smoke}"
SIM_EXTRA_ARGS="${SIM_EXTRA_ARGS:-}"

echo "[ITER-3] Tier A1 diagnostic (geometry-branch)"
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/run_readback_tierA.py" \
  --sim-script "${SIM_SCRIPT}" \
  --mode A1 \
  --seed "${SEED}" \
  --nx "${NX}" \
  --ny "${NY}" \
  --dx "${DX}" \
  --beta-rb "${BETA_RB}" \
  --out-dir "${OUT_DIR}" \
  --sim-extra-args "${SIM_EXTRA_ARGS}"

echo "[ITER-3] Tier A2 default (one-step recompute)"
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/run_readback_tierA.py" \
  --sim-script "${SIM_SCRIPT}" \
  --mode A2 \
  --seed "${SEED}" \
  --nx "${NX}" \
  --ny "${NY}" \
  --dx "${DX}" \
  --beta-rb "${BETA_RB}" \
  --out-dir "${OUT_DIR}" \
  --sim-extra-args "${SIM_EXTRA_ARGS}"

echo "[ITER-3 DONE] Tier A artifacts available under ${ROOT_DIR}/${OUT_DIR}"
