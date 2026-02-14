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
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-50}"
STORAGE_MODE="${STORAGE_MODE:-memory}"
BETA_RB="${BETA_RB:-1.0}"
FRG_BETA2="${FRG_BETA2:-2.0}"
FIELD_INTERVAL="${FIELD_INTERVAL:-1}"
FIELD_DOWNSAMPLE="${FIELD_DOWNSAMPLE:-2}"
OUT_DIR="${OUT_DIR:-outputs/readback-baseline/ci-smoke}"
THRESHOLDS_JSON="${THRESHOLDS_JSON:-${ROOT_DIR}/experiments/scripts/readback_thresholds.json}"
SIM_EXTRA_ARGS="${SIM_EXTRA_ARGS:-}"
STRICT_VERIFY="${STRICT_VERIFY:-0}"

echo "[CI-SMOKE] 1/3 validate telemetry schema"
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/validate_readback_schema.py"

echo "[CI-SMOKE] 2/3 run short Tier B packet"
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/run_readback_tierB.py" \
  --sim-script "${SIM_SCRIPT}" \
  --seed "${SEED}" \
  --nx "${NX}" \
  --ny "${NY}" \
  --dx "${DX}" \
  --steps "${STEPS}" \
  --snapshot-interval "${SNAPSHOT_INTERVAL}" \
  --storage-mode "${STORAGE_MODE}" \
  --beta-rb "${BETA_RB}" \
  --frg-beta2 "${FRG_BETA2}" \
  --field-interval "${FIELD_INTERVAL}" \
  --field-downsample "${FIELD_DOWNSAMPLE}" \
  --thresholds-json "${THRESHOLDS_JSON}" \
  --sim-extra-args "${SIM_EXTRA_ARGS}" \
  --out-dir "${OUT_DIR}"

echo "[CI-SMOKE] 3/3 verify packet completeness"
set +e
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/scripts/verify_readback_audit_packet.py" \
  --packets-root "${ROOT_DIR}/${OUT_DIR}" \
  --out-json "${ROOT_DIR}/${OUT_DIR}/verification.json" \
  --out-md "${ROOT_DIR}/${OUT_DIR}/verification.md"
VERIFY_EXIT=$?
set -e

if [[ ${VERIFY_EXIT} -ne 0 ]]; then
  if [[ "${STRICT_VERIFY}" == "1" ]]; then
    echo "[CI-SMOKE] verification failed (STRICT_VERIFY=1)."
    exit ${VERIFY_EXIT}
  fi
  echo "[CI-SMOKE] verification reported gaps (STRICT_VERIFY=0). See ${ROOT_DIR}/${OUT_DIR}/verification.md"
fi

echo "[CI-SMOKE DONE] ${ROOT_DIR}/${OUT_DIR}"
