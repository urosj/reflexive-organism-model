#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/outputs/v16-iter1-baseline"
mkdir -p "${OUT_DIR}/v15" "${OUT_DIR}/v16"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

SIM_V15_SCRIPT="${SIM_V15_SCRIPT:-simulations/active/simulation-v15-cuda.py}"
SIM_V16_SCRIPT="${SIM_V16_SCRIPT:-simulations/active/simulation-v16-cuda.py}"

if [[ ! -f "${ROOT_DIR}/${SIM_V15_SCRIPT}" ]]; then
  echo "[ERROR] Missing baseline script: ${SIM_V15_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${ROOT_DIR}/${SIM_V16_SCRIPT}" ]]; then
  echo "[WARN] ${SIM_V16_SCRIPT} not found. Falling back to ${SIM_V15_SCRIPT} for scaffold runs."
  SIM_V16_SCRIPT="${SIM_V15_SCRIPT}"
fi

SEED="${SEED:-1}"
NX="${NX:-512}"
NY="${NY:-512}"
DX="${DX:-0.1}"
HEADLESS_STEPS="${HEADLESS_STEPS:-2000}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-50}"
STORAGE_MODE="${STORAGE_MODE:-memory}"
FPS="${FPS:-10}"
ANIMATE_INTERVAL="${ANIMATE_INTERVAL:-100}"
OPERATOR_DIAGNOSTICS="${OPERATOR_DIAGNOSTICS:-1}"
OPERATOR_DIAGNOSTICS_INTERVAL="${OPERATOR_DIAGNOSTICS_INTERVAL:-10}"

V15_BASELINE_EXTRA_ARGS="${V15_BASELINE_EXTRA_ARGS:-}"
V16_BASELINE_EXTRA_ARGS="${V16_BASELINE_EXTRA_ARGS:-}"

V15_EXTRA_ARGS=()
if [[ -n "${V15_BASELINE_EXTRA_ARGS}" ]]; then
  read -r -a V15_EXTRA_ARGS <<<"${V15_BASELINE_EXTRA_ARGS}"
fi

V16_EXTRA_ARGS=()
if [[ -n "${V16_BASELINE_EXTRA_ARGS}" ]]; then
  read -r -a V16_EXTRA_ARGS <<<"${V16_BASELINE_EXTRA_ARGS}"
fi

V16_OPERATOR_ARGS=()
if [[ "${OPERATOR_DIAGNOSTICS}" != "0" ]]; then
  V16_OPERATOR_ARGS+=(--operator-diagnostics --operator-diagnostics-interval "${OPERATOR_DIAGNOSTICS_INTERVAL}")
fi

run_and_capture() {
  local name="$1"
  local log_file="$2"
  shift 2

  echo "[RUN] ${name}"
  (cd "${ROOT_DIR}" && "$@") >"${log_file}" 2>&1

  if [[ -f "${ROOT_DIR}/simulation_output.mp4" ]]; then
    mv "${ROOT_DIR}/simulation_output.mp4" "${OUT_DIR}/${name}/simulation_output.mp4"
  elif [[ -f "${ROOT_DIR}/simulation_output.gif" ]]; then
    mv "${ROOT_DIR}/simulation_output.gif" "${OUT_DIR}/${name}/simulation_output.gif"
  else
    echo "[WARN] ${name}: no animation output file found" >>"${log_file}"
  fi

  tail -n 8 "${log_file}" >"${OUT_DIR}/${name}/tail.txt"
}

run_and_capture \
  "v15" \
  "${OUT_DIR}/v15/run.log" \
  "${PYTHON_BIN}" "${SIM_V15_SCRIPT}" \
    --headless \
    --headless-steps "${HEADLESS_STEPS}" \
    --nx "${NX}" --ny "${NY}" --dx "${DX}" \
    --seed "${SEED}" \
    --snapshot-interval "${SNAPSHOT_INTERVAL}" \
    --storage-mode "${STORAGE_MODE}" \
    --fps "${FPS}" \
    --animate-interval "${ANIMATE_INTERVAL}" \
    --closure-mode soft \
    "${V15_EXTRA_ARGS[@]}"

run_and_capture \
  "v16" \
  "${OUT_DIR}/v16/run.log" \
  "${PYTHON_BIN}" "${SIM_V16_SCRIPT}" \
    --headless \
    --headless-steps "${HEADLESS_STEPS}" \
    --nx "${NX}" --ny "${NY}" --dx "${DX}" \
    --seed "${SEED}" \
    --snapshot-interval "${SNAPSHOT_INTERVAL}" \
    --storage-mode "${STORAGE_MODE}" \
    --fps "${FPS}" \
    --animate-interval "${ANIMATE_INTERVAL}" \
    --closure-mode soft \
    "${V16_OPERATOR_ARGS[@]}" \
    "${V16_EXTRA_ARGS[@]}"

echo "[DONE] v16 baseline artifacts stored under ${OUT_DIR}"
