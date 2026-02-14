#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/outputs/v16-ablations"
mkdir -p "${OUT_DIR}/core-only" "${OUT_DIR}/core-events" "${OUT_DIR}/soft" "${OUT_DIR}/full" "${OUT_DIR}/nonlocal-off" "${OUT_DIR}/nonlocal-on"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

SIM_SCRIPT="${SIM_SCRIPT:-simulations/active/simulation-v16-cuda.py}"
if [[ ! -f "${ROOT_DIR}/${SIM_SCRIPT}" ]]; then
  echo "[WARN] ${SIM_SCRIPT} not found. Falling back to simulations/active/simulation-v15-cuda.py for scaffold runs."
  SIM_SCRIPT="simulations/active/simulation-v15-cuda.py"
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

COMMON_EXTRA_ARGS="${COMMON_EXTRA_ARGS:-}"
NONLOCAL_OFF_EXTRA_ARGS="${NONLOCAL_OFF_EXTRA_ARGS:-}"
NONLOCAL_ON_EXTRA_ARGS="${NONLOCAL_ON_EXTRA_ARGS:-}"

COMMON_EXTRA=()
if [[ -n "${COMMON_EXTRA_ARGS}" ]]; then
  read -r -a COMMON_EXTRA <<<"${COMMON_EXTRA_ARGS}"
fi

NONLOCAL_OFF_EXTRA=()
if [[ -n "${NONLOCAL_OFF_EXTRA_ARGS}" ]]; then
  read -r -a NONLOCAL_OFF_EXTRA <<<"${NONLOCAL_OFF_EXTRA_ARGS}"
fi

NONLOCAL_ON_EXTRA=()
if [[ -n "${NONLOCAL_ON_EXTRA_ARGS}" ]]; then
  read -r -a NONLOCAL_ON_EXTRA <<<"${NONLOCAL_ON_EXTRA_ARGS}"
fi

COMMON_ARGS=(
  --headless
  --headless-steps "${HEADLESS_STEPS}"
  --nx "${NX}"
  --ny "${NY}"
  --dx "${DX}"
  --seed "${SEED}"
  --snapshot-interval "${SNAPSHOT_INTERVAL}"
  --storage-mode "${STORAGE_MODE}"
  --fps "${FPS}"
  --animate-interval "${ANIMATE_INTERVAL}"
)
if [[ "${OPERATOR_DIAGNOSTICS}" != "0" ]]; then
  COMMON_ARGS+=(--operator-diagnostics --operator-diagnostics-interval "${OPERATOR_DIAGNOSTICS_INTERVAL}")
fi

run_case() {
  local name="$1"
  shift
  local log_file="${OUT_DIR}/${name}/run.log"

  echo "[RUN] ${name}"
  (cd "${ROOT_DIR}" && "${PYTHON_BIN}" "${SIM_SCRIPT}" "${COMMON_ARGS[@]}" "${COMMON_EXTRA[@]}" "$@") >"${log_file}" 2>&1

  if [[ -f "${ROOT_DIR}/simulation_output.mp4" ]]; then
    mv "${ROOT_DIR}/simulation_output.mp4" "${OUT_DIR}/${name}/simulation_output.mp4"
  elif [[ -f "${ROOT_DIR}/simulation_output.gif" ]]; then
    mv "${ROOT_DIR}/simulation_output.gif" "${OUT_DIR}/${name}/simulation_output.gif"
  else
    echo "[WARN] ${name}: no animation output file found" >>"${log_file}"
  fi

  tail -n 10 "${log_file}" >"${OUT_DIR}/${name}/tail.txt"
}

run_case "core-only" \
  --closure-mode off

run_case "core-events" \
  --closure-mode off \
  --events-control-in-core

run_case "soft" \
  --closure-mode soft

run_case "full" \
  --closure-mode full

# Nonlocal ablation hooks: pass profile-specific flags via NONLOCAL_*_EXTRA_ARGS.
run_case "nonlocal-off" \
  --closure-mode off \
  --nonlocal-mode off \
  "${NONLOCAL_OFF_EXTRA[@]}"

run_case "nonlocal-on" \
  --closure-mode off \
  --nonlocal-mode on \
  "${NONLOCAL_ON_EXTRA[@]}"

echo "[DONE] v16 ablation artifacts stored under ${OUT_DIR}"
