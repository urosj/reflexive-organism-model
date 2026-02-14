#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/outputs/v15-ablations"
mkdir -p "${OUT_DIR}/core-only" "${OUT_DIR}/core-events" "${OUT_DIR}/full"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
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

run_case() {
  local name="$1"
  shift
  local log_file="${OUT_DIR}/${name}/run.log"

  echo "[RUN] ${name}"
  (cd "${ROOT_DIR}" && "${PYTHON_BIN}" simulations/active/simulation-v15-cuda.py "${COMMON_ARGS[@]}" "$@") >"${log_file}" 2>&1

  if [[ -f "${ROOT_DIR}/simulation_output.mp4" ]]; then
    mv "${ROOT_DIR}/simulation_output.mp4" "${OUT_DIR}/${name}/simulation_output.mp4"
  elif [[ -f "${ROOT_DIR}/simulation_output.gif" ]]; then
    mv "${ROOT_DIR}/simulation_output.gif" "${OUT_DIR}/${name}/simulation_output.gif"
  else
    echo "[WARN] ${name}: no animation output file found" >>"${log_file}"
  fi

  tail -n 8 "${log_file}" >"${OUT_DIR}/${name}/tail.txt"
}

# L0 only (no L1 control signal in core, no L2 closure identities)
run_case "core-only" \
  --closure-mode off

# L0 + L1 signal in core (still no L2 closure identities)
run_case "core-events" \
  --closure-mode off \
  --events-control-in-core

# L0 + L1 + L2 hard closure policy
run_case "full" \
  --closure-mode full

echo "[DONE] Ablation artifacts stored under ${OUT_DIR}"
