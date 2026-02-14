#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/outputs/v15-iter1-baseline"
mkdir -p "${OUT_DIR}/v13" "${OUT_DIR}/v14"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
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

  tail -n 5 "${log_file}" >"${OUT_DIR}/${name}/tail.txt"
}

run_and_capture \
  "v13" \
  "${OUT_DIR}/v13/run.log" \
  "${PYTHON_BIN}" simulations/active/simulation-v13-cuda.py \
    --headless \
    --headless-steps 2000 \
    --nx 512 --ny 512 --dx 0.1 \
    --seed 1 \
    --snapshot-interval 50 \
    --storage-mode memory \
    --fps 10 \
    --animate-interval 100

run_and_capture \
  "v14" \
  "${OUT_DIR}/v14/run.log" \
  "${PYTHON_BIN}" simulations/active/simulation-v14-cuda.py \
    --headless \
    --headless-steps 2000 \
    --nx 512 --ny 512 --dx 0.1 \
    --seed 1 \
    --snapshot-interval 50 \
    --storage-mode memory \
    --fps 10 \
    --animate-interval 100 \
    --closure-softness 0.6 \
    --spark-softness 0.08 \
    --collapse-softness 0.5

echo "[DONE] Baseline artifacts stored under ${OUT_DIR}"
