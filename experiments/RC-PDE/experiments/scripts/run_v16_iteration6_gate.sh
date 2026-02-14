#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/outputs/v16-iter6-gate"
mkdir -p "${OUT_DIR}/throughput" "${OUT_DIR}/snapshot-overhead"

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
NX="${NX:-1024}"
NY="${NY:-1024}"
DX="${DX:-0.1}"
STORAGE_MODE="${STORAGE_MODE:-disk}"
FPS="${FPS:-10}"
ANIMATE_INTERVAL="${ANIMATE_INTERVAL:-100}"
OPERATOR_DIAGNOSTICS="${OPERATOR_DIAGNOSTICS:-1}"
OPERATOR_DIAGNOSTICS_INTERVAL="${OPERATOR_DIAGNOSTICS_INTERVAL:-10}"

THROUGHPUT_STEPS="${THROUGHPUT_STEPS:-2000}"
THROUGHPUT_SNAPSHOT_INTERVAL="${THROUGHPUT_SNAPSHOT_INTERVAL:-5000}"

OVERHEAD_STEPS="${OVERHEAD_STEPS:-2000}"
OVERHEAD_INTERVALS="${OVERHEAD_INTERVALS:-10 25 50 100}"
ADAPTIVE_DOMAIN_STRENGTH="${ADAPTIVE_DOMAIN_STRENGTH:-0.30}"
ADAPTIVE_DOMAIN_INTERVAL="${ADAPTIVE_DOMAIN_INTERVAL:-50}"

COMMON_EXTRA_ARGS="${COMMON_EXTRA_ARGS:-}"
NONLOCAL_OFF_EXTRA_ARGS="${NONLOCAL_OFF_EXTRA_ARGS:-}"
NONLOCAL_ON_EXTRA_ARGS="${NONLOCAL_ON_EXTRA_ARGS:-}"
NONLOCAL_OVERHEAD_EXTRA_ARGS="${NONLOCAL_OVERHEAD_EXTRA_ARGS:-}"

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

NONLOCAL_OVERHEAD_EXTRA=()
if [[ -n "${NONLOCAL_OVERHEAD_EXTRA_ARGS}" ]]; then
  read -r -a NONLOCAL_OVERHEAD_EXTRA <<<"${NONLOCAL_OVERHEAD_EXTRA_ARGS}"
fi

run_case() {
  local run_dir="$1"
  local run_name="$2"
  shift 2

  mkdir -p "${run_dir}/${run_name}"
  local log_file="${run_dir}/${run_name}/run.log"

  echo "[RUN] ${run_name}"
  (
    cd "${ROOT_DIR}"
    /usr/bin/time -f "wall_seconds=%e" \
      "${PYTHON_BIN}" "${SIM_SCRIPT}" "${COMMON_EXTRA[@]}" "$@"
  ) >"${log_file}" 2>&1

  if [[ -f "${ROOT_DIR}/simulation_output.mp4" ]]; then
    mv "${ROOT_DIR}/simulation_output.mp4" "${run_dir}/${run_name}/simulation_output.mp4"
  elif [[ -f "${ROOT_DIR}/simulation_output.gif" ]]; then
    mv "${ROOT_DIR}/simulation_output.gif" "${run_dir}/${run_name}/simulation_output.gif"
  else
    echo "[WARN] ${run_name}: no animation output file found" >>"${log_file}"
  fi

  tail -n 12 "${log_file}" >"${run_dir}/${run_name}/tail.txt"
}

COMMON_ARGS=(
  --headless
  --nx "${NX}"
  --ny "${NY}"
  --dx "${DX}"
  --seed "${SEED}"
  --storage-mode "${STORAGE_MODE}"
  --fps "${FPS}"
  --animate-interval "${ANIMATE_INTERVAL}"
)
if [[ "${OPERATOR_DIAGNOSTICS}" != "0" ]]; then
  COMMON_ARGS+=(--operator-diagnostics --operator-diagnostics-interval "${OPERATOR_DIAGNOSTICS_INTERVAL}")
fi

run_case "${OUT_DIR}/throughput" "core-only" \
  "${COMMON_ARGS[@]}" \
  --headless-steps "${THROUGHPUT_STEPS}" \
  --snapshot-interval "${THROUGHPUT_SNAPSHOT_INTERVAL}" \
  --closure-mode off

run_case "${OUT_DIR}/throughput" "core-events" \
  "${COMMON_ARGS[@]}" \
  --headless-steps "${THROUGHPUT_STEPS}" \
  --snapshot-interval "${THROUGHPUT_SNAPSHOT_INTERVAL}" \
  --closure-mode off \
  --events-control-in-core

run_case "${OUT_DIR}/throughput" "soft" \
  "${COMMON_ARGS[@]}" \
  --headless-steps "${THROUGHPUT_STEPS}" \
  --snapshot-interval "${THROUGHPUT_SNAPSHOT_INTERVAL}" \
  --closure-mode soft

run_case "${OUT_DIR}/throughput" "full" \
  "${COMMON_ARGS[@]}" \
  --headless-steps "${THROUGHPUT_STEPS}" \
  --snapshot-interval "${THROUGHPUT_SNAPSHOT_INTERVAL}" \
  --closure-mode full

run_case "${OUT_DIR}/throughput" "nonlocal-off" \
  "${COMMON_ARGS[@]}" \
  --headless-steps "${THROUGHPUT_STEPS}" \
  --snapshot-interval "${THROUGHPUT_SNAPSHOT_INTERVAL}" \
  --closure-mode off \
  --nonlocal-mode off \
  "${NONLOCAL_OFF_EXTRA[@]}"

run_case "${OUT_DIR}/throughput" "nonlocal-on" \
  "${COMMON_ARGS[@]}" \
  --headless-steps "${THROUGHPUT_STEPS}" \
  --snapshot-interval "${THROUGHPUT_SNAPSHOT_INTERVAL}" \
  --closure-mode off \
  --nonlocal-mode on \
  "${NONLOCAL_ON_EXTRA[@]}"

run_case "${OUT_DIR}/throughput" "adaptive-soft" \
  "${COMMON_ARGS[@]}" \
  --headless-steps "${THROUGHPUT_STEPS}" \
  --snapshot-interval "${THROUGHPUT_SNAPSHOT_INTERVAL}" \
  --closure-mode soft \
  --nonlocal-mode on \
  --domain-mode adaptive \
  --domain-adapt-strength "${ADAPTIVE_DOMAIN_STRENGTH}" \
  --domain-adapt-interval "${ADAPTIVE_DOMAIN_INTERVAL}" \
  "${NONLOCAL_OVERHEAD_EXTRA[@]}"

for interval in ${OVERHEAD_INTERVALS}; do
  run_case "${OUT_DIR}/snapshot-overhead" "soft-int${interval}" \
    "${COMMON_ARGS[@]}" \
    --headless-steps "${OVERHEAD_STEPS}" \
    --snapshot-interval "${interval}" \
    --closure-mode soft \
    --nonlocal-mode on \
    "${NONLOCAL_OVERHEAD_EXTRA[@]}"

  run_case "${OUT_DIR}/snapshot-overhead" "adaptive-soft-int${interval}" \
    "${COMMON_ARGS[@]}" \
    --headless-steps "${OVERHEAD_STEPS}" \
    --snapshot-interval "${interval}" \
    --closure-mode soft \
    --nonlocal-mode on \
    --domain-mode adaptive \
    --domain-adapt-strength "${ADAPTIVE_DOMAIN_STRENGTH}" \
    --domain-adapt-interval "${ADAPTIVE_DOMAIN_INTERVAL}" \
    "${NONLOCAL_OVERHEAD_EXTRA[@]}"
done

echo "[DONE] v16 iteration-6 gate logs stored in ${OUT_DIR}"
echo "[NEXT] Summarize with:"
echo "  python experiments/scripts/summarize_run_logs.py --root ${OUT_DIR} --csv ${OUT_DIR}/summary.csv --md ${OUT_DIR}/summary.md"
