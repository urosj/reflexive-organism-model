#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/outputs/v15-iter6-gate"
mkdir -p "${OUT_DIR}/throughput" "${OUT_DIR}/snapshot-overhead"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

SEED="${SEED:-1}"
NX="${NX:-1024}"
NY="${NY:-1024}"
DX="${DX:-0.1}"
STORAGE_MODE="${STORAGE_MODE:-disk}"
FPS="${FPS:-10}"
ANIMATE_INTERVAL="${ANIMATE_INTERVAL:-100}"

# Throughput profiles: set interval > steps to minimize snapshot/animation overhead.
THROUGHPUT_STEPS="${THROUGHPUT_STEPS:-2000}"
THROUGHPUT_SNAPSHOT_INTERVAL="${THROUGHPUT_SNAPSHOT_INTERVAL:-5000}"

# Snapshot-overhead sweep: vary interval in a fixed soft-closure profile.
OVERHEAD_STEPS="${OVERHEAD_STEPS:-2000}"
OVERHEAD_INTERVALS="${OVERHEAD_INTERVALS:-10 25 50 100}"

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
      "${PYTHON_BIN}" simulations/active/simulation-v15-cuda.py "$@"
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

# 1) Throughput gate across closure modes.
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

# 2) Snapshot interval overhead sweep (soft mode).
for interval in ${OVERHEAD_INTERVALS}; do
  run_case "${OUT_DIR}/snapshot-overhead" "soft-int${interval}" \
    "${COMMON_ARGS[@]}" \
    --headless-steps "${OVERHEAD_STEPS}" \
    --snapshot-interval "${interval}" \
    --closure-mode soft
done

echo "[DONE] Iteration-6 gate logs stored in ${OUT_DIR}"
echo "[NEXT] Summarize with:"
echo "  python experiments/scripts/summarize_run_logs.py --root ${OUT_DIR} --csv ${OUT_DIR}/summary.csv --md ${OUT_DIR}/summary.md"
