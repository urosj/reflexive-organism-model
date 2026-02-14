#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/clean_artifacts.sh [--full] [--help]

Default cleanup:
  - Removes all __pycache__/ directories
  - Removes all .pytest_cache/ directories
  - Removes Python bytecode artifacts (*.pyc, *.pyo)

With --full:
  - Performs default cleanup, plus removes:
    - snapshots/
    - outputs/
    - venv/
USAGE
}

full_cleanup=0

while (($# > 0)); do
  case "$1" in
    --full)
      full_cleanup=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "[CLEAN] Repo root: ${ROOT_DIR}"

# Remove cache directories and bytecode artifacts.
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

echo "[CLEAN] Removed Python cache artifacts (__pycache__, .pytest_cache, *.pyc, *.pyo)."

if [[ "${full_cleanup}" -eq 1 ]]; then
  for path in snapshots outputs venv; do
    if [[ -e "${path}" ]]; then
      rm -rf "${path}"
      echo "[CLEAN] Removed ${path}/"
    else
      echo "[CLEAN] Skipped ${path}/ (not present)"
    fi
  done
fi

echo "[DONE] Cleanup complete."
