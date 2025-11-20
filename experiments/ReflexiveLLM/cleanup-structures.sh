#!/usr/bin/env bash
# Cleanup helper for SentientLLM artifacts.
# Removes on-disk structures created by runs (memory store, indices, policies, visuals).

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve paths with env fallbacks
mem_store_dir="${MEM_STORE_DIR:-memory_store}"
assembly_index_path="${ASSEMBLY_INDEX_PATH:-assembly_index.json}"
self_model_path="${SELF_MODEL_PATH:-self_model.json}"
policy_path="${POLICY_PATH:-meta_policy.pt}"
viz_dir="${VISUALIZATION_DIR:-visualizations}"
model_weights_path="${MODEL_WEIGHTS_PATH:-model_weights.json}"

# Helper to safely remove a path relative to the repo root
remove_path() {
  local target="$1"
  [ -z "$target" ] && return 0
  # Only allow removal inside the repo root
  if [[ "$target" != /* ]]; then
    target="$root_dir/$target"
  fi
  if [[ "$target" == "/" || "$target" == "$root_dir" ]]; then
    echo "Skipping unsafe path: $target"
    return 0
  fi
  if [ -e "$target" ]; then
    echo "Removing $target"
    rm -rf -- "$target"
  fi
}

remove_path "$mem_store_dir"
remove_path "$assembly_index_path"
remove_path "$self_model_path"
remove_path "$policy_path"
remove_path "$viz_dir"
remove_path "$model_weights_path"

echo "Cleanup complete."
