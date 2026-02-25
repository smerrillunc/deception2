#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/longleaf_common.sh"
activate_deception_env

GPU_IDS_RAW="${GPU_IDS:-0 1 2 3 4 5}"
GPU_IDS_RAW="${GPU_IDS_RAW//,/ }"
read -r -a GPU_IDS_ARR <<< "$GPU_IDS_RAW"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_IDS_ARR[@]}}"

if [[ ${#GPU_IDS_ARR[@]} -lt "$NUM_SHARDS" ]]; then
  echo "GPU_IDS count (${#GPU_IDS_ARR[@]}) must be >= NUM_SHARDS ($NUM_SHARDS)."
  exit 1
fi

select_model_name

REASONING_FLAG="${REASONING_FLAG:---is_reasoning_model}"
SEED_BASE="${SEED_BASE:-0}"
MAX_GAMES="${MAX_GAMES:-1000}"
MAX_TURNS="${MAX_TURNS:-1000}"
LABEL_FILTER="${LABEL_FILTER:-deceptive_only}"
validate_label_filter "$LABEL_FILTER"
set_target_counts_from_label_filter "$LABEL_FILTER" "1000"

MODEL_TAG="${MODEL_NAME//\//_}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
OUT_BASE="${OUT_BASE:-$MINING_ROOT/$MODEL_TAG/$DATE_TAG}"
SCRIPT_PATH="$SRC_ROOT/deception_miner.py"
mkdir -p "$OUT_BASE"

echo "Launching $NUM_SHARDS BS miner shards"
echo "GPUs: ${GPU_IDS_ARR[*]:0:$NUM_SHARDS}"
echo "Model: $MODEL_NAME"
echo "Label filter: $LABEL_FILTER"

declare -a pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_IDS_ARR[$i]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    out_dir="$OUT_BASE/gpu_${gpu}"
    mkdir -p "$out_dir"
    seed=$((SEED_BASE + i * 10000))

    cmd=(
      "$PYTHON_BIN" "$SCRIPT_PATH"
      --game bs
      --model_name "$MODEL_NAME"
      --output_dir "$out_dir"
      --seed "$seed"
      --max_games "$MAX_GAMES"
      --max_turns "$MAX_TURNS"
      --label_filter "$LABEL_FILTER"
      --target_deceptive "$TARGET_DECEPTIVE"
      --target_truthful "$TARGET_TRUTHFUL"
      --log_every "${LOG_EVERY:-25}"
    )
    if [[ -n "$REASONING_FLAG" ]]; then
      cmd+=("$REASONING_FLAG")
    fi

    "${cmd[@]}" > "$out_dir/run.log" 2>&1
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "One or more BS miner shards failed."
  exit 1
fi

echo "All BS miner shards complete: $OUT_BASE"
