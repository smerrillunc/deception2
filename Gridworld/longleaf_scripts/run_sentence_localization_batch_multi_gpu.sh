#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/longleaf_common.sh"
activate_deception_env

GAME="${GAME:-gridworld}"
if [[ "$GAME" != "gridworld" && "$GAME" != "bs" && "$GAME" != "auto" ]]; then
  echo "Invalid GAME=$GAME. Expected one of: gridworld, bs, auto"
  exit 1
fi

GPU_IDS_RAW="${GPU_IDS:-0 1 2 3 4}"
GPU_IDS_RAW="${GPU_IDS_RAW//,/ }"
read -r -a GPU_IDS_ARR <<< "$GPU_IDS_RAW"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_IDS_ARR[@]}}"
if [[ ${#GPU_IDS_ARR[@]} -lt "$NUM_SHARDS" ]]; then
  echo "GPU_IDS count (${#GPU_IDS_ARR[@]}) must be >= NUM_SHARDS ($NUM_SHARDS)."
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
MODEL_TAG_RAW="${MODEL_NAME//\//_}"
MODEL_TAG_BASE="${MODEL_NAME##*/}"
LABEL_FILTER="${LABEL_FILTER:-deceptive_only}"
validate_label_filter "$LABEL_FILTER"
AUTO_BUILD_DATASET="${AUTO_BUILD_DATASET:-0}"

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ "$GAME" == "bs" ]]; then
    bs_sentence_root="${BS_SENTENCE_ROOT:-$PROJECT_ROOT/BS/Results/SentencePipeline/v1}"
    cand_a="$bs_sentence_root/$MODEL_TAG_BASE"
    cand_b="$bs_sentence_root/$MODEL_TAG_RAW"
  else
    cand_a="$SENTENCE_ROOT/$MODEL_TAG_BASE"
    cand_b="$SENTENCE_ROOT/$MODEL_TAG_RAW"
  fi

  if [[ -d "$cand_a" ]]; then
    DATA_DIR="$cand_a"
  elif [[ -d "$cand_b" ]]; then
    DATA_DIR="$cand_b"
  else
    DATA_DIR="$cand_a"
  fi
fi

examples_path="$DATA_DIR/examples.jsonl"
if [[ ! -f "$examples_path" ]]; then
  if [[ "$AUTO_BUILD_DATASET" == "1" ]]; then
    echo "examples.jsonl missing. Building sentence dataset before launching shards..."
    GAME="$GAME" MODEL_NAME="$MODEL_NAME" OUT_DIR="$DATA_DIR" LABEL_FILTER="$LABEL_FILTER"       "$script_dir/run_sentence_dataset.sh"
  else
    echo "Missing examples file: $examples_path"
    exit 1
  fi
fi

export MODEL_NAME
export GAME
export DATA_DIR
export LABEL_FILTER
export SKIP_GPU_LIST=1

echo "Launching $NUM_SHARDS localization shards"
echo "GPUs: ${GPU_IDS_ARR[*]:0:$NUM_SHARDS}"
echo "GAME: $GAME"
echo "MODEL_NAME: $MODEL_NAME"

declare -a pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_IDS_ARR[$i]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SHARD_ID="$i"
    export NUM_SHARDS="$NUM_SHARDS"
    "$script_dir/run_sentence_localization_batch.sh"
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
  echo "One or more Gridworld localization shards failed."
  exit 1
fi

echo "All Gridworld localization shards complete."
