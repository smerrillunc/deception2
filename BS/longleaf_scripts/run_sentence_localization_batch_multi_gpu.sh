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

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
MODEL_TAG_RAW="${MODEL_NAME//\//_}"
MODEL_TAG_BASE="${MODEL_NAME##*/}"
LABEL_FILTER="${LABEL_FILTER:-truthful_only}"
TRUTHFUL_LIMIT="${TRUTHFUL_LIMIT:-3000}"
validate_label_filter "$LABEL_FILTER"

AUTO_BUILD_DATASET="${AUTO_BUILD_DATASET:-}"
if [[ -z "$AUTO_BUILD_DATASET" ]]; then
  if [[ "$LABEL_FILTER" == "truthful_only" ]]; then
    AUTO_BUILD_DATASET=1
  else
    AUTO_BUILD_DATASET=0
  fi
fi

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ "$LABEL_FILTER" == "truthful_only" ]]; then
    cand_a="$SENTENCE_ROOT/${MODEL_TAG_BASE}_truthful_${TRUTHFUL_LIMIT}"
    cand_b="$SENTENCE_ROOT/${MODEL_TAG_RAW}_truthful_${TRUTHFUL_LIMIT}"
    if [[ -d "$cand_a" ]]; then
      DATA_DIR="$cand_a"
    elif [[ -d "$cand_b" ]]; then
      DATA_DIR="$cand_b"
    elif [[ -d "$SENTENCE_ROOT/$MODEL_TAG_BASE" ]]; then
      DATA_DIR="$cand_a"
    else
      DATA_DIR="$cand_b"
    fi
  else
    cand_a="$SENTENCE_ROOT/$MODEL_TAG_BASE"
    cand_b="$SENTENCE_ROOT/$MODEL_TAG_RAW"
    if [[ -d "$cand_a" ]]; then
      DATA_DIR="$cand_a"
    elif [[ -d "$cand_b" ]]; then
      DATA_DIR="$cand_b"
    else
      DATA_DIR="$cand_a"
    fi
  fi
fi

export MODEL_NAME
export LABEL_FILTER
export TRUTHFUL_LIMIT
export DATA_DIR
export AUTO_BUILD_DATASET
export SKIP_GPU_LIST=1

examples_path="$DATA_DIR/examples.jsonl"
if [[ ! -f "$examples_path" ]]; then
  if [[ "$AUTO_BUILD_DATASET" == "1" ]]; then
    echo "examples.jsonl not found at $examples_path"
    echo "Building sentence dataset once before launching shards..."
    build_limit="${LIMIT:-0}"
    if [[ "$LABEL_FILTER" == "truthful_only" && "$build_limit" == "0" ]]; then
      build_limit="$TRUTHFUL_LIMIT"
    fi
    MODEL_NAME="$MODEL_NAME" OUT_DIR="$DATA_DIR" LABEL_FILTER="$LABEL_FILTER" LIMIT="$build_limit"       "$script_dir/run_sentence_dataset.sh"
  else
    echo "Missing examples file: $examples_path"
    exit 1
  fi
fi

echo "Launching $NUM_SHARDS BS localization shards"
echo "GPUs: ${GPU_IDS_ARR[*]:0:$NUM_SHARDS}"
echo "Model: $MODEL_NAME"
echo "Label filter: $LABEL_FILTER"
echo "DATA_DIR: $DATA_DIR"

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
  echo "One or more BS localization shards failed."
  exit 1
fi

echo "All BS localization shards complete."
