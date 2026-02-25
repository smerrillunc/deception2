#!/usr/bin/env bash
set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

GPU_IDS=(${GPU_IDS:-2 3 4 5 6 7})
NUM_SHARDS=${NUM_SHARDS:-6}

if [[ ${#GPU_IDS[@]} -ne $NUM_SHARDS ]]; then
  echo "GPU_IDS count (${#GPU_IDS[@]}) must equal NUM_SHARDS ($NUM_SHARDS)."
  exit 1
fi

echo "Launching $NUM_SHARDS shards across GPUs: ${GPU_IDS[*]}"
export MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
RESULTS_ROOT="/playpen-ssd/smerrill/deception2/BS/Results"
SENTENCE_ROOT="$RESULTS_ROOT/SentencePipeline/v1"
MODEL_TAG_RAW="${MODEL_NAME//\//_}"
MODEL_TAG_BASE="${MODEL_NAME##*/}"
LABEL_FILTER="${LABEL_FILTER:-truthful_only}"
TRUTHFUL_LIMIT="${TRUTHFUL_LIMIT:-3000}"
if [[ "$LABEL_FILTER" != "all" && "$LABEL_FILTER" != "deceptive_only" && "$LABEL_FILTER" != "truthful_only" ]]; then
  echo "Invalid LABEL_FILTER=$LABEL_FILTER. Expected one of: all, deceptive_only, truthful_only"
  exit 1
fi
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
    CAND_A="$SENTENCE_ROOT/${MODEL_TAG_BASE}_truthful_${TRUTHFUL_LIMIT}"
    CAND_B="$SENTENCE_ROOT/${MODEL_TAG_RAW}_truthful_${TRUTHFUL_LIMIT}"
    if [[ -d "$CAND_A" ]]; then
      export DATA_DIR="$CAND_A"
    elif [[ -d "$CAND_B" ]]; then
      export DATA_DIR="$CAND_B"
    elif [[ -d "$SENTENCE_ROOT/$MODEL_TAG_BASE" ]]; then
      export DATA_DIR="$CAND_A"
    else
      export DATA_DIR="$CAND_B"
    fi
  else
    CAND_A="$SENTENCE_ROOT/${MODEL_TAG_BASE}"
    CAND_B="$SENTENCE_ROOT/${MODEL_TAG_RAW}"
    if [[ -d "$CAND_A" ]]; then
      export DATA_DIR="$CAND_A"
    elif [[ -d "$CAND_B" ]]; then
      export DATA_DIR="$CAND_B"
    else
      export DATA_DIR="$CAND_A"
    fi
  fi
fi
export LABEL_FILTER
export TRUTHFUL_LIMIT
export AUTO_BUILD_DATASET
export SKIP_GPU_LIST=1

echo "Label filter: $LABEL_FILTER"
echo "Model: $MODEL_NAME"
echo "DATA_DIR: ${DATA_DIR:-}"
echo "AUTO_BUILD_DATASET: $AUTO_BUILD_DATASET"

EXAMPLES_PATH="$DATA_DIR/examples.jsonl"
if [[ ! -f "$EXAMPLES_PATH" ]]; then
  if [[ "$AUTO_BUILD_DATASET" == "1" ]]; then
    echo "examples.jsonl not found at $EXAMPLES_PATH"
    echo "Building sentence dataset once before launching shards..."
    BUILD_LIMIT="${LIMIT:-0}"
    if [[ "$LABEL_FILTER" == "truthful_only" && "$BUILD_LIMIT" == "0" ]]; then
      BUILD_LIMIT="$TRUTHFUL_LIMIT"
    fi
    MODEL_NAME="$MODEL_NAME" OUT_DIR="$DATA_DIR" LABEL_FILTER="$LABEL_FILTER" LIMIT="$BUILD_LIMIT" \
      /playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_dataset.sh
  else
    echo "Missing examples file: $EXAMPLES_PATH"
    echo "Set AUTO_BUILD_DATASET=1 or build dataset first."
    exit 1
  fi
fi

pids=()
for i in "${!GPU_IDS[@]}"; do
  GPU=${GPU_IDS[$i]}
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export SHARD_ID="$i"
    export NUM_SHARDS="$NUM_SHARDS"
    /playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_localization_batch.sh
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
  echo "One or more localization shards failed."
  exit 1
fi
echo "✓ All shards complete."
