#!/usr/bin/env bash
set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception
hash -r

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: conda env not active after 'conda activate deception'."
  exit 1
fi
PYTHON_BIN="$CONDA_PREFIX/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: expected python not found at $PYTHON_BIN"
  exit 1
fi
echo "Python in env: $PYTHON_BIN"

GPU_IDS=(${GPU_IDS:-2 3 4 5 6 7})
NUM_SHARDS=${NUM_SHARDS:-6}

if [[ ${#GPU_IDS[@]} -ne $NUM_SHARDS ]]; then
  echo "GPU_IDS count (${#GPU_IDS[@]}) must equal NUM_SHARDS ($NUM_SHARDS)."
  exit 1
fi
MODEL_NAME="mistralai/Ministral-3-8B-Reasoning-2512"
echo "Launching $NUM_SHARDS shards across GPUs: ${GPU_IDS[*]}"
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "MODEL_NAME is required."
  echo "Example:"
  echo "  MODEL_NAME=mistralai/Ministral-3-8B-Reasoning-2512 $0"
  exit 1
fi
export MODEL_NAME
RESULTS_ROOT="/playpen-ssd/smerrill/deception2/BS/Results"
SENTENCE_ROOT="$RESULTS_ROOT/SentencePipeline/v1"
MODEL_TAG_RAW="${MODEL_NAME//\//_}"
MODEL_TAG_BASE="${MODEL_NAME##*/}"
LABEL_FILTER="all"

if [[ -z "${DATA_DIR:-}" ]]; then
  CAND_A="$SENTENCE_ROOT/${MODEL_TAG_BASE}"
  CAND_B="$SENTENCE_ROOT/${MODEL_TAG_RAW}"
  if [[ -d "$CAND_A" ]]; then
    DATA_DIR="$CAND_A"
  elif [[ -d "$CAND_B" ]]; then
    DATA_DIR="$CAND_B"
  else
    DATA_DIR="$CAND_A"
  fi
fi

EXAMPLES_PATH="${EXAMPLES_PATH:-$DATA_DIR/examples.jsonl}"
if [[ ! -f "$EXAMPLES_PATH" ]]; then
  echo "Missing examples file: $EXAMPLES_PATH"
  echo "Run dataset build first:"
  echo "  MODEL_NAME=$MODEL_NAME OUT_DIR=$DATA_DIR /playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_dataset.sh"
  exit 1
fi

export DATA_DIR
export LABEL_FILTER
export SKIP_GPU_LIST=1

echo "Label filter: $LABEL_FILTER"
echo "Model: $MODEL_NAME"
echo "DATA_DIR: ${DATA_DIR:-}"
echo "EXAMPLES_PATH: $EXAMPLES_PATH"


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
