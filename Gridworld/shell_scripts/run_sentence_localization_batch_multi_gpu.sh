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
NUM_SHARDS=${NUM_SHARDS:-${#GPU_IDS[@]}}

if [[ ${#GPU_IDS[@]} -lt $NUM_SHARDS ]]; then
  echo "GPU_IDS count (${#GPU_IDS[@]}) must be >= NUM_SHARDS ($NUM_SHARDS)."
  exit 1
fi

GAME="${GAME:-gridworld}"
if [[ "$GAME" != "gridworld" && "$GAME" != "bs" && "$GAME" != "auto" ]]; then
  echo "Invalid GAME=$GAME. Expected one of: gridworld, bs, auto"
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-}"
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "MODEL_NAME is required."
  echo "Example:"
  echo "  MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B $0"
  exit 1
fi
MODEL_TAG="${MODEL_NAME//\//_}"
LABEL_FILTER="all"

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ "$GAME" == "bs" ]]; then
    DATA_DIR="/playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1/${MODEL_TAG}"
  else
    DATA_DIR="/playpen-ssd/smerrill/deception2/Gridworld/Results/SentencePipeline/v1/${MODEL_TAG}"
  fi
fi
export DATA_DIR

EXAMPLES_PATH="${EXAMPLES_PATH:-$DATA_DIR/examples.jsonl}"

if [[ ! -f "$EXAMPLES_PATH" ]]; then
  echo "Missing examples file: $EXAMPLES_PATH"
  echo "Run dataset build first:"
  echo "  GAME=$GAME MODEL_NAME=$MODEL_NAME OUT_DIR=$DATA_DIR /playpen-ssd/smerrill/deception2/Gridworld/shell_scripts/run_sentence_dataset.sh"
  exit 1
fi

echo "Launching $NUM_SHARDS localization shards across GPUs: ${GPU_IDS[*]:0:$NUM_SHARDS}"
echo "GAME: $GAME"
echo "MODEL_NAME: $MODEL_NAME"
echo "DATA_DIR: $DATA_DIR"
echo "LABEL_FILTER: $LABEL_FILTER"

export MODEL_NAME
export GAME
export DATA_DIR
export LABEL_FILTER
export SKIP_GPU_LIST=1

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU=${GPU_IDS[$i]}
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export SHARD_ID="$i"
    export NUM_SHARDS="$NUM_SHARDS"
    /playpen-ssd/smerrill/deception2/Gridworld/shell_scripts/run_sentence_localization_batch.sh
  ) &
done

wait
echo "All sentence localization shards complete."
