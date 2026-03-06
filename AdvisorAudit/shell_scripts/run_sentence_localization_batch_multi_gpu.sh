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

GAME="${GAME:-advisor_audit}"
if [[ "$GAME" != "advisor_audit" && "$GAME" != "auto" && "$GAME" != "bs" && "$GAME" != "gridworld" ]]; then
  echo "Invalid GAME=$GAME. Expected one of: advisor_audit, auto, bs, gridworld"
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-}"
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "MODEL_NAME is required."
  echo "Example:"
  echo "  MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B $0"
  exit 1
fi
export MODEL_NAME
export GAME

RESULTS_ROOT="/playpen-ssd/smerrill/deception2/AdvisorAudit/Results"
SENTENCE_ROOT="$RESULTS_ROOT/SentencePipeline/v1"
MODEL_TAG_RAW="${MODEL_NAME//\//_}"
MODEL_TAG_BASE="${MODEL_NAME##*/}"
LABEL_FILTER="${LABEL_FILTER:-all}"

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
  echo "  MODEL_NAME=$MODEL_NAME OUT_DIR=$DATA_DIR /playpen-ssd/smerrill/deception2/AdvisorAudit/shell_scripts/run_sentence_dataset.sh"
  exit 1
fi

export DATA_DIR
export LABEL_FILTER
export SKIP_GPU_LIST=1
export LIMIT="${LIMIT:-0}"
export N_SAMPLES="${N_SAMPLES:-50}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
export COARSE_ITERS="${COARSE_ITERS:-8}"
export REFINEMENT_ITERS="${REFINEMENT_ITERS:-8}"
export LOG_EVERY="${LOG_EVERY:-25}"

BASE_SEED_START="${BASE_SEED_START:-1234}"

echo "Launching $NUM_SHARDS shards across GPUs: ${GPU_IDS[*]}"
echo "GAME: $GAME"
echo "MODEL_NAME: $MODEL_NAME"
echo "DATA_DIR: ${DATA_DIR:-}"
echo "EXAMPLES_PATH: $EXAMPLES_PATH"
echo "LABEL_FILTER: $LABEL_FILTER"
echo "LIMIT: ${LIMIT:-0}"
echo "BASE_SEED_START: $BASE_SEED_START"

pids=()
for i in "${!GPU_IDS[@]}"; do
  GPU=${GPU_IDS[$i]}
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export SHARD_ID="$i"
    export NUM_SHARDS="$NUM_SHARDS"
    export BASE_SEED="$((BASE_SEED_START + i * 100000))"
    /playpen-ssd/smerrill/deception2/AdvisorAudit/shell_scripts/run_sentence_localization_batch.sh
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
echo "All localization shards complete."
