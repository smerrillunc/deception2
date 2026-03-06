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

if [[ -z "${SKIP_GPU_LIST:-}" ]]; then
  echo "Available GPUs:"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  echo ""
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Using GPU: $CUDA_VISIBLE_DEVICES"
  echo ""
else
  if [[ ! -t 0 ]]; then
    export CUDA_VISIBLE_DEVICES="0"
    echo "Using GPU: $CUDA_VISIBLE_DEVICES (non-interactive default)"
    echo ""
  else
    read -r -p "Enter the GPU ID you want to use (e.g., 0): " GPU
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo "Using GPU: $CUDA_VISIBLE_DEVICES"
    echo ""
  fi
fi

GAME="${GAME:-advisor_audit}"
if [[ "$GAME" != "advisor_audit" && "$GAME" != "auto" && "$GAME" != "bs" && "$GAME" != "gridworld" ]]; then
  echo "Invalid GAME=$GAME. Expected one of: advisor_audit, auto, bs, gridworld"
  exit 1
fi

if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "MODEL_NAME is required."
  echo "Example:"
  echo "  MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B $0"
  exit 1
fi

RESULTS_ROOT="/playpen-ssd/smerrill/deception2/AdvisorAudit/Results"
SENTENCE_ROOT="$RESULTS_ROOT/SentencePipeline/v1"
MODEL_TAG_RAW="${MODEL_NAME//\//_}"
MODEL_TAG_BASE="${MODEL_NAME##*/}"
LABEL_FILTER="${LABEL_FILTER:-all}"
DATA_DIR="${DATA_DIR:-}"

N_SAMPLES="${N_SAMPLES:-50}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_P="${TOP_P:-0.5}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
BASE_SEED="${BASE_SEED:-1234}"
METHOD="${METHOD:-adaptive}"
MODE="${MODE:-prefix}"
COARSE_ITERS="${COARSE_ITERS:-8}"
REFINEMENT_ITERS="${REFINEMENT_ITERS:-8}"
MIN_VALID="${MIN_VALID:-3}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
LOG_EVERY="${LOG_EVERY:-25}"
LIMIT="${LIMIT:-0}"
TEXT_FIELD="${TEXT_FIELD:-action_reasoning}"

if [[ -z "$DATA_DIR" ]]; then
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
SENTENCES_PATH="${SENTENCES_PATH:-$DATA_DIR/sentences.jsonl}"
OUT_DIR="${OUT_DIR:-$DATA_DIR/localization}"
JSONL_PATH="${JSONL_PATH:-$DATA_DIR/localization.jsonl}"

SCRIPT="/playpen-ssd/smerrill/deception2/src/sentence_localization_batch.py"

if [[ ! -f "$EXAMPLES_PATH" ]]; then
  echo "Missing examples file: $EXAMPLES_PATH"
  echo "Run dataset build first:"
  echo "  MODEL_NAME=$MODEL_NAME /playpen-ssd/smerrill/deception2/AdvisorAudit/shell_scripts/run_sentence_dataset.sh"
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$SCRIPT"
  --game "$GAME"
  --examples_path "$EXAMPLES_PATH"
  --model_name "$MODEL_NAME"
  --jsonl_path "$JSONL_PATH"
  --n_samples "$N_SAMPLES"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --repetition_penalty "$REPETITION_PENALTY"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --base_seed "$BASE_SEED"
  --method "$METHOD"
  --mode "$MODE"
  --coarse_iters "$COARSE_ITERS"
  --refinement_iters "$REFINEMENT_ITERS"
  --min_valid "$MIN_VALID"
  --label_filter "$LABEL_FILTER"
  --shard_id "$SHARD_ID"
  --num_shards "$NUM_SHARDS"
  --log_every "$LOG_EVERY"
  --text_field "$TEXT_FIELD"
)

if [[ -f "$SENTENCES_PATH" ]]; then
  CMD+=(--sentences_path "$SENTENCES_PATH")
fi
if [[ "$OUT_DIR" != "none" ]]; then
  mkdir -p "$OUT_DIR"
  CMD+=(--out_dir "$OUT_DIR")
fi
if [[ "$LIMIT" != "0" ]]; then
  CMD+=(--limit "$LIMIT")
fi

echo "GAME: $GAME"
echo "MODEL_NAME: $MODEL_NAME"
echo "EXAMPLES_PATH: $EXAMPLES_PATH"
if [[ -f "$SENTENCES_PATH" ]]; then
  echo "SENTENCES_PATH: $SENTENCES_PATH"
else
  echo "SENTENCES_PATH: (not found, using on-the-fly sentence split)"
fi
echo "JSONL_PATH: $JSONL_PATH"
if [[ "$OUT_DIR" != "none" ]]; then
  echo "OUT_DIR: $OUT_DIR"
fi
echo "LABEL_FILTER: $LABEL_FILTER"
echo "LIMIT: $LIMIT"
echo "BASE_SEED: $BASE_SEED"

echo "Running localization..."
"${CMD[@]}"

echo "Batch localization complete."
