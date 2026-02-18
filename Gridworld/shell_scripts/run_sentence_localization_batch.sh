#!/usr/bin/env bash
set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

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

if [[ -z "${MODEL_NAME:-}" ]]; then
  if [[ ! -t 0 ]]; then
    MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  else
    echo "Select a model:"
    echo "  1) deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    echo "  2) deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    echo "  3) deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    echo ""
    read -r -p "Enter model number (1-3): " MODEL_CHOICE
    case "$MODEL_CHOICE" in
      1) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ;;
      2) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" ;;
      3) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" ;;
      *) echo "Invalid model selection: $MODEL_CHOICE"; exit 1 ;;
    esac
  fi
fi

GAME="${GAME:-gridworld}"
if [[ "$GAME" != "gridworld" && "$GAME" != "bs" && "$GAME" != "auto" ]]; then
  echo "Invalid GAME=$GAME. Expected one of: gridworld, bs, auto"
  exit 1
fi

MODEL_TAG="${MODEL_NAME//\//_}"

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ "$GAME" == "bs" ]]; then
    DATA_DIR="/playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1/${MODEL_TAG}"
  else
    DATA_DIR="/playpen-ssd/smerrill/deception2/Gridworld/Results/SentencePipeline/v1/${MODEL_TAG}"
  fi
fi

EXAMPLES_PATH="${EXAMPLES_PATH:-$DATA_DIR/examples.jsonl}"
SENTENCES_PATH="${SENTENCES_PATH:-$DATA_DIR/sentences.jsonl}"
OUT_DIR="${OUT_DIR:-$DATA_DIR/localization}"
JSONL_PATH="${JSONL_PATH:-$DATA_DIR/localization.jsonl}"
AUTO_BUILD_DATASET="${AUTO_BUILD_DATASET:-0}"

N_SAMPLES="${N_SAMPLES:-50}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_P="${TOP_P:-0.5}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
METHOD="${METHOD:-adaptive}"
MODE="${MODE:-prefix}"
ONLY_DECEPTIVE="${ONLY_DECEPTIVE:-1}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
LOG_EVERY="${LOG_EVERY:-25}"

SCRIPT="/playpen-ssd/smerrill/deception2/src/sentence_localization_batch.py"

if [[ ! -f "$EXAMPLES_PATH" ]]; then
  if [[ "$AUTO_BUILD_DATASET" == "1" ]]; then
    echo "examples.jsonl not found. Auto-building sentence dataset first..."
    GAME="$GAME" MODEL_NAME="$MODEL_NAME" DATA_DIR="$DATA_DIR" \
      /playpen-ssd/smerrill/deception2/Gridworld/shell_scripts/run_sentence_dataset.sh
  else
    echo "Missing examples file: $EXAMPLES_PATH"
    echo "Run dataset build first:"
    echo "  GAME=$GAME MODEL_NAME=$MODEL_NAME /playpen-ssd/smerrill/deception2/Gridworld/shell_scripts/run_sentence_dataset.sh"
    echo "Or rerun with AUTO_BUILD_DATASET=1"
    exit 1
  fi
fi

CMD=(
  python "$SCRIPT"
  --game "$GAME"
  --examples_path "$EXAMPLES_PATH"
  --model_name "$MODEL_NAME"
  --jsonl_path "$JSONL_PATH"
  --n_samples "$N_SAMPLES"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --repetition_penalty "$REPETITION_PENALTY"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --method "$METHOD"
  --mode "$MODE"
  --shard_id "$SHARD_ID"
  --num_shards "$NUM_SHARDS"
  --log_every "$LOG_EVERY"
)

if [[ -f "$SENTENCES_PATH" ]]; then
  CMD+=(--sentences_path "$SENTENCES_PATH")
fi

if [[ "$OUT_DIR" != "none" ]]; then
  mkdir -p "$OUT_DIR"
  CMD+=(--out_dir "$OUT_DIR")
fi

if [[ "$ONLY_DECEPTIVE" == "1" ]]; then
  CMD+=(--only_deceptive)
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

echo "Running localization..."
"${CMD[@]}"

echo "Batch localization complete."
