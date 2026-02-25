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
  echo "✓ Using GPU: $CUDA_VISIBLE_DEVICES"
  echo ""
else
  if [[ ! -t 0 ]]; then
    export CUDA_VISIBLE_DEVICES="0"
    echo "✓ Using GPU: $CUDA_VISIBLE_DEVICES (non-interactive default)"
    echo ""
  else
    read -p "Enter the GPU ID you want to use (e.g., 0): " GPU
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo "✓ Using GPU: $CUDA_VISIBLE_DEVICES"
    echo ""
  fi
fi

if [[ -z "${MODEL_NAME:-}" ]]; then
  if [[ ! -t 0 ]]; then
    # Non-interactive: pick a reasonable default to avoid hanging
    MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  else
    echo "Select a model:"
    echo "  1) deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    echo "  2) deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    echo "  3) deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    echo ""

    read -p "Enter model number (1–3): " MODEL_CHOICE

    case "$MODEL_CHOICE" in
        1) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ;;
        2) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" ;;
        3) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" ;;
        *) echo "❌ Invalid model selection: $MODEL_CHOICE"; exit 1 ;;
    esac
  fi
fi

GAME="bs"
RESULTS_ROOT="/playpen-ssd/smerrill/deception2/BS/Results"
SENTENCE_ROOT="$RESULTS_ROOT/SentencePipeline/v1"
MODEL_TAG_RAW="${MODEL_NAME//\//_}"
MODEL_TAG_BASE="${MODEL_NAME##*/}"
LABEL_FILTER="${LABEL_FILTER:-deceptive_only}"
ONLY_DECEPTIVE="${ONLY_DECEPTIVE:-0}"
ONLY_TRUTHFUL="${ONLY_TRUTHFUL:-0}"
TRUTHFUL_LIMIT="${TRUTHFUL_LIMIT:-3000}"
DATA_DIR="${DATA_DIR:-}"
AUTO_BUILD_DATASET="${AUTO_BUILD_DATASET:-0}"

N_SAMPLES="${N_SAMPLES:-50}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_P="${TOP_P:-0.5}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
METHOD="${METHOD:-adaptive}"
MODE="${MODE:-prefix}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
LOG_EVERY="${LOG_EVERY:-25}"

if [[ "$ONLY_DECEPTIVE" == "1" && "$ONLY_TRUTHFUL" == "1" ]]; then
  echo "Cannot set both ONLY_DECEPTIVE=1 and ONLY_TRUTHFUL=1"
  exit 1
fi
if [[ "$ONLY_DECEPTIVE" == "1" ]]; then
  LABEL_FILTER="deceptive_only"
fi
if [[ "$ONLY_TRUTHFUL" == "1" ]]; then
  LABEL_FILTER="truthful_only"
fi
if [[ "$LABEL_FILTER" != "all" && "$LABEL_FILTER" != "deceptive_only" && "$LABEL_FILTER" != "truthful_only" ]]; then
  echo "Invalid LABEL_FILTER=$LABEL_FILTER. Expected one of: all, deceptive_only, truthful_only"
  exit 1
fi

if [[ -z "$DATA_DIR" ]]; then
  if [[ "$LABEL_FILTER" == "truthful_only" ]]; then
    CAND_A="$SENTENCE_ROOT/${MODEL_TAG_BASE}_truthful_${TRUTHFUL_LIMIT}"
    CAND_B="$SENTENCE_ROOT/${MODEL_TAG_RAW}_truthful_${TRUTHFUL_LIMIT}"
    if [[ -d "$CAND_A" ]]; then
      DATA_DIR="$CAND_A"
    elif [[ -d "$CAND_B" ]]; then
      DATA_DIR="$CAND_B"
    elif [[ -d "$SENTENCE_ROOT/$MODEL_TAG_BASE" ]]; then
      DATA_DIR="$CAND_A"
    else
      DATA_DIR="$CAND_B"
    fi
  else
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
fi

EXAMPLES_PATH="${EXAMPLES_PATH:-$DATA_DIR/examples.jsonl}"
SENTENCES_PATH="${SENTENCES_PATH:-$DATA_DIR/sentences.jsonl}"
OUT_DIR="${OUT_DIR:-$DATA_DIR/localization}"
JSONL_PATH="${JSONL_PATH:-$DATA_DIR/localization.jsonl}"

SCRIPT="/playpen-ssd/smerrill/deception2/src/sentence_localization_batch.py"

if [[ ! -f "$EXAMPLES_PATH" ]]; then
  if [[ "$AUTO_BUILD_DATASET" == "1" ]]; then
    echo "examples.jsonl not found. Auto-building sentence dataset first..."
    BUILD_LIMIT="${LIMIT:-0}"
    if [[ "$LABEL_FILTER" == "truthful_only" && "$BUILD_LIMIT" == "0" ]]; then
      BUILD_LIMIT="$TRUTHFUL_LIMIT"
    fi
    MODEL_NAME="$MODEL_NAME" OUT_DIR="$DATA_DIR" LABEL_FILTER="$LABEL_FILTER" LIMIT="$BUILD_LIMIT" \
      /playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_dataset.sh
  else
    echo "Missing examples file: $EXAMPLES_PATH"
    echo "Run dataset build first:"
    echo "  MODEL_NAME=$MODEL_NAME /playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_dataset.sh"
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
  --label_filter "$LABEL_FILTER"
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

echo "Running localization..."
"${CMD[@]}"

echo "Batch localization complete."
