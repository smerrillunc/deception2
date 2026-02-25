#!/usr/bin/env bash
set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

GAME="bs"

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

RESULTS_ROOT="/playpen-ssd/smerrill/deception2/BS/Results"
MINING_ROOT="$RESULTS_ROOT/DeceptionMining"
SENTENCE_ROOT="$RESULTS_ROOT/SentencePipeline/v1"
MODEL_TAG_RAW="${MODEL_NAME//\//_}"
MODEL_TAG_BASE="${MODEL_NAME##*/}"
MODEL_TAG="$MODEL_TAG_BASE"

if [[ -d "$MINING_ROOT/$MODEL_TAG_RAW" && ! -d "$MINING_ROOT/$MODEL_TAG_BASE" ]]; then
  MODEL_TAG="$MODEL_TAG_RAW"
elif [[ -d "$SENTENCE_ROOT/$MODEL_TAG_RAW" && ! -d "$SENTENCE_ROOT/$MODEL_TAG_BASE" ]]; then
  MODEL_TAG="$MODEL_TAG_RAW"
fi

INPUT_ROOT="${INPUT_ROOT:-$MINING_ROOT/${MODEL_TAG}}"
if [[ -n "${OUT_DIR+x}" ]]; then
  OUT_DIR="${OUT_DIR:-$SENTENCE_ROOT/${MODEL_TAG}}"
elif [[ -n "${DATA_DIR:-}" ]]; then
  # Back-compat: allow callers that pass DATA_DIR instead of OUT_DIR.
  OUT_DIR="$DATA_DIR"
else
  OUT_DIR="$SENTENCE_ROOT/${MODEL_TAG}"
fi
TEXT_FIELD="${TEXT_FIELD:-action_reasoning}"
FALLBACK_TEXT_FIELD="${FALLBACK_TEXT_FIELD:-action_raw_text}"
LABEL_FILTER="${LABEL_FILTER:-deceptive_only}"
ONLY_DECEPTIVE="${ONLY_DECEPTIVE:-0}"
ONLY_TRUTHFUL="${ONLY_TRUTHFUL:-0}"
INCLUDE_MESSAGES="${INCLUDE_MESSAGES:-1}"
LIMIT="${LIMIT:-0}"

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

mkdir -p "$OUT_DIR"

SCRIPT="/playpen-ssd/smerrill/deception2/src/build_sentence_dataset.py"
CMD=(
  python "$SCRIPT"
  --input_root "$INPUT_ROOT"
  --out_dir "$OUT_DIR"
  --text_field "$TEXT_FIELD"
  --fallback_text_field "$FALLBACK_TEXT_FIELD"
  --label_filter "$LABEL_FILTER"
)

if [[ "$INCLUDE_MESSAGES" == "1" ]]; then
  CMD+=(--include_messages)
fi
if [[ "$LIMIT" != "0" ]]; then
  CMD+=(--limit "$LIMIT")
fi

echo "GAME: $GAME"
echo "MODEL_NAME: $MODEL_NAME"
echo "INPUT_ROOT: $INPUT_ROOT"
echo "OUT_DIR: $OUT_DIR"
echo "TEXT_FIELD: $TEXT_FIELD"
echo "FALLBACK_TEXT_FIELD: $FALLBACK_TEXT_FIELD"
echo "LABEL_FILTER: $LABEL_FILTER"

echo "Building sentence dataset..."
"${CMD[@]}"

echo "Sentence dataset complete."
