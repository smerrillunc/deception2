#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/longleaf_common.sh"
activate_deception_env

select_single_gpu
select_model_name

GAME="bs"
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
validate_label_filter "$LABEL_FILTER"

if [[ -z "$DATA_DIR" ]]; then
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

EXAMPLES_PATH="${EXAMPLES_PATH:-$DATA_DIR/examples.jsonl}"
SENTENCES_PATH="${SENTENCES_PATH:-$DATA_DIR/sentences.jsonl}"
OUT_DIR="${OUT_DIR:-$DATA_DIR/localization}"
JSONL_PATH="${JSONL_PATH:-$DATA_DIR/localization.jsonl}"

SCRIPT_PATH="$SRC_ROOT/sentence_localization_batch.py"

if [[ ! -f "$EXAMPLES_PATH" ]]; then
  if [[ "$AUTO_BUILD_DATASET" == "1" ]]; then
    echo "examples.jsonl missing. Building sentence dataset first..."
    build_limit="${LIMIT:-0}"
    if [[ "$LABEL_FILTER" == "truthful_only" && "$build_limit" == "0" ]]; then
      build_limit="$TRUTHFUL_LIMIT"
    fi
    MODEL_NAME="$MODEL_NAME" OUT_DIR="$DATA_DIR" LABEL_FILTER="$LABEL_FILTER" LIMIT="$build_limit"       "$script_dir/run_sentence_dataset.sh"
  else
    echo "Missing examples file: $EXAMPLES_PATH"
    echo "Run dataset build first (or AUTO_BUILD_DATASET=1)."
    exit 1
  fi
fi

CMD=(
  "$PYTHON_BIN" "$SCRIPT_PATH"
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
echo "LABEL_FILTER: $LABEL_FILTER"
echo "EXAMPLES_PATH: $EXAMPLES_PATH"
echo "JSONL_PATH: $JSONL_PATH"
"${CMD[@]}"

echo "Batch localization complete: $JSONL_PATH"
