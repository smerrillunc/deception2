#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/longleaf_common.sh"
activate_deception_env

GAME="${GAME:-gridworld}"
if [[ "$GAME" != "gridworld" && "$GAME" != "bs" && "$GAME" != "auto" ]]; then
  echo "Invalid GAME=$GAME. Expected one of: gridworld, bs, auto"
  exit 1
fi

select_model_name
MODEL_TAG="$(resolve_model_tag "$MODEL_NAME" "$MINING_ROOT")"

if [[ -z "${INPUT_ROOT:-}" ]]; then
  if [[ "$GAME" == "bs" ]]; then
    INPUT_ROOT="${BS_MINING_ROOT:-$PROJECT_ROOT/BS/Results/DeceptionMining}/$MODEL_TAG"
  else
    INPUT_ROOT="$MINING_ROOT/$MODEL_TAG"
  fi
fi

if [[ -n "${OUT_DIR+x}" ]]; then
  OUT_DIR="${OUT_DIR:-$SENTENCE_ROOT/$MODEL_TAG}"
elif [[ -n "${DATA_DIR:-}" ]]; then
  OUT_DIR="$DATA_DIR"
else
  if [[ "$GAME" == "bs" ]]; then
    OUT_DIR="${BS_SENTENCE_ROOT:-$PROJECT_ROOT/BS/Results/SentencePipeline/v1}/$MODEL_TAG"
  else
    OUT_DIR="$SENTENCE_ROOT/$MODEL_TAG"
  fi
fi

TEXT_FIELD="${TEXT_FIELD:-action_reasoning}"
FALLBACK_TEXT_FIELD="${FALLBACK_TEXT_FIELD:-action_raw_text}"
LABEL_FILTER="${LABEL_FILTER:-deceptive_only}"
ONLY_DECEPTIVE="${ONLY_DECEPTIVE:-0}"
ONLY_TRUTHFUL="${ONLY_TRUTHFUL:-0}"
INCLUDE_MESSAGES="${INCLUDE_MESSAGES:-0}"
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
validate_label_filter "$LABEL_FILTER"

mkdir -p "$OUT_DIR"
SCRIPT_PATH="$SRC_ROOT/build_sentence_dataset.py"
CMD=(
  "$PYTHON_BIN" "$SCRIPT_PATH"
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

"${CMD[@]}"

echo "Sentence dataset complete: $OUT_DIR"
