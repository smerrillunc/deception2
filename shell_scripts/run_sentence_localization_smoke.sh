#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/playpen-ssd/smerrill/conda_envs/deception/bin/python}"
REPO_ROOT="/playpen-ssd/smerrill/deception2"

MINING_DIR=""
ENVIRONMENT=""
MODEL_NAME=""
GPU_ID="${CUDA_VISIBLE_DEVICES:-}"
PIPELINE_DIR=""
TEXT_FIELD="${TEXT_FIELD:-action_reasoning}"
FALLBACK_TEXT_FIELD="${FALLBACK_TEXT_FIELD:-action_raw_text}"
LABEL_FILTER="${LABEL_FILTER:-all}"
BUILD_LIMIT="${BUILD_LIMIT:-0}"
LOCALIZE_LIMIT="${LOCALIZE_LIMIT:-0}"
WRITE_JSONL="${WRITE_JSONL:-0}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BASE_SEED="${BASE_SEED:-1234}"
METHOD="${METHOD:-adaptive}"
MODE="${MODE:-prefix}"
COARSE_ITERS="${COARSE_ITERS:-6}"
REFINEMENT_ITERS="${REFINEMENT_ITERS:-6}"
MIN_VALID="${MIN_VALID:-2}"
EXTRA_BUILD_ARGS=()
EXTRA_LOCALIZE_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_sentence_localization_smoke.sh --mining_dir DIR [options] [-- build/localize extra args]

Required:
  --mining_dir DIR              Directory containing deception_samples.jsonl and ideally meta.json

Optional:
  --env ENV                     One of: bs, gridworld, advisor_audit. If omitted, infer from meta.json.
  --model_name MODEL            If omitted, infer from meta.json.
  --gpu GPU                     Optional single GPU id; otherwise uses existing CUDA_VISIBLE_DEVICES
  --pipeline_dir DIR            Output root for examples/sentences/localization
  --text_field FIELD            Default: action_reasoning
  --fallback_text_field FIELD   Default: action_raw_text
  --build_limit N               Optional build-time example limit
  --localize_limit N            Optional localization example limit
  --write_jsonl                 Also write localization.jsonl (default: off)
  --n_samples N                 Default: 8
  --max_new_tokens N            Default: 4096
  --help                        Show this message

Anything after `--` is forwarded to the localization command.
EOF
}

meta_field() {
  local meta_path="$1"
  local field_name="$2"
  "$PYTHON_BIN" - "$meta_path" "$field_name" <<'PY'
import json, sys
from pathlib import Path

meta_path = Path(sys.argv[1])
field_name = sys.argv[2]
if not meta_path.exists():
    raise SystemExit(0)
try:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
value = data.get(field_name)
if value is None:
    raise SystemExit(0)
print(value)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mining_dir)
      MINING_DIR="$2"
      shift 2
      ;;
    --env)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --pipeline_dir)
      PIPELINE_DIR="$2"
      shift 2
      ;;
    --text_field)
      TEXT_FIELD="$2"
      shift 2
      ;;
    --fallback_text_field)
      FALLBACK_TEXT_FIELD="$2"
      shift 2
      ;;
    --label_filter)
      LABEL_FILTER="$2"
      shift 2
      ;;
    --build_limit)
      BUILD_LIMIT="$2"
      shift 2
      ;;
    --localize_limit)
      LOCALIZE_LIMIT="$2"
      shift 2
      ;;
    --write_jsonl)
      WRITE_JSONL="1"
      shift
      ;;
    --n_samples)
      N_SAMPLES="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top_p)
      TOP_P="$2"
      shift 2
      ;;
    --repetition_penalty)
      REPETITION_PENALTY="$2"
      shift 2
      ;;
    --max_new_tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --base_seed)
      BASE_SEED="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --coarse_iters)
      COARSE_ITERS="$2"
      shift 2
      ;;
    --refinement_iters)
      REFINEMENT_ITERS="$2"
      shift 2
      ;;
    --min_valid)
      MIN_VALID="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_LOCALIZE_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MINING_DIR" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

MINING_DIR="$(realpath "$MINING_DIR")"
META_PATH="$MINING_DIR/meta.json"

if [[ -z "$ENVIRONMENT" && -f "$META_PATH" ]]; then
  ENVIRONMENT="$(meta_field "$META_PATH" "game" || true)"
fi
if [[ -z "$MODEL_NAME" && -f "$META_PATH" ]]; then
  MODEL_NAME="$(meta_field "$META_PATH" "model_name" || true)"
fi

if [[ -z "$ENVIRONMENT" || -z "$MODEL_NAME" ]]; then
  echo "Could not infer env/model_name from $META_PATH. Pass --env and --model_name explicitly." >&2
  exit 1
fi

case "$ENVIRONMENT" in
  bs|gridworld|advisor_audit)
    ;;
  *)
    echo "Unsupported env: $ENVIRONMENT" >&2
    exit 1
    ;;
esac

MODEL_TAG="${MODEL_NAME//\//_}"
RUN_TAG="$(basename "$MINING_DIR")"
if [[ -z "$PIPELINE_DIR" ]]; then
  PIPELINE_DIR="$REPO_ROOT/testing/localization_smoke/$ENVIRONMENT/$MODEL_TAG/$RUN_TAG"
fi
mkdir -p "$PIPELINE_DIR"

EXAMPLES_PATH="$PIPELINE_DIR/examples.jsonl"
SENTENCES_PATH="$PIPELINE_DIR/sentences.jsonl"
LOCALIZATION_DIR="$PIPELINE_DIR/localization"
LOCALIZATION_JSONL="$PIPELINE_DIR/localization.jsonl"

BUILD_CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/src/build_sentence_dataset.py"
  --input_root "$MINING_DIR"
  --out_dir "$PIPELINE_DIR"
  --text_field "$TEXT_FIELD"
  --fallback_text_field "$FALLBACK_TEXT_FIELD"
  --label_filter "$LABEL_FILTER"
  --target_deceptive 0
  --target_truthful 0
  --include_messages
)
if [[ "$BUILD_LIMIT" != "0" ]]; then
  BUILD_CMD+=(--limit "$BUILD_LIMIT")
fi
if [[ ${#EXTRA_BUILD_ARGS[@]} -gt 0 ]]; then
  BUILD_CMD+=("${EXTRA_BUILD_ARGS[@]}")
fi

LOCALIZE_CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/src/sentence_localization_batch.py"
  --game "$ENVIRONMENT"
  --examples_path "$EXAMPLES_PATH"
  --sentences_path "$SENTENCES_PATH"
  --model_name "$MODEL_NAME"
  --out_dir "$LOCALIZATION_DIR"
  --overwrite
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
  --text_field "$TEXT_FIELD"
)
if [[ "$LOCALIZE_LIMIT" != "0" ]]; then
  LOCALIZE_CMD+=(--limit "$LOCALIZE_LIMIT")
fi
if [[ "$WRITE_JSONL" == "1" ]]; then
  LOCALIZE_CMD+=(--jsonl_path "$LOCALIZATION_JSONL")
fi
if [[ ${#EXTRA_LOCALIZE_ARGS[@]} -gt 0 ]]; then
  LOCALIZE_CMD+=("${EXTRA_LOCALIZE_ARGS[@]}")
fi

echo "Mining dir: $MINING_DIR"
echo "Environment: $ENVIRONMENT"
echo "Model: $MODEL_NAME"
echo "Pipeline dir: $PIPELINE_DIR"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo ""
echo "Building sentence dataset:"
printf ' %q' "${BUILD_CMD[@]}"
printf '\n'
"${BUILD_CMD[@]}"

echo ""
echo "Running localization:"
printf ' %q' "${LOCALIZE_CMD[@]}"
printf '\n'
"${LOCALIZE_CMD[@]}"

echo ""
echo "Done."
echo "Examples: $EXAMPLES_PATH"
echo "Sentences: $SENTENCES_PATH"
echo "Localization dir: $LOCALIZATION_DIR"
if [[ "$WRITE_JSONL" == "1" ]]; then
  echo "Localization JSONL: $LOCALIZATION_JSONL"
fi
