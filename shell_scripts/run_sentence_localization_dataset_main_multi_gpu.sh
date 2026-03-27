#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/playpen-ssd/smerrill/conda_envs/deception/bin/python}"
REPO_ROOT="/playpen-ssd/smerrill/deception2"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/DatasetMain}"

ENVIRONMENT=""
MODEL_NAME=""
GPU_IDS_STR="${GPU_IDS:-}"

N_SAMPLES="${N_SAMPLES:-50}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
BASE_SEED_START="${BASE_SEED_START:-1234}"
METHOD="${METHOD:-adaptive}"
MODE="${MODE:-prefix}"
TEXT_FIELD="${TEXT_FIELD:-action_reasoning}"
LABEL_FILTER="${LABEL_FILTER:-all}"
LIMIT="${LIMIT:-0}"
LOG_EVERY="${LOG_EVERY:-25}"
COARSE_ITERS="${COARSE_ITERS:-8}"
REFINEMENT_ITERS="${REFINEMENT_ITERS:-8}"
MIN_VALID="${MIN_VALID:-3}"
MIN_STEP_SIZE="${MIN_STEP_SIZE:-1}"
MIN_SPACING="${MIN_SPACING:-1}"
OVERWRITE="${OVERWRITE:-0}"
WRITE_JSONL="${WRITE_JSONL:-0}"
JSONL_BASENAME="${JSONL_BASENAME:-localization.jsonl}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

usage() {
  cat <<'EOF'
Usage:
  run_sentence_localization_dataset_main_multi_gpu.sh --env ENV --model_name MODEL --gpu_ids "2 3 4 5"

Required:
  --env ENV                  One of: bs, gridworld, advisor_audit, interview, car_sales
  --model_name MODEL         Hugging Face / vLLM model name
  --gpu_ids "2 3 4 5"        Space-separated GPU ids on the current machine

Optional:
  --dataset_root DIR         Default: /playpen-ssd/smerrill/deception2/DatasetMain
  --text_field FIELD         Default: action_reasoning
  --gpu_memory_utilization F Default: 0.9
  --method adaptive|full     Default: adaptive
  --mode prefix|sentence_only
  --limit N                  Optional example limit
  --overwrite                Pass --overwrite to the localizer
  --write_jsonl              Also write localization.jsonl
  --help                     Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --gpu_ids)
      GPU_IDS_STR="$2"
      shift 2
      ;;
    --dataset_root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --text_field)
      TEXT_FIELD="$2"
      shift 2
      ;;
    --gpu_memory_utilization)
      GPU_MEMORY_UTILIZATION="$2"
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
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --write_jsonl)
      WRITE_JSONL=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ENVIRONMENT" || -z "$MODEL_NAME" || -z "$GPU_IDS_STR" ]]; then
  usage >&2
  exit 1
fi

case "$ENVIRONMENT" in
  bs|gridworld|advisor_audit|interview|car_sales)
    ;;
  *)
    echo "Unsupported env: $ENVIRONMENT" >&2
    exit 1
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

GPU_IDS_ARRAY=($GPU_IDS_STR)
NUM_SHARDS=${#GPU_IDS_ARRAY[@]}
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "No GPUs provided." >&2
  exit 1
fi

MODEL_TAIL="${MODEL_NAME##*/}"
DATA_DIR="$DATASET_ROOT/$ENVIRONMENT/$MODEL_TAIL"
EXAMPLES_PATH="$DATA_DIR/examples.jsonl"
SENTENCES_PATH="$DATA_DIR/sentences.jsonl"
OUT_DIR="$DATA_DIR/localization"

if [[ ! -f "$EXAMPLES_PATH" ]]; then
  echo "Missing examples file: $EXAMPLES_PATH" >&2
  exit 1
fi
if [[ ! -f "$SENTENCES_PATH" ]]; then
  echo "Missing sentences file: $SENTENCES_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "Environment: $ENVIRONMENT"
echo "Model: $MODEL_NAME"
echo "Dataset dir: $DATA_DIR"
echo "GPUs: ${GPU_IDS_ARRAY[*]}"
echo "Localization dir: $OUT_DIR"

pids=()
pid_gpus=()
for idx in "${!GPU_IDS_ARRAY[@]}"; do
  gpu="${GPU_IDS_ARRAY[$idx]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    cmd=(
      "$PYTHON_BIN" "$REPO_ROOT/src/sentence_localization_batch.py"
      --game "$ENVIRONMENT"
      --examples_path "$EXAMPLES_PATH"
      --sentences_path "$SENTENCES_PATH"
      --model_name "$MODEL_NAME"
      --out_dir "$OUT_DIR"
      --n_samples "$N_SAMPLES"
      --temperature "$TEMPERATURE"
      --top_p "$TOP_P"
      --repetition_penalty "$REPETITION_PENALTY"
      --max_new_tokens "$MAX_NEW_TOKENS"
      --base_seed "$((BASE_SEED_START + idx * 100000))"
      --method "$METHOD"
      --mode "$MODE"
      --coarse_iters "$COARSE_ITERS"
      --refinement_iters "$REFINEMENT_ITERS"
      --min_valid "$MIN_VALID"
      --min_step_size "$MIN_STEP_SIZE"
      --min_spacing "$MIN_SPACING"
      --label_filter "$LABEL_FILTER"
      --text_field "$TEXT_FIELD"
      --shard_id "$idx"
      --num_shards "$NUM_SHARDS"
      --log_every "$LOG_EVERY"
      --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    )
    if [[ "$LIMIT" -gt 0 ]]; then
      cmd+=(--limit "$LIMIT")
    fi
    if [[ "$OVERWRITE" == "1" ]]; then
      cmd+=(--overwrite)
    fi
    if [[ "$WRITE_JSONL" == "1" ]]; then
      cmd+=(--jsonl_path "$DATA_DIR/$JSONL_BASENAME")
    fi
    "${cmd[@]}" > "$OUT_DIR/run_gpu_$gpu.log" 2>&1
  ) &
  pids+=("$!")
  pid_gpus+=("$gpu")
done

failed=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    failed=$((failed + 1))
    gpu="${pid_gpus[$idx]}"
    echo "Localization worker failed on GPU $gpu. Tail of log:"
    tail -n 60 "$OUT_DIR/run_gpu_$gpu.log" || true
  fi
done

if (( failed > 0 )); then
  echo "$failed localization worker(s) failed." >&2
  exit 1
fi

echo "Sentence localization complete."
echo "Output dir: $OUT_DIR"
