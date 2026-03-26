#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/playpen-ssd/smerrill/conda_envs/deception/bin/python}"
REPO_ROOT="/playpen-ssd/smerrill/deception2"

ENVIRONMENT=""
MODEL_NAME=""
GPU_ID="${CUDA_VISIBLE_DEVICES:-}"
OUTPUT_DIR=""
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
SAMPLES_PER_STATE="${SAMPLES_PER_STATE:-4}"
TARGET_DECEPTIVE="${TARGET_DECEPTIVE:-2}"
TARGET_TRUTHFUL="${TARGET_TRUTHFUL:-2}"
MAX_GAMES="${MAX_GAMES:-10}"
MAX_EPISODES="${MAX_EPISODES:-10}"
MAX_TURNS="${MAX_TURNS:-40}"
REASONING_MODE="${REASONING_MODE:-auto}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_deception_miner_smoke.sh --env {bs|gridworld|car_sales|advisor_audit} --model_name MODEL [options] [-- extra args]

Options:
  --env ENV                     One of: bs, gridworld, car_sales, advisor_audit
  --model_name MODEL            Hugging Face / vLLM model name
  --gpu GPU                     Optional single GPU id; otherwise uses existing CUDA_VISIBLE_DEVICES
  --output_dir PATH             Optional explicit output dir
  --run_tag TAG                 Optional subdir name under testing/miner_smoke (default: timestamp)
  --samples_per_state N         Default: 4
  --target_deceptive N          Default: 2
  --target_truthful N           Default: 2
  --max_games N                 Default for bs/gridworld: 10
  --max_episodes N              Default for advisor_audit: 10
  --max_turns N                 Default: 40
  --reasoning {auto|on|off}     Default: auto
  --help                        Show this message

Anything after `--` is forwarded to the underlying miner.
EOF
}

guess_reasoning_flag() {
  local mode="$1"
  local model_name="$2"
  local env="$3"
  local lower
  lower="$(printf '%s' "$model_name" | tr '[:upper:]' '[:lower:]')"

  if [[ "$mode" == "on" ]]; then
    printf '%s' "--is_reasoning_model"
    return
  fi

  if [[ "$mode" == "off" ]]; then
    if [[ "$env" == "advisor_audit" ]]; then
      printf '%s' "--no-is_reasoning_model"
    fi
    return
  fi

  if [[ "$lower" == *"r1"* || "$lower" == *"qwq"* || "$lower" == *"gpt-oss"* || "$lower" == *"reason"* || "$lower" == *"thinking"* ]]; then
    printf '%s' "--is_reasoning_model"
  fi
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
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run_tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --samples_per_state)
      SAMPLES_PER_STATE="$2"
      shift 2
      ;;
    --target_deceptive)
      TARGET_DECEPTIVE="$2"
      shift 2
      ;;
    --target_truthful)
      TARGET_TRUTHFUL="$2"
      shift 2
      ;;
    --max_games)
      MAX_GAMES="$2"
      shift 2
      ;;
    --max_episodes)
      MAX_EPISODES="$2"
      shift 2
      ;;
    --max_turns)
      MAX_TURNS="$2"
      shift 2
      ;;
    --reasoning)
      REASONING_MODE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ENVIRONMENT" || -z "$MODEL_NAME" ]]; then
  usage >&2
  exit 1
fi

case "$ENVIRONMENT" in
  bs|gridworld|car_sales|advisor_audit)
    ;;
  *)
    echo "Unsupported --env: $ENVIRONMENT" >&2
    exit 1
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

MODEL_TAG="${MODEL_NAME//\//_}"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$REPO_ROOT/testing/miner_smoke/$ENVIRONMENT/$MODEL_TAG/$RUN_TAG"
fi
mkdir -p "$OUTPUT_DIR"

REASONING_FLAG="$(guess_reasoning_flag "$REASONING_MODE" "$MODEL_NAME" "$ENVIRONMENT")"

if [[ "$ENVIRONMENT" == "advisor_audit" ]]; then
  SCRIPT="$REPO_ROOT/AdvisorAudit/src/financial_advisor_deception_miner.py"
  CMD=(
    "$PYTHON_BIN" "$SCRIPT"
    --model_name "$MODEL_NAME"
    --output_dir "$OUTPUT_DIR"
    --seed 0
    --samples_per_state "$SAMPLES_PER_STATE"
    --max_episodes "$MAX_EPISODES"
    --max_turns "$MAX_TURNS"
    --target_deceptive "$TARGET_DECEPTIVE"
    --target_truthful "$TARGET_TRUTHFUL"
    --label_filter all
    --log_every 5
  )
else
  SCRIPT="$REPO_ROOT/src/deception_miner.py"
  CMD=(
    "$PYTHON_BIN" "$SCRIPT"
    --game "$ENVIRONMENT"
    --model_name "$MODEL_NAME"
    --output_dir "$OUTPUT_DIR"
    --seed 0
    --samples_per_state "$SAMPLES_PER_STATE"
    --max_games "$MAX_GAMES"
    --max_turns "$MAX_TURNS"
    --target_deceptive "$TARGET_DECEPTIVE"
    --target_truthful "$TARGET_TRUTHFUL"
    --label_filter all
    --log_every 5
  )
  if [[ "$ENVIRONMENT" == "gridworld" ]]; then
    CMD+=(--max_steps 25 --history_window 10)
  elif [[ "$ENVIRONMENT" == "car_sales" ]]; then
    CMD+=(--car_sales_max_rounds 4 --history_window 12)
  fi
fi

if [[ -n "$REASONING_FLAG" ]]; then
  CMD+=("$REASONING_FLAG")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Environment: $ENVIRONMENT"
echo "Model: $MODEL_NAME"
echo "Output dir: $OUTPUT_DIR"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo "Running:"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
