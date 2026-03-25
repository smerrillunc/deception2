#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/playpen-ssd/smerrill/conda_envs/deception/bin/python}"
REPO_ROOT="/playpen-ssd/smerrill/deception2"
MINER_SCRIPT="$REPO_ROOT/src/deception_miner.py"
DEFAULT_CONVERSATIONS_PATH="$REPO_ROOT/Interview/Data/interview_conversation_seeds.jsonl"
DEFAULT_OUTPUT_ROOT="$REPO_ROOT/Interview/Results/deception_miner"

MODEL_NAME=""
CONVERSATIONS_PATH="$DEFAULT_CONVERSATIONS_PATH"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
TOTAL_TARGET_EXAMPLES="${TOTAL_TARGET_EXAMPLES:-5000}"
TARGET_DECEPTIVE_TOTAL="${TARGET_DECEPTIVE_TOTAL:-}"
TARGET_TRUTHFUL_TOTAL="${TARGET_TRUTHFUL_TOTAL:-}"
SAMPLES_PER_STATE="${SAMPLES_PER_STATE:-16}"
MAX_GAMES=""
MAX_TURNS="${MAX_TURNS:-4}"
BASE_SEED="${BASE_SEED:-0}"
PRIVATE_PROFILE_NAME="${PRIVATE_PROFILE_NAME:-}"
RESUME=1
LOG_EVERY="${LOG_EVERY:-25}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_P="${TOP_P:-0.5}"
MAX_TOKENS="${MAX_TOKENS:-10000}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
MAX_RETRIES="${MAX_RETRIES:-3}"
REASONING_FLAG="--is_reasoning_model"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_interview_deception_miner_multi_gpu.sh --model_name MODEL [options] [-- extra miner args]

Options:
  --model_name MODEL                  Hugging Face / vLLM model name
  --conversations_path PATH           JSONL conversation seeds file
  --gpus CSV                          Comma-separated GPU ids, default: 0,1,2,3
  --output_root PATH                  Root directory for run outputs
  --run_tag TAG                       Subdirectory name under output_root
  --total_target_examples N           Default: 5000; split 50/50 deceptive/truthful unless totals are overridden
  --target_deceptive_total N          Optional total deceptive target across all shards
  --target_truthful_total N           Optional total truthful target across all shards
  --samples_per_state N               Default: 16
  --max_games N                       Optional cap on conversation seeds considered before sharding
  --max_turns N                       Default: 4
  --base_seed N                       Default: 0
  --private_profile_name NAME         Optional fixed interview private profile
  --resume / --no-resume              Default: resume enabled
  --log_every N                       Default: 25
  --temperature F                     Default: 0.5
  --top_p F                           Default: 0.5
  --max_tokens N                      Default: 10000
  --repetition_penalty F              Default: 1.2
  --max_retries N                     Default: 3
  --reasoning / --no-reasoning        Default: reasoning enabled
  --help                              Show this message

Anything after `--` is forwarded to each miner worker unchanged.

How work is shared:
- shard 0 processes conversation indices 0, N, 2N, ...
- shard 1 processes conversation indices 1, N+1, 2N+1, ...
- and so on, where N = number of GPUs / shards
EOF
}

quota_for_shard() {
  local total="$1"
  local num_shards="$2"
  local shard_index="$3"
  local base remainder
  base=$(( total / num_shards ))
  remainder=$(( total % num_shards ))
  if (( shard_index < remainder )); then
    echo $(( base + 1 ))
  else
    echo "$base"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --conversations_path)
      CONVERSATIONS_PATH="$2"
      shift 2
      ;;
    --gpus)
      GPU_IDS_CSV="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run_tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --total_target_examples)
      TOTAL_TARGET_EXAMPLES="$2"
      shift 2
      ;;
    --target_deceptive_total)
      TARGET_DECEPTIVE_TOTAL="$2"
      shift 2
      ;;
    --target_truthful_total)
      TARGET_TRUTHFUL_TOTAL="$2"
      shift 2
      ;;
    --samples_per_state)
      SAMPLES_PER_STATE="$2"
      shift 2
      ;;
    --max_games)
      MAX_GAMES="$2"
      shift 2
      ;;
    --max_turns)
      MAX_TURNS="$2"
      shift 2
      ;;
    --base_seed)
      BASE_SEED="$2"
      shift 2
      ;;
    --private_profile_name)
      PRIVATE_PROFILE_NAME="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --no-resume)
      RESUME=0
      shift
      ;;
    --log_every)
      LOG_EVERY="$2"
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
    --max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --repetition_penalty)
      REPETITION_PENALTY="$2"
      shift 2
      ;;
    --max_retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --reasoning)
      REASONING_FLAG="--is_reasoning_model"
      shift
      ;;
    --no-reasoning)
      REASONING_FLAG=""
      shift
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

if [[ -z "$MODEL_NAME" ]]; then
  echo "--model_name is required." >&2
  usage >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$MINER_SCRIPT" ]]; then
  echo "Miner script not found: $MINER_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$CONVERSATIONS_PATH" ]]; then
  echo "Conversation seed file not found: $CONVERSATIONS_PATH" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_CSV"
if [[ "${#GPU_IDS_ARRAY[@]}" -eq 0 ]]; then
  echo "No GPUs provided." >&2
  exit 1
fi

NUM_SHARDS="${#GPU_IDS_ARRAY[@]}"

if [[ -z "$TARGET_DECEPTIVE_TOTAL" || -z "$TARGET_TRUTHFUL_TOTAL" ]]; then
  TARGET_DECEPTIVE_TOTAL=$(( TOTAL_TARGET_EXAMPLES / 2 ))
  TARGET_TRUTHFUL_TOTAL=$(( TOTAL_TARGET_EXAMPLES - TARGET_DECEPTIVE_TOTAL ))
fi

if [[ -z "$MAX_GAMES" ]]; then
  MAX_GAMES="$(grep -cve '^[[:space:]]*$' "$CONVERSATIONS_PATH")"
fi

MODEL_TAG="${MODEL_NAME//\//_}"
RUN_DIR="$OUTPUT_ROOT/$MODEL_TAG/$RUN_TAG"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

echo "Interview miner multi-GPU run"
echo "Model: $MODEL_NAME"
echo "Conversations: $CONVERSATIONS_PATH"
echo "GPUs: ${GPU_IDS_ARRAY[*]}"
echo "Num shards: $NUM_SHARDS"
echo "Run dir: $RUN_DIR"
echo "Max games: $MAX_GAMES"
echo "Total target deceptive: $TARGET_DECEPTIVE_TOTAL"
echo "Total target truthful: $TARGET_TRUTHFUL_TOTAL"
echo "Resume: $RESUME"

pids=()
pid_names=()

for shard_idx in "${!GPU_IDS_ARRAY[@]}"; do
  gpu="${GPU_IDS_ARRAY[$shard_idx]}"
  shard_target_deceptive="$(quota_for_shard "$TARGET_DECEPTIVE_TOTAL" "$NUM_SHARDS" "$shard_idx")"
  shard_target_truthful="$(quota_for_shard "$TARGET_TRUTHFUL_TOTAL" "$NUM_SHARDS" "$shard_idx")"

  if (( shard_target_deceptive == 0 && shard_target_truthful == 0 )); then
    echo "Skipping shard $shard_idx on GPU $gpu because both targets are zero."
    continue
  fi

  shard_dir="$RUN_DIR/shard_$(printf '%02d' "$shard_idx")"
  mkdir -p "$shard_dir"
  log_path="$LOG_DIR/shard_$(printf '%02d' "$shard_idx")_gpu_${gpu}.log"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    cmd=(
      "$PYTHON_BIN" "$MINER_SCRIPT"
      --game interview
      --model_name "$MODEL_NAME"
      --output_dir "$shard_dir"
      --interview_conversations_path "$CONVERSATIONS_PATH"
      --samples_per_state "$SAMPLES_PER_STATE"
      --max_games "$MAX_GAMES"
      --max_turns "$MAX_TURNS"
      --target_deceptive "$shard_target_deceptive"
      --target_truthful "$shard_target_truthful"
      --label_filter all
      --log_every "$LOG_EVERY"
      --temperature "$TEMPERATURE"
      --top_p "$TOP_P"
      --max_tokens "$MAX_TOKENS"
      --repetition_penalty "$REPETITION_PENALTY"
      --max_retries "$MAX_RETRIES"
      --seed "$(( BASE_SEED + shard_idx * 100000 ))"
      --shard_index "$shard_idx"
      --num_shards "$NUM_SHARDS"
    )
    if [[ -n "$PRIVATE_PROFILE_NAME" ]]; then
      cmd+=(--interview_private_profile_name "$PRIVATE_PROFILE_NAME")
    fi
    if [[ "$RESUME" == "1" ]]; then
      cmd+=(--resume)
    else
      cmd+=(--no-resume)
    fi
    if [[ -n "$REASONING_FLAG" ]]; then
      cmd+=("$REASONING_FLAG")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      cmd+=("${EXTRA_ARGS[@]}")
    fi

    {
      echo "Shard: $shard_idx / $NUM_SHARDS"
      echo "GPU: $gpu"
      echo "Output dir: $shard_dir"
      echo "Target deceptive: $shard_target_deceptive"
      echo "Target truthful: $shard_target_truthful"
      echo "Command:"
      printf ' %q' "${cmd[@]}"
      printf '\n\n'
      "${cmd[@]}"
    } > "$log_path" 2>&1
  ) &

  pids+=("$!")
  pid_names+=("shard=$shard_idx,gpu=$gpu")
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No workers were launched." >&2
  exit 1
fi

failed=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    failed=$(( failed + 1 ))
    ident="${pid_names[$idx]}"
    shard="$(echo "$ident" | sed -E 's/shard=([0-9]+),gpu=.*/\1/')"
    gpu="$(echo "$ident" | sed -E 's/shard=[0-9]+,gpu=([0-9]+).*/\1/')"
    log_path="$LOG_DIR/shard_$(printf '%02d' "$shard")_gpu_${gpu}.log"
    echo "Worker failed: $ident" >&2
    echo "Tail of log ($log_path):" >&2
    tail -n 80 "$log_path" >&2 || true
  fi
done

if (( failed > 0 )); then
  echo "$failed worker(s) failed." >&2
  exit 1
fi

MERGED_DIR="$RUN_DIR/merged"
mkdir -p "$MERGED_DIR"
MERGED_SAMPLES="$MERGED_DIR/deception_samples.jsonl"
MERGED_PROCESSED="$MERGED_DIR/processed_interview_conversations.jsonl"
: > "$MERGED_SAMPLES"
: > "$MERGED_PROCESSED"

for shard_dir in "$RUN_DIR"/shard_*; do
  [[ -d "$shard_dir" ]] || continue
  if [[ -f "$shard_dir/deception_samples.jsonl" ]]; then
    cat "$shard_dir/deception_samples.jsonl" >> "$MERGED_SAMPLES"
  fi
  if [[ -f "$shard_dir/processed_interview_conversations.jsonl" ]]; then
    cat "$shard_dir/processed_interview_conversations.jsonl" >> "$MERGED_PROCESSED"
  fi
done

echo
echo "All interview miner workers completed successfully."
echo "Run dir: $RUN_DIR"
echo "Merged samples: $MERGED_SAMPLES"
echo "Merged processed manifest: $MERGED_PROCESSED"
