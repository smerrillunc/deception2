#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"
REPO_ROOT="/playpen-ssd/smerrill/deception2"
GENERATOR_SCRIPT="$REPO_ROOT/Interview/src/generate_interview_conversation_seeds_vllm.py"
DEFAULT_OUTPUT_ROOT="$REPO_ROOT/Interview/Data/generated_seed_runs"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
TOTAL_CONVERSATIONS="${TOTAL_CONVERSATIONS:-5000}"
TURNS_PER_CONVERSATION="${TURNS_PER_CONVERSATION:-4}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-1.0}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-600}"
REQUEST_PAUSE_SECONDS="${REQUEST_PAUSE_SECONDS:-0.0}"
MAX_RETRIES="${MAX_RETRIES:-5}"
BASE_SEED="${BASE_SEED:-0}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LOG_EVERY="${LOG_EVERY:-25}"
RESUME=1
TRUST_REMOTE_CODE=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_interview_seed_generation_vllm_multi_gpu.sh [options] [-- extra generator args]

Options:
  --model_name MODEL                  Default: Qwen/Qwen2.5-14B-Instruct
  --gpus CSV                          Comma-separated GPU ids, default: 0,1,2,3
  --output_root PATH                  Root directory for shard outputs
  --run_tag TAG                       Subdirectory name under output_root
  --total_conversations N             Default: 5000
  --turns_per_conversation N          Default: 4
  --temperature F                     Default: 0.9
  --top_p F                           Default: 1.0
  --max_output_tokens N               Default: 600
  --request_pause_seconds F           Default: 0.0
  --max_retries N                     Default: 5
  --base_seed N                       Default: 0
  --dtype NAME                        Default: bfloat16
  --max_model_len N                   Default: 4096
  --gpu_memory_utilization F          Default: 0.9
  --tensor_parallel_size N            Default: 1
  --log_every N                       Default: 25
  --resume / --no-resume              Default: resume enabled
  --trust_remote_code                 Enable trust_remote_code
  --help                              Show this message

Anything after `--` is forwarded to each generator worker unchanged.

How work is shared:
- shard 0 generates conversation indices 0, N, 2N, ...
- shard 1 generates conversation indices 1, N+1, 2N+1, ...
- and so on, where N = number of GPUs / shards
EOF
}

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi

  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    printf '%s\n' "${CONDA_PREFIX}/bin/python"
    return
  fi

  if [[ -x "/playpen-ssd/smerrill/conda_envs/deception/bin/python" ]]; then
    printf '%s\n' "/playpen-ssd/smerrill/conda_envs/deception/bin/python"
    return
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name)
      MODEL_NAME="$2"
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
    --total_conversations)
      TOTAL_CONVERSATIONS="$2"
      shift 2
      ;;
    --turns_per_conversation)
      TURNS_PER_CONVERSATION="$2"
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
    --max_output_tokens)
      MAX_OUTPUT_TOKENS="$2"
      shift 2
      ;;
    --request_pause_seconds)
      REQUEST_PAUSE_SECONDS="$2"
      shift 2
      ;;
    --max_retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --base_seed)
      BASE_SEED="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --max_model_len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --gpu_memory_utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --tensor_parallel_size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --log_every)
      LOG_EVERY="$2"
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
    --trust_remote_code)
      TRUST_REMOTE_CODE=1
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

if ! PYTHON_BIN="$(resolve_python_bin)"; then
  echo "Could not find a usable python interpreter." >&2
  echo "Set PYTHON_BIN explicitly or activate the desired conda environment first." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$GENERATOR_SCRIPT" ]]; then
  echo "Generator script not found: $GENERATOR_SCRIPT" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_CSV"
if [[ "${#GPU_IDS_ARRAY[@]}" -eq 0 ]]; then
  echo "No GPUs provided." >&2
  exit 1
fi

NUM_SHARDS="${#GPU_IDS_ARRAY[@]}"
MODEL_TAG="${MODEL_NAME//\//_}"
RUN_DIR="$OUTPUT_ROOT/$MODEL_TAG/$RUN_TAG"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

echo "Interview seed-generation multi-GPU run"
echo "Model: $MODEL_NAME"
echo "GPUs: ${GPU_IDS_ARRAY[*]}"
echo "Num shards: $NUM_SHARDS"
echo "Run dir: $RUN_DIR"
echo "Total conversations: $TOTAL_CONVERSATIONS"
echo "Resume: $RESUME"

pids=()
pid_names=()

for shard_idx in "${!GPU_IDS_ARRAY[@]}"; do
  gpu="${GPU_IDS_ARRAY[$shard_idx]}"
  shard_dir="$RUN_DIR/shard_$(printf '%02d' "$shard_idx")"
  shard_output="$shard_dir/interview_conversation_seeds.jsonl"
  log_path="$LOG_DIR/shard_$(printf '%02d' "$shard_idx")_gpu_${gpu}.log"
  mkdir -p "$shard_dir"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    cmd=(
      "$PYTHON_BIN" "$GENERATOR_SCRIPT"
      --model_name "$MODEL_NAME"
      --output_path "$shard_output"
      --total_conversations "$TOTAL_CONVERSATIONS"
      --turns_per_conversation "$TURNS_PER_CONVERSATION"
      --temperature "$TEMPERATURE"
      --top_p "$TOP_P"
      --max_output_tokens "$MAX_OUTPUT_TOKENS"
      --request_pause_seconds "$REQUEST_PAUSE_SECONDS"
      --max_retries "$MAX_RETRIES"
      --seed "$(( BASE_SEED + shard_idx * 100000 ))"
      --dtype "$DTYPE"
      --max_model_len "$MAX_MODEL_LEN"
      --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
      --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
      --log_every "$LOG_EVERY"
      --run_tag "$RUN_TAG"
      --shard_index "$shard_idx"
      --num_shards "$NUM_SHARDS"
    )
    if [[ "$RESUME" == "1" ]]; then
      cmd+=(--resume)
    else
      cmd+=(--no-resume)
    fi
    if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
      cmd+=(--trust_remote_code)
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      cmd+=("${EXTRA_ARGS[@]}")
    fi

    {
      echo "Shard: $shard_idx / $NUM_SHARDS"
      echo "GPU: $gpu"
      echo "Output path: $shard_output"
      echo "Command:"
      printf ' %q' "${cmd[@]}"
      printf '\n\n'
      "${cmd[@]}"
    } > "$log_path" 2>&1
  ) &

  pids+=("$!")
  pid_names+=("shard=$shard_idx,gpu=$gpu")
done

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
    tail -n 120 "$log_path" >&2 || true
  fi
done

if (( failed > 0 )); then
  echo "$failed worker(s) failed." >&2
  exit 1
fi

MERGED_DIR="$RUN_DIR/merged"
MERGED_OUTPUT="$MERGED_DIR/interview_conversation_seeds.jsonl"
mkdir -p "$MERGED_DIR"
: > "$MERGED_OUTPUT"

for shard_dir in "$RUN_DIR"/shard_*; do
  [[ -d "$shard_dir" ]] || continue
  if [[ -f "$shard_dir/interview_conversation_seeds.jsonl" ]]; then
    cat "$shard_dir/interview_conversation_seeds.jsonl" >> "$MERGED_OUTPUT"
  fi
done

echo
echo "All interview seed-generation workers completed successfully."
echo "Run dir: $RUN_DIR"
echo "Merged output: $MERGED_OUTPUT"
