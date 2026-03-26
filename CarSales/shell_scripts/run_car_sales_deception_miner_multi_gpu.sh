#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"
REPO_ROOT="/playpen-ssd/smerrill/deception2"
MINER_SCRIPT="$REPO_ROOT/src/deception_miner.py"
DEFAULT_OUTPUT_ROOT="$REPO_ROOT/CarSales/Results/deception_miner"

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
GPU_IDS_CSV="${GPU_IDS:-2}"
OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d)}"
TOTAL_TARGET_EXAMPLES="${TOTAL_TARGET_EXAMPLES:-5000}"
TARGET_DECEPTIVE_TOTAL="${TARGET_DECEPTIVE_TOTAL:-}"
TARGET_TRUTHFUL_TOTAL="${TARGET_TRUTHFUL_TOTAL:-}"
MAX_GAMES="${MAX_GAMES:-}"
SAMPLES_PER_STATE="${SAMPLES_PER_STATE:-25}"
CAR_SALES_MAX_ROUNDS="${CAR_SALES_MAX_ROUNDS:-4}"
MAX_TURNS="${MAX_TURNS:-}"
BASE_SEED="${BASE_SEED:-0}"
SCENARIO_NAME="${SCENARIO_NAME:-}"
RESUME=1
LOG_EVERY="${LOG_EVERY:-25}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-10000}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_STATE_RESAMPLE_ROUNDS="${MAX_STATE_RESAMPLE_ROUNDS:-0}"
REASONING_FLAG="--is_reasoning_model"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_car_sales_deception_miner_multi_gpu.sh --model_name MODEL [options] [-- extra miner args]

Options:
  --model_name MODEL                  Hugging Face / vLLM model name
  --gpus CSV                          Comma-separated GPU ids, default: 0,1,2,3
  --output_root PATH                  Root directory for run outputs
  --run_tag TAG                       Subdirectory name under output_root
  --total_target_examples N           Total saved examples target across all shards, default: 5000
  --target_deceptive_total N          Optional total deceptive target across all shards
  --target_truthful_total N           Optional total truthful target across all shards
  --max_games N                       Optional cap on total games considered before sharding
  --samples_per_state N               Default: 25
  --car_sales_max_rounds N            Default: 4
  --max_turns N                       Defaults to car_sales_max_rounds
  --scenario_name NAME                Optional fixed CarSales scenario name
  --base_seed N                       Default: 0
  --resume / --no-resume              Default: resume enabled
  --log_every N                       Default: 25
  --temperature F                     Default: 0.9
  --top_p F                           Default: 0.9
  --max_tokens N                      Default: 10000
  --repetition_penalty F              Default: 1.2
  --max_retries N                     Default: 3
  --max_state_resample_rounds N       Default: 0 (unlimited)
  --reasoning / --no-reasoning        Default: reasoning enabled
  --help                              Show this message

Anything after `--` is forwarded to each miner worker unchanged.

How work is shared:
- shard 0 processes game indices 0, N, 2N, ...
- shard 1 processes game indices 1, N+1, 2N+1, ...
- and so on, where N = number of GPUs / shards

CarSales paired mining:
- The miner resamples each seller-response state until it finds one deceptive and one truthful example.
- Each game has up to car_sales_max_rounds seller-response states.
- So one CarSales game can contribute up to car_sales_max_rounds deceptive examples and
  up to car_sales_max_rounds truthful examples.
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

ceil_div() {
  local num="$1"
  local den="$2"
  echo $(( (num + den - 1) / den ))
}

games_assigned_to_shard() {
  local total_games="$1"
  local num_shards="$2"
  local shard_index="$3"
  if (( total_games <= shard_index )); then
    echo 0
  else
    echo $(( (total_games - 1 - shard_index) / num_shards + 1 ))
  fi
}

required_global_games() {
  local target_deceptive_total="$1"
  local target_truthful_total="$2"
  local num_shards="$3"
  local rounds_per_game="$4"
  local required_total=0
  local shard_idx shard_deceptive shard_truthful shard_needed_games truthful_needed_games shard_required_total

  for (( shard_idx=0; shard_idx<num_shards; shard_idx++ )); do
    shard_deceptive="$(quota_for_shard "$target_deceptive_total" "$num_shards" "$shard_idx")"
    shard_truthful="$(quota_for_shard "$target_truthful_total" "$num_shards" "$shard_idx")"
    shard_needed_games="$(ceil_div "$shard_deceptive" "$rounds_per_game")"
    truthful_needed_games="$(ceil_div "$shard_truthful" "$rounds_per_game")"
    if (( truthful_needed_games > shard_needed_games )); then
      shard_needed_games="$truthful_needed_games"
    fi
    if (( shard_needed_games <= 0 )); then
      shard_required_total=0
    else
      shard_required_total=$(( shard_idx + 1 + (shard_needed_games - 1) * num_shards ))
    fi
    if (( shard_required_total > required_total )); then
      required_total="$shard_required_total"
    fi
  done

  echo "$required_total"
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
    --max_games)
      MAX_GAMES="$2"
      shift 2
      ;;
    --samples_per_state)
      SAMPLES_PER_STATE="$2"
      shift 2
      ;;
    --car_sales_max_rounds)
      CAR_SALES_MAX_ROUNDS="$2"
      shift 2
      ;;
    --max_turns)
      MAX_TURNS="$2"
      shift 2
      ;;
    --scenario_name)
      SCENARIO_NAME="$2"
      shift 2
      ;;
    --base_seed)
      BASE_SEED="$2"
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
    --max_state_resample_rounds)
      MAX_STATE_RESAMPLE_ROUNDS="$2"
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

if ! PYTHON_BIN="$(resolve_python_bin)"; then
  echo "Could not find a usable python interpreter." >&2
  echo "Set PYTHON_BIN explicitly or activate the desired conda environment first." >&2
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

IFS=',' read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_CSV"
if [[ "${#GPU_IDS_ARRAY[@]}" -eq 0 ]]; then
  echo "No GPUs provided." >&2
  exit 1
fi

NUM_SHARDS="${#GPU_IDS_ARRAY[@]}"

if (( CAR_SALES_MAX_ROUNDS <= 0 )); then
  echo "--car_sales_max_rounds must be positive." >&2
  exit 1
fi

if [[ -z "$TARGET_DECEPTIVE_TOTAL" ]]; then
  TARGET_DECEPTIVE_TOTAL=$(( (TOTAL_TARGET_EXAMPLES + 1) / 2 ))
fi

if [[ -z "$TARGET_TRUTHFUL_TOTAL" ]]; then
  TARGET_TRUTHFUL_TOTAL=$(( TOTAL_TARGET_EXAMPLES / 2 ))
fi

if (( TARGET_DECEPTIVE_TOTAL < 0 || TARGET_TRUTHFUL_TOTAL < 0 )); then
  echo "Target counts must be non-negative." >&2
  exit 1
fi

if [[ -z "$MAX_GAMES" ]]; then
  MAX_GAMES="$(required_global_games "$TARGET_DECEPTIVE_TOTAL" "$TARGET_TRUTHFUL_TOTAL" "$NUM_SHARDS" "$CAR_SALES_MAX_ROUNDS")"
fi

if [[ -z "$MAX_TURNS" ]]; then
  MAX_TURNS="$CAR_SALES_MAX_ROUNDS"
fi

if (( MAX_GAMES <= 0 )); then
  echo "--max_games must be positive." >&2
  exit 1
fi

max_label_capacity=$(( MAX_GAMES * CAR_SALES_MAX_ROUNDS ))
if (( TARGET_DECEPTIVE_TOTAL > max_label_capacity || TARGET_TRUTHFUL_TOTAL > max_label_capacity )); then
  echo "Global targets exceed aggregate max capacity ($max_label_capacity)." >&2
  echo "Increase --max_games or lower the target counts." >&2
  exit 1
fi

for shard_idx in "${!GPU_IDS_ARRAY[@]}"; do
  shard_target_deceptive="$(quota_for_shard "$TARGET_DECEPTIVE_TOTAL" "$NUM_SHARDS" "$shard_idx")"
  shard_target_truthful="$(quota_for_shard "$TARGET_TRUTHFUL_TOTAL" "$NUM_SHARDS" "$shard_idx")"
  shard_games="$(games_assigned_to_shard "$MAX_GAMES" "$NUM_SHARDS" "$shard_idx")"
  shard_capacity=$(( shard_games * CAR_SALES_MAX_ROUNDS ))
  if (( shard_target_deceptive > shard_capacity )); then
    echo "Shard $shard_idx deceptive target ($shard_target_deceptive) exceeds its capacity ($shard_capacity)." >&2
    echo "Increase --max_games or lower --target_deceptive_total." >&2
    exit 1
  fi
  if (( shard_target_truthful > shard_capacity )); then
    echo "Shard $shard_idx truthful target ($shard_target_truthful) exceeds its capacity ($shard_capacity)." >&2
    echo "Increase --max_games or lower --target_truthful_total." >&2
    exit 1
  fi
done

MODEL_TAG="${MODEL_NAME//\//_}"
RUN_DIR="$OUTPUT_ROOT/$MODEL_TAG/$RUN_TAG"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

echo "CarSales miner multi-GPU run"
echo "Model: $MODEL_NAME"
echo "Python: $PYTHON_BIN"
echo "GPUs: ${GPU_IDS_ARRAY[*]}"
echo "Num shards: $NUM_SHARDS"
echo "Run dir: $RUN_DIR"
echo "Scenario: ${SCENARIO_NAME:-auto-cycle}"
echo "CarSales max rounds: $CAR_SALES_MAX_ROUNDS"
echo "Max turns: $MAX_TURNS"
echo "Max games: $MAX_GAMES"
echo "Total target deceptive: $TARGET_DECEPTIVE_TOTAL"
echo "Total target truthful: $TARGET_TRUTHFUL_TOTAL"
echo "Total target examples: $(( TARGET_DECEPTIVE_TOTAL + TARGET_TRUTHFUL_TOTAL ))"
echo "Max state resample rounds: $MAX_STATE_RESAMPLE_ROUNDS"
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
      --game car_sales
      --model_name "$MODEL_NAME"
      --output_dir "$shard_dir"
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
      --max_state_resample_rounds "$MAX_STATE_RESAMPLE_ROUNDS"
      --car_sales_max_rounds "$CAR_SALES_MAX_ROUNDS"
      --seed "$(( BASE_SEED + shard_idx * 100000 ))"
      --shard_index "$shard_idx"
      --num_shards "$NUM_SHARDS"
    )
    if [[ -n "$SCENARIO_NAME" ]]; then
      cmd+=(--car_sales_scenario_name "$SCENARIO_NAME")
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
MERGED_PROCESSED="$MERGED_DIR/processed_car_sales_games.jsonl"
: > "$MERGED_SAMPLES"
: > "$MERGED_PROCESSED"

for shard_dir in "$RUN_DIR"/shard_*; do
  [[ -d "$shard_dir" ]] || continue
  if [[ -f "$shard_dir/deception_samples.jsonl" ]]; then
    cat "$shard_dir/deception_samples.jsonl" >> "$MERGED_SAMPLES"
  fi
  if [[ -f "$shard_dir/processed_car_sales_games.jsonl" ]]; then
    cat "$shard_dir/processed_car_sales_games.jsonl" >> "$MERGED_PROCESSED"
  fi
done

echo
echo "All CarSales miner workers completed successfully."
echo "Run dir: $RUN_DIR"
echo "Merged samples: $MERGED_SAMPLES"
echo "Merged processed manifest: $MERGED_PROCESSED"
