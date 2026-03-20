#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENVIRONMENT="all"
MODEL_NAME=""
GPU_IDS_STR="${GPU_IDS:-2 3 4 5 6}"
FORCE_MINE="${FORCE_MINE:-0}"
SKIP_MINING="${SKIP_MINING:-0}"
OVERWRITE_LOCALIZATION="${OVERWRITE_LOCALIZATION:-0}"
WRITE_JSONL="${WRITE_JSONL:-0}"
MINE_ONLY="${MINE_ONLY:-0}"

CONDA_ENV="${CONDA_ENV:-deception}"
PYTHON_BIN="${PYTHON_BIN:-}"

TARGET_DECEPTIVE="${TARGET_DECEPTIVE:-2500}"
TARGET_TRUTHFUL="${TARGET_TRUTHFUL:-2500}"
TEXT_FIELD="${TEXT_FIELD:-action_reasoning}"
FALLBACK_TEXT_FIELD="${FALLBACK_TEXT_FIELD:-action_raw_text}"
BUILD_LABEL_FILTER="${BUILD_LABEL_FILTER:-all}"

MINE_TEMPERATURE="${MINE_TEMPERATURE:-0.5}"
MINE_TOP_P="${MINE_TOP_P:-0.5}"
MINE_MAX_RETRIES="${MINE_MAX_RETRIES:-3}"
MINE_BASE_SEED="${MINE_BASE_SEED:-0}"
MAX_TURNS="${MAX_TURNS:-1000}"
MAX_GAMES="${MAX_GAMES:-3000}"
MAX_EPISODES="${MAX_EPISODES:-3000}"
BS_MINE_SAMPLES_PER_STATE="${BS_MINE_SAMPLES_PER_STATE:-1}"
GRIDWORLD_MINE_SAMPLES_PER_STATE="${GRIDWORLD_MINE_SAMPLES_PER_STATE:-1}"
ADVISOR_MINE_SAMPLES_PER_STATE="${ADVISOR_MINE_SAMPLES_PER_STATE:-32}"

N_SAMPLES="${N_SAMPLES:-100}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
BASE_SEED_START="${BASE_SEED_START:-1234}"
METHOD="${METHOD:-adaptive}"
MODE="${MODE:-prefix}"
COARSE_ITERS="${COARSE_ITERS:-8}"
REFINEMENT_ITERS="${REFINEMENT_ITERS:-8}"
MIN_VALID="${MIN_VALID:-3}"
MIN_STEP_SIZE="${MIN_STEP_SIZE:-1}"
MIN_SPACING="${MIN_SPACING:-1}"
LOG_EVERY="${LOG_EVERY:-25}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
JSONL_BASENAME="${JSONL_BASENAME:-localization.jsonl}"
RUN_TAG="${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/DatasetMain}"

usage() {
  cat <<'EOF'
Usage:
  run_targeted_localization_pipeline.sh --model_name MODEL [options]

Options:
  --env ENV                  One of: bs, gridworld, advisor_audit, all (default: all)
  --model_name MODEL         Hugging Face / vLLM model name
  --gpu_ids "2 3 4 5"        GPUs to use on the current host
  --dataset_root PATH        Output root (default: deception2/DatasetMain)
  --force_mine               Ignore existing mined counts and mine again
  --skip_mining              Never mine; fail later if existing data is insufficient
  --overwrite_localization   Pass --overwrite to sentence_localization_batch.py
  --write_jsonl              Also write shard JSONL outputs during localization
  --mine_only                Stop after ensuring mined data reaches target counts
  --help                     Show this message

Most tuning is driven by environment variables. Defaults match:
  target_deceptive=2500, target_truthful=2500
  n_samples=100, temperature=0.7, top_p=0.9, repetition_penalty=1.1
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
    --force_mine)
      FORCE_MINE="1"
      shift
      ;;
    --skip_mining)
      SKIP_MINING="1"
      shift
      ;;
    --overwrite_localization)
      OVERWRITE_LOCALIZATION="1"
      shift
      ;;
    --write_jsonl)
      WRITE_JSONL="1"
      shift
      ;;
    --mine_only)
      MINE_ONLY="1"
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

if [[ -z "$MODEL_NAME" ]]; then
  usage >&2
  exit 1
fi

case "$ENVIRONMENT" in
  bs|gridworld|advisor_audit|all)
    ;;
  *)
    echo "Unsupported env: $ENVIRONMENT" >&2
    exit 1
    ;;
esac

activate_python() {
  if [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]]; then
    return
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    return
  fi

  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    if [[ -f "/playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh" ]]; then
      # Beowulf / playpen default.
      # shellcheck disable=SC1091
      source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
      conda activate "$CONDA_ENV"
    elif command -v conda >/dev/null 2>&1; then
      # shellcheck disable=SC1091
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$CONDA_ENV"
    fi
  fi

  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    return
  fi

  PYTHON_BIN="$(command -v python)"
}

guess_reasoning_flag() {
  local model_name="$1"
  local env_name="$2"
  local lower
  lower="$(printf '%s' "$model_name" | tr '[:upper:]' '[:lower:]')"
  if [[ "$lower" == *"r1"* || "$lower" == *"qwq"* || "$lower" == *"gpt-oss"* || "$lower" == *"reason"* || "$lower" == *"thinking"* ]]; then
    printf '%s' "--is_reasoning_model"
    return
  fi
  if [[ "$env_name" == "advisor_audit" ]]; then
    printf '%s' "--no-is_reasoning_model"
  fi
}

count_labels_in_root() {
  local root="$1"
  "$PYTHON_BIN" - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    print("0 0 0 0")
    raise SystemExit(0)

total = dec = tru = unk = 0
for path in sorted(root.rglob("deception_samples.jsonl")):
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                rec = json.loads(line)
                val = rec.get("deceptive")
                if val is True:
                    dec += 1
                elif val is False:
                    tru += 1
                else:
                    unk += 1
    except Exception:
        continue
print(f"{total} {dec} {tru} {unk}")
PY
}

count_examples_file() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("0 0 0 0")
    raise SystemExit(0)

total = dec = tru = unk = 0
with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        rec = json.loads(line)
        val = rec.get("deceptive")
        if val is True:
            dec += 1
        elif val is False:
            tru += 1
        else:
            unk += 1
print(f"{total} {dec} {tru} {unk}")
PY
}

activate_python

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Could not resolve a Python executable." >&2
  exit 1
fi

GPU_IDS_ARRAY=($GPU_IDS_STR)
NUM_WORKERS=${#GPU_IDS_ARRAY[@]}
if [[ "$NUM_WORKERS" -lt 1 ]]; then
  echo "No GPUs provided." >&2
  exit 1
fi

if [[ "$ENVIRONMENT" == "all" ]]; then
  ENVIRONMENTS_ARRAY=(bs gridworld advisor_audit)
else
  ENVIRONMENTS_ARRAY=("$ENVIRONMENT")
fi

MODEL_TAG_RAW="${MODEL_NAME//\//_}"
run_single_environment() {
  local ENVIRONMENT="$1"
  local MODEL_TAG_BASE="${MODEL_NAME##*/}"
  local ENV_ROOT GAME_NAME RAW_MODEL_ROOT BASE_MODEL_ROOT
  local raw_total raw_dec raw_tru raw_unk base_total base_dec base_tru base_unk
  local MODEL_ROOT existing_total existing_dec existing_tru existing_unk
  local remaining_dec remaining_tru need_mining mine_label_filter
  local per_worker_dec per_worker_tru reasoning_flag mine_run_root
  local failed ds_total ds_dec ds_tru ds_unk PIPELINE_DIR EXAMPLES_PATH SENTENCES_PATH
  local LOCALIZATION_DIR loc_failed
  local pids pid_gpus loc_pids loc_pid_gpus

  case "$ENVIRONMENT" in
    bs)
      ENV_ROOT="$REPO_ROOT/BS"
      GAME_NAME="bs"
      ;;
    gridworld)
      ENV_ROOT="$REPO_ROOT/Gridworld"
      GAME_NAME="gridworld"
      ;;
    advisor_audit)
      ENV_ROOT="$REPO_ROOT/AdvisorAudit"
      GAME_NAME="advisor_audit"
      ;;
  esac

  RAW_MODEL_ROOT="$ENV_ROOT/Results/DeceptionMining/$MODEL_TAG_RAW"
  BASE_MODEL_ROOT="$ENV_ROOT/Results/DeceptionMining/$MODEL_TAG_BASE"

  read -r raw_total raw_dec raw_tru raw_unk <<<"$(count_labels_in_root "$RAW_MODEL_ROOT")"
  read -r base_total base_dec base_tru base_unk <<<"$(count_labels_in_root "$BASE_MODEL_ROOT")"

  MODEL_ROOT="$RAW_MODEL_ROOT"
  existing_total=$raw_total
  existing_dec=$raw_dec
  existing_tru=$raw_tru
  existing_unk=$raw_unk
  if (( base_total > raw_total )); then
    MODEL_ROOT="$BASE_MODEL_ROOT"
    existing_total=$base_total
    existing_dec=$base_dec
    existing_tru=$base_tru
    existing_unk=$base_unk
  fi

  mkdir -p "$MODEL_ROOT"

  remaining_dec=$(( TARGET_DECEPTIVE - existing_dec ))
  remaining_tru=$(( TARGET_TRUTHFUL - existing_tru ))
  if (( remaining_dec < 0 )); then remaining_dec=0; fi
  if (( remaining_tru < 0 )); then remaining_tru=0; fi

  need_mining=0
  if [[ "$FORCE_MINE" == "1" ]]; then
    need_mining=1
  else
    if (( remaining_dec > 0 || remaining_tru > 0 )); then
      need_mining=1
    fi
  fi
  if [[ "$SKIP_MINING" == "1" ]]; then
    need_mining=0
  fi

  if [[ "$FORCE_MINE" == "1" ]]; then
    if (( TARGET_DECEPTIVE > 0 )); then remaining_dec=$TARGET_DECEPTIVE; fi
    if (( TARGET_TRUTHFUL > 0 )); then remaining_tru=$TARGET_TRUTHFUL; fi
  fi

  echo "Environment: $ENVIRONMENT"
  echo "Model: $MODEL_NAME"
  echo "Python: $PYTHON_BIN"
  echo "GPUs: ${GPU_IDS_ARRAY[*]}"
  echo "Using mining root: $MODEL_ROOT"
  echo "Existing mined counts: total=$existing_total deceptive=$existing_dec truthful=$existing_tru unknown=$existing_unk"
  echo "Target mined counts: deceptive=$TARGET_DECEPTIVE truthful=$TARGET_TRUTHFUL"

  if [[ "$need_mining" == "1" ]]; then
    mine_label_filter="all"
    if (( remaining_dec > 0 && remaining_tru == 0 )); then
      mine_label_filter="deceptive_only"
    elif (( remaining_tru > 0 && remaining_dec == 0 )); then
      mine_label_filter="truthful_only"
    fi

    per_worker_dec=0
    per_worker_tru=0
    if (( remaining_dec > 0 )); then
      per_worker_dec=$(( (remaining_dec + NUM_WORKERS - 1) / NUM_WORKERS ))
    fi
    if (( remaining_tru > 0 )); then
      per_worker_tru=$(( (remaining_tru + NUM_WORKERS - 1) / NUM_WORKERS ))
    fi

    reasoning_flag="$(guess_reasoning_flag "$MODEL_NAME" "$ENVIRONMENT")"
    mine_run_root="$MODEL_ROOT/$RUN_TAG"
    mkdir -p "$mine_run_root"

    echo "Mining additional data into: $mine_run_root"
    echo "Mining label filter: $mine_label_filter"
    echo "Per-worker targets: deceptive=$per_worker_dec truthful=$per_worker_tru"

    pids=()
    pid_gpus=()
    for idx in "${!GPU_IDS_ARRAY[@]}"; do
      gpu="${GPU_IDS_ARRAY[$idx]}"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        out_dir="$mine_run_root/gpu_$gpu"
        mkdir -p "$out_dir"
        seed=$(( MINE_BASE_SEED + idx * 10000 ))

        if [[ "$ENVIRONMENT" == "advisor_audit" ]]; then
          cmd=(
            "$PYTHON_BIN" "$REPO_ROOT/AdvisorAudit/src/financial_advisor_deception_miner.py"
            --model_name "$MODEL_NAME"
            --output_dir "$out_dir"
            --seed "$seed"
            --max_episodes "$MAX_EPISODES"
            --max_turns "$MAX_TURNS"
            --samples_per_state "$ADVISOR_MINE_SAMPLES_PER_STATE"
            --label_filter "$mine_label_filter"
            --target_deceptive "$per_worker_dec"
            --target_truthful "$per_worker_tru"
            --temperature "$MINE_TEMPERATURE"
            --top_p "$MINE_TOP_P"
            --max_tokens 4096
            --repetition_penalty 1.1
            --max_retries "$MINE_MAX_RETRIES"
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
            --log_every "$LOG_EVERY"
          )
        elif [[ "$ENVIRONMENT" == "gridworld" ]]; then
          cmd=(
            "$PYTHON_BIN" "$REPO_ROOT/src/deception_miner.py"
            --game "$GAME_NAME"
            --model_name "$MODEL_NAME"
            --output_dir "$out_dir"
            --seed "$seed"
            --max_games "$MAX_GAMES"
            --max_turns "$MAX_TURNS"
            --samples_per_state "$GRIDWORLD_MINE_SAMPLES_PER_STATE"
            --label_filter "$mine_label_filter"
            --target_deceptive "$per_worker_dec"
            --target_truthful "$per_worker_tru"
            --temperature "$MINE_TEMPERATURE"
            --top_p "$MINE_TOP_P"
            --max_tokens 10000
            --repetition_penalty 1.2
            --max_retries "$MINE_MAX_RETRIES"
            --log_every "$LOG_EVERY"
          )
        else
          cmd=(
            "$PYTHON_BIN" "$REPO_ROOT/src/deception_miner.py"
            --game "$GAME_NAME"
            --model_name "$MODEL_NAME"
            --output_dir "$out_dir"
            --seed "$seed"
            --max_games "$MAX_GAMES"
            --max_turns "$MAX_TURNS"
            --samples_per_state "$BS_MINE_SAMPLES_PER_STATE"
            --label_filter "$mine_label_filter"
            --target_deceptive "$per_worker_dec"
            --target_truthful "$per_worker_tru"
            --temperature "$MINE_TEMPERATURE"
            --top_p "$MINE_TOP_P"
            --max_tokens 10000
            --repetition_penalty 1.2
            --max_retries "$MINE_MAX_RETRIES"
            --log_every "$LOG_EVERY"
          )
        fi

        if [[ -n "$reasoning_flag" ]]; then
          cmd+=("$reasoning_flag")
        fi
        if [[ "$ENVIRONMENT" == "gridworld" ]]; then
          cmd+=(
            --grid_width "${GRID_WIDTH:-9}"
            --grid_height "${GRID_HEIGHT:-9}"
            --wall_prob "${WALL_PROB:-0.18}"
            --max_tries "${GRID_MAX_TRIES:-200}"
            --max_steps "${GRID_MAX_STEPS:-60}"
            --view_radius "${GRID_VIEW_RADIUS:-2}"
            --history_window "${GRID_HISTORY_WINDOW:-15}"
            --auto_move_explorer
          )
        fi

        "${cmd[@]}" > "$out_dir/run.log" 2>&1
      ) &
      pids+=("$!")
      pid_gpus+=("$gpu")
    done

    failed=0
    for idx in "${!pids[@]}"; do
      if ! wait "${pids[$idx]}"; then
        failed=$((failed + 1))
        gpu="${pid_gpus[$idx]}"
        echo "Mining worker failed on GPU $gpu. Tail of log:"
        tail -n 60 "$mine_run_root/gpu_$gpu/run.log" || true
      fi
    done
    if (( failed > 0 )); then
      echo "$failed mining worker(s) failed." >&2
      exit 1
    fi

    read -r existing_total existing_dec existing_tru existing_unk <<<"$(count_labels_in_root "$MODEL_ROOT")"
    echo "Post-mining counts: total=$existing_total deceptive=$existing_dec truthful=$existing_tru unknown=$existing_unk"
  else
    echo "Skipping mining."
  fi

  if [[ "$MINE_ONLY" == "1" ]]; then
    echo "Mine-only mode enabled; skipping sentence dataset build and localization."
    echo "Mining root: $MODEL_ROOT"
    return
  fi

  PIPELINE_DIR="$DATASET_ROOT/$ENVIRONMENT/$MODEL_TAG_BASE"
  mkdir -p "$PIPELINE_DIR"

  echo "Building sentence dataset into: $PIPELINE_DIR"
  "$PYTHON_BIN" "$REPO_ROOT/src/build_sentence_dataset.py" \
    --input_root "$MODEL_ROOT" \
    --out_dir "$PIPELINE_DIR" \
    --text_field "$TEXT_FIELD" \
    --fallback_text_field "$FALLBACK_TEXT_FIELD" \
    --label_filter "$BUILD_LABEL_FILTER" \
    --target_deceptive "$TARGET_DECEPTIVE" \
    --target_truthful "$TARGET_TRUTHFUL" \
    --include_messages

  EXAMPLES_PATH="$PIPELINE_DIR/examples.jsonl"
  SENTENCES_PATH="$PIPELINE_DIR/sentences.jsonl"
  read -r ds_total ds_dec ds_tru ds_unk <<<"$(count_examples_file "$EXAMPLES_PATH")"
  echo "Built dataset counts: total=$ds_total deceptive=$ds_dec truthful=$ds_tru unknown=$ds_unk"

  if (( ds_dec < TARGET_DECEPTIVE || ds_tru < TARGET_TRUTHFUL )); then
    echo "Dataset build did not reach target counts. Need deceptive=$TARGET_DECEPTIVE truthful=$TARGET_TRUTHFUL." >&2
    exit 1
  fi

  LOCALIZATION_DIR="$PIPELINE_DIR/localization"
  mkdir -p "$LOCALIZATION_DIR"

  echo "Starting multi-GPU localization into: $LOCALIZATION_DIR"
  echo "Localization params: n_samples=$N_SAMPLES temperature=$TEMPERATURE top_p=$TOP_P repetition_penalty=$REPETITION_PENALTY"

  loc_pids=()
  loc_pid_gpus=()
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
        --out_dir "$LOCALIZATION_DIR"
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
        --label_filter all
        --text_field "$TEXT_FIELD"
        --shard_id "$idx"
        --num_shards "$NUM_WORKERS"
        --log_every "$LOG_EVERY"
      )
      if [[ "$OVERWRITE_LOCALIZATION" == "1" ]]; then
        cmd+=(--overwrite)
      fi
      if [[ "$WRITE_JSONL" == "1" ]]; then
        cmd+=(--jsonl_path "$PIPELINE_DIR/$JSONL_BASENAME")
      fi
      "${cmd[@]}" > "$LOCALIZATION_DIR/run_gpu_$gpu.log" 2>&1
    ) &
    loc_pids+=("$!")
    loc_pid_gpus+=("$gpu")
  done

  loc_failed=0
  for idx in "${!loc_pids[@]}"; do
    if ! wait "${loc_pids[$idx]}"; then
      loc_failed=$((loc_failed + 1))
      gpu="${loc_pid_gpus[$idx]}"
      echo "Localization worker failed on GPU $gpu. Tail of log:"
      tail -n 60 "$LOCALIZATION_DIR/run_gpu_$gpu.log" || true
    fi
  done
  if (( loc_failed > 0 )); then
    echo "$loc_failed localization worker(s) failed." >&2
    exit 1
  fi

  echo "Pipeline complete."
  echo "Mining root: $MODEL_ROOT"
  echo "Sentence dataset: $PIPELINE_DIR"
  echo "Localization dir: $LOCALIZATION_DIR"
}

for env_name in "${ENVIRONMENTS_ARRAY[@]}"; do
  echo ""
  echo "================================================================"
  echo "Running targeted localization pipeline for env=$env_name model=$MODEL_NAME"
  echo "================================================================"
  run_single_environment "$env_name"
done
