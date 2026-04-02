#!/bin/bash
#SBATCH --job-name=commit_prefix_cpu
#SBATCH --output=commit_prefix_%A_%a.out
#SBATCH --error=commit_prefix_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=2-00:00:00

set -euo pipefail

# ---------------- User parameters ----------------
CONDA_ENV="${CONDA_ENV:-deception}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
RECENT_WINDOW_TOKENS="${RECENT_WINDOW_TOKENS:-128}"
FEATURE_SET="${FEATURE_SET:-core}"
NUM_LAYER_BLOCKS="${NUM_LAYER_BLOCKS:-4}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
WRITE_EVERY_EXAMPLES="${WRITE_EVERY_EXAMPLES:-32}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
OVERWRITE="${OVERWRITE:-0}"
STRICT="${STRICT:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CLEAN_STALE_OUTPUTS="${CLEAN_STALE_OUTPUTS:-1}"
STATE_CACHE_DIR="${STATE_CACHE_DIR:-}"

# Sharding:
# - NUM_SHARDS is the total shard count.
# - SHARD_ID defaults to SLURM_ARRAY_TASK_ID (if using --array), else 0.
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SRC_ROOT="$PROJECT_ROOT/src"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"

# Dataset / environment selection.
GAME="${GAME:-bs}"   # bs | gridworld | advisor_audit | interview | car_sales
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
# ---------------- End parameters -----------------

module load anaconda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

build_job_name() {
  local env_name="$1"
  local model_tail="$2"
  local job_name="commit_prefix_${env_name}_${model_tail}"

  job_name="${job_name//[^[:alnum:]_.-]/_}"
  printf '%s' "${job_name:0:120}"
}

MODEL_TAIL="${MODEL_NAME##*/}"
JOB_NAME="$(build_job_name "$GAME" "$MODEL_TAIL")"
DATA_DIR="${DATA_DIR:-$DATASET_ROOT/$GAME/$MODEL_TAIL}"
LOCALIZATION_DIR="$DATA_DIR/localization"
EXAMPLES_PATH="$DATA_DIR/examples.jsonl"
SHARD_OUT_DIR="${SHARD_OUT_DIR:-$DATA_DIR/commitment_prefix_features_shards}"
OUT_PATH="${OUT_PATH:-$SHARD_OUT_DIR/commitment_prefix_features_shard_${SHARD_ID}_of_${NUM_SHARDS}.parquet}"
TMP_OUT_PATH="${OUT_PATH}.tmp"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  if scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME" >/dev/null 2>&1; then
    echo "SLURM job name: $JOB_NAME"
  else
    echo "Warning: failed to update SLURM job name to $JOB_NAME" >&2
  fi
fi

if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "NUM_SHARDS must be a positive integer. Got: $NUM_SHARDS"
  exit 1
fi
if ! [[ "$SHARD_ID" =~ ^[0-9]+$ ]]; then
  echo "SHARD_ID must be a non-negative integer. Got: $SHARD_ID"
  exit 1
fi
if [[ "$SHARD_ID" -ge "$NUM_SHARDS" ]]; then
  echo "SHARD_ID ($SHARD_ID) must be in [0, NUM_SHARDS) with NUM_SHARDS=$NUM_SHARDS"
  exit 1
fi
if [[ ! -f "$EXAMPLES_PATH" ]]; then
  echo "Missing examples file: $EXAMPLES_PATH"
  exit 1
fi
if [[ ! -d "$LOCALIZATION_DIR" ]]; then
  echo "Missing localization directory: $LOCALIZATION_DIR"
  exit 1
fi

mkdir -p "$SHARD_OUT_DIR"

if [[ "$OVERWRITE" != "1" ]]; then
  if [[ -e "$OUT_PATH" && "$SKIP_EXISTING" == "1" ]]; then
    echo "Shard output already exists; skipping shard $SHARD_ID:"
    echo "  $OUT_PATH"
    exit 0
  fi

  if [[ ! -e "$OUT_PATH" && -e "$TMP_OUT_PATH" ]]; then
    if [[ "$CLEAN_STALE_OUTPUTS" == "1" ]]; then
      echo "Removing stale temporary shard output before rerun:"
      echo "  $TMP_OUT_PATH"
      rm -f "$TMP_OUT_PATH"
    else
      echo "Temporary shard output exists and cleanup is disabled:"
      echo "  $TMP_OUT_PATH"
      echo "Set CLEAN_STALE_OUTPUTS=1 or delete it manually before rerunning."
      exit 1
    fi
  fi
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false

CMD=(
  python "$SRC_ROOT/commitment_prefix_features.py"
  "$DATA_DIR"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --recent-window-tokens "$RECENT_WINDOW_TOKENS"
  --feature-set "$FEATURE_SET"
  --num-layer-blocks "$NUM_LAYER_BLOCKS"
  --write-every-examples "$WRITE_EVERY_EXAMPLES"
  --progress-every "$PROGRESS_EVERY"
  --num-shards "$NUM_SHARDS"
  --shard-id "$SHARD_ID"
  --output "$OUT_PATH"
)
if [[ "$MAX_EXAMPLES" -gt 0 ]]; then
  CMD+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ -n "$STATE_CACHE_DIR" ]]; then
  CMD+=(--state-cache-dir "$STATE_CACHE_DIR")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ "$STRICT" == "1" ]]; then
  CMD+=(--strict)
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi

echo "Command to run:"
printf '%q ' "${CMD[@]}"
echo
echo "Running shard $SHARD_ID of $NUM_SHARDS"
echo "Dataset dir: $DATA_DIR"
echo "Localization dir: $LOCALIZATION_DIR"
echo "Shard output: $OUT_PATH"
echo "CPU threads per task: $THREADS"

"${CMD[@]}"

echo "commitment_prefix_features shard complete."
