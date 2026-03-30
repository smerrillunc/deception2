#!/bin/bash
#SBATCH --job-name=attention_feat2_cpu
#SBATCH --output=attention_feat2_%A_%a.out
#SBATCH --error=attention_feat2_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=2-00:00:00

set -euo pipefail

# ---------------- User parameters ----------------
# For the 14B model, raise --mem near the top of this file before submitting.
CONDA_ENV="deception"
DEVICE="cpu"
DTYPE="float32"
ATTN_IMPLEMENTATION="eager"
RECENT_WINDOW_TOKENS=64
MAX_EXAMPLES=0
WRITE_EVERY_EXAMPLES=32
PROGRESS_EVERY=10
OVERWRITE=0
STRICT=0
TRUST_REMOTE_CODE=0

# Sharding:
# - NUM_SHARDS is the total shard count.
# - SHARD_ID defaults to SLURM_ARRAY_TASK_ID (if using --array), else 0.
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"

PROJECT_ROOT="${PROJECT_ROOT:-/work/users/s/m/smerrill/deception2}"
SRC_ROOT="$PROJECT_ROOT/src"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"

# Dataset / environment selection.
GAME='gridworld'   # advisor_audit | bs | gridworld | interview | car_sales
MODEL_NAME='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# ---------------- End parameters -----------------

module load anaconda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

build_job_name() {
  local env_name="$1"
  local model_tail="$2"
  local job_name="attn_feat2_${env_name}_${model_tail}"

  job_name="${job_name//[^[:alnum:]_.-]/_}"
  printf '%s' "${job_name:0:120}"
}

MODEL_TAIL="${MODEL_NAME##*/}"
JOB_NAME="$(build_job_name "$GAME" "$MODEL_TAIL")"
DATA_DIR="${DATA_DIR:-$DATASET_ROOT/$GAME/$MODEL_TAIL}"
SHARD_OUT_DIR="${SHARD_OUT_DIR:-$DATA_DIR/attention_features2_shards}"
OUT_PATH="${OUT_PATH:-$SHARD_OUT_DIR/attention_features2_shard_${SHARD_ID}_of_${NUM_SHARDS}.parquet}"
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
if [[ ! -d "$DATA_DIR/localization" ]]; then
  echo "Missing localization directory: $DATA_DIR/localization"
  exit 1
fi

mkdir -p "$SHARD_OUT_DIR"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false

CMD=(
  python "$SRC_ROOT/attention_features2.py"
  "$DATA_DIR"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --recent-window-tokens "$RECENT_WINDOW_TOKENS"
  --write-every-examples "$WRITE_EVERY_EXAMPLES"
  --progress-every "$PROGRESS_EVERY"
  --num-shards "$NUM_SHARDS"
  --shard-id "$SHARD_ID"
  --output "$OUT_PATH"
)
if [[ "$MAX_EXAMPLES" -gt 0 ]]; then
  CMD+=(--max-examples "$MAX_EXAMPLES")
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
echo "Shard output: $OUT_PATH"
echo "CPU threads per task: $THREADS"

"${CMD[@]}"

echo "attention_features2 shard complete."
