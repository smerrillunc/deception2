#!/bin/bash
#SBATCH --job-name=prefix_feat_act_cpu
#SBATCH --output=prefix_feat_act_%A_%a.out
#SBATCH --error=prefix_feat_act_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=2-00:00:00

set -euo pipefail

# ---------------- User parameters ----------------
# For larger models on CPU, you may want to raise --mem above.
PYTHON_BIN="${PYTHON_BIN:-/work/users/s/m/smerrill/.conda/envs/deception/bin/python}"
DEVICE="cpu"
DTYPE="bfloat16"          # auto | float32 | float16 | bfloat16
ATTN_IMPLEMENTATION="eager"
RECENT_WINDOW_TOKENS=64
NUM_PREFIX_SENTENCES=5
COMPRESSION="lzf"         # lzf | gzip | none
GZIP_LEVEL=4
MAX_EXAMPLES=0
WRITE_EVERY_EXAMPLES=32
PROGRESS_EVERY=10
OVERWRITE=0
STRICT=0
TRUST_REMOTE_CODE=0
SKIP_EXISTING=1
CLEAN_PARTIAL_OUTPUTS=1

# Sharding:
# - NUM_SHARDS is the total shard count.
# - SHARD_ID defaults to SLURM_ARRAY_TASK_ID (if using --array), else 0.
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"

PROJECT_ROOT="${PROJECT_ROOT:-/work/users/s/m/smerrill/deception2}"
SRC_ROOT="$PROJECT_ROOT/src"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"

# Dataset / environment selection.
GAME='interview'   # advisor_audit | bs | gridworld | interview | car_sales
MODEL_NAME='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# ---------------- End parameters -----------------

build_job_name() {
  local env_name="$1"
  local model_tail="$2"
  local job_name="prefix_feat_act_${env_name}_${model_tail}"
  job_name="${job_name//[^[:alnum:]_.-]/_}"
  printf '%s' "${job_name:0:120}"
}

MODEL_TAIL="${MODEL_NAME##*/}"
JOB_NAME="$(build_job_name "$GAME" "$MODEL_TAIL")"
DATA_DIR="${DATA_DIR:-$DATASET_ROOT/$GAME/$MODEL_TAIL}"
FEATURE_SHARD_OUT_DIR="${FEATURE_SHARD_OUT_DIR:-$DATA_DIR/prefix_deception_feature_shards}"
ACTIVATION_SHARD_OUT_DIR="${ACTIVATION_SHARD_OUT_DIR:-$DATA_DIR/prefix_deception_activation_shards}"
FEATURE_OUT_PATH="${FEATURE_OUT_PATH:-$FEATURE_SHARD_OUT_DIR/prefix_deception_features_shard_${SHARD_ID}_of_${NUM_SHARDS}.parquet}"
ACTIVATION_OUT_PATH="${ACTIVATION_OUT_PATH:-$ACTIVATION_SHARD_OUT_DIR/prefix_deception_activations_shard_${SHARD_ID}_of_${NUM_SHARDS}.h5}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  if scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME" >/dev/null 2>&1; then
    echo "SLURM job name: $JOB_NAME"
  else
    echo "Warning: failed to update SLURM job name to $JOB_NAME" >&2
  fi
fi

if ! [[ "$NUM_SHARDS" =~ ^[0-9]+$ ]] || [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "NUM_SHARDS must be a positive integer. Got: $NUM_SHARDS" >&2
  exit 1
fi
if ! [[ "$SHARD_ID" =~ ^[0-9]+$ ]]; then
  echo "SHARD_ID must be a non-negative integer. Got: $SHARD_ID" >&2
  exit 1
fi
if [[ "$SHARD_ID" -ge "$NUM_SHARDS" ]]; then
  echo "SHARD_ID ($SHARD_ID) must be in [0, NUM_SHARDS) with NUM_SHARDS=$NUM_SHARDS" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$SRC_ROOT/deception_prefix_feature_and_activation_extractor.py" ]]; then
  echo "Missing script: $SRC_ROOT/deception_prefix_feature_and_activation_extractor.py" >&2
  exit 1
fi
if [[ ! -d "$DATA_DIR/localization" ]]; then
  echo "Missing localization directory: $DATA_DIR/localization" >&2
  exit 1
fi

mkdir -p "$FEATURE_SHARD_OUT_DIR" "$ACTIVATION_SHARD_OUT_DIR"

feature_exists=0
activation_exists=0
if [[ -e "$FEATURE_OUT_PATH" ]]; then
  feature_exists=1
fi
if [[ -e "$ACTIVATION_OUT_PATH" ]]; then
  activation_exists=1
fi

if [[ "$OVERWRITE" != "1" ]]; then
  if [[ "$feature_exists" == "1" && "$activation_exists" == "1" && "$SKIP_EXISTING" == "1" ]]; then
    echo "Both shard outputs already exist; skipping shard $SHARD_ID"
    echo "  Feature:    $FEATURE_OUT_PATH"
    echo "  Activation: $ACTIVATION_OUT_PATH"
    exit 0
  fi

  if [[ "$feature_exists" != "$activation_exists" ]]; then
    if [[ "$CLEAN_PARTIAL_OUTPUTS" == "1" ]]; then
      echo "Detected partial shard outputs; removing existing partial files before rerun"
      [[ -e "$FEATURE_OUT_PATH" ]] && rm -f "$FEATURE_OUT_PATH"
      [[ -e "$ACTIVATION_OUT_PATH" ]] && rm -f "$ACTIVATION_OUT_PATH"
    else
      echo "Detected partial shard outputs for shard $SHARD_ID:" >&2
      echo "  Feature exists:    $feature_exists" >&2
      echo "  Activation exists: $activation_exists" >&2
      echo "Set CLEAN_PARTIAL_OUTPUTS=1 or delete the partial outputs manually." >&2
      exit 1
    fi
  fi
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false

echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -c "import sys, torch; print('python=', sys.executable); print('torch=', torch.__version__)"

CMD=(
  "$PYTHON_BIN" "$SRC_ROOT/deception_prefix_feature_and_activation_extractor.py"
  "$DATA_DIR"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --recent-window-tokens "$RECENT_WINDOW_TOKENS"
  --num-prefix-sentences "$NUM_PREFIX_SENTENCES"
  --write-every-examples "$WRITE_EVERY_EXAMPLES"
  --progress-every "$PROGRESS_EVERY"
  --num-shards "$NUM_SHARDS"
  --shard-id "$SHARD_ID"
  --feature-output "$FEATURE_OUT_PATH"
  --activation-output "$ACTIVATION_OUT_PATH"
  --compression "$COMPRESSION"
  --gzip-level "$GZIP_LEVEL"
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

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
echo "Running shard $SHARD_ID of $NUM_SHARDS"
echo "Dataset dir: $DATA_DIR"
echo "Feature output: $FEATURE_OUT_PATH"
echo "Activation output: $ACTIVATION_OUT_PATH"
echo "CPU threads per task: $THREADS"

"${CMD[@]}"

echo "prefix feature + activation shard complete."
