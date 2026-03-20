#!/bin/bash
#SBATCH --job-name=sentence_loc
#SBATCH --output=sentence_loc_%A_%a.out
#SBATCH --error=sentence_loc_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40g
#SBATCH --time=6-23:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

set -euo pipefail

# ---------------- User parameters ----------------
CONDA_ENV="deception"
N_SAMPLES=100
TEMPERATURE=0.7
TOP_P=0.9
REPETITION_PENALTY=1.1
MAX_NEW_TOKENS=10000
METHOD="adaptive"    # adaptive | full
MODE="prefix"        # prefix | sentence_only
TEXT_FIELD="action_reasoning"
LIMIT=0              # 0 means no limit.
LOG_EVERY=25
OVERWRITE=0          # 1 => pass --overwrite
WRITE_JSONL=0        # 1 => also write localization.jsonl
JSONL_BASENAME="localization.jsonl"

# Sharding:
# - NUM_SHARDS is total shard count.
# - SHARD_ID defaults to SLURM_ARRAY_TASK_ID (if using --array), else 0.
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"

PROJECT_ROOT="/work/users/s/m/smerrill/deception2"
SRC_ROOT="$PROJECT_ROOT/src"
DATASET_ROOT="$PROJECT_ROOT/DatasetMain"

# Dataset / model selection.
# Examples:
# GAME='advisor_audit'
# MODEL_NAME='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
#
# For instruction models you may want:
# METHOD="full"
# TEXT_FIELD="reasoning"
GAME='bs'   # bs | gridworld | advisor_audit
MODEL_NAME='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# ---------------- End parameters -----------------

module load anaconda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

MODEL_TAIL="${MODEL_NAME##*/}"
DATA_DIR="$DATASET_ROOT/$GAME/$MODEL_TAIL"

if [[ -z "${DATA_DIR:-}" ]]; then
  echo "DATA_DIR is not set. Set DATA_DIR near the top of this script."
  exit 1
fi

EXAMPLES_PATH="$DATA_DIR/examples.jsonl"
SENTENCES_PATH="$DATA_DIR/sentences.jsonl"
OUT_DIR="$DATA_DIR/localization"

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
  echo "Build the DatasetMain dataset first and rerun."
  exit 1
fi

mkdir -p "$OUT_DIR"

CMD=(
  conda run -n "$CONDA_ENV" python "$SRC_ROOT/sentence_localization_batch.py"
  --game "$GAME"
  --examples_path "$EXAMPLES_PATH"
  --model_name "$MODEL_NAME"
  --n_samples "$N_SAMPLES"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --repetition_penalty "$REPETITION_PENALTY"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --method "$METHOD"
  --mode "$MODE"
  --label_filter all
  --shard_id "$SHARD_ID"
  --num_shards "$NUM_SHARDS"
  --log_every "$LOG_EVERY"
  --out_dir "$OUT_DIR"
  --text_field "$TEXT_FIELD"
)
if [[ -f "$SENTENCES_PATH" ]]; then
  CMD+=(--sentences_path "$SENTENCES_PATH")
fi
if [[ "$LIMIT" -gt 0 ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ "$WRITE_JSONL" == "1" ]]; then
  CMD+=(--jsonl_path "$DATA_DIR/$JSONL_BASENAME")
fi

echo "Command to run:"
printf '%q ' "${CMD[@]}"
echo
echo "Running shard $SHARD_ID of $NUM_SHARDS"
echo "Dataset dir: $DATA_DIR"

"${CMD[@]}"

echo "Sentence localization complete (per-example JSONs written to $OUT_DIR)."
