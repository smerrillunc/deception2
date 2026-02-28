#!/bin/bash
#SBATCH --job-name=gw_loc_deceptive
#SBATCH --output=gw_loc_deceptive_%j.out
#SBATCH --error=gw_loc_deceptive_%j.err
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
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
N_SAMPLES=50
TEMPERATURE=0.5
TOP_P=0.5
REPETITION_PENALTY=1.2
MAX_NEW_TOKENS=10000
METHOD="adaptive"    # adaptive | full
MODE="prefix"        # prefix | sentence_only
LIMIT=0              # 0 means no limit.
LOG_EVERY=25
# Sharding:
# - NUM_SHARDS is total shard count.
# - SHARD_ID defaults to SLURM_ARRAY_TASK_ID (if using --array), else 0.
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"

# ---------------- End parameters -----------------

module load anaconda
conda activate "$CONDA_ENV"

PROJECT_ROOT="/work/users/s/m/smerrill/deception2"
SRC_ROOT="$PROJECT_ROOT/src"

GAME="bs"
# DATA_DIR="/work/users/s/m/smerrill/deception2/BS/Results/SentencePipeline/v1/DeepSeek-R1-Distill-Qwen-14B_deceptive" # complete
DATA_DIR="/work/users/s/m/smerrill/deception2/BS/Results/SentencePipeline/v1/DeepSeek-R1-Distill-Qwen-14B_truthful" # 33005650
# DATA_DIR="/work/users/s/m/smerrill/deception2/BS/Results/SentencePipeline/v1/DeepSeek-R1-Distill-Qwen-7B_deceptive" # complete
# DATA_DIR="/work/users/s/m/smerrill/deception2/BS/Results/SentencePipeline/v1/DeepSeek-R1-Distill-Qwen-7B_truthful" # running

# GAME="gridworld"
# DATA_DIR="/work/users/s/m/smerrill/deception2/Gridworld/Results/SentencePipeline/v1/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_deceptive" # complete
# DATA_DIR="/work/users/s/m/smerrill/deception2/Gridworld/Results/SentencePipeline/v1/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_truthful" # JOB 33005216


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
  echo "Build sentence data first (deceptive_only) and rerun."
  exit 1
fi

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
)
if [[ -f "$SENTENCES_PATH" ]]; then
  CMD+=(--sentences_path "$SENTENCES_PATH")
fi
if [[ "$LIMIT" -gt 0 ]]; then
  CMD+=(--limit "$LIMIT")
fi

echo "Command to run:"
printf '%q ' "${CMD[@]}"
echo
echo "Running shard $SHARD_ID of $NUM_SHARDS"

"${CMD[@]}"

echo "Sentence localization complete (per-example JSONs written to $OUT_DIR)."
