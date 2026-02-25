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
  --shard_id 0
  --num_shards 1
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

"${CMD[@]}"

echo "Sentence localization complete (per-example JSONs written to $OUT_DIR)."
