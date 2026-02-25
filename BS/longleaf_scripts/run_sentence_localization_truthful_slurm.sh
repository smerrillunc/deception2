#!/bin/bash
#SBATCH --job-name=bs_loc_truthful
#SBATCH --output=bs_loc_truthful_%j.out
#SBATCH --error=bs_loc_truthful_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=23:00:00
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

set -euo pipefail

# ---------------- User parameters ----------------
CONDA_ENV="deception"
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TRUTHFUL_LIMIT=3000
N_SAMPLES=50
TEMPERATURE=0.5
TOP_P=0.5
REPETITION_PENALTY=1.2
MAX_NEW_TOKENS=10000
METHOD="adaptive"    # adaptive | full
MODE="prefix"        # prefix | sentence_only
SHARD_ID=0
NUM_SHARDS=1
LIMIT=0              # 0 means no limit.
LOG_EVERY=25
DATA_DIR_OVERRIDE="" # Optional absolute data directory.
# ---------------- End parameters -----------------

module load anaconda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$ENV_ROOT/.." && pwd)"
SRC_ROOT="$PROJECT_ROOT/src"
SENTENCE_ROOT="$ENV_ROOT/Results/SentencePipeline/v1"

MODEL_TAG_BASE="${MODEL_NAME##*/}"
MODEL_TAG_RAW="${MODEL_NAME//\//_}"

if [[ -n "$DATA_DIR_OVERRIDE" ]]; then
  DATA_DIR="$DATA_DIR_OVERRIDE"
else
  CANDIDATES=(
    "$SENTENCE_ROOT/${MODEL_TAG_BASE}_truthful_${TRUTHFUL_LIMIT}"
    "$SENTENCE_ROOT/${MODEL_TAG_RAW}_truthful_${TRUTHFUL_LIMIT}"
    "$SENTENCE_ROOT/${MODEL_TAG_BASE}"
    "$SENTENCE_ROOT/${MODEL_TAG_RAW}"
  )
  DATA_DIR="${CANDIDATES[0]}"
  for cand in "${CANDIDATES[@]}"; do
    if [[ -d "$cand" ]]; then
      DATA_DIR="$cand"
      break
    fi
  done
fi

EXAMPLES_PATH="$DATA_DIR/examples.jsonl"
SENTENCES_PATH="$DATA_DIR/sentences.jsonl"
OUT_DIR="$DATA_DIR/localization_truthful"
JSONL_PATH="$DATA_DIR/localization_truthful.jsonl"

if [[ ! -f "$EXAMPLES_PATH" ]]; then
  echo "Missing examples file: $EXAMPLES_PATH"
  echo "Build sentence data first (truthful_only) and rerun."
  exit 1
fi

mkdir -p "$OUT_DIR"

CMD=(
  conda run -n "$CONDA_ENV" python "$SRC_ROOT/sentence_localization_batch.py"
  --game bs
  --examples_path "$EXAMPLES_PATH"
  --model_name "$MODEL_NAME"
  --jsonl_path "$JSONL_PATH"
  --n_samples "$N_SAMPLES"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --repetition_penalty "$REPETITION_PENALTY"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --method "$METHOD"
  --mode "$MODE"
  --label_filter truthful_only
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

echo "MODEL_NAME: $MODEL_NAME"
echo "DATA_DIR: $DATA_DIR"
echo "OUT_DIR: $OUT_DIR"
echo "JSONL_PATH: $JSONL_PATH"
"${CMD[@]}"

echo "BS truthful localization complete."
