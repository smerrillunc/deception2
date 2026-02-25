#!/bin/bash
#SBATCH --job-name=gw_loc_deceptive
#SBATCH --output=gw_loc_deceptive_%j.out
#SBATCH --error=gw_loc_deceptive_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=23:00:00
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# ---------------- User parameters ----------------
CONDA_ENV="deception"
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
GAME="gridworld" # gridworld | bs
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

PROJECT_ROOT = "/work/users/s/m/smerrill/deception2"
SRC_ROOT="$PROJECT_ROOT/src"

# DATA_DIR="/work/users/s/m/smerrill/deception2/BS/Results/SentencePipeline/v1/DeepSeek-R1-Distill-Qwen-14B_deceptive" # complete
# DATA_DIR="/work/users/s/m/smerrill/deception2/BS/Results/SentencePipeline/v1/DeepSeek-R1-Distill-Qwen-14B_truthful" 
# DATA_DIR="/work/users/s/m/smerrill/deception2/BS/Results/SentencePipeline/v1/DeepSeek-R1-Distill-Qwen-7B_deceptive" # complete
# DATA_DIR="/work/users/s/m/smerrill/deception2/BS/Results/SentencePipeline/v1/DeepSeek-R1-Distill-Qwen-7B_truthful" # running
# DATA_DIR="/work/users/s/m/smerrill/deception2/BS/Results/SentencePipeline/v1/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_deceptive" # complete
# DATA_DIR="/work/users/s/m/smerrill/deception2/BS/Results/SentencePipeline/v1/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_truthful"


EXAMPLES_PATH="$DATA_DIR/examples.jsonl"
SENTENCES_PATH="$DATA_DIR/sentences.jsonl"
OUT_DIR="$DATA_DIR/localization_deceptive"
JSONL_PATH="$DATA_DIR/localization_deceptive.jsonl"

if [[ ! -f "$EXAMPLES_PATH" ]]; then
  echo "Missing examples file: $EXAMPLES_PATH"
  echo "Build sentence data first (deceptive_only) and rerun."
  exit 1
fi

conda run -n "$CONDA_ENV" python "$SRC_ROOT/sentence_localization_batch.py" \
  --game "$GAME" \
  --examples_path "$EXAMPLES_PATH" \
  --model_name "$MODEL_NAME" \
  --jsonl_path "$JSONL_PATH" \
  --n_samples "$N_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --repetition_penalty "$REPETITION_PENALTY" \
  --max_new_tokens "$MAX_NEW_TOKENS" \ 
  --method "$METHOD" \ 
  --mode "$MODE" \ 
  --label_filter deceptive_only \ 
  --shard_id 0 \ 
  --num_shards 1 \ 
  --log_every "$LOG_EVERY" \ 
  --out_dir "$OUT_DIR" \ 

echo "Gridworld deceptive localization complete."
