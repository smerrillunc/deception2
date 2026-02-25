#!/bin/bash
#SBATCH --job-name=bs_miner
#SBATCH --output=bs_miner_%j.out
#SBATCH --error=bs_miner_%j.err
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
IS_REASONING_MODEL=1
SEED=0
MAX_GAMES=1000
MAX_TURNS=1000
LABEL_FILTER="deceptive_only"  # all | deceptive_only | truthful_only
TARGET_PER_LABEL=1000
LOG_EVERY=25
DATE_TAG="$(date +%Y-%m-%d)"
OUT_DIR_OVERRIDE=""            # Optional absolute output directory.
# ---------------- End parameters -----------------

case "$LABEL_FILTER" in
  truthful_only)
    TARGET_DECEPTIVE=0
    TARGET_TRUTHFUL="$TARGET_PER_LABEL"
    LABEL_TAG="truthful"
    ;;
  deceptive_only)
    TARGET_DECEPTIVE="$TARGET_PER_LABEL"
    TARGET_TRUTHFUL=0
    LABEL_TAG="deceptive"
    ;;
  all)
    TARGET_DECEPTIVE=0
    TARGET_TRUTHFUL=0
    LABEL_TAG="all"
    ;;
  *)
    echo "Invalid LABEL_FILTER=$LABEL_FILTER. Expected one of: all, deceptive_only, truthful_only"
    exit 1
    ;;
esac

module load anaconda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$ENV_ROOT/.." && pwd)"
SRC_ROOT="$PROJECT_ROOT/src"
RESULTS_ROOT="$ENV_ROOT/Results"
MINING_ROOT="$RESULTS_ROOT/DeceptionMining"
MODEL_TAG="${MODEL_NAME//\//_}"

if [[ -n "$OUT_DIR_OVERRIDE" ]]; then
  OUT_DIR="$OUT_DIR_OVERRIDE"
else
  JOB_TAG="${SLURM_JOB_ID:-manual}"
  OUT_DIR="$MINING_ROOT/$MODEL_TAG/$DATE_TAG/${LABEL_TAG}_job_${JOB_TAG}"
fi
mkdir -p "$OUT_DIR"

CMD=(
  conda run -n "$CONDA_ENV" python "$SRC_ROOT/deception_miner.py"
  --game bs
  --model_name "$MODEL_NAME"
  --output_dir "$OUT_DIR"
  --seed "$SEED"
  --max_games "$MAX_GAMES"
  --max_turns "$MAX_TURNS"
  --label_filter "$LABEL_FILTER"
  --target_deceptive "$TARGET_DECEPTIVE"
  --target_truthful "$TARGET_TRUTHFUL"
  --log_every "$LOG_EVERY"
)
if [[ "$IS_REASONING_MODEL" == "1" ]]; then
  CMD+=(--is_reasoning_model)
fi

echo "MODEL_NAME: $MODEL_NAME"
echo "LABEL_FILTER: $LABEL_FILTER"
echo "OUT_DIR: $OUT_DIR"
"${CMD[@]}" | tee "$OUT_DIR/run.log"

echo "BS deception mining complete."
