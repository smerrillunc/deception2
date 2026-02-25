#!/bin/bash
#SBATCH --job-name=deception_miner
#SBATCH --output=deception_miner_%j.out
#SBATCH --error=deception_miner_%j.err
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
GAME="bs"                   # bs | gridworld
IS_REASONING_MODEL=1        # 1 to pass --is_reasoning_model

TEMPERATURE=0.5
TOP_P=0.5
MAX_TOKENS=10000
REPETITION_PENALTY=1.2
MAX_RETRIES=3
SAMPLES_PER_STATE=10

MAX_GAMES=1000
MAX_TURNS=1000
LABEL_FILTER="all"          # all | deceptive_only | truthful_only
TARGET_DECEPTIVE=0
TARGET_TRUTHFUL=0
SEED=0
LOG_EVERY=50

# BS environment args
NUM_PLAYERS=4
CARDS_PER_PLAYER=5

# Gridworld environment args
GRID_WIDTH=9
GRID_HEIGHT=9
WALL_PROB=0.18
MAX_TRIES=200
MAX_STEPS=60
VIEW_RADIUS=2
HISTORY_WINDOW=15
AUTO_MOVE_EXPLORER=1        # 1 => --auto_move_explorer, 0 => --no-auto_move_explorer

# Set one OUTPUT_DIR.
# OUTPUT_DIR="/work/users/s/m/smerrill/deception2/BS/Results/DeceptionMining/DeepSeek-R1-Distill-Qwen-7B/$(date +%Y-%m-%d)/gpu_0"
# OUTPUT_DIR="/work/users/s/m/smerrill/deception2/Gridworld/Results/DeceptionMining/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B/$(date +%Y-%m-%d)/gpu_0"
OUTPUT_DIR=""
# ---------------- End parameters -----------------

module load anaconda
conda activate "$CONDA_ENV"

PROJECT_ROOT="/work/users/s/m/smerrill/deception2"
SRC_ROOT="$PROJECT_ROOT/src"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR is not set. Set OUTPUT_DIR near the top of this script."
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

CMD=(
  conda run -n "$CONDA_ENV" python "$SRC_ROOT/deception_miner.py"
  --game "$GAME"
  --model_name "$MODEL_NAME"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --max_tokens "$MAX_TOKENS"
  --repetition_penalty "$REPETITION_PENALTY"
  --max_retries "$MAX_RETRIES"
  --samples_per_state "$SAMPLES_PER_STATE"
  --max_games "$MAX_GAMES"
  --max_turns "$MAX_TURNS"
  --label_filter "$LABEL_FILTER"
  --target_deceptive "$TARGET_DECEPTIVE"
  --target_truthful "$TARGET_TRUTHFUL"
  --output_dir "$OUTPUT_DIR"
  --seed "$SEED"
  --log_every "$LOG_EVERY"
)

if [[ "$IS_REASONING_MODEL" == "1" ]]; then
  CMD+=(--is_reasoning_model)
fi

if [[ "$GAME" == "bs" ]]; then
  CMD+=(--num_players "$NUM_PLAYERS" --cards_per_player "$CARDS_PER_PLAYER")
elif [[ "$GAME" == "gridworld" ]]; then
  CMD+=(
    --grid_width "$GRID_WIDTH"
    --grid_height "$GRID_HEIGHT"
    --wall_prob "$WALL_PROB"
    --max_tries "$MAX_TRIES"
    --max_steps "$MAX_STEPS"
    --view_radius "$VIEW_RADIUS"
    --history_window "$HISTORY_WINDOW"
  )
  if [[ "$AUTO_MOVE_EXPLORER" == "1" ]]; then
    CMD+=(--auto_move_explorer)
  else
    CMD+=(--no-auto_move_explorer)
  fi
else
  echo "Invalid GAME=$GAME. Expected bs or gridworld."
  exit 1
fi

echo "Command to run:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Deception mining complete."
