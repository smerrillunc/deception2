#!/usr/bin/env bash

###############################################
# Single-GPU Gridworld deception miner
###############################################

set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
read -r -p "Enter the GPU ID you want to use (e.g., 0): " GPU
export CUDA_VISIBLE_DEVICES="$GPU"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

echo ""
echo "Select a model:"
echo "  1) deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
echo "  2) deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
echo "  3) deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
echo ""

read -r -p "Enter model number (1-3): " MODEL_CHOICE

REASONING_FLAG=""
case "$MODEL_CHOICE" in
    1)
        MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        REASONING_FLAG="--is_reasoning_model"
        ;;
    2)
        MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        REASONING_FLAG="--is_reasoning_model"
        ;;
    3)
        MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        REASONING_FLAG="--is_reasoning_model"
        ;;
    *)
        echo "Invalid model selection: $MODEL_CHOICE"
        exit 1
        ;;
esac

MODEL_TAG="${MODEL_NAME//\//_}"

SEED_BASE=${SEED_BASE:-0}
MAX_GAMES=${MAX_GAMES:-1000}
MAX_TURNS=${MAX_TURNS:-1000}
TARGET_DECEPTIVE=${TARGET_DECEPTIVE:-1000}

GRID_WIDTH=${GRID_WIDTH:-9}
GRID_HEIGHT=${GRID_HEIGHT:-9}
WALL_PROB=${WALL_PROB:-0.18}
MAX_TRIES=${MAX_TRIES:-200}
MAX_STEPS=${MAX_STEPS:-60}
VIEW_RADIUS=${VIEW_RADIUS:-2}
HISTORY_WINDOW=${HISTORY_WINDOW:-15}
AUTO_MOVE_EXPLORER=${AUTO_MOVE_EXPLORER:-1}

if [[ "$AUTO_MOVE_EXPLORER" == "0" ]]; then
    AUTO_MOVE_FLAG="--no-auto_move_explorer"
else
    AUTO_MOVE_FLAG="--auto_move_explorer"
fi

OUT_BASE="/playpen-ssd/smerrill/deception2/Gridworld/Results/DeceptionMining/${MODEL_TAG}/$(date +%Y-%m-%d)"
SCRIPT="/playpen-ssd/smerrill/deception2/src/deception_miner.py"
OUT_DIR="$OUT_BASE/gpu_$CUDA_VISIBLE_DEVICES"
mkdir -p "$OUT_DIR"

echo "Output dir: $OUT_DIR"
echo "Model: $MODEL_NAME"
echo "Running miner..."

CMD=(
    python "$SCRIPT"
    --game gridworld
    --model_name "$MODEL_NAME"
    --output_dir "$OUT_DIR"
    --seed "$SEED_BASE"
    --max_games "$MAX_GAMES"
    --max_turns "$MAX_TURNS"
    --target_deceptive "$TARGET_DECEPTIVE"
    --grid_width "$GRID_WIDTH"
    --grid_height "$GRID_HEIGHT"
    --wall_prob "$WALL_PROB"
    --max_tries "$MAX_TRIES"
    --max_steps "$MAX_STEPS"
    --view_radius "$VIEW_RADIUS"
    --history_window "$HISTORY_WINDOW"
    "$AUTO_MOVE_FLAG"
    --log_every 25
)

if [[ -n "$REASONING_FLAG" ]]; then
    CMD+=("$REASONING_FLAG")
fi

"${CMD[@]}" > "$OUT_DIR/run.log" 2>&1

echo ""
echo "Gridworld deception mining complete."
