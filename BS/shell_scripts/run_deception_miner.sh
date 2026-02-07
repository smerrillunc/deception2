#!/usr/bin/env bash

###############################################
# Parallel BS deception miner (6 GPUs)
###############################################

set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

read -p "Enter the GPU ID you want to use (e.g., 0): " GPU

export CUDA_VISIBLE_DEVICES="$GPU"
echo "✓ Using GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# ---------------------------
# Model selection (edit or extend)
# ---------------------------
echo "Select a model:"
echo "  1) deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
echo "  2) deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
echo "  3) deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
echo ""

read -p "Enter model number (1–3): " MODEL_CHOICE

REASONING=""
case "$MODEL_CHOICE" in
    1)
        MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        REASONING="--is_reasoning_model"
        ;;
    2)
        MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        REASONING="--is_reasoning_model"
        ;;
    3)
        MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        REASONING="--is_reasoning_model"
        ;;
    *)
        echo "❌ Invalid model selection: $MODEL_CHOICE"
        exit 1
        ;;
esac

echo "✓ Using model: $MODEL_NAME"
if [[ -n "$REASONING" ]]; then
    echo "✓ REASONING enabled"
fi
echo ""

# ---------------------------
# Run configuration
# ---------------------------
SEED_BASE=0
MAX_GAMES=1000
MAX_TURNS=1000
TARGET_DECEPTIVE=1000   # per GPU

OUT_BASE="/playpen-ssd/smerrill/deception2/BS/Results/DeceptionMining/$(date +%Y-%m-%d)"
mkdir -p "$OUT_BASE"

SCRIPT="/playpen-ssd/smerrill/deception2/BS/src/bs_deception_miner.py"

OUT_DIR="$OUT_BASE/gpu_$CUDA_VISIBLE_DEVICES"
mkdir -p "$OUT_DIR"

echo "Output dir: $OUT_DIR"
echo "Launching worker on GPU $CUDA_VISIBLE_DEVICES"
echo ""

python "$SCRIPT" \
    --model_name "$MODEL_NAME" \
    $REASONING \
    --output_dir "$OUT_DIR" \
    --seed "$SEED_BASE" \
    --max_games "$MAX_GAMES" \
    --max_turns "$MAX_TURNS" \
    --target_deceptive "$TARGET_DECEPTIVE" \
    --log_every 25 \
    > "$OUT_DIR/run.log" 2>&1

echo ""
echo "✓ Deception mining complete"
