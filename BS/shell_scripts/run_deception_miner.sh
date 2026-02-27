#!/usr/bin/env bash

###############################################
# Parallel BS deception miner (6 GPUs)
###############################################

set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

hash -r
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: conda env not active after 'conda activate deception'."
    exit 1
fi
PYTHON_BIN="$CONDA_PREFIX/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: expected python not found at $PYTHON_BIN"
    exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Note: inherited VIRTUAL_ENV is set to: $VIRTUAL_ENV"
    echo "Using conda python explicitly: $PYTHON_BIN"
fi

echo "Python in env: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys
print("Python executable:", sys.executable)
try:
    import vllm  # noqa: F401
except Exception as e:
    print("ERROR: could not import vllm in this environment:", repr(e))
    raise
PY

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
LABEL_FILTER="${LABEL_FILTER:-deceptive_only}"
if [[ "$LABEL_FILTER" != "all" && "$LABEL_FILTER" != "deceptive_only" && "$LABEL_FILTER" != "truthful_only" ]]; then
    echo "Invalid LABEL_FILTER=$LABEL_FILTER. Expected one of: all, deceptive_only, truthful_only"
    exit 1
fi
if [[ "$LABEL_FILTER" == "truthful_only" ]]; then
    TARGET_DECEPTIVE=${TARGET_DECEPTIVE:-0}
    TARGET_TRUTHFUL=${TARGET_TRUTHFUL:-1000}
elif [[ "$LABEL_FILTER" == "deceptive_only" ]]; then
    TARGET_DECEPTIVE=${TARGET_DECEPTIVE:-1000}
    TARGET_TRUTHFUL=${TARGET_TRUTHFUL:-0}
else
    TARGET_DECEPTIVE=${TARGET_DECEPTIVE:-0}
    TARGET_TRUTHFUL=${TARGET_TRUTHFUL:-0}
fi

MODEL_TAG="${MODEL_NAME//\//_}"
OUT_BASE="/playpen-ssd/smerrill/deception2/BS/Results/DeceptionMining/${MODEL_TAG}/$(date +%Y-%m-%d)"
mkdir -p "$OUT_BASE"

SCRIPT="/playpen-ssd/smerrill/deception2/src/deception_miner.py"

OUT_DIR="$OUT_BASE/gpu_$CUDA_VISIBLE_DEVICES"
mkdir -p "$OUT_DIR"

echo "Output dir: $OUT_DIR"
echo "Launching worker on GPU $CUDA_VISIBLE_DEVICES"
echo "Label filter: $LABEL_FILTER"
echo ""

"$PYTHON_BIN" "$SCRIPT" \
    --game bs \
    --model_name "$MODEL_NAME" \
    $REASONING \
    --output_dir "$OUT_DIR" \
    --seed "$SEED_BASE" \
    --max_games "$MAX_GAMES" \
    --max_turns "$MAX_TURNS" \
    --label_filter "$LABEL_FILTER" \
    --target_deceptive "$TARGET_DECEPTIVE" \
    --target_truthful "$TARGET_TRUTHFUL" \
    --log_every 25 \
    > "$OUT_DIR/run.log" 2>&1

echo ""
echo "✓ Deception mining complete"
