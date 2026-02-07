#!/usr/bin/env bash
set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

GPU_IDS=(${GPU_IDS:-2 3 4 5 6 7})
NUM_SHARDS=${NUM_SHARDS:-6}

if [[ ${#GPU_IDS[@]} -ne $NUM_SHARDS ]]; then
  echo "GPU_IDS count (${#GPU_IDS[@]}) must equal NUM_SHARDS ($NUM_SHARDS)."
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
REASONING_FLAG="${REASONING_FLAG:---is_reasoning_model}"

SEED_BASE=${SEED_BASE:-0}
MAX_GAMES=${MAX_GAMES:-1000}
MAX_TURNS=${MAX_TURNS:-1000}
TARGET_DECEPTIVE=${TARGET_DECEPTIVE:-1000}

OUT_BASE="/playpen-ssd/smerrill/deception2/BS/Results/DeceptionMining/$(date +%Y-%m-%d)"
mkdir -p "$OUT_BASE"

SCRIPT="/playpen-ssd/smerrill/deception2/BS/src/bs_deception_miner.py"

echo "Launching $NUM_SHARDS miners across GPUs: ${GPU_IDS[*]}"
echo "Model: $MODEL_NAME"

for i in "${!GPU_IDS[@]}"; do
  GPU=${GPU_IDS[$i]}
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    OUT_DIR="$OUT_BASE/gpu_$GPU"
    mkdir -p "$OUT_DIR"
    SEED=$((SEED_BASE + i * 10000))

    python "$SCRIPT" \
      --model_name "$MODEL_NAME" \
      $REASONING_FLAG \
      --output_dir "$OUT_DIR" \
      --seed "$SEED" \
      --max_games "$MAX_GAMES" \
      --max_turns "$MAX_TURNS" \
      --target_deceptive "$TARGET_DECEPTIVE" \
      --log_every 25 \
      > "$OUT_DIR/run.log" 2>&1
  ) &
done

wait
echo "✓ All miners complete."
