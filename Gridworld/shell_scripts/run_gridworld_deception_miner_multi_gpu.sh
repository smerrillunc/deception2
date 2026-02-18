#!/usr/bin/env bash
set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

GPU_IDS=(${GPU_IDS:-2 3 4 5 6 7})
NUM_SHARDS=${NUM_SHARDS:-${#GPU_IDS[@]}}

if [[ ${#GPU_IDS[@]} -lt $NUM_SHARDS ]]; then
  echo "GPU_IDS count (${#GPU_IDS[@]}) must be >= NUM_SHARDS ($NUM_SHARDS)."
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
REASONING_FLAG="${REASONING_FLAG:---is_reasoning_model}"
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
mkdir -p "$OUT_BASE"

echo "Launching $NUM_SHARDS miners across GPUs: ${GPU_IDS[*]:0:$NUM_SHARDS}"
echo "Model: $MODEL_NAME"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  GPU=${GPU_IDS[$i]}
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    OUT_DIR="$OUT_BASE/gpu_$GPU"
    mkdir -p "$OUT_DIR"
    SEED=$((SEED_BASE + i * 10000))

    CMD=(
      python "$SCRIPT"
      --game gridworld
      --model_name "$MODEL_NAME"
      --output_dir "$OUT_DIR"
      --seed "$SEED"
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
  ) &
done

wait
echo "All gridworld miners complete."
