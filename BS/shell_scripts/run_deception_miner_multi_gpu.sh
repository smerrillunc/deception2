#!/usr/bin/env bash
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

GPU_IDS=(${GPU_IDS:-2 3 4 5 6 7})
NUM_SHARDS=${NUM_SHARDS:-6}

if [[ ${#GPU_IDS[@]} -ne $NUM_SHARDS ]]; then
  echo "GPU_IDS count (${#GPU_IDS[@]}) must equal NUM_SHARDS ($NUM_SHARDS)."
  exit 1
fi

#MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
MODEL_NAME="${MODEL_NAME:-mistralai/Ministral-3-8B-Reasoning-2512}"
REASONING_FLAG="${REASONING_FLAG:---is_reasoning_model}"
MODEL_TAG="${MODEL_NAME//\//_}"

SEED_BASE=${SEED_BASE:-0}
MAX_GAMES=${MAX_GAMES:-1000}
MAX_TURNS=${MAX_TURNS:-1000}
LABEL_FILTER="${LABEL_FILTER:-all}"
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

OUT_BASE="/playpen-ssd/smerrill/deception2/BS/Results/DeceptionMining/${MODEL_TAG}/$(date +%Y-%m-%d)"
mkdir -p "$OUT_BASE"

SCRIPT="/playpen-ssd/smerrill/deception2/src/deception_miner.py"

echo "Launching $NUM_SHARDS miners across GPUs: ${GPU_IDS[*]}"
echo "Model: $MODEL_NAME"
echo "Label filter: $LABEL_FILTER"

PIDS=()
PID_GPUS=()

for i in "${!GPU_IDS[@]}"; do
  GPU=${GPU_IDS[$i]}
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    OUT_DIR="$OUT_BASE/gpu_$GPU"
    mkdir -p "$OUT_DIR"
    SEED=$((SEED_BASE + i * 10000))

    "$PYTHON_BIN" "$SCRIPT" \
      --game bs \
      --model_name "$MODEL_NAME" \
      $REASONING_FLAG \
      --output_dir "$OUT_DIR" \
      --seed "$SEED" \
      --max_games "$MAX_GAMES" \
      --max_turns "$MAX_TURNS" \
      --label_filter "$LABEL_FILTER" \
      --target_deceptive "$TARGET_DECEPTIVE" \
      --target_truthful "$TARGET_TRUTHFUL" \
      --log_every 25 \
      > "$OUT_DIR/run.log" 2>&1
  ) &
  PIDS+=("$!")
  PID_GPUS+=("$GPU")
done

FAILED=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  gpu="${PID_GPUS[$i]}"
  out_dir="$OUT_BASE/gpu_$gpu"
  if wait "$pid"; then
    echo "✓ GPU $gpu finished successfully."
  else
    FAILED=$((FAILED + 1))
    echo "✗ GPU $gpu failed. See: $out_dir/run.log"
    echo "--- tail($out_dir/run.log) ---"
    tail -n 60 "$out_dir/run.log" || true
    echo "------------------------------"
  fi
done

if [[ "$FAILED" -gt 0 ]]; then
  echo "✗ $FAILED miner(s) failed."
  exit 1
fi

echo "✓ All miners complete."
