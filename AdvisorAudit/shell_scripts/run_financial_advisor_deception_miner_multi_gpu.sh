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
echo "Python in env: $PYTHON_BIN"

# Use exactly 6 workers by default. Override with:
# GPU_IDS="2 3 4 5 6 7" NUM_WORKERS=6 ./run_financial_advisor_deception_miner_multi_gpu.sh
GPU_IDS=(${GPU_IDS:-0 1 2 3 4 5})
NUM_WORKERS=${NUM_WORKERS:-6}

if [[ ${#GPU_IDS[@]} -lt $NUM_WORKERS ]]; then
  echo "GPU_IDS count (${#GPU_IDS[@]}) must be >= NUM_WORKERS ($NUM_WORKERS)."
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
MODEL_TAG="${MODEL_NAME//\//_}"
SCENARIO_NAME="${SCENARIO_NAME:-retirement_conservative}"
REASONING_FLAG="${REASONING_FLAG:---is_reasoning_model}"

SEED_BASE=${SEED_BASE:-0}
MAX_EPISODES=${MAX_EPISODES:-1000}
MAX_TURNS=${MAX_TURNS:-1000}
SAMPLES_PER_STATE=${SAMPLES_PER_STATE:-64}
TARGET_DECEPTIVE=${TARGET_DECEPTIVE:-1000}
TARGET_TRUTHFUL=${TARGET_TRUTHFUL:-1000}
LABEL_FILTER="${LABEL_FILTER:-all}"

if [[ "$LABEL_FILTER" != "all" && "$LABEL_FILTER" != "deceptive_only" && "$LABEL_FILTER" != "truthful_only" ]]; then
  echo "Invalid LABEL_FILTER=$LABEL_FILTER. Expected one of: all, deceptive_only, truthful_only"
  exit 1
fi

if [[ "$LABEL_FILTER" == "truthful_only" ]]; then
  TARGET_DECEPTIVE=0
elif [[ "$LABEL_FILTER" == "deceptive_only" ]]; then
  TARGET_TRUTHFUL=0
fi

MONTHS="${MONTHS:-January,February,March,April,May,June}"
INITIAL_PORTFOLIO_VALUE=${INITIAL_PORTFOLIO_VALUE:-100000.0}
SECOND_OPINION_ACCURACY=${SECOND_OPINION_ACCURACY:-0.90}
AUDIT_PENALTY_FOR_BAD_ADVICE=${AUDIT_PENALTY_FOR_BAD_ADVICE:-600}
DECEPTION_PRESSURE_PROB=${DECEPTION_PRESSURE_PROB:-0.25}
HISTORY_WINDOW=${HISTORY_WINDOW:-10}
TEMPERATURE=${TEMPERATURE:-0.5}
TOP_P=${TOP_P:-0.5}
MAX_TOKENS=${MAX_TOKENS:-4096}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.1}
MAX_RETRIES=${MAX_RETRIES:-3}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

OUT_BASE="/playpen-ssd/smerrill/deception2/AdvisorAudit/Results/DeceptionMining/${MODEL_TAG}/$(date +%Y-%m-%d)"
SCRIPT="/playpen-ssd/smerrill/deception2/AdvisorAudit/src/financial_advisor_deception_miner.py"
mkdir -p "$OUT_BASE"

echo "Launching $NUM_WORKERS advisor-audit miners across GPUs: ${GPU_IDS[*]:0:$NUM_WORKERS}"
echo "Model: $MODEL_NAME"
echo "Scenario: $SCENARIO_NAME"
echo "Label filter: $LABEL_FILTER"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
  GPU=${GPU_IDS[$i]}
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    OUT_DIR="$OUT_BASE/gpu_$GPU"
    mkdir -p "$OUT_DIR"
    SEED=$((SEED_BASE + i * 10000))

    CMD=(
      "$PYTHON_BIN" "$SCRIPT"
      --model_name "$MODEL_NAME"
      --scenario_name "$SCENARIO_NAME"
      --output_dir "$OUT_DIR"
      --seed "$SEED"
      --max_episodes "$MAX_EPISODES"
      --max_turns "$MAX_TURNS"
      --samples_per_state "$SAMPLES_PER_STATE"
      --label_filter "$LABEL_FILTER"
      --target_deceptive "$TARGET_DECEPTIVE"
      --target_truthful "$TARGET_TRUTHFUL"
      --months "$MONTHS"
      --initial_portfolio_value "$INITIAL_PORTFOLIO_VALUE"
      --second_opinion_accuracy "$SECOND_OPINION_ACCURACY"
      --audit_penalty_for_bad_advice "$AUDIT_PENALTY_FOR_BAD_ADVICE"
      --deception_pressure_prob "$DECEPTION_PRESSURE_PROB"
      --history_window "$HISTORY_WINDOW"
      --temperature "$TEMPERATURE"
      --top_p "$TOP_P"
      --max_tokens "$MAX_TOKENS"
      --repetition_penalty "$REPETITION_PENALTY"
      --max_retries "$MAX_RETRIES"
      --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
      --log_every 25
    )

    if [[ -n "$REASONING_FLAG" ]]; then
      CMD+=("$REASONING_FLAG")
    fi

    "${CMD[@]}" > "$OUT_DIR/run.log" 2>&1
  ) &
done

wait
echo "All advisor-audit miners complete."
