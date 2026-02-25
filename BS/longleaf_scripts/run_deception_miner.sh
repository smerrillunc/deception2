#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/longleaf_common.sh"
activate_deception_env

select_single_gpu
select_model_name

REASONING_FLAG="${REASONING_FLAG:---is_reasoning_model}"
SEED_BASE="${SEED_BASE:-0}"
MAX_GAMES="${MAX_GAMES:-1000}"
MAX_TURNS="${MAX_TURNS:-1000}"
LABEL_FILTER="${LABEL_FILTER:-deceptive_only}"
validate_label_filter "$LABEL_FILTER"
set_target_counts_from_label_filter "$LABEL_FILTER" "1000"

MODEL_TAG="${MODEL_NAME//\//_}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
OUT_BASE="${OUT_BASE:-$MINING_ROOT/$MODEL_TAG/$DATE_TAG}"
OUT_DIR="${OUT_DIR:-$OUT_BASE/gpu_${CUDA_VISIBLE_DEVICES}}"
SCRIPT_PATH="$SRC_ROOT/deception_miner.py"

mkdir -p "$OUT_DIR"

CMD=(
  "$PYTHON_BIN" "$SCRIPT_PATH"
  --game bs
  --model_name "$MODEL_NAME"
  --output_dir "$OUT_DIR"
  --seed "$SEED_BASE"
  --max_games "$MAX_GAMES"
  --max_turns "$MAX_TURNS"
  --label_filter "$LABEL_FILTER"
  --target_deceptive "$TARGET_DECEPTIVE"
  --target_truthful "$TARGET_TRUTHFUL"
  --log_every "${LOG_EVERY:-25}"
)
if [[ -n "$REASONING_FLAG" ]]; then
  CMD+=("$REASONING_FLAG")
fi

{
  echo "PROJECT_ROOT: $PROJECT_ROOT"
  echo "MODEL_NAME: $MODEL_NAME"
  echo "LABEL_FILTER: $LABEL_FILTER"
  echo "OUT_DIR: $OUT_DIR"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  printf 'Command: %q ' "${CMD[@]}"
  echo
} | tee "$OUT_DIR/launch.log"

"${CMD[@]}" > "$OUT_DIR/run.log" 2>&1

echo "BS deception mining complete: $OUT_DIR"
