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

GRID_WIDTH="${GRID_WIDTH:-9}"
GRID_HEIGHT="${GRID_HEIGHT:-9}"
WALL_PROB="${WALL_PROB:-0.18}"
MAX_TRIES="${MAX_TRIES:-200}"
MAX_STEPS="${MAX_STEPS:-60}"
VIEW_RADIUS="${VIEW_RADIUS:-2}"
HISTORY_WINDOW="${HISTORY_WINDOW:-15}"
AUTO_MOVE_EXPLORER="${AUTO_MOVE_EXPLORER:-1}"

if [[ "$AUTO_MOVE_EXPLORER" == "0" ]]; then
  AUTO_MOVE_FLAG="--no-auto_move_explorer"
else
  AUTO_MOVE_FLAG="--auto_move_explorer"
fi

MODEL_TAG="${MODEL_NAME//\//_}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
OUT_BASE="${OUT_BASE:-$MINING_ROOT/$MODEL_TAG/$DATE_TAG}"
OUT_DIR="${OUT_DIR:-$OUT_BASE/gpu_${CUDA_VISIBLE_DEVICES}}"
SCRIPT_PATH="$SRC_ROOT/deception_miner.py"
mkdir -p "$OUT_DIR"

CMD=(
  "$PYTHON_BIN" "$SCRIPT_PATH"
  --game gridworld
  --model_name "$MODEL_NAME"
  --output_dir "$OUT_DIR"
  --seed "$SEED_BASE"
  --max_games "$MAX_GAMES"
  --max_turns "$MAX_TURNS"
  --label_filter "$LABEL_FILTER"
  --target_deceptive "$TARGET_DECEPTIVE"
  --target_truthful "$TARGET_TRUTHFUL"
  --grid_width "$GRID_WIDTH"
  --grid_height "$GRID_HEIGHT"
  --wall_prob "$WALL_PROB"
  --max_tries "$MAX_TRIES"
  --max_steps "$MAX_STEPS"
  --view_radius "$VIEW_RADIUS"
  --history_window "$HISTORY_WINDOW"
  "$AUTO_MOVE_FLAG"
  --log_every "${LOG_EVERY:-25}"
)
if [[ -n "$REASONING_FLAG" ]]; then
  CMD+=("$REASONING_FLAG")
fi

"${CMD[@]}" > "$OUT_DIR/run.log" 2>&1

echo "Gridworld deception mining complete: $OUT_DIR"
