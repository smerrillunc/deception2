#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
OUT_DIR="${OUT_DIR:-BS/Results/WerewolfDeceptionMining/$(date +%Y-%m-%d)/demo}"

python BS/src/werewolf_deception_miner.py \
  --model_name "$MODEL_NAME" \
  --is_reasoning_model \
  --output_dir "$OUT_DIR" \
  --num_players "${NUM_PLAYERS:-5}" \
  --max_games "${MAX_GAMES:-20}" \
  --max_turns "${MAX_TURNS:-40}" \
  --target_deceptive "${TARGET_DECEPTIVE:-0}" \
  --seed "${SEED:-0}"
