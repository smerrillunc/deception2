#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="$SCRIPT_DIR/run_targeted_localization_pipeline.sh"

MODELS=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

for model in "${MODELS[@]}"; do
  echo ""
  echo "============================================================"
  echo "Running pipeline for model=$model across all environments"
  echo "============================================================"
  "$PIPELINE_SCRIPT" --model_name "$model" "$@"
done
