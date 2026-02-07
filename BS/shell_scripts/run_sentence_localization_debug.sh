#!/usr/bin/env bash
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

echo "Select a model:"
echo "  1) deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
echo "  2) deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
echo "  3) deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
echo ""

read -p "Enter model number (1–3): " MODEL_CHOICE

case "$MODEL_CHOICE" in
    1) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ;;
    2) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" ;;
    3) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" ;;
    *) echo "❌ Invalid model selection: $MODEL_CHOICE"; exit 1 ;;
esac

DATA_DIR="/playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1"
read -p "Enter example_id or index to debug: " EXAMPLE_ID

python /playpen-ssd/smerrill/deception2/BS/src/sentence_localization_debug.py \
  --examples_path "$DATA_DIR/examples.jsonl" \
  --sentences_path "$DATA_DIR/sentences.jsonl" \
  --example_id "$EXAMPLE_ID" \
  --model_name "$MODEL_NAME" \
  --out_path "$DATA_DIR/sentence_localization_debug_${EXAMPLE_ID//\//_}.json"

echo "✓ Debug JSON written to $DATA_DIR"
