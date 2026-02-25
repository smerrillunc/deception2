#!/usr/bin/env bash
set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ OPENAI_API_KEY is not set."
  echo "Set it with: export OPENAI_API_KEY=your_key"
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-gpt-4.1-mini}"
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
echo "✓ Using OpenAI model: $MODEL_NAME"
echo ""

DATA_DIR="/playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1/DeepSeek-R1-Distill-Qwen-14B"
TAXONOMY="/playpen-ssd/smerrill/deception2/BS/config/sentence_taxonomy.json"
OUT_PATH="$DATA_DIR/tags.jsonl"

CUDA_VISIBLE_DEVICES=7 python /playpen-ssd/smerrill/deception2/BS/src/tag_sentences_llm.py \
  --sentences_path "$DATA_DIR/sentences.jsonl" \
  --taxonomy_path "$TAXONOMY" \
  --out_path "$OUT_PATH" \
  --model_name "$MODEL_NAME" \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_tokens 10000 \
  --max_model_len 10000 \
  --backend vllm \
  --vllm_batch_size 64

echo "✓ Tags written to $OUT_PATH"
