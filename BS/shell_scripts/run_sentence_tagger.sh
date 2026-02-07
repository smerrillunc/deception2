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

MODEL_NAME="${MODEL_NAME:-gpt-4-mini}"
echo "✓ Using OpenAI model: $MODEL_NAME"
echo ""

DATA_DIR="/playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1"
TAXONOMY="/playpen-ssd/smerrill/deception2/BS/config/sentence_taxonomy.json"
OUT_PATH="$DATA_DIR/tags.jsonl"

python /playpen-ssd/smerrill/deception2/BS/src/tag_sentences_llm.py \
  --sentences_path "$DATA_DIR/sentences.jsonl" \
  --taxonomy_path "$TAXONOMY" \
  --out_path "$OUT_PATH" \
  --model_name "$MODEL_NAME" \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_tokens 512

echo "✓ Tags written to $OUT_PATH"
