#!/usr/bin/env bash
set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

PIPELINE_DATE=$(date +%Y-%m-%d)
echo "Pipeline date: $PIPELINE_DATE"

MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
export MODEL_NAME

echo "==> 1) Multi-GPU deception mining"
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_deception_miner_multi_gpu.sh

echo "==> 2) Build sentence dataset"
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_dataset.sh

echo "==> 3) Tag sentences (OpenAI API)"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ OPENAI_API_KEY is not set."
  echo "Set it with: export OPENAI_API_KEY=your_key"
  exit 1
fi
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_tagger.sh

echo "==> 4) Multi-GPU sentence-level localization"
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_localization_batch_multi_gpu.sh

echo "==> 5) Merge localization shards"
python /playpen-ssd/smerrill/deception2/BS/src/merge_localization_jsonl.py \
  --input /playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1/localization.jsonl.shard*.jsonl \
  --out_path /playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1/localization.merged.jsonl \
  --dedupe

echo "==> 6) Extract sentence features"
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_features.sh

echo "✓ Full pipeline complete."
