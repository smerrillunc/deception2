#!/usr/bin/env bash
set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

if [[ -z "${SKIP_GPU_LIST:-}" ]]; then
  echo "Available GPUs:"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  echo ""
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "✓ Using GPU: $CUDA_VISIBLE_DEVICES"
  echo ""
else
  if [[ ! -t 0 ]]; then
    export CUDA_VISIBLE_DEVICES="0"
    echo "✓ Using GPU: $CUDA_VISIBLE_DEVICES (non-interactive default)"
    echo ""
  else
    read -p "Enter the GPU ID you want to use (e.g., 0): " GPU
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo "✓ Using GPU: $CUDA_VISIBLE_DEVICES"
    echo ""
  fi
fi

if [[ -z "${MODEL_NAME:-}" ]]; then
  if [[ ! -t 0 ]]; then
    # Non-interactive: pick a reasonable default to avoid hanging
    MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  else
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
  fi
fi

DATA_DIR="/playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1"
OUT_DIR="${OUT_DIR:-/playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1/localization}"
if [[ "$OUT_DIR" != "none" ]]; then
  mkdir -p "$OUT_DIR"
fi
JSONL_PATH="${JSONL_PATH:-/playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1/localization.jsonl}"

SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

echo "✓ MODEL_NAME: $MODEL_NAME (shard $SHARD_ID / $NUM_SHARDS)"

CMD=(python /playpen-ssd/smerrill/deception2/BS/src/sentence_localization_batch.py
  --examples_path "$DATA_DIR/examples.jsonl"
  --sentences_path "$DATA_DIR/sentences.jsonl"
  --jsonl_path "$JSONL_PATH"
  --model_name "$MODEL_NAME"
  --n_samples 25
  --temperature 0.5
  --top_p 0.5
  --repetition_penalty 1.2
  --max_new_tokens 10000
  --mode "prefix"
  --only_deceptive
  --shard_id "$SHARD_ID"
  --num_shards "$NUM_SHARDS"
  --log_every 25
)

if [[ "$OUT_DIR" != "none" ]]; then
  CMD+=(--out_dir "$OUT_DIR")
fi

"${CMD[@]}"

echo "✓ Batch localization complete in $OUT_DIR"
