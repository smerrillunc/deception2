#!/usr/bin/env bash
set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

GPU_IDS=(${GPU_IDS:-2 3 4 5 6 7})
NUM_SHARDS=${NUM_SHARDS:-6}

if [[ ${#GPU_IDS[@]} -ne $NUM_SHARDS ]]; then
  echo "GPU_IDS count (${#GPU_IDS[@]}) must equal NUM_SHARDS ($NUM_SHARDS)."
  exit 1
fi

echo "Launching $NUM_SHARDS shards across GPUs: ${GPU_IDS[*]}"
export MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
export SKIP_GPU_LIST=1

for i in "${!GPU_IDS[@]}"; do
  GPU=${GPU_IDS[$i]}
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export SHARD_ID="$i"
    export NUM_SHARDS="$NUM_SHARDS"
    /playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_localization_batch.sh
  ) &
done

wait
echo "✓ All shards complete."
