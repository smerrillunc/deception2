#!/usr/bin/env bash
set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

DATA_DIR="/playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1"

if [[ -z "${LOCALIZATION_PATH:-}" ]]; then
  if [[ -f "$DATA_DIR/localization.merged.jsonl" ]]; then
    LOCALIZATION_PATH="$DATA_DIR/localization.merged.jsonl"
  elif [[ -f "$DATA_DIR/localization.jsonl" ]]; then
    LOCALIZATION_PATH="$DATA_DIR/localization.jsonl"
  else
    LOCALIZATION_PATH="$DATA_DIR/localization"
  fi
fi
OUT_PATH="${OUT_PATH:-$DATA_DIR/sentence_features.jsonl}"

python /playpen-ssd/smerrill/deception2/BS/src/extract_sentence_features.py \
  --examples_path "$DATA_DIR/examples.jsonl" \
  --sentences_path "$DATA_DIR/sentences.jsonl" \
  --tags_path "$DATA_DIR/tags.jsonl" \
  --localization_path "$LOCALIZATION_PATH" \
  --out_path "$OUT_PATH" \
  --label_strategy "delta_threshold" \
  --delta_threshold 0.1

echo "✓ Features written to $OUT_PATH"
