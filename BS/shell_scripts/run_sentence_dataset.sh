#!/usr/bin/env bash
set -euo pipefail

echo "Activating conda environment: deception"
source /playpen-ssd/smerrill/miniconda/etc/profile.d/conda.sh
conda activate deception

INPUT_ROOT="/playpen-ssd/smerrill/deception2/BS/Results/DeceptionMining"
OUT_DIR="/playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1"

python /playpen-ssd/smerrill/deception2/BS/src/build_sentence_dataset.py \
  --input_root "$INPUT_ROOT" \
  --out_dir "$OUT_DIR" \
  --text_field "action_reasoning" \
  --include_messages \
  --only_deceptive

echo "✓ Sentence dataset built at $OUT_DIR"
