#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_SCRIPT="${RUN_SCRIPT:-$SCRIPT_DIR/run_main3_cross_model_ood_slurm.sh}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-rc_ssriva_pi}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/rebuttal/results}"
RUN_NAME="${RUN_NAME:-OOD_Modeling_main3_cross_model_ood_xgb_pca_128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/$RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/slurm_logs}"
MODEL_PRESETS="${MODEL_PRESETS:-gptoss20b,llama8b,qwen7b,qwen14b}"
FEATURE_SPACES="${FEATURE_SPACES:-baseline_tfidf_last_sentence_text,baseline_tfidf_prefix_text,attention_only,activation_pca_final,attention_plus_activation_pca_final,baseline_raw_final}"
FEATURE_SIZES="${FEATURE_SIZES:-128}"
MODEL_FAMILY="${MODEL_FAMILY:-xgb}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-100g}"
TIME_LIMIT="${TIME_LIMIT:-0-16:00:00}"
DRY_RUN="${DRY_RUN:-0}"

LOG_OUT="${LOG_OUT:-$LOG_ROOT/main3_cross_model_ood_%j.out}"
LOG_ERR="${LOG_ERR:-$LOG_ROOT/main3_cross_model_ood_%j.err}"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT" >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT"

cmd=(
  sbatch
  --account "$SBATCH_ACCOUNT"
  --job-name "m3modood_xgb"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem "$MEMORY"
  --time "$TIME_LIMIT"
  --output "$LOG_OUT"
  --error "$LOG_ERR"
  "$RUN_SCRIPT"
)

echo "Submitting Main3 cross-model OOD run"
echo "Run script: $RUN_SCRIPT"
echo "Dataset root: $DATASET_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Log root: $LOG_ROOT"
echo "Model presets: $MODEL_PRESETS"
echo "Feature spaces: $FEATURE_SPACES"
echo "Feature sizes: $FEATURE_SIZES"
echo "Model family: $MODEL_FAMILY"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    DATASET_ROOT="$DATASET_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    LOG_ROOT="$LOG_ROOT" \
    MODEL_PRESETS="$MODEL_PRESETS" \
    FEATURE_SPACES="$FEATURE_SPACES" \
    FEATURE_SIZES="$FEATURE_SIZES" \
    MODEL_FAMILY="$MODEL_FAMILY"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

job_id="$({
  env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    DATASET_ROOT="$DATASET_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    LOG_ROOT="$LOG_ROOT" \
    MODEL_PRESETS="$MODEL_PRESETS" \
    FEATURE_SPACES="$FEATURE_SPACES" \
    FEATURE_SIZES="$FEATURE_SIZES" \
    MODEL_FAMILY="$MODEL_FAMILY" \
    "${cmd[@]}"
})"

echo "Submitted job: $job_id"
echo "Check queue with: squeue -u ${USER:-$LOGNAME}"
