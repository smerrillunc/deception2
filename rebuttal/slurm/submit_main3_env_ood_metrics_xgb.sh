#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_SCRIPT="${RUN_SCRIPT:-$SCRIPT_DIR/run_main3_env_ood_metrics_slurm.sh}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-rc_ssriva_pi}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/rebuttal/results}"
RUN_NAME="${RUN_NAME:-OOD_Modeling_main3_env_ood_metrics_qwen14b_xgb_pca_128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/$RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/slurm_logs}"
MODEL_PRESET="${MODEL_PRESET:-qwen14b}"
SCENARIOS="${SCENARIOS:-holdout_env_ood}"
FEATURE_SPACES="${FEATURE_SPACES:-baseline_tfidf_last_sentence_text,baseline_tfidf_prefix_text,attention_only,attention_grounding_only,attention_concentration_only,attention_grounding_transition_only,attention_concentration_transition_only,activation_pca_final,activation_pca_delta_last2,activation_pca_delta_prev4mean,attention_plus_activation_pca_final,attention_plus_activation_pca_delta_last2,attention_plus_activation_pca_delta_prev4mean,baseline_raw_final}"
FEATURE_SIZES="${FEATURE_SIZES:-128}"
MODEL_FAMILY="${MODEL_FAMILY:-xgb}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-100g}"
TIME_LIMIT="${TIME_LIMIT:-0-16:00:00}"
DRY_RUN="${DRY_RUN:-0}"

LOG_OUT="${LOG_OUT:-$LOG_ROOT/main3_env_ood_%j.out}"
LOG_ERR="${LOG_ERR:-$LOG_ROOT/main3_env_ood_%j.err}"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT" >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT"

cmd=(
  sbatch
  --account "$SBATCH_ACCOUNT"
  --job-name "m3envood_xgb"
  --cpus-per-task "$CPUS_PER_TASK"
  --mem "$MEMORY"
  --time "$TIME_LIMIT"
  --output "$LOG_OUT"
  --error "$LOG_ERR"
  "$RUN_SCRIPT"
)

echo "Submitting Main3 environment-OOD metrics run"
echo "Run script: $RUN_SCRIPT"
echo "Dataset root: $DATASET_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Log root: $LOG_ROOT"
echo "Model preset: $MODEL_PRESET"
echo "Scenarios: $SCENARIOS"
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
    MODEL_PRESET="$MODEL_PRESET" \
    SCENARIOS="$SCENARIOS" \
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
    MODEL_PRESET="$MODEL_PRESET" \
    SCENARIOS="$SCENARIOS" \
    FEATURE_SPACES="$FEATURE_SPACES" \
    FEATURE_SIZES="$FEATURE_SIZES" \
    MODEL_FAMILY="$MODEL_FAMILY" \
    "${cmd[@]}"
})"

echo "Submitted job: $job_id"
echo "Check queue with: squeue -u ${USER:-$LOGNAME}"
