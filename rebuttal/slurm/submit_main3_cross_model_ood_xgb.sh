#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_SCRIPT="${RUN_SCRIPT:-$SCRIPT_DIR/run_main3_cross_model_ood_slurm.sh}"
AGGREGATE_RUN_SCRIPT="${AGGREGATE_RUN_SCRIPT:-$SCRIPT_DIR/run_main3_cross_model_ood_aggregate_slurm.sh}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-rc_ssriva_pi}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/rebuttal/results}"
RUN_NAME="${RUN_NAME:-OOD_Modeling_main3_cross_model_ood_xgb_pca_128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/$RUN_NAME}"
SHARDS_ROOT="${SHARDS_ROOT:-$OUTPUT_ROOT/shards}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/slurm_logs}"
NOTEBOOK_OUTPUT="${NOTEBOOK_OUTPUT:-$PROJECT_ROOT/rebuttal/notebooks/main3_cross_model_ood_analysis.ipynb}"
MODEL_PRESETS="${MODEL_PRESETS:-gptoss20b,llama8b,qwen7b,qwen14b}"
FEATURE_SPACES="${FEATURE_SPACES:-baseline_tfidf_last_sentence_text,baseline_tfidf_prefix_text,attention_only,activation_pca_final,attention_plus_activation_pca_final,baseline_raw_final}"
FEATURE_SIZES="${FEATURE_SIZES:-128}"
MODEL_FAMILY="${MODEL_FAMILY:-xgb}"
CONDA_ENV="${CONDA_ENV:-deception}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-100g}"
TIME_LIMIT="${TIME_LIMIT:-0-16:00:00}"
AGGREGATE_CPUS_PER_TASK="${AGGREGATE_CPUS_PER_TASK:-2}"
AGGREGATE_MEMORY="${AGGREGATE_MEMORY:-32g}"
AGGREGATE_TIME_LIMIT="${AGGREGATE_TIME_LIMIT:-0-02:00:00}"
TOP_FEATURES_TO_SHOW="${TOP_FEATURES_TO_SHOW:-20}"
BUILD_NOTEBOOK="${BUILD_NOTEBOOK:-1}"
FORCE_RERUN_SHARDS="${FORCE_RERUN_SHARDS:-0}"
DRY_RUN="${DRY_RUN:-0}"

parse_csv_list() {
  local raw="$1"
  local part=""
  local out=()
  IFS=',' read -r -a parts <<< "$raw"
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    if [[ -n "$part" ]]; then
      out+=("$part")
    fi
  done
  printf '%s\n' "${out[@]}"
}

slugify() {
  local raw="$1"
  raw="${raw//\//_}"
  raw="${raw// /_}"
  raw="${raw//,/__}"
  raw="${raw//[^[:alnum:]_.-]/_}"
  printf '%s' "$raw"
}

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$AGGREGATE_RUN_SCRIPT" ]]; then
  echo "Missing aggregate run script: $AGGREGATE_RUN_SCRIPT" >&2
  exit 1
fi

mapfile -t MODEL_PRESET_LIST < <(parse_csv_list "$MODEL_PRESETS")
if [[ ${#MODEL_PRESET_LIST[@]} -eq 0 ]]; then
  echo "MODEL_PRESETS did not contain any model presets." >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT" "$OUTPUT_ROOT" "$SHARDS_ROOT" "$LOG_ROOT"

echo "Submitting Main3 cross-model OOD run"
echo "Shard run script: $RUN_SCRIPT"
echo "Aggregate run script: $AGGREGATE_RUN_SCRIPT"
echo "Dataset root: $DATASET_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Shards root: $SHARDS_ROOT"
echo "Log root: $LOG_ROOT"
echo "Model presets: ${MODEL_PRESET_LIST[*]}"
echo "Feature spaces: $FEATURE_SPACES"
echo "Feature sizes: $FEATURE_SIZES"
echo "Model family: $MODEL_FAMILY"
echo "Conda env: $CONDA_ENV"
echo "Build notebook after aggregate: $BUILD_NOTEBOOK"
echo "Force rerun completed shards: $FORCE_RERUN_SHARDS"

submitted_count=0
skipped_count=0
submitted_job_ids=()

for model_key in "${MODEL_PRESET_LIST[@]}"; do
  model_slug="$(slugify "$model_key")"
  shard_output_root="$SHARDS_ROOT/$model_slug"
  shard_log_root="$LOG_ROOT/shards"
  shard_job_name="m3mod_${model_slug}"
  shard_log_out="$shard_log_root/${shard_job_name}_%j.out"
  shard_log_err="$shard_log_root/${shard_job_name}_%j.err"
  shard_done_path="$shard_output_root/all_transfer_metrics.csv"

  mkdir -p "$shard_output_root" "$shard_log_root"

  echo
  echo "Shard: train model=$model_key"
  echo "  Shard output: $shard_output_root"

  if [[ "$FORCE_RERUN_SHARDS" != "1" && -s "$shard_done_path" ]]; then
    echo "  Skipping existing shard output: $shard_done_path"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  shard_cmd=(
    sbatch
    --account "$SBATCH_ACCOUNT"
    --job-name "$shard_job_name"
    --cpus-per-task "$CPUS_PER_TASK"
    --mem "$MEMORY"
    --time "$TIME_LIMIT"
    --output "$shard_log_out"
    --error "$shard_log_err"
    "$RUN_SCRIPT"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN SHARD:'
    printf ' %q' env \
      PROJECT_ROOT="$PROJECT_ROOT" \
      DATASET_ROOT="$DATASET_ROOT" \
      OUTPUT_ROOT="$shard_output_root" \
      SHARDS_ROOT="$SHARDS_ROOT" \
      LOG_ROOT="$LOG_ROOT" \
      NOTEBOOK_OUTPUT="$NOTEBOOK_OUTPUT" \
      CONDA_ENV="$CONDA_ENV" \
      MODEL_PRESETS="$MODEL_PRESETS" \
      TRAIN_MODEL_PRESETS="$model_key" \
      FEATURE_SPACES="$FEATURE_SPACES" \
      FEATURE_SIZES="$FEATURE_SIZES" \
      MODEL_FAMILY="$MODEL_FAMILY" \
      BUILD_NOTEBOOK="0"
    printf ' %q' "${shard_cmd[@]}"
    printf '\n'
    continue
  fi

  shard_submit_output="$({
    env \
      PROJECT_ROOT="$PROJECT_ROOT" \
      DATASET_ROOT="$DATASET_ROOT" \
      OUTPUT_ROOT="$shard_output_root" \
      SHARDS_ROOT="$SHARDS_ROOT" \
      LOG_ROOT="$LOG_ROOT" \
      NOTEBOOK_OUTPUT="$NOTEBOOK_OUTPUT" \
      CONDA_ENV="$CONDA_ENV" \
      MODEL_PRESETS="$MODEL_PRESETS" \
      TRAIN_MODEL_PRESETS="$model_key" \
      FEATURE_SPACES="$FEATURE_SPACES" \
      FEATURE_SIZES="$FEATURE_SIZES" \
      MODEL_FAMILY="$MODEL_FAMILY" \
      BUILD_NOTEBOOK="0" \
      "${shard_cmd[@]}"
  })"
  shard_job_id="$(printf '%s\n' "$shard_submit_output" | awk '{print $NF}')"
  if [[ -z "$shard_job_id" ]]; then
    echo "Failed to parse sbatch output for shard: $shard_submit_output" >&2
    exit 1
  fi
  echo "  Submitted shard job: $shard_job_id"
  echo "  Logs: $shard_log_out / $shard_log_err"
  submitted_job_ids+=("$shard_job_id")
  submitted_count=$((submitted_count + 1))
done

aggregate_log_root="$LOG_ROOT/aggregate"
aggregate_log_out="$aggregate_log_root/main3_cross_model_ood_aggregate_%j.out"
aggregate_log_err="$aggregate_log_root/main3_cross_model_ood_aggregate_%j.err"
mkdir -p "$aggregate_log_root"

aggregate_cmd=(
  sbatch
  --account "$SBATCH_ACCOUNT"
  --job-name "m3mod_agg"
  --cpus-per-task "$AGGREGATE_CPUS_PER_TASK"
  --mem "$AGGREGATE_MEMORY"
  --time "$AGGREGATE_TIME_LIMIT"
  --output "$aggregate_log_out"
  --error "$aggregate_log_err"
)

if [[ ${#submitted_job_ids[@]} -gt 0 ]]; then
  dependency_value="afterok:$(IFS=:; echo "${submitted_job_ids[*]}")"
  aggregate_cmd+=(--dependency "$dependency_value")
fi

aggregate_cmd+=("$AGGREGATE_RUN_SCRIPT")

echo
echo "Shard jobs submitted: $submitted_count"
echo "Shard jobs skipped: $skipped_count"
echo "Aggregate logs: $aggregate_log_out / $aggregate_log_err"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY RUN AGGREGATE:'
  printf ' %q' env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    SHARDS_ROOT="$SHARDS_ROOT" \
    NOTEBOOK_OUTPUT="$NOTEBOOK_OUTPUT" \
    CONDA_ENV="$CONDA_ENV" \
    FEATURE_SIZES="$FEATURE_SIZES" \
    TOP_FEATURES_TO_SHOW="$TOP_FEATURES_TO_SHOW" \
    BUILD_NOTEBOOK="$BUILD_NOTEBOOK"
  printf ' %q' "${aggregate_cmd[@]}"
  printf '\n'
  exit 0
fi

aggregate_submit_output="$({
  env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    SHARDS_ROOT="$SHARDS_ROOT" \
    NOTEBOOK_OUTPUT="$NOTEBOOK_OUTPUT" \
    CONDA_ENV="$CONDA_ENV" \
    FEATURE_SIZES="$FEATURE_SIZES" \
    TOP_FEATURES_TO_SHOW="$TOP_FEATURES_TO_SHOW" \
    BUILD_NOTEBOOK="$BUILD_NOTEBOOK" \
    "${aggregate_cmd[@]}"
})"
aggregate_job_id="$(printf '%s\n' "$aggregate_submit_output" | awk '{print $NF}')"
if [[ -z "$aggregate_job_id" ]]; then
  echo "Failed to parse sbatch output for aggregate job: $aggregate_submit_output" >&2
  exit 1
fi

echo "Submitted aggregate job: $aggregate_job_id"
if [[ ${#submitted_job_ids[@]} -gt 0 ]]; then
  echo "Aggregate dependency: afterok on ${#submitted_job_ids[@]} shard jobs"
fi
echo "Check queue with: squeue -u ${USER:-$LOGNAME}"
