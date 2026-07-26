#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_SCRIPT="${RUN_SCRIPT:-$SCRIPT_DIR/run_main3_env_ood_metrics_slurm.sh}"
AGGREGATE_RUN_SCRIPT="${AGGREGATE_RUN_SCRIPT:-$SCRIPT_DIR/run_main3_env_ood_aggregate_slurm.sh}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-rc_ssriva_pi}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/rebuttal/results}"
RUN_NAME="${RUN_NAME:-OOD_Modeling_main3_env_ood_metrics_qwen14b_xgb_pca_128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/$RUN_NAME}"
SHARDS_ROOT="${SHARDS_ROOT:-$OUTPUT_ROOT/shards}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/slurm_logs}"
NOTEBOOK_OUTPUT="${NOTEBOOK_OUTPUT:-$PROJECT_ROOT/rebuttal/notebooks/main3_env_ood_analysis.ipynb}"
MODEL_PRESET="${MODEL_PRESET:-qwen14b}"
SCENARIOS="${SCENARIOS:-holdout_env_ood}"
ENV_NAMES_CSV="${ENV_NAMES_CSV:-AdvisorAudit,BS,CarSales,Gridworld,Interview}"
FEATURE_SPACES="${FEATURE_SPACES:-baseline_tfidf_last_sentence_text,baseline_tfidf_prefix_text,attention_only,attention_grounding_only,attention_concentration_only,attention_grounding_transition_only,attention_concentration_transition_only,activation_pca_final,activation_pca_delta_last2,activation_pca_delta_prev4mean,attention_plus_activation_pca_final,attention_plus_activation_pca_delta_last2,attention_plus_activation_pca_delta_prev4mean,baseline_raw_final}"
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

build_train_env_label() {
  local scenario_name="$1"
  local env_name="$2"
  case "$scenario_name" in
    holdout_env_ood)
      printf 'All except %s' "$env_name"
      ;;
    single_source_ood)
      printf '%s' "$env_name"
      ;;
    *)
      echo "Unsupported scenario: $scenario_name" >&2
      return 1
      ;;
  esac
}

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$AGGREGATE_RUN_SCRIPT" ]]; then
  echo "Missing aggregate run script: $AGGREGATE_RUN_SCRIPT" >&2
  exit 1
fi

mapfile -t SCENARIO_LIST < <(parse_csv_list "$SCENARIOS")
mapfile -t ENV_NAMES < <(parse_csv_list "$ENV_NAMES_CSV")

if [[ ${#SCENARIO_LIST[@]} -eq 0 ]]; then
  echo "SCENARIOS did not contain any scenarios." >&2
  exit 1
fi
if [[ ${#ENV_NAMES[@]} -eq 0 ]]; then
  echo "ENV_NAMES_CSV did not contain any environments." >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT" "$OUTPUT_ROOT" "$SHARDS_ROOT" "$LOG_ROOT"

echo "Submitting Main3 environment-OOD metrics run"
echo "Shard run script: $RUN_SCRIPT"
echo "Aggregate run script: $AGGREGATE_RUN_SCRIPT"
echo "Dataset root: $DATASET_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Shards root: $SHARDS_ROOT"
echo "Log root: $LOG_ROOT"
echo "Model preset: $MODEL_PRESET"
echo "Scenarios: ${SCENARIO_LIST[*]}"
echo "Environments: ${ENV_NAMES[*]}"
echo "Feature spaces: $FEATURE_SPACES"
echo "Feature sizes: $FEATURE_SIZES"
echo "Model family: $MODEL_FAMILY"
echo "Conda env: $CONDA_ENV"
echo "Build notebook after aggregate: $BUILD_NOTEBOOK"
echo "Force rerun completed shards: $FORCE_RERUN_SHARDS"

submitted_count=0
skipped_count=0
submitted_job_ids=()

for scenario_name in "${SCENARIO_LIST[@]}"; do
  for env_name in "${ENV_NAMES[@]}"; do
    train_env_label="$(build_train_env_label "$scenario_name" "$env_name")"
    scenario_slug="$(slugify "$scenario_name")"
    env_slug="$(slugify "$env_name")"
    shard_output_root="$SHARDS_ROOT/$scenario_slug/$env_slug"
    shard_log_root="$LOG_ROOT/shards/$scenario_slug"
    shard_job_name="m3env_${scenario_slug}_${env_slug}"
    shard_log_out="$shard_log_root/${shard_job_name}_%j.out"
    shard_log_err="$shard_log_root/${shard_job_name}_%j.err"
    shard_done_path="$shard_output_root/all_transfer_metrics.csv"

    mkdir -p "$shard_output_root" "$shard_log_root"

    echo
    echo "Shard: scenario=$scenario_name env=$env_name"
    echo "  Train env label: $train_env_label"
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
        MODEL_PRESET="$MODEL_PRESET" \
        SCENARIOS="$scenario_name" \
        TRAIN_ENV_LABELS="$train_env_label" \
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
        MODEL_PRESET="$MODEL_PRESET" \
        SCENARIOS="$scenario_name" \
        TRAIN_ENV_LABELS="$train_env_label" \
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
done

aggregate_log_root="$LOG_ROOT/aggregate"
aggregate_log_out="$aggregate_log_root/main3_env_ood_aggregate_%j.out"
aggregate_log_err="$aggregate_log_root/main3_env_ood_aggregate_%j.err"
mkdir -p "$aggregate_log_root"

aggregate_cmd=(
  sbatch
  --account "$SBATCH_ACCOUNT"
  --job-name "m3env_agg"
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
