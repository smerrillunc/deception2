#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUN_SCRIPT="$SCRIPT_DIR/run_ood_modeling_main3_consistency_ablation_slurm.sh"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-rc_amcavoy_pi}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/Results/OOD_Modeling_main3_consistency_xgb_pca_64_128_256}"
LOG_ROOT="${LOG_ROOT:-$RESULTS_ROOT/slurm_logs}"
FEATURE_SIZES="${FEATURE_SIZES:-64,128,256}"
RUN_TAG="${RUN_TAG:-pca_64_128_256}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-100g}"
TIME_LIMIT="${TIME_LIMIT:-1-12:00:00}"
THREADS="${THREADS:-$CPUS_PER_TASK}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
SHOW_PLOTS="${SHOW_PLOTS:-0}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_PRESETS_CSV="${MODEL_PRESETS_CSV:-qwen7b,gptoss20b,llama8b}"
SCENARIO_KEYS_CSV="${SCENARIO_KEYS_CSV:-single_source_ood,holdout_env_ood}"

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

mapfile -t MODELS < <(parse_csv_list "$MODEL_PRESETS_CSV")
mapfile -t SCENARIOS < <(parse_csv_list "$SCENARIO_KEYS_CSV")

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "MODEL_PRESETS_CSV did not contain any model presets." >&2
  exit 1
fi
if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
  echo "SCENARIO_KEYS_CSV did not contain any scenarios." >&2
  exit 1
fi

slugify() {
  local raw="$1"
  raw="${raw//\//_}"
  raw="${raw//[^[:alnum:]_.-]/_}"
  printf '%s' "$raw"
}

build_bundle_name() {
  local model_key="$1"
  local scenario_key="$2"
  local feature_tag
  feature_tag="$(slugify "pca_${FEATURE_SIZES//,/__}")"
  printf '%s__%s__xgb__%s' "$model_key" "$scenario_key" "$feature_tag"
}

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "DATASET_ROOT does not exist: $DATASET_ROOT" >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"

submitted_count=0

echo "Submitting Main3 consistency ablations"
echo "Run script: $RUN_SCRIPT"
echo "Account: $SBATCH_ACCOUNT"
echo "Dataset root: $DATASET_ROOT"
echo "Results root: $RESULTS_ROOT"
echo "Log root: $LOG_ROOT"
echo "Feature sizes: $FEATURE_SIZES"
echo "Model presets: ${MODELS[*]}"
echo "Scenarios: ${SCENARIOS[*]}"
echo "CPUs per task: $CPUS_PER_TASK"
echo "Memory: $MEMORY"
echo "Time limit: $TIME_LIMIT"

for model_key in "${MODELS[@]}"; do
  for scenario_key in "${SCENARIOS[@]}"; do
    bundle_name="$(build_bundle_name "$model_key" "$scenario_key")"
    output_root="$RESULTS_ROOT/$bundle_name"
    job_name="oodm3_${model_key}_${scenario_key}_xgb"
    log_out="$LOG_ROOT/${job_name}_%j.out"
    log_err="$LOG_ROOT/${job_name}_%j.err"

    echo
    echo "Bundle: $bundle_name"
    echo "  Model preset: $model_key"
    echo "  Scenario: $scenario_key"
    echo "  Output root: $output_root"

    cmd=(
      sbatch
      --account "$SBATCH_ACCOUNT"
      --job-name "$job_name"
      --cpus-per-task "$CPUS_PER_TASK"
      --mem "$MEMORY"
      --time "$TIME_LIMIT"
      --output "$log_out"
      --error "$log_err"
      "$RUN_SCRIPT"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY RUN:'
      printf ' %q' env         PROJECT_ROOT="$PROJECT_ROOT"         DATASET_ROOT="$DATASET_ROOT"         RESULTS_ROOT="$RESULTS_ROOT"         MODEL_PRESET="$model_key"         TRAIN_MODEL="xgb"         SCENARIOS="$scenario_key"         FEATURE_SIZES="$FEATURE_SIZES"         RUN_TAG="$RUN_TAG"         RUN_NAME="$bundle_name"         OUTPUT_ROOT="$output_root"         THREADS="$THREADS"         DISABLE_TQDM="$DISABLE_TQDM"         SHOW_PLOTS="$SHOW_PLOTS"
      printf ' %q' "${cmd[@]}"
      printf '
'
      continue
    fi

    job_id="$({
      env         PROJECT_ROOT="$PROJECT_ROOT"         DATASET_ROOT="$DATASET_ROOT"         RESULTS_ROOT="$RESULTS_ROOT"         MODEL_PRESET="$model_key"         TRAIN_MODEL="xgb"         SCENARIOS="$scenario_key"         FEATURE_SIZES="$FEATURE_SIZES"         RUN_TAG="$RUN_TAG"         RUN_NAME="$bundle_name"         OUTPUT_ROOT="$output_root"         THREADS="$THREADS"         DISABLE_TQDM="$DISABLE_TQDM"         SHOW_PLOTS="$SHOW_PLOTS"         "${cmd[@]}"
    })"

    echo "  Submitted job: $job_id"
    echo "  Logs: $log_out / $log_err"
    submitted_count=$((submitted_count + 1))
  done
done

echo
echo "Submitted $submitted_count jobs total."
echo "Check queue with: squeue -u ${USER:-$LOGNAME}"
