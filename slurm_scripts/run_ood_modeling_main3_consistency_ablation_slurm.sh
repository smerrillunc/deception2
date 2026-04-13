#!/bin/bash
#SBATCH --job-name=ood_main3
#SBATCH --output=ood_main3_%j.out
#SBATCH --error=ood_main3_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=1-12:00:00

set -euo pipefail

# Example:
# sbatch --account=rc_amcavoy_pi \
#   --export=ALL,PROJECT_ROOT=/work/users/s/m/smerrill/deception2,DATASET_ROOT=/work/users/s/m/smerrill/deception2/DatasetMain,MODEL_DIRNAME=DeepSeek-R1-Distill-Qwen-7B \
#   slurm_scripts/run_ood_modeling_main3_consistency_ablation_slurm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_project_root() {
  local start_dir=""
  local candidate=""
  local resolved=""

  for start_dir in "${PROJECT_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "${PWD:-}" "$SCRIPT_DIR"; do
    if [[ -z "$start_dir" ]] || [[ ! -d "$start_dir" ]]; then
      continue
    fi
    resolved="$(cd "$start_dir" && pwd)"
    candidate="$resolved"
    while true; do
      if [[ -f "$candidate/src/run_ood_modeling_main3_consistency_ablation.py" ]]; then
        printf '%s' "$candidate"
        return 0
      fi
      if [[ "$candidate" == "/" ]]; then
        break
      fi
      candidate="$(dirname "$candidate")"
    done
  done
  return 1
}

PROJECT_ROOT="${PROJECT_ROOT:-}"
if [[ -z "$PROJECT_ROOT" || ! -f "$PROJECT_ROOT/src/run_ood_modeling_main3_consistency_ablation.py" ]]; then
  PROJECT_ROOT="$(find_project_root || true)"
fi
if [[ -z "$PROJECT_ROOT" ]]; then
  echo "Could not resolve PROJECT_ROOT." >&2
  echo "Set PROJECT_ROOT explicitly when submitting, e.g." >&2
  echo "  sbatch --export=ALL,PROJECT_ROOT=/work/users/s/m/smerrill/deception2,DATASET_ROOT=/work/users/s/m/smerrill/deception2/DatasetMain ..." >&2
  exit 1
fi

RUNNER_PATH="${RUNNER_PATH:-$PROJECT_ROOT/src/run_ood_modeling_main3_consistency_ablation.py}"

MODEL_DIRNAME="${MODEL_DIRNAME:-DeepSeek-R1-Distill-Qwen-7B}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/Results/OOD_Modeling_main3_consistency}"
RUN_TAG="${RUN_TAG:-kgrid}"

FEATURE_SIZES="${FEATURE_SIZES:-32,64,128,256}"
ATTENTION_TOP_K="${ATTENTION_TOP_K:-}"
C_GRID="${C_GRID:-0.1}"
SEED="${SEED:-42}"
VAL_SIZE="${VAL_SIZE:-0.20}"
DELTA_THRESHOLD="${DELTA_THRESHOLD:-0.30}"
PER_ROOT_LIMIT="${PER_ROOT_LIMIT:-4}"
ROOT_BATCH_SIZE="${ROOT_BATCH_SIZE:-8}"
DECISION_THRESHOLD_MODE="${DECISION_THRESHOLD_MODE:-train_balanced_accuracy}"
MODEL_SELECTION_OBJECTIVE="${MODEL_SELECTION_OBJECTIVE:-mean_ood_auroc_oracle}"
TOP_FEATURES_TO_SHOW="${TOP_FEATURES_TO_SHOW:-20}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-0}"
SHOW_PLOTS="${SHOW_PLOTS:-0}"

CONDA_ENV="${CONDA_ENV:-deception}"
PYTHON_BIN="${PYTHON_BIN:-}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

slugify() {
  local raw="$1"
  raw="${raw//\//_}"
  raw="${raw//[^[:alnum:]_.-]/_}"
  printf '%s' "$raw"
}

MODEL_SLUG="$(slugify "$MODEL_DIRNAME")"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/$MODEL_SLUG/$RUN_TAG}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "DATASET_ROOT does not exist: $DATASET_ROOT" >&2
  exit 1
fi
if [[ ! -f "$RUNNER_PATH" ]]; then
  echo "Runner script not found: $RUNNER_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

build_job_name() {
  local model_slug="$1"
  local run_tag="$2"
  local job_name="ood_main3_${model_slug}_${run_tag}"
  job_name="${job_name//[^[:alnum:]_.-]/_}"
  printf '%s' "${job_name:0:120}"
}

JOB_NAME="$(build_job_name "$MODEL_SLUG" "$RUN_TAG")"
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  if scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME" >/dev/null 2>&1; then
    echo "SLURM job name: $JOB_NAME"
  else
    echo "Warning: failed to update SLURM job name to $JOB_NAME" >&2
  fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
  module load anaconda
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1

CMD=(
  "$PYTHON_BIN" "$RUNNER_PATH"
  --model-dirname "$MODEL_DIRNAME"
  --dataset-root "$DATASET_ROOT"
  --output-root "$OUTPUT_ROOT"
  --feature-sizes "$FEATURE_SIZES"
  --c-grid "$C_GRID"
  --seed "$SEED"
  --val-size "$VAL_SIZE"
  --delta-threshold "$DELTA_THRESHOLD"
  --per-root-limit "$PER_ROOT_LIMIT"
  --root-batch-size "$ROOT_BATCH_SIZE"
  --decision-threshold-mode "$DECISION_THRESHOLD_MODE"
  --model-selection-objective "$MODEL_SELECTION_OBJECTIVE"
  --top-features-to-show "$TOP_FEATURES_TO_SHOW"
)

if [[ -n "$ATTENTION_TOP_K" ]]; then
  CMD+=(--attention-top-k "$ATTENTION_TOP_K")
fi
if [[ "$FORCE_REBUILD" == "1" ]]; then
  CMD+=(--force-rebuild)
fi
if [[ "$DISABLE_TQDM" == "1" ]]; then
  CMD+=(--disable-tqdm)
fi
if [[ "$SHOW_PLOTS" == "1" ]]; then
  CMD+=(--show-plots)
fi

echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "RUNNER_PATH: $RUNNER_PATH"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "MODEL_DIRNAME: $MODEL_DIRNAME"
echo "DATASET_ROOT: $DATASET_ROOT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "FEATURE_SIZES: $FEATURE_SIZES"
echo "THREADS: $THREADS"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "OOD Main3 consistency ablation complete."
