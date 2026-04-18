#!/bin/bash
#SBATCH --job-name=ood_main3
#SBATCH --output=ood_main3_%j.out
#SBATCH --error=ood_main3_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100g
#SBATCH --time=1-12:00:00

set -euo pipefail

# Example:
# sbatch --account=rc_amcavoy_pi \
#   --export=ALL,PROJECT_ROOT=/work/users/s/m/smerrill/deception2,DATASET_ROOT=/work/users/s/m/smerrill/deception2/DatasetMain,MODEL_PRESET=qwen7b,TRAIN_MODEL=logreg \
#   slurm_scripts/run_ood_modeling_main3_consistency_ablation_slurm.sh
#
# Most explicit version:
# sbatch --account=rc_amcavoy_pi \
#   --export=ALL,PROJECT_ROOT=/work/users/s/m/smerrill/deception2,RUNNER_PATH=/work/users/s/m/smerrill/deception2/src/run_ood_modeling_main3_consistency_ablation.py,DATASET_ROOT=/work/users/s/m/smerrill/deception2/DatasetMain,MODEL_PRESET=llama8b,TRAIN_MODEL=xgb,RUN_TAG=baseline \
#   /work/users/s/m/smerrill/deception2/slurm_scripts/run_ood_modeling_main3_consistency_ablation_slurm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_project_root() {
  local start_dir=""
  local candidate=""
  local resolved=""
  local user_first=""
  local user_second=""

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

  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    for candidate in \
      "$SLURM_SUBMIT_DIR/deception2" \
      "$SLURM_SUBMIT_DIR/../deception2" \
      "$SLURM_SUBMIT_DIR/../../deception2"; do
      if [[ -f "$candidate/src/run_ood_modeling_main3_consistency_ablation.py" ]]; then
        printf '%s' "$(cd "$candidate" && pwd)"
        return 0
      fi
    done
  fi

  if [[ -n "${HOME:-}" && -f "$HOME/deception2/src/run_ood_modeling_main3_consistency_ablation.py" ]]; then
    printf '%s' "$(cd "$HOME/deception2" && pwd)"
    return 0
  fi

  if [[ -n "${USER:-}" && "${#USER}" -ge 2 ]]; then
    user_first="${USER:0:1}"
    user_second="${USER:1:1}"
    candidate="/work/users/$user_first/$user_second/$USER/deception2"
    if [[ -f "$candidate/src/run_ood_modeling_main3_consistency_ablation.py" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  fi
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

slugify() {
  local raw="$1"
  raw="${raw//\//_}"
  raw="${raw//[^[:alnum:]_.-]/_}"
  printf '%s' "$raw"
}

first_csv_value() {
  local raw="${1:-}"
  raw="${raw%%,*}"
  raw="${raw//[[:space:]]/}"
  printf '%s' "$raw"
}

normalize_train_model() {
  local raw="${1:-logreg}"
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$raw" in
    logreg|logistic|logistic_regression|lr)
      printf 'logreg'
      ;;
    xgb|xgboost)
      printf 'xgboost'
      ;;
    *)
      echo "Unsupported TRAIN_MODEL: $1" >&2
      echo "Supported values: logreg, xgb" >&2
      return 1
      ;;
  esac
}

resolve_model_dirname() {
  local raw="${1:-qwen7b}"
  raw="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$raw" in
    qwen7b|deepseek_qwen7b|deepseek-r1-distill-qwen-7b)
      printf 'DeepSeek-R1-Distill-Qwen-7B'
      ;;
    qwen14b|deepseek_qwen14b|deepseek-r1-distill-qwen-14b)
      printf 'DeepSeek-R1-Distill-Qwen-14B'
      ;;
    llama8b|deepseek_llama8b|deepseek-r1-distill-llama-8b)
      printf 'DeepSeek-R1-Distill-Llama-8B'
      ;;
    gptoss20b|gpt-oss-20b|oss20b)
      printf 'gpt-oss-20b'
      ;;
    *)
      echo "Unsupported MODEL_PRESET: $1" >&2
      echo "Supported values: qwen7b, qwen14b, llama8b, gptoss20b" >&2
      return 1
      ;;
  esac
}

resolve_model_key() {
  local model_dirname="$1"
  case "$model_dirname" in
    DeepSeek-R1-Distill-Qwen-7B)
      printf 'qwen7b'
      ;;
    DeepSeek-R1-Distill-Qwen-14B)
      printf 'qwen14b'
      ;;
    DeepSeek-R1-Distill-Llama-8B)
      printf 'llama8b'
      ;;
    gpt-oss-20b)
      printf 'gptoss20b'
      ;;
    *)
      printf '%s' "$(slugify "$model_dirname")"
      ;;
  esac
}

MODEL_PRESET="${MODEL_PRESET:-qwen7b}"
if [[ -n "${MODEL_DIRNAME:-}" ]]; then
  MODEL_DIRNAME="$MODEL_DIRNAME"
  MODEL_PRESET_RESOLVED="custom"
else
  MODEL_DIRNAME="$(resolve_model_dirname "$MODEL_PRESET")"
  MODEL_PRESET_RESOLVED="$MODEL_PRESET"
fi
MODEL_KEY="$(resolve_model_key "$MODEL_DIRNAME")"
TRAIN_MODEL="${TRAIN_MODEL:-logreg}"
TRAIN_MODEL_FAMILY="$(normalize_train_model "$TRAIN_MODEL")"
TRAIN_MODEL_KEY="$TRAIN_MODEL_FAMILY"
if [[ "$TRAIN_MODEL_FAMILY" == "xgboost" ]]; then
  TRAIN_MODEL_KEY="xgb"
fi
SCENARIOS="${SCENARIOS:-single_source_ood}"
SCENARIO_KEY="$(slugify "${SCENARIOS//,/__}")"

DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/Results/OOD_Modeling_main3_consistency_FINAL}"
RUN_TAG="${RUN_TAG:-baseline}"
RUN_NAME="${RUN_NAME:-${MODEL_KEY}__${SCENARIO_KEY}__${TRAIN_MODEL_KEY}__${RUN_TAG}}"

FEATURE_SIZES="${FEATURE_SIZES:-128}"
ATTENTION_TOP_K="${ATTENTION_TOP_K:-}"
LOGREG_C="${LOGREG_C:-}"
if [[ -z "$LOGREG_C" && -n "${C_GRID:-}" ]]; then
  LOGREG_C="$(first_csv_value "$C_GRID")"
fi
LOGREG_C="${LOGREG_C:-0.1}"
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
XGB_MAX_DEPTH="${XGB_MAX_DEPTH:-}"
if [[ -z "$XGB_MAX_DEPTH" && -n "${XGB_MAX_DEPTH_GRID:-}" ]]; then
  XGB_MAX_DEPTH="$(first_csv_value "$XGB_MAX_DEPTH_GRID")"
fi
XGB_MAX_DEPTH="${XGB_MAX_DEPTH:-5}"
XGB_N_ESTIMATORS="${XGB_N_ESTIMATORS:-300}"
XGB_LEARNING_RATE="${XGB_LEARNING_RATE:-0.05}"
XGB_SUBSAMPLE="${XGB_SUBSAMPLE:-0.8}"
XGB_COLSAMPLE_BYTREE="${XGB_COLSAMPLE_BYTREE:-0.8}"
XGB_REG_LAMBDA="${XGB_REG_LAMBDA:-1.0}"
XGB_MIN_CHILD_WEIGHT="${XGB_MIN_CHILD_WEIGHT:-1.0}"
XGB_GAMMA="${XGB_GAMMA:-0.0}"
XGB_N_JOBS="${XGB_N_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
XGB_IMPORTANCE_TYPE="${XGB_IMPORTANCE_TYPE:-gain}"

CONDA_ENV="${CONDA_ENV:-deception}"
PYTHON_BIN="${PYTHON_BIN:-}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$RESULTS_ROOT/$RUN_NAME}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "DATASET_ROOT does not exist: $DATASET_ROOT" >&2
  exit 1
fi
if [[ ! -f "$RUNNER_PATH" ]]; then
  echo "Runner script not found: $RUNNER_PATH" >&2
  echo "PROJECT_ROOT=$PROJECT_ROOT" >&2
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}" >&2
  echo "PWD=${PWD:-unset}" >&2
  echo "If you are still seeing a /var/spool/slurmd path here, the longleaf copy of this launcher is stale." >&2
  echo "Resync the updated launcher or submit with explicit PROJECT_ROOT and RUNNER_PATH." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

build_job_name() {
  local run_name="$1"
  local job_name="ood_main3_${run_name}"
  job_name="${job_name//[^[:alnum:]_.-]/_}"
  printf '%s' "${job_name:0:120}"
}

JOB_NAME="$(build_job_name "$RUN_NAME")"
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
export OOD_MAIN3_COMPANION_MODEL_FAMILY="$TRAIN_MODEL_FAMILY"
export OOD_MAIN3_COMPANION_LOGREG_C="$LOGREG_C"
export OOD_MAIN3_COMPANION_XGB_MAX_DEPTH="$XGB_MAX_DEPTH"
export OOD_MAIN3_COMPANION_XGB_N_ESTIMATORS="$XGB_N_ESTIMATORS"
export OOD_MAIN3_COMPANION_XGB_LEARNING_RATE="$XGB_LEARNING_RATE"
export OOD_MAIN3_COMPANION_XGB_SUBSAMPLE="$XGB_SUBSAMPLE"
export OOD_MAIN3_COMPANION_XGB_COLSAMPLE_BYTREE="$XGB_COLSAMPLE_BYTREE"
export OOD_MAIN3_COMPANION_XGB_REG_LAMBDA="$XGB_REG_LAMBDA"
export OOD_MAIN3_COMPANION_XGB_MIN_CHILD_WEIGHT="$XGB_MIN_CHILD_WEIGHT"
export OOD_MAIN3_COMPANION_XGB_GAMMA="$XGB_GAMMA"
export OOD_MAIN3_COMPANION_XGB_N_JOBS="$XGB_N_JOBS"
export OOD_MAIN3_COMPANION_XGB_IMPORTANCE_TYPE="$XGB_IMPORTANCE_TYPE"

CMD=(
  "$PYTHON_BIN" "$RUNNER_PATH"
  --model-dirname "$MODEL_DIRNAME"
  --dataset-root "$DATASET_ROOT"
  --output-root "$OUTPUT_ROOT"
  --model-family "$TRAIN_MODEL_FAMILY"
  --scenarios "$SCENARIOS"
  --feature-sizes "$FEATURE_SIZES"
  --logreg-c "$LOGREG_C"
  --xgb-max-depth "$XGB_MAX_DEPTH"
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
echo "MODEL_PRESET: $MODEL_PRESET_RESOLVED"
echo "MODEL_KEY: $MODEL_KEY"
echo "MODEL_DIRNAME: $MODEL_DIRNAME"
echo "TRAIN_MODEL: $TRAIN_MODEL_KEY"
echo "TRAIN_MODEL_FAMILY: $TRAIN_MODEL_FAMILY"
echo "SCENARIOS: $SCENARIOS"
echo "DATASET_ROOT: $DATASET_ROOT"
echo "RUN_TAG: $RUN_TAG"
echo "RUN_NAME: $RUN_NAME"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "FEATURE_SIZES: $FEATURE_SIZES"
echo "OOD_MAIN3_COMPANION_MODEL_FAMILY: ${OOD_MAIN3_COMPANION_MODEL_FAMILY}"
echo "LOGREG_C: $LOGREG_C"
if [[ "$TRAIN_MODEL_FAMILY" == "xgboost" ]]; then
  echo "XGB_MAX_DEPTH: $XGB_MAX_DEPTH"
  echo "XGB_N_ESTIMATORS: $XGB_N_ESTIMATORS"
  echo "XGB_LEARNING_RATE: $XGB_LEARNING_RATE"
  echo "XGB_SUBSAMPLE: $XGB_SUBSAMPLE"
  echo "XGB_COLSAMPLE_BYTREE: $XGB_COLSAMPLE_BYTREE"
fi
echo "THREADS: $THREADS"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "OOD Main3 consistency ablation complete."
