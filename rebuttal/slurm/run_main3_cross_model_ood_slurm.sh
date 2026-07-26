#!/bin/bash
#SBATCH --job-name=main3_model_ood
#SBATCH --output=main3_model_ood_%j.out
#SBATCH --error=main3_model_ood_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100g
#SBATCH --time=0-16:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNNER_PATH="${RUNNER_PATH:-$PROJECT_ROOT/rebuttal/scripts/run_main3_cross_model_ood.py}"
NOTEBOOK_BUILDER_PATH="${NOTEBOOK_BUILDER_PATH:-$PROJECT_ROOT/rebuttal/notebooks/build_main3_cross_model_ood_notebook.py}"
NOTEBOOK_OUTPUT="${NOTEBOOK_OUTPUT:-$PROJECT_ROOT/rebuttal/notebooks/main3_cross_model_ood_analysis.ipynb}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/rebuttal/results/OOD_Modeling_main3_cross_model_ood_xgb_pca_128}"
MODEL_PRESETS="${MODEL_PRESETS:-gptoss20b,llama8b,qwen7b,qwen14b}"
FEATURE_SPACES="${FEATURE_SPACES:-baseline_tfidf_last_sentence_text,baseline_tfidf_prefix_text,attention_only,activation_pca_final,attention_plus_activation_pca_final,baseline_raw_final}"
FEATURE_SIZES="${FEATURE_SIZES:-128}"
MODEL_FAMILY="${MODEL_FAMILY:-xgb}"
SEED="${SEED:-42}"
VAL_SIZE="${VAL_SIZE:-0.20}"
DELTA_THRESHOLD="${DELTA_THRESHOLD:-0.30}"
MIN_NUM_VALID="${MIN_NUM_VALID:-11}"
MIN_SENTENCE_ALPHA_WORDS="${MIN_SENTENCE_ALPHA_WORDS:-4}"
ROOT_BATCH_SIZE="${ROOT_BATCH_SIZE:-8}"
FALLBACK_ATTENTION_TOP_K="${FALLBACK_ATTENTION_TOP_K:-128}"
DECISION_THRESHOLD_MODE="${DECISION_THRESHOLD_MODE:-train_balanced_accuracy}"
MODEL_SELECTION_OBJECTIVE="${MODEL_SELECTION_OBJECTIVE:-mean_ood_auroc_oracle}"
ACTIVATION_ALIGNMENT_MODE="${ACTIVATION_ALIGNMENT_MODE:-truncate_to_min_hidden_dim}"
LOGREG_C="${LOGREG_C:-0.1}"
XGB_MAX_DEPTH="${XGB_MAX_DEPTH:-5}"
XGB_N_ESTIMATORS="${XGB_N_ESTIMATORS:-200}"
XGB_LEARNING_RATE="${XGB_LEARNING_RATE:-0.05}"
XGB_SUBSAMPLE="${XGB_SUBSAMPLE:-0.8}"
XGB_COLSAMPLE_BYTREE="${XGB_COLSAMPLE_BYTREE:-0.8}"
XGB_REG_LAMBDA="${XGB_REG_LAMBDA:-1.0}"
XGB_MIN_CHILD_WEIGHT="${XGB_MIN_CHILD_WEIGHT:-1.0}"
XGB_GAMMA="${XGB_GAMMA:-0.0}"
XGB_N_JOBS="${XGB_N_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
XGB_EVAL_METRIC="${XGB_EVAL_METRIC:-aucpr}"
XGB_EARLY_STOPPING_ROUNDS="${XGB_EARLY_STOPPING_ROUNDS:-30}"
CALIBRATION_BINS="${CALIBRATION_BINS:-10}"
FIXED_RECALL_LEVELS="${FIXED_RECALL_LEVELS:-0.5,0.8,0.9,0.95}"
TOP_FEATURES_TO_SHOW="${TOP_FEATURES_TO_SHOW:-20}"
FORCE_REBUILD_REDUCTIONS="${FORCE_REBUILD_REDUCTIONS:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
EXCLUDE_MULTILINE_SENTENCES="${EXCLUDE_MULTILINE_SENTENCES:-0}"
BUILD_NOTEBOOK="${BUILD_NOTEBOOK:-1}"
CONDA_ENV="${CONDA_ENV:-deception}"

PYTHON_BIN="${PYTHON_BIN:-}"
USE_UV_RUN="${USE_UV_RUN:-0}"

if [[ "$USE_UV_RUN" != "1" && -z "$PYTHON_BIN" ]]; then
  module load anaconda
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi

if [[ "$USE_UV_RUN" != "1" && ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

cmd=()
if [[ "$USE_UV_RUN" == "1" ]]; then
  cmd=(uv run python "$RUNNER_PATH")
else
  cmd=("$PYTHON_BIN" "$RUNNER_PATH")
fi

cmd+=(
  --dataset-root "$DATASET_ROOT"
  --output-root "$OUTPUT_ROOT"
  --model-presets "$MODEL_PRESETS"
  --feature-spaces "$FEATURE_SPACES"
  --feature-sizes "$FEATURE_SIZES"
  --model-family "$MODEL_FAMILY"
  --seed "$SEED"
  --val-size "$VAL_SIZE"
  --delta-threshold "$DELTA_THRESHOLD"
  --min-num-valid "$MIN_NUM_VALID"
  --min-sentence-alpha-words "$MIN_SENTENCE_ALPHA_WORDS"
  --root-batch-size "$ROOT_BATCH_SIZE"
  --fallback-attention-top-k "$FALLBACK_ATTENTION_TOP_K"
  --decision-threshold-mode "$DECISION_THRESHOLD_MODE"
  --model-selection-objective "$MODEL_SELECTION_OBJECTIVE"
  --activation-alignment-mode "$ACTIVATION_ALIGNMENT_MODE"
  --logreg-c "$LOGREG_C"
  --xgb-max-depth "$XGB_MAX_DEPTH"
  --xgb-n-estimators "$XGB_N_ESTIMATORS"
  --xgb-learning-rate "$XGB_LEARNING_RATE"
  --xgb-subsample "$XGB_SUBSAMPLE"
  --xgb-colsample-bytree "$XGB_COLSAMPLE_BYTREE"
  --xgb-reg-lambda "$XGB_REG_LAMBDA"
  --xgb-min-child-weight "$XGB_MIN_CHILD_WEIGHT"
  --xgb-gamma "$XGB_GAMMA"
  --xgb-n-jobs "$XGB_N_JOBS"
  --xgb-eval-metric "$XGB_EVAL_METRIC"
  --xgb-early-stopping-rounds "$XGB_EARLY_STOPPING_ROUNDS"
  --calibration-bins "$CALIBRATION_BINS"
  --fixed-recall-levels "$FIXED_RECALL_LEVELS"
  --top-features-to-show "$TOP_FEATURES_TO_SHOW"
)

if [[ "$FORCE_REBUILD_REDUCTIONS" == "1" ]]; then
  cmd+=(--force-rebuild-reductions)
fi
if [[ "$DISABLE_TQDM" == "1" ]]; then
  cmd+=(--disable-tqdm)
fi
if [[ "$EXCLUDE_MULTILINE_SENTENCES" == "1" ]]; then
  cmd+=(--exclude-multiline-sentences)
fi

mkdir -p "$OUTPUT_ROOT"

echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "RUNNER_PATH: $RUNNER_PATH"
echo "NOTEBOOK_BUILDER_PATH: $NOTEBOOK_BUILDER_PATH"
echo "NOTEBOOK_OUTPUT: $NOTEBOOK_OUTPUT"
echo "DATASET_ROOT: $DATASET_ROOT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "MODEL_PRESETS: $MODEL_PRESETS"
echo "FEATURE_SPACES: $FEATURE_SPACES"
echo "FEATURE_SIZES: $FEATURE_SIZES"
echo "MODEL_FAMILY: $MODEL_FAMILY"
echo "CONDA_ENV: $CONDA_ENV"
echo "PYTHON_BIN: ${PYTHON_BIN:-uv run python}"
echo "XGB_N_ESTIMATORS: $XGB_N_ESTIMATORS"
echo "XGB_EVAL_METRIC: $XGB_EVAL_METRIC"
echo "XGB_EARLY_STOPPING_ROUNDS: $XGB_EARLY_STOPPING_ROUNDS"
echo "Activation alignment: $ACTIVATION_ALIGNMENT_MODE"
echo "Python entrypoint: ${cmd[*]}"

if [[ "$USE_UV_RUN" != "1" ]]; then
  "$PYTHON_BIN" -c "import sys; print('python=', sys.executable)"
fi

"${cmd[@]}"

if [[ "$BUILD_NOTEBOOK" == "1" ]]; then
  echo "Refreshing notebook: $NOTEBOOK_OUTPUT"
  if [[ "$USE_UV_RUN" == "1" ]]; then
    uv run python "$NOTEBOOK_BUILDER_PATH" --results-dir "$OUTPUT_ROOT" --output "$NOTEBOOK_OUTPUT"
  else
    "$PYTHON_BIN" "$NOTEBOOK_BUILDER_PATH" --results-dir "$OUTPUT_ROOT" --output "$NOTEBOOK_OUTPUT"
  fi
fi
