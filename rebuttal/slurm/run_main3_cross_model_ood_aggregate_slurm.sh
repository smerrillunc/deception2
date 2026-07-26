#!/bin/bash
#SBATCH --job-name=main3_model_ood_agg
#SBATCH --output=main3_model_ood_agg_%j.out
#SBATCH --error=main3_model_ood_agg_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32g
#SBATCH --time=0-02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
AGGREGATE_SCRIPT_PATH="${AGGREGATE_SCRIPT_PATH:-$PROJECT_ROOT/rebuttal/scripts/aggregate_main3_cross_model_ood_results.py}"
NOTEBOOK_BUILDER_PATH="${NOTEBOOK_BUILDER_PATH:-$PROJECT_ROOT/rebuttal/notebooks/build_main3_cross_model_ood_notebook.py}"
NOTEBOOK_OUTPUT="${NOTEBOOK_OUTPUT:-$PROJECT_ROOT/rebuttal/notebooks/main3_cross_model_ood_analysis.ipynb}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/rebuttal/results/OOD_Modeling_main3_cross_model_ood_xgb_pca_128}"
SHARDS_ROOT="${SHARDS_ROOT:-$OUTPUT_ROOT/shards}"
FEATURE_SIZES="${FEATURE_SIZES:-128}"
TOP_FEATURES_TO_SHOW="${TOP_FEATURES_TO_SHOW:-20}"
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
  cmd=(uv run python "$AGGREGATE_SCRIPT_PATH")
else
  cmd=("$PYTHON_BIN" "$AGGREGATE_SCRIPT_PATH")
fi

cmd+=(
  --output-root "$OUTPUT_ROOT"
  --shards-root "$SHARDS_ROOT"
  --feature-sizes "$FEATURE_SIZES"
  --top-features-to-show "$TOP_FEATURES_TO_SHOW"
)

mkdir -p "$OUTPUT_ROOT" "$SHARDS_ROOT"

echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "AGGREGATE_SCRIPT_PATH: $AGGREGATE_SCRIPT_PATH"
echo "NOTEBOOK_BUILDER_PATH: $NOTEBOOK_BUILDER_PATH"
echo "NOTEBOOK_OUTPUT: $NOTEBOOK_OUTPUT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "SHARDS_ROOT: $SHARDS_ROOT"
echo "FEATURE_SIZES: $FEATURE_SIZES"
echo "TOP_FEATURES_TO_SHOW: $TOP_FEATURES_TO_SHOW"
echo "BUILD_NOTEBOOK: $BUILD_NOTEBOOK"
echo "CONDA_ENV: $CONDA_ENV"
echo "PYTHON_BIN: ${PYTHON_BIN:-uv run python}"
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
