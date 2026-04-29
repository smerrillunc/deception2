#!/bin/bash
#SBATCH --job-name=dm_loc_sum
#SBATCH --output=dataset_scripts/logs/dm_loc_sum_%j.out
#SBATCH --error=dataset_scripts/logs/dm_loc_sum_%j.err
#SBATCH --chdir=/work/users/s/m/smerrill/deception2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=2-00:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/users/s/m/smerrill/deception2}"
SCRIPT_PATH="$PROJECT_ROOT/dataset_scripts/datasetmain_localization_dataset_summary.py"
DATASETMAIN_ROOT="${DATASETMAIN_ROOT:-$PROJECT_ROOT/DatasetMain}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/dataset_scripts/outputs/datasetmain_localization_dataset_summary}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-}"
CONDA_ENV="${CONDA_ENV:-deception}"
MAX_FILES_PER_BUNDLE="${MAX_FILES_PER_BUNDLE:-}"
TOKEN_COUNT_MODE="${TOKEN_COUNT_MODE:-hf}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROGRESS_LEVEL="${PROGRESS_LEVEL:-bundle}"
FORCE_REBUILD_BUNDLE_SUMMARY="${FORCE_REBUILD_BUNDLE_SUMMARY:-0}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$PROJECT_ROOT/dataset_scripts/logs" "$OUTPUT_DIR"

CMD=(
  python "$SCRIPT_PATH"
  --repo-root "$PROJECT_ROOT"
  --datasetmain-root "$DATASETMAIN_ROOT"
  --output-dir "$OUTPUT_DIR"
  --token-count-mode "$TOKEN_COUNT_MODE"
  --num-workers "$NUM_WORKERS"
  --progress-level "$PROGRESS_LEVEL"
)

if [[ -n "$HF_CACHE_ROOT" ]]; then
  CMD+=(--hf-cache-root "$HF_CACHE_ROOT")
fi

if [[ -n "$MAX_FILES_PER_BUNDLE" ]]; then
  CMD+=(--max-files-per-bundle "$MAX_FILES_PER_BUNDLE")
fi

if [[ "$FORCE_REBUILD_BUNDLE_SUMMARY" == "1" ]]; then
  CMD+=(--force-rebuild-bundle-summary)
fi

if [[ "$SHOW_PROGRESS" == "1" ]]; then
  CMD+=(--show-progress)
else
  CMD+=(--no-show-progress)
fi

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
