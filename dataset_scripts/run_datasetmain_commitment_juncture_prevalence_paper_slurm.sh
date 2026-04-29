#!/bin/bash
#SBATCH --job-name=dm_cj_prev
#SBATCH --output=dataset_scripts/logs/dm_cj_prev_%j.out
#SBATCH --error=dataset_scripts/logs/dm_cj_prev_%j.err
#SBATCH --chdir=/work/users/s/m/smerrill/deception2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=2-00:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/users/s/m/smerrill/deception2}"
SCRIPT_PATH="$PROJECT_ROOT/dataset_scripts/datasetmain_commitment_juncture_prevalence_paper.py"
DATASETMAIN_ROOT="${DATASETMAIN_ROOT:-$PROJECT_ROOT/DatasetMain}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/dataset_scripts/outputs/datasetmain_commitment_juncture_prevalence_paper}"
CONDA_ENV="${CONDA_ENV:-deception}"
MAX_JSON_FILES_PER_BUNDLE="${MAX_JSON_FILES_PER_BUNDLE:-}"
BOOTSTRAP_NUM_RESAMPLES="${BOOTSTRAP_NUM_RESAMPLES:-1000}"
PROGRESS_LEVEL="${PROGRESS_LEVEL:-bundle}"
FORCE_REBUILD_SUMMARIES="${FORCE_REBUILD_SUMMARIES:-0}"
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
  --bootstrap-num-resamples "$BOOTSTRAP_NUM_RESAMPLES"
  --progress-level "$PROGRESS_LEVEL"
)

if [[ -n "$MAX_JSON_FILES_PER_BUNDLE" ]]; then
  CMD+=(--max-json-files-per-bundle "$MAX_JSON_FILES_PER_BUNDLE")
fi

if [[ "$FORCE_REBUILD_SUMMARIES" == "1" ]]; then
  CMD+=(--force-rebuild-summaries)
fi

if [[ "$SHOW_PROGRESS" == "1" ]]; then
  CMD+=(--show-progress)
else
  CMD+=(--no-show-progress)
fi

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
