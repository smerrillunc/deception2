#!/bin/bash
#SBATCH --job-name=dm_cj_tau
#SBATCH --output=dataset_scripts/logs/dm_cj_tau_%j.out
#SBATCH --error=dataset_scripts/logs/dm_cj_tau_%j.err
#SBATCH --chdir=/work/users/s/m/smerrill/deception2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80g
#SBATCH --time=2-00:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/users/s/m/smerrill/deception2}"
SCRIPT_PATH="$PROJECT_ROOT/dataset_scripts/commitment_juncture_threshold.py"
DATASETMAIN_ROOT="${DATASETMAIN_ROOT:-$PROJECT_ROOT/DatasetMain}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/dataset_scripts/outputs/commitment_juncture_threshold}"
CONDA_ENV="${CONDA_ENV:-deception}"
MAX_JSON_FILES_PER_BUNDLE="${MAX_JSON_FILES_PER_BUNDLE:-}"
MIN_VALID="${MIN_VALID:-10}"
PREFERRED_SOURCE="${PREFERRED_SOURCE:-localization_json}"
DEFAULT_TAU="${DEFAULT_TAU:-0.3}"
PROGRESS_LEVEL="${PROGRESS_LEVEL:-bundle}"
LOAD_ARTIFACTS_IF_AVAILABLE="${LOAD_ARTIFACTS_IF_AVAILABLE:-1}"
SAVE_PAIR_ARTIFACTS="${SAVE_PAIR_ARTIFACTS:-1}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export MPLBACKEND=Agg

mkdir -p "$PROJECT_ROOT/dataset_scripts/logs" "$OUTPUT_DIR"

CMD=(
  python "$SCRIPT_PATH"
  --repo-root "$PROJECT_ROOT"
  --datasetmain-root "$DATASETMAIN_ROOT"
  --output-dir "$OUTPUT_DIR"
  --min-valid "$MIN_VALID"
  --preferred-source "$PREFERRED_SOURCE"
  --default-tau "$DEFAULT_TAU"
  --progress-level "$PROGRESS_LEVEL"
)

if [[ -n "$MAX_JSON_FILES_PER_BUNDLE" ]]; then
  CMD+=(--max-json-files-per-bundle "$MAX_JSON_FILES_PER_BUNDLE")
fi

if [[ "$LOAD_ARTIFACTS_IF_AVAILABLE" == "1" ]]; then
  CMD+=(--load-artifacts-if-available)
else
  CMD+=(--no-load-artifacts-if-available)
fi

if [[ "$SAVE_PAIR_ARTIFACTS" == "1" ]]; then
  CMD+=(--save-pair-artifacts)
else
  CMD+=(--no-save-pair-artifacts)
fi

if [[ "$SHOW_PROGRESS" == "1" ]]; then
  CMD+=(--show-progress)
else
  CMD+=(--no-show-progress)
fi

printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
