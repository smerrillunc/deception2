#!/bin/bash
#SBATCH --job-name=loc_rebuttal
#SBATCH --output=loc_rebuttal_%A_%a.out
#SBATCH --error=loc_rebuttal_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40g
#SBATCH --time=6-23:00:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/users/s/m/smerrill/deception2}"
CONDA_ENV="${CONDA_ENV:-deception}"
RUN_NAME="${RUN_NAME:-localization_fulltrace_vs_adaptive_rebuttal_v1}"
MANIFEST_KIND="${MANIFEST_KIND:-all}"   # all | adaptive | full
TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

case "$MANIFEST_KIND" in
  all) MANIFEST_FILENAME="run_manifest.csv" ;;
  adaptive) MANIFEST_FILENAME="run_manifest_adaptive.csv" ;;
  full) MANIFEST_FILENAME="run_manifest_full.csv" ;;
  *)
    echo "Unsupported MANIFEST_KIND: $MANIFEST_KIND" >&2
    exit 1
    ;;
esac

MANIFEST_PATH="${MANIFEST_PATH:-$PROJECT_ROOT/rebuttal/results/$RUN_NAME/$MANIFEST_FILENAME}"
TASK_RUNNER="$PROJECT_ROOT/rebuttal/scripts/run_localization_fulltrace_rebuttal_task.py"

if [[ ! -f "$TASK_RUNNER" ]]; then
  echo "Missing task runner: $TASK_RUNNER" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Missing manifest: $MANIFEST_PATH" >&2
  echo "Run the prep script first on the /work repo clone." >&2
  exit 1
fi

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Project root: $PROJECT_ROOT"
echo "Manifest kind: $MANIFEST_KIND"
echo "Manifest path: $MANIFEST_PATH"
echo "Task index: $TASK_INDEX"
echo "Conda env: $CONDA_ENV"

python "$TASK_RUNNER" \
  --manifest-path "$MANIFEST_PATH" \
  --task-index "$TASK_INDEX" \
  --project-root "$PROJECT_ROOT"
