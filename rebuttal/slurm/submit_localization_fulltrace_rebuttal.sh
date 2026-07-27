#!/bin/bash
set -euo pipefail

RUN_NAME="${1:-localization_fulltrace_vs_adaptive_rebuttal_v1}"
MANIFEST_KIND="${2:-all}"   # all | adaptive | full
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-rc_amcavoy_pi}"
PROJECT_ROOT="${PROJECT_ROOT:-/work/users/s/m/smerrill/deception2}"

case "$MANIFEST_KIND" in
  all) MANIFEST_FILENAME="run_manifest.csv" ;;
  adaptive) MANIFEST_FILENAME="run_manifest_adaptive.csv" ;;
  full) MANIFEST_FILENAME="run_manifest_full.csv" ;;
  *)
    echo "Unsupported manifest kind: $MANIFEST_KIND" >&2
    echo "Expected one of: all, adaptive, full" >&2
    exit 1
    ;;
esac

RUN_ROOT="$PROJECT_ROOT/rebuttal/results/$RUN_NAME"
MANIFEST_PATH="$RUN_ROOT/$MANIFEST_FILENAME"
RUN_SCRIPT="$PROJECT_ROOT/rebuttal/slurm/run_localization_fulltrace_rebuttal_slurm.sh"
LOG_DIR="$RUN_ROOT/slurm_logs"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Missing manifest: $MANIFEST_PATH" >&2
  echo "Run the prep script first on the /work repo clone." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

TASK_COUNT="$(python -c "import pandas as pd; import sys; df = pd.read_csv(sys.argv[1]); print(len(df))" "$MANIFEST_PATH")"
if ! [[ "$TASK_COUNT" =~ ^[0-9]+$ ]] || [[ "$TASK_COUNT" -lt 1 ]]; then
  echo "Manifest has no runnable tasks: $MANIFEST_PATH" >&2
  exit 1
fi

JOB_NAME="locreb_${MANIFEST_KIND}"
ARRAY_SPEC="0-$((TASK_COUNT - 1))"

echo "Submitting $TASK_COUNT localization tasks"
echo "Run root: $RUN_ROOT"
echo "Manifest kind: $MANIFEST_KIND"
echo "Manifest path: $MANIFEST_PATH"
echo "Using account: $SBATCH_ACCOUNT"

ARRAY_JOB_ID="$(sbatch \
  --account "$SBATCH_ACCOUNT" \
  --job-name "$JOB_NAME" \
  --output "$LOG_DIR/${MANIFEST_KIND}_%A_%a.out" \
  --error "$LOG_DIR/${MANIFEST_KIND}_%A_%a.err" \
  --parsable \
  --array="$ARRAY_SPEC" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",RUN_NAME="$RUN_NAME",MANIFEST_KIND="$MANIFEST_KIND",MANIFEST_PATH="$MANIFEST_PATH" \
  "$RUN_SCRIPT")"

echo "Submitted array job: $ARRAY_JOB_ID"
echo "Array tasks: ${ARRAY_JOB_ID}_[${ARRAY_SPEC}]"
echo "Check status:"
echo "  squeue -j $ARRAY_JOB_ID -r"
