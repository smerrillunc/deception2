#!/bin/bash
set -euo pipefail

# Usage:
#   ./submit_sentence_localization_batch.sh          # defaults to 8 shards
#   ./submit_sentence_localization_batch.sh 16       # 16 shards

N="${1:-8}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-rc_amcavoy_pi}"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
  echo "N must be a positive integer. Got: $N"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_sentence_localization_slurm.sh"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT"
  exit 1
fi

extract_literal_assignment() {
  local key="$1"
  local raw_value

  raw_value="$(sed -n "s/^${key}=//p" "$RUN_SCRIPT" | head -n 1)"
  raw_value="${raw_value%%#*}"
  raw_value="${raw_value#"${raw_value%%[![:space:]]*}"}"
  raw_value="${raw_value%"${raw_value##*[![:space:]]}"}"

  if [[ "$raw_value" == \"*\" && "$raw_value" == *\" ]]; then
    raw_value="${raw_value:1:-1}"
  elif [[ "$raw_value" == \'*\' && "$raw_value" == *\' ]]; then
    raw_value="${raw_value:1:-1}"
  fi

  printf '%s' "$raw_value"
}

build_job_name() {
  local env_name="$1"
  local model_tail="$2"
  local job_name="sentence_loc_${env_name}_${model_tail}"

  job_name="${job_name//[^[:alnum:]_.-]/_}"
  printf '%s' "${job_name:0:120}"
}

GAME="$(extract_literal_assignment GAME)"
MODEL_NAME="$(extract_literal_assignment MODEL_NAME)"
MODEL_TAIL="${MODEL_NAME##*/}"
JOB_NAME="$(build_job_name "$GAME" "$MODEL_TAIL")"

echo "Submitting $N shard jobs (array 0-$((N-1)))"
echo "Using run script: $RUN_SCRIPT"
echo "Using account: $SBATCH_ACCOUNT"
echo "Using job name: $JOB_NAME"

ARRAY_JOB_ID="$(sbatch --account "$SBATCH_ACCOUNT" --job-name "$JOB_NAME" --parsable --array=0-$((N-1)) --export=ALL,NUM_SHARDS="$N" "$RUN_SCRIPT")"

echo "Submitted array job: $ARRAY_JOB_ID"
echo "Array tasks: ${ARRAY_JOB_ID}_[0-$((N-1))]"
echo "Check tasks:"
echo "  squeue -j $ARRAY_JOB_ID -r"
echo "  scontrol show job $ARRAY_JOB_ID"
