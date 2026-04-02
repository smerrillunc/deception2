#!/bin/bash
set -euo pipefail

# Usage:
#   ./submit_commitment_prefix_features_batch.sh
#   ./submit_commitment_prefix_features_batch.sh 50
#   ./submit_commitment_prefix_features_batch.sh 50 bs gridworld

N="${1:-50}"
if [[ $# -gt 0 ]]; then
  shift
fi

DEFAULT_ENVIRONMENTS=(bs gridworld advisor_audit interview car_sales)
if [[ $# -gt 0 ]]; then
  ENVIRONMENTS=("$@")
else
  ENVIRONMENTS=("${DEFAULT_ENVIRONMENTS[@]}")
fi

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-rc_amcavoy_pi}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
  echo "N must be a positive integer. Got: $N"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_project_root() {
  local candidate
  for candidate in \
    "${PROJECT_ROOT:-}" \
    "/work/users/s/m/smerrill/deception2" \
    "/playpen-ssd/smerrill/deception2" \
    "$(cd "$SCRIPT_DIR/.." && pwd)"
  do
    if [[ -n "$candidate" && -d "$candidate/src" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

PROJECT_ROOT="$(resolve_project_root)" || {
  echo "Could not resolve PROJECT_ROOT. Set PROJECT_ROOT explicitly before submitting."
  exit 1
}
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"
RUN_SCRIPT="$SCRIPT_DIR/run_commitment_prefix_features_cpu_slurm.sh"
MODEL_TAIL="${MODEL_NAME##*/}"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT"
  exit 1
fi

is_known_environment() {
  case "$1" in
    bs|gridworld|advisor_audit|interview|car_sales) return 0 ;;
    *) return 1 ;;
  esac
}

build_job_name() {
  local env_name="$1"
  local model_tail="$2"
  local job_name="commit_prefix_${env_name}_${model_tail}"

  job_name="${job_name//[^[:alnum:]_.-]/_}"
  printf '%s' "${job_name:0:120}"
}

for env_name in "${ENVIRONMENTS[@]}"; do
  if ! is_known_environment "$env_name"; then
    echo "Unknown environment: $env_name"
    echo "Valid environments: ${DEFAULT_ENVIRONMENTS[*]}"
    exit 1
  fi

  data_dir="$DATASET_ROOT/$env_name/$MODEL_TAIL"
  if [[ ! -f "$data_dir/examples.jsonl" ]]; then
    echo "Missing examples file for $env_name: $data_dir/examples.jsonl"
    exit 1
  fi
  if [[ ! -d "$data_dir/localization" ]]; then
    echo "Missing localization directory for $env_name: $data_dir/localization"
    exit 1
  fi
done

declare -a JOB_IDS=()

echo "Submitting ${#ENVIRONMENTS[@]} environment batch(es)"
echo "Shards per environment: $N"
echo "Using run script: $RUN_SCRIPT"
echo "Using account: $SBATCH_ACCOUNT"
echo "Using model: $MODEL_NAME"

for env_name in "${ENVIRONMENTS[@]}"; do
  job_name="$(build_job_name "$env_name" "$MODEL_TAIL")"
  echo
  echo "Submitting environment: $env_name"
  echo "Using job name: $job_name"

  array_job_id="$(sbatch \
    --account "$SBATCH_ACCOUNT" \
    --job-name "$job_name" \
    --parsable \
    --array=0-$((N-1)) \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",DATASET_ROOT="$DATASET_ROOT",NUM_SHARDS="$N",GAME="$env_name",MODEL_NAME="$MODEL_NAME" \
    "$RUN_SCRIPT")"

  JOB_IDS+=("$array_job_id")
  echo "Submitted array job for $env_name: $array_job_id"
  echo "Array tasks: ${array_job_id}_[0-$((N-1))]"
done

echo
echo "Submitted ${#JOB_IDS[@]} array job(s):"
for idx in "${!JOB_IDS[@]}"; do
  echo "  ${ENVIRONMENTS[$idx]}: ${JOB_IDS[$idx]}"
done

echo "Check tasks:"
echo "  squeue -u $USER"
