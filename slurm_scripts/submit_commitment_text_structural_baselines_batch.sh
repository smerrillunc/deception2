#!/bin/bash
set -euo pipefail

# Usage:
#   ./submit_commitment_text_structural_baselines_batch.sh
#   ./submit_commitment_text_structural_baselines_batch.sh bs gridworld
#   MODEL_KEYS="qwen7b llama8b gptoss" ./submit_commitment_text_structural_baselines_batch.sh
#
# Notes:
# - qwen14b is intentionally excluded from the default model list for now.
# - The alias "interrview" is accepted and normalized to "interview".

DEFAULT_ENVIRONMENTS=(bs gridworld advisor_audit interview car_sales)
DEFAULT_MODEL_KEYS=(qwen7b llama8b gptoss)

declare -A MODEL_TAIL_BY_KEY=(
  [qwen7b]="DeepSeek-R1-Distill-Qwen-7B"
  [qwen14b]="DeepSeek-R1-Distill-Qwen-14B"
  [llama8b]="DeepSeek-R1-Distill-Llama-8B"
  [gptoss]="gpt-oss-20b"
  [gpt-oss20b]="gpt-oss-20b"
  [gpt-oss-20b]="gpt-oss-20b"
)

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-rc_amcavoy_pi}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_commitment_text_structural_baselines_cpu_slurm.sh"

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

normalize_environment() {
  case "$1" in
    interrview) printf '%s\n' "interview" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

is_known_environment() {
  case "$1" in
    bs|gridworld|advisor_audit|interview|car_sales) return 0 ;;
    *) return 1 ;;
  esac
}

parse_model_keys() {
  local raw="${MODEL_KEYS:-}"
  local normalized
  if [[ -z "$raw" ]]; then
    printf '%s\n' "${DEFAULT_MODEL_KEYS[@]}"
    return 0
  fi

  raw="${raw//,/ }"
  for normalized in $raw; do
    printf '%s\n' "$normalized"
  done
}

build_job_name() {
  local env_name="$1"
  local model_key="$2"
  local job_name="commit_txt_struct_${env_name}_${model_key}"
  job_name="${job_name//[^[:alnum:]_.-]/_}"
  printf '%s' "${job_name:0:120}"
}

PROJECT_ROOT="$(resolve_project_root)" || {
  echo "Could not resolve PROJECT_ROOT. Set PROJECT_ROOT explicitly before submitting." >&2
  exit 1
}
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  ENVIRONMENTS=()
  for env_name in "$@"; do
    ENVIRONMENTS+=("$(normalize_environment "$env_name")")
  done
else
  ENVIRONMENTS=("${DEFAULT_ENVIRONMENTS[@]}")
fi

MODEL_KEYS_LIST=()
while IFS= read -r model_key; do
  [[ -n "$model_key" ]] && MODEL_KEYS_LIST+=("$model_key")
done < <(parse_model_keys)

if [[ "${#MODEL_KEYS_LIST[@]}" -eq 0 ]]; then
  echo "No model keys specified." >&2
  exit 1
fi

for env_name in "${ENVIRONMENTS[@]}"; do
  if ! is_known_environment "$env_name"; then
    echo "Unknown environment: $env_name" >&2
    echo "Valid environments: ${DEFAULT_ENVIRONMENTS[*]}" >&2
    exit 1
  fi
done

for model_key in "${MODEL_KEYS_LIST[@]}"; do
  if [[ -z "${MODEL_TAIL_BY_KEY[$model_key]:-}" ]]; then
    echo "Unknown model key: $model_key" >&2
    echo "Known model keys: ${!MODEL_TAIL_BY_KEY[*]}" >&2
    exit 1
  fi
done

for env_name in "${ENVIRONMENTS[@]}"; do
  for model_key in "${MODEL_KEYS_LIST[@]}"; do
    model_tail="${MODEL_TAIL_BY_KEY[$model_key]}"
    data_dir="$DATASET_ROOT/$env_name/$model_tail"
    if [[ ! -f "$data_dir/examples.jsonl" ]]; then
      echo "Missing examples file for $env_name / $model_key: $data_dir/examples.jsonl" >&2
      exit 1
    fi
    if [[ ! -d "$data_dir/localization" ]]; then
      echo "Missing localization directory for $env_name / $model_key: $data_dir/localization" >&2
      exit 1
    fi
  done
done

declare -a JOB_IDS=()
declare -a JOB_LABELS=()

echo "Submitting commitment text/structural baseline jobs"
echo "Using run script: $RUN_SCRIPT"
echo "Using account: $SBATCH_ACCOUNT"
echo "Project root: $PROJECT_ROOT"
echo "Dataset root: $DATASET_ROOT"
echo "Environments: ${ENVIRONMENTS[*]}"
echo "Models: ${MODEL_KEYS_LIST[*]}"
echo "qwen14b is excluded by default."

for env_name in "${ENVIRONMENTS[@]}"; do
  for model_key in "${MODEL_KEYS_LIST[@]}"; do
    model_tail="${MODEL_TAIL_BY_KEY[$model_key]}"
    job_name="$(build_job_name "$env_name" "$model_key")"
    echo
    echo "Submitting environment=$env_name model=$model_key"
    echo "  model tail: $model_tail"
    echo "  job name:   $job_name"

    job_id="$(sbatch \
      --account "$SBATCH_ACCOUNT" \
      --job-name "$job_name" \
      --parsable \
      --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",DATASET_ROOT="$DATASET_ROOT",GAME="$env_name",MODEL_KEY="$model_key",MODEL_TAIL="$model_tail" \
      "$RUN_SCRIPT")"

    JOB_IDS+=("$job_id")
    JOB_LABELS+=("${env_name}:${model_key}")
    echo "  submitted:  $job_id"
  done
done

echo
echo "Submitted ${#JOB_IDS[@]} job(s):"
for idx in "${!JOB_IDS[@]}"; do
  echo "  ${JOB_LABELS[$idx]} -> ${JOB_IDS[$idx]}"
done

echo "Check jobs:"
echo "  squeue -u $USER"
