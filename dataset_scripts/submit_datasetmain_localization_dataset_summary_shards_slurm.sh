#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/users/s/m/smerrill/deception2}"
WORKER_SCRIPT="$PROJECT_ROOT/dataset_scripts/run_datasetmain_localization_dataset_summary_slurm.sh"
DATASETMAIN_ROOT="${DATASETMAIN_ROOT:-$PROJECT_ROOT/DatasetMain}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$PROJECT_ROOT/dataset_scripts/outputs/datasetmain_localization_dataset_summary_sharded}"
SHARD_OUTPUT_ROOT="${SHARD_OUTPUT_ROOT:-$BASE_OUTPUT_DIR/shards}"
COMBINE_OUTPUT_DIR="${COMBINE_OUTPUT_DIR:-$BASE_OUTPUT_DIR/combined}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-}"
CONDA_ENV="${CONDA_ENV:-deception}"
MAX_FILES_PER_BUNDLE="${MAX_FILES_PER_BUNDLE:-}"
TOKEN_COUNT_MODE="${TOKEN_COUNT_MODE:-hf}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROGRESS_LEVEL="${PROGRESS_LEVEL:-bundle}"
FORCE_REBUILD_BUNDLE_SUMMARY="${FORCE_REBUILD_BUNDLE_SUMMARY:-0}"
SHOW_PROGRESS="${SHOW_PROGRESS:-0}"
LOAD_BUNDLE_SUMMARY_CACHE="${LOAD_BUNDLE_SUMMARY_CACHE:-1}"
SAVE_BUNDLE_SUMMARY_CACHE="${SAVE_BUNDLE_SUMMARY_CACHE:-1}"
SUBMIT_COMBINE="${SUBMIT_COMBINE:-1}"

mkdir -p "$SHARD_OUTPUT_ROOT" "$COMBINE_OUTPUT_DIR"

if [[ ! -d "$DATASETMAIN_ROOT" ]]; then
  echo "DatasetMain root does not exist: $DATASETMAIN_ROOT" >&2
  exit 1
fi

slugify() {
  local raw_slug="$1"
  local slug
  slug="$(printf '%s' "$raw_slug" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^[._-]+//; s/[._-]+$//')"
  if [[ -z "$slug" ]]; then
    slug="shard"
  fi
  printf '%s\n' "$slug"
}

mapfile -t SHARD_ROWS < <(
  while IFS= read -r bundle_dir; do
    env_name="$(basename "$(dirname "$bundle_dir")")"
    model_name="$(basename "$bundle_dir")"
    shard_slug="$(slugify "${env_name}__${model_name}")"
    printf '%s\t%s\t%s\t%s\n' "$env_name" "$model_name" "$shard_slug" "$bundle_dir"
  done < <(find "$DATASETMAIN_ROOT" -mindepth 2 -maxdepth 2 -type d | LC_ALL=C sort)
)

if [[ ${#SHARD_ROWS[@]} -eq 0 ]]; then
  echo "No DatasetMain dataset/model shards were found under $DATASETMAIN_ROOT." >&2
  exit 1
fi

declare -a JOB_IDS=()

for shard_row in "${SHARD_ROWS[@]}"; do
  IFS=$'\t' read -r env_name model_name shard_slug bundle_dir <<< "$shard_row"
  if [[ -z "${env_name:-}" || -z "${model_name:-}" || -z "${shard_slug:-}" ]]; then
    echo "Malformed shard row: $shard_row" >&2
    exit 1
  fi

  shard_output_dir="$SHARD_OUTPUT_ROOT/$shard_slug"
  mkdir -p "$shard_output_dir"

  job_id="$(
    sbatch \
      --parsable \
      --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",DATASETMAIN_ROOT="$DATASETMAIN_ROOT",OUTPUT_DIR="$shard_output_dir",HF_CACHE_ROOT="$HF_CACHE_ROOT",CONDA_ENV="$CONDA_ENV",MAX_FILES_PER_BUNDLE="$MAX_FILES_PER_BUNDLE",TOKEN_COUNT_MODE="$TOKEN_COUNT_MODE",NUM_WORKERS="$NUM_WORKERS",PROGRESS_LEVEL="$PROGRESS_LEVEL",FORCE_REBUILD_BUNDLE_SUMMARY="$FORCE_REBUILD_BUNDLE_SUMMARY",SHOW_PROGRESS="$SHOW_PROGRESS",ENV_NAME="$env_name",MODEL_NAME="$model_name",LOAD_BUNDLE_SUMMARY_CACHE="$LOAD_BUNDLE_SUMMARY_CACHE",SAVE_BUNDLE_SUMMARY_CACHE="$SAVE_BUNDLE_SUMMARY_CACHE" \
      "$WORKER_SCRIPT"
  )"
  JOB_IDS+=("$job_id")
  printf 'Submitted shard %s / %s -> %s (job %s)\n' "$env_name" "$model_name" "$shard_output_dir" "$job_id"
done

printf 'Submitted %d shard jobs.\n' "${#JOB_IDS[@]}"

if [[ "$SUBMIT_COMBINE" != "1" ]]; then
  echo "Shard outputs will land under: $SHARD_OUTPUT_ROOT"
  echo "Combine later with:"
  printf '  sbatch --export=ALL,PROJECT_ROOT=%q,DATASETMAIN_ROOT=%q,OUTPUT_DIR=%q,HF_CACHE_ROOT=%q,CONDA_ENV=%q,TOKEN_COUNT_MODE=%q,NUM_WORKERS=1,PROGRESS_LEVEL=bundle,FORCE_REBUILD_BUNDLE_SUMMARY=0,SHOW_PROGRESS=0,COMBINE_SHARD_OUTPUT_ROOT=%q,LOAD_BUNDLE_SUMMARY_CACHE=0,SAVE_BUNDLE_SUMMARY_CACHE=0 %q\n' \
    "$PROJECT_ROOT" "$DATASETMAIN_ROOT" "$COMBINE_OUTPUT_DIR" "$HF_CACHE_ROOT" "$CONDA_ENV" "$TOKEN_COUNT_MODE" "$SHARD_OUTPUT_ROOT" "$WORKER_SCRIPT"
  exit 0
fi

dependency_spec="$(IFS=:; echo "${JOB_IDS[*]}")"
combine_job_id="$(
  sbatch \
    --parsable \
    --dependency="afterok:$dependency_spec" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",DATASETMAIN_ROOT="$DATASETMAIN_ROOT",OUTPUT_DIR="$COMBINE_OUTPUT_DIR",HF_CACHE_ROOT="$HF_CACHE_ROOT",CONDA_ENV="$CONDA_ENV",TOKEN_COUNT_MODE="$TOKEN_COUNT_MODE",NUM_WORKERS=1,PROGRESS_LEVEL=bundle,FORCE_REBUILD_BUNDLE_SUMMARY=0,SHOW_PROGRESS=0,COMBINE_SHARD_OUTPUT_ROOT="$SHARD_OUTPUT_ROOT",LOAD_BUNDLE_SUMMARY_CACHE=0,SAVE_BUNDLE_SUMMARY_CACHE=0 \
    "$WORKER_SCRIPT"
)"

printf 'Submitted combine job -> %s (depends on %d shard jobs)\n' "$combine_job_id" "${#JOB_IDS[@]}"
echo "Shard outputs: $SHARD_OUTPUT_ROOT"
echo "Combined output: $COMBINE_OUTPUT_DIR"
