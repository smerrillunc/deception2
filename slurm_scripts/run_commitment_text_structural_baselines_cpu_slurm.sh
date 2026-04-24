#!/bin/bash
#SBATCH --job-name=commit_txt_struct
#SBATCH --output=commit_txt_struct_%j.out
#SBATCH --error=commit_txt_struct_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=2-00:00:00

set -euo pipefail

# ---------------- User parameters ----------------
CONDA_ENV="${CONDA_ENV:-deception}"
RECENT_WINDOW_SENTENCES="${RECENT_WINDOW_SENTENCES:-5}"
DELTA_THRESHOLD="${DELTA_THRESHOLD:-0.3}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
WRITE_EVERY_EXAMPLES="${WRITE_EVERY_EXAMPLES:-32}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_TFIDF_CACHE="${OVERWRITE_TFIDF_CACHE:-0}"
STRICT="${STRICT:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
COMPUTE_TFIDF="${COMPUTE_TFIDF:-1}"
TFIDF_TEXT_FIELDS="${TFIDF_TEXT_FIELDS:-last_sentence_text,prefix_text}"
TFIDF_MAX_FEATURES="${TFIDF_MAX_FEATURES:-20000}"
TFIDF_MIN_NGRAM="${TFIDF_MIN_NGRAM:-1}"
TFIDF_MAX_NGRAM="${TFIDF_MAX_NGRAM:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CLEAN_STALE_OUTPUTS="${CLEAN_STALE_OUTPUTS:-1}"
# ---------------- End parameters -----------------

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

normalize_environment() {
  case "$1" in
    interrview) printf '%s\n' "interview" ;;
    *) printf '%s\n' "$1" ;;
  esac
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
SRC_ROOT="$PROJECT_ROOT/src"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/DatasetMain}"

GAME="$(normalize_environment "${GAME:-bs}")"
MODEL_KEY="${MODEL_KEY:-qwen7b}"
MODEL_TAIL="${MODEL_TAIL:-DeepSeek-R1-Distill-Qwen-7B}"

module load anaconda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

PYTHON_BIN="${PYTHON_BIN:-$CONDA_PREFIX/bin/python}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

JOB_NAME="$(build_job_name "$GAME" "$MODEL_KEY")"
DATA_DIR="${DATA_DIR:-$DATASET_ROOT/$GAME/$MODEL_TAIL}"
LOCALIZATION_DIR="$DATA_DIR/localization"
EXAMPLES_PATH="$DATA_DIR/examples.jsonl"
OUTPUT_PATH="${OUTPUT_PATH:-$DATA_DIR/commitment_text_structural_baselines.parquet}"
TMP_OUTPUT_PATH="${OUTPUT_PATH}.tmp"
TFIDF_CACHE_DIR="${TFIDF_CACHE_DIR:-$DATA_DIR/commitment_text_baseline_tfidf_cache}"

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  if scontrol update JobId="$SLURM_JOB_ID" JobName="$JOB_NAME" >/dev/null 2>&1; then
    echo "SLURM job name: $JOB_NAME"
  else
    echo "Warning: failed to update SLURM job name to $JOB_NAME" >&2
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$SRC_ROOT/deception_prefix_text_structural_baseline_extractor.py" ]]; then
  echo "Missing script: $SRC_ROOT/deception_prefix_text_structural_baseline_extractor.py" >&2
  exit 1
fi
if [[ ! -f "$EXAMPLES_PATH" ]]; then
  echo "Missing examples file: $EXAMPLES_PATH" >&2
  exit 1
fi
if [[ ! -d "$LOCALIZATION_DIR" ]]; then
  echo "Missing localization directory: $LOCALIZATION_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

if [[ -e "$TMP_OUTPUT_PATH" ]]; then
  if [[ "$CLEAN_STALE_OUTPUTS" == "1" ]]; then
    echo "Removing stale temporary output: $TMP_OUTPUT_PATH"
    rm -f "$TMP_OUTPUT_PATH"
  else
    echo "Temporary output exists and cleanup is disabled: $TMP_OUTPUT_PATH" >&2
    echo "Set CLEAN_STALE_OUTPUTS=1 or delete it manually before rerunning." >&2
    exit 1
  fi
fi

IFS=',' read -r -a TFIDF_FIELDS_ARRAY <<< "$TFIDF_TEXT_FIELDS"
EXPECTED_TFIDF_ARTIFACTS=0
for text_field in "${TFIDF_FIELDS_ARRAY[@]}"; do
  text_field="${text_field//[[:space:]]/}"
  if [[ -n "$text_field" ]]; then
    EXPECTED_TFIDF_ARTIFACTS=$((EXPECTED_TFIDF_ARTIFACTS + 1))
  fi
done

tfidf_cache_complete() {
  local npz_count
  local joblib_count
  local feature_names_count
  local json_count

  if [[ "$COMPUTE_TFIDF" != "1" ]]; then
    return 0
  fi
  if [[ "$EXPECTED_TFIDF_ARTIFACTS" -le 0 ]] || [[ ! -d "$TFIDF_CACHE_DIR" ]]; then
    return 1
  fi

  npz_count=$(find "$TFIDF_CACHE_DIR" -maxdepth 1 -name '*.npz' | wc -l)
  joblib_count=$(find "$TFIDF_CACHE_DIR" -maxdepth 1 -name '*.joblib' | wc -l)
  feature_names_count=$(find "$TFIDF_CACHE_DIR" -maxdepth 1 -name '*__feature_names.npy' | wc -l)
  json_count=$(find "$TFIDF_CACHE_DIR" -maxdepth 1 -name '*.json' | wc -l)

  [[ "$npz_count" -ge "$EXPECTED_TFIDF_ARTIFACTS" ]] \
    && [[ "$joblib_count" -ge "$EXPECTED_TFIDF_ARTIFACTS" ]] \
    && [[ "$feature_names_count" -ge "$EXPECTED_TFIDF_ARTIFACTS" ]] \
    && [[ "$json_count" -ge "$EXPECTED_TFIDF_ARTIFACTS" ]]
}

AUTO_OVERWRITE_OUTPUT=0
if [[ -e "$OUTPUT_PATH" ]]; then
  if [[ "$SKIP_EXISTING" == "1" ]] && tfidf_cache_complete; then
    echo "Output parquet and TF-IDF cache already exist; skipping."
    echo "  Output: $OUTPUT_PATH"
    echo "  TF-IDF cache: $TFIDF_CACHE_DIR"
    exit 0
  fi
  AUTO_OVERWRITE_OUTPUT=1
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false

CMD=(
  "$PYTHON_BIN" "$SRC_ROOT/deception_prefix_text_structural_baseline_extractor.py"
  "$DATA_DIR"
  --output "$OUTPUT_PATH"
  --recent-window-sentences "$RECENT_WINDOW_SENTENCES"
  --delta-threshold "$DELTA_THRESHOLD"
  --write-every-examples "$WRITE_EVERY_EXAMPLES"
  --progress-every "$PROGRESS_EVERY"
)

if [[ "$MAX_EXAMPLES" -gt 0 ]]; then
  CMD+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ "$OVERWRITE" == "1" || "$AUTO_OVERWRITE_OUTPUT" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ "$STRICT" == "1" ]]; then
  CMD+=(--strict)
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi
if [[ "$COMPUTE_TFIDF" == "1" ]]; then
  CMD+=(
    --compute-tfidf
    --tfidf-text-fields "$TFIDF_TEXT_FIELDS"
    --tfidf-cache-dir "$TFIDF_CACHE_DIR"
    --tfidf-max-features "$TFIDF_MAX_FEATURES"
    --tfidf-min-ngram "$TFIDF_MIN_NGRAM"
    --tfidf-max-ngram "$TFIDF_MAX_NGRAM"
  )
  if [[ "$OVERWRITE_TFIDF_CACHE" == "1" ]]; then
    CMD+=(--overwrite-tfidf-cache)
  fi
fi

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
echo "Dataset dir: $DATA_DIR"
echo "Localization dir: $LOCALIZATION_DIR"
echo "Output parquet: $OUTPUT_PATH"
if [[ "$COMPUTE_TFIDF" == "1" ]]; then
  echo "TF-IDF cache dir: $TFIDF_CACHE_DIR"
fi
echo "CPU threads per task: $THREADS"
if [[ "$AUTO_OVERWRITE_OUTPUT" == "1" && "$OVERWRITE" != "1" ]]; then
  echo "Output already existed; rerunning with --overwrite to regenerate parquet before TF-IDF caching."
fi

"${CMD[@]}"

echo "commitment text/structural baseline extraction complete."
