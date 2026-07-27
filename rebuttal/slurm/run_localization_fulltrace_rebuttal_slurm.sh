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
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
RUN_NAME="${RUN_NAME:-localization_fulltrace_vs_adaptive_rebuttal_v1}"
MANIFEST_KIND="${MANIFEST_KIND:-full}"   # full | all(alias of full)
TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

case "$MANIFEST_KIND" in
  all) MANIFEST_FILENAME="run_manifest.csv" ;;
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

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi

  if [[ -n "${CONDA_ENV_PATH:-}" && -x "${CONDA_ENV_PATH}/bin/python" ]]; then
    printf '%s\n' "${CONDA_ENV_PATH}/bin/python"
    return
  fi

  local candidates=(
    "/work/users/s/m/smerrill/conda_envs/deception/bin/python"
    "/playpen-ssd/smerrill/conda_envs/deception/bin/python"
  )
  local cand
  for cand in "${candidates[@]}"; do
    if [[ -x "$cand" ]]; then
      printf '%s\n' "$cand"
      return
    fi
  done

  return 1
}

if ! PYTHON_BIN="$(resolve_python_bin)"; then
  module load anaconda
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHON_BIN

# Longleaf's MKL defaults can conflict with libgomp-loaded deps inside the
# localization subprocess. Force the GNU threading layer unless the user has
# explicitly overridden it.
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-/tmp/vllm}"
mkdir -p "$VLLM_CONFIG_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Manifest kind: $MANIFEST_KIND"
echo "Manifest path: $MANIFEST_PATH"
echo "Task index: $TASK_INDEX"
echo "Conda env: $CONDA_ENV"
echo "Conda env path: ${CONDA_ENV_PATH:-<unset>}"
echo "Python bin: $PYTHON_BIN"
echo "MKL_THREADING_LAYER: $MKL_THREADING_LAYER"
echo "VLLM_CONFIG_ROOT: $VLLM_CONFIG_ROOT"

"$PYTHON_BIN" -c "import sys; print('python=', sys.executable)"

"$PYTHON_BIN" "$TASK_RUNNER" \
  --manifest-path "$MANIFEST_PATH" \
  --task-index "$TASK_INDEX" \
  --project-root "$PROJECT_ROOT"
