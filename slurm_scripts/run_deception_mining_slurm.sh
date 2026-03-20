#!/bin/bash
#SBATCH --job-name=mine_pipeline
#SBATCH --output=mine_pipeline_%j.out
#SBATCH --error=mine_pipeline_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=2-00:00:00
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4

set -euo pipefail

if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "Set MODEL_NAME when submitting this job. ENVIRONMENT defaults to all." >&2
  echo 'Example: sbatch --export=ALL,MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B slurm_scripts/run_deception_mining_slurm.sh' >&2
  exit 1
fi

module load anaconda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-deception}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
  NUM_GPUS=1
fi
if (( NUM_GPUS < 1 )); then
  NUM_GPUS=1
fi

GPU_IDS=""
for ((i = 0; i < NUM_GPUS; i++)); do
  if [[ -z "$GPU_IDS" ]]; then
    GPU_IDS="$i"
  else
    GPU_IDS="$GPU_IDS $i"
  fi
done

export GPU_IDS
export PYTHON_BIN="$CONDA_PREFIX/bin/python"

echo "MODEL_NAME: $MODEL_NAME"
echo "ENVIRONMENT: ${ENVIRONMENT:-all}"
echo "GPU_IDS: $GPU_IDS"

"$REPO_ROOT/shell_scripts/run_targeted_localization_pipeline.sh" \
  --model_name "$MODEL_NAME" \
  --env "${ENVIRONMENT:-all}" \
  --mine_only
