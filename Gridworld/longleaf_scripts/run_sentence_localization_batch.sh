#!/usr/bin/env bash
set -euo pipefail

# Submit single-GPU SLURM localization jobs.
# TARGET: deceptive | truthful | both
TARGET="${TARGET:-deceptive}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

submit_job() {
  local script_path="$1"
  echo "Submitting: $script_path"
  sbatch "$script_path"
}

case "$TARGET" in
  deceptive)
    submit_job "$script_dir/run_sentence_localization_deceptive_slurm.sh"
    ;;
  truthful)
    submit_job "$script_dir/run_sentence_localization_truthful_slurm.sh"
    ;;
  both)
    submit_job "$script_dir/run_sentence_localization_deceptive_slurm.sh"
    submit_job "$script_dir/run_sentence_localization_truthful_slurm.sh"
    ;;
  *)
    echo "Invalid TARGET=$TARGET. Expected one of: deceptive, truthful, both"
    exit 1
    ;;
esac
