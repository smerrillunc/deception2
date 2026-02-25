#!/usr/bin/env bash
set -euo pipefail

# Submit single-GPU BS miner SLURM job.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$script_dir/run_deception_miner_slurm.sh"
