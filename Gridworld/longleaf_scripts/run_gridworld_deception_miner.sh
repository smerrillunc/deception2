#!/usr/bin/env bash
set -euo pipefail

# Submit single-GPU Gridworld miner SLURM job.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$script_dir/run_gridworld_deception_miner_slurm.sh"
