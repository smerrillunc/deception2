#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entrypoint: Longleaf uses single-GPU jobs.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$script_dir/run_gridworld_deception_miner_slurm.sh"
