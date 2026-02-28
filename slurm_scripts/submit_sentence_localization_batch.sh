#!/bin/bash
set -euo pipefail

# Usage:
#   ./submit_sentence_localization_batch.sh          # defaults to 8 shards
#   ./submit_sentence_localization_batch.sh 16       # 16 shards

N="${1:-8}"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
  echo "N must be a positive integer. Got: $N"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_sentence_localization_slurm.sh"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Missing run script: $RUN_SCRIPT"
  exit 1
fi

echo "Submitting $N shard jobs (array 0-$((N-1)))"
echo "Using run script: $RUN_SCRIPT"

sbatch --array=0-$((N-1)) --export=ALL,NUM_SHARDS="$N" "$RUN_SCRIPT"
