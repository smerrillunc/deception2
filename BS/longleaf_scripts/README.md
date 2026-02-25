# Longleaf Scripts (BS)

These scripts mirror the BS pipeline scripts but avoid hardcoded `/playpen-ssd/...` paths.

Defaults:
- `PROJECT_ROOT`: inferred from script location (`.../deception2`)
- `RESULTS_ROOT`: `$PROJECT_ROOT/BS/Results`
- `CONDA_ENV`: `deception`
- `CONDA_SH`: auto-discovered from common conda locations (or set manually)

Common usage examples:

```bash
cd /work/users/s/m/smerrill/deception2/BS/longleaf_scripts
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B LABEL_FILTER=truthful_only ./run_deception_miner_multi_gpu.sh
```

```bash
cd /work/users/s/m/smerrill/deception2/BS/longleaf_scripts
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B LABEL_FILTER=truthful_only AUTO_BUILD_DATASET=1 ./run_sentence_localization_batch_multi_gpu.sh
```

Sync full results folders from this machine to Longleaf:

```bash
cd /playpen-ssd/smerrill/deception2/BS/longleaf_scripts
LONGLEAF_HOST=smerrill@longleaf.unc.edu REMOTE_ROOT=/work/users/s/m/smerrill/deception2 ./sync_qwen14b_to_longleaf.sh
```

This syncs these full directories by default:
- `/playpen-ssd/smerrill/deception2/BS/Results`
- `/playpen-ssd/smerrill/deception2/Gridworld/Results`

Optional sync flags:
- `DRY_RUN=1` to preview transfers
- `SSH_KEY_PATH=~/.ssh/id_ed25519 SSH_IDENTITIES_ONLY=1` to force a specific key
- `USE_SSH_MUX=1` (default) to prompt for password once and reuse the SSH session
- `CLOSE_SSH_MUX=1` to close the shared SSH connection when the sync finishes
- `RSYNC_DELETE=1` to delete remote files not present locally
- `GRIDWORLD_RESULTS_DIR=/playpen-ssd/smerrill/deception2/Gridwolrd/Results` if your local folder is misspelled

Run Qwen-14B localization on Longleaf after sync:

```bash
cd /work/users/s/m/smerrill/deception2/BS/longleaf_scripts
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
LABEL_FILTER=truthful_only \
DATA_DIR=/work/users/s/m/smerrill/deception2/BS/Results/SentencePipeline/v1/DeepSeek-R1-Distill-Qwen-14B \
AUTO_BUILD_DATASET=0 \
./run_sentence_localization_batch_multi_gpu.sh
```
