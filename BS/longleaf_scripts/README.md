# Longleaf Scripts (BS)

These scripts mirror the BS pipeline scripts but avoid hardcoded `/playpen-ssd/...` paths.

Defaults:
- `PROJECT_ROOT`: inferred from script location (`.../deception2`)
- `RESULTS_ROOT`: `$PROJECT_ROOT/BS/Results`
- `CONDA_ENV`: `deception`
- `CONDA_SH`: auto-discovered from common conda locations (or set manually)

Common usage examples:

```bash
cd /work/users/s/m/smerrill/deception/deception2/BS/longleaf_scripts
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B LABEL_FILTER=truthful_only ./run_deception_miner_multi_gpu.sh
```

```bash
cd /work/users/s/m/smerrill/deception/deception2/BS/longleaf_scripts
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B LABEL_FILTER=truthful_only AUTO_BUILD_DATASET=1 ./run_sentence_localization_batch_multi_gpu.sh
```
