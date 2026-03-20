# Deception2

This repo contains three deception-mining environments plus a shared sentence-localization pipeline.

The current dataset flow is:

1. Mine deception examples into each environment's `Results/DeceptionMining/...`
2. Build sentence datasets into `DatasetMain/{environment}/{model_tail}/`
3. Run sentence localization into `DatasetMain/{environment}/{model_tail}/localization/`

## Environments

- `BS/`
- `Gridworld/`
- `AdvisorAudit/`

Environment-specific code that must stay local lives under each environment's `src/`. Shared pipeline code lives in `src/`.

## Main Paths

- Mined examples:
  - `BS/Results/DeceptionMining/...`
  - `Gridworld/Results/DeceptionMining/...`
  - `AdvisorAudit/Results/DeceptionMining/...`
- Built datasets:
  - `DatasetMain/bs/{model_tail}/`
  - `DatasetMain/gridworld/{model_tail}/`
  - `DatasetMain/advisor_audit/{model_tail}/`
- Shared code:
  - `src/build_sentence_dataset.py`
  - `src/sentence_localization_batch.py`
  - `src/deception_miner.py`

Each `DatasetMain/{environment}/{model_tail}/` directory is expected to contain:

- `examples.jsonl`
- `sentences.jsonl`
- `manifest.json`
- `localization/` after localization runs

## Beowulf

### Full Model Pipeline

Use the shared shell driver to mine any missing data, build `DatasetMain`, and localize across `bs`, `gridworld`, and `advisor_audit` for one model:

```bash
/playpen-ssd/smerrill/deception2/shell_scripts/run_targeted_localization_pipeline.sh --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --gpu_ids "2 3 5"
```

By default this:

- targets `2500` deceptive and `2500` truthful examples
- uses `temperature=0.7`, `top_p=0.9`, `repetition_penalty=1.1`
- uses `n_samples=100` for localization
- writes outputs into `DatasetMain/{environment}/{model_tail}/`

To restrict to one environment:

```bash
/playpen-ssd/smerrill/deception2/shell_scripts/run_targeted_localization_pipeline.sh --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --env bs --gpu_ids "2 3 5"
```

To mine only:

```bash
/playpen-ssd/smerrill/deception2/shell_scripts/run_targeted_localization_pipeline.sh --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --mine_only --gpu_ids "2 3 5"
```

### Build `DatasetMain` Only

If mined outputs already exist, build datasets directly:

```bash
/playpen-ssd/smerrill/deception2/shell_scripts/run_build_dataset_main.sh
```

Single model:

```bash
/playpen-ssd/smerrill/deception2/shell_scripts/run_build_dataset_main.sh --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

Single model + single environment:

```bash
/playpen-ssd/smerrill/deception2/shell_scripts/run_build_dataset_main.sh --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --env bs
```

`run_build_dataset_main.sh` auto-selects the better of:

- `{model_name with / replaced by _}`
- `{model_tail}`

under each environment's `Results/DeceptionMining/`.

## Longleaf / SLURM

### Sentence Localization

Edit the top-of-file parameters in:

- `slurm_scripts/run_sentence_localization_slurm.sh`

Important editable fields include:

- `GAME`
- `MODEL_NAME`
- `N_SAMPLES`
- `TEMPERATURE`
- `TOP_P`
- `REPETITION_PENALTY`
- `METHOD`
- `TEXT_FIELD`

The script localizes from:

- `/work/users/s/m/smerrill/deception2/DatasetMain/{game}/{model_tail}/examples.jsonl`
- `/work/users/s/m/smerrill/deception2/DatasetMain/{game}/{model_tail}/sentences.jsonl`

and writes to:

- `/work/users/s/m/smerrill/deception2/DatasetMain/{game}/{model_tail}/localization/`

Submit an array job with:

```bash
/playpen-ssd/smerrill/deception2/slurm_scripts/submit_sentence_localization_batch.sh
```

or:

```bash
/playpen-ssd/smerrill/deception2/slurm_scripts/submit_sentence_localization_batch.sh 16
```

By default the submit helper uses:

- `SBATCH_ACCOUNT=rc_amcavoy_pi`

Override that if needed:

```bash
SBATCH_ACCOUNT=rc_nopigiven5_pi /playpen-ssd/smerrill/deception2/slurm_scripts/submit_sentence_localization_batch.sh 16
```

### Mining Only

`slurm_scripts/run_deception_mining_slurm.sh` runs the shared targeted miner in `mine_only` mode. It is useful when you want to fill in missing mined examples before building `DatasetMain`.

Submit it with:

```bash
sbatch --export=ALL,MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B /playpen-ssd/smerrill/deception2/slurm_scripts/run_deception_mining_slurm.sh
```

## Notes

- Shared pipeline code should live in `src/`, not duplicated under environment directories.
- `run_sentence_localization_slurm.sh` is intentionally manual/editable at the top, so it is easy to switch between reasoning and instruction-tuned models.
- For instruction-tuned models, it is often useful to set:
  - `METHOD="full"`
  - `TEXT_FIELD="reasoning"`
