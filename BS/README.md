# Deception2 / BS

End-to-end pipeline for mining deceptive BS examples, splitting into sentences, tagging with a reasoning-function taxonomy, and running sentence-level localization debug.

**Directory layout**
- `src/`: core scripts
- `shell_scripts/`: runnable pipeline scripts
- `config/`: taxonomy + config
- `Results/`: outputs

## Pipeline Overview
1. Mine deceptive examples from real game flows with `bs_deception_miner.py`
2. Build sentence dataset from `action_reasoning`
3. LLM-tag each sentence using the taxonomy
4. Run sentence-level localization debug on a chosen example
5. Run batch sentence-level localization
6. Extract sentence features for modeling

## Taxonomy
Defined in `config/sentence_taxonomy.json` (8 categories from Venhoff et al. 2025) and fully parameterizable.

## Running the Pipeline

### 1) Deception mining
```bash
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_deception_miner.sh
```
Output example:
`/playpen-ssd/smerrill/deception2/BS/Results/DeceptionMining/YYYY-MM-DD/gpu_X/deception_samples.jsonl`

#### Multi-GPU mining (6 GPUs)
```bash
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_deception_miner_multi_gpu.sh
```
This launches one miner per GPU with distinct seeds.

### Full multi-GPU pipeline
```bash
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
OPENAI_API_KEY=your_key \
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_full_pipeline_multi_gpu.sh
```
GPU-accelerated steps (mining + localization) run across all GPUs. Tagging uses OpenAI API on CPU.

### 2) Build sentence dataset
```bash
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_dataset.sh
```
Outputs:
- `Results/SentencePipeline/v1/examples.jsonl`
- `Results/SentencePipeline/v1/sentences.jsonl`

Note: `--include_messages` is enabled so the localization debug step can rebuild prompts from the stored messages.

### 3) Tag sentences with LLM
```bash
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_tagger.sh
```
Output:
- `Results/SentencePipeline/v1/tags.jsonl`

Requires `OPENAI_API_KEY` in your environment. Default model is `gpt-4-mini`
(override with `MODEL_NAME=...`).

### 4) Sentence-level localization debug
```bash
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_localization_debug.sh
```
This prompts for an `example_id` from `examples.jsonl` and writes:
- `Results/SentencePipeline/v1/sentence_localization_debug_<example_id>.json`

### 5) Batch sentence-level localization
```bash
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_localization_batch.sh
```
Output directory:
- `Results/SentencePipeline/v1/localization/`

JSONL output (per-shard if sharded):
- `Results/SentencePipeline/v1/localization.jsonl` (or `.shardN.jsonl`)

Tip: For faster runs with less filesystem overhead, set `OUT_DIR=none` to skip per-example JSON files and only write JSONL.

#### Multi-GPU batch localization (6 GPUs)
```bash
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_localization_batch_multi_gpu.sh
```
This shards examples by index across GPUs using `SHARD_ID/NUM_SHARDS`.

To merge shards into one JSONL:
```bash
python /playpen-ssd/smerrill/deception2/BS/src/merge_localization_jsonl.py \
  --input /playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1/localization.jsonl.shard*.jsonl \
  --out_path /playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline/v1/localization.merged.jsonl \
  --dedupe
```

### 6) Extract sentence features
```bash
/playpen-ssd/smerrill/deception2/BS/shell_scripts/run_sentence_features.sh
```
Output:
- `Results/SentencePipeline/v1/sentence_features.jsonl`

## Core Scripts
- `src/bs_deception_miner.py`: mines deceptive BS examples
- `src/build_sentence_dataset.py`: builds `examples.jsonl` + `sentences.jsonl`
- `src/tag_sentences_llm.py`: tags sentences using taxonomy
- `src/sentence_localization_debug.py`: runs sentence-level deception localization
- `src/sentence_localization_batch.py`: runs batch sentence-level localization
- `src/merge_localization_jsonl.py`: merges shard JSONL outputs
- `src/extract_sentence_features.py`: builds a modeling dataset with text + localization features
- `src/sentence_pipeline.py`: shared sentence utilities + taxonomy support

## Notes
- Sentence splitting uses a simple regex; swap in spaCy if you want higher-quality segmentation.
- If you want to tag `action_raw_text` instead, change `--text_field` in `run_sentence_dataset.sh`.
