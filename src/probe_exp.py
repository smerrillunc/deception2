# %% [markdown]
# # Cross-Environment Binary Probes From Counterfactual Generations
# 
# This notebook uses the paired compromise exactly:
# 
# - for each localized prefix, keep exactly 1 truthful generation and 1 deceptive generation
# - drop the prefix if it does not have both classes available
# - input text = `full_generation_text`
# - target label = `is_truthful`
# - compare the held-out prefix-mean probe score against the original continuation deception rate from the localization file, which was computed from all generations
# 
# This gives a much smaller, balanced dataset while preserving the full counterfactual continuation-rate target at the prefix level.
# 
# We only train binary truthful-vs-deceptive probes here.
# 

# %% [markdown]
# ## 1. Imports and configuration
# 
# The default dataset/model choice is `DeepSeek-R1-Distill-Qwen-7B` because it is the one variant that currently has localization files for all five environments inside `DatasetMain`.
# 

# %%
import importlib
import json
import sys
import warnings
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import pandas as pd

NOTEBOOK_ROOT = Path('/playpen-ssd/smerrill/deception2/Notebooks')
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

import multienv_commitment_probe_lib as multienv_probe_module
mcp = importlib.reload(multienv_probe_module)

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 220)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

MODEL_VARIANT = 'DeepSeek-R1-Distill-Qwen-7B'
HF_MODEL_NAME = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

OUTPUT_ROOT = NOTEBOOK_ROOT / 'multienv_counterfactual_binary_probe_paired_outputs'
RUN_ROOT = OUTPUT_ROOT / MODEL_VARIANT
ACTIVATION_CACHE_DIR = RUN_ROOT / 'paired_counterfactual_generation_final_layer_cache'

MAX_EXAMPLES_PER_ENV = 0
VAL_SIZE = 0.20
RANDOM_STATE = 42
GENERATION_SELECTION_MODE = 'paired_1p1n'

HIDDEN_STATE_DEVICE = 'cuda'
HIDDEN_STATE_DTYPE = 'bfloat16'
EXTRACTION_BATCH_SIZE = 4
LOCAL_FILES_ONLY = None
OVERWRITE_ACTIVATION_CACHE = False

PROBE_ALPHA = 1e-4
PROBE_MAX_ITER = 2_000
FEATURE_REPRESENTATIONS = ['raw', 'unit', 'layernorm', 'delta_unit', 'delta_unit_pca8']
SELECTED_REPRESENTATION = 'delta_unit'

SAVE_OUTPUTS = True
RUN_ROOT.mkdir(parents=True, exist_ok=True)

ENV_DATASET_ROOTS = mcp.build_env_dataset_roots(MODEL_VARIANT)
config_df = pd.DataFrame(
    [
        {'setting': 'model_variant', 'value': MODEL_VARIANT},
        {'setting': 'hf_model_name', 'value': HF_MODEL_NAME},
        {'setting': 'output_root', 'value': str(RUN_ROOT)},
        {'setting': 'activation_cache_dir', 'value': str(ACTIVATION_CACHE_DIR)},
        {'setting': 'max_examples_per_env', 'value': MAX_EXAMPLES_PER_ENV},
        {'setting': 'val_size', 'value': VAL_SIZE},
        {'setting': 'random_state', 'value': RANDOM_STATE},
        {'setting': 'generation_selection_mode', 'value': GENERATION_SELECTION_MODE},
        {'setting': 'hidden_state_device', 'value': HIDDEN_STATE_DEVICE},
        {'setting': 'hidden_state_dtype', 'value': HIDDEN_STATE_DTYPE},
        {'setting': 'extraction_batch_size', 'value': EXTRACTION_BATCH_SIZE},
        {'setting': 'feature_representations', 'value': ', '.join(FEATURE_REPRESENTATIONS)},
        {'setting': 'selected_representation', 'value': SELECTED_REPRESENTATION},
    ]
)
config_df


# %% [markdown]
# ## 2. Build the paired counterfactual-generation dataset
# 
# Exact row construction:
# 
# 1. open a localization JSON file
# 2. iterate over every `history` entry
# 3. iterate over every `generations` entry inside that history item
# 4. keep a generation candidate only if:
#    - `parse_error is None`
#    - `full_generation_text` is non-empty
#    - `is_truthful` is present
# 5. for each prefix, choose exactly:
#    - 1 truthful generation
#    - 1 deceptive generation
# 6. drop the prefix if one side is missing
# 7. train on `full_generation_text -> is_truthful`
# 
# We still keep the parent prefix metadata on each row, especially:
# - `sentence_idx`
# - `prefix_idx`
# - `global_prefix_id`
# - `continuation_deception_rate`
# 
# That `continuation_deception_rate` remains the original value from the localization history, so it still reflects the full counterfactual set rather than the paired subset.
# 
# The split is assigned at the state level inside each environment, so generations from the same underlying state do not leak across train and validation.
# 

# %%
examples_df, generation_df, summary_tables = mcp.build_multienv_counterfactual_generation_dataset(
    ENV_DATASET_ROOTS,
    max_examples_per_env=MAX_EXAMPLES_PER_ENV,
    selection_mode=GENERATION_SELECTION_MODE,
    random_state=RANDOM_STATE,
)
generation_df = mcp.assign_env_state_splits(
    generation_df,
    val_size=VAL_SIZE,
    random_state=RANDOM_STATE,
)

split_summary_df = (
    generation_df[['env_name', 'global_state_id', 'split']]
    .drop_duplicates(['env_name', 'global_state_id'])
    .groupby(['env_name', 'split'], as_index=False)
    .agg(n_states=('global_state_id', 'size'))
    .sort_values(['env_name', 'split'])
    .reset_index(drop=True)
)

#display(summary_tables['example_summary'])
#display(summary_tables['prefix_summary'])
#display(summary_tables['generation_summary'])
##display(summary_tables['pairing_summary'])
#display(split_summary_df)


# %% [markdown]
# ## 3. Extract or load final-layer hidden states from `full_generation_text`
# 
# This uses the full counterfactual generation text for each row and stores the final hidden state at the last token of that text.
# 

# %%
activation_cache = mcp.extract_or_load_generation_final_layer_activations(
    generation_df,
    cache_dir=ACTIVATION_CACHE_DIR,
    model_name=HF_MODEL_NAME,
    text_col='full_generation_text',
    device=HIDDEN_STATE_DEVICE,
    dtype_str=HIDDEN_STATE_DTYPE,
    batch_size=EXTRACTION_BATCH_SIZE,
    local_files_only=LOCAL_FILES_ONLY,
    overwrite=OVERWRITE_ACTIVATION_CACHE,
    show_progress=True,
)

#activation_cache


# %% [markdown]
# ## 4. Prepare hidden-state representations once
# 
# The key optimization for the normalization ablation is that we only load the raw final-layer cache once. Every representation below is derived from this cached matrix in RAM, so testing normalization schemes does not rerun the model.
# 

# %%
feature_bundle = mcp.prepare_feature_representations(
    generation_df,
    activation_cache,
)

#{
#    'n_rows': int(feature_bundle.feature_matrix.shape[0]),
#    'd_model': int(feature_bundle.feature_matrix.shape[1]),
#    'representations_to_test': FEATURE_REPRESENTATIONS,
#}


# %% [markdown]
# ## 5. Binary probe representation sweep
# 
# For each representation:
# - fit the binary truthful-vs-deceptive probe on one source environment at a time
# - score every environment on validation states
# - summarize both in-domain and OOD AUROC
# - show a pooled diagonal validation confusion matrix
# 

# %%
binary_sweep = mcp.run_binary_representation_sweep(
    generation_df,
    prepared_features=feature_bundle,
    representations=FEATURE_REPRESENTATIONS,
    random_state=RANDOM_STATE,
    alpha=PROBE_ALPHA,
    max_iter=PROBE_MAX_ITER,
)

#display(binary_sweep.summary_df)
#display(binary_sweep.timing_df)
mcp.plot_binary_transfer_heatmap_grid(binary_sweep)
mcp.plot_binary_representation_confusion_matrices(binary_sweep)


# %% [markdown]
# ## 6. Prefix-level probe score vs continuation deception rate
# 
# The binary probe is trained on individual counterfactual generations, but the continuation deception rate is a prefix-level quantity.
# 
# So for this comparison we aggregate the held-out binary probe scores back to the prefix level:
# 
# - one prefix gets exactly two kept generation rows in this dataset
# - we take the mean probe score across those rows
# - then we compare that mean prefix score against the prefix's original continuation deception rate from the localization history
# 

# %%
binary_result = binary_sweep.results_by_representation[SELECTED_REPRESENTATION]
prefix_prediction_df, binary_prefix_corr_df = mcp.build_binary_prefix_continuation_correlations(binary_result)

#display(binary_prefix_corr_df)
#display(prefix_prediction_df.head())

mcp.plot_binary_prefix_probe_vs_continuation(prefix_prediction_df, binary_prefix_corr_df)
mcp.plot_binary_prefix_mean_trajectory_overlay(prefix_prediction_df)


# %% [markdown]
# ## 7. Save exact-number outputs
# 
# The saved tables are:
# - `example_summary.csv`
# - `prefix_summary.csv`
# - `generation_summary.csv`
# - `split_summary.csv`
# - `binary_representation_summary.csv`
# - `binary_representation_transfer.csv`
# - `binary_representation_confusion.csv`
# - `binary_representation_timing.csv`
# - `selected_binary_val_predictions.parquet`
# - `selected_binary_prefix_predictions.parquet`
# - `binary_prefix_continuation_correlations.csv`
# 

# %%
if SAVE_OUTPUTS:
    summary_tables['example_summary'].to_csv(RUN_ROOT / 'example_summary.csv', index=False)
    summary_tables['prefix_summary'].to_csv(RUN_ROOT / 'prefix_summary.csv', index=False)
    summary_tables['generation_summary'].to_csv(RUN_ROOT / 'generation_summary.csv', index=False)
    split_summary_df.to_csv(RUN_ROOT / 'split_summary.csv', index=False)

    binary_sweep.summary_df.to_csv(RUN_ROOT / 'binary_representation_summary.csv', index=False)
    binary_sweep.transfer_df.to_csv(RUN_ROOT / 'binary_representation_transfer.csv', index=False)
    binary_sweep.confusion_df.to_csv(RUN_ROOT / 'binary_representation_confusion.csv', index=False)
    binary_sweep.timing_df.to_csv(RUN_ROOT / 'binary_representation_timing.csv', index=False)
    binary_result.matrix_df.to_csv(RUN_ROOT / f'binary_transfer_matrix__{SELECTED_REPRESENTATION}.csv')
    binary_result.diagonal_prediction_df.to_parquet(RUN_ROOT / 'selected_binary_val_predictions.parquet', index=False)
    prefix_prediction_df.to_parquet(RUN_ROOT / 'selected_binary_prefix_predictions.parquet', index=False)
    binary_prefix_corr_df.to_csv(RUN_ROOT / 'binary_prefix_continuation_correlations.csv', index=False)

    config_payload = {
        'model_variant': MODEL_VARIANT,
        'hf_model_name': HF_MODEL_NAME,
        'max_examples_per_env': MAX_EXAMPLES_PER_ENV,
        'val_size': VAL_SIZE,
        'random_state': RANDOM_STATE,
        'generation_selection_mode': GENERATION_SELECTION_MODE,
        'hidden_state_device': HIDDEN_STATE_DEVICE,
        'hidden_state_dtype': HIDDEN_STATE_DTYPE,
        'extraction_batch_size': EXTRACTION_BATCH_SIZE,
        'probe_alpha': PROBE_ALPHA,
        'probe_max_iter': PROBE_MAX_ITER,
        'feature_representations': FEATURE_REPRESENTATIONS,
        'selected_representation': SELECTED_REPRESENTATION,
        'activation_cache_dir': str(ACTIVATION_CACHE_DIR),
    }
    (RUN_ROOT / 'run_config.json').write_text(json.dumps(config_payload, indent=2), encoding='utf-8')
    print(f'Saved outputs to: {RUN_ROOT}')
else:
    print('SAVE_OUTPUTS is False; no files were written.')



