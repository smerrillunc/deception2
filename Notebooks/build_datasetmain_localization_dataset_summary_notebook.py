from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


NOTEBOOK_PATH = Path(__file__).with_name("datasetmain_localization_dataset_summary.ipynb")


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(
            """# DatasetMain Localization Dataset Summary

This notebook summarizes the full DatasetMain localization corpus directly from the raw localization JSON files such as:

- `DatasetMain/bs/DeepSeek-R1-Distill-Llama-8B/localization/sentence_localization_2026-03-11_13-53-37_game_0_turn_0_state_0_sample_8.json`

It reports:

1. The requested per-model table:
   - Avg. reasoning sent.
   - Avg. reasoning tokens
   - Avg. prefixes localized
   - Avg. continuation tokens
2. Per-environment breakdowns.
3. Raw bundle inventory counts and sizes.
4. Whole-dataset totals for files, localized sentences, continuations, tokens, sentences, words, and disk size.

Definitions used here:

- `Reasoning sent.`: number of localized reasoning sentences in each localization file, i.e. `len(history)`.
- `Reasoning tokens`: by default this notebook uses the real tokenizer for the model that generated each localization file.
  - `DeepSeek-R1-Distill-Qwen-7B` files use the Qwen-7B tokenizer.
  - `DeepSeek-R1-Distill-Qwen-14B` files use the Qwen-14B tokenizer.
  - `DeepSeek-R1-Distill-Llama-8B` files use the Llama-8B tokenizer.
  - `gpt-oss-20b` files use the GPT-OSS-20B tokenizer.
- `Prefixes localized`: number of localized prefixes in each file, i.e. `len(history)`.
- `Continuation tokens`: average real-tokenizer token count of each saved `gen_text` continuation.
- `Words` and sentence counts remain model-agnostic text counts.
- `Expanded dataset totals`: for every continuation, we count both the continuation text and the full `prefix_text` that produced it. This is the dataset view that matches localization examples as actual prompt-continuation pairs.

Progress:

- The notebook now supports `tqdm` progress bars.
- If you keep `NUM_WORKERS = 1` and `PROGRESS_LEVEL = 'file'`, you get the most informative ETA because the bar advances file-by-file.
- If you increase `NUM_WORKERS`, progress falls back to bundle-level updates.

Caching:

- The notebook caches the expensive per-bundle summary under `ARTIFACT_DIR`.
- After the first full run, future runs can load the cached bundle summary instead of reparsing all localization JSONs.
- Set `FORCE_REBUILD_BUNDLE_SUMMARY = True` to rebuild from raw files.
"""
        ),
        new_code_cell(
            """from __future__ import annotations

from pathlib import Path
import importlib
import json
import os

import pandas as pd
from IPython.display import Markdown, display

import datasetmain_localization_dataset_summary_lib as dsum


dsum = importlib.reload(dsum)

DATASETMAIN_ROOT = dsum.DATASETMAIN_ROOT
MAX_FILES_PER_BUNDLE = None
TOKEN_COUNT_MODE = 'hf'
NUM_WORKERS = 1
SHOW_PROGRESS = True
PROGRESS_LEVEL = 'file'
EXPECTED_FILES_PER_BUNDLE = 5000
SAVE_ARTIFACTS = True
LOAD_BUNDLE_SUMMARY_CACHE = True
SAVE_BUNDLE_SUMMARY_CACHE = True
FORCE_REBUILD_BUNDLE_SUMMARY = False
ARTIFACT_DIR = Path('/playpen-ssd/smerrill/deception2/Notebooks/datasetmain_localization_dataset_summary_outputs')
SUMMARY_CACHE_VERSION = 'localization_dataset_summary_bundle_cache_v2'

pd.options.display.max_columns = 200
pd.options.display.max_colwidth = 220


def md(text: str) -> None:
    display(Markdown(text))


def ensure_artifact_dir() -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


def maybe_save_table(df: pd.DataFrame, stem: str) -> None:
    if not SAVE_ARTIFACTS:
        return
    out_dir = ensure_artifact_dir()
    df.to_csv(out_dir / f'{stem}.csv', index=False)


def display_table(df: pd.DataFrame, *, float_cols: dict[str, str] | None = None, int_cols: list[str] | None = None) -> None:
    format_map = {}
    for column, fmt in (float_cols or {}).items():
        if column in df.columns:
            format_map[column] = fmt
    for column in int_cols or []:
        if column in df.columns:
            format_map[column] = '{:,.0f}'
    display(df.style.hide(axis='index').format(format_map, na_rep=''))


def bundle_summary_cache_paths() -> dict[str, Path]:
    out_dir = ensure_artifact_dir()
    return {
        'metadata': out_dir / 'bundle_summary_cache_metadata.json',
        'bundle': out_dir / 'bundle_df.pkl',
    }


def build_bundle_summary_cache_metadata() -> dict[str, object]:
    return {
        'cache_version': SUMMARY_CACHE_VERSION,
        'dataset_root': str(DATASETMAIN_ROOT),
        'max_files_per_bundle': MAX_FILES_PER_BUNDLE,
        'token_count_mode': str(TOKEN_COUNT_MODE),
        'expected_files_per_bundle': int(EXPECTED_FILES_PER_BUNDLE),
    }


def has_complete_bundle_summary_cache() -> bool:
    paths = bundle_summary_cache_paths()
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        metadata = json.loads(paths['metadata'].read_text(encoding='utf-8'))
    except Exception:
        return False
    return metadata == build_bundle_summary_cache_metadata()


def load_bundle_summary_cache() -> pd.DataFrame:
    return pd.read_pickle(bundle_summary_cache_paths()['bundle'])


def save_bundle_summary_cache(bundle_df: pd.DataFrame) -> None:
    if not SAVE_BUNDLE_SUMMARY_CACHE:
        return
    paths = bundle_summary_cache_paths()
    bundle_df.to_pickle(paths['bundle'])
    paths['metadata'].write_text(
        json.dumps(build_bundle_summary_cache_metadata(), indent=2, sort_keys=True),
        encoding='utf-8',
    )
"""
        ),
        new_code_cell(
            """bundle_summary_source = 'raw_json'
if LOAD_BUNDLE_SUMMARY_CACHE and (not FORCE_REBUILD_BUNDLE_SUMMARY) and has_complete_bundle_summary_cache():
    bundle_df = load_bundle_summary_cache()
    bundle_summary_source = 'cache'
else:
    bundle_df = dsum.build_bundle_summary_df(
        DATASETMAIN_ROOT,
        max_files_per_bundle=MAX_FILES_PER_BUNDLE,
        num_workers=NUM_WORKERS,
        token_count_mode=TOKEN_COUNT_MODE,
        show_progress=SHOW_PROGRESS,
        progress_level=PROGRESS_LEVEL,
    )
    save_bundle_summary_cache(bundle_df)

model_df = dsum.summarize_groups(bundle_df, ['model_display'])
env_df = dsum.summarize_groups(bundle_df, ['env_display'])
env_model_df = dsum.summarize_groups(bundle_df, ['model_display', 'env_display'])

requested_model_table_df = dsum.make_requested_summary_table(model_df)
requested_env_table_df = dsum.make_requested_summary_table(env_df, include_model=False, include_environment=True)
requested_env_model_table_df = dsum.make_requested_summary_table(env_model_df, include_environment=True)

bundle_inventory_table_df = bundle_df.loc[
    :,
    [
        'model_display',
        'env_display',
        'file_count',
        'localized_prefix_total',
        'continuation_total',
        'file_size_tb',
        'avg_continuations_per_prefix',
    ],
].rename(
    columns={
        'model_display': 'Model',
        'env_display': 'Environment',
        'file_count': 'Localization Files',
        'localized_prefix_total': 'Localized Sentences',
        'continuation_total': 'Continuations',
        'file_size_tb': 'File Size (TB)',
        'avg_continuations_per_prefix': 'Avg. Continuations / Prefix',
    }
)
bundle_inventory_table_df['Gap vs 5000 Files'] = bundle_inventory_table_df['Localization Files'] - EXPECTED_FILES_PER_BUNDLE

model_totals_table_df = model_df.loc[
    :,
    [
        'model_display',
        'file_count',
        'localized_prefix_total',
        'continuation_total',
        'expanded_dataset_token_total',
        'expanded_dataset_word_total',
        'expanded_dataset_sentence_total',
        'file_size_tb',
    ],
].rename(
    columns={
        'model_display': 'Model',
        'file_count': 'Localization Files',
        'localized_prefix_total': 'Localized Sentences',
        'continuation_total': 'Continuations',
        'expanded_dataset_token_total': 'Expanded Dataset Tokens',
        'expanded_dataset_word_total': 'Expanded Dataset Words',
        'expanded_dataset_sentence_total': 'Expanded Dataset Sentences',
        'file_size_tb': 'File Size (TB)',
    }
)

env_totals_table_df = env_df.loc[
    :,
    [
        'env_display',
        'file_count',
        'localized_prefix_total',
        'continuation_total',
        'expanded_dataset_token_total',
        'expanded_dataset_word_total',
        'expanded_dataset_sentence_total',
        'file_size_tb',
    ],
].rename(
    columns={
        'env_display': 'Environment',
        'file_count': 'Localization Files',
        'localized_prefix_total': 'Localized Sentences',
        'continuation_total': 'Continuations',
        'expanded_dataset_token_total': 'Expanded Dataset Tokens',
        'expanded_dataset_word_total': 'Expanded Dataset Words',
        'expanded_dataset_sentence_total': 'Expanded Dataset Sentences',
        'file_size_tb': 'File Size (TB)',
    }
)

totals_table_df = dsum.make_total_summary_table(bundle_df)

md('## Run Summary')
source_text = 'cached bundle summary' if bundle_summary_source == 'cache' else 'raw localization JSONs'
md(
    f'Processed `{int(bundle_df["file_count"].sum()):,}` raw localization JSON files '
    f'across `{len(bundle_df):,}` model x environment bundles using token mode `{TOKEN_COUNT_MODE}`. '
    f'Loaded from {source_text}. '
    f'`NUM_WORKERS={NUM_WORKERS}`, `SHOW_PROGRESS={SHOW_PROGRESS}`, `PROGRESS_LEVEL={PROGRESS_LEVEL}`, '
    f'`MAX_FILES_PER_BUNDLE={MAX_FILES_PER_BUNDLE}`, `FORCE_REBUILD_BUNDLE_SUMMARY={FORCE_REBUILD_BUNDLE_SUMMARY}`.'
)

maybe_save_table(bundle_df, 'bundle_summary_raw')
maybe_save_table(model_df, 'model_summary_raw')
maybe_save_table(env_df, 'environment_summary_raw')
maybe_save_table(env_model_df, 'environment_model_summary_raw')
"""
        ),
        new_markdown_cell(
            """## Requested Per-Model Table

This is the closest match to the table sketch in the request. `Avg. continuation tokens` is the average token count of each saved continuation `gen_text`, while the prompt-side tokens are accounted for separately in the totals sections below.
"""
        ),
        new_code_cell(
            """display_table(
    requested_model_table_df,
    float_cols={
        'Avg. reasoning sent.': '{:,.1f}',
        'Avg. reasoning tokens': '{:,.1f}',
        'Avg. prefixes localized': '{:,.1f}',
        'Avg. continuation tokens': '{:,.1f}',
    },
)

maybe_save_table(requested_model_table_df, 'requested_model_table')
"""
        ),
        new_markdown_cell(
            """## Per-Environment Breakdowns"""
        ),
        new_code_cell(
            """md('### Environment Pooled Across Models')
display_table(
    requested_env_table_df,
    float_cols={
        'Avg. reasoning sent.': '{:,.1f}',
        'Avg. reasoning tokens': '{:,.1f}',
        'Avg. prefixes localized': '{:,.1f}',
        'Avg. continuation tokens': '{:,.1f}',
    },
)

md('### Model x Environment')
display_table(
    requested_env_model_table_df,
    float_cols={
        'Avg. reasoning sent.': '{:,.1f}',
        'Avg. reasoning tokens': '{:,.1f}',
        'Avg. prefixes localized': '{:,.1f}',
        'Avg. continuation tokens': '{:,.1f}',
    },
)

maybe_save_table(requested_env_table_df, 'requested_environment_table')
maybe_save_table(requested_env_model_table_df, 'requested_env_model_table')
"""
        ),
        new_markdown_cell(
            """## Raw Bundle Inventory

This table is useful for sanity-checking the dataset footprint directly from the localization files.
"""
        ),
        new_code_cell(
            """display_table(
    bundle_inventory_table_df,
    float_cols={
        'File Size (TB)': '{:,.6f}',
        'Avg. Continuations / Prefix': '{:,.1f}',
        'Gap vs 5000 Files': '{:,.0f}',
    },
    int_cols=['Localization Files', 'Localized Sentences', 'Continuations'],
)

non_exact_bundle_count_df = bundle_inventory_table_df.loc[
    ~bundle_inventory_table_df['Localization Files'].eq(EXPECTED_FILES_PER_BUNDLE)
].reset_index(drop=True)
if not non_exact_bundle_count_df.empty:
    md('### Bundles Not Exactly 5000 Files')
    display_table(
        non_exact_bundle_count_df,
        float_cols={
            'File Size (TB)': '{:,.6f}',
            'Avg. Continuations / Prefix': '{:,.1f}',
            'Gap vs 5000 Files': '{:,.0f}',
        },
        int_cols=['Localization Files', 'Localized Sentences', 'Continuations'],
    )

maybe_save_table(bundle_inventory_table_df, 'bundle_inventory')
maybe_save_table(non_exact_bundle_count_df, 'bundle_inventory_non_5000')
"""
        ),
        new_markdown_cell(
            """## Dataset Totals

The totals below separate the unique reasoning corpus from the expanded localization dataset. The expanded totals count every prompt prefix once for every saved continuation generated from that prefix.
"""
        ),
        new_code_cell(
            """display_table(totals_table_df)
maybe_save_table(totals_table_df, 'dataset_totals')
"""
        ),
        new_markdown_cell(
            """## Totals by Model and Environment"""
        ),
        new_code_cell(
            """md('### Totals by Model')
display_table(
    model_totals_table_df,
    float_cols={'File Size (TB)': '{:,.6f}'},
    int_cols=[
        'Localization Files',
        'Localized Sentences',
        'Continuations',
        'Expanded Dataset Tokens',
        'Expanded Dataset Words',
        'Expanded Dataset Sentences',
    ],
)

md('### Totals by Environment')
display_table(
    env_totals_table_df,
    float_cols={'File Size (TB)': '{:,.6f}'},
    int_cols=[
        'Localization Files',
        'Localized Sentences',
        'Continuations',
        'Expanded Dataset Tokens',
        'Expanded Dataset Words',
        'Expanded Dataset Sentences',
    ],
)

maybe_save_table(model_totals_table_df, 'model_totals')
maybe_save_table(env_totals_table_df, 'environment_totals')
"""
        ),
        new_markdown_cell(
            """## Handy Objects

The main in-memory tables after running the notebook are:

- `bundle_df`
- `model_df`
- `env_df`
- `env_model_df`
- `requested_model_table_df`
- `requested_env_table_df`
- `requested_env_model_table_df`
- `totals_table_df`

Useful run modes:

1. Exact tokenizer counts with detailed ETA:
   - `TOKEN_COUNT_MODE = 'hf'`
   - `NUM_WORKERS = 1`
   - `PROGRESS_LEVEL = 'file'`
2. Faster but coarser progress:
   - `TOKEN_COUNT_MODE = 'hf'`
   - `NUM_WORKERS > 1`
   - `PROGRESS_LEVEL = 'bundle'`
3. Cheap smoke test:
   - `MAX_FILES_PER_BUNDLE = 1`
4. Force-refresh the saved summary cache:
   - `FORCE_REBUILD_BUNDLE_SUMMARY = True`
"""
        ),
    ]

    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
    )


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(nbformat.writes(notebook), encoding='utf-8')
    print(f'Wrote {NOTEBOOK_PATH}')


if __name__ == '__main__':
    main()
