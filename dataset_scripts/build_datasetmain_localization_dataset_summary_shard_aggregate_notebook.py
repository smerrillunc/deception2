from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


NOTEBOOK_PATH = Path(__file__).with_name("datasetmain_localization_dataset_summary_shard_aggregate.ipynb")


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(
            """# DatasetMain Localization Shard Aggregate

This notebook aggregates the per-shard outputs produced by the sharded DatasetMain localization summary workflow.

Expected layout:

- `dataset_scripts/outputs/datasetmain_localization_dataset_summary_sharded/shards/<env>__<model>/bundle_summary_raw.csv`
- Optional parse issue tables in the same shard directories.

The notebook:

1. Loads every shard bundle summary.
2. Combines them into the same aggregate tables produced by `datasetmain_localization_dataset_summary.py --combine-shard-output-root ...`.
3. Saves the combined CSV outputs under `ARTIFACT_DIR` for convenience.
"""
        ),
        new_code_cell(
            """from __future__ import annotations

from pathlib import Path
import importlib

import pandas as pd
from IPython.display import Markdown, display

import datasetmain_localization_dataset_summary as dsum_main
import datasetmain_localization_dataset_summary_lib as dsum


dsum_main = importlib.reload(dsum_main)
dsum = importlib.reload(dsum)

REPO_ROOT = Path('/playpen-ssd/smerrill/deception2')
SHARD_OUTPUT_ROOT = REPO_ROOT / 'dataset_scripts' / 'outputs' / 'datasetmain_localization_dataset_summary_sharded' / 'shards'
ARTIFACT_DIR = REPO_ROOT / 'dataset_scripts' / 'outputs' / 'datasetmain_localization_dataset_summary_sharded' / 'combined_notebook'
EXPECTED_FILES_PER_BUNDLE = 5000
SAVE_ARTIFACTS = True

pd.options.display.max_columns = 200
pd.options.display.max_colwidth = 220


def md(text: str) -> None:
    display(Markdown(text))


def ensure_artifact_dir() -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


def display_table(
    df: pd.DataFrame,
    *,
    float_cols: dict[str, str] | None = None,
    int_cols: list[str] | None = None,
) -> None:
    format_map: dict[str, str] = {}
    for column, fmt in (float_cols or {}).items():
        if column in df.columns:
            format_map[column] = fmt
    for column in int_cols or []:
        if column in df.columns:
            format_map[column] = '{:,.0f}'
    display(df.style.hide(axis='index').format(format_map, na_rep=''))
"""
        ),
        new_code_cell(
            """bundle_df, parse_issue_df, shard_dirs = dsum_main._load_combined_shard_outputs(dsum, SHARD_OUTPUT_ROOT)
tables = dsum_main._build_output_tables(
    dsum,
    bundle_df,
    parse_issue_df,
    expected_files_per_bundle=EXPECTED_FILES_PER_BUNDLE,
)

if SAVE_ARTIFACTS:
    ensure_artifact_dir()
    dsum_main._save_output_tables(
        ARTIFACT_DIR,
        bundle_df=tables['bundle_df'],
        parse_issue_df=parse_issue_df,
        model_df=tables['model_df'],
        env_df=tables['env_df'],
        env_model_df=tables['env_model_df'],
        requested_model_table_df=tables['requested_model_table_df'],
        requested_env_table_df=tables['requested_env_table_df'],
        requested_env_model_table_df=tables['requested_env_model_table_df'],
        bundle_inventory_table_df=tables['bundle_inventory_table_df'],
        non_exact_bundle_count_df=tables['non_exact_bundle_count_df'],
        totals_table_df=tables['totals_table_df'],
        dataset_overview_table_df=tables['dataset_overview_table_df'],
        model_totals_table_df=tables['model_totals_table_df'],
        env_totals_table_df=tables['env_totals_table_df'],
        paper_model_scale_table_df=tables['paper_model_scale_table_df'],
        paper_env_scale_table_df=tables['paper_env_scale_table_df'],
        paper_env_model_scale_table_df=tables['paper_env_model_scale_table_df'],
        parse_issue_summary_df=tables['parse_issue_summary_df'],
    )

recovered_total = int(pd.to_numeric(bundle_df.get('recovered_json_file_count', 0), errors='coerce').fillna(0).sum()) if not bundle_df.empty else 0
skipped_total = int(pd.to_numeric(bundle_df.get('skipped_json_file_count', 0), errors='coerce').fillna(0).sum()) if not bundle_df.empty else 0

md(
    f\"Loaded **{len(shard_dirs):,}** shard directories from `{SHARD_OUTPUT_ROOT}`.  \"
    f\"Processed **{int(bundle_df['file_count'].sum()) if not bundle_df.empty else 0:,}** localization files, \"
    f\"recovered **{recovered_total:,}**, skipped **{skipped_total:,}**.\"
)
"""
        ),
        new_code_cell(
            """md('## Paper Overview')
display_table(
    tables['dataset_overview_table_df'],
    float_cols={'Value': '{:,.3f}'},
)

md('## Paper Model Scale Table')
display_table(
    tables['paper_model_scale_table_df'],
    float_cols={
        'Share of Traces (%)': '{:,.1f}',
        'Share of Expanded Tokens (%)': '{:,.1f}',
        'Avg. reasoning sent.': '{:,.1f}',
        'Avg. reasoning tokens': '{:,.1f}',
        'Avg. words / reasoning sent.': '{:,.2f}',
        'Avg. continuations / trace': '{:,.2f}',
        'Avg. continuation tokens': '{:,.1f}',
        'Expanded token multiplier': '{:,.2f}',
        'Recovery rate (%)': '{:,.2f}',
        'Skip rate (%)': '{:,.2f}',
        'File Size (TB)': '{:,.4f}',
    },
)

md('## Paper Environment Scale Table')
display_table(
    tables['paper_env_scale_table_df'],
    float_cols={
        'Share of Traces (%)': '{:,.1f}',
        'Share of Expanded Tokens (%)': '{:,.1f}',
        'Avg. reasoning sent.': '{:,.1f}',
        'Avg. reasoning tokens': '{:,.1f}',
        'Avg. words / reasoning sent.': '{:,.2f}',
        'Avg. continuations / trace': '{:,.2f}',
        'Avg. continuation tokens': '{:,.1f}',
        'Expanded token multiplier': '{:,.2f}',
        'Recovery rate (%)': '{:,.2f}',
        'Skip rate (%)': '{:,.2f}',
        'File Size (TB)': '{:,.4f}',
    },
)

md('## Paper Environment x Model Scale Table')
display_table(
    tables['paper_env_model_scale_table_df'],
    float_cols={
        'Share of Traces (%)': '{:,.1f}',
        'Share of Expanded Tokens (%)': '{:,.1f}',
        'Avg. reasoning sent.': '{:,.1f}',
        'Avg. reasoning tokens': '{:,.1f}',
        'Avg. words / reasoning sent.': '{:,.2f}',
        'Avg. continuations / trace': '{:,.2f}',
        'Avg. continuation tokens': '{:,.1f}',
        'Expanded token multiplier': '{:,.2f}',
        'Recovery rate (%)': '{:,.2f}',
        'Skip rate (%)': '{:,.2f}',
        'File Size (TB)': '{:,.4f}',
    },
)
"""
        ),
        new_code_cell(
            """md('## Requested Model Summary')
display_table(
    tables['requested_model_table_df'],
    float_cols={
        'Avg. reasoning sent.': '{:,.1f}',
        'Avg. reasoning tokens': '{:,.1f}',
        'Avg. reasoning words': '{:,.1f}',
        'Avg. words / reasoning sent.': '{:,.2f}',
        'Avg. localized traces / reasoning trace': '{:,.2f}',
        'Avg. continuation tokens': '{:,.1f}',
    },
)

md('## Requested Environment Summary')
display_table(
    tables['requested_env_table_df'],
    float_cols={
        'Avg. reasoning sent.': '{:,.1f}',
        'Avg. reasoning tokens': '{:,.1f}',
        'Avg. reasoning words': '{:,.1f}',
        'Avg. words / reasoning sent.': '{:,.2f}',
        'Avg. localized traces / reasoning trace': '{:,.2f}',
        'Avg. continuation tokens': '{:,.1f}',
    },
)

md('## Requested Environment x Model Summary')
display_table(
    tables['requested_env_model_table_df'],
    float_cols={
        'Avg. reasoning sent.': '{:,.1f}',
        'Avg. reasoning tokens': '{:,.1f}',
        'Avg. reasoning words': '{:,.1f}',
        'Avg. words / reasoning sent.': '{:,.2f}',
        'Avg. localized traces / reasoning trace': '{:,.2f}',
        'Avg. continuation tokens': '{:,.1f}',
    },
)
"""
        ),
        new_code_cell(
            """md('## Bundle Inventory')
display_table(
    tables['bundle_inventory_table_df'],
    float_cols={
        'File Size (TB)': '{:,.4f}',
        'Recovery Rate (%)': '{:,.2f}',
        'Skip Rate (%)': '{:,.2f}',
        'Avg. Continuations / Prefix': '{:,.2f}',
    },
    int_cols=[
        'Localization Files',
        'Attempted JSON Files',
        'Recovered JSON Files',
        'Skipped Broken JSON Files',
        'Localized Sentences',
        'Continuations',
        'Gap vs 5000 Files',
    ],
)

md('## Dataset Totals')
display_table(
    tables['totals_table_df'],
    float_cols={'Value': '{:,.4f}'},
)
"""
        ),
        new_code_cell(
            """md('## Parse Issue Summary')
tables['parse_issue_summary_df']"""
        ),
        new_code_cell(
            """md('## Parse Issues Raw')
parse_issue_df"""
        ),
    ]
    return new_notebook(cells=cells)


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(nbformat.writes(notebook), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
