from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


NOTEBOOK_PATH = Path(__file__).with_name("datasetmain_commitment_juncture_prevalence_paper.ipynb")


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(
            """# DatasetMain Commitment Juncture Paper Tables

This notebook builds paper-ready commitment-prevalence tables directly from the raw DatasetMain localization JSONs.

Definitions:
- Deceptive / truthful example: explicit `deceptive` label from each bundle's `examples.jsonl`, with localization-internal labels used only as a fallback when the example record is missing.
- Deceptive commitment: `delta_deception_rate > 0.3`
- Truthful commitment: `delta_deception_rate < -0.3`
- Commitment example location: the first qualifying commitment sentence as a fraction of the full reasoning-trace length.
  - `Model x Environment`: `mean [bootstrap 95% CI]`
  - `Model` pooled across environments: `mean [bootstrap 95% CI]`

The notebook caches parsed example summaries under `ARTIFACT_DIR` so future runs can skip the expensive raw-JSON scan. Set `FORCE_REBUILD_SUMMARIES = True` to rebuild the cache from scratch.
"""
        ),
        new_code_cell(
            """from __future__ import annotations

from pathlib import Path
import importlib
import json

import pandas as pd
from IPython.display import Markdown, display

import datasetmain_commitment_juncture_prevalence_lib as cj


cj = importlib.reload(cj)

DATASETMAIN_ROOT = cj.DATASETMAIN_ROOT
DELTA_THRESHOLD = cj.DELTA_DECEPTION_THRESHOLD
MAX_JSON_FILES_PER_BUNDLE = None
BOOTSTRAP_NUM_RESAMPLES = cj.BOOTSTRAP_NUM_RESAMPLES
SHOW_PROGRESS = True
PROGRESS_LEVEL = 'bundle'
SAVE_ARTIFACTS = True
LOAD_SUMMARY_CACHE = True
SAVE_SUMMARY_CACHE = True
FORCE_REBUILD_SUMMARIES = False
ARTIFACT_DIR = Path('/playpen-ssd/smerrill/deception2/Notebooks/datasetmain_commitment_juncture_prevalence_outputs')
SUMMARY_CACHE_VERSION = 'prevalence_example_cache_v2'

pd.options.display.max_columns = 200


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


def display_paper_table(df: pd.DataFrame) -> None:
    formatters = {}
    for column in df.columns:
        if column.endswith('Examples'):
            formatters[column] = '{:,}'
        elif column.endswith('Fraction'):
            formatters[column] = '{:.1%}'
    display(
        df.style
        .hide(axis='index')
        .format(formatters, na_rep='')
    )


def summary_cache_paths() -> dict[str, Path]:
    out_dir = ensure_artifact_dir()
    return {
        'metadata': out_dir / 'summary_cache_metadata.json',
        'inventory': out_dir / 'inventory_df.pkl',
        'example': out_dir / 'example_df.pkl',
        'parse_error': out_dir / 'parse_error_df.pkl',
    }


def build_summary_cache_metadata() -> dict[str, object]:
    return {
        'cache_version': SUMMARY_CACHE_VERSION,
        'dataset_root': str(DATASETMAIN_ROOT),
        'delta_threshold': float(DELTA_THRESHOLD),
        'max_json_files_per_bundle': MAX_JSON_FILES_PER_BUNDLE,
    }


def has_complete_summary_cache() -> bool:
    paths = summary_cache_paths()
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        metadata = json.loads(paths['metadata'].read_text(encoding='utf-8'))
    except Exception:
        return False
    return metadata == build_summary_cache_metadata()


def load_summary_cache() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = summary_cache_paths()
    inventory_df = pd.read_pickle(paths['inventory'])
    example_df = pd.read_pickle(paths['example'])
    parse_error_df = pd.read_pickle(paths['parse_error'])
    return inventory_df, example_df, parse_error_df


def save_summary_cache(
    inventory_df: pd.DataFrame,
    example_df: pd.DataFrame,
    parse_error_df: pd.DataFrame,
) -> None:
    if not SAVE_SUMMARY_CACHE:
        return
    paths = summary_cache_paths()
    inventory_df.to_pickle(paths['inventory'])
    example_df.to_pickle(paths['example'])
    parse_error_df.to_pickle(paths['parse_error'])
    paths['metadata'].write_text(
        json.dumps(build_summary_cache_metadata(), indent=2, sort_keys=True),
        encoding='utf-8',
    )
"""
        ),
        new_code_cell(
            """if not hasattr(cj, 'load_datasetmain_localization_example_df'):
    cj = importlib.reload(cj)

summary_source = 'raw_json'
if LOAD_SUMMARY_CACHE and (not FORCE_REBUILD_SUMMARIES) and has_complete_summary_cache():
    inventory_df, example_df, parse_error_df = load_summary_cache()
    summary_source = 'cache'
else:
    inventory_df, example_df, parse_error_df = cj.load_datasetmain_localization_example_df(
        DATASETMAIN_ROOT,
        max_json_files_per_bundle=MAX_JSON_FILES_PER_BUNDLE,
        show_progress=SHOW_PROGRESS,
        progress_level=PROGRESS_LEVEL,
    )
    save_summary_cache(inventory_df, example_df, parse_error_df)

coverage_table_df = inventory_df.loc[
    :,
    [
        'model_display',
        'env_display',
        'json_file_count',
        'loaded_examples',
        'usable_examples',
        'unusable_examples',
    ],
].rename(
    columns={
        'model_display': 'Model',
        'env_display': 'Environment',
        'json_file_count': 'Localization JSONs',
        'loaded_examples': 'Summarized Examples',
        'usable_examples': 'Usable Examples',
        'unusable_examples': 'Unusable Examples',
    }
)

md('## Localization Coverage')
display(coverage_table_df.style.hide(axis='index'))

memory_mib = example_df.memory_usage(deep=True).sum() / (1024 ** 2) if not example_df.empty else 0.0
source_text = 'cached summary pickles' if summary_source == 'cache' else 'raw localization JSONs'
md(
    f'Loaded `{len(example_df):,}` example summaries from {source_text}. '
    f'Parse errors: `{len(parse_error_df):,}`. Example summary frame memory: `{memory_mib:.1f} MiB`. '
    f'Progress settings: `SHOW_PROGRESS={SHOW_PROGRESS}`, `PROGRESS_LEVEL={PROGRESS_LEVEL}`. '
    f'Cache status: `summary_source={summary_source}`, `FORCE_REBUILD_SUMMARIES={FORCE_REBUILD_SUMMARIES}`.'
)

maybe_save_table(coverage_table_df, 'json_localization_coverage')
maybe_save_table(parse_error_df, 'parse_errors')
"""
        ),
        new_code_cell(
            """env_model_stats_df = cj.build_commitment_example_statistics(
    example_df,
    ['model_display', 'env_display'],
    bootstrap_location_ci=True,
    bootstrap_num_resamples=BOOTSTRAP_NUM_RESAMPLES,
)
model_stats_df = cj.build_commitment_example_statistics(
    example_df,
    ['model_display'],
    bootstrap_location_ci=True,
    bootstrap_num_resamples=BOOTSTRAP_NUM_RESAMPLES,
)

paper_env_model_table_df = cj.make_commitment_fraction_location_table(
    env_model_stats_df,
    location_interval_style='bootstrap_ci',
)
paper_model_table_df = cj.make_commitment_fraction_location_table(
    model_stats_df,
    location_interval_style='bootstrap_ci',
)

md('## Model Pooled Across Environments')
display_paper_table(paper_model_table_df)

md('## Model x Environment')
display_paper_table(paper_env_model_table_df)

maybe_save_table(env_model_stats_df, 'env_model_stats_raw')
maybe_save_table(model_stats_df, 'model_stats_raw')
maybe_save_table(paper_env_model_table_df, 'env_model_paper_table')
maybe_save_table(paper_model_table_df, 'model_paper_table')
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
    NOTEBOOK_PATH.write_text(nbformat.writes(notebook), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
