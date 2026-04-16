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
  - `Model` pooled across environments: `mean +/- SE`

The notebook only renders the two paper tables, while the backing `env_model_stats_df` and `model_stats_df` data frames retain the numeric location summary columns.
"""
        ),
        new_code_cell(
            """from __future__ import annotations

from pathlib import Path
import importlib

import pandas as pd
from IPython.display import Markdown, display

import datasetmain_commitment_juncture_prevalence_lib as cj


cj = importlib.reload(cj)

DATASETMAIN_ROOT = cj.DATASETMAIN_ROOT
MAX_JSON_FILES_PER_BUNDLE = None
BOOTSTRAP_NUM_RESAMPLES = cj.BOOTSTRAP_NUM_RESAMPLES
SAVE_ARTIFACTS = False
ARTIFACT_DIR = Path('/playpen-ssd/smerrill/deception2/Notebooks/datasetmain_commitment_juncture_prevalence_outputs')

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
    display(
        df.style
        .hide(axis='index')
        .format(
            {
                'Deceptive Commitment Example Fraction': '{:.3f}',
                'Truthful Commitment Example Fraction': '{:.3f}',
            },
            na_rep='',
        )
    )
"""
        ),
        new_code_cell(
            """if not hasattr(cj, 'load_datasetmain_localization_example_df'):
    cj = importlib.reload(cj)

inventory_df, example_df, parse_error_df = cj.load_datasetmain_localization_example_df(
    DATASETMAIN_ROOT,
    max_json_files_per_bundle=MAX_JSON_FILES_PER_BUNDLE,
)

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
md(
    f'Parsed `{len(example_df):,}` example summaries from `{int(coverage_table_df["Localization JSONs"].sum()):,}` localization JSONs. '
    f'Parse errors: `{len(parse_error_df):,}`. Example summary frame memory: `{memory_mib:.1f} MiB`.'
)

maybe_save_table(coverage_table_df, 'json_localization_coverage')
"""
        ),
        new_code_cell(
            """env_model_stats_df = cj.build_commitment_example_statistics(
    example_df,
    ['model_display', 'env_display'],
    bootstrap_location_ci=True,
    bootstrap_num_resamples=BOOTSTRAP_NUM_RESAMPLES,
)
model_stats_df = cj.build_commitment_example_statistics(example_df, ['model_display'])

paper_env_model_table_df = cj.make_commitment_paper_table(
    env_model_stats_df,
    location_interval_style='bootstrap_ci',
)
paper_model_table_df = cj.make_commitment_paper_table(model_stats_df)

md('## Model x Environment')
display_paper_table(paper_env_model_table_df)

md('## Model Pooled Across Environments')
display_paper_table(paper_model_table_df)

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
