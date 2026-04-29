from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


NOTEBOOK_PATH = Path(__file__).with_name("commitment_juncture_threshold.ipynb")


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(
            """# DatasetMain Commitment Juncture Threshold Sweep

This notebook examines commitment junctures under multiple threshold values `tau`.

Definitions used here:

- A commitment juncture is a change in **consecutive counterfactual deception rate** between adjacent saved localization sentences.
- "Consecutive sentences" means consecutive entries in each saved localization trace after sorting by the saved sentence index.
- Positive commitment: `Delta_k = p_k - p_{k-1} > tau`
- Negative commitment: `Delta_k = p_k - p_{k-1} < -tau`
- We require `num_valid > 10` on **both** sides of the pair before counting a juncture.
- Coverage is the share of examples with at least one qualifying juncture among examples that have at least one valid consecutive sentence pair.

This notebook now uses artifact-first loading:

1. If saved CSV artifacts already exist in `ARTIFACT_DIR`, it loads those first.
2. If required artifacts are missing, it falls back to the raw localization JSON pass.
3. Newly computed tables and bucket summaries are written back to `ARTIFACT_DIR` when `SAVE_ARTIFACTS = True`.

The notebook reports:

1. Overall threshold-sensitivity tables.
2. Per-model threshold tables.
3. Bucketed `|Delta_k|` histograms with bins `0.1-0.2`, `0.2-0.3`, `0.3-0.4`, `0.4-0.5`, and `> 0.5`.
4. Per-model-by-environment tables at `tau = 0.3`.
"""
        ),
        new_code_cell(
            """from __future__ import annotations

from pathlib import Path
import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

STYLE_DIR = Path('/playpen-ssd/smerrill/deception2/styles')
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from neurips import COLORS, FIGURE_SIZES, add_figure_note, apply_style, style_axes, style_panel_title

import datasetmain_commitment_juncture_prevalence_lib as cj
import datasetmain_commitment_juncture_threshold_lib as cjt


cj = importlib.reload(cj)
cjt = importlib.reload(cjt)

DATASETMAIN_ROOT = cjt.DATASETMAIN_ROOT
TAU_VALUES = cjt.TAU_VALUES
DEFAULT_TAU = 0.3
MIN_VALID = cjt.DEFAULT_MIN_VALID
PREFERRED_SOURCE = cjt.DEFAULT_SOURCE_KIND
EXPECTED_JSONS_PER_BUNDLE = 5000
MIN_EXPECTED_JSONS_PER_BUNDLE = 4900
INCLUDE_SENTENCE_TEXT = False
MAX_JSON_FILES_PER_BUNDLE = None
SHOW_PROGRESS = True
PROGRESS_LEVEL = 'bundle'
LOAD_ARTIFACTS_IF_AVAILABLE = True
SAVE_ARTIFACTS = True
SAVE_PAIR_ARTIFACTS = True
ARTIFACT_DIR = Path('/playpen-ssd/smerrill/deception2/Notebooks/commitment_juncture_threshold_outputs')
PAIR_ARTIFACT_PATH = ARTIFACT_DIR / 'valid_pair_df.parquet'

REQUIRED_TABLE_ARTIFACT_STEMS = [
    'inventory',
    'positive_overall_table',
    'negative_overall_table',
    'positive_model_table_all_tau',
    'negative_model_table_all_tau',
    'positive_env_model_table_tau_0p3',
    'negative_env_model_table_tau_0p3',
]
OPTIONAL_TABLE_ARTIFACT_STEMS = [
    'non_exact_json_counts',
    'parse_warnings',
    'positive_overall_summary_raw',
    'negative_overall_summary_raw',
    'positive_model_summary_raw',
    'negative_model_summary_raw',
    'positive_env_model_summary_raw',
    'negative_env_model_summary_raw',
    'positive_model_table_tau_0p3',
    'negative_model_table_tau_0p3',
    'positive_env_model_table_all_tau',
    'negative_env_model_table_all_tau',
]
REQUIRED_BUCKET_ARTIFACT_STEMS = [
    'positive_delta_bucket_overall',
    'negative_delta_bucket_overall',
    'positive_delta_bucket_by_model',
    'negative_delta_bucket_by_model',
]

DELTA_BUCKET_LABELS = ['0.1 - 0.2', '0.2 - 0.3', '0.3 - 0.4', '0.4 - 0.5', '> 0.5']
DELTA_BUCKET_BINS = [0.1, 0.2, 0.3, 0.4, 0.5, np.inf]
HIST_COLORS = {
    'positive': COLORS['blue'],
    'negative': '#B8C7E0',
}

apply_style()
pd.options.display.max_columns = 200
pd.options.display.max_colwidth = 220
pd.options.display.width = 220


# Objects filled either from artifacts or from the raw JSON pass.
inventory_df = pd.DataFrame()
prefix_df = pd.DataFrame()
pair_df = pd.DataFrame()
valid_pair_df = pd.DataFrame()
parse_error_df = pd.DataFrame()


def md(text: str) -> None:
    display(Markdown(text))


def ensure_artifact_dir() -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


def artifact_csv_path(stem: str) -> Path:
    return ARTIFACT_DIR / f'{stem}.csv'


def artifact_exists(stem: str) -> bool:
    return artifact_csv_path(stem).exists()


def artifact_group_exists(stems: list[str] | tuple[str, ...]) -> bool:
    return all(artifact_exists(stem) for stem in stems)


def load_artifact_table(stem: str) -> pd.DataFrame:
    path = artifact_csv_path(stem)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def maybe_save_table(df: pd.DataFrame, stem: str) -> None:
    if not SAVE_ARTIFACTS or df.empty:
        return
    out_dir = ensure_artifact_dir()
    df.to_csv(out_dir / f'{stem}.csv', index=False)


def maybe_save_pair_artifact(df: pd.DataFrame) -> None:
    if not SAVE_ARTIFACTS or not SAVE_PAIR_ARTIFACTS or df.empty:
        return
    out_dir = ensure_artifact_dir()
    df.to_parquet(out_dir / PAIR_ARTIFACT_PATH.name, index=False)


def display_threshold_table(df: pd.DataFrame) -> None:
    if df.empty:
        md('_No rows available._')
        return
    format_map = {}
    for column in ['Threshold']:
        if column in df.columns:
            format_map[column] = '{:.1f}'
    for column in ['Coverage', 'Share']:
        if column in df.columns:
            format_map[column] = '{:.1%}'
    for column in ['Mean Delta', 'Mean Δ_k', 'Pre-rate', 'Post-rate']:
        if column in df.columns:
            format_map[column] = '{:.3f}'
    for column in ['Examples', 'Pairs', 'Total directional pairs']:
        if column in df.columns:
            format_map[column] = '{:.0f}'
    display(df.style.hide(axis='index').format(format_map, na_rep=''))


def make_table(
    summary_df: pd.DataFrame,
    *,
    include_group_columns: bool = True,
    include_counts: bool = True,
) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    return cjt.format_threshold_summary_table(
        summary_df,
        include_group_columns=include_group_columns,
        include_counts=include_counts,
    ).rename(columns={'Mean Delta': 'Mean Δ_k'})


def build_inventory_tables(
    inventory_raw_df: pd.DataFrame,
    parse_error_raw_df: pd.DataFrame,
    pair_raw_df: pd.DataFrame,
    valid_pair_raw_df: pd.DataFrame,
    prefix_raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    inventory_table = inventory_raw_df.loc[
        :,
        [
            'model_display',
            'env_display',
            'source_kind',
            'json_file_count',
            'loaded_examples',
            'loaded_rows',
        ],
    ].rename(
        columns={
            'model_display': 'Model',
            'env_display': 'Environment',
            'source_kind': 'Loaded From',
            'json_file_count': 'Localization JSONs',
            'loaded_examples': 'Examples',
            'loaded_rows': 'Prefix Rows',
        }
    )
    inventory_table['Gap vs 5000'] = inventory_table['Localization JSONs'] - EXPECTED_JSONS_PER_BUNDLE
    inventory_table['Near 5k'] = inventory_table['Localization JSONs'].between(MIN_EXPECTED_JSONS_PER_BUNDLE, EXPECTED_JSONS_PER_BUNDLE)

    non_exact_json_count_table = inventory_table.loc[
        ~inventory_table['Localization JSONs'].eq(EXPECTED_JSONS_PER_BUNDLE),
        ['Model', 'Environment', 'Localization JSONs', 'Gap vs 5000'],
    ].reset_index(drop=True)

    if parse_error_raw_df.empty:
        parse_error_table = pd.DataFrame()
    else:
        parse_error_table = parse_error_raw_df.loc[
            :,
            [column for column in ['bundle_dir', 'path', 'source_kind', 'error'] if column in parse_error_raw_df.columns],
        ].drop_duplicates()

    valid_example_count = int(valid_pair_raw_df.loc[:, cjt.EXAMPLE_KEY_COLUMNS].drop_duplicates().shape[0]) if not valid_pair_raw_df.empty else 0
    json_count_min = int(inventory_raw_df['json_file_count'].min()) if not inventory_raw_df.empty else 0
    json_count_max = int(inventory_raw_df['json_file_count'].max()) if not inventory_raw_df.empty else 0
    summary_text = (
        f'Read directly from raw localization JSON files. Preferred source: `{PREFERRED_SOURCE}`. '
        f'Per-bundle JSON counts range from `{json_count_min:,}` to `{json_count_max:,}` with an expected target of about `{EXPECTED_JSONS_PER_BUNDLE:,}` per model x environment. '
        f'Loaded `{len(prefix_raw_df):,}` prefix rows and `{len(pair_raw_df):,}` consecutive sentence pairs. '
        f'Valid pairs after requiring `num_valid > {MIN_VALID}` on both sides: `{len(valid_pair_raw_df):,}` across `{valid_example_count:,}` examples. '
        f'Bundle parse warnings/errors captured: `{len(parse_error_raw_df):,}`. '
        f'Progress settings: `SHOW_PROGRESS={SHOW_PROGRESS}`, `PROGRESS_LEVEL={PROGRESS_LEVEL}`.'
    )
    return inventory_table, non_exact_json_count_table, parse_error_table, summary_text


def build_summary_payload(pair_raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    positive_overall_summary = cjt.summarize_threshold_sweep(
        pair_raw_df,
        tau_values=TAU_VALUES,
        polarity='positive',
    )
    negative_overall_summary = cjt.summarize_threshold_sweep(
        pair_raw_df,
        tau_values=TAU_VALUES,
        polarity='negative',
    )
    positive_model_summary = cjt.summarize_threshold_sweep(
        pair_raw_df,
        tau_values=TAU_VALUES,
        polarity='positive',
        groupby_columns=['model_display'],
    )
    negative_model_summary = cjt.summarize_threshold_sweep(
        pair_raw_df,
        tau_values=TAU_VALUES,
        polarity='negative',
        groupby_columns=['model_display'],
    )
    positive_env_model_summary = cjt.summarize_threshold_sweep(
        pair_raw_df,
        tau_values=TAU_VALUES,
        polarity='positive',
        groupby_columns=['model_display', 'env_display'],
    )
    negative_env_model_summary = cjt.summarize_threshold_sweep(
        pair_raw_df,
        tau_values=TAU_VALUES,
        polarity='negative',
        groupby_columns=['model_display', 'env_display'],
    )

    positive_overall_table = make_table(positive_overall_summary, include_group_columns=False, include_counts=True)
    negative_overall_table = make_table(negative_overall_summary, include_group_columns=False, include_counts=True)
    positive_model_table = make_table(positive_model_summary, include_group_columns=True, include_counts=True)
    negative_model_table = make_table(negative_model_summary, include_group_columns=True, include_counts=True)
    positive_env_model_table = make_table(positive_env_model_summary, include_group_columns=True, include_counts=True)
    negative_env_model_table = make_table(negative_env_model_summary, include_group_columns=True, include_counts=True)

    return {
        'positive_overall_summary_df': positive_overall_summary,
        'negative_overall_summary_df': negative_overall_summary,
        'positive_model_summary_df': positive_model_summary,
        'negative_model_summary_df': negative_model_summary,
        'positive_env_model_summary_df': positive_env_model_summary,
        'negative_env_model_summary_df': negative_env_model_summary,
        'positive_overall_table_df': positive_overall_table,
        'negative_overall_table_df': negative_overall_table,
        'positive_model_table_df': positive_model_table,
        'negative_model_table_df': negative_model_table,
        'positive_env_model_table_df': positive_env_model_table,
        'negative_env_model_table_df': negative_env_model_table,
        'positive_model_focus_table_df': positive_model_table.loc[positive_model_table['Threshold'].astype(float).eq(DEFAULT_TAU)].reset_index(drop=True),
        'negative_model_focus_table_df': negative_model_table.loc[negative_model_table['Threshold'].astype(float).eq(DEFAULT_TAU)].reset_index(drop=True),
        'positive_env_model_focus_table_df': positive_env_model_table.loc[positive_env_model_table['Threshold'].astype(float).eq(DEFAULT_TAU)].reset_index(drop=True),
        'negative_env_model_focus_table_df': negative_env_model_table.loc[negative_env_model_table['Threshold'].astype(float).eq(DEFAULT_TAU)].reset_index(drop=True),
    }


def maybe_save_summary_payload(payload: dict[str, pd.DataFrame]) -> None:
    save_map = {
        'positive_overall_summary_df': 'positive_overall_summary_raw',
        'negative_overall_summary_df': 'negative_overall_summary_raw',
        'positive_model_summary_df': 'positive_model_summary_raw',
        'negative_model_summary_df': 'negative_model_summary_raw',
        'positive_env_model_summary_df': 'positive_env_model_summary_raw',
        'negative_env_model_summary_df': 'negative_env_model_summary_raw',
        'positive_overall_table_df': 'positive_overall_table',
        'negative_overall_table_df': 'negative_overall_table',
        'positive_model_table_df': 'positive_model_table_all_tau',
        'negative_model_table_df': 'negative_model_table_all_tau',
        'positive_model_focus_table_df': 'positive_model_table_tau_0p3',
        'negative_model_focus_table_df': 'negative_model_table_tau_0p3',
        'positive_env_model_table_df': 'positive_env_model_table_all_tau',
        'negative_env_model_table_df': 'negative_env_model_table_all_tau',
        'positive_env_model_focus_table_df': 'positive_env_model_table_tau_0p3',
        'negative_env_model_focus_table_df': 'negative_env_model_table_tau_0p3',
    }
    for key, stem in save_map.items():
        maybe_save_table(payload.get(key, pd.DataFrame()), stem)


def ensure_raw_threshold_inputs_loaded(reason: str = '') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global inventory_df, prefix_df, parse_error_df, pair_df, valid_pair_df
    if not pair_df.empty:
        return inventory_df, prefix_df, parse_error_df, pair_df, valid_pair_df

    if reason:
        md(f'### Raw Recompute\\nMissing cached inputs for `{reason}`. Falling back to the raw localization JSON pass.')

    inventory_df, prefix_df, parse_error_df = cjt.load_datasetmain_threshold_prefix_df(
        DATASETMAIN_ROOT,
        include_sentence_text=INCLUDE_SENTENCE_TEXT,
        max_json_files_per_bundle=MAX_JSON_FILES_PER_BUNDLE,
        preferred_source=PREFERRED_SOURCE,
        show_progress=SHOW_PROGRESS,
        progress_level=PROGRESS_LEVEL,
    )
    pair_df = cjt.build_consecutive_pair_df(prefix_df, min_valid=MIN_VALID)
    valid_pair_df = pair_df.loc[pair_df['pair_is_valid'].fillna(False)].copy()
    maybe_save_pair_artifact(valid_pair_df)
    return inventory_df, prefix_df, parse_error_df, pair_df, valid_pair_df


def build_delta_bucket_df(
    pair_source_df: pd.DataFrame,
    *,
    polarity: str,
    groupby_columns: list[str] | None = None,
) -> pd.DataFrame:
    if pair_source_df.empty:
        base_columns = list(groupby_columns or []) + ['delta_bucket', 'Pairs', 'Total directional pairs', 'Share', 'Direction']
        return pd.DataFrame(columns=base_columns)

    valid_df = pair_source_df.loc[pair_source_df['pair_is_valid'].fillna(False)].copy()
    polarity_key = str(polarity).strip().lower()
    if polarity_key == 'positive':
        subset = valid_df.loc[valid_df['delta_deception_rate'].gt(0.1)].copy()
        direction_label = 'Toward deception'
    elif polarity_key == 'negative':
        subset = valid_df.loc[valid_df['delta_deception_rate'].lt(-0.1)].copy()
        direction_label = 'Toward truthfulness'
    else:
        raise ValueError(f'Unsupported polarity={polarity!r}')

    if subset.empty:
        base_columns = list(groupby_columns or []) + ['delta_bucket', 'Pairs', 'Total directional pairs', 'Share', 'Direction']
        return pd.DataFrame(columns=base_columns)

    subset['delta_magnitude'] = subset['delta_deception_rate'].abs()
    subset['delta_bucket'] = pd.cut(
        subset['delta_magnitude'],
        bins=DELTA_BUCKET_BINS,
        labels=DELTA_BUCKET_LABELS,
        right=False,
        include_lowest=True,
    )

    group_columns = list(groupby_columns or [])
    bucket_columns = group_columns + ['delta_bucket']
    bucket_df = (
        subset.groupby(bucket_columns, observed=True, as_index=False)
        .size()
        .rename(columns={'size': 'Pairs'})
    )
    if group_columns:
        total_df = (
            subset.groupby(group_columns, observed=True, as_index=False)
            .size()
            .rename(columns={'size': 'Total directional pairs'})
        )
        bucket_df = bucket_df.merge(total_df, on=group_columns, how='left')
    else:
        bucket_df['Total directional pairs'] = int(len(subset))

    bucket_df['Share'] = bucket_df['Pairs'] / bucket_df['Total directional pairs']
    bucket_df['Direction'] = direction_label
    bucket_df['delta_bucket'] = pd.Categorical(bucket_df['delta_bucket'], categories=DELTA_BUCKET_LABELS, ordered=True)

    sort_columns = []
    if 'model_display' in group_columns:
        bucket_df['_model_sort'] = bucket_df['model_display'].map(cj._model_sort_key)
        sort_columns.append('_model_sort')
    if 'env_display' in group_columns:
        bucket_df['_env_sort'] = bucket_df['env_display'].map(cj._env_sort_key)
        sort_columns.append('_env_sort')
    sort_columns.append('delta_bucket')
    bucket_df = bucket_df.sort_values(sort_columns).drop(columns=['_model_sort', '_env_sort'], errors='ignore').reset_index(drop=True)
    bucket_df['delta_bucket'] = bucket_df['delta_bucket'].astype(str)
    return bucket_df


def maybe_save_bucket_payload(payload: dict[str, pd.DataFrame]) -> None:
    save_map = {
        'positive_bucket_overall_df': 'positive_delta_bucket_overall',
        'negative_bucket_overall_df': 'negative_delta_bucket_overall',
        'positive_bucket_by_model_df': 'positive_delta_bucket_by_model',
        'negative_bucket_by_model_df': 'negative_delta_bucket_by_model',
    }
    for key, stem in save_map.items():
        maybe_save_table(payload.get(key, pd.DataFrame()), stem)


def build_bucket_payload(pair_source_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        'positive_bucket_overall_df': build_delta_bucket_df(pair_source_df, polarity='positive'),
        'negative_bucket_overall_df': build_delta_bucket_df(pair_source_df, polarity='negative'),
        'positive_bucket_by_model_df': build_delta_bucket_df(pair_source_df, polarity='positive', groupby_columns=['model_display']),
        'negative_bucket_by_model_df': build_delta_bucket_df(pair_source_df, polarity='negative', groupby_columns=['model_display']),
    }


def count_label(value: float) -> str:
    numeric = float(value)
    if numeric >= 1_000_000:
        return f'{numeric / 1_000_000:.1f}M'
    if numeric >= 1_000:
        return f'{numeric / 1_000:.1f}k'
    return f'{numeric:.0f}'


def plot_bucket_histograms(positive_bucket_df: pd.DataFrame, negative_bucket_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True)
    panel_specs = [
        ('A. Toward deception', positive_bucket_df, HIST_COLORS['positive'], 'Valid pairs with Δ_k > 0.1'),
        ('B. Toward truthfulness', negative_bucket_df, HIST_COLORS['negative'], 'Valid pairs with Δ_k < -0.1'),
    ]
    max_pairs = 0.0
    for _, bucket_df, _, _ in panel_specs:
        if not bucket_df.empty:
            max_pairs = max(max_pairs, float(pd.to_numeric(bucket_df['Pairs'], errors='coerce').max()))
    y_max = 1.22 * max(max_pairs, 1.0)

    for axis_index, (ax, title, bucket_df, color, subtitle) in enumerate(zip(axes, *zip(*panel_specs))):
        plot_df = bucket_df.copy()
        if plot_df.empty:
            style_panel_title(ax, title)
            style_axes(ax, ylabel='Pairs' if axis_index == 0 else None, xlabel='|Δ_k| bucket', ylim=(0, 1), grid_axis='y')
            ax.text(0.5, 0.5, 'No qualifying pairs', transform=ax.transAxes, ha='center', va='center', color=COLORS['muted_ink'])
            continue

        plot_df['delta_bucket'] = pd.Categorical(plot_df['delta_bucket'], categories=DELTA_BUCKET_LABELS, ordered=True)
        plot_df = plot_df.set_index('delta_bucket').reindex(DELTA_BUCKET_LABELS).reset_index()
        plot_df['Pairs'] = pd.to_numeric(plot_df['Pairs'], errors='coerce').fillna(0.0)
        plot_df['Share'] = pd.to_numeric(plot_df['Share'], errors='coerce').fillna(0.0)

        x_positions = np.arange(len(DELTA_BUCKET_LABELS))
        bars = ax.bar(
            x_positions,
            plot_df['Pairs'],
            width=0.68,
            color=color,
            edgecolor=COLORS['light_gray'],
            linewidth=0.8,
            zorder=3,
        )
        style_panel_title(ax, title)
        style_axes(ax, ylabel='Pairs' if axis_index == 0 else None, xlabel='|Δ_k| bucket', ylim=(0, y_max), grid_axis='y')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(DELTA_BUCKET_LABELS)
        ax.tick_params(axis='x', labelrotation=0)
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, ha='left', va='top', fontsize=8.6, color=COLORS['muted_ink'])
        ax.text(0.98, 0.98, f'n = {int(plot_df["Pairs"].sum()):,}', transform=ax.transAxes, ha='right', va='top', fontsize=8.6, color=COLORS['muted_ink'])

        for bar, row in zip(bars, plot_df.itertuples(index=False)):
            height = float(bar.get_height())
            label = f'{count_label(height)}\\n{100.0 * float(row.Share):.1f}%'
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015 * y_max,
                label,
                ha='center',
                va='bottom',
                fontsize=7.2,
                color=COLORS['ink'],
                clip_on=False,
            )

    add_figure_note(
        fig,
        'Bars show counts of valid same-direction junctures in each |Δ_k| bucket; labels report count and within-direction share.',
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig
"""
        ),
        new_code_cell(
            """tables_loaded_from_artifacts = False
load_summary_message = ''

inventory_table_df = pd.DataFrame()
non_exact_json_count_df = pd.DataFrame()
parse_error_table_df = pd.DataFrame()
positive_overall_summary_df = pd.DataFrame()
negative_overall_summary_df = pd.DataFrame()
positive_model_summary_df = pd.DataFrame()
negative_model_summary_df = pd.DataFrame()
positive_env_model_summary_df = pd.DataFrame()
negative_env_model_summary_df = pd.DataFrame()
positive_overall_table_df = pd.DataFrame()
negative_overall_table_df = pd.DataFrame()
positive_model_table_df = pd.DataFrame()
negative_model_table_df = pd.DataFrame()
positive_env_model_table_df = pd.DataFrame()
negative_env_model_table_df = pd.DataFrame()
positive_model_focus_table_df = pd.DataFrame()
negative_model_focus_table_df = pd.DataFrame()
positive_env_model_focus_table_df = pd.DataFrame()
negative_env_model_focus_table_df = pd.DataFrame()

if LOAD_ARTIFACTS_IF_AVAILABLE and artifact_group_exists(REQUIRED_TABLE_ARTIFACT_STEMS):
    tables_loaded_from_artifacts = True
    inventory_table_df = load_artifact_table('inventory')
    non_exact_json_count_df = load_artifact_table('non_exact_json_counts')
    parse_error_table_df = load_artifact_table('parse_warnings')
    positive_overall_summary_df = load_artifact_table('positive_overall_summary_raw')
    negative_overall_summary_df = load_artifact_table('negative_overall_summary_raw')
    positive_model_summary_df = load_artifact_table('positive_model_summary_raw')
    negative_model_summary_df = load_artifact_table('negative_model_summary_raw')
    positive_env_model_summary_df = load_artifact_table('positive_env_model_summary_raw')
    negative_env_model_summary_df = load_artifact_table('negative_env_model_summary_raw')
    positive_overall_table_df = load_artifact_table('positive_overall_table')
    negative_overall_table_df = load_artifact_table('negative_overall_table')
    positive_model_table_df = load_artifact_table('positive_model_table_all_tau')
    negative_model_table_df = load_artifact_table('negative_model_table_all_tau')
    positive_env_model_table_df = load_artifact_table('positive_env_model_table_all_tau')
    negative_env_model_table_df = load_artifact_table('negative_env_model_table_all_tau')
    positive_model_focus_table_df = load_artifact_table('positive_model_table_tau_0p3')
    negative_model_focus_table_df = load_artifact_table('negative_model_table_tau_0p3')
    positive_env_model_focus_table_df = load_artifact_table('positive_env_model_table_tau_0p3')
    negative_env_model_focus_table_df = load_artifact_table('negative_env_model_table_tau_0p3')

    if positive_model_focus_table_df.empty and not positive_model_table_df.empty:
        positive_model_focus_table_df = positive_model_table_df.loc[
            positive_model_table_df['Threshold'].astype(float).eq(DEFAULT_TAU)
        ].reset_index(drop=True)
    if negative_model_focus_table_df.empty and not negative_model_table_df.empty:
        negative_model_focus_table_df = negative_model_table_df.loc[
            negative_model_table_df['Threshold'].astype(float).eq(DEFAULT_TAU)
        ].reset_index(drop=True)
    if positive_env_model_focus_table_df.empty and not positive_env_model_table_df.empty:
        positive_env_model_focus_table_df = positive_env_model_table_df.loc[
            positive_env_model_table_df['Threshold'].astype(float).eq(DEFAULT_TAU)
        ].reset_index(drop=True)
    if negative_env_model_focus_table_df.empty and not negative_env_model_table_df.empty:
        negative_env_model_focus_table_df = negative_env_model_table_df.loc[
            negative_env_model_table_df['Threshold'].astype(float).eq(DEFAULT_TAU)
        ].reset_index(drop=True)

    load_summary_message = (
        f'Loaded saved threshold tables from `{ARTIFACT_DIR}`. '
        'The expensive raw localization JSON scan was skipped for the table sections.'
    )
else:
    inventory_df, prefix_df, parse_error_df, pair_df, valid_pair_df = ensure_raw_threshold_inputs_loaded(reason='threshold tables')
    inventory_table_df, non_exact_json_count_df, parse_error_table_df, load_summary_message = build_inventory_tables(
        inventory_df,
        parse_error_df,
        pair_df,
        valid_pair_df,
        prefix_df,
    )
    summary_payload = build_summary_payload(pair_df)
    positive_overall_summary_df = summary_payload['positive_overall_summary_df']
    negative_overall_summary_df = summary_payload['negative_overall_summary_df']
    positive_model_summary_df = summary_payload['positive_model_summary_df']
    negative_model_summary_df = summary_payload['negative_model_summary_df']
    positive_env_model_summary_df = summary_payload['positive_env_model_summary_df']
    negative_env_model_summary_df = summary_payload['negative_env_model_summary_df']
    positive_overall_table_df = summary_payload['positive_overall_table_df']
    negative_overall_table_df = summary_payload['negative_overall_table_df']
    positive_model_table_df = summary_payload['positive_model_table_df']
    negative_model_table_df = summary_payload['negative_model_table_df']
    positive_env_model_table_df = summary_payload['positive_env_model_table_df']
    negative_env_model_table_df = summary_payload['negative_env_model_table_df']
    positive_model_focus_table_df = summary_payload['positive_model_focus_table_df']
    negative_model_focus_table_df = summary_payload['negative_model_focus_table_df']
    positive_env_model_focus_table_df = summary_payload['positive_env_model_focus_table_df']
    negative_env_model_focus_table_df = summary_payload['negative_env_model_focus_table_df']

    maybe_save_table(inventory_table_df, 'inventory')
    maybe_save_table(non_exact_json_count_df, 'non_exact_json_counts')
    maybe_save_table(parse_error_table_df, 'parse_warnings')
    maybe_save_summary_payload(summary_payload)

md('## Load Summary')
md(load_summary_message)
display(inventory_table_df.style.hide(axis='index'))

if not non_exact_json_count_df.empty:
    md('### Bundles Not Exactly 5000 JSONs')
    display(non_exact_json_count_df.style.hide(axis='index'))

if not parse_error_table_df.empty:
    md('### Parse Warnings')
    display(parse_error_table_df.style.hide(axis='index'))
"""
        ),
        new_markdown_cell(
            """## Overall Threshold Tables

These are the pooled DatasetMain threshold-sensitivity tables for positive and negative commitment junctures.
"""
        ),
        new_code_cell(
            """md('### Positive Commitments: Toward Deception')
display_threshold_table(positive_overall_table_df)

md('### Negative Commitments: Toward Truthfulness')
display_threshold_table(negative_overall_table_df)
"""
        ),
        new_markdown_cell(
            """## Per-Model Breakdown

These tables keep the same thresholds but split the summary by model.
"""
        ),
        new_code_cell(
            """md('### Positive Commitments by Model')
display_threshold_table(positive_model_table_df)

md('### Negative Commitments by Model')
display_threshold_table(negative_model_table_df)
"""
        ),
        new_markdown_cell(
            """## Delta Histograms

The histogram section uses saved bucket summaries when available. If those bucket summaries are missing, it will try a saved pair-level parquet cache next, and only then fall back to the full raw JSON pass.
"""
        ),
        new_code_cell(
            """bucket_tables_loaded_from_artifacts = False
bucket_message = ''
positive_bucket_overall_df = pd.DataFrame()
negative_bucket_overall_df = pd.DataFrame()
positive_bucket_by_model_df = pd.DataFrame()
negative_bucket_by_model_df = pd.DataFrame()

if LOAD_ARTIFACTS_IF_AVAILABLE and artifact_group_exists(REQUIRED_BUCKET_ARTIFACT_STEMS):
    bucket_tables_loaded_from_artifacts = True
    positive_bucket_overall_df = load_artifact_table('positive_delta_bucket_overall')
    negative_bucket_overall_df = load_artifact_table('negative_delta_bucket_overall')
    positive_bucket_by_model_df = load_artifact_table('positive_delta_bucket_by_model')
    negative_bucket_by_model_df = load_artifact_table('negative_delta_bucket_by_model')
    bucket_message = f'Loaded saved delta-bucket summaries from `{ARTIFACT_DIR}`.'
else:
    if pair_df.empty and PAIR_ARTIFACT_PATH.exists():
        valid_pair_df = pd.read_parquet(PAIR_ARTIFACT_PATH)
        pair_df = valid_pair_df.copy()
        bucket_message = f'Loaded `{PAIR_ARTIFACT_PATH.name}` and rebuilt the delta-bucket summaries without re-scanning raw JSON files.'
    elif pair_df.empty:
        inventory_df, prefix_df, parse_error_df, pair_df, valid_pair_df = ensure_raw_threshold_inputs_loaded(reason='delta histograms')
        bucket_message = 'Computed delta-bucket summaries from the raw localization JSON files because no cached bucket artifacts were present.'
    else:
        bucket_message = 'Reused in-memory `pair_df` to build the delta-bucket summaries.'

    bucket_payload = build_bucket_payload(pair_df)
    positive_bucket_overall_df = bucket_payload['positive_bucket_overall_df']
    negative_bucket_overall_df = bucket_payload['negative_bucket_overall_df']
    positive_bucket_by_model_df = bucket_payload['positive_bucket_by_model_df']
    negative_bucket_by_model_df = bucket_payload['negative_bucket_by_model_df']
    maybe_save_bucket_payload(bucket_payload)

md(bucket_message)

bucket_histogram_fig = plot_bucket_histograms(
    positive_bucket_overall_df,
    negative_bucket_overall_df,
)
plt.show()

bucket_overall_display_df = pd.concat(
    [positive_bucket_overall_df, negative_bucket_overall_df],
    ignore_index=True,
)
if not bucket_overall_display_df.empty:
    bucket_overall_display_df = bucket_overall_display_df.rename(
        columns={
            'Direction': 'Direction',
            'delta_bucket': '|Δ_k| bucket',
            'Pairs': 'Pairs',
            'Share': 'Share',
        }
    )
    md('### Bucket Counts Used For The Histogram')
    display_threshold_table(bucket_overall_display_df)
"""
        ),
        new_markdown_cell(
            """## Per Model x Environment

For readability, this section keeps the model x environment breakdown at the default threshold `tau = 0.3`.
"""
        ),
        new_code_cell(
            """md(f'### Positive Commitments by Model x Environment at tau={DEFAULT_TAU:.1f}')
display_threshold_table(positive_env_model_focus_table_df)

md(f'### Negative Commitments by Model x Environment at tau={DEFAULT_TAU:.1f}')
display_threshold_table(negative_env_model_focus_table_df)
"""
        ),
        new_markdown_cell(
            """## Handy Objects

Useful data frames left in memory after running the notebook:

- `inventory_table_df`: bundle-level load summary table
- `positive_overall_table_df`, `negative_overall_table_df`
- `positive_model_table_df`, `negative_model_table_df`
- `positive_env_model_focus_table_df`, `negative_env_model_focus_table_df`
- `positive_bucket_overall_df`, `negative_bucket_overall_df`
- `positive_bucket_by_model_df`, `negative_bucket_by_model_df`

If the notebook loaded only saved artifacts, `prefix_df`, `pair_df`, and `valid_pair_df` may stay empty unless the histogram section had to rebuild them.
"""
        ),
    ]

    return new_notebook(
        cells=cells,
        metadata={
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
                'version': '3',
            },
        },
    )


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(nbformat.writes(notebook), encoding='utf-8')
    print(f'Wrote {NOTEBOOK_PATH}')


if __name__ == '__main__':
    main()
