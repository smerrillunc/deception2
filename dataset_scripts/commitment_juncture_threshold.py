from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_scripts_common import (
    ensure_dir,
    ensure_import_paths,
    resolve_datasetmain_root,
    resolve_output_dir,
    resolve_repo_root,
    save_csv,
    utc_now_iso,
    write_json,
)


SCRIPT_NAME = Path(__file__).stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone runner for commitment_juncture_threshold.ipynb.",
    )
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--datasetmain-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--min-valid", type=int, default=10)
    parser.add_argument("--preferred-source", choices=["localization_json", "parquet", "auto"], default="localization_json")
    parser.add_argument("--default-tau", type=float, default=0.3)
    parser.add_argument("--max-json-files-per-bundle", type=int, default=None)
    parser.add_argument("--progress-level", choices=["bundle", "file"], default="bundle")

    parser.set_defaults(show_progress=True)
    parser.add_argument("--show-progress", dest="show_progress", action="store_true")
    parser.add_argument("--no-show-progress", dest="show_progress", action="store_false")

    parser.set_defaults(load_artifacts_if_available=True)
    parser.add_argument("--load-artifacts-if-available", dest="load_artifacts_if_available", action="store_true")
    parser.add_argument("--no-load-artifacts-if-available", dest="load_artifacts_if_available", action="store_false")

    parser.set_defaults(save_artifacts=True)
    parser.add_argument("--save-artifacts", dest="save_artifacts", action="store_true")
    parser.add_argument("--no-save-artifacts", dest="save_artifacts", action="store_false")

    parser.set_defaults(save_pair_artifacts=True)
    parser.add_argument("--save-pair-artifacts", dest="save_pair_artifacts", action="store_true")
    parser.add_argument("--no-save-pair-artifacts", dest="save_pair_artifacts", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    repo_root = resolve_repo_root(args.repo_root)
    datasetmain_root = resolve_datasetmain_root(repo_root, args.datasetmain_root)
    output_dir = ensure_dir(resolve_output_dir(repo_root, SCRIPT_NAME, args.output_dir))

    ensure_import_paths(repo_root, include_styles=True)

    from neurips import COLORS, add_figure_note, style_axes, style_panel_title
    import datasetmain_commitment_juncture_prevalence_lib as cj
    import datasetmain_commitment_juncture_threshold_lib as cjt

    cj = importlib.reload(cj)
    cjt = importlib.reload(cjt)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    expected_jsons_per_bundle = 5000
    min_expected_jsons_per_bundle = 4900
    tau_values = cjt.TAU_VALUES
    focus_tau_stem = f"tau_{str(args.default_tau).replace('.', 'p')}"
    required_table_artifact_stems = [
        "inventory",
        "positive_overall_table",
        "negative_overall_table",
        "positive_model_table_all_tau",
        "negative_model_table_all_tau",
        f"positive_env_model_table_{focus_tau_stem}",
        f"negative_env_model_table_{focus_tau_stem}",
    ]
    required_bucket_artifact_stems = [
        "positive_delta_bucket_overall",
        "negative_delta_bucket_overall",
        "positive_delta_bucket_by_model",
        "negative_delta_bucket_by_model",
    ]
    delta_bucket_labels = ["0.1 - 0.2", "0.2 - 0.3", "0.3 - 0.4", "0.4 - 0.5", "> 0.5"]
    delta_bucket_bins = [0.1, 0.2, 0.3, 0.4, 0.5, np.inf]
    hist_colors = {
        "positive": COLORS["blue"],
        "negative": "#B8C7E0",
    }

    inventory_df = pd.DataFrame()
    prefix_df = pd.DataFrame()
    parse_error_df = pd.DataFrame()
    pair_df = pd.DataFrame()
    valid_pair_df = pd.DataFrame()
    pair_artifact_path = output_dir / "valid_pair_df.parquet"

    def artifact_csv_path(stem: str) -> Path:
        return output_dir / f"{stem}.csv"

    def artifact_exists(stem: str) -> bool:
        return artifact_csv_path(stem).exists()

    def artifact_group_exists(stems: list[str]) -> bool:
        return all(artifact_exists(stem) for stem in stems)

    def load_artifact_table(stem: str) -> pd.DataFrame:
        path = artifact_csv_path(stem)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def maybe_save_table(df: pd.DataFrame, stem: str) -> Path | None:
        if (not args.save_artifacts) or df.empty:
            return None
        return save_csv(df, output_dir, stem)

    def maybe_save_pair_artifact(df: pd.DataFrame) -> Path | None:
        if (not args.save_artifacts) or (not args.save_pair_artifacts) or df.empty:
            return None
        df.to_parquet(pair_artifact_path, index=False)
        return pair_artifact_path

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
        ).rename(columns={"Mean Delta": "Mean Δ_k"})

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
                "model_display",
                "env_display",
                "source_kind",
                "json_file_count",
                "loaded_examples",
                "loaded_rows",
            ],
        ].rename(
            columns={
                "model_display": "Model",
                "env_display": "Environment",
                "source_kind": "Loaded From",
                "json_file_count": "Localization JSONs",
                "loaded_examples": "Examples",
                "loaded_rows": "Prefix Rows",
            }
        )
        inventory_table["Gap vs 5000"] = inventory_table["Localization JSONs"] - expected_jsons_per_bundle
        inventory_table["Near 5k"] = inventory_table["Localization JSONs"].between(
            min_expected_jsons_per_bundle,
            expected_jsons_per_bundle,
        )

        non_exact_json_count_table = inventory_table.loc[
            ~inventory_table["Localization JSONs"].eq(expected_jsons_per_bundle),
            ["Model", "Environment", "Localization JSONs", "Gap vs 5000"],
        ].reset_index(drop=True)

        if parse_error_raw_df.empty:
            parse_error_table = pd.DataFrame()
        else:
            parse_error_table = parse_error_raw_df.loc[
                :,
                [column for column in ["bundle_dir", "path", "source_kind", "error"] if column in parse_error_raw_df.columns],
            ].drop_duplicates()

        valid_example_count = (
            int(valid_pair_raw_df.loc[:, cjt.EXAMPLE_KEY_COLUMNS].drop_duplicates().shape[0])
            if not valid_pair_raw_df.empty
            else 0
        )
        json_count_min = int(inventory_raw_df["json_file_count"].min()) if not inventory_raw_df.empty else 0
        json_count_max = int(inventory_raw_df["json_file_count"].max()) if not inventory_raw_df.empty else 0
        summary_text = (
            f"Read directly from raw localization JSON files. Preferred source: {args.preferred_source}. "
            f"Per-bundle JSON counts range from {json_count_min:,} to {json_count_max:,} with an expected target "
            f"of about {expected_jsons_per_bundle:,} per model x environment. Loaded {len(prefix_raw_df):,} prefix rows "
            f"and {len(pair_raw_df):,} consecutive sentence pairs. Valid pairs after requiring num_valid > {args.min_valid} "
            f"on both sides: {len(valid_pair_raw_df):,} across {valid_example_count:,} examples. "
            f"Bundle parse warnings/errors captured: {len(parse_error_raw_df):,}. "
            f"Progress settings: SHOW_PROGRESS={args.show_progress}, PROGRESS_LEVEL={args.progress_level}."
        )
        return inventory_table, non_exact_json_count_table, parse_error_table, summary_text

    def build_summary_payload(pair_raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        positive_overall_summary = cjt.summarize_threshold_sweep(
            pair_raw_df,
            tau_values=tau_values,
            polarity="positive",
        )
        negative_overall_summary = cjt.summarize_threshold_sweep(
            pair_raw_df,
            tau_values=tau_values,
            polarity="negative",
        )
        positive_model_summary = cjt.summarize_threshold_sweep(
            pair_raw_df,
            tau_values=tau_values,
            polarity="positive",
            groupby_columns=["model_display"],
        )
        negative_model_summary = cjt.summarize_threshold_sweep(
            pair_raw_df,
            tau_values=tau_values,
            polarity="negative",
            groupby_columns=["model_display"],
        )
        positive_env_model_summary = cjt.summarize_threshold_sweep(
            pair_raw_df,
            tau_values=tau_values,
            polarity="positive",
            groupby_columns=["model_display", "env_display"],
        )
        negative_env_model_summary = cjt.summarize_threshold_sweep(
            pair_raw_df,
            tau_values=tau_values,
            polarity="negative",
            groupby_columns=["model_display", "env_display"],
        )

        positive_overall_table = make_table(positive_overall_summary, include_group_columns=False, include_counts=True)
        negative_overall_table = make_table(negative_overall_summary, include_group_columns=False, include_counts=True)
        positive_model_table = make_table(positive_model_summary, include_group_columns=True, include_counts=True)
        negative_model_table = make_table(negative_model_summary, include_group_columns=True, include_counts=True)
        positive_env_model_table = make_table(positive_env_model_summary, include_group_columns=True, include_counts=True)
        negative_env_model_table = make_table(negative_env_model_summary, include_group_columns=True, include_counts=True)

        return {
            "positive_overall_summary_df": positive_overall_summary,
            "negative_overall_summary_df": negative_overall_summary,
            "positive_model_summary_df": positive_model_summary,
            "negative_model_summary_df": negative_model_summary,
            "positive_env_model_summary_df": positive_env_model_summary,
            "negative_env_model_summary_df": negative_env_model_summary,
            "positive_overall_table_df": positive_overall_table,
            "negative_overall_table_df": negative_overall_table,
            "positive_model_table_df": positive_model_table,
            "negative_model_table_df": negative_model_table,
            "positive_env_model_table_df": positive_env_model_table,
            "negative_env_model_table_df": negative_env_model_table,
            "positive_model_focus_table_df": positive_model_table.loc[
                positive_model_table["Threshold"].astype(float).eq(args.default_tau)
            ].reset_index(drop=True),
            "negative_model_focus_table_df": negative_model_table.loc[
                negative_model_table["Threshold"].astype(float).eq(args.default_tau)
            ].reset_index(drop=True),
            "positive_env_model_focus_table_df": positive_env_model_table.loc[
                positive_env_model_table["Threshold"].astype(float).eq(args.default_tau)
            ].reset_index(drop=True),
            "negative_env_model_focus_table_df": negative_env_model_table.loc[
                negative_env_model_table["Threshold"].astype(float).eq(args.default_tau)
            ].reset_index(drop=True),
        }

    def maybe_save_summary_payload(payload: dict[str, pd.DataFrame]) -> None:
        save_map = {
            "positive_overall_summary_df": "positive_overall_summary_raw",
            "negative_overall_summary_df": "negative_overall_summary_raw",
            "positive_model_summary_df": "positive_model_summary_raw",
            "negative_model_summary_df": "negative_model_summary_raw",
            "positive_env_model_summary_df": "positive_env_model_summary_raw",
            "negative_env_model_summary_df": "negative_env_model_summary_raw",
            "positive_overall_table_df": "positive_overall_table",
            "negative_overall_table_df": "negative_overall_table",
            "positive_model_table_df": "positive_model_table_all_tau",
            "negative_model_table_df": "negative_model_table_all_tau",
            "positive_model_focus_table_df": f"positive_model_table_{focus_tau_stem}",
            "negative_model_focus_table_df": f"negative_model_table_{focus_tau_stem}",
            "positive_env_model_table_df": "positive_env_model_table_all_tau",
            "negative_env_model_table_df": "negative_env_model_table_all_tau",
            "positive_env_model_focus_table_df": f"positive_env_model_table_{focus_tau_stem}",
            "negative_env_model_focus_table_df": f"negative_env_model_table_{focus_tau_stem}",
        }
        for key, stem in save_map.items():
            maybe_save_table(payload.get(key, pd.DataFrame()), stem)

    def ensure_raw_threshold_inputs_loaded() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        nonlocal inventory_df, prefix_df, parse_error_df, pair_df, valid_pair_df
        if not pair_df.empty:
            return inventory_df, prefix_df, parse_error_df, pair_df, valid_pair_df

        inventory_df, prefix_df, parse_error_df = cjt.load_datasetmain_threshold_prefix_df(
            datasetmain_root,
            include_sentence_text=False,
            max_json_files_per_bundle=args.max_json_files_per_bundle,
            preferred_source=args.preferred_source,
            show_progress=args.show_progress,
            progress_level=args.progress_level,
        )
        pair_df = cjt.build_consecutive_pair_df(prefix_df, min_valid=args.min_valid)
        valid_pair_df = pair_df.loc[pair_df["pair_is_valid"].fillna(False)].copy()
        maybe_save_pair_artifact(valid_pair_df)
        return inventory_df, prefix_df, parse_error_df, pair_df, valid_pair_df

    def build_delta_bucket_df(
        pair_source_df: pd.DataFrame,
        *,
        polarity: str,
        groupby_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if pair_source_df.empty:
            base_columns = list(groupby_columns or []) + [
                "delta_bucket",
                "Pairs",
                "Total directional pairs",
                "Share",
                "Direction",
            ]
            return pd.DataFrame(columns=base_columns)

        valid_df = pair_source_df.loc[pair_source_df["pair_is_valid"].fillna(False)].copy()
        polarity_key = str(polarity).strip().lower()
        if polarity_key == "positive":
            subset = valid_df.loc[valid_df["delta_deception_rate"].gt(0.1)].copy()
            direction_label = "Toward deception"
        elif polarity_key == "negative":
            subset = valid_df.loc[valid_df["delta_deception_rate"].lt(-0.1)].copy()
            direction_label = "Toward truthfulness"
        else:
            raise ValueError(f"Unsupported polarity={polarity!r}")

        if subset.empty:
            base_columns = list(groupby_columns or []) + [
                "delta_bucket",
                "Pairs",
                "Total directional pairs",
                "Share",
                "Direction",
            ]
            return pd.DataFrame(columns=base_columns)

        subset["delta_magnitude"] = subset["delta_deception_rate"].abs()
        subset["delta_bucket"] = pd.cut(
            subset["delta_magnitude"],
            bins=delta_bucket_bins,
            labels=delta_bucket_labels,
            right=False,
            include_lowest=True,
        )

        group_columns = list(groupby_columns or [])
        bucket_columns = group_columns + ["delta_bucket"]
        bucket_df = (
            subset.groupby(bucket_columns, observed=True, as_index=False)
            .size()
            .rename(columns={"size": "Pairs"})
        )
        if group_columns:
            total_df = (
                subset.groupby(group_columns, observed=True, as_index=False)
                .size()
                .rename(columns={"size": "Total directional pairs"})
            )
            bucket_df = bucket_df.merge(total_df, on=group_columns, how="left")
        else:
            bucket_df["Total directional pairs"] = int(len(subset))

        bucket_df["Share"] = bucket_df["Pairs"] / bucket_df["Total directional pairs"]
        bucket_df["Direction"] = direction_label
        bucket_df["delta_bucket"] = pd.Categorical(
            bucket_df["delta_bucket"],
            categories=delta_bucket_labels,
            ordered=True,
        )

        sort_columns: list[str] = []
        if "model_display" in group_columns:
            bucket_df["_model_sort"] = bucket_df["model_display"].map(cj._model_sort_key)
            sort_columns.append("_model_sort")
        if "env_display" in group_columns:
            bucket_df["_env_sort"] = bucket_df["env_display"].map(cj._env_sort_key)
            sort_columns.append("_env_sort")
        sort_columns.append("delta_bucket")
        bucket_df = (
            bucket_df.sort_values(sort_columns)
            .drop(columns=["_model_sort", "_env_sort"], errors="ignore")
            .reset_index(drop=True)
        )
        bucket_df["delta_bucket"] = bucket_df["delta_bucket"].astype(str)
        return bucket_df

    def build_bucket_payload(pair_source_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {
            "positive_bucket_overall_df": build_delta_bucket_df(pair_source_df, polarity="positive"),
            "negative_bucket_overall_df": build_delta_bucket_df(pair_source_df, polarity="negative"),
            "positive_bucket_by_model_df": build_delta_bucket_df(
                pair_source_df,
                polarity="positive",
                groupby_columns=["model_display"],
            ),
            "negative_bucket_by_model_df": build_delta_bucket_df(
                pair_source_df,
                polarity="negative",
                groupby_columns=["model_display"],
            ),
        }

    def maybe_save_bucket_payload(payload: dict[str, pd.DataFrame]) -> None:
        save_map = {
            "positive_bucket_overall_df": "positive_delta_bucket_overall",
            "negative_bucket_overall_df": "negative_delta_bucket_overall",
            "positive_bucket_by_model_df": "positive_delta_bucket_by_model",
            "negative_bucket_by_model_df": "negative_delta_bucket_by_model",
        }
        for key, stem in save_map.items():
            maybe_save_table(payload.get(key, pd.DataFrame()), stem)

    def count_label(value: float) -> str:
        numeric = float(value)
        if numeric >= 1_000_000:
            return f"{numeric / 1_000_000:.1f}M"
        if numeric >= 1_000:
            return f"{numeric / 1_000:.1f}k"
        return f"{numeric:.0f}"

    def plot_bucket_histograms(positive_bucket_df: pd.DataFrame, negative_bucket_df: pd.DataFrame) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True)
        panel_specs = [
            ("A. Toward deception", positive_bucket_df, hist_colors["positive"], "Valid pairs with Δ_k > 0.1"),
            ("B. Toward truthfulness", negative_bucket_df, hist_colors["negative"], "Valid pairs with Δ_k < -0.1"),
        ]

        max_pairs = 0.0
        for _, bucket_df, _, _ in panel_specs:
            if not bucket_df.empty:
                max_pairs = max(max_pairs, float(pd.to_numeric(bucket_df["Pairs"], errors="coerce").max()))
        y_max = 1.22 * max(max_pairs, 1.0)

        for axis_index, (ax, panel_spec) in enumerate(zip(axes, panel_specs, strict=True)):
            plot_title, plot_df_source, plot_color, plot_subtitle = panel_spec
            plot_df = plot_df_source.copy()
            if plot_df.empty:
                style_panel_title(ax, plot_title)
                style_axes(ax, ylabel="Pairs" if axis_index == 0 else None, xlabel="|Δ_k| bucket", ylim=(0, 1), grid_axis="y")
                ax.text(
                    0.5,
                    0.5,
                    "No qualifying pairs",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color=COLORS["muted_ink"],
                )
                continue

            plot_df["delta_bucket"] = pd.Categorical(plot_df["delta_bucket"], categories=delta_bucket_labels, ordered=True)
            plot_df = plot_df.set_index("delta_bucket").reindex(delta_bucket_labels).reset_index()
            plot_df["Pairs"] = pd.to_numeric(plot_df["Pairs"], errors="coerce").fillna(0.0)
            plot_df["Share"] = pd.to_numeric(plot_df["Share"], errors="coerce").fillna(0.0)

            x_positions = np.arange(len(delta_bucket_labels))
            bars = ax.bar(
                x_positions,
                plot_df["Pairs"],
                width=0.68,
                color=plot_color,
                edgecolor=COLORS["light_gray"],
                linewidth=0.8,
                zorder=3,
            )
            style_panel_title(ax, plot_title)
            style_axes(ax, ylabel="Pairs" if axis_index == 0 else None, xlabel="|Δ_k| bucket", ylim=(0, y_max), grid_axis="y")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(delta_bucket_labels)
            ax.text(0.02, 0.98, plot_subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color=COLORS["muted_ink"])
            ax.text(
                0.98,
                0.98,
                f"n = {int(plot_df['Pairs'].sum()):,}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.6,
                color=COLORS["muted_ink"],
            )

            for bar, row in zip(bars, plot_df.itertuples(index=False), strict=False):
                height = float(bar.get_height())
                label = f"{count_label(height)}\n{100.0 * float(row.Share):.1f}%"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.015 * y_max,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    color=COLORS["ink"],
                    clip_on=False,
                )

        add_figure_note(
            fig,
            "Bars show counts of valid same-direction junctures in each |Δ_k| bucket; labels report count and within-direction share.",
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        return fig

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

    tables_loaded_from_artifacts = False
    load_summary_message = ""

    if args.load_artifacts_if_available and artifact_group_exists(required_table_artifact_stems):
        tables_loaded_from_artifacts = True
        inventory_table_df = load_artifact_table("inventory")
        non_exact_json_count_df = load_artifact_table("non_exact_json_counts")
        parse_error_table_df = load_artifact_table("parse_warnings")
        positive_overall_summary_df = load_artifact_table("positive_overall_summary_raw")
        negative_overall_summary_df = load_artifact_table("negative_overall_summary_raw")
        positive_model_summary_df = load_artifact_table("positive_model_summary_raw")
        negative_model_summary_df = load_artifact_table("negative_model_summary_raw")
        positive_env_model_summary_df = load_artifact_table("positive_env_model_summary_raw")
        negative_env_model_summary_df = load_artifact_table("negative_env_model_summary_raw")
        positive_overall_table_df = load_artifact_table("positive_overall_table")
        negative_overall_table_df = load_artifact_table("negative_overall_table")
        positive_model_table_df = load_artifact_table("positive_model_table_all_tau")
        negative_model_table_df = load_artifact_table("negative_model_table_all_tau")
        positive_env_model_table_df = load_artifact_table("positive_env_model_table_all_tau")
        negative_env_model_table_df = load_artifact_table("negative_env_model_table_all_tau")
        positive_model_focus_table_df = load_artifact_table(f"positive_model_table_{focus_tau_stem}")
        negative_model_focus_table_df = load_artifact_table(f"negative_model_table_{focus_tau_stem}")
        positive_env_model_focus_table_df = load_artifact_table(f"positive_env_model_table_{focus_tau_stem}")
        negative_env_model_focus_table_df = load_artifact_table(f"negative_env_model_table_{focus_tau_stem}")

        if positive_model_focus_table_df.empty and not positive_model_table_df.empty:
            positive_model_focus_table_df = positive_model_table_df.loc[
                positive_model_table_df["Threshold"].astype(float).eq(args.default_tau)
            ].reset_index(drop=True)
        if negative_model_focus_table_df.empty and not negative_model_table_df.empty:
            negative_model_focus_table_df = negative_model_table_df.loc[
                negative_model_table_df["Threshold"].astype(float).eq(args.default_tau)
            ].reset_index(drop=True)
        if positive_env_model_focus_table_df.empty and not positive_env_model_table_df.empty:
            positive_env_model_focus_table_df = positive_env_model_table_df.loc[
                positive_env_model_table_df["Threshold"].astype(float).eq(args.default_tau)
            ].reset_index(drop=True)
        if negative_env_model_focus_table_df.empty and not negative_env_model_table_df.empty:
            negative_env_model_focus_table_df = negative_env_model_table_df.loc[
                negative_env_model_table_df["Threshold"].astype(float).eq(args.default_tau)
            ].reset_index(drop=True)

        load_summary_message = (
            f"Loaded saved threshold tables from {output_dir}. "
            "The expensive raw localization JSON scan was skipped for the table sections."
        )
    else:
        inventory_df, prefix_df, parse_error_df, pair_df, valid_pair_df = ensure_raw_threshold_inputs_loaded()
        inventory_table_df, non_exact_json_count_df, parse_error_table_df, load_summary_message = build_inventory_tables(
            inventory_df,
            parse_error_df,
            pair_df,
            valid_pair_df,
            prefix_df,
        )
        summary_payload = build_summary_payload(pair_df)
        positive_overall_summary_df = summary_payload["positive_overall_summary_df"]
        negative_overall_summary_df = summary_payload["negative_overall_summary_df"]
        positive_model_summary_df = summary_payload["positive_model_summary_df"]
        negative_model_summary_df = summary_payload["negative_model_summary_df"]
        positive_env_model_summary_df = summary_payload["positive_env_model_summary_df"]
        negative_env_model_summary_df = summary_payload["negative_env_model_summary_df"]
        positive_overall_table_df = summary_payload["positive_overall_table_df"]
        negative_overall_table_df = summary_payload["negative_overall_table_df"]
        positive_model_table_df = summary_payload["positive_model_table_df"]
        negative_model_table_df = summary_payload["negative_model_table_df"]
        positive_env_model_table_df = summary_payload["positive_env_model_table_df"]
        negative_env_model_table_df = summary_payload["negative_env_model_table_df"]
        positive_model_focus_table_df = summary_payload["positive_model_focus_table_df"]
        negative_model_focus_table_df = summary_payload["negative_model_focus_table_df"]
        positive_env_model_focus_table_df = summary_payload["positive_env_model_focus_table_df"]
        negative_env_model_focus_table_df = summary_payload["negative_env_model_focus_table_df"]

        maybe_save_table(inventory_table_df, "inventory")
        maybe_save_table(non_exact_json_count_df, "non_exact_json_counts")
        maybe_save_table(parse_error_table_df, "parse_warnings")
        maybe_save_summary_payload(summary_payload)

    bucket_tables_loaded_from_artifacts = False
    bucket_message = ""
    positive_bucket_overall_df = pd.DataFrame()
    negative_bucket_overall_df = pd.DataFrame()
    positive_bucket_by_model_df = pd.DataFrame()
    negative_bucket_by_model_df = pd.DataFrame()

    if args.load_artifacts_if_available and artifact_group_exists(required_bucket_artifact_stems):
        bucket_tables_loaded_from_artifacts = True
        positive_bucket_overall_df = load_artifact_table("positive_delta_bucket_overall")
        negative_bucket_overall_df = load_artifact_table("negative_delta_bucket_overall")
        positive_bucket_by_model_df = load_artifact_table("positive_delta_bucket_by_model")
        negative_bucket_by_model_df = load_artifact_table("negative_delta_bucket_by_model")
        bucket_message = f"Loaded saved delta-bucket summaries from {output_dir}."
    else:
        if pair_df.empty and pair_artifact_path.exists():
            valid_pair_df = pd.read_parquet(pair_artifact_path)
            pair_df = valid_pair_df.copy()
            bucket_message = (
                f"Loaded {pair_artifact_path.name} and rebuilt the delta-bucket summaries "
                "without re-scanning raw JSON files."
            )
        elif pair_df.empty:
            inventory_df, prefix_df, parse_error_df, pair_df, valid_pair_df = ensure_raw_threshold_inputs_loaded()
            bucket_message = "Computed delta-bucket summaries from the raw localization JSON files because no cached bucket artifacts were present."
        else:
            bucket_message = "Reused in-memory pair_df to build the delta-bucket summaries."

        bucket_payload = build_bucket_payload(pair_df)
        positive_bucket_overall_df = bucket_payload["positive_bucket_overall_df"]
        negative_bucket_overall_df = bucket_payload["negative_bucket_overall_df"]
        positive_bucket_by_model_df = bucket_payload["positive_bucket_by_model_df"]
        negative_bucket_by_model_df = bucket_payload["negative_bucket_by_model_df"]
        maybe_save_bucket_payload(bucket_payload)

    figure_path_png = output_dir / "delta_bucket_histogram.png"
    figure_path_pdf = output_dir / "delta_bucket_histogram.pdf"
    bucket_histogram_fig = plot_bucket_histograms(positive_bucket_overall_df, negative_bucket_overall_df)
    bucket_histogram_fig.savefig(figure_path_png, bbox_inches="tight")
    bucket_histogram_fig.savefig(figure_path_pdf, bbox_inches="tight")
    plt.close(bucket_histogram_fig)

    metadata_path = write_json(
        {
            "completed_at_utc": utc_now_iso(),
            "script": str(Path(__file__).resolve()),
            "repo_root": repo_root,
            "datasetmain_root": datasetmain_root,
            "output_dir": output_dir,
            "pair_artifact_path": pair_artifact_path,
            "tables_loaded_from_artifacts": tables_loaded_from_artifacts,
            "bucket_tables_loaded_from_artifacts": bucket_tables_loaded_from_artifacts,
            "load_summary_message": load_summary_message,
            "bucket_message": bucket_message,
            "min_valid": args.min_valid,
            "preferred_source": args.preferred_source,
            "default_tau": args.default_tau,
            "focus_tau_stem": focus_tau_stem,
            "max_json_files_per_bundle": args.max_json_files_per_bundle,
            "show_progress": args.show_progress,
            "progress_level": args.progress_level,
            "load_artifacts_if_available": args.load_artifacts_if_available,
            "save_artifacts": args.save_artifacts,
            "save_pair_artifacts": args.save_pair_artifacts,
            "inventory_rows": int(len(inventory_table_df)),
            "positive_overall_rows": int(len(positive_overall_table_df)),
            "negative_overall_rows": int(len(negative_overall_table_df)),
            "positive_bucket_rows": int(len(positive_bucket_overall_df)),
            "negative_bucket_rows": int(len(negative_bucket_overall_df)),
            "figure_png": figure_path_png,
            "figure_pdf": figure_path_pdf,
        },
        output_dir / "run_metadata.json",
    )

    print("Commitment juncture threshold run complete.")
    print(load_summary_message)
    print(bucket_message)
    print(f"Dataset root: {datasetmain_root}")
    print(f"Output dir: {output_dir}")
    print(f"Inventory rows: {len(inventory_table_df):,}")
    print(f"Positive overall rows: {len(positive_overall_table_df):,}")
    print(f"Negative overall rows: {len(negative_overall_table_df):,}")
    print(f"Histogram PNG: {figure_path_png}")
    print(f"Histogram PDF: {figure_path_pdf}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
