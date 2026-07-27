#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


THIS_FILE = Path(__file__).resolve()
NOTEBOOK_DIR = THIS_FILE.parent
REBUTTAL_ROOT = NOTEBOOK_DIR.parent
DEFAULT_RESULTS_ROOT = REBUTTAL_ROOT / "results"
DEFAULT_RUN_NAME = "localization_fulltrace_vs_adaptive_rebuttal_v1"
DEFAULT_NOTEBOOK_PATH = NOTEBOOK_DIR / "localization_fulltrace_rebuttal_analysis.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the localization dataset-adaptive-vs-full rebuttal notebook."
    )
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_NOTEBOOK_PATH))
    return parser.parse_args()


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
            # Localization Dataset-Adaptive-vs-Full Rebuttal Analysis

            This notebook is for the small matched localization comparison:

            - 10 selected examples per model x environment bundle
            - dataset adaptive localization used as the reference traces
            - the same subset rerun only with exhaustive `full` localization
            - quantitative summaries of:
              - dataset-adaptive probe coverage
              - dataset-adaptive-vs-full peak and boundary agreement
              - model-level summaries in addition to model x environment summaries
              - prevalence of gradual and multi-peak exhaustive traces
              - case-study curves

            If the analysis CSVs are missing or stale, run the refresh cell below.
            """
        ),
        code(
            f"""
            from __future__ import annotations

            import json
            import subprocess
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from pandas.errors import EmptyDataError
            from IPython.display import Markdown, display

            NOTEBOOK_CWD = Path.cwd().resolve()
            SEARCH_ROOTS = [NOTEBOOK_CWD, *NOTEBOOK_CWD.parents]
            REPO_ROOT = next(
                (
                    root
                    for root in SEARCH_ROOTS
                    if (root / "rebuttal").exists() and (root / "src" / "sentence_localization_batch.py").exists()
                ),
                NOTEBOOK_CWD,
            )
            RESULTS_ROOT = Path({str(Path(args.results_root).expanduser().resolve())!r})
            if not RESULTS_ROOT.exists():
                RESULTS_ROOT = REPO_ROOT / "rebuttal" / "results"
            RUN_NAME = {args.run_name!r}
            RUN_ROOT = RESULTS_ROOT / RUN_NAME
            ANALYSIS_ROOT = RUN_ROOT / "analysis"
            FIGURES_ROOT = ANALYSIS_ROOT / "figures"
            ANALYSIS_SCRIPT = REPO_ROOT / "rebuttal" / "scripts" / "analyze_localization_fulltrace_rebuttal.py"

            pd.options.display.max_columns = 200
            pd.options.display.max_colwidth = 220
            pd.options.display.width = 240

            print("REPO_ROOT:", REPO_ROOT)
            print("RUN_ROOT:", RUN_ROOT)
            print("ANALYSIS_ROOT:", ANALYSIS_ROOT)
            """
        ),
        code(
            """
            def md(text: str) -> None:
                display(Markdown(text))


            def read_json(path: Path, default=None):
                if not path.exists():
                    return default
                return json.loads(path.read_text(encoding="utf-8"))


            def read_csv(path: Path) -> pd.DataFrame:
                if not path.exists() or path.stat().st_size == 0:
                    return pd.DataFrame()
                try:
                    return pd.read_csv(path)
                except EmptyDataError:
                    return pd.DataFrame()


            def ordered_commitment_threshold_rate_columns(df: pd.DataFrame, metric: str) -> list[str]:
                prefix = f"commitment_sentence_{metric}_rate_tau_"
                ordered: list[tuple[float, str]] = []
                for column in df.columns:
                    if not str(column).startswith(prefix):
                        continue
                    suffix = str(column)[len(prefix):]
                    try:
                        tau = float(suffix.replace("neg_", "-").replace("_", "."))
                    except ValueError:
                        continue
                    ordered.append((tau, str(column)))
                ordered.sort(key=lambda item: item[0])
                return [column for _, column in ordered]


            def refresh_analysis() -> subprocess.CompletedProcess:
                cmd = [
                    sys.executable,
                    str(ANALYSIS_SCRIPT),
                    "--run-name",
                    RUN_NAME,
                    "--results-root",
                    str(RESULTS_ROOT),
                ]
                print("Running:", " ".join(cmd))
                completed = subprocess.run(cmd, text=True, capture_output=True)
                print(completed.stdout)
                if completed.returncode != 0:
                    print(completed.stderr)
                return completed


            completion_summary = read_json(ANALYSIS_ROOT / "completion_summary.json", default={}) or {}
            selection_df = read_csv(RUN_ROOT / "selected_examples.csv")
            bundle_selection_df = read_csv(RUN_ROOT / "bundle_summary.csv")
            bundle_completion_df = read_csv(ANALYSIS_ROOT / "bundle_completion_summary.csv")
            model_completion_df = read_csv(ANALYSIS_ROOT / "model_completion_summary.csv")
            per_example_df = read_csv(ANALYSIS_ROOT / "per_example_metrics.csv")
            curve_points_df = read_csv(ANALYSIS_ROOT / "curve_points.csv")
            adaptive_bundle_df = read_csv(ANALYSIS_ROOT / "adaptive_vs_full_summary_by_bundle.csv")
            adaptive_model_df = read_csv(ANALYSIS_ROOT / "adaptive_vs_full_summary_by_model.csv")
            adaptive_overall_df = read_csv(ANALYSIS_ROOT / "adaptive_vs_full_summary_overall.csv")
            trace_shape_df = read_csv(ANALYSIS_ROOT / "trace_shape_prevalence_by_bundle.csv")
            trace_shape_model_df = read_csv(ANALYSIS_ROOT / "trace_shape_prevalence_by_model.csv")
            case_studies_df = read_csv(ANALYSIS_ROOT / "case_studies.csv")
            print(
                "Analysis status:",
                {
                    "expected_examples": completion_summary.get("expected_examples"),
                    "completed_dataset_adaptive_examples": completion_summary.get("completed_dataset_adaptive_examples"),
                    "completed_full_examples": completion_summary.get("completed_full_examples"),
                    "completed_paired_examples": completion_summary.get("completed_paired_examples"),
                },
            )
            """
        ),
        md("## Refresh Analysis"),
        code(
            """
            refresh = False
            if refresh:
                refresh_analysis()
            else:
                print("Set refresh = True and rerun this cell to recompute the analysis outputs.")
            """
        ),
        md("## Selection Inventory"),
        code(
            """
            md(
                f"- Selected examples: **{len(selection_df)}**\\n"
                f"- Bundles: **{bundle_selection_df.shape[0]}**"
            )
            bundle_selection_df
            """
        ),
        md("## Completion"),
        code(
            """
            completion_summary
            """
        ),
        code(
            """
            bundle_completion_df
            """
        ),
        md("## Trace Definitions"),
        code(
            """
            {
                "multi_peak_definition": completion_summary.get("multi_peak_definition", {}),
                "commitment_metric_note": completion_summary.get("commitment_metric_note", ""),
                "commitment_delta_thresholds": completion_summary.get("commitment_delta_thresholds", []),
                "peak_metric_note": completion_summary.get("peak_metric_note", ""),
                "gradual_definition": completion_summary.get("gradual_definition", {}),
                "shape_thresholds": completion_summary.get("shape_thresholds", {}),
            }
            """
        ),
        md("## Model Completion"),
        code(
            """
            model_completion_df
            """
        ),
        md("## Overall Adaptive-vs-Full Summary"),
        code(
            """
            adaptive_overall_df
            """
        ),
        code(
            """
            exact_tau_columns = ordered_commitment_threshold_rate_columns(adaptive_overall_df, "exact")
            within_one_tau_columns = ordered_commitment_threshold_rate_columns(adaptive_overall_df, "within_one")
            commitment_columns = [
                column
                for column in [
                    "num_examples",
                    *exact_tau_columns,
                    *within_one_tau_columns,
                    "commitment_text_normalized_match_rate",
                ]
                if column in adaptive_overall_df.columns
            ]
            if commitment_columns:
                adaptive_overall_df.loc[:, commitment_columns]
            else:
                print("No commitment-agreement columns available yet.")
            """
        ),
        md("## Model Summary"),
        code(
            """
            adaptive_model_df
            """
        ),
        code(
            """
            exact_tau_columns = ordered_commitment_threshold_rate_columns(adaptive_model_df, "exact")
            within_one_tau_columns = ordered_commitment_threshold_rate_columns(adaptive_model_df, "within_one")
            model_commitment_columns = [
                column
                for column in [
                    "model_display",
                    "num_examples",
                    *exact_tau_columns,
                    *within_one_tau_columns,
                    "commitment_text_normalized_match_rate",
                ]
                if column in adaptive_model_df.columns
            ]
            if model_commitment_columns:
                adaptive_model_df.loc[:, model_commitment_columns]
            else:
                print("No model-level commitment-agreement columns available yet.")
            """
        ),
        code(
            """
            if not adaptive_model_df.empty:
                plot_df = adaptive_model_df.copy()
                plot_df["model_label"] = plot_df["model_display"].astype(str)
                peak_column = (
                    "adaptive_any_peak_probe_within_one_rate"
                    if "adaptive_any_peak_probe_within_one_rate" in plot_df.columns
                    else "peak_within_one_rate"
                )
                peak_label = (
                    "Any peak probed within one"
                    if peak_column == "adaptive_any_peak_probe_within_one_rate"
                    else "Peak within one"
                )
                commitment_column = (
                    "commitment_sentence_exact_rate"
                    if "commitment_sentence_exact_rate" in plot_df.columns
                    else "commitment_sentence_within_one_rate"
                )
                commitment_label = (
                    "Commitment exact"
                    if commitment_column == "commitment_sentence_exact_rate"
                    else "Commitment within one"
                )
                x = np.arange(len(plot_df))
                width = 0.32
                fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
                ax.bar(
                    x - width / 2.0,
                    plot_df[peak_column],
                    width=width,
                    label=peak_label,
                    color="#3D6E70",
                )
                ax.bar(
                    x + width / 2.0,
                    plot_df[commitment_column],
                    width=width,
                    label=commitment_label,
                    color="#C9774D",
                )
                ax.set_xticks(x)
                ax.set_xticklabels(plot_df["model_label"], rotation=20)
                ax.set_ylim(0.0, 1.02)
                ax.set_ylabel("Agreement rate")
                ax.set_title("Dataset adaptive vs exhaustive agreement by model")
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend()
                plt.show()
            else:
                print("No paired adaptive/full results available yet.")
            """
        ),
        md("## Bundle Summary"),
        code(
            """
            adaptive_bundle_df
            """
        ),
        code(
            """
            exact_tau_columns = ordered_commitment_threshold_rate_columns(adaptive_bundle_df, "exact")
            within_one_tau_columns = ordered_commitment_threshold_rate_columns(adaptive_bundle_df, "within_one")
            bundle_commitment_columns = [
                column
                for column in [
                    "env_display",
                    "model_display",
                    "num_examples",
                    *exact_tau_columns,
                    *within_one_tau_columns,
                    "commitment_text_normalized_match_rate",
                ]
                if column in adaptive_bundle_df.columns
            ]
            if bundle_commitment_columns:
                adaptive_bundle_df.loc[:, bundle_commitment_columns]
            else:
                print("No bundle-level commitment-agreement columns available yet.")
            """
        ),
        code(
            """
            if not adaptive_bundle_df.empty:
                plot_df = adaptive_bundle_df.copy()
                plot_df["bundle_label"] = plot_df["env_display"].astype(str) + "\\n" + plot_df["model_display"].astype(str)
                peak_column = (
                    "adaptive_any_peak_probe_within_one_rate"
                    if "adaptive_any_peak_probe_within_one_rate" in plot_df.columns
                    else "peak_within_one_rate"
                )
                peak_label = (
                    "Any peak probed within one"
                    if peak_column == "adaptive_any_peak_probe_within_one_rate"
                    else "Peak within one"
                )
                commitment_column = (
                    "commitment_sentence_exact_rate"
                    if "commitment_sentence_exact_rate" in plot_df.columns
                    else "commitment_sentence_within_one_rate"
                )
                commitment_label = (
                    "Commitment exact"
                    if commitment_column == "commitment_sentence_exact_rate"
                    else "Commitment within one"
                )
                x = np.arange(len(plot_df))
                width = 0.32
                fig, ax = plt.subplots(figsize=(14.5, 5.4), constrained_layout=True)
                ax.bar(
                    x - width / 2.0,
                    plot_df[peak_column],
                    width=width,
                    label=peak_label,
                    color="#3D6E70",
                )
                ax.bar(
                    x + width / 2.0,
                    plot_df[commitment_column],
                    width=width,
                    label=commitment_label,
                    color="#C9774D",
                )
                ax.set_xticks(x)
                ax.set_xticklabels(plot_df["bundle_label"], rotation=60)
                ax.set_ylim(0.0, 1.02)
                ax.set_ylabel("Agreement rate")
                ax.set_title("Dataset adaptive vs exhaustive agreement by bundle")
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend()
                plt.show()
            else:
                print("No paired adaptive/full results available yet.")
            """
        ),
        md("## Trace Shapes"),
        code(
            """
            trace_shape_model_df
            """
        ),
        code(
            """
            if not trace_shape_model_df.empty:
                pivot = (
                    trace_shape_model_df.pivot_table(
                        index=["model_display"],
                        columns="trace_shape_label",
                        values="fraction",
                        fill_value=0.0,
                    )
                    .reset_index()
                )
                model_labels = pivot["model_display"].astype(str)
                categories = ["multi_peak", "gradual", "sharp_or_other"]
                colors = {
                    "multi_peak": "#2B6CB0",
                    "gradual": "#38A169",
                    "sharp_or_other": "#D69E2E",
                }
                bottoms = np.zeros(len(pivot), dtype=float)
                fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
                for category in categories:
                    values = pivot[category].to_numpy(dtype=float) if category in pivot.columns else np.zeros(len(pivot), dtype=float)
                    ax.bar(model_labels, values, bottom=bottoms, color=colors[category], label=category.replace("_", " "))
                    bottoms += values
                ax.set_ylabel("Fraction of exhaustive traces")
                ax.set_ylim(0.0, 1.02)
                ax.set_title("Gradual and multi-peak trace prevalence by model")
                ax.tick_params(axis="x", rotation=20)
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend()
                plt.show()
            else:
                print("No exhaustive trace-shape outputs available yet.")
            """
        ),
        code(
            """
            trace_shape_df
            """
        ),
        code(
            """
            if not trace_shape_df.empty:
                pivot = (
                    trace_shape_df.pivot_table(
                        index=["env_display", "model_display"],
                        columns="trace_shape_label",
                        values="fraction",
                        fill_value=0.0,
                    )
                    .reset_index()
                )
                bundle_labels = pivot["env_display"].astype(str) + "\\n" + pivot["model_display"].astype(str)
                categories = ["multi_peak", "gradual", "sharp_or_other"]
                colors = {
                    "multi_peak": "#2B6CB0",
                    "gradual": "#38A169",
                    "sharp_or_other": "#D69E2E",
                }
                bottoms = np.zeros(len(pivot), dtype=float)
                fig, ax = plt.subplots(figsize=(14.5, 5.7), constrained_layout=True)
                for category in categories:
                    values = pivot[category].to_numpy(dtype=float) if category in pivot.columns else np.zeros(len(pivot), dtype=float)
                    ax.bar(bundle_labels, values, bottom=bottoms, color=colors[category], label=category.replace("_", " "))
                    bottoms += values
                ax.set_ylabel("Fraction of exhaustive traces")
                ax.set_ylim(0.0, 1.02)
                ax.set_title("Gradual and multi-peak trace prevalence by bundle")
                ax.tick_params(axis="x", rotation=60)
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend()
                plt.show()
            else:
                print("No exhaustive trace-shape outputs available yet.")
            """
        ),
        md("## Example-Level Metrics"),
        code(
            """
            per_example_df.head(20)
            """
        ),
        md("## Case Studies"),
        code(
            """
            case_studies_df
            """
        ),
        code(
            """
            def plot_example(bundle_key: str, example_id: str) -> None:
                subset = curve_points_df.loc[
                    curve_points_df["bundle_key"].astype(str).eq(str(bundle_key))
                    & curve_points_df["example_id"].astype(str).eq(str(example_id))
                ].copy()
                if subset.empty:
                    print(f"No curve rows found for {bundle_key} / {example_id}")
                    return
                full_df = subset.loc[subset["method"].astype(str).eq("full")].sort_values("sentence_idx")
                adaptive_df = subset.loc[subset["method"].astype(str).eq("adaptive")].sort_values("sentence_idx")
                metrics_row = per_example_df.loc[
                    per_example_df["bundle_key"].astype(str).eq(str(bundle_key))
                    & per_example_df["example_id"].astype(str).eq(str(example_id))
                ]
                if metrics_row.empty:
                    print(f"No metric row found for {bundle_key} / {example_id}")
                    return
                metric = metrics_row.iloc[0]
                fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
                ax.plot(full_df["sentence_number"], full_df["deception_rate"], marker="o", linewidth=2.4, color="black", label="Full")
                ax.scatter(adaptive_df["sentence_number"], adaptive_df["deception_rate"], s=70, color="#C05621", label="Dataset adaptive probes", zorder=4)

                peak_string = str(metric.get("full_prominent_peak_sentence_indices") or "")
                for peak_text in [value for value in peak_string.split(",") if value.strip()]:
                    ax.axvline(int(peak_text), color="#3182CE", linewidth=1.1, alpha=0.35)

                if pd.notna(metric.get("adaptive_right_sentence_end_idx")):
                    ax.axvline(int(metric["adaptive_right_sentence_end_idx"]), color="#DD6B20", linestyle="--", linewidth=1.4, alpha=0.75, label="Dataset adaptive right boundary")
                if pd.notna(metric.get("full_boundary_sentence_end_idx")):
                    ax.axvline(int(metric["full_boundary_sentence_end_idx"]), color="#2F855A", linestyle=":", linewidth=1.6, alpha=0.85, label="Full boundary")

                ax.set_title(
                    f"{metric['env_display']} / {metric['model_display']}\\n"
                    f"{metric['example_id']} | shape={metric['trace_shape_label']}"
                )
                ax.set_xlabel("Sentence index")
                ax.set_ylabel("Deception rate")
                ax.set_xticks(full_df["sentence_number"])
                ax.set_ylim(-0.02, 1.02)
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best")
                plt.show()


            if not case_studies_df.empty:
                first_case = case_studies_df.iloc[0]
                plot_example(first_case["bundle_key"], first_case["example_id"])
            else:
                print("No case studies available yet.")
            """
        ),
        md("## Saved Figures"),
        code(
            """
            if FIGURES_ROOT.exists():
                sorted(path.name for path in FIGURES_ROOT.glob("*.png"))
            else:
                print("No figures directory found yet.")
            """
        ),
    ]

    nbf.write(nb, output_path)
    print(f"Wrote notebook to: {output_path}")


if __name__ == "__main__":
    main()
