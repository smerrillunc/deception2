#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_main3_env_ood_metrics as env_ood


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate sharded Main3 environment-OOD outputs into one notebook-friendly results directory."
        )
    )
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--shards-root", type=str, default="")
    parser.add_argument("--feature-sizes", type=str, default="")
    parser.add_argument("--top-features-to-show", type=int, default=20)
    return parser.parse_args()


def read_csv_if_present(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def concat_frames(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [read_csv_if_present(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def dedupe_frame(df: pd.DataFrame, *, subset: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    if subset is None:
        return df.drop_duplicates().reset_index(drop=True)
    available_subset = [column for column in subset if column in df.columns]
    if not available_subset:
        return df.drop_duplicates().reset_index(drop=True)
    return df.drop_duplicates(subset=available_subset, keep="first").reset_index(drop=True)


def resolve_feature_sizes(args: argparse.Namespace, config_df: pd.DataFrame) -> list[int]:
    if str(args.feature_sizes).strip():
        return env_ood.parse_int_csv(args.feature_sizes)
    if not config_df.empty and {"setting", "value"}.issubset(config_df.columns):
        feature_size_rows = config_df.loc[config_df["setting"].eq("feature_sizes"), "value"]
        if not feature_size_rows.empty and str(feature_size_rows.iloc[0]).strip():
            return env_ood.parse_int_csv(str(feature_size_rows.iloc[0]))
    return list(env_ood.DEFAULT_FEATURE_SIZES)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    shards_root = (
        Path(args.shards_root).expanduser().resolve()
        if str(args.shards_root).strip()
        else (output_root / "shards").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    metric_paths = sorted(shards_root.rglob("all_transfer_metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No shard metrics found under {shards_root}")
    shard_dirs = [path.parent for path in metric_paths]

    shard_manifest_df = pd.DataFrame(
        [{"shard_dir": str(shard_dir)} for shard_dir in shard_dirs]
    )
    shard_manifest_df.to_csv(output_root / "shard_manifest.csv", index=False)

    config_df = dedupe_frame(
        concat_frames([shard_dir / "config.csv" for shard_dir in shard_dirs]),
        subset=["setting", "value"],
    )
    config_df = pd.concat(
        [
            config_df,
            pd.DataFrame(
                [
                    {"setting": "aggregate_output_root", "value": str(output_root)},
                    {"setting": "aggregate_shards_root", "value": str(shards_root)},
                    {"setting": "aggregate_shard_count", "value": len(shard_dirs)},
                ]
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["setting", "value"], keep="first")
    config_df.to_csv(output_root / "config.csv", index=False)

    bundle_inventory_df = dedupe_frame(
        concat_frames([shard_dir / "bundle_inventory.csv" for shard_dir in shard_dirs]),
        subset=["bundle_key"],
    )
    bundle_inventory_df.to_csv(output_root / "bundle_inventory.csv", index=False)

    split_summary_df = dedupe_frame(
        concat_frames([shard_dir / "split_summary.csv" for shard_dir in shard_dirs]),
        subset=["bundle_key"],
    )
    split_summary_df.to_csv(output_root / "split_summary.csv", index=False)

    feature_space_catalog_df = dedupe_frame(
        concat_frames([shard_dir / "feature_space_catalog.csv" for shard_dir in shard_dirs]),
        subset=["feature_space"],
    )
    feature_space_catalog_df.to_csv(output_root / "feature_space_catalog.csv", index=False)

    attention_lookup_df = dedupe_frame(
        concat_frames([shard_dir / "attention_reduction_lookup.csv" for shard_dir in shard_dirs]),
        subset=["feature"],
    )
    if not attention_lookup_df.empty:
        attention_lookup_df.to_csv(output_root / "attention_reduction_lookup.csv", index=False)

    all_transfer_metrics_df = concat_frames(metric_paths)
    all_model_selection_df = concat_frames([shard_dir / "all_model_selection.csv" for shard_dir in shard_dirs])
    all_coefficients_df = concat_frames([shard_dir / "all_coefficients.csv" for shard_dir in shard_dirs])
    all_calibration_df = concat_frames([shard_dir / "all_calibration_curves.csv" for shard_dir in shard_dirs])
    all_fpr_df = concat_frames([shard_dir / "all_fpr_at_recall.csv" for shard_dir in shard_dirs])

    all_transfer_metrics_df.to_csv(output_root / "all_transfer_metrics.csv", index=False)
    all_model_selection_df.to_csv(output_root / "all_model_selection.csv", index=False)
    all_coefficients_df.to_csv(output_root / "all_coefficients.csv", index=False)
    all_calibration_df.to_csv(output_root / "all_calibration_curves.csv", index=False)
    all_fpr_df.to_csv(output_root / "all_fpr_at_recall.csv", index=False)

    if all_transfer_metrics_df.empty:
        raise RuntimeError("Aggregated shard outputs produced no transfer metrics.")

    selected_scenarios = [
        scenario_name
        for scenario_name in env_ood.SCENARIO_TITLES
        if scenario_name in set(all_transfer_metrics_df["scenario_name"].dropna().astype(str).tolist())
    ]
    _, train_axis_labels_by_scenario = env_ood.build_experiment_run_specs(selected_scenarios)
    feature_sizes = resolve_feature_sizes(args, config_df)

    transfer_summary_df = env_ood.summarize_transfer_metrics_env(all_transfer_metrics_df)
    train_env_model_summary_df = env_ood.summarize_train_env_models(all_transfer_metrics_df)
    target_env_breakdown_df = env_ood.summarize_target_env_breakdown(all_transfer_metrics_df)
    confusion_summary_df = env_ood.summarize_confusion_counts_env(all_transfer_metrics_df)
    family_panel_selection_df = env_ood.build_family_panel_selection_env(transfer_summary_df, feature_sizes)
    best_family_models_df = env_ood.build_best_family_models_env(
        train_env_model_summary_df,
        family_panel_selection_df,
    )
    selected_panel_confusion_val_df = env_ood.build_selected_panel_confusion_summary_env(
        all_transfer_metrics_df,
        family_panel_selection_df,
        eval_role="val",
    )
    selected_panel_confusion_ood_df = env_ood.build_selected_panel_confusion_summary_env(
        all_transfer_metrics_df,
        family_panel_selection_df,
        eval_role="ood",
    )
    selected_panel_confusion_df = (
        pd.concat(
            [selected_panel_confusion_val_df, selected_panel_confusion_ood_df],
            ignore_index=True,
        )
        if not selected_panel_confusion_val_df.empty or not selected_panel_confusion_ood_df.empty
        else pd.DataFrame()
    )

    transfer_summary_df.to_csv(output_root / "transfer_summary.csv", index=False)
    train_env_model_summary_df.to_csv(output_root / "train_env_model_summary.csv", index=False)
    target_env_breakdown_df.to_csv(output_root / "target_env_breakdown_summary.csv", index=False)
    confusion_summary_df.to_csv(output_root / "confusion_summary.csv", index=False)
    family_panel_selection_df.to_csv(output_root / "best_feature_space_by_target_size_family.csv", index=False)
    best_family_models_df.to_csv(output_root / "best_model_by_target_size_family.csv", index=False)
    selected_panel_confusion_val_df.to_csv(output_root / "selected_panel_confusion_val.csv", index=False)
    selected_panel_confusion_ood_df.to_csv(output_root / "selected_panel_confusion_ood.csv", index=False)
    selected_panel_confusion_df.to_csv(output_root / "selected_panel_confusion_all.csv", index=False)

    env_ood.export_selected_family_panel_tables_env(
        output_root,
        all_transfer_metrics_df,
        family_panel_selection_df,
        selected_panel_confusion_df,
        train_axis_labels_by_scenario,
    )

    top_feature_tables: list[pd.DataFrame] = []
    top_feature_dir = env_ood.ensure_dir(output_root / "best_features")
    for row in best_family_models_df.itertuples(index=False):
        feature_table = (
            all_coefficients_df.loc[
                all_coefficients_df["target_name"].eq(row.target_name)
                & all_coefficients_df["feature_space"].eq(row.feature_space)
                & all_coefficients_df["feature_size_label"].eq(row.feature_size_label)
                & all_coefficients_df["train_env"].eq(row.train_env)
            ]
            .sort_values(["abs_coefficient", "feature"], ascending=[False, True], na_position="last")
            .reset_index(drop=True)
        )
        if feature_table.empty:
            continue
        feature_table.insert(0, "importance_rank", range(1, len(feature_table) + 1))
        top_feature_df = feature_table.head(int(args.top_features_to_show)).copy()
        top_feature_tables.append(top_feature_df)
        scenario_dir = env_ood.ensure_dir(top_feature_dir / str(row.scenario_name))
        out_name = f"{row.target_name}__{row.feature_family_group}__{row.requested_feature_size_label}__top_features.csv"
        top_feature_df.to_csv(scenario_dir / out_name, index=False)
    top_features_for_best_models_df = (
        pd.concat(top_feature_tables, ignore_index=True) if top_feature_tables else pd.DataFrame()
    )
    top_features_for_best_models_df.to_csv(output_root / "top_features_for_best_models.csv", index=False)

    print(f"Aggregated shards root: {shards_root}")
    print(f"Output root: {output_root}")
    print(f"Shard count: {len(shard_dirs)}")
    print(f"Transfer rows: {len(all_transfer_metrics_df):,}")
    print(f"Model selections: {len(all_model_selection_df):,}")
    print(f"Calibration rows: {len(all_calibration_df):,}")
    print(f"FPR rows: {len(all_fpr_df):,}")


if __name__ == "__main__":
    main()
