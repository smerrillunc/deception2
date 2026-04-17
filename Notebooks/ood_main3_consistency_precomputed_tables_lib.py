from __future__ import annotations

import math
import os
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NOTEBOOK_ROOT = Path("/playpen-ssd/smerrill/deception2/Notebooks")
DEFAULT_RESULTS_COLLECTION_ROOT = Path("/playpen-ssd/smerrill/deception2/Results/OOD_Modeling_main3_consistency")
DEFAULT_OUTPUT_BUNDLE_GLOB = "OOD_Modeling_main3_consistency_ablation_outputs__*"

ENV_ORDER = ["AdvisorAudit", "BS", "CarSales", "Gridworld", "Interview"]
TARGET_ORDER = ["delta_pos_gt_0_3", "delta_neg_lt_neg_0_3", "delta_abs_gt_0_3"]
MAIN_PAPER_TARGET_ORDER = ["delta_pos_gt_0_3", "delta_neg_lt_neg_0_3"]
MODEL_ORDER = ["GPT-OSS-20B", "Llama-8B", "Qwen-7B", "Qwen-14B"]

TARGET_TITLE_OVERRIDES = {
    "delta_pos_gt_0_3": "delta_deception_rate > 0.3",
    "delta_neg_lt_neg_0_3": "delta_deception_rate < -0.3",
    "delta_abs_gt_0_3": "|delta_deception_rate| > 0.3",
}
MODEL_NAME_OVERRIDES = {
    "gptoss20b": "GPT-OSS-20B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "llama8b": "Llama-8B",
    "deepseek-r1-distill-llama-8b": "Llama-8B",
    "qwen7b": "Qwen-7B",
    "deepseek-r1-distill-qwen-7b": "Qwen-7B",
    "qwen14b": "Qwen-14B",
    "deepseek-r1-distill-qwen-14b": "Qwen-14B",
}

CORE_FEATURE_ORDER = [
    "Baseline (Activation only: raw)",
    "Activation only: PCA final",
    "Activation only: PCA final - prev",
    "Activation only: PCA final - mean(prev 4)",
    "Attention only",
    "Attention + PCA final",
    "Attention + PCA final - prev",
    "Attention + PCA final - mean(prev 4)",
]
ATTENTION_SUBSET_FEATURE_ORDER = [
    "Attention only",
    "Attention only: grounding",
    "Attention only: concentration",
    "Attention only: grounding transition",
    "Attention only: concentration transition",
]

FEATURE_SET_ALIASES = {
    "baseline_raw_final": "Baseline (Activation only: raw)",
    "baseline: raw final": "Baseline (Activation only: raw)",
    "baseline (activation only: raw)": "Baseline (Activation only: raw)",
    "activation_raw": "Baseline (Activation only: raw)",
    "activation raw": "Baseline (Activation only: raw)",
    "activation: raw": "Baseline (Activation only: raw)",
    "activation only: raw": "Baseline (Activation only: raw)",
    "activation only: raw final": "Baseline (Activation only: raw)",
    "activation_raw_final": "Baseline (Activation only: raw)",
    "activation_pca_final": "Activation only: PCA final",
    "activation only: pca final": "Activation only: PCA final",
    "activation_pca_delta_last2": "Activation only: PCA final - prev",
    "activation only: pca final - previous": "Activation only: PCA final - prev",
    "activation only: pca final - prev": "Activation only: PCA final - prev",
    "activation_pca_delta_prev4mean": "Activation only: PCA final - mean(prev 4)",
    "activation only: pca final - mean(prev 4)": "Activation only: PCA final - mean(prev 4)",
    "attention_only": "Attention only",
    "attention only": "Attention only",
    "attention_grounding_only": "Attention only: grounding",
    "attention only: grounding": "Attention only: grounding",
    "attention_concentration_only": "Attention only: concentration",
    "attention only: concentration": "Attention only: concentration",
    "attention_grounding_transition_only": "Attention only: grounding transition",
    "attention only: grounding transition": "Attention only: grounding transition",
    "attention_concentration_transition_only": "Attention only: concentration transition",
    "attention only: concentration transition": "Attention only: concentration transition",
    "attention_plus_activation_raw_final": "Attention + raw final",
    "attention + raw final": "Attention + raw final",
    "attention_plus_activation_pca_final": "Attention + PCA final",
    "attention + pca final": "Attention + PCA final",
    "attention_plus_activation_pca_delta_last2": "Attention + PCA final - prev",
    "attention + pca final - previous": "Attention + PCA final - prev",
    "attention + pca final - prev": "Attention + PCA final - prev",
    "attention_plus_activation_pca_delta_prev4mean": "Attention + PCA final - mean(prev 4)",
    "attention + pca final - mean(prev 4)": "Attention + PCA final - mean(prev 4)",
}


def _clean_key(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def canonical_feature_set(feature_space: object, feature_space_title: object) -> str:
    for value in [feature_space, feature_space_title]:
        clean = _clean_key(value)
        if clean in FEATURE_SET_ALIASES:
            return FEATURE_SET_ALIASES[clean]
    if pd.notna(feature_space_title) and str(feature_space_title).strip():
        return str(feature_space_title).strip()
    return "" if pd.isna(feature_space) else str(feature_space).strip()


def canonical_target_title(target_name: object, target_title: object) -> str:
    if pd.notna(target_title) and str(target_title).strip():
        return str(target_title).strip()
    key = "" if pd.isna(target_name) else str(target_name).strip()
    return TARGET_TITLE_OVERRIDES.get(key, key)


def _slug_to_model_name(slug: str) -> str:
    clean = _clean_key(slug).replace(" ", "-")
    return MODEL_NAME_OVERRIDES.get(clean, slug)


def canonical_model_name(model_dirname: object, fallback_name: str) -> str:
    if pd.notna(model_dirname) and str(model_dirname).strip():
        clean = _clean_key(model_dirname).replace(" ", "-")
        if clean in MODEL_NAME_OVERRIDES:
            return MODEL_NAME_OVERRIDES[clean]
        for key, value in MODEL_NAME_OVERRIDES.items():
            if key in clean:
                return value
        return str(model_dirname).strip()
    for token in re.split(r"__|[_\-]", fallback_name):
        mapped = _slug_to_model_name(token)
        if mapped != token:
            return mapped
    return fallback_name


def _bundle_sort_key(path: Path) -> tuple[int, str]:
    return (0, path.name)


def _model_sort_key(model_name: str) -> tuple[int, str]:
    try:
        return (MODEL_ORDER.index(model_name), model_name)
    except ValueError:
        return (len(MODEL_ORDER), model_name)


def _target_sort_key(target_name: str) -> tuple[int, str]:
    try:
        return (TARGET_ORDER.index(target_name), target_name)
    except ValueError:
        return (len(TARGET_ORDER), target_name)


def _parse_multi_path_env(raw: str | None) -> list[Path]:
    if not raw:
        return []
    parts = [part.strip() for part in re.split(r"[,\n]|" + re.escape(os.pathsep), raw) if part.strip()]
    return [Path(part).expanduser().resolve() for part in parts]


def is_bundle_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        (path / name).exists()
        for name in ["all_transfer_metrics.csv", "transfer_summary.csv", "feature_run_summary.csv", "config.csv"]
    )


def discover_bundle_dirs() -> list[Path]:
    explicit_paths = _parse_multi_path_env(os.environ.get("OOD_MAIN3_PRECOMPUTED_BUNDLE_ROOTS"))
    candidates: list[Path] = []
    if explicit_paths:
        candidates.extend(explicit_paths)
    else:
        output_root_env = os.environ.get("OOD_MAIN3_PRECOMPUTED_OUTPUT_ROOT")
        if output_root_env:
            candidates.append(Path(output_root_env).expanduser().resolve())
        candidates.extend(sorted(NOTEBOOK_ROOT.glob(DEFAULT_OUTPUT_BUNDLE_GLOB), key=_bundle_sort_key))
        collection_root = Path(
            os.environ.get("OOD_MAIN3_COLLECTION_RESULTS_ROOT", str(DEFAULT_RESULTS_COLLECTION_ROOT))
        ).expanduser().resolve()
        if collection_root.exists():
            candidates.extend(sorted((path for path in collection_root.iterdir() if path.is_dir()), key=_bundle_sort_key))

    bundle_dirs: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if is_bundle_dir(candidate):
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                bundle_dirs.append(resolved)
            continue
        if candidate.is_dir():
            for child in sorted((path for path in candidate.iterdir() if path.is_dir()), key=_bundle_sort_key):
                if not is_bundle_dir(child):
                    continue
                resolved = child.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                bundle_dirs.append(resolved)
    return bundle_dirs


def maybe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _column_or_empty(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(dtype=object)


def read_config_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    config_df = pd.read_csv(path)
    if {"setting", "value"}.issubset(config_df.columns):
        return {str(row.setting): str(row.value) for row in config_df.itertuples(index=False)}
    return {}


def build_inventory_row(bundle_dir: Path, model_name: str, summary_df: pd.DataFrame, metrics_df: pd.DataFrame) -> dict[str, object]:
    feature_sets = sorted(
        set(summary_df.get("feature_set", pd.Series(dtype=str)).dropna().astype(str).tolist())
        | set(metrics_df.get("feature_set", pd.Series(dtype=str)).dropna().astype(str).tolist())
    )
    targets = sorted(
        set(summary_df.get("target_name", pd.Series(dtype=str)).dropna().astype(str).tolist())
        | set(metrics_df.get("target_name", pd.Series(dtype=str)).dropna().astype(str).tolist()),
        key=_target_sort_key,
    )
    return {
        "Bundle": bundle_dir.name,
        "Model": model_name,
        "Has transfer_summary": (bundle_dir / "transfer_summary.csv").exists(),
        "Has feature_run_summary": (bundle_dir / "feature_run_summary.csv").exists(),
        "Has all_transfer_metrics": (bundle_dir / "all_transfer_metrics.csv").exists(),
        "Targets": ", ".join(targets),
        "Feature Sets": ", ".join(feature_sets),
    }


def load_bundle_frames(bundle_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]] | None:
    config_map = read_config_map(bundle_dir / "config.csv")
    summary_df = maybe_read_csv(bundle_dir / "feature_run_summary.csv")
    if summary_df.empty:
        summary_df = maybe_read_csv(bundle_dir / "transfer_summary.csv")
    metrics_df = maybe_read_csv(bundle_dir / "all_transfer_metrics.csv")

    if summary_df.empty and metrics_df.empty:
        return None

    model_name = canonical_model_name(config_map.get("model_dirname"), bundle_dir.name)
    bundle_id = str(bundle_dir.resolve())

    if not summary_df.empty:
        summary_df = summary_df.copy()
        if "feature_space" not in summary_df.columns:
            summary_df["feature_space"] = summary_df.get("feature_space_title", pd.Series(dtype=object))
        summary_df["bundle_id"] = bundle_id
        summary_df["bundle_name"] = bundle_dir.name
        summary_df["Model"] = model_name
        summary_df["target_title"] = [
            canonical_target_title(target_name, target_title)
            for target_name, target_title in zip(
                summary_df.get("target_name", pd.Series(dtype=object)),
                summary_df.get("target_title", pd.Series(dtype=object)),
                strict=False,
            )
        ]
        summary_df["feature_set"] = [
            canonical_feature_set(feature_space, feature_space_title)
            for feature_space, feature_space_title in zip(
                summary_df.get("feature_space", pd.Series(dtype=object)),
                summary_df.get("feature_space_title", pd.Series(dtype=object)),
                strict=False,
            )
        ]

    if not metrics_df.empty:
        metrics_df = metrics_df.copy()
        if "feature_space" not in metrics_df.columns:
            metrics_df["feature_space"] = metrics_df.get("feature_space_title", pd.Series(dtype=object))
        metrics_df["bundle_id"] = bundle_id
        metrics_df["bundle_name"] = bundle_dir.name
        metrics_df["Model"] = model_name
        metrics_df["target_title"] = [
            canonical_target_title(target_name, target_title)
            for target_name, target_title in zip(
                metrics_df.get("target_name", pd.Series(dtype=object)),
                metrics_df.get("target_title", pd.Series(dtype=object)),
                strict=False,
            )
        ]
        metrics_df["feature_set"] = [
            canonical_feature_set(feature_space, feature_space_title)
            for feature_space, feature_space_title in zip(
                metrics_df.get("feature_space", pd.Series(dtype=object)),
                metrics_df.get("feature_space_title", pd.Series(dtype=object)),
                strict=False,
            )
        ]
        metrics_df["auroc"] = pd.to_numeric(metrics_df.get("auroc", pd.Series(dtype=float)), errors="coerce")
        if "train_env" in metrics_df.columns:
            metrics_df["train_env"] = metrics_df["train_env"].astype(str)
        if "test_env" in metrics_df.columns:
            metrics_df["test_env"] = metrics_df["test_env"].astype(str)
        if "eval_role" in metrics_df.columns:
            metrics_df["eval_role"] = metrics_df["eval_role"].astype(str)

    inventory_row = build_inventory_row(bundle_dir, model_name, summary_df, metrics_df)
    return summary_df, metrics_df, inventory_row


def load_cross_run_bundle_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory_rows: list[dict[str, object]] = []
    summary_frames: list[pd.DataFrame] = []
    metrics_frames: list[pd.DataFrame] = []

    for bundle_dir in discover_bundle_dirs():
        loaded = load_bundle_frames(bundle_dir)
        if loaded is None:
            continue
        summary_df, metrics_df, inventory_row = loaded
        inventory_rows.append(inventory_row)
        if not summary_df.empty:
            summary_frames.append(summary_df)
        if not metrics_df.empty:
            metrics_frames.append(metrics_df)

    inventory_df = pd.DataFrame(inventory_rows)
    if not inventory_df.empty:
        inventory_df["_model_sort"] = pd.Categorical(
            inventory_df["Model"],
            categories=MODEL_ORDER,
            ordered=True,
        )
        inventory_df = inventory_df.sort_values(["_model_sort", "Model", "Bundle"]).drop(columns="_model_sort").reset_index(drop=True)
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    return inventory_df, summary_df, metrics_df


def _stderr(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return float("nan")
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def _mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def _bundle_mean_and_se(df: pd.DataFrame, value_col: str) -> tuple[float, float]:
    clean_df = df.loc[pd.to_numeric(df[value_col], errors="coerce").notna(), ["bundle_id", value_col]].copy()
    if clean_df.empty:
        return float("nan"), float("nan")
    bundle_means = clean_df.groupby("bundle_id", as_index=False)[value_col].mean()
    if len(bundle_means) >= 2:
        return _mean(bundle_means[value_col]), _stderr(bundle_means[value_col])
    return _mean(clean_df[value_col]), _stderr(clean_df[value_col])


def _fallback_summary_row(summary_slice: pd.DataFrame, env_count: int) -> tuple[float, float, float, float]:
    val_mean = _mean(summary_slice.get("mean_val_auroc", pd.Series(dtype=float)))
    mean_ood = _mean(summary_slice.get("mean_ood_auroc", pd.Series(dtype=float)))
    val_se = float("nan")
    ood_std = _mean(summary_slice.get("std_ood_auroc", pd.Series(dtype=float)))
    if not math.isfinite(ood_std):
        ood_se = float("nan")
    else:
        num_pairs = max(env_count * max(env_count - 1, 0), 1)
        ood_se = float(ood_std / math.sqrt(num_pairs))
    return val_mean, val_se, mean_ood, ood_se


def available_models(summary_df: pd.DataFrame, metrics_df: pd.DataFrame, target_name: str) -> list[str]:
    if "Model" in summary_df.columns:
        summary_models = summary_df.loc[
            _column_or_empty(summary_df, "target_name").eq(target_name),
            "Model",
        ].dropna().astype(str).tolist()
    else:
        summary_models = []
    if "Model" in metrics_df.columns:
        metric_models = metrics_df.loc[
            _column_or_empty(metrics_df, "target_name").eq(target_name),
            "Model",
        ].dropna().astype(str).tolist()
    else:
        metric_models = []
    models = sorted(
        set(summary_models) | set(metric_models),
        key=_model_sort_key,
    )
    return models


def target_rows(summary_df: pd.DataFrame, metrics_df: pd.DataFrame) -> list[tuple[str, str]]:
    target_names = sorted(
        set(_column_or_empty(summary_df, "target_name").dropna().astype(str).tolist())
        | set(_column_or_empty(metrics_df, "target_name").dropna().astype(str).tolist()),
        key=_target_sort_key,
    )
    rows: list[tuple[str, str]] = []
    for target_name in target_names:
        title_series = pd.concat(
            [
                summary_df.loc[_column_or_empty(summary_df, "target_name").eq(target_name), "target_title"]
                if "target_title" in summary_df.columns
                else pd.Series(dtype=object),
                metrics_df.loc[_column_or_empty(metrics_df, "target_name").eq(target_name), "target_title"]
                if "target_title" in metrics_df.columns
                else pd.Series(dtype=object),
            ],
            ignore_index=True,
        ).dropna()
        target_title = str(title_series.iloc[0]) if not title_series.empty else TARGET_TITLE_OVERRIDES.get(target_name, target_name)
        rows.append((target_name, target_title))
    return rows


def main_paper_target_rows(summary_df: pd.DataFrame, metrics_df: pd.DataFrame) -> list[tuple[str, str]]:
    target_lookup = {target_name: target_title for target_name, target_title in target_rows(summary_df, metrics_df)}
    rows: list[tuple[str, str]] = []
    for target_name in MAIN_PAPER_TARGET_ORDER:
        if target_name in target_lookup:
            rows.append((target_name, target_lookup[target_name]))
    return rows


def build_feature_summary_table(
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    target_name: str,
    feature_order: list[str],
) -> pd.DataFrame:
    models = available_models(summary_df, metrics_df, target_name)
    rows: list[dict[str, object]] = []
    env_count = len(ENV_ORDER)
    for model_name in models:
        for feature_set in feature_order:
            metric_slice = metrics_df.loc[
                metrics_df["target_name"].eq(target_name)
                & metrics_df["Model"].eq(model_name)
                & metrics_df["feature_set"].eq(feature_set)
            ].copy()
            if not metric_slice.empty:
                val_slice = metric_slice.loc[metric_slice["eval_role"].eq("val")].copy()
                ood_slice = metric_slice.loc[metric_slice["eval_role"].eq("ood")].copy()
                validation_auroc, validation_auroc_se = _bundle_mean_and_se(val_slice, "auroc")
                mean_ood_auroc, mean_ood_auroc_se = _bundle_mean_and_se(ood_slice, "auroc")
            else:
                summary_slice = summary_df.loc[
                    summary_df["target_name"].eq(target_name)
                    & summary_df["Model"].eq(model_name)
                    & summary_df["feature_set"].eq(feature_set)
                ].copy()
                if summary_slice.empty:
                    validation_auroc = float("nan")
                    validation_auroc_se = float("nan")
                    mean_ood_auroc = float("nan")
                    mean_ood_auroc_se = float("nan")
                else:
                    validation_auroc, validation_auroc_se, mean_ood_auroc, mean_ood_auroc_se = _fallback_summary_row(summary_slice, env_count)
            rows.append(
                {
                    "Model": model_name,
                    "Feature Set": feature_set,
                    "Validation AUROC": validation_auroc,
                    "Validation AUROC SE": validation_auroc_se,
                    "Mean OOD AUROC": mean_ood_auroc,
                    "Mean OOD AUROC SE": mean_ood_auroc_se,
                }
            )
    return pd.DataFrame(rows)


def build_environment_summary_table(
    metrics_df: pd.DataFrame,
    *,
    target_name: str,
    feature_order: list[str],
) -> pd.DataFrame:
    models = available_models(pd.DataFrame(), metrics_df, target_name)
    if metrics_df.empty or not models:
        return pd.DataFrame(columns=["Model", "Feature Set", "Environment", "Validation AUROC", "Validation AUROC SE", "OOD AUROC", "OOD AUROC SE"])

    target_slice = metrics_df.loc[metrics_df["target_name"].eq(target_name)].copy()
    val_df = (
        target_slice.loc[target_slice["eval_role"].eq("val"), ["bundle_id", "Model", "feature_set", "train_env", "auroc"]]
        .rename(columns={"auroc": "validation_auroc"})
        .drop_duplicates()
    )
    ood_df = (
        target_slice.loc[target_slice["eval_role"].eq("ood"), ["bundle_id", "Model", "feature_set", "train_env", "test_env", "auroc"]]
        .rename(columns={"auroc": "ood_auroc", "test_env": "Environment"})
        .copy()
    )
    merged_df = ood_df.merge(
        val_df,
        on=["bundle_id", "Model", "feature_set", "train_env"],
        how="left",
        validate="many_to_one",
    )

    rows: list[dict[str, object]] = []
    for model_name in models:
        for feature_set in feature_order:
            for environment in ENV_ORDER:
                env_slice = merged_df.loc[
                    merged_df["Model"].eq(model_name)
                    & merged_df["feature_set"].eq(feature_set)
                    & merged_df["Environment"].eq(environment)
                ].copy()
                if env_slice.empty:
                    validation_auroc = float("nan")
                    validation_auroc_se = float("nan")
                    ood_auroc = float("nan")
                    ood_auroc_se = float("nan")
                else:
                    validation_auroc, validation_auroc_se = _bundle_mean_and_se(env_slice, "validation_auroc")
                    ood_auroc, ood_auroc_se = _bundle_mean_and_se(env_slice, "ood_auroc")
                rows.append(
                    {
                        "Model": model_name,
                        "Feature Set": feature_set,
                        "Environment": environment,
                        "Validation AUROC": validation_auroc,
                        "Validation AUROC SE": validation_auroc_se,
                        "OOD AUROC": ood_auroc,
                        "OOD AUROC SE": ood_auroc_se,
                    }
                )
    return pd.DataFrame(rows)


def _format_auroc_with_se(value: Any, se_value: Any) -> str:
    mean_numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(mean_numeric):
        return ""
    se_numeric = pd.to_numeric(pd.Series([se_value]), errors="coerce").iloc[0]
    if pd.isna(se_numeric):
        return f"{float(mean_numeric):.3f}"
    return f"{float(mean_numeric):.3f} +/- {float(se_numeric):.3f}"


def build_main_paper_ood_matrix(
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    target_name: str,
    feature_order: list[str] | None = None,
) -> pd.DataFrame:
    feature_order = CORE_FEATURE_ORDER if feature_order is None else feature_order
    source_df = build_feature_summary_table(
        summary_df,
        metrics_df,
        target_name=target_name,
        feature_order=feature_order,
    )
    if source_df.empty:
        return pd.DataFrame(columns=["Model", *feature_order])

    models = [model_name for model_name in MODEL_ORDER if model_name in set(source_df["Model"].astype(str))]
    extra_models = sorted(
        [model_name for model_name in source_df["Model"].dropna().astype(str).unique().tolist() if model_name not in models],
        key=_model_sort_key,
    )
    ordered_models = [*models, *extra_models]

    rows: list[dict[str, str]] = []
    for model_name in ordered_models:
        row: dict[str, str] = {"Model": model_name}
        model_slice = source_df.loc[source_df["Model"].eq(model_name)].copy()
        for feature_set in feature_order:
            feature_slice = model_slice.loc[model_slice["Feature Set"].eq(feature_set)]
            if feature_slice.empty:
                row[feature_set] = ""
                continue
            first_row = feature_slice.iloc[0]
            row[feature_set] = _format_auroc_with_se(
                first_row.get("Mean OOD AUROC"),
                first_row.get("Mean OOD AUROC SE"),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_feature_set_choice(
    requested_feature_set: str,
    *,
    feature_order: list[str],
) -> str:
    resolved_feature_set = canonical_feature_set(requested_feature_set, requested_feature_set)
    if resolved_feature_set in feature_order:
        return resolved_feature_set

    clean_requested = _clean_key(requested_feature_set)
    for feature_set in feature_order:
        if _clean_key(feature_set) == clean_requested:
            return feature_set

    raise ValueError(
        "Unknown feature set for main-paper heatmaps: "
        f"{requested_feature_set!r}. "
        f"Choose one of: {', '.join(feature_order)}"
    )


def select_main_paper_heatmap_feature_sets(
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    target_name: str,
    feature_order: list[str] | None = None,
    requested_feature_set: str | None = None,
) -> pd.DataFrame:
    feature_order = CORE_FEATURE_ORDER if feature_order is None else feature_order
    source_df = build_feature_summary_table(
        summary_df,
        metrics_df,
        target_name=target_name,
        feature_order=feature_order,
    )
    if source_df.empty:
        return pd.DataFrame(columns=["Model", "Feature Set", "Mean OOD AUROC", "Mean OOD AUROC SE"])

    if requested_feature_set is not None:
        resolved_feature_set = resolve_feature_set_choice(
            requested_feature_set,
            feature_order=feature_order,
        )
        selected_df = (
            source_df.loc[source_df["Feature Set"].eq(resolved_feature_set)]
            .reset_index(drop=True)
        )
        return selected_df.loc[:, ["Model", "Feature Set", "Mean OOD AUROC", "Mean OOD AUROC SE"]]

    feature_rank = {feature_name: idx for idx, feature_name in enumerate(feature_order)}
    ranked_df = source_df.copy()
    ranked_df["_model_sort"] = pd.Categorical(
        ranked_df["Model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    ranked_df["_feature_rank"] = [
        feature_rank.get(str(feature_set), len(feature_rank))
        for feature_set in ranked_df["Feature Set"]
    ]
    ranked_df = ranked_df.sort_values(
        ["_model_sort", "Model", "Mean OOD AUROC", "_feature_rank", "Feature Set"],
        ascending=[True, True, False, True, True],
        na_position="last",
        kind="mergesort",
    )
    selected_df = (
        ranked_df.drop_duplicates(subset=["Model"], keep="first")
        .drop(columns=["_model_sort", "_feature_rank"])
        .reset_index(drop=True)
    )
    return selected_df.loc[:, ["Model", "Feature Set", "Mean OOD AUROC", "Mean OOD AUROC SE"]]


def build_model_train_eval_auroc_matrices(
    metrics_df: pd.DataFrame,
    *,
    target_name: str,
    model_name: str,
    feature_set: str,
    env_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    env_order = ENV_ORDER if env_order is None else env_order
    mean_df = pd.DataFrame(np.nan, index=env_order, columns=env_order, dtype=float)
    se_df = pd.DataFrame(np.nan, index=env_order, columns=env_order, dtype=float)

    if metrics_df.empty:
        return mean_df, se_df

    subset = metrics_df.loc[
        metrics_df["target_name"].eq(target_name)
        & metrics_df["Model"].eq(model_name)
        & metrics_df["feature_set"].eq(feature_set)
    ].copy()
    if subset.empty:
        return mean_df, se_df

    for train_env in env_order:
        val_slice = subset.loc[
            subset["eval_role"].eq("val")
            & subset["train_env"].eq(train_env)
        ]
        if not val_slice.empty:
            val_mean, val_se = _bundle_mean_and_se(val_slice, "auroc")
            mean_df.loc[train_env, train_env] = val_mean
            se_df.loc[train_env, train_env] = val_se

        for eval_env in env_order:
            if train_env == eval_env:
                continue
            ood_slice = subset.loc[
                subset["eval_role"].eq("ood")
                & subset["train_env"].eq(train_env)
                & subset["test_env"].eq(eval_env)
            ]
            if ood_slice.empty:
                continue
            ood_mean, ood_se = _bundle_mean_and_se(ood_slice, "auroc")
            mean_df.loc[train_env, eval_env] = ood_mean
            se_df.loc[train_env, eval_env] = ood_se

    return mean_df, se_df


def _paper_heatmap_limits(mean_matrices: list[pd.DataFrame]) -> tuple[float, float]:
    finite_values = np.asarray(
        np.concatenate([matrix.to_numpy(dtype=float).ravel() for matrix in mean_matrices]),
        dtype=float,
    )
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    vmin = max(0.0, float(finite_values.min()) - 0.02)
    vmax = min(1.0, float(finite_values.max()) + 0.02)
    if vmax - vmin < 0.08:
        midpoint = (vmin + vmax) / 2.0
        vmin = max(0.0, midpoint - 0.04)
        vmax = min(1.0, midpoint + 0.04)
    return vmin, vmax


def _format_heatmap_annotation(mean_value: Any, se_value: Any) -> str:
    mean_numeric = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
    if pd.isna(mean_numeric):
        return "NA"
    se_numeric = pd.to_numeric(pd.Series([se_value]), errors="coerce").iloc[0]
    if pd.isna(se_numeric):
        return f"{float(mean_numeric):.3f}"
    return f"{float(mean_numeric):.3f}\n+/- {float(se_numeric):.3f}"


def _annotation_color(cmap: Any, *, value: float, vmin: float, vmax: float) -> str:
    if not math.isfinite(value):
        return "#555555"
    if vmax <= vmin:
        return "black"
    normalized = (float(value) - vmin) / (vmax - vmin)
    normalized = min(max(normalized, 0.0), 1.0)
    r, g, b, _ = cmap(normalized)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.58 else "white"


def _wrap_panel_title(text: str, *, width: int = 26) -> str:
    return textwrap.fill(str(text), width=width, break_long_words=False)


def plot_main_paper_model_heatmaps(
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    target_name: str,
    target_title: str,
    feature_order: list[str] | None = None,
    shared_feature_set: str | None = None,
    env_order: list[str] | None = None,
    nrows: int = 2,
    ncols: int = 2,
) -> tuple[plt.Figure, pd.DataFrame]:
    feature_order = CORE_FEATURE_ORDER if feature_order is None else feature_order
    env_order = ENV_ORDER if env_order is None else env_order
    selected_df = select_main_paper_heatmap_feature_sets(
        summary_df,
        metrics_df,
        target_name=target_name,
        feature_order=feature_order,
        requested_feature_set=shared_feature_set,
    )
    if selected_df.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.axis("off")
        ax.text(0.5, 0.5, "No heatmap data available.", ha="center", va="center", fontsize=12)
        return fig, selected_df

    resolved_shared_feature_set = None
    if shared_feature_set is not None:
        resolved_shared_feature_set = resolve_feature_set_choice(
            shared_feature_set,
            feature_order=feature_order,
        )

    panel_count = int(nrows) * int(ncols)
    ordered_models = selected_df["Model"].dropna().astype(str).tolist()
    if len(ordered_models) > panel_count:
        raise ValueError(
            f"Requested a {nrows}x{ncols} heatmap grid for {len(ordered_models)} models. "
            "Increase `nrows` or `ncols`."
        )

    matrix_lookup: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    mean_matrices: list[pd.DataFrame] = []
    for _, row in selected_df.iterrows():
        mean_df, se_df = build_model_train_eval_auroc_matrices(
            metrics_df,
            target_name=target_name,
            model_name=str(row["Model"]),
            feature_set=str(row["Feature Set"]),
            env_order=env_order,
        )
        matrix_lookup[str(row["Model"])] = (mean_df, se_df)
        mean_matrices.append(mean_df)

    vmin, vmax = _paper_heatmap_limits(mean_matrices)
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#f3f4f6")

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13.8, 10.6),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)
    image = None

    for idx, ax in enumerate(axes.flat):
        panel_label = chr(ord("A") + idx)
        ax.text(
            -0.16,
            1.08,
            panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
        )

        if idx >= len(selected_df):
            ax.set_facecolor("#fafafa")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linestyle((0, (3, 3)))
                spine.set_linewidth(1.0)
                spine.set_edgecolor("#9ca3af")
            ax.text(
                0.5,
                0.5,
                "Reserved for\nfuture model",
                ha="center",
                va="center",
                fontsize=12,
                color="#6b7280",
            )
            continue

        row = selected_df.iloc[idx]
        model_name = str(row["Model"])
        feature_set = str(row["Feature Set"])
        mean_df, se_df = matrix_lookup[model_name]
        matrix = np.ma.masked_invalid(mean_df.to_numpy(dtype=float))
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_facecolor("#f8fafc")
        ax.set_xticks(np.arange(len(env_order)))
        ax.set_xticklabels(env_order, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(np.arange(len(env_order)))
        ax.set_yticklabels(env_order, fontsize=9)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, len(env_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(env_order), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if resolved_shared_feature_set is None:
            panel_title = f"{model_name}\n{_wrap_panel_title(feature_set)}"
        else:
            panel_title = model_name
        ax.set_title(panel_title, fontsize=11.5, fontweight="bold", pad=10)

        for row_idx in range(mean_df.shape[0]):
            for col_idx in range(mean_df.shape[1]):
                mean_value = mean_df.iat[row_idx, col_idx]
                se_value = se_df.iat[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    _format_heatmap_annotation(mean_value, se_value),
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color=_annotation_color(cmap, value=float(mean_value), vmin=vmin, vmax=vmax),
                    linespacing=1.15,
                )

    title_lines = [target_title]
    if resolved_shared_feature_set is not None:
        title_lines.append(f"Feature set: {resolved_shared_feature_set}")
    #title_lines.append("Diagonal = validation AUROC; off-diagonal = OOD AUROC")
    fig.suptitle("\n".join(title_lines), fontsize=16, fontweight="bold")
    fig.supxlabel("Evaluation Environment", fontsize=13)
    fig.supylabel("Training Environment", fontsize=13)

    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.03,
            pad=0.02,
            shrink=0.92,
            label="AUROC",
        )

    return fig, selected_df


def style_metric_table(df: pd.DataFrame):
    if df.empty:
        return df
    format_map = {
        column: "{:.3f}"
        for column in df.columns
        if "AUROC" in column
    }
    return df.style.hide(axis="index").format(format_map)


def style_text_table(df: pd.DataFrame):
    if df.empty:
        return df
    return df.style.hide(axis="index")


def missing_feature_sets(table_df: pd.DataFrame) -> dict[str, list[str]]:
    if table_df.empty:
        return {}
    missing: dict[str, list[str]] = {}
    for model_name, model_df in table_df.groupby("Model", sort=False):
        missing_features = [
            str(row["Feature Set"])
            for _, row in model_df.iterrows()
            if pd.isna(row.get("Validation AUROC")) and pd.isna(row.get("Mean OOD AUROC", row.get("OOD AUROC")))
        ]
        if missing_features:
            missing[str(model_name)] = missing_features
    return missing


def render_missing_feature_note(missing: dict[str, list[str]]) -> str | None:
    if not missing:
        return None
    lines = ["Missing feature runs in the currently discovered bundles:"]
    for model_name in sorted(missing, key=_model_sort_key):
        lines.append(f"- {model_name}: {', '.join(missing[model_name])}")
    return "\n".join(lines)
