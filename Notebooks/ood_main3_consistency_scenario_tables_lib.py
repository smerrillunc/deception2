from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython import get_ipython
    from IPython.display import Markdown, display
except ImportError:  # pragma: no cover
    get_ipython = None
    Markdown = None

    def display(obj: Any) -> None:
        print(obj)


NOTEBOOK_DIR = Path(__file__).resolve().parent
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

import ood_main3_consistency_precomputed_tables_lib as base_tables


DEFAULT_RESULTS_ROOT = Path(
    "/playpen-ssd/smerrill/deception2/Results/OOD_Modeling_main3_consistency_xgb_pca_64_128_256"
)
SCENARIO_TITLE_OVERRIDES = {
    "single_source_ood": "Train on 1 environment; evaluate OOD on the other 4",
    "holdout_env_ood": "Train on 4 environments; evaluate OOD on the held-out environment",
}
FAMILY_ORDER = ["attention_only", "activation_only", "attention_plus_activation", "baseline"]
FAMILY_DISPLAY = {
    "attention_only": "Attention only",
    "activation_only": "Activation only",
    "attention_plus_activation": "Attention + activation",
    "baseline": "Baseline",
}
PANEL_TITLE_OVERRIDES = {
    "delta_pos_gt_0_3": "Deceptive commitment prediction",
    "delta_neg_lt_neg_0_3": "Honest commitment prediction",
}
FEATURE_GROUP_SPECS = [
    (
        "Activation",
        [
            "Baseline (Activation only: raw)",
            "Activation only: PCA final",
            "Activation only: PCA final - prev",
            "Activation only: PCA final - mean(prev 4)",
        ],
    ),
    ("Attention", ["Attention only"]),
    (
        "Combined",
        [
            "Attention + PCA final",
            "Attention + PCA final - prev",
            "Attention + PCA final - mean(prev 4)",
        ],
    ),
]
SCENARIO_FEATURE_ORDER = [feature_name for _, feature_names in FEATURE_GROUP_SPECS for feature_name in feature_names]
ATTENTION_ABLATION_GROUP_SPECS = [
    (
        "Attention",
        list(base_tables.ATTENTION_SUBSET_FEATURE_ORDER),
    ),
]
ATTENTION_ABLATION_FEATURE_ORDER = [feature_name for _, feature_names in ATTENTION_ABLATION_GROUP_SPECS for feature_name in feature_names]
FEATURE_SHORT_LABELS = {
    "Baseline (Activation only: raw)": "\\textbf{Raw}",
    "Activation only: PCA final": "\\shortstack[c]{\\textbf{PCA}\\\\\\textbf{final}}",
    "Activation only: PCA final - prev": "\\shortstack[c]{\\textbf{PCA final}\\\\\\textbf{$-$ prev}}",
    "Activation only: PCA final - mean(prev 4)": "\\shortstack[c]{\\textbf{PCA final}\\\\\\textbf{$-$ mean(prev 4)}}",
    "Attention only": "\\shortstack[c]{\\textbf{Attention}\\\\\\textbf{only}}",
    "Attention only: grounding": "\\shortstack[c]{\\textbf{Attention:}\\\\\\textbf{grounding only}}",
    "Attention only: concentration": "\\shortstack[c]{\\textbf{Attention:}\\\\\\textbf{concentration only}}",
    "Attention only: grounding transition": "\\shortstack[c]{\\textbf{Attention:}\\\\\\textbf{grounding transition only}}",
    "Attention only: concentration transition": "\\shortstack[c]{\\textbf{Attention:}\\\\\\textbf{concentration transition only}}",
    "Attention + PCA final": "\\shortstack[c]{\\textbf{Attention}\\\\\\textbf{+ PCA final}}",
    "Attention + PCA final - prev": "\\shortstack[c]{\\textbf{Attention +}\\\\\\textbf{PCA final $-$ prev}}",
    "Attention + PCA final - mean(prev 4)": "\\shortstack[c]{\\textbf{Attention + PCA final}\\\\\\textbf{$-$ mean(prev 4)}}",
}
FEATURE_MARKDOWN_LABELS = {
    "Baseline (Activation only: raw)": "Raw",
    "Activation only: PCA final": "PCA final",
    "Activation only: PCA final - prev": "PCA final - prev",
    "Activation only: PCA final - mean(prev 4)": "PCA final - mean(prev 4)",
    "Attention only": "Attention only",
    "Attention only: grounding": "Attention: grounding only",
    "Attention only: concentration": "Attention: concentration only",
    "Attention only: grounding transition": "Attention: grounding transition only",
    "Attention only: concentration transition": "Attention: concentration transition only",
    "Attention + PCA final": "Attention + PCA final",
    "Attention + PCA final - prev": "Attention + PCA final - prev",
    "Attention + PCA final - mean(prev 4)": "Attention + PCA final - mean(prev 4)",
}


pd.options.display.max_columns = 200


def md(text: str) -> None:
    if Markdown is not None and _is_notebook_shell():
        display(Markdown(text))
    else:
        print(text)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _is_notebook_shell() -> bool:
    if get_ipython is None or get_ipython() is None:
        return False
    return get_ipython().__class__.__name__ == "ZMQInteractiveShell"


def display_text_table(df: pd.DataFrame) -> None:
    if _is_notebook_shell():
        display(base_tables.style_text_table(df))
    else:
        print(df.to_string(index=False))


def _series_or_empty(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(dtype=object)


def _text_or_empty(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _first_present(values: pd.Series, default: object = pd.NA) -> object:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return value
    return default


def _mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def _std(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return float("nan")
    return float(clean.std(ddof=1))


def _stderr(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return float("nan")
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def _minimum(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.min())


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_numeric = pd.to_numeric(values, errors="coerce")
    weight_numeric = pd.to_numeric(weights, errors="coerce")
    keep_mask = value_numeric.notna() & weight_numeric.notna() & weight_numeric.gt(0)
    if not keep_mask.any():
        return _mean(values)
    return float(np.average(value_numeric.loc[keep_mask], weights=weight_numeric.loc[keep_mask]))


def _stderr_from_series(values: pd.Series) -> float:
    return _stderr(values)


def _format_pm_value(mean_value: Any, se_value: Any, *, style: str) -> str:
    mean_numeric = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
    if pd.isna(mean_numeric):
        return ""
    se_numeric = pd.to_numeric(pd.Series([se_value]), errors="coerce").iloc[0]
    if style == "latex":
        if pd.isna(se_numeric):
            return f"{float(mean_numeric):.3f}"
        return f"{float(mean_numeric):.3f} $\\\\pm$ {float(se_numeric):.3f}"
    if pd.isna(se_numeric):
        return f"{float(mean_numeric):.3f}"
    return f"{float(mean_numeric):.3f} ± {float(se_numeric):.3f}"


def _format_ood_validation_cell(
    ood_mean: Any,
    ood_se: Any,
    validation_mean: Any,
    validation_se: Any,
    *,
    style: str,
) -> str:
    ood_text = _format_pm_value(ood_mean, ood_se, style=style)
    validation_text = _format_pm_value(validation_mean, validation_se, style=style)
    if not ood_text and not validation_text:
        return ""
    if not ood_text:
        ood_text = "NA"
    if not validation_text:
        validation_text = "NA"
    return f"{ood_text} ({validation_text})"


def format_mean_se(values: pd.Series) -> str:
    mean_value = _mean(values)
    se_value = _stderr(values)
    if not np.isfinite(mean_value):
        return ""
    if np.isfinite(se_value):
        return f"{mean_value:.3f} +/- {se_value:.3f}"
    return f"{mean_value:.3f}"


def join_unique_text(values: pd.Series) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values.dropna().astype(str):
        clean = value.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return " | ".join(ordered)


def maybe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_config_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    config_df = pd.read_csv(path)
    if {"setting", "value"}.issubset(config_df.columns):
        return {str(row.setting): str(row.value) for row in config_df.itertuples(index=False)}
    return {}


def is_bundle_dir(path: Path) -> bool:
    return path.is_dir() and (path / "all_transfer_metrics.csv").exists()


def discover_bundle_dirs(root: Path) -> list[Path]:
    if is_bundle_dir(root):
        return [root.resolve()]
    if not root.exists():
        return []
    return sorted(
        [path.resolve() for path in root.iterdir() if is_bundle_dir(path)],
        key=lambda path: path.name,
    )


def requested_feature_size_label(requested_feature_size: int) -> str:
    return f"k{int(requested_feature_size):03d}"


def infer_requested_feature_sizes(metrics_df: pd.DataFrame) -> list[int]:
    labels = _series_or_empty(metrics_df, "feature_size_label").dropna().astype(str)
    sizes: set[int] = set()
    for label in labels:
        match = re.fullmatch(r"k(\d+)", label.strip().lower())
        if match is not None:
            sizes.add(int(match.group(1)))
    return sorted(sizes)


def model_sort_key(model_name: str) -> tuple[int, str]:
    try:
        return (base_tables.MODEL_ORDER.index(model_name), model_name)
    except ValueError:
        return (len(base_tables.MODEL_ORDER), model_name)


def family_sort_key(family_name: str) -> tuple[int, str]:
    try:
        return (FAMILY_ORDER.index(family_name), family_name)
    except ValueError:
        return (len(FAMILY_ORDER), family_name)


def env_sort_key(env_name: str) -> tuple[int, str]:
    try:
        return (base_tables.ENV_ORDER.index(env_name), env_name)
    except ValueError:
        return (len(base_tables.ENV_ORDER), env_name)


def train_axis_labels_for_scenario(scenario_name: str) -> list[str]:
    if scenario_name == "single_source_ood":
        return list(base_tables.ENV_ORDER)
    return [f"All except {env_name}" for env_name in base_tables.ENV_ORDER]


def target_rows(metrics_df: pd.DataFrame) -> list[tuple[str, str]]:
    names = sorted(
        _series_or_empty(metrics_df, "target_name").dropna().astype(str).unique().tolist(),
        key=lambda value: (
            base_tables.TARGET_ORDER.index(value)
            if value in base_tables.TARGET_ORDER
            else len(base_tables.TARGET_ORDER),
            value,
        ),
    )
    rows: list[tuple[str, str]] = []
    for target_name in names:
        target_slice = metrics_df.loc[_series_or_empty(metrics_df, "target_name").astype(str).eq(target_name)]
        target_title = base_tables.canonical_target_title(
            target_name,
            _first_present(_series_or_empty(target_slice, "target_title")),
        )
        rows.append((target_name, target_title))
    return rows


def _family_requested_size_mask(
    metrics_df: pd.DataFrame,
    *,
    family_name: str,
    requested_feature_size: int,
) -> pd.Series:
    labels = _series_or_empty(metrics_df, "feature_size_label").fillna("").astype(str).str.lower()
    numeric_sizes = pd.to_numeric(_series_or_empty(metrics_df, "feature_size"), errors="coerce")
    requested_label = requested_feature_size_label(requested_feature_size).lower()
    requested_numeric = int(requested_feature_size)

    if family_name == "attention_only":
        return pd.Series(True, index=metrics_df.index)
    if family_name == "baseline":
        return labels.eq("raw_final") | labels.eq("") | numeric_sizes.isna()
    return labels.eq(requested_label) | numeric_sizes.eq(requested_numeric)


def _rank_feature_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    ranked = candidate_df.copy()
    ranked = ranked.sort_values(
        [
            "mean_ood_auroc",
            "min_ood_auroc",
            "mean_val_auroc",
            "selected_feature_count",
            "selected_feature_space_title",
            "selected_feature_space",
        ],
        ascending=[False, False, False, False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked


def _rank_train_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    ranked = candidate_df.copy()
    ranked = ranked.sort_values(
        [
            "mean_ood_auroc",
            "min_ood_auroc",
            "source_val_auroc",
            "train_env",
        ],
        ascending=[False, False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked


def rebuild_scenario_summaries_from_metrics(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    requested_sizes = infer_requested_feature_sizes(metrics_df)
    if not requested_sizes:
        return pd.DataFrame(), pd.DataFrame()

    panel_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []

    for target_name, target_title in target_rows(metrics_df):
        target_slice = metrics_df.loc[_series_or_empty(metrics_df, "target_name").astype(str).eq(target_name)].copy()
        if target_slice.empty:
            continue

        scenario_name = _text_or_empty(_first_present(_series_or_empty(target_slice, "scenario_name")))
        scenario_title = _text_or_empty(
            _first_present(_series_or_empty(target_slice, "scenario_title"), SCENARIO_TITLE_OVERRIDES.get(scenario_name, ""))
        )

        for requested_feature_size in requested_sizes:
            requested_label = requested_feature_size_label(requested_feature_size)
            for family_name in FAMILY_ORDER:
                family_slice = target_slice.loc[
                    _series_or_empty(target_slice, "feature_family_group").astype(str).eq(family_name)
                    & _family_requested_size_mask(
                        target_slice,
                        family_name=family_name,
                        requested_feature_size=requested_feature_size,
                    )
                ].copy()
                if family_slice.empty:
                    continue

                feature_rows: list[dict[str, object]] = []
                for (feature_space, feature_space_title), feature_df in family_slice.groupby(
                    ["feature_space", "feature_space_title"],
                    dropna=False,
                    sort=False,
                ):
                    val_df = feature_df.loc[_series_or_empty(feature_df, "eval_role").astype(str).eq("val")].copy()
                    ood_df = feature_df.loc[_series_or_empty(feature_df, "eval_role").astype(str).eq("ood")].copy()

                    feature_rows.append(
                        {
                            "scenario_name": scenario_name,
                            "scenario_title": scenario_title,
                            "target_name": target_name,
                            "target_title": target_title,
                            "requested_feature_size": requested_feature_size,
                            "requested_feature_size_label": requested_label,
                            "feature_family_group": family_name,
                            "selected_feature_space": _text_or_empty(feature_space),
                            "selected_feature_space_title": _text_or_empty(
                                feature_space_title
                                if pd.notna(feature_space_title)
                                else base_tables.canonical_feature_set(feature_space, feature_space_title)
                            ),
                            "source_feature_size": _first_present(_series_or_empty(feature_df, "feature_size")),
                            "source_feature_size_label": _first_present(_series_or_empty(feature_df, "feature_size_label")),
                            "attention_feature_count": _first_present(_series_or_empty(feature_df, "attention_feature_count")),
                            "activation_feature_count": _first_present(_series_or_empty(feature_df, "activation_feature_count")),
                            "selected_feature_count": _first_present(_series_or_empty(feature_df, "selected_feature_count")),
                            "effective_activation_pca_dim": _first_present(
                                _series_or_empty(feature_df, "effective_activation_pca_dim")
                            ),
                            "mean_val_auroc": _mean(_series_or_empty(val_df, "auroc")),
                            "mean_ood_auroc": _mean(_series_or_empty(ood_df, "auroc")),
                            "min_ood_auroc": _minimum(_series_or_empty(ood_df, "auroc")),
                            "std_ood_auroc": _std(_series_or_empty(ood_df, "auroc")),
                            "mean_ood_average_precision": _mean(_series_or_empty(ood_df, "average_precision")),
                        }
                    )

                feature_summary_df = _rank_feature_candidates(pd.DataFrame(feature_rows))
                if feature_summary_df.empty:
                    continue
                selected_feature_row = feature_summary_df.iloc[0].to_dict()
                panel_rows.append(selected_feature_row)

                selected_feature_space = str(selected_feature_row["selected_feature_space"])
                selected_feature_slice = family_slice.loc[
                    _series_or_empty(family_slice, "feature_space").astype(str).eq(selected_feature_space)
                ].copy()
                if selected_feature_slice.empty:
                    continue

                train_rows: list[dict[str, object]] = []
                for train_env, train_df in selected_feature_slice.groupby("train_env", dropna=False, sort=False):
                    val_df = train_df.loc[_series_or_empty(train_df, "eval_role").astype(str).eq("val")].copy()
                    ood_df = train_df.loc[_series_or_empty(train_df, "eval_role").astype(str).eq("ood")].copy()
                    train_rows.append(
                        {
                            "scenario_name": scenario_name,
                            "scenario_title": scenario_title,
                            "target_name": target_name,
                            "target_title": target_title,
                            "requested_feature_size": requested_feature_size,
                            "requested_feature_size_label": requested_label,
                            "feature_family_group": family_name,
                            "feature_space": selected_feature_row["selected_feature_space"],
                            "feature_space_title": selected_feature_row["selected_feature_space_title"],
                            "feature_size": selected_feature_row["source_feature_size"],
                            "feature_size_label": selected_feature_row["source_feature_size_label"],
                            "train_env": _text_or_empty(train_env),
                            "source_envs": join_unique_text(_series_or_empty(train_df, "source_envs")),
                            "source_env_count": _first_present(_series_or_empty(train_df, "source_env_count")),
                            "heldout_env": join_unique_text(_series_or_empty(train_df, "heldout_env")),
                            "source_val_auroc": _mean(_series_or_empty(val_df, "auroc")),
                            "mean_ood_auroc": _mean(_series_or_empty(ood_df, "auroc")),
                            "min_ood_auroc": _minimum(_series_or_empty(ood_df, "auroc")),
                            "std_ood_auroc": _std(_series_or_empty(ood_df, "auroc")),
                            "mean_ood_average_precision": _mean(_series_or_empty(ood_df, "average_precision")),
                            "mean_ood_balanced_accuracy": _mean(_series_or_empty(ood_df, "balanced_accuracy")),
                            "attention_feature_count": selected_feature_row["attention_feature_count"],
                            "activation_feature_count": selected_feature_row["activation_feature_count"],
                            "selected_feature_count": selected_feature_row["selected_feature_count"],
                            "chosen_c": _first_present(_series_or_empty(train_df, "chosen_c")),
                            "chosen_max_depth": _first_present(_series_or_empty(train_df, "chosen_max_depth")),
                            "decision_threshold": _first_present(_series_or_empty(train_df, "decision_threshold")),
                            "effective_activation_pca_dim": selected_feature_row["effective_activation_pca_dim"],
                            "selected_features_path": join_unique_text(_series_or_empty(train_df, "selected_features_path")),
                            "coefficients_path": join_unique_text(_series_or_empty(train_df, "coefficients_path")),
                        }
                    )

                train_summary_df = _rank_train_candidates(pd.DataFrame(train_rows))
                if train_summary_df.empty:
                    continue
                model_rows.append(train_summary_df.iloc[0].to_dict())

    panel_df = pd.DataFrame(panel_rows)
    model_df = pd.DataFrame(model_rows)
    return panel_df, model_df


def load_scenario_bundle_frames(
    results_root: Path,
    scenario_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory_rows: list[dict[str, object]] = []
    panel_frames: list[pd.DataFrame] = []
    model_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []

    for bundle_dir in discover_bundle_dirs(results_root):
        metrics_df = maybe_read_csv(bundle_dir / "all_transfer_metrics.csv")
        if metrics_df.empty:
            continue
        if "scenario_name" in metrics_df.columns:
            metrics_df = metrics_df.loc[_series_or_empty(metrics_df, "scenario_name").astype(str).eq(scenario_name)].copy()
        if metrics_df.empty:
            continue

        config_map = read_config_map(bundle_dir / "config.csv")
        model_name = base_tables.canonical_model_name(config_map.get("model_dirname"), bundle_dir.name)
        bundle_id = str(bundle_dir.resolve())

        metrics_df = metrics_df.copy()
        metrics_df["bundle_id"] = bundle_id
        metrics_df["bundle_name"] = bundle_dir.name
        metrics_df["Model"] = model_name
        metrics_df["target_title"] = [
            base_tables.canonical_target_title(target_name, target_title)
            for target_name, target_title in zip(
                _series_or_empty(metrics_df, "target_name"),
                _series_or_empty(metrics_df, "target_title"),
                strict=False,
            )
        ]
        metrics_df["feature_set"] = [
            base_tables.canonical_feature_set(feature_space, feature_space_title)
            for feature_space, feature_space_title in zip(
                _series_or_empty(metrics_df, "feature_space"),
                _series_or_empty(metrics_df, "feature_space_title"),
                strict=False,
            )
        ]
        numeric_columns = [
            "auroc",
            "average_precision",
            "accuracy",
            "balanced_accuracy",
            "feature_size",
            "attention_feature_count",
            "activation_feature_count",
            "selected_feature_count",
            "source_env_count",
            "chosen_c",
            "chosen_max_depth",
            "decision_threshold",
            "effective_activation_pca_dim",
        ]
        for column in numeric_columns:
            if column in metrics_df.columns:
                metrics_df[column] = pd.to_numeric(metrics_df[column], errors="coerce")
        for column in [
            "train_env",
            "test_env",
            "heldout_env",
            "feature_family_group",
            "feature_space",
            "feature_space_title",
            "feature_set",
            "feature_size_label",
            "eval_role",
        ]:
            if column in metrics_df.columns:
                metrics_df[column] = metrics_df[column].astype(str)

        panel_df, model_df = rebuild_scenario_summaries_from_metrics(metrics_df)
        for df in (panel_df, model_df):
            if df.empty:
                continue
            df["bundle_id"] = bundle_id
            df["bundle_name"] = bundle_dir.name
            df["Model"] = model_name

        requested_sizes = infer_requested_feature_sizes(metrics_df)
        available_targets = ", ".join(target_name for target_name, _ in target_rows(metrics_df))
        inventory_rows.append(
            {
                "Model": model_name,
                "Bundle": bundle_dir.name,
                "Scenario": scenario_name,
                "Targets": available_targets,
                "Requested Sizes": ", ".join(str(size) for size in requested_sizes),
                "Path": bundle_id,
            }
        )
        if not panel_df.empty:
            panel_frames.append(panel_df)
        if not model_df.empty:
            model_frames.append(model_df)
        metric_frames.append(metrics_df)

    inventory_df = pd.DataFrame(inventory_rows)
    if not inventory_df.empty:
        inventory_df = inventory_df.sort_values(
            ["Model", "Bundle"],
            key=lambda column: column.map(lambda value: model_sort_key(value)[0] if column.name == "Model" else value),
        ).reset_index(drop=True)
    panel_df = pd.concat(panel_frames, ignore_index=True) if panel_frames else pd.DataFrame()
    model_df = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
    metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    return inventory_df, panel_df, model_df, metrics_df


def _feature_requested_size_mask(
    metrics_df: pd.DataFrame,
    *,
    feature_set: str,
    requested_feature_size: int,
) -> pd.Series:
    canonical_feature_set = base_tables.canonical_feature_set(feature_set, feature_set)
    labels = _series_or_empty(metrics_df, "feature_size_label").fillna("").astype(str).str.lower()
    numeric_sizes = pd.to_numeric(_series_or_empty(metrics_df, "feature_size"), errors="coerce")
    requested_label = requested_feature_size_label(requested_feature_size).lower()
    requested_numeric = int(requested_feature_size)

    if canonical_feature_set.startswith("Attention only"):
        return labels.eq("all_attention") | labels.eq("")
    if canonical_feature_set == "Baseline (Activation only: raw)":
        return labels.eq("raw_final") | labels.eq("") | numeric_sizes.isna()
    return labels.eq(requested_label) | numeric_sizes.eq(requested_numeric)


def ordered_models_from_metrics(metrics_df: pd.DataFrame) -> list[str]:
    models = sorted(_series_or_empty(metrics_df, "Model").dropna().astype(str).unique().tolist(), key=model_sort_key)
    return models


def build_feature_metric_rows(
    metrics_df: pd.DataFrame,
    *,
    scenario_name: str,
    target_name: str,
    requested_feature_size: int,
    feature_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_order = SCENARIO_FEATURE_ORDER if feature_order is None else feature_order
    target_slice = metrics_df.loc[_series_or_empty(metrics_df, "target_name").astype(str).eq(target_name)].copy()
    if target_slice.empty:
        return pd.DataFrame(), pd.DataFrame()

    split_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for model_name in ordered_models_from_metrics(target_slice):
        model_slice = target_slice.loc[_series_or_empty(target_slice, "Model").astype(str).eq(model_name)].copy()
        for feature_set in feature_order:
            feature_slice = model_slice.loc[
                _series_or_empty(model_slice, "feature_set").astype(str).eq(feature_set)
                & _feature_requested_size_mask(
                    model_slice,
                    feature_set=feature_set,
                    requested_feature_size=requested_feature_size,
                )
            ].copy()
            if feature_slice.empty:
                continue

            for train_env, split_df in feature_slice.groupby("train_env", dropna=False, sort=False):
                val_df = split_df.loc[_series_or_empty(split_df, "eval_role").astype(str).eq("val")].copy()
                ood_df = split_df.loc[_series_or_empty(split_df, "eval_role").astype(str).eq("ood")].copy()
                validation_auroc = _weighted_mean(_series_or_empty(val_df, "auroc"), _series_or_empty(val_df, "n_rows"))
                ood_auroc = _mean(_series_or_empty(ood_df, "auroc"))
                environment_name = (
                    _text_or_empty(_first_present(_series_or_empty(ood_df, "test_env"), _first_present(_series_or_empty(split_df, "heldout_env"))))
                    if scenario_name == "holdout_env_ood"
                    else ""
                )
                split_rows.append(
                    {
                        "bundle_id": _text_or_empty(_first_present(_series_or_empty(split_df, "bundle_id"))),
                        "Model": model_name,
                        "Feature Set": feature_set,
                        "Training Source": _text_or_empty(train_env),
                        "Training Split": _text_or_empty(train_env),
                        "Held-Out Environment": environment_name,
                        "Validation AUROC": validation_auroc,
                        "OOD AUROC": ood_auroc,
                    }
                )

                if scenario_name == "holdout_env_ood":
                    detail_rows.append(
                        {
                            "bundle_id": _text_or_empty(_first_present(_series_or_empty(split_df, "bundle_id"))),
                            "Model": model_name,
                            "Feature Set": feature_set,
                            "Training Split": _text_or_empty(train_env),
                            "Held-Out Environment": environment_name,
                            "Validation AUROC": validation_auroc,
                            "OOD AUROC": ood_auroc,
                        }
                    )
                else:
                    for eval_env, env_df in ood_df.groupby("test_env", dropna=False, sort=False):
                        detail_rows.append(
                            {
                                "bundle_id": _text_or_empty(_first_present(_series_or_empty(env_df, "bundle_id"))),
                                "Model": model_name,
                                "Feature Set": feature_set,
                                "Training Source": _text_or_empty(train_env),
                                "Evaluation Environment": _text_or_empty(eval_env),
                                "Validation AUROC": validation_auroc,
                                "OOD AUROC": _mean(_series_or_empty(env_df, "auroc")),
                            }
                        )

    split_metrics_df = pd.DataFrame(split_rows)
    detail_metrics_df = pd.DataFrame(detail_rows)
    return split_metrics_df, detail_metrics_df


def _build_metric_stats_table(
    source_df: pd.DataFrame,
    *,
    value_column: str,
    group_column: str | None = None,
    feature_order: list[str] | None = None,
) -> pd.DataFrame:
    feature_order = SCENARIO_FEATURE_ORDER if feature_order is None else feature_order
    rows: list[dict[str, object]] = []
    models = ordered_models_from_metrics(source_df)
    group_values = [None] if group_column is None else [
        str(value)
        for value in source_df[group_column].dropna().astype(str).unique().tolist()
    ]
    if group_column is not None:
        group_values = sorted(group_values, key=env_sort_key)

    for group_value in group_values:
        for model_name in models:
            for feature_set in feature_order:
                subset = source_df.loc[
                    _series_or_empty(source_df, "Model").astype(str).eq(model_name)
                    & _series_or_empty(source_df, "Feature Set").astype(str).eq(feature_set)
                ].copy()
                if group_column is not None:
                    subset = subset.loc[_series_or_empty(subset, group_column).astype(str).eq(str(group_value))]
                if subset.empty:
                    mean_value = float("nan")
                    se_value = float("nan")
                else:
                    mean_value = _mean(_series_or_empty(subset, value_column))
                    se_value = _stderr_from_series(_series_or_empty(subset, value_column))
                row: dict[str, object] = {
                    "Model": model_name,
                    "Feature Set": feature_set,
                    "Mean": mean_value,
                    "SE": se_value,
                }
                if group_column is not None:
                    row[group_column] = group_value
                rows.append(row)
    return pd.DataFrame(rows)


def _build_combined_metric_stats_table(
    source_df: pd.DataFrame,
    *,
    ood_column: str,
    validation_column: str,
    group_column: str | None = None,
    feature_order: list[str] | None = None,
) -> pd.DataFrame:
    feature_order = SCENARIO_FEATURE_ORDER if feature_order is None else feature_order
    rows: list[dict[str, object]] = []
    models = ordered_models_from_metrics(source_df)
    group_values = [None] if group_column is None else [
        str(value)
        for value in source_df[group_column].dropna().astype(str).unique().tolist()
    ]
    if group_column is not None:
        group_values = sorted(group_values, key=env_sort_key)

    for group_value in group_values:
        for model_name in models:
            for feature_set in feature_order:
                subset = source_df.loc[
                    _series_or_empty(source_df, "Model").astype(str).eq(model_name)
                    & _series_or_empty(source_df, "Feature Set").astype(str).eq(feature_set)
                ].copy()
                if group_column is not None:
                    subset = subset.loc[_series_or_empty(subset, group_column).astype(str).eq(str(group_value))]
                row: dict[str, object] = {
                    "Model": model_name,
                    "Feature Set": feature_set,
                    "OOD Mean": _mean(_series_or_empty(subset, ood_column)),
                    "OOD SE": _stderr_from_series(_series_or_empty(subset, ood_column)),
                    "Validation Mean": _mean(_series_or_empty(subset, validation_column)),
                    "Validation SE": _stderr_from_series(_series_or_empty(subset, validation_column)),
                }
                if group_column is not None:
                    row[group_column] = group_value
                rows.append(row)
    return pd.DataFrame(rows)


def build_wide_metric_table(
    stats_df: pd.DataFrame,
    *,
    style: str,
    feature_order: list[str] | None = None,
    bold_best: bool = False,
) -> pd.DataFrame:
    feature_order = SCENARIO_FEATURE_ORDER if feature_order is None else feature_order
    if stats_df.empty:
        return pd.DataFrame(columns=["Model", *feature_order])

    models = ordered_models_from_metrics(stats_df)
    value_lookup = {
        (str(row["Model"]), str(row["Feature Set"])): (
            row.get("Mean", float("nan")),
            row.get("SE", float("nan")),
        )
        for _, row in stats_df.iterrows()
    }

    rows: list[dict[str, object]] = []
    for model_name in models:
        row: dict[str, object] = {"Model": model_name}
        best_value = float("nan")
        if bold_best:
            candidate_values = [
                pd.to_numeric(pd.Series([value_lookup.get((model_name, feature_set), (float("nan"), float("nan")))[0]]), errors="coerce").iloc[0]
                for feature_set in feature_order
            ]
            finite_candidates = [float(value) for value in candidate_values if pd.notna(value)]
            if finite_candidates:
                best_value = max(finite_candidates)
        for feature_set in feature_order:
            mean_value, se_value = value_lookup.get((model_name, feature_set), (float("nan"), float("nan")))
            formatted = _format_pm_value(mean_value, se_value, style=style)
            mean_numeric = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
            if bold_best and formatted and pd.notna(mean_numeric) and np.isclose(float(mean_numeric), float(best_value)):
                formatted = f"\\textbf{{{formatted}}}" if style == "latex" else f"**{formatted}**"
            row[feature_set] = formatted
        rows.append(row)
    return pd.DataFrame(rows)


def build_wide_combined_metric_table(
    stats_df: pd.DataFrame,
    *,
    style: str,
    feature_order: list[str] | None = None,
    bold_best: bool = False,
) -> pd.DataFrame:
    feature_order = SCENARIO_FEATURE_ORDER if feature_order is None else feature_order
    if stats_df.empty:
        return pd.DataFrame(columns=["Model", *feature_order])

    models = ordered_models_from_metrics(stats_df)
    value_lookup = {
        (str(row["Model"]), str(row["Feature Set"])): (
            row.get("OOD Mean", float("nan")),
            row.get("OOD SE", float("nan")),
            row.get("Validation Mean", float("nan")),
            row.get("Validation SE", float("nan")),
        )
        for _, row in stats_df.iterrows()
    }

    rows: list[dict[str, object]] = []
    for model_name in models:
        row: dict[str, object] = {"Model": model_name}
        best_value = float("nan")
        if bold_best:
            candidate_values = [
                pd.to_numeric(
                    pd.Series([value_lookup.get((model_name, feature_set), (float("nan"), float("nan"), float("nan"), float("nan")))[0]]),
                    errors="coerce",
                ).iloc[0]
                for feature_set in feature_order
            ]
            finite_candidates = [float(value) for value in candidate_values if pd.notna(value)]
            if finite_candidates:
                best_value = max(finite_candidates)
        for feature_set in feature_order:
            ood_mean, ood_se, validation_mean, validation_se = value_lookup.get(
                (model_name, feature_set),
                (float("nan"), float("nan"), float("nan"), float("nan")),
            )
            formatted = _format_ood_validation_cell(
                ood_mean,
                ood_se,
                validation_mean,
                validation_se,
                style=style,
            )
            mean_numeric = pd.to_numeric(pd.Series([ood_mean]), errors="coerce").iloc[0]
            if bold_best and formatted and pd.notna(mean_numeric) and np.isclose(float(mean_numeric), float(best_value)):
                formatted = f"\\textbf{{{formatted}}}" if style == "latex" else f"**{formatted}**"
            row[feature_set] = formatted
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_row_label(feature_name: str) -> str:
    return FEATURE_MARKDOWN_LABELS.get(feature_name, feature_name)


def build_transposed_combined_metric_table(
    stats_df: pd.DataFrame,
    *,
    style: str,
    feature_order: list[str] | None = None,
    bold_best: bool = False,
) -> pd.DataFrame:
    feature_order = SCENARIO_FEATURE_ORDER if feature_order is None else feature_order
    if stats_df.empty:
        return pd.DataFrame(columns=["Feature Set"])

    models = ordered_models_from_metrics(stats_df)
    value_lookup = {
        (str(row["Feature Set"]), str(row["Model"])): (
            row.get("OOD Mean", float("nan")),
            row.get("OOD SE", float("nan")),
            row.get("Validation Mean", float("nan")),
            row.get("Validation SE", float("nan")),
        )
        for _, row in stats_df.iterrows()
    }

    best_by_model: dict[str, float] = {}
    if bold_best:
        for model_name in models:
            candidate_values = [
                pd.to_numeric(
                    pd.Series([value_lookup.get((feature_set, model_name), (float("nan"), float("nan"), float("nan"), float("nan")))[0]]),
                    errors="coerce",
                ).iloc[0]
                for feature_set in feature_order
            ]
            finite_candidates = [float(value) for value in candidate_values if pd.notna(value)]
            if finite_candidates:
                best_by_model[model_name] = max(finite_candidates)

    rows: list[dict[str, object]] = []
    for feature_set in feature_order:
        row: dict[str, object] = {"Feature Set": _feature_row_label(feature_set)}
        for model_name in models:
            ood_mean, ood_se, validation_mean, validation_se = value_lookup.get(
                (feature_set, model_name),
                (float("nan"), float("nan"), float("nan"), float("nan")),
            )
            formatted = _format_ood_validation_cell(
                ood_mean,
                ood_se,
                validation_mean,
                validation_se,
                style=style,
            )
            mean_numeric = pd.to_numeric(pd.Series([ood_mean]), errors="coerce").iloc[0]
            best_value = best_by_model.get(model_name, float("nan"))
            if bold_best and formatted and pd.notna(mean_numeric) and pd.notna(best_value) and np.isclose(float(mean_numeric), float(best_value)):
                formatted = f"\\textbf{{{formatted}}}" if style == "latex" else f"**{formatted}**"
            row[model_name] = formatted
        rows.append(row)
    return pd.DataFrame(rows)


def render_markdown_feature_table(formatted_df: pd.DataFrame) -> str:
    if formatted_df.empty:
        return ""
    try:
        return formatted_df.to_markdown(index=False)
    except Exception:
        return formatted_df.to_string(index=False)


def render_latex_feature_table(
    formatted_df: pd.DataFrame,
    *,
    panel_title: str,
    feature_order: list[str] | None = None,
    feature_group_specs: list[tuple[str, list[str]]] | None = None,
) -> str:
    feature_order = SCENARIO_FEATURE_ORDER if feature_order is None else feature_order
    if feature_group_specs is None:
        feature_group_specs = FEATURE_GROUP_SPECS
    if formatted_df.empty:
        return ""

    filtered_group_specs: list[tuple[str, list[str]]] = []
    for group_title, feature_names in feature_group_specs:
        filtered_names = [feature_name for feature_name in feature_names if feature_name in feature_order]
        if filtered_names:
            filtered_group_specs.append((group_title, filtered_names))

    lines = [
        "\\begin{table*}[h!]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "",
        f"\\textbf{{{panel_title}}} \\\\",
        "\\vspace{0.25em}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * (len(formatted_df.columns) - 1) + "}",
        "\\toprule",
        "\\textbf{Feature Set}",
    ]
    model_columns = [column for column in formatted_df.columns if column != "Feature Set"]
    header_tail = " & ".join(f"\\textbf{{{column}}}" for column in model_columns)
    lines[-1] += " & " + header_tail + " \\\\"
    lines.append("\\midrule")

    for group_idx, (group_title, group_feature_names) in enumerate(filtered_group_specs):
        group_rows = [feature_name for feature_name in group_feature_names if _feature_row_label(feature_name) in formatted_df["Feature Set"].tolist()]
        if not group_rows:
            continue
        if group_idx > 0:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{len(model_columns) + 1}}}{{l}}{{\\textbf{{{group_title}}}}} \\\\")
        for feature_name in group_rows:
            row = formatted_df.loc[formatted_df["Feature Set"].astype(str).eq(_feature_row_label(feature_name))].iloc[0]
            row_values = [str(row.get(column, "")) for column in model_columns]
            lines.append(str(row["Feature Set"]) + " & " + " & ".join(row_values) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def render_simple_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def render_simple_latex_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return df.to_latex(index=False, escape=False)


def _display_latex_block(latex_text: str) -> None:
    if not latex_text:
        return
    md(f"```latex\n{latex_text}\n```")


def _detail_table_for_style(df: pd.DataFrame, *, style: str) -> pd.DataFrame:
    if df.empty or style == "markdown":
        return df
    out = df.copy()
    for column in out.columns:
        if column in {"Model", "Training Source", "Evaluation Environment", "Training Split", "Held-Out Environment"}:
            continue
        out[column] = [
            str(value).replace("±", "$\\pm$") if str(value).strip() else ""
            for value in out[column]
        ]
    return out


def _rename_feature_columns(df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.rename(columns={feature_name: FEATURE_MARKDOWN_LABELS.get(feature_name, feature_name) for feature_name in feature_order})


def transpose_detail_table(
    df: pd.DataFrame,
    *,
    id_columns: list[str],
    feature_order: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*id_columns, "Feature Set"])

    models = ordered_models_from_metrics(df)
    context_df = df[id_columns].drop_duplicates().reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for _, context_row in context_df.iterrows():
        mask = pd.Series(True, index=df.index)
        for column in id_columns:
            mask &= _series_or_empty(df, column).astype(str).eq(str(context_row[column]))
        context_slice = df.loc[mask].copy()
        for feature_set in feature_order:
            row: dict[str, object] = {
                column: context_row[column]
                for column in id_columns
            }
            row["Feature Set"] = _feature_row_label(feature_set)
            for model_name in models:
                value_slice = context_slice.loc[_series_or_empty(context_slice, "Model").astype(str).eq(model_name)]
                row[model_name] = _text_or_empty(_first_present(_series_or_empty(value_slice, feature_set), ""))
            rows.append(row)
    return pd.DataFrame(rows)


def build_single_source_validation_detail_table(
    split_metrics_df: pd.DataFrame,
    *,
    feature_order: list[str] | None = None,
) -> pd.DataFrame:
    feature_order = SCENARIO_FEATURE_ORDER if feature_order is None else feature_order
    if split_metrics_df.empty:
        return pd.DataFrame(columns=["Model", "Training Source", *feature_order])

    rows: list[dict[str, object]] = []
    models = ordered_models_from_metrics(split_metrics_df)
    for model_name in models:
        model_slice = split_metrics_df.loc[_series_or_empty(split_metrics_df, "Model").astype(str).eq(model_name)].copy()
        split_labels = sorted(_series_or_empty(model_slice, "Training Source").dropna().astype(str).unique().tolist(), key=env_sort_key)
        for split_label in split_labels:
            row: dict[str, object] = {"Model": model_name, "Training Source": split_label}
            split_slice = model_slice.loc[_series_or_empty(model_slice, "Training Source").astype(str).eq(split_label)].copy()
            for feature_set in feature_order:
                feature_slice = split_slice.loc[_series_or_empty(split_slice, "Feature Set").astype(str).eq(feature_set)]
                row[feature_set] = _format_ood_validation_cell(
                    _mean(_series_or_empty(feature_slice, "OOD AUROC")),
                    _stderr_from_series(_series_or_empty(feature_slice, "OOD AUROC")),
                    _mean(_series_or_empty(feature_slice, "Validation AUROC")),
                    _stderr_from_series(_series_or_empty(feature_slice, "Validation AUROC")),
                    style="markdown",
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_single_source_ood_by_environment_table(
    detail_metrics_df: pd.DataFrame,
    *,
    feature_order: list[str] | None = None,
) -> pd.DataFrame:
    feature_order = SCENARIO_FEATURE_ORDER if feature_order is None else feature_order
    if detail_metrics_df.empty:
        return pd.DataFrame(columns=["Model", "Evaluation Environment", *feature_order])

    rows: list[dict[str, object]] = []
    models = ordered_models_from_metrics(detail_metrics_df)
    environments = sorted(
        _series_or_empty(detail_metrics_df, "Evaluation Environment").dropna().astype(str).unique().tolist(),
        key=env_sort_key,
    )
    for model_name in models:
        model_slice = detail_metrics_df.loc[_series_or_empty(detail_metrics_df, "Model").astype(str).eq(model_name)].copy()
        for environment_name in environments:
            row: dict[str, object] = {"Model": model_name, "Evaluation Environment": environment_name}
            env_slice = model_slice.loc[
                _series_or_empty(model_slice, "Evaluation Environment").astype(str).eq(environment_name)
            ].copy()
            for feature_set in feature_order:
                feature_slice = env_slice.loc[_series_or_empty(env_slice, "Feature Set").astype(str).eq(feature_set)]
                row[feature_set] = _format_ood_validation_cell(
                    _mean(_series_or_empty(feature_slice, "OOD AUROC")),
                    _stderr_from_series(_series_or_empty(feature_slice, "OOD AUROC")),
                    _mean(_series_or_empty(feature_slice, "Validation AUROC")),
                    _stderr_from_series(_series_or_empty(feature_slice, "Validation AUROC")),
                    style="markdown",
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_holdout_split_detail_table(
    detail_metrics_df: pd.DataFrame,
    *,
    feature_order: list[str] | None = None,
) -> pd.DataFrame:
    feature_order = SCENARIO_FEATURE_ORDER if feature_order is None else feature_order
    if detail_metrics_df.empty:
        return pd.DataFrame(columns=["Model", "Training Split", "Held-Out Environment", *feature_order])

    rows: list[dict[str, object]] = []
    models = ordered_models_from_metrics(detail_metrics_df)
    for model_name in models:
        model_slice = detail_metrics_df.loc[_series_or_empty(detail_metrics_df, "Model").astype(str).eq(model_name)].copy()
        split_labels = sorted(
            _series_or_empty(model_slice, "Training Split").dropna().astype(str).unique().tolist(),
            key=lambda value: env_sort_key(value.replace("All except ", "")),
        )
        for split_label in split_labels:
            split_slice = model_slice.loc[_series_or_empty(model_slice, "Training Split").astype(str).eq(split_label)].copy()
            row: dict[str, object] = {
                "Model": model_name,
                "Training Split": split_label,
                "Held-Out Environment": _text_or_empty(_first_present(_series_or_empty(split_slice, "Held-Out Environment"))),
            }
            for feature_set in feature_order:
                feature_slice = split_slice.loc[_series_or_empty(split_slice, "Feature Set").astype(str).eq(feature_set)]
                row[feature_set] = _format_ood_validation_cell(
                    _mean(_series_or_empty(feature_slice, "OOD AUROC")),
                    _stderr_from_series(_series_or_empty(feature_slice, "OOD AUROC")),
                    _mean(_series_or_empty(feature_slice, "Validation AUROC")),
                    _stderr_from_series(_series_or_empty(feature_slice, "Validation AUROC")),
                    style="markdown",
                )
            rows.append(row)
    return pd.DataFrame(rows)


def selected_panel_rows(
    panel_df: pd.DataFrame,
    model_df: pd.DataFrame,
    *,
    target_name: str,
    requested_feature_size: int,
) -> pd.DataFrame:
    if panel_df.empty:
        return pd.DataFrame()

    panel_slice = panel_df.loc[
        _series_or_empty(panel_df, "target_name").astype(str).eq(target_name)
        & pd.to_numeric(_series_or_empty(panel_df, "requested_feature_size"), errors="coerce").eq(int(requested_feature_size))
    ].copy()
    if panel_slice.empty:
        return pd.DataFrame()

    join_cols = [
        "bundle_id",
        "Model",
        "scenario_name",
        "target_name",
        "requested_feature_size",
        "requested_feature_size_label",
        "feature_family_group",
    ]
    extra_model_cols = [
        "feature_space",
        "feature_space_title",
        "feature_size",
        "feature_size_label",
        "train_env",
        "source_envs",
        "source_env_count",
        "heldout_env",
        "source_val_auroc",
        "mean_ood_auroc",
        "min_ood_auroc",
        "std_ood_auroc",
        "selected_features_path",
        "coefficients_path",
    ]
    model_slice = model_df.loc[
        _series_or_empty(model_df, "target_name").astype(str).eq(target_name)
        & pd.to_numeric(_series_or_empty(model_df, "requested_feature_size"), errors="coerce").eq(int(requested_feature_size)),
        [column for column in join_cols + extra_model_cols if column in model_df.columns],
    ].copy()
    if not model_slice.empty:
        panel_slice = panel_slice.merge(
            model_slice,
            on=join_cols,
            how="left",
            validate="one_to_one",
            suffixes=("", "_best_model"),
        )

    panel_slice["Feature Family"] = panel_slice["feature_family_group"].map(FAMILY_DISPLAY).fillna(panel_slice["feature_family_group"])
    panel_slice = panel_slice.sort_values(
        ["Model", "feature_family_group"],
        key=lambda column: (
            column.map(lambda value: model_sort_key(str(value))[0])
            if column.name == "Model"
            else column.map(lambda value: family_sort_key(str(value))[0])
        ),
    ).reset_index(drop=True)
    return panel_slice


def build_family_summary_table(
    panel_df: pd.DataFrame,
    model_df: pd.DataFrame,
    *,
    target_name: str,
    requested_feature_size: int,
    scenario_name: str,
) -> pd.DataFrame:
    selected_df = selected_panel_rows(
        panel_df,
        model_df,
        target_name=target_name,
        requested_feature_size=requested_feature_size,
    )
    if selected_df.empty:
        return pd.DataFrame()

    training_column = "Training Source(s)"
    if scenario_name == "holdout_env_ood":
        training_column = "Training Split"

    rows: list[dict[str, object]] = []
    for (model_name, family_name), family_df in selected_df.groupby(["Model", "feature_family_group"], dropna=False, sort=False):
        rows.append(
            {
                "Model": model_name,
                "Feature Family": FAMILY_DISPLAY.get(str(family_name), str(family_name)),
                "Selected Feature Set": join_unique_text(_series_or_empty(family_df, "selected_feature_space_title")),
                training_column: join_unique_text(_series_or_empty(family_df, "train_env")),
                "Validation AUROC": format_mean_se(_series_or_empty(family_df, "source_val_auroc")),
                "Mean OOD AUROC": format_mean_se(_series_or_empty(family_df, "mean_ood_auroc")),
                "Min OOD AUROC": format_mean_se(_series_or_empty(family_df, "min_ood_auroc")),
                "_model_sort": model_sort_key(str(model_name))[0],
                "_family_sort": family_sort_key(str(family_name))[0],
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["_model_sort", "_family_sort", "Model", "Feature Family"]).drop(
        columns=["_model_sort", "_family_sort"]
    ).reset_index(drop=True)


def build_environment_summary_table(
    panel_df: pd.DataFrame,
    model_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    target_name: str,
    requested_feature_size: int,
    scenario_name: str,
) -> pd.DataFrame:
    selected_df = selected_panel_rows(
        panel_df,
        model_df,
        target_name=target_name,
        requested_feature_size=requested_feature_size,
    )
    if selected_df.empty or metrics_df.empty:
        return pd.DataFrame()

    training_column = "Training Source(s)"
    environment_column = "Environment"
    if scenario_name == "holdout_env_ood":
        training_column = "Training Split"
        environment_column = "Held-Out Environment"

    metric_rows: list[dict[str, object]] = []
    for row in selected_df.itertuples(index=False):
        feature_space = getattr(row, "feature_space", getattr(row, "selected_feature_space", None))
        feature_size_label = getattr(row, "feature_size_label", getattr(row, "source_feature_size_label", None))
        train_env = getattr(row, "train_env", None)
        if feature_space is None or feature_size_label is None or train_env is None or pd.isna(train_env):
            continue
        subset = metrics_df.loc[
            _series_or_empty(metrics_df, "bundle_id").astype(str).eq(str(row.bundle_id))
            & _series_or_empty(metrics_df, "Model").astype(str).eq(str(row.Model))
            & _series_or_empty(metrics_df, "target_name").astype(str).eq(str(row.target_name))
            & _series_or_empty(metrics_df, "feature_space").astype(str).eq(str(feature_space))
            & _series_or_empty(metrics_df, "feature_size_label").astype(str).eq(str(feature_size_label))
            & _series_or_empty(metrics_df, "train_env").astype(str).eq(str(train_env))
        ].copy()
        if subset.empty:
            continue
        val_df = subset.loc[_series_or_empty(subset, "eval_role").astype(str).eq("val")].copy()
        ood_df = subset.loc[_series_or_empty(subset, "eval_role").astype(str).eq("ood")].copy()
        source_val_auroc = _mean(_series_or_empty(val_df, "auroc")) if not val_df.empty else float("nan")
        for environment, env_df in ood_df.groupby("test_env", dropna=False, sort=False):
            metric_rows.append(
                {
                    "Model": row.Model,
                    "feature_family_group": row.feature_family_group,
                    "Feature Family": FAMILY_DISPLAY.get(str(row.feature_family_group), str(row.feature_family_group)),
                    environment_column: str(environment),
                    "Selected Feature Set": getattr(
                        row,
                        "selected_feature_space_title",
                        getattr(row, "feature_space_title", str(feature_space)),
                    ),
                    training_column: str(train_env),
                    "validation_auroc": source_val_auroc,
                    "ood_auroc": _mean(_series_or_empty(env_df, "auroc")),
                }
            )
    raw_df = pd.DataFrame(metric_rows)
    if raw_df.empty:
        return raw_df

    rows: list[dict[str, object]] = []
    for (model_name, family_name, environment), env_df in raw_df.groupby(
        ["Model", "feature_family_group", environment_column],
        dropna=False,
        sort=False,
    ):
        rows.append(
            {
                "Model": model_name,
                "Feature Family": FAMILY_DISPLAY.get(str(family_name), str(family_name)),
                environment_column: environment,
                "Selected Feature Set": join_unique_text(_series_or_empty(env_df, "Selected Feature Set")),
                training_column: join_unique_text(_series_or_empty(env_df, training_column)),
                "Validation AUROC": format_mean_se(_series_or_empty(env_df, "validation_auroc")),
                "OOD AUROC": format_mean_se(_series_or_empty(env_df, "ood_auroc")),
                "_model_sort": model_sort_key(str(model_name))[0],
                "_family_sort": family_sort_key(str(family_name))[0],
                "_env_sort": env_sort_key(str(environment))[0],
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["_model_sort", "_family_sort", "_env_sort", environment_column]).drop(
        columns=["_model_sort", "_family_sort", "_env_sort"]
    ).reset_index(drop=True)


def build_transfer_matrix_for_row(
    row: pd.Series,
    metrics_df: pd.DataFrame,
    *,
    train_axis_labels: list[str],
) -> pd.DataFrame:
    feature_space = row.get("feature_space", row.get("selected_feature_space"))
    feature_size_label = row.get("feature_size_label", row.get("source_feature_size_label"))
    train_env = row.get("train_env")
    subset = metrics_df.loc[
        _series_or_empty(metrics_df, "bundle_id").astype(str).eq(str(row["bundle_id"]))
        & _series_or_empty(metrics_df, "Model").astype(str).eq(str(row["Model"]))
        & _series_or_empty(metrics_df, "target_name").astype(str).eq(str(row["target_name"]))
        & _series_or_empty(metrics_df, "feature_space").astype(str).eq(str(feature_space))
        & _series_or_empty(metrics_df, "feature_size_label").astype(str).eq(str(feature_size_label))
        & _series_or_empty(metrics_df, "train_env").astype(str).eq(str(train_env))
    ].copy()
    matrix_df = pd.DataFrame(index=train_axis_labels, columns=base_tables.ENV_ORDER, dtype=float)
    for metric_row in subset.itertuples(index=False):
        matrix_df.loc[str(metric_row.train_env), str(metric_row.test_env)] = (
            float(metric_row.auroc) if pd.notna(metric_row.auroc) else float("nan")
        )
    return matrix_df


def plot_selected_family_transfer_panels(
    panel_df: pd.DataFrame,
    model_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    model_name: str,
    target_name: str,
    target_title: str,
    requested_feature_size: int,
    scenario_name: str,
    save_heatmaps: bool,
    heatmap_export_dir: Path,
) -> None:
    selected_df = selected_panel_rows(
        panel_df,
        model_df,
        target_name=target_name,
        requested_feature_size=requested_feature_size,
    )
    selected_df = selected_df.loc[_series_or_empty(selected_df, "Model").astype(str).eq(model_name)].copy()
    if selected_df.empty:
        return
    selected_df = selected_df.sort_values(
        "feature_family_group",
        key=lambda column: column.map(lambda value: family_sort_key(str(value))[0]),
    )

    train_axis_labels = train_axis_labels_for_scenario(scenario_name)
    y_axis_label = "Training source(s)" if scenario_name == "single_source_ood" else "Training split"

    matrices: dict[str, pd.DataFrame] = {}
    finite_values: list[np.ndarray] = []
    for family_name in FAMILY_ORDER:
        family_rows = selected_df.loc[_series_or_empty(selected_df, "feature_family_group").astype(str).eq(family_name)]
        if family_rows.empty:
            continue
        row = family_rows.iloc[0]
        matrix_df = build_transfer_matrix_for_row(
            row,
            metrics_df,
            train_axis_labels=train_axis_labels,
        )
        matrices[family_name] = matrix_df
        finite_values.append(matrix_df.to_numpy(dtype=float).ravel())

    flattened = np.concatenate(finite_values) if finite_values else np.array([], dtype=float)
    flattened = flattened[np.isfinite(flattened)]
    vmin = float(np.min(flattened)) if flattened.size else 0.0
    vmax = float(np.max(flattened)) if flattened.size else 1.0

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 2)
    image = None

    for idx, family_name in enumerate(FAMILY_ORDER):
        ax = axes.flat[idx]
        family_rows = selected_df.loc[_series_or_empty(selected_df, "feature_family_group").astype(str).eq(family_name)]
        if family_rows.empty or family_name not in matrices:
            ax.axis("off")
            continue
        row = family_rows.iloc[0]
        matrix_df = matrices[family_name].reindex(index=train_axis_labels, columns=base_tables.ENV_ORDER)
        matrix = np.ma.masked_invalid(matrix_df.to_numpy(dtype=float))
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(base_tables.ENV_ORDER)))
        ax.set_xticklabels(base_tables.ENV_ORDER, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(train_axis_labels)))
        ax.set_yticklabels(train_axis_labels)
        ax.set_xlabel("Evaluation env")
        ax.set_ylabel(y_axis_label)
        ax.set_title(
            f"{FAMILY_DISPLAY.get(family_name, family_name)}\n"
            f"{row['selected_feature_space_title']}\n"
            f"train = {row['train_env']}",
            fontsize=10.0,
        )
        midpoint = (vmin + vmax) / 2.0 if np.isfinite(vmin) and np.isfinite(vmax) else 0.5
        for row_idx in range(matrix_df.shape[0]):
            for col_idx in range(matrix_df.shape[1]):
                value = matrix_df.iat[row_idx, col_idx]
                text = "nan" if not np.isfinite(value) else f"{value:.3f}"
                text_color = "white" if np.isfinite(value) and value < midpoint else "black"
                ax.text(col_idx, row_idx, text, ha="center", va="center", color=text_color, fontsize=8.0)

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="AUROC")

    scenario_note = (
        "diagonal = source validation AUROC; off-diagonal = OOD AUROC"
        if scenario_name == "single_source_ood"
        else "source-env columns show validation AUROC; held-out column shows OOD AUROC"
    )
    fig.suptitle(
        f"{model_name} | {target_title} | PCA size {requested_feature_size}\n{scenario_note}",
        fontsize=14,
    )

    if save_heatmaps:
        heatmap_export_dir.mkdir(parents=True, exist_ok=True)
        out_path = heatmap_export_dir / (
            f"{slugify(model_name)}__{target_name}__k{int(requested_feature_size):03d}__family_transfer_heatmaps.png"
        )
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        md(f"Saved figure: `{out_path}`")
    plt.show()
    plt.close(fig)


def render_scenario_notebook(
    *,
    scenario_name: str,
    scenario_title: str | None = None,
    requested_feature_sizes: list[int] | None = None,
    results_root: Path | None = None,
    show_heatmaps: bool = True,
    save_heatmaps: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolved_results_root = (
        Path(os.environ.get("OOD_MAIN3_SCENARIO_RESULTS_ROOT", str(DEFAULT_RESULTS_ROOT))).expanduser().resolve()
        if results_root is None
        else Path(results_root).expanduser().resolve()
    )
    resolved_scenario_title = scenario_title or SCENARIO_TITLE_OVERRIDES.get(scenario_name, scenario_name)
    requested_feature_sizes = [64, 128, 256] if requested_feature_sizes is None else list(requested_feature_sizes)
    heatmap_export_dir = resolved_results_root / "notebook_exports" / scenario_name

    inventory_df, panel_df, model_df, metrics_df = load_scenario_bundle_frames(
        resolved_results_root,
        scenario_name,
    )

    if inventory_df.empty:
        md(f"_No populated bundle directories were found for scenario `{scenario_name}` under `{resolved_results_root}`._")
        return inventory_df, panel_df, model_df, metrics_df

    available_sizes = sorted(
        set(pd.to_numeric(_series_or_empty(panel_df, "requested_feature_size"), errors="coerce").dropna().astype(int).tolist())
    )
    selected_sizes = [size for size in requested_feature_sizes if size in available_sizes] or available_sizes
    available_target_rows = target_rows(metrics_df)

    md("## Bundle Inventory")
    display(inventory_df)

    md("## Scenario Summary")
    md(
        f"- Scenario: `{scenario_name}`\n"
        f"- Description: {resolved_scenario_title}\n"
        f"- Results root: `{resolved_results_root}`\n"
        f"- Requested PCA sizes present: {', '.join(str(size) for size in selected_sizes) if selected_sizes else 'none'}"
    )

    if not available_target_rows:
        md("_No targets were available after filtering the saved metrics._")
        return inventory_df, panel_df, model_df, metrics_df

    for target_name, target_title in available_target_rows:
        md(f"## {target_title}")
        target_has_any_rows = False
        for requested_feature_size in selected_sizes:
            split_metrics_df, detail_metrics_df = build_feature_metric_rows(
                metrics_df,
                scenario_name=scenario_name,
                target_name=target_name,
                requested_feature_size=requested_feature_size,
            )
            if split_metrics_df.empty:
                continue

            target_has_any_rows = True
            md(f"### PCA size {requested_feature_size}")
            if scenario_name == "holdout_env_ood":
                md(
                    "_Previous heatmaps showed many `nan` cells because each training split has only one OOD destination. "
                    "The tables below use the saved split-level metrics directly: weighted validation AUROC across the four source environments, "
                    "and OOD AUROC on the held-out environment._"
                )
            else:
                md(
                    "_The tables below use explicit feature-set metrics directly instead of selecting one best family first. "
                    "Aggregated tables average over source environments; detail tables break results out by source or evaluation environment._"
                )

            aggregated_ood_stats_df = _build_combined_metric_stats_table(
                split_metrics_df,
                ood_column="OOD AUROC",
                validation_column="Validation AUROC",
            )
            aggregated_markdown_df = build_transposed_combined_metric_table(
                aggregated_ood_stats_df,
                style="markdown",
                feature_order=SCENARIO_FEATURE_ORDER,
                bold_best=True,
            )
            aggregated_latex_df = build_transposed_combined_metric_table(
                aggregated_ood_stats_df,
                style="latex",
                feature_order=SCENARIO_FEATURE_ORDER,
                bold_best=True,
            )
            panel_title = PANEL_TITLE_OVERRIDES.get(target_name, target_title)

            md("#### 1. Aggregated AUROC Table (Markdown)")
            md(render_markdown_feature_table(aggregated_markdown_df))

            md("#### 2. Aggregated AUROC Table (LaTeX)")
            _display_latex_block(
                render_latex_feature_table(
                    aggregated_latex_df,
                    panel_title=f"{panel_title} | PCA size {requested_feature_size}",
                    feature_order=SCENARIO_FEATURE_ORDER,
                    feature_group_specs=FEATURE_GROUP_SPECS,
                )
            )

            if scenario_name == "holdout_env_ood":
                holdout_detail_df = build_holdout_split_detail_table(detail_metrics_df)
                holdout_detail_df = transpose_detail_table(
                    holdout_detail_df,
                    id_columns=["Training Split", "Held-Out Environment"],
                    feature_order=SCENARIO_FEATURE_ORDER,
                )
                md("#### 3. Split-Level AUROC Table (Markdown)")
                md(render_simple_markdown_table(_detail_table_for_style(holdout_detail_df, style="markdown")))
                md("#### 4. Split-Level AUROC Table (LaTeX)")
                _display_latex_block(
                    render_simple_latex_table(_detail_table_for_style(holdout_detail_df, style="latex"))
                )
            else:
                validation_detail_df = build_single_source_validation_detail_table(split_metrics_df)
                validation_detail_df = transpose_detail_table(
                    validation_detail_df,
                    id_columns=["Training Source"],
                    feature_order=SCENARIO_FEATURE_ORDER,
                )
                ood_detail_df = build_single_source_ood_by_environment_table(detail_metrics_df)
                ood_detail_df = transpose_detail_table(
                    ood_detail_df,
                    id_columns=["Evaluation Environment"],
                    feature_order=SCENARIO_FEATURE_ORDER,
                )
                md("#### 3. AUROC By Training Source (Markdown)")
                md(render_simple_markdown_table(_detail_table_for_style(validation_detail_df, style="markdown")))
                md("#### 4. AUROC By Training Source (LaTeX)")
                _display_latex_block(
                    render_simple_latex_table(_detail_table_for_style(validation_detail_df, style="latex"))
                )
                md("#### 5. AUROC By Evaluation Environment (Markdown)")
                md(render_simple_markdown_table(_detail_table_for_style(ood_detail_df, style="markdown")))
                md("#### 6. AUROC By Evaluation Environment (LaTeX)")
                _display_latex_block(
                    render_simple_latex_table(_detail_table_for_style(ood_detail_df, style="latex"))
                )

            if show_heatmaps and not panel_df.empty:
                md("_Legacy family-selection heatmaps are omitted here because they are sparse and can be misleading for these scenario-specific outputs._")

        attention_panel_title = PANEL_TITLE_OVERRIDES.get(target_name, target_title)
        attention_requested_feature_size = selected_sizes[0] if selected_sizes else None
        if attention_requested_feature_size is not None:
            attention_split_metrics_df, attention_detail_metrics_df = build_feature_metric_rows(
                metrics_df,
                scenario_name=scenario_name,
                target_name=target_name,
                requested_feature_size=attention_requested_feature_size,
                feature_order=ATTENTION_ABLATION_FEATURE_ORDER,
            )
            if not attention_split_metrics_df.empty:
                target_has_any_rows = True
                md("### Attention-Only Feature Family Ablation")
                md(
                    "_These attention-only subsets do not depend on activation PCA size, so this ablation is shown once per target. "
                    "It compares full attention features against grounding, concentration, grounding-transition, and concentration-transition subsets._"
                )

                attention_aggregated_ood_stats_df = _build_combined_metric_stats_table(
                    attention_split_metrics_df,
                    ood_column="OOD AUROC",
                    validation_column="Validation AUROC",
                    feature_order=ATTENTION_ABLATION_FEATURE_ORDER,
                )
                attention_aggregated_markdown_df = build_transposed_combined_metric_table(
                    attention_aggregated_ood_stats_df,
                    style="markdown",
                    feature_order=ATTENTION_ABLATION_FEATURE_ORDER,
                    bold_best=True,
                )
                attention_aggregated_latex_df = build_transposed_combined_metric_table(
                    attention_aggregated_ood_stats_df,
                    style="latex",
                    feature_order=ATTENTION_ABLATION_FEATURE_ORDER,
                    bold_best=True,
                )

                md("#### 1. Attention-Only Aggregated AUROC Table (Markdown)")
                md(render_markdown_feature_table(attention_aggregated_markdown_df))

                md("#### 2. Attention-Only Aggregated AUROC Table (LaTeX)")
                _display_latex_block(
                    render_latex_feature_table(
                        attention_aggregated_latex_df,
                        panel_title=f"{attention_panel_title} | attention-only ablations",
                        feature_order=ATTENTION_ABLATION_FEATURE_ORDER,
                        feature_group_specs=ATTENTION_ABLATION_GROUP_SPECS,
                    )
                )

                if scenario_name == "holdout_env_ood":
                    attention_holdout_detail_df = build_holdout_split_detail_table(
                        attention_detail_metrics_df,
                        feature_order=ATTENTION_ABLATION_FEATURE_ORDER,
                    )
                    attention_holdout_detail_df = transpose_detail_table(
                        attention_holdout_detail_df,
                        id_columns=["Training Split", "Held-Out Environment"],
                        feature_order=ATTENTION_ABLATION_FEATURE_ORDER,
                    )
                    md("#### 3. Attention-Only Split-Level AUROC Table (Markdown)")
                    md(render_simple_markdown_table(_detail_table_for_style(attention_holdout_detail_df, style="markdown")))
                    md("#### 4. Attention-Only Split-Level AUROC Table (LaTeX)")
                    _display_latex_block(
                        render_simple_latex_table(_detail_table_for_style(attention_holdout_detail_df, style="latex"))
                    )
                else:
                    attention_validation_detail_df = build_single_source_validation_detail_table(
                        attention_split_metrics_df,
                        feature_order=ATTENTION_ABLATION_FEATURE_ORDER,
                    )
                    attention_validation_detail_df = transpose_detail_table(
                        attention_validation_detail_df,
                        id_columns=["Training Source"],
                        feature_order=ATTENTION_ABLATION_FEATURE_ORDER,
                    )
                    attention_ood_detail_df = build_single_source_ood_by_environment_table(
                        attention_detail_metrics_df,
                        feature_order=ATTENTION_ABLATION_FEATURE_ORDER,
                    )
                    attention_ood_detail_df = transpose_detail_table(
                        attention_ood_detail_df,
                        id_columns=["Evaluation Environment"],
                        feature_order=ATTENTION_ABLATION_FEATURE_ORDER,
                    )
                    md("#### 3. Attention-Only AUROC By Training Source (Markdown)")
                    md(render_simple_markdown_table(_detail_table_for_style(attention_validation_detail_df, style="markdown")))
                    md("#### 4. Attention-Only AUROC By Training Source (LaTeX)")
                    _display_latex_block(
                        render_simple_latex_table(_detail_table_for_style(attention_validation_detail_df, style="latex"))
                    )
                    md("#### 5. Attention-Only AUROC By Evaluation Environment (Markdown)")
                    md(render_simple_markdown_table(_detail_table_for_style(attention_ood_detail_df, style="markdown")))
                    md("#### 6. Attention-Only AUROC By Evaluation Environment (LaTeX)")
                    _display_latex_block(
                        render_simple_latex_table(_detail_table_for_style(attention_ood_detail_df, style="latex"))
                    )

        if not target_has_any_rows:
            md("_No rows were available for this target in the discovered bundles._")

    return inventory_df, panel_df, model_df, metrics_df
