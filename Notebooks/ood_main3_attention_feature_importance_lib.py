from __future__ import annotations

import math
import os
import re
import sys
import textwrap
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
import ood_main3_consistency_scenario_tables_lib as scenario_tables


DEFAULT_RESULTS_ROOT = scenario_tables.DEFAULT_RESULTS_ROOT
DEFAULT_EXPORT_ROOT = DEFAULT_RESULTS_ROOT / "notebook_exports" / "attention_feature_importance"

SCENARIO_LAYOUT = [
    (
        "single_source_ood",
        "1. Single Source",
        "Train on 1 environment and average feature importance across the five source-specific models for each LLM.",
    ),
    (
        "holdout_env_ood",
        "2. Multi-source",
        "Train on 4 environments, hold one environment out, and average feature importance across the five held-out splits for each LLM.",
    ),
]

GROUP_ORDER = ["grounding", "concentration", "grounding_transition", "concentration_transition"]
GROUP_DISPLAY = {
    "grounding": "Grounding",
    "concentration": "Concentration",
    "grounding_transition": "Grounding transition",
    "concentration_transition": "Concentration transition",
}
GROUP_COLORS = {
    "grounding": "#1d4ed8",
    "concentration": "#d97706",
    "grounding_transition": "#0f766e",
    "concentration_transition": "#b91c1c",
}

BAND_ORDER = ["early", "mid", "late"]
BAND_DISPLAY = {"early": "Early", "mid": "Mid", "late": "Late"}
HEAD_SUMMARY_DISPLAY = {"mean": "Head Mean", "std": "Head Std", "max": "Head Max"}
BAND_STAT_DISPLAY = {"mean": "Mean", "std": "Std", "min": "Min", "max": "Max"}
PANEL_TITLE_FONTSIZE = 12.5
PANEL_TITLE_PAD = 9
PAPER_METRIC_DISPLAY = {
    "current_vs_prev": "Local grounding",
    "current_vs_prior": "History grounding",
    "recent_vs_early": "Recency bias",
    "prev_share_of_prior": "Previous-sentence share",
    "entropy_prior": "Prior entropy",
    "entropy_full": "Full entropy",
    "top1_prior": "Top-1 prior mass",
    "top5_prior": "Top-5 prior mass",
    "herfindahl_prior": "Prior Herfindahl",
    "effective_support_prior": "Prior effective support",
    "current_share_total": "Current-sentence share",
}
TRANSITION_DISPLAY = {
    "delta": "Delta",
    "slope3": "Slope",
    "devrun": "Running Deviation",
    "min_gap": "Min Gap",
    "max_gap": "Max Gap",
}

PLOT_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#111827",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#111827",
    "xtick.color": "#111827",
    "ytick.color": "#111827",
    "text.color": "#111827",
    "axes.titleweight": "semibold",
    "axes.grid": False,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

RESULTS_MARKER = "/deception2/Results/"
LOCAL_RESULTS_PREFIX = "/playpen-ssd/smerrill/deception2/Results/"
FIXED_ATTENTION_FEATURE_SPACE = "attention_only"
TRUTHFUL_TARGET_NAME = "delta_neg_lt_neg_0_3"
DECEPTIVE_TARGET_NAME = "delta_pos_gt_0_3"
COMBINED_TARGET_NAME = "truthful_deceptive_average"
COMBINED_TARGET_NAMES = (DECEPTIVE_TARGET_NAME, TRUTHFUL_TARGET_NAME)
COMBINED_TARGET_TITLE = "Average of truthful and deceptive commitment"


pd.options.display.max_columns = 200


def _is_notebook_shell() -> bool:
    if get_ipython is None or get_ipython() is None:
        return False
    return get_ipython().__class__.__name__ == "ZMQInteractiveShell"


def md(text: str) -> None:
    if Markdown is not None and _is_notebook_shell():
        display(Markdown(text))
    else:
        print(text)


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


def _stderr(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return float("nan")
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_numeric = pd.to_numeric(values, errors="coerce")
    weight_numeric = pd.to_numeric(weights, errors="coerce")
    keep_mask = value_numeric.notna() & weight_numeric.notna() & weight_numeric.gt(0)
    if not keep_mask.any():
        clean = value_numeric.dropna()
        return float(clean.mean()) if not clean.empty else float("nan")
    return float(np.average(value_numeric.loc[keep_mask], weights=weight_numeric.loc[keep_mask]))


def _format_pm(mean_value: Any, se_value: Any) -> str:
    mean_numeric = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
    if pd.isna(mean_numeric):
        return ""
    se_numeric = pd.to_numeric(pd.Series([se_value]), errors="coerce").iloc[0]
    if pd.isna(se_numeric):
        return f"{float(mean_numeric):.3f}"
    return f"{float(mean_numeric):.3f} +/- {float(se_numeric):.3f}"


def _format_pct(value: Any, digits: int = 1) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return f"{100.0 * float(numeric):.{digits}f}%"


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _display_table(df: pd.DataFrame) -> None:
    if df.empty:
        md("_No rows available._")
        return
    if _is_notebook_shell():
        display(df)
    else:
        print(df.to_string(index=False))


def resolve_saved_path(raw_path: object) -> Path:
    path_text = _text_or_empty(raw_path)
    if not path_text:
        return Path("")
    path = Path(path_text)
    if path.exists():
        return path
    if RESULTS_MARKER in path_text:
        suffix = path_text.split(RESULTS_MARKER, 1)[1]
        candidate = Path(LOCAL_RESULTS_PREFIX) / suffix
        if candidate.exists():
            return candidate
    return path


def humanize_token(value: object) -> str:
    text = _text_or_empty(value)
    if not text:
        return ""
    text = text.replace("__pca", " PCA")
    text = text.replace("devrun_", "")
    text = text.replace("_", " ")
    return text


def _title_token(value: object) -> str:
    text = humanize_token(value)
    if not text:
        return ""
    return " ".join(token[:1].upper() + token[1:] for token in text.split())


def feature_display_name(row: pd.Series) -> str:
    metric_key = _text_or_empty(row.get("metric_name"))
    transition_key = _text_or_empty(row.get("transition_prefix"))
    metric_label = PAPER_METRIC_DISPLAY.get(metric_key, _title_token(metric_key))
    transition_label = TRANSITION_DISPLAY.get(transition_key, _title_token(transition_key))
    if metric_label and transition_label:
        return f"{transition_label}({metric_label})"
    if metric_label:
        return metric_label
    if transition_label:
        return transition_label
    feature = _text_or_empty(row.get("feature"))
    if not feature:
        return ""
    return textwrap.shorten(feature.replace("__", " | ").replace("_", " "), width=58, placeholder="...")


def feature_context_label(row: pd.Series) -> str:
    head_summary = HEAD_SUMMARY_DISPLAY.get(_text_or_empty(row.get("head_summary")), _title_token(row.get("head_summary")))
    band = BAND_DISPLAY.get(_text_or_empty(row.get("band")), _title_token(row.get("band")))
    band_stat = BAND_STAT_DISPLAY.get(_text_or_empty(row.get("band_stat")), _title_token(row.get("band_stat")))
    layer_stat = ""
    if band and band_stat:
        layer_stat = f"{band} Layer {band_stat}"
    elif band:
        layer_stat = band
    elif band_stat:
        layer_stat = band_stat
    parts = [part for part in [head_summary, layer_stat] if part]
    return " | ".join(parts)


def compact_feature_label(row: pd.Series) -> str:
    feature_label = feature_display_name(row)
    context_label = feature_context_label(row)
    if feature_label and context_label:
        return f"{feature_label}\n{context_label}"
    if feature_label:
        return feature_label
    feature = _text_or_empty(row.get("feature"))
    if not feature:
        return ""
    return textwrap.shorten(feature.replace("__", " | ").replace("_", " "), width=58, placeholder="...")


def target_title(target_name: str) -> str:
    if target_name == COMBINED_TARGET_NAME:
        return COMBINED_TARGET_TITLE
    return scenario_tables.PANEL_TITLE_OVERRIDES.get(
        target_name,
        base_tables.TARGET_TITLE_OVERRIDES.get(target_name, target_name),
    )


def compact_attention_space_label(value: object) -> str:
    text = _text_or_empty(value)
    if not text:
        return ""
    for prefix in ["Attention only | ", "Attention only: "]:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    if text == "Attention only":
        return ""
    return text


def selection_heading_label(selection_row: pd.Series) -> str:
    model_name = _text_or_empty(selection_row.get("Model"))
    suffix = compact_attention_space_label(selection_row.get("feature_space_title"))
    if model_name and suffix:
        return f"{model_name} | {suffix}"
    return model_name or suffix


def _run_level_attention_metrics(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for train_env, split_df in feature_df.groupby("train_env", dropna=False, sort=False):
        val_df = split_df.loc[_series_or_empty(split_df, "eval_role").astype(str).eq("val")].copy()
        ood_df = split_df.loc[_series_or_empty(split_df, "eval_role").astype(str).eq("ood")].copy()
        rows.append(
            {
                "train_env": _text_or_empty(train_env),
                "heldout_env": _text_or_empty(_first_present(_series_or_empty(split_df, "heldout_env"))),
                "validation_auroc": _weighted_mean(_series_or_empty(val_df, "auroc"), _series_or_empty(val_df, "n_rows")),
                "ood_auroc": pd.to_numeric(_series_or_empty(ood_df, "auroc"), errors="coerce").dropna().mean(),
                "selected_features_path": _text_or_empty(_first_present(_series_or_empty(split_df, "selected_features_path"))),
                "coefficients_path": _text_or_empty(_first_present(_series_or_empty(split_df, "coefficients_path"))),
                "feature_space_title": _text_or_empty(_first_present(_series_or_empty(split_df, "feature_space_title"))),
                "selected_feature_count": pd.to_numeric(
                    pd.Series([_first_present(_series_or_empty(split_df, "selected_feature_count"))]),
                    errors="coerce",
                ).iloc[0],
            }
        )
    return pd.DataFrame(rows)


def select_attention_only_feature_space(metrics_df: pd.DataFrame, *, target_name: str) -> pd.DataFrame:
    target_slice = metrics_df.loc[
        _series_or_empty(metrics_df, "target_name").astype(str).eq(target_name)
        & _series_or_empty(metrics_df, "feature_family_group").astype(str).eq("attention_only")
        & _series_or_empty(metrics_df, "feature_space").astype(str).eq(FIXED_ATTENTION_FEATURE_SPACE)
    ].copy()
    if target_slice.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_cols = ["bundle_id", "bundle_name", "Model", "target_name", "feature_space", "feature_space_title", "feature_set"]
    for keys, feature_df in target_slice.groupby(group_cols, dropna=False, sort=False):
        run_df = _run_level_attention_metrics(feature_df)
        if run_df.empty:
            continue
        bundle_id, bundle_name, model_name, target_key, feature_space, feature_space_title, feature_set = keys
        rows.append(
            {
                "bundle_id": _text_or_empty(bundle_id),
                "bundle_name": _text_or_empty(bundle_name),
                "Model": _text_or_empty(model_name),
                "target_name": _text_or_empty(target_key),
                "feature_space": _text_or_empty(feature_space),
                "feature_space_title": _text_or_empty(feature_space_title),
                "feature_set": _text_or_empty(feature_set),
                "mean_ood_auroc": pd.to_numeric(run_df["ood_auroc"], errors="coerce").dropna().mean(),
                "ood_auroc_se": _stderr(run_df["ood_auroc"]),
                "mean_validation_auroc": pd.to_numeric(run_df["validation_auroc"], errors="coerce").dropna().mean(),
                "validation_auroc_se": _stderr(run_df["validation_auroc"]),
                "selected_feature_count": pd.to_numeric(run_df["selected_feature_count"], errors="coerce").dropna().max(),
                "training_splits": int(len(run_df)),
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    summary_df = summary_df.sort_values("Model", key=lambda column: column.map(scenario_tables.model_sort_key)).reset_index(drop=True)
    return summary_df


def load_attention_importance_long_df(metrics_df: pd.DataFrame, selection_row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = metrics_df.loc[
        _series_or_empty(metrics_df, "bundle_id").astype(str).eq(_text_or_empty(selection_row.get("bundle_id")))
        & _series_or_empty(metrics_df, "Model").astype(str).eq(_text_or_empty(selection_row.get("Model")))
        & _series_or_empty(metrics_df, "target_name").astype(str).eq(_text_or_empty(selection_row.get("target_name")))
        & _series_or_empty(metrics_df, "feature_space").astype(str).eq(_text_or_empty(selection_row.get("feature_space")))
    ].copy()
    run_metrics_df = _run_level_attention_metrics(subset)
    if run_metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    metadata_cols = [
        "feature",
        "selected_rank",
        "global_rank",
        "consistency_score",
        "sign_direction",
        "feature_root",
        "family",
        "metric_name",
        "metric_group",
        "attention_feature_group",
        "transition_prefix",
        "head_summary",
        "band",
        "band_stat",
    ]

    merged_frames: list[pd.DataFrame] = []
    for run_row in run_metrics_df.itertuples(index=False):
        coefficients_path = resolve_saved_path(run_row.coefficients_path)
        selected_features_path = resolve_saved_path(run_row.selected_features_path)
        if not coefficients_path.exists():
            continue

        coef_df = pd.read_csv(coefficients_path)
        if "feature_weight" not in coef_df.columns and "coefficient" in coef_df.columns:
            coef_df["feature_weight"] = coef_df["coefficient"]
        if "abs_feature_weight" not in coef_df.columns:
            coef_df["abs_feature_weight"] = pd.to_numeric(coef_df.get("feature_weight"), errors="coerce").abs()

        if selected_features_path.exists():
            selected_df = pd.read_csv(selected_features_path)
            available_cols = [column for column in metadata_cols if column in selected_df.columns]
            if available_cols:
                coef_df = coef_df.merge(
                    selected_df.loc[:, available_cols].drop_duplicates(subset=["feature"]),
                    on="feature",
                    how="left",
                    validate="one_to_one",
                )

        total_weight = pd.to_numeric(coef_df["feature_weight"], errors="coerce").fillna(0.0).sum()
        if total_weight <= 0:
            total_weight = pd.to_numeric(coef_df["abs_feature_weight"], errors="coerce").fillna(0.0).sum()
            coef_df["importance_share"] = pd.to_numeric(coef_df["abs_feature_weight"], errors="coerce").fillna(0.0) / max(total_weight, 1e-12)
        else:
            coef_df["importance_share"] = pd.to_numeric(coef_df["feature_weight"], errors="coerce").fillna(0.0) / total_weight

        coef_df["run_id"] = _text_or_empty(run_row.train_env)
        coef_df["split_label"] = _text_or_empty(run_row.train_env)
        coef_df["heldout_env"] = _text_or_empty(run_row.heldout_env)
        coef_df["validation_auroc"] = run_row.validation_auroc
        coef_df["ood_auroc"] = run_row.ood_auroc
        merged_frames.append(coef_df)

    long_df = pd.concat(merged_frames, ignore_index=True) if merged_frames else pd.DataFrame()
    if long_df.empty:
        return long_df, run_metrics_df

    for column in metadata_cols:
        if column != "feature" and column not in long_df.columns:
            long_df[column] = ""
    long_df["attention_feature_group"] = _series_or_empty(long_df, "attention_feature_group").fillna("").astype(str)
    long_df["band"] = _series_or_empty(long_df, "band").fillna("").astype(str)
    long_df["band_stat"] = _series_or_empty(long_df, "band_stat").fillna("").astype(str)
    long_df["head_summary"] = _series_or_empty(long_df, "head_summary").fillna("").astype(str)
    long_df["metric_name"] = _series_or_empty(long_df, "metric_name").fillna("").astype(str)
    return long_df, run_metrics_df


def summarize_attention_feature_importance(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    run_ids = sorted(_series_or_empty(long_df, "run_id").dropna().astype(str).unique().tolist())
    if not run_ids:
        return pd.DataFrame()

    importance_matrix = (
        long_df.pivot_table(index="feature", columns="run_id", values="importance_share", aggfunc="sum", fill_value=0.0)
        .reindex(columns=run_ids, fill_value=0.0)
    )
    selected_matrix = (
        long_df.assign(selected_flag=1)
        .pivot_table(index="feature", columns="run_id", values="selected_flag", aggfunc="max", fill_value=0)
        .reindex(index=importance_matrix.index, columns=run_ids, fill_value=0)
    )

    n_runs = len(run_ids)
    mean_share = importance_matrix.mean(axis=1)
    std_share = importance_matrix.std(axis=1, ddof=1) if n_runs > 1 else pd.Series(0.0, index=importance_matrix.index)
    se_share = std_share / math.sqrt(n_runs) if n_runs > 1 else pd.Series(np.nan, index=importance_matrix.index)
    metadata_df = (
        long_df.sort_values(["feature", "selected_rank"], na_position="last", kind="mergesort")
        .drop_duplicates(subset=["feature"])
        .set_index("feature")
    )
    observed_rank_df = long_df.groupby("feature", as_index=True).agg(
        mean_selected_rank=("selected_rank", lambda values: pd.to_numeric(values, errors="coerce").dropna().mean()),
        mean_global_rank=("global_rank", lambda values: pd.to_numeric(values, errors="coerce").dropna().mean()),
        mean_consistency_score=("consistency_score", lambda values: pd.to_numeric(values, errors="coerce").dropna().mean()),
    )

    summary_df = pd.DataFrame(
        {
            "feature": importance_matrix.index,
            "mean_importance_share": mean_share.to_numpy(dtype=float),
            "std_importance_share": std_share.to_numpy(dtype=float),
            "se_importance_share": se_share.to_numpy(dtype=float),
            "selected_in_runs": selected_matrix.sum(axis=1).to_numpy(dtype=float),
            "selection_rate": (selected_matrix.sum(axis=1) / max(n_runs, 1)).to_numpy(dtype=float),
        }
    ).set_index("feature")
    summary_df = summary_df.join(observed_rank_df, how="left")
    summary_df = summary_df.join(
        metadata_df.loc[
            :,
            [
                column
                for column in [
                    "feature_root",
                    "family",
                    "metric_name",
                    "metric_group",
                    "attention_feature_group",
                    "transition_prefix",
                    "head_summary",
                    "band",
                    "band_stat",
                    "sign_direction",
                ]
                if column in metadata_df.columns
            ],
        ],
        how="left",
    ).reset_index()

    summary_df = summary_df.sort_values(
        ["mean_importance_share", "selection_rate", "mean_selected_rank", "feature"],
        ascending=[False, False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return summary_df


def build_group_band_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(index=GROUP_ORDER, columns=BAND_ORDER, dtype=float)

    run_ids = sorted(_series_or_empty(long_df, "run_id").dropna().astype(str).unique().tolist())
    index = pd.MultiIndex.from_product([run_ids, GROUP_ORDER, BAND_ORDER], names=["run_id", "attention_feature_group", "band"])
    grouped = (
        long_df.groupby(["run_id", "attention_feature_group", "band"], as_index=True)["importance_share"]
        .sum()
        .reindex(index, fill_value=0.0)
        .reset_index()
    )
    summary_df = (
        grouped.groupby(["attention_feature_group", "band"], as_index=False)
        .agg(mean_share=("importance_share", "mean"), se_share=("importance_share", _stderr))
    )
    matrix_df = (
        summary_df.pivot(index="attention_feature_group", columns="band", values="mean_share")
        .reindex(index=GROUP_ORDER, columns=BAND_ORDER)
        .fillna(0.0)
    )
    return matrix_df


def build_group_band_display_table(matrix_df: pd.DataFrame) -> pd.DataFrame:
    if matrix_df.empty:
        return pd.DataFrame()
    out = matrix_df.reindex(index=GROUP_ORDER, columns=BAND_ORDER).fillna(0.0).copy().reset_index()
    out = out.rename(columns={"attention_feature_group": "Attention Family"})
    out["Attention Family"] = out["Attention Family"].map(GROUP_DISPLAY).fillna(out["Attention Family"])
    for band in BAND_ORDER:
        out[BAND_DISPLAY[band]] = [_format_pct(value, digits=1) for value in pd.to_numeric(out[band], errors="coerce")]
    return out.loc[:, ["Attention Family"] + [BAND_DISPLAY[band] for band in BAND_ORDER]]


def build_display_selection_table(selection_df: pd.DataFrame) -> pd.DataFrame:
    if selection_df.empty:
        return pd.DataFrame()
    out = selection_df.loc[:, ["Model", "mean_ood_auroc", "ood_auroc_se", "mean_validation_auroc", "validation_auroc_se", "training_splits", "selected_feature_count"]].copy()
    out = out.rename(
        columns={
            "training_splits": "Training Splits",
            "selected_feature_count": "Selected Features",
        }
    )
    out["OOD AUROC"] = [_format_pm(mean_value, se_value) for mean_value, se_value in zip(out["mean_ood_auroc"], out["ood_auroc_se"], strict=False)]
    out["Validation AUROC"] = [
        _format_pm(mean_value, se_value)
        for mean_value, se_value in zip(out["mean_validation_auroc"], out["validation_auroc_se"], strict=False)
    ]
    out["Selected Features"] = pd.to_numeric(out["Selected Features"], errors="coerce").round().astype("Int64")
    out = out.drop(columns=["mean_ood_auroc", "ood_auroc_se", "mean_validation_auroc", "validation_auroc_se"])
    return out


def build_top_feature_table(summary_df: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    out = summary_df.head(top_k).copy()
    out["Mean Importance Share"] = [_format_pct(value, digits=2) for value in out["mean_importance_share"]]
    out["SE"] = [_format_pct(value, digits=2) for value in out["se_importance_share"]]
    out["Group"] = out["attention_feature_group"].map(GROUP_DISPLAY).fillna(out["attention_feature_group"])
    out["Band"] = out["band"].map(BAND_DISPLAY).fillna(out["band"])
    out["Stat"] = out["band_stat"].map(BAND_STAT_DISPLAY).fillna(out["band_stat"])
    out["Head Summary"] = out["head_summary"].map(HEAD_SUMMARY_DISPLAY).fillna(out["head_summary"])
    out["Feature"] = [feature_display_name(row) for _, row in out.iterrows()]
    out["Rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out.loc[:, ["Rank", "Feature", "Mean Importance Share", "SE", "Group", "Band", "Stat", "Head Summary"]]


def build_model_payloads(metrics_df: pd.DataFrame, *, target_name: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    selection_df = select_attention_only_feature_space(metrics_df, target_name=target_name)
    payloads: list[dict[str, object]] = []
    for row in selection_df.to_dict(orient="records"):
        selection_row = pd.Series(row)
        long_df, run_metrics_df = load_attention_importance_long_df(metrics_df, selection_row)
        feature_summary_df = summarize_attention_feature_importance(long_df)
        group_band_df = build_group_band_matrix(long_df)
        payloads.append(
            {
                "selection": selection_row,
                "run_metrics_df": run_metrics_df,
                "long_df": long_df,
                "feature_summary_df": feature_summary_df,
                "group_band_df": group_band_df,
            }
        )
    payloads.sort(key=lambda payload: scenario_tables.model_sort_key(_text_or_empty(payload["selection"].get("Model"))))
    return selection_df, payloads


def build_combined_target_payloads(
    scenario_results: dict[str, tuple[pd.DataFrame, list[dict[str, object]]]],
    *,
    target_names: tuple[str, ...],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if any(target_name not in scenario_results for target_name in target_names):
        return pd.DataFrame(), []

    payload_maps: dict[str, dict[str, dict[str, object]]] = {}
    model_sets: list[set[str]] = []
    for target_name in target_names:
        _, payloads = scenario_results[target_name]
        model_map = {
            _text_or_empty(payload["selection"].get("Model")): payload
            for payload in payloads
            if _text_or_empty(payload["selection"].get("Model"))
        }
        if not model_map:
            return pd.DataFrame(), []
        payload_maps[target_name] = model_map
        model_sets.append(set(model_map))

    common_models = sorted(set.intersection(*model_sets), key=scenario_tables.model_sort_key)
    rows: list[dict[str, object]] = []
    combined_payloads: list[dict[str, object]] = []
    for model_name in common_models:
        component_payloads = [(target_name, payload_maps[target_name][model_name]) for target_name in target_names]
        long_frames: list[pd.DataFrame] = []
        run_metric_frames: list[pd.DataFrame] = []
        ood_values: list[float] = []
        validation_values: list[float] = []
        selected_feature_counts: list[float] = []
        training_splits_total = 0

        for target_name, payload in component_payloads:
            selection_row = payload["selection"]
            ood_values.append(pd.to_numeric(pd.Series([selection_row.get("mean_ood_auroc")]), errors="coerce").iloc[0])
            validation_values.append(pd.to_numeric(pd.Series([selection_row.get("mean_validation_auroc")]), errors="coerce").iloc[0])
            selected_feature_counts.append(pd.to_numeric(pd.Series([selection_row.get("selected_feature_count")]), errors="coerce").iloc[0])
            training_splits_total += int(pd.to_numeric(pd.Series([selection_row.get("training_splits")]), errors="coerce").fillna(0).iloc[0])

            long_df = payload["long_df"].copy()
            if not long_df.empty:
                long_df["target_name"] = target_name
                long_df["run_id"] = f"{target_name}::" + _series_or_empty(long_df, "run_id").fillna("").astype(str)
                long_df["split_label"] = target_title(target_name) + " | " + _series_or_empty(long_df, "split_label").fillna("").astype(str)
                long_frames.append(long_df)

            run_metrics_df = payload["run_metrics_df"].copy()
            if not run_metrics_df.empty:
                run_metrics_df["target_name"] = target_name
                run_metrics_df["target_title"] = target_title(target_name)
                run_metrics_df["train_env"] = target_title(target_name) + " | " + _series_or_empty(run_metrics_df, "train_env").fillna("").astype(str)
                run_metric_frames.append(run_metrics_df)

        if not long_frames:
            continue

        combined_long_df = pd.concat(long_frames, ignore_index=True)
        combined_run_metrics_df = pd.concat(run_metric_frames, ignore_index=True) if run_metric_frames else pd.DataFrame()
        feature_summary_df = summarize_attention_feature_importance(combined_long_df)
        group_band_df = build_group_band_matrix(combined_long_df)

        ood_series = pd.to_numeric(pd.Series(ood_values), errors="coerce").dropna()
        validation_series = pd.to_numeric(pd.Series(validation_values), errors="coerce").dropna()
        selected_feature_series = pd.to_numeric(pd.Series(selected_feature_counts), errors="coerce").dropna()
        selection_row = pd.Series(
            {
                "Model": model_name,
                "target_name": COMBINED_TARGET_NAME,
                "feature_space": FIXED_ATTENTION_FEATURE_SPACE,
                "feature_space_title": "",
                "mean_ood_auroc": float(ood_series.mean()) if not ood_series.empty else float("nan"),
                "ood_auroc_se": _stderr(ood_series),
                "mean_validation_auroc": float(validation_series.mean()) if not validation_series.empty else float("nan"),
                "validation_auroc_se": _stderr(validation_series),
                "selected_feature_count": float(selected_feature_series.mean()) if not selected_feature_series.empty else float("nan"),
                "training_splits": int(training_splits_total),
            }
        )
        rows.append(selection_row.to_dict())
        combined_payloads.append(
            {
                "selection": selection_row,
                "run_metrics_df": combined_run_metrics_df,
                "long_df": combined_long_df,
                "feature_summary_df": feature_summary_df,
                "group_band_df": group_band_df,
            }
        )

    selection_df = pd.DataFrame(rows)
    if not selection_df.empty:
        selection_df = selection_df.sort_values("Model", key=lambda column: column.map(scenario_tables.model_sort_key)).reset_index(drop=True)
    combined_payloads.sort(key=lambda payload: scenario_tables.model_sort_key(_text_or_empty(payload["selection"].get("Model"))))
    return selection_df, combined_payloads


def save_figure(fig: plt.Figure, export_root: Path, *, filename: str, export_figures: bool) -> Path | None:
    if not export_figures:
        return None
    export_root.mkdir(parents=True, exist_ok=True)
    out_path = export_root / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    return out_path


def plot_top_features_across_models(
    payloads: list[dict[str, object]],
    *,
    scenario_heading: str,
    target_name: str,
    export_root: Path,
    top_k: int,
    export_figures: bool,
) -> Path | None:
    if not payloads:
        return None

    n_models = len(payloads)
    figure_height = max(5.8, 0.48 * top_k + 2.0)

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, n_models, figsize=(6.0 * n_models, figure_height), constrained_layout=True)
        axes_array = np.atleast_1d(axes)

        legend_handles: dict[str, Any] = {}
        max_share = 0.0
        for payload in payloads:
            feature_summary_df = payload["feature_summary_df"]
            if feature_summary_df.empty:
                continue
            max_share = max(
                max_share,
                float(pd.to_numeric(feature_summary_df.head(top_k)["mean_importance_share"], errors="coerce").max()),
            )
        max_share = max(max_share, 1e-4)

        for ax, payload in zip(axes_array, payloads, strict=False):
            selection_row = payload["selection"]
            feature_summary_df = payload["feature_summary_df"].head(top_k).iloc[::-1].copy()
            colors = [GROUP_COLORS.get(_text_or_empty(value), "#6b7280") for value in feature_summary_df["attention_feature_group"]]
            y_positions = np.arange(len(feature_summary_df))
            ax.barh(
                y_positions,
                100.0 * pd.to_numeric(feature_summary_df["mean_importance_share"], errors="coerce"),
                xerr=100.0 * pd.to_numeric(feature_summary_df["se_importance_share"], errors="coerce").fillna(0.0),
                color=colors,
                edgecolor="none",
                alpha=0.95,
                error_kw={"ecolor": "#374151", "elinewidth": 1.0, "capsize": 2.0},
            )
            ax.set_yticks(y_positions)
            ax.set_yticklabels([compact_feature_label(row) for _, row in feature_summary_df.iterrows()], fontsize=8.8)
            ax.set_xlabel("Mean importance share (%)", fontsize=10)
            ax.set_xlim(0.0, 100.0 * max_share * 1.18)
            ax.set_title(selection_heading_label(selection_row), fontsize=PANEL_TITLE_FONTSIZE, pad=PANEL_TITLE_PAD)
            ax.grid(axis="x", alpha=0.20, linewidth=0.6)
            for row in feature_summary_df.itertuples(index=False):
                group_name = _text_or_empty(getattr(row, "attention_feature_group", ""))
                if group_name not in legend_handles:
                    legend_handles[group_name] = plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS.get(group_name, "#6b7280"))

        legend_labels = [GROUP_DISPLAY.get(name, name) for name in legend_handles]
        if legend_handles:
            fig.legend(
                [legend_handles[name] for name in legend_handles],
                legend_labels,
                ncol=min(4, len(legend_handles)),
                loc="lower center",
                bbox_to_anchor=(0.5, -0.08),
                frameon=False,
                fontsize=10,
            )

        out_path = save_figure(
            fig,
            export_root,
            filename=f"{_slugify(scenario_heading)}__{target_name}__top_attention_features.png",
            export_figures=export_figures,
        )
        plt.show()
        plt.close(fig)
        return out_path


def plot_group_band_heatmaps_across_models(
    payloads: list[dict[str, object]],
    *,
    scenario_heading: str,
    target_name: str,
    export_root: Path,
    export_figures: bool,
) -> Path | None:
    if not payloads:
        return None

    n_models = len(payloads)
    vmax = 0.0
    for payload in payloads:
        matrix_df = payload["group_band_df"]
        if matrix_df.empty:
            continue
        vmax = max(vmax, float(np.nanmax(matrix_df.to_numpy(dtype=float))))
    vmax = max(vmax, 1e-6)

    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.8), constrained_layout=True)
        axes_array = np.atleast_1d(axes)
        image = None
        for ax, payload in zip(axes_array, payloads, strict=False):
            selection_row = payload["selection"]
            matrix_df = payload["group_band_df"].reindex(index=GROUP_ORDER, columns=BAND_ORDER).fillna(0.0)
            heat_values = 100.0 * matrix_df.to_numpy(dtype=float)
            image = ax.imshow(heat_values, cmap="YlOrBr", vmin=0.0, vmax=100.0 * vmax, aspect="auto")
            ax.set_xticks(np.arange(len(BAND_ORDER)))
            ax.set_xticklabels([BAND_DISPLAY[band] for band in BAND_ORDER], fontsize=9)
            ax.set_yticks(np.arange(len(GROUP_ORDER)))
            ax.set_yticklabels([GROUP_DISPLAY[group_name] for group_name in GROUP_ORDER], fontsize=9)
            ax.set_title(selection_heading_label(selection_row), fontsize=PANEL_TITLE_FONTSIZE, pad=PANEL_TITLE_PAD)
            for row_idx in range(matrix_df.shape[0]):
                for col_idx in range(matrix_df.shape[1]):
                    value = heat_values[row_idx, col_idx]
                    text_color = "white" if value > 50.0 * vmax else "#111827"
                    ax.text(col_idx, row_idx, f"{value:.1f}%", ha="center", va="center", fontsize=8.6, color=text_color)
            ax.set_xlabel("Layer band", fontsize=10)
            ax.set_ylabel("Attention family", fontsize=10)

        if image is not None:
            fig.colorbar(image, ax=axes_array.ravel().tolist(), fraction=0.035, pad=0.03, label="Mean importance share (%)")
        out_path = save_figure(
            fig,
            export_root,
            filename=f"{_slugify(scenario_heading)}__{target_name}__attention_group_band_heatmaps.png",
            export_figures=export_figures,
        )
        plt.show()
        plt.close(fig)
        return out_path


def render_combined_target_average_section(
    *,
    scenario_results: dict[str, tuple[pd.DataFrame, list[dict[str, object]]]],
    section_heading: str,
    export_root: Path,
    top_k: int,
    export_figures: bool,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    selection_df, payloads = build_combined_target_payloads(
        scenario_results,
        target_names=COMBINED_TARGET_NAMES,
    )
    if selection_df.empty or not payloads:
        return pd.DataFrame(), []

    md(f"### {COMBINED_TARGET_TITLE}")
    md(
        "This section averages the saved feature-importance summaries from the deceptive-commitment target "
        "(`delta_deception_rate > 0.3`) and the truthful-commitment target (`delta_deception_rate < -0.3`)."
    )

    md("#### By Model")
    _display_table(build_display_selection_table(selection_df))

    top_path = plot_top_features_across_models(
        payloads,
        scenario_heading=section_heading,
        target_name=COMBINED_TARGET_NAME,
        export_root=export_root,
        top_k=top_k,
        export_figures=export_figures,
    )
    if top_path is not None:
        md(f"Saved figure: `{top_path}`")

    heatmap_path = plot_group_band_heatmaps_across_models(
        payloads,
        scenario_heading=section_heading,
        target_name=COMBINED_TARGET_NAME,
        export_root=export_root,
        export_figures=export_figures,
    )
    if heatmap_path is not None:
        md(f"Saved figure: `{heatmap_path}`")

    md("#### Attention Family By Layer Band Tables")
    for payload in payloads:
        selection_row = payload["selection"]
        md(f"##### {selection_heading_label(selection_row)}")
        _display_table(build_group_band_display_table(payload["group_band_df"]))

    md("#### Top Attention Features By Model")
    for payload in payloads:
        selection_row = payload["selection"]
        md(
            f"##### {selection_heading_label(selection_row)}\n"
            f"- OOD AUROC: `{_format_pm(selection_row['mean_ood_auroc'], selection_row['ood_auroc_se'])}`\n"
            f"- Validation AUROC: `{_format_pm(selection_row['mean_validation_auroc'], selection_row['validation_auroc_se'])}`\n"
            f"- Training splits averaged: `{int(selection_row['training_splits'])}`"
        )
        _display_table(build_top_feature_table(payload["feature_summary_df"], top_k=top_k))

    return selection_df, payloads


def render_scenario_section(
    *,
    metrics_df: pd.DataFrame,
    scenario_name: str,
    section_heading: str,
    section_blurb: str,
    export_root: Path,
    top_k: int,
    export_figures: bool,
) -> dict[str, tuple[pd.DataFrame, list[dict[str, object]]]]:
    md(f"## {section_heading}")
    md(
        f"{section_blurb}\n\n"
        "Method: for each model and target, use the fixed full feature space, "
        "then average normalized XGBoost gain shares across the relevant training splits. "
        "The figures do not average across models."
    )

    scenario_results: dict[str, tuple[pd.DataFrame, list[dict[str, object]]]] = {}
    target_rows = scenario_tables.target_rows(metrics_df)
    if not target_rows:
        md("_No saved targets were available for this scenario._")
        return scenario_results

    for target_name_key, target_heading in target_rows:
        md(f"### {target_heading}")
        selection_df, payloads = build_model_payloads(metrics_df, target_name=target_name_key)
        if selection_df.empty or not payloads:
            md("_No saved feature-importance artifacts were available for this target._")
            continue

        scenario_results[target_name_key] = (selection_df, payloads)

        summary_table_df = build_display_selection_table(selection_df)
        md("#### By Model")
        _display_table(summary_table_df)

        top_path = plot_top_features_across_models(
            payloads,
            scenario_heading=section_heading,
            target_name=target_name_key,
            export_root=export_root,
            top_k=top_k,
            export_figures=export_figures,
        )
        if top_path is not None:
            md(f"Saved figure: `{top_path}`")

        heatmap_path = plot_group_band_heatmaps_across_models(
            payloads,
            scenario_heading=section_heading,
            target_name=target_name_key,
            export_root=export_root,
            export_figures=export_figures,
        )
        if heatmap_path is not None:
            md(f"Saved figure: `{heatmap_path}`")

        md("#### Attention Family By Layer Band Tables")
        for payload in payloads:
            selection_row = payload["selection"]
            md(f"##### {selection_heading_label(selection_row)}")
            _display_table(build_group_band_display_table(payload["group_band_df"]))

        md("#### Top Attention Features By Model")
        for payload in payloads:
            selection_row = payload["selection"]
            md(
                f"##### {selection_heading_label(selection_row)}\n"
                f"- OOD AUROC: `{_format_pm(selection_row['mean_ood_auroc'], selection_row['ood_auroc_se'])}`\n"
                f"- Validation AUROC: `{_format_pm(selection_row['mean_validation_auroc'], selection_row['validation_auroc_se'])}`\n"
                f"- Training splits averaged: `{int(selection_row['training_splits'])}`"
            )
            _display_table(build_top_feature_table(payload["feature_summary_df"], top_k=top_k))

    combined_selection_df, combined_payloads = render_combined_target_average_section(
        scenario_results=scenario_results,
        section_heading=section_heading,
        export_root=export_root,
        top_k=top_k,
        export_figures=export_figures,
    )
    if not combined_selection_df.empty and combined_payloads:
        scenario_results[COMBINED_TARGET_NAME] = (combined_selection_df, combined_payloads)

    return scenario_results


def render_attention_feature_importance_notebook(
    *,
    results_root: Path | None = None,
    top_k: int = 12,
    export_figures: bool = False,
) -> dict[str, dict[str, tuple[pd.DataFrame, list[dict[str, object]]]]]:
    resolved_results_root = (
        Path(os.environ.get("OOD_MAIN3_SCENARIO_RESULTS_ROOT", str(DEFAULT_RESULTS_ROOT))).expanduser().resolve()
        if results_root is None
        else Path(results_root).expanduser().resolve()
    )
    export_root = DEFAULT_EXPORT_ROOT if results_root is None else Path(results_root).expanduser().resolve() / "notebook_exports" / "attention_feature_importance"

    md("# OOD Main3 Attention Feature Importance")
    md(
        f"Results root: `{resolved_results_root}`\n\n"
        "This notebook focuses on attention-feature importance for OOD generalization in the saved XGBoost runs. "
        "It uses the saved per-split coefficient files, averages feature importance across datasets within a model, and keeps models separate."
    )

    all_results: dict[str, dict[str, tuple[pd.DataFrame, list[dict[str, object]]]]] = {}
    for scenario_name, section_heading, section_blurb in SCENARIO_LAYOUT:
        _, _, _, metrics_df = scenario_tables.load_scenario_bundle_frames(resolved_results_root, scenario_name)
        if metrics_df.empty:
            md(f"## {section_heading}")
            md("_No populated bundle directories were found for this scenario._")
            all_results[scenario_name] = {}
            continue
        all_results[scenario_name] = render_scenario_section(
            metrics_df=metrics_df,
            scenario_name=scenario_name,
            section_heading=section_heading,
            section_blurb=section_blurb,
            export_root=export_root / scenario_name,
            top_k=top_k,
            export_figures=export_figures,
        )

    return all_results
