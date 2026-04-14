# %% [markdown]
# # OOD Main3 Consistency Ablation: Precomputed Analysis
#
# This notebook-style script reads the saved outputs from
# `OOD_Modeling_main3_consistency_ablation.py` and builds interpretable plots
# without rerunning any modeling.
#
# It focuses on:
# - AUROC transfer matrices where the diagonal is source-validation AUROC
#   and the off-diagonal cells are OOD AUROC
# - OOD confusion matrices aggregated across off-diagonal train/test pairs
# - saved feature-weight views for every `(feature_space, feature_size)` run
#
# By default it prefers output roots like:
# `OOD_Modeling_main3_consistency_ablation_outputs__deepseek_r1_distill_qwen_7b__logreg`
# and falls back to the legacy non-model-family root if needed.
#
# You can override the output bundle with:
# `OOD_MAIN3_PRECOMPUTED_OUTPUT_ROOT=/abs/path/to/output_root`
#
# Optional knobs:
# - `OOD_MAIN3_PRECOMPUTED_MODEL_FAMILY=logreg|xgboost`
# - `OOD_MAIN3_PRECOMPUTED_TOP_FEATURES=15`
# - `OOD_MAIN3_PRECOMPUTED_EXPORT=1`

# %%
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython import get_ipython
    from IPython.display import Markdown, display
except ImportError:  # pragma: no cover - notebook convenience fallback
    get_ipython = None
    Markdown = None

    def display(obj: Any) -> None:
        print(obj)


# %%
def locate_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "deception2" / "Notebooks").exists() and (candidate / "deception2" / "src").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate repo root from {start}")


def notebook_anchor() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


def md(text: str) -> None:
    shell_name = ""
    if get_ipython is not None and get_ipython() is not None:
        shell_name = get_ipython().__class__.__name__
    if Markdown is not None and shell_name == "ZMQInteractiveShell":
        display(Markdown(text))
    else:
        print(text)


def first_non_null(series: pd.Series, default: Any = np.nan) -> Any:
    for value in series:
        if pd.notna(value):
            return value
    return default


def min_numeric(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.min()) if not numeric.empty else float("nan")


def max_numeric(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else float("nan")


def mean_numeric(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else float("nan")


def int_mean(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return int(round(float(numeric.mean()))) if not numeric.empty else 0


def shorten_label(text: str, width: int = 42) -> str:
    return text if len(text) <= width else f"{text[: width - 3]}..."


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def normalize_model_family(raw_value: str | None) -> str:
    value = str(raw_value or "logreg").strip().lower()
    if value in {"logreg", "logistic", "logistic_regression", "lr"}:
        return "logreg"
    if value in {"xgboost", "xgb"}:
        return "xgboost"
    return value


def format_feature_size_label(feature_size: Any, feature_size_label: Any) -> str:
    label = "" if pd.isna(feature_size_label) else str(feature_size_label).strip()
    if label and label.lower() != "nan":
        return label
    numeric = pd.to_numeric(pd.Series([feature_size]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return f"k{int(numeric):03d}"
    return "unspecified"


def make_feature_run_key(feature_space: str, feature_size_label: str) -> str:
    return f"{feature_space}__{feature_size_label}"


def make_directory_run_key(feature_space: str, feature_size_label: str) -> str:
    return f"{feature_space}/{feature_size_label}"


def format_feature_run_title(
    feature_space_title: str,
    feature_size_label: str,
    model_family_title: str,
) -> str:
    return f"{feature_space_title} [{feature_size_label}; {model_family_title}]"


def default_output_root(notebook_root: Path) -> Path:
    model_slug = "deepseek_r1_distill_qwen_7b"
    preferred_family = normalize_model_family(
        os.environ.get(
            "OOD_MAIN3_PRECOMPUTED_MODEL_FAMILY",
            os.environ.get(
                "OOD_MAIN3_COMPANION_MODEL_FAMILY",
                os.environ.get("OOD_MAIN3_MODEL_FAMILY", "logreg"),
            ),
        )
    )
    candidates = [
        notebook_root / f"OOD_Modeling_main3_consistency_ablation_outputs__{model_slug}__{preferred_family}",
        notebook_root / f"OOD_Modeling_main3_consistency_ablation_outputs__{model_slug}",
    ]
    for candidate in candidates:
        if (candidate / "config.csv").exists():
            return candidate
    return candidates[0]


def load_or_rebuild_csv(
    *,
    primary_path: Path,
    fallback_glob: str,
    description: str,
) -> pd.DataFrame:
    if primary_path.exists():
        return pd.read_csv(primary_path)

    fallback_paths = sorted(MODEL_SELECTION_ROOT.rglob(fallback_glob)) if MODEL_SELECTION_ROOT.exists() else []
    if not fallback_paths:
        raise FileNotFoundError(
            f"Missing {description} file: {primary_path}\n"
            f"No fallback files matching `{fallback_glob}` were found under {MODEL_SELECTION_ROOT}"
        )

    rebuilt_df = pd.concat([pd.read_csv(path) for path in fallback_paths], ignore_index=True)
    rebuilt_df.to_csv(primary_path, index=False)
    print(f"Rebuilt {primary_path.name} from {len(fallback_paths)} per-selection files.")
    return rebuilt_df


def ensure_run_columns(df: pd.DataFrame, *, config: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if "feature_size" not in out.columns:
        out["feature_size"] = pd.NA
    if "feature_size_label" not in out.columns:
        out["feature_size_label"] = out.get("feature_size", pd.Series([pd.NA] * len(out))).apply(
            lambda value: format_feature_size_label(value, pd.NA)
        )
    else:
        out["feature_size_label"] = [
            format_feature_size_label(feature_size, feature_size_label)
            for feature_size, feature_size_label in zip(
                out.get("feature_size", pd.Series([pd.NA] * len(out))),
                out["feature_size_label"],
                strict=False,
            )
        ]

    model_family = str(config.get("model_family", "logreg"))
    model_family_title = str(config.get("model_family_title", "Logistic regression"))
    model_weight_kind = str(config.get("model_weight_kind", "coefficient"))

    if "model_family" not in out.columns:
        out["model_family"] = model_family
    if "model_family_title" not in out.columns:
        out["model_family_title"] = model_family_title
    if "model_weight_kind" not in out.columns:
        out["model_weight_kind"] = model_weight_kind
    if "feature_space_attention_subset_key" not in out.columns:
        out["feature_space_attention_subset_key"] = ""
    if "feature_space_attention_subset_title" not in out.columns:
        out["feature_space_attention_subset_title"] = ""
    if "feature_space_title" not in out.columns and "feature_space" in out.columns:
        out["feature_space_title"] = out["feature_space"].astype(str)
    if "feature_family_group" not in out.columns:
        out["feature_family_group"] = ""

    out["feature_run_key"] = [
        make_feature_run_key(str(feature_space), str(feature_size_label))
        for feature_space, feature_size_label in zip(out["feature_space"], out["feature_size_label"], strict=False)
    ]
    out["feature_run_title"] = [
        format_feature_run_title(
            str(feature_space_title),
            str(feature_size_label),
            str(model_family_title_value),
        )
        for feature_space_title, feature_size_label, model_family_title_value in zip(
            out["feature_space_title"],
            out["feature_size_label"],
            out["model_family_title"],
            strict=False,
        )
    ]
    return out


NOTEBOOK_ROOT = locate_repo_root(notebook_anchor()) / "deception2" / "Notebooks"
OUTPUT_ROOT = Path(
    os.environ.get("OOD_MAIN3_PRECOMPUTED_OUTPUT_ROOT", str(default_output_root(NOTEBOOK_ROOT)))
).expanduser().resolve()

CONFIG_PATH = OUTPUT_ROOT / "config.csv"
TRANSFER_METRICS_PATH = OUTPUT_ROOT / "all_transfer_metrics.csv"
MODEL_SELECTION_PATH = OUTPUT_ROOT / "all_model_selection.csv"
COEFFICIENTS_PATH = OUTPUT_ROOT / "all_coefficients.csv"
MODEL_SELECTION_ROOT = OUTPUT_ROOT / "model_selection"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

CONFIG_DF = pd.read_csv(CONFIG_PATH)
CONFIG = {str(row.setting): row.value for row in CONFIG_DF.itertuples(index=False)}
DELTA_THRESHOLD = float(CONFIG.get("delta_threshold", 0.3))
ENV_ORDER = [part.strip() for part in str(CONFIG.get("env_order", "")).split(",") if part.strip()]
if not ENV_ORDER:
    ENV_ORDER = ["AdvisorAudit", "BS", "CarSales", "Gridworld", "Interview"]

TARGET_SPECS = {
    "delta_pos_gt_0_3": {
        "title": f"delta_deception_rate > {DELTA_THRESHOLD:.1f}",
        "negative_label": f"<= {DELTA_THRESHOLD:.1f}",
        "positive_label": f"> {DELTA_THRESHOLD:.1f}",
    },
    "delta_neg_lt_neg_0_3": {
        "title": f"delta_deception_rate < -{DELTA_THRESHOLD:.1f}",
        "negative_label": f">= -{DELTA_THRESHOLD:.1f}",
        "positive_label": f"< -{DELTA_THRESHOLD:.1f}",
    },
}

TOP_N_FEATURES = int(os.environ.get("OOD_MAIN3_PRECOMPUTED_TOP_FEATURES", "15"))
EXPORT_FIGURES = os.environ.get("OOD_MAIN3_PRECOMPUTED_EXPORT", "1") == "1"
FIGURE_ROOT = OUTPUT_ROOT / "precomputed_interpretable_plots"
if EXPORT_FIGURES:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

md(
    "\n".join(
        [
            "## Output Bundle",
            f"- `OUTPUT_ROOT = {OUTPUT_ROOT}`",
            f"- `TOP_N_FEATURES = {TOP_N_FEATURES}`",
            f"- `EXPORT_FIGURES = {EXPORT_FIGURES}`",
        ]
    )
)


# %%
transfer_metrics_df = load_or_rebuild_csv(
    primary_path=TRANSFER_METRICS_PATH,
    fallback_glob="*__transfer_metrics.csv",
    description="transfer metrics",
)
model_selection_df = load_or_rebuild_csv(
    primary_path=MODEL_SELECTION_PATH,
    fallback_glob="*__selection_summary.csv",
    description="model selection",
)
all_coefficients_df = load_or_rebuild_csv(
    primary_path=COEFFICIENTS_PATH,
    fallback_glob="*__coefficients.csv",
    description="coefficients",
)

transfer_metrics_df = ensure_run_columns(transfer_metrics_df, config=CONFIG)
model_selection_df = ensure_run_columns(model_selection_df, config=CONFIG)
all_coefficients_df = ensure_run_columns(all_coefficients_df, config=CONFIG) if not all_coefficients_df.empty else all_coefficients_df

feature_space_title_lookup = (
    model_selection_df.loc[:, ["feature_space", "feature_space_title"]]
    .drop_duplicates()
    .set_index("feature_space")["feature_space_title"]
    .to_dict()
)

target_title_lookup = (
    transfer_metrics_df.loc[:, ["target_name", "target_title"]]
    .drop_duplicates()
    .set_index("target_name")["target_title"]
    .to_dict()
)

directory_feature_runs: dict[str, list[str]] = {}
transfer_feature_runs: dict[str, list[str]] = {}
missing_transfer_directories: dict[str, list[str]] = {}

for target_name in TARGET_SPECS:
    target_dir = OUTPUT_ROOT / "model_selection" / target_name
    dirs = (
        sorted(
            make_directory_run_key(path.parent.name, path.name)
            for path in target_dir.glob("*/*")
            if path.is_dir()
        )
        if target_dir.exists()
        else []
    )
    present_in_transfer = sorted(
        make_directory_run_key(str(row.feature_space), str(row.feature_size_label))
        for row in (
            transfer_metrics_df.loc[transfer_metrics_df["target_name"].eq(target_name), ["feature_space", "feature_size_label"]]
            .drop_duplicates()
            .itertuples(index=False)
        )
    )
    directory_feature_runs[target_name] = dirs
    transfer_feature_runs[target_name] = present_in_transfer
    missing_transfer_directories[target_name] = [name for name in dirs if name not in set(present_in_transfer)]

display(
    pd.DataFrame(
        [
            {
                "target_name": target_name,
                "run_directories_found": len(directory_feature_runs[target_name]),
                "feature_runs_with_transfer_rows": len(transfer_feature_runs[target_name]),
                "directories_without_transfer_rows": ", ".join(missing_transfer_directories[target_name]) or "",
            }
            for target_name in TARGET_SPECS
        ]
    )
)


# %%
def save_figure(fig: plt.Figure, *, target_name: str, feature_run_key: str, suffix: str) -> Path | None:
    if not EXPORT_FIGURES:
        return None
    out_dir = FIGURE_ROOT / target_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(feature_run_key)}__{suffix}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    return out_path


def build_feature_space_overview(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "target_name",
        "feature_space",
        "feature_space_title",
        "feature_family_group",
        "feature_space_attention_subset_key",
        "feature_space_attention_subset_title",
        "feature_size",
        "feature_size_label",
        "model_family",
        "model_family_title",
        "model_weight_kind",
    ]
    for group_values, subset in metrics_df.groupby(group_cols, sort=False, dropna=False):
        (
            target_name,
            feature_space,
            feature_space_title,
            feature_family_group,
            attention_subset_key,
            attention_subset_title,
            feature_size,
            feature_size_label,
            model_family,
            model_family_title,
            model_weight_kind,
        ) = group_values
        val_subset = subset.loc[subset["eval_role"].eq("val")]
        ood_subset = subset.loc[subset["eval_role"].eq("ood")]
        rows.append(
            {
                "target_name": target_name,
                "target_title": first_non_null(subset["target_title"], default=target_title_lookup.get(target_name, target_name)),
                "target_short_label": "> 0.3" if target_name == "delta_pos_gt_0_3" else "< -0.3",
                "feature_space": feature_space,
                "feature_space_title": feature_space_title,
                "feature_family_group": feature_family_group,
                "feature_space_attention_subset_key": attention_subset_key,
                "feature_space_attention_subset_title": attention_subset_title,
                "feature_size": feature_size,
                "feature_size_label": feature_size_label,
                "model_family": model_family,
                "model_family_title": model_family_title,
                "model_weight_kind": model_weight_kind,
                "feature_run_key": make_feature_run_key(str(feature_space), str(feature_size_label)),
                "feature_run_title": format_feature_run_title(
                    str(feature_space_title),
                    str(feature_size_label),
                    str(model_family_title),
                ),
                "selected_feature_count": int_mean(val_subset["selected_feature_count"]),
                "mean_val_accuracy": mean_numeric(val_subset["accuracy"]),
                "mean_ood_accuracy": mean_numeric(ood_subset["accuracy"]),
                "min_ood_accuracy": min_numeric(ood_subset["accuracy"]),
                "mean_val_balanced_accuracy": mean_numeric(val_subset["balanced_accuracy"]),
                "mean_ood_balanced_accuracy": mean_numeric(ood_subset["balanced_accuracy"]),
                "mean_val_auroc": mean_numeric(val_subset["auroc"]),
                "mean_ood_auroc": mean_numeric(ood_subset["auroc"]),
                "min_ood_auroc": min_numeric(ood_subset["auroc"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target_name", "mean_ood_auroc", "mean_val_auroc"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_source_env_summary(
    metrics_df: pd.DataFrame,
    *,
    target_name: str,
    feature_space: str,
    feature_size_label: str,
) -> pd.DataFrame:
    subset = metrics_df.loc[
        metrics_df["target_name"].eq(target_name)
        & metrics_df["feature_space"].eq(feature_space)
        & metrics_df["feature_size_label"].eq(feature_size_label)
    ].copy()
    val_summary = (
        subset.loc[subset["eval_role"].eq("val"), ["train_env", "accuracy", "balanced_accuracy", "auroc"]]
        .rename(
            columns={
                "accuracy": "val_accuracy",
                "balanced_accuracy": "val_balanced_accuracy",
                "auroc": "val_auroc",
            }
        )
        .reset_index(drop=True)
    )
    ood_summary = (
        subset.loc[subset["eval_role"].eq("ood")]
        .groupby("train_env", as_index=False)
        .agg(
            mean_ood_accuracy=("accuracy", "mean"),
            min_ood_accuracy=("accuracy", "min"),
            mean_ood_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_ood_auroc=("auroc", "mean"),
        )
    )
    merged = val_summary.merge(ood_summary, on="train_env", how="outer", validate="one_to_one")
    env_categorical = pd.Categorical(merged["train_env"], categories=ENV_ORDER, ordered=True)
    return merged.assign(train_env=env_categorical).sort_values("train_env").reset_index(drop=True)


def build_transfer_matrix(
    metrics_df: pd.DataFrame,
    *,
    target_name: str,
    feature_space: str,
    feature_size_label: str,
    metric: str = "auroc",
) -> pd.DataFrame:
    subset = metrics_df.loc[
        metrics_df["target_name"].eq(target_name)
        & metrics_df["feature_space"].eq(feature_space)
        & metrics_df["feature_size_label"].eq(feature_size_label)
    ].copy()
    matrix_df = pd.DataFrame(np.nan, index=ENV_ORDER, columns=ENV_ORDER, dtype=float)
    for row in subset.itertuples(index=False):
        value = getattr(row, metric)
        if row.eval_role == "val":
            matrix_df.loc[str(row.train_env), str(row.train_env)] = float(value)
        else:
            matrix_df.loc[str(row.train_env), str(row.test_env)] = float(value)
    return matrix_df


def build_ood_confusion_matrix(
    metrics_df: pd.DataFrame,
    *,
    target_name: str,
    feature_space: str,
    feature_size_label: str,
) -> pd.DataFrame | None:
    subset = metrics_df.loc[
        metrics_df["target_name"].eq(target_name)
        & metrics_df["feature_space"].eq(feature_space)
        & metrics_df["feature_size_label"].eq(feature_size_label)
        & metrics_df["eval_role"].eq("ood")
    ]
    if subset.empty:
        return None
    target_spec = TARGET_SPECS[target_name]
    return pd.DataFrame(
        [
            [int(subset["tn"].sum()), int(subset["fp"].sum())],
            [int(subset["fn"].sum()), int(subset["tp"].sum())],
        ],
        index=[f"Actual {target_spec['negative_label']}", f"Actual {target_spec['positive_label']}"],
        columns=[f"Pred {target_spec['negative_label']}", f"Pred {target_spec['positive_label']}"],
    )


def load_feature_importance_data(
    model_df: pd.DataFrame,
    *,
    target_name: str,
    feature_space: str,
    feature_size_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    subset = model_df.loc[
        model_df["target_name"].eq(target_name)
        & model_df["feature_space"].eq(feature_space)
        & model_df["feature_size_label"].eq(feature_size_label)
    ].copy()
    env_categorical = pd.Categorical(subset["train_env"], categories=ENV_ORDER, ordered=True)
    subset = subset.assign(train_env=env_categorical).sort_values("train_env").reset_index(drop=True)

    merged_frames: list[pd.DataFrame] = []
    rank_cols = [
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

    for row in subset.itertuples(index=False):
        coef_df = pd.read_csv(Path(row.coefficients_path))
        if "feature_weight" not in coef_df.columns and "coefficient" in coef_df.columns:
            coef_df["feature_weight"] = coef_df["coefficient"]
        if "abs_feature_weight" not in coef_df.columns and "abs_coefficient" in coef_df.columns:
            coef_df["abs_feature_weight"] = coef_df["abs_coefficient"]
        if "feature_weight_kind" not in coef_df.columns:
            coef_df["feature_weight_kind"] = str(getattr(row, "model_weight_kind", "coefficient"))
        selected_path = Path(row.selected_features_path)
        if selected_path.exists():
            selected_df = pd.read_csv(selected_path)
            available_cols = [col for col in rank_cols if col in selected_df.columns]
            if available_cols:
                coef_df = coef_df.merge(
                    selected_df.loc[:, available_cols].drop_duplicates(subset=["feature"]),
                    on="feature",
                    how="left",
                    validate="one_to_one",
                )
        coef_df["train_env"] = str(row.train_env)
        coef_df["feature_run_title"] = str(row.feature_run_title)
        coef_df["feature_family_group"] = str(row.feature_family_group)
        coef_df["model_family_title"] = str(row.model_family_title)
        merged_frames.append(coef_df)

    if not merged_frames:
        empty = pd.DataFrame()
        return empty, empty, "coefficient", str(first_non_null(subset["model_family_title"], default="Model"))

    long_df = pd.concat(merged_frames, ignore_index=True)
    for col in rank_cols:
        if col != "feature" and col not in long_df.columns:
            long_df[col] = np.nan

    weight_kind = str(first_non_null(long_df["feature_weight_kind"], default="coefficient"))
    model_family_title = str(first_non_null(long_df["model_family_title"], default="Model"))
    summary_df = (
        long_df.groupby("feature", as_index=False)
        .agg(
            mean_weight=("feature_weight", "mean"),
            mean_abs_weight=("abs_feature_weight", "mean"),
            max_abs_weight=("abs_feature_weight", "max"),
            selected_in_sources=("train_env", "nunique"),
            global_rank=("global_rank", min_numeric),
            selected_rank=("selected_rank", min_numeric),
            consistency_score=("consistency_score", max_numeric),
            sign_direction=("sign_direction", first_non_null),
            feature_root=("feature_root", first_non_null),
            family=("family", first_non_null),
            metric_name=("metric_name", first_non_null),
            metric_group=("metric_group", first_non_null),
            attention_feature_group=("attention_feature_group", first_non_null),
            transition_prefix=("transition_prefix", first_non_null),
            feature_weight_kind=("feature_weight_kind", first_non_null),
        )
        .sort_values(["mean_abs_weight", "global_rank", "selected_rank"], ascending=[False, True, True], na_position="last")
        .reset_index(drop=True)
    )
    summary_df["mean_coefficient"] = summary_df["mean_weight"]
    summary_df["mean_abs_coefficient"] = summary_df["mean_abs_weight"]
    return long_df, summary_df, weight_kind, model_family_title


def plot_auroc_and_confusion(
    matrix_df: pd.DataFrame,
    confusion_df: pd.DataFrame | None,
    *,
    target_name: str,
    feature_run_key: str,
    feature_run_title: str,
) -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)

    matrix = np.ma.masked_invalid(matrix_df.to_numpy(dtype=float))
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="lightgray")
    im = axes[0].imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[0].set_xticks(np.arange(len(ENV_ORDER)))
    axes[0].set_xticklabels(ENV_ORDER, rotation=35, ha="right")
    axes[0].set_yticks(np.arange(len(ENV_ORDER)))
    axes[0].set_yticklabels(ENV_ORDER)
    axes[0].set_xlabel("Test environment")
    axes[0].set_ylabel("Train environment")
    axes[0].set_title("AUROC transfer matrix", fontsize=11)
    for idx in range(len(ENV_ORDER)):
        axes[0].add_patch(plt.Rectangle((idx - 0.5, idx - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.2))
    for row_idx in range(matrix_df.shape[0]):
        for col_idx in range(matrix_df.shape[1]):
            value = matrix_df.iat[row_idx, col_idx]
            text = "nan" if not np.isfinite(value) else f"{value:.3f}"
            color = "white" if np.isfinite(value) and value < 0.55 else "black"
            axes[0].text(col_idx, row_idx, text, ha="center", va="center", fontsize=8.2, color=color)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="AUROC")

    axes[1].set_title("Summed OOD confusion counts", fontsize=11)
    if confusion_df is None:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No saved OOD confusion counts", ha="center", va="center")
    else:
        confusion = confusion_df.to_numpy(dtype=float)
        vmax = float(np.nanmax(confusion)) if np.isfinite(confusion).any() else 1.0
        im_conf = axes[1].imshow(confusion, cmap=plt.cm.Blues, vmin=0.0, vmax=max(vmax, 1.0))
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(confusion_df.columns, rotation=15, ha="right")
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(confusion_df.index)
        row_sums = confusion.sum(axis=1, keepdims=True)
        row_rates = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums > 0)
        midpoint = max(vmax, 1.0) / 2.0
        for row_idx in range(2):
            for col_idx in range(2):
                axes[1].text(
                    col_idx,
                    row_idx,
                    f"{int(confusion[row_idx, col_idx])}\n({100.0 * row_rates[row_idx, col_idx]:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=9.0,
                    color="white" if confusion[row_idx, col_idx] > midpoint else "black",
                )
        fig.colorbar(im_conf, ax=axes[1], fraction=0.046, pad=0.04, label="Count")

    target_title = target_title_lookup.get(target_name, TARGET_SPECS[target_name]["title"])
    fig.suptitle(
        f"{target_title} | {feature_run_title}\n"
        "Diagonal = source validation AUROC, off-diagonal = OOD AUROC",
        fontsize=13,
    )
    out_path = save_figure(fig, target_name=target_name, feature_run_key=feature_run_key, suffix="auroc_and_confusion")
    if out_path is not None:
        print(f"Saved {out_path}")
    plt.show()
    plt.close(fig)
    return out_path


def plot_feature_importances(
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    target_name: str,
    feature_run_key: str,
    feature_run_title: str,
    model_family_title: str,
    weight_kind: str,
    top_n: int,
) -> Path | None:
    if long_df.empty or summary_df.empty:
        print(f"No saved feature weights available for {feature_run_key}")
        return None

    uses_signed_weights = weight_kind == "coefficient"
    top_df = summary_df.head(top_n).copy()
    ordered_features = top_df["feature"].tolist()
    bar_df = top_df.iloc[::-1].reset_index(drop=True)
    weight_matrix = (
        long_df.loc[long_df["feature"].isin(ordered_features), ["train_env", "feature", "feature_weight"]]
        .pivot_table(index="train_env", columns="feature", values="feature_weight", aggfunc="first")
        .reindex(index=ENV_ORDER, columns=ordered_features)
    )

    fig = plt.figure(figsize=(17.0, max(6.0, 0.42 * len(bar_df) + 2.0)), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35])
    ax_bar = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])

    color_lookup = {
        "positive": "#2b6cb0",
        "negative": "#c05621",
        "mixed": "#6b7280",
        "not_ranked": "#6b7280",
    }
    colors = [color_lookup.get(str(direction), "#6b7280") for direction in bar_df["sign_direction"]]
    y_positions = np.arange(len(bar_df))
    ax_bar.barh(y_positions, bar_df["mean_abs_weight"], color=colors)
    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels([shorten_label(value) for value in bar_df["feature"]], fontsize=9)
    ax_bar.set_xlabel("Mean |coefficient|" if uses_signed_weights else "Mean feature importance")
    ax_bar.set_title(
        f"Top {len(bar_df)} features by {'mean absolute coefficient' if uses_signed_weights else 'mean importance'}",
        fontsize=11,
    )
    ax_bar.grid(axis="x", alpha=0.25, linewidth=0.6)
    for idx, row in enumerate(bar_df.itertuples(index=False)):
        rank_text = f" | rank {int(row.global_rank)}" if np.isfinite(row.global_rank) else ""
        value_text = f"{row.mean_weight:+.3f}" if uses_signed_weights else f"{row.mean_weight:.3f}"
        ax_bar.text(
            float(row.mean_abs_weight) + 0.01,
            idx,
            f"{value_text}{rank_text}",
            va="center",
            fontsize=8.5,
        )

    heat_values = weight_matrix.to_numpy(dtype=float)
    if uses_signed_weights:
        vmax = float(np.nanmax(np.abs(heat_values))) if np.isfinite(heat_values).any() else 1.0
        heat = ax_heat.imshow(
            heat_values,
            cmap="coolwarm",
            vmin=-max(vmax, 1e-6),
            vmax=max(vmax, 1e-6),
            aspect="auto",
        )
    else:
        vmax = float(np.nanmax(heat_values)) if np.isfinite(heat_values).any() else 1.0
        heat = ax_heat.imshow(
            heat_values,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=max(vmax, 1e-6),
            aspect="auto",
        )
    ax_heat.set_xticks(np.arange(len(ordered_features)))
    ax_heat.set_xticklabels([shorten_label(value, width=28) for value in ordered_features], rotation=45, ha="right", fontsize=8)
    ax_heat.set_yticks(np.arange(len(ENV_ORDER)))
    ax_heat.set_yticklabels(ENV_ORDER)
    ax_heat.set_xlabel("Feature")
    ax_heat.set_ylabel("Train environment")
    ax_heat.set_title("Signed feature weights by source environment" if uses_signed_weights else "Feature importances by source environment", fontsize=11)
    if weight_matrix.shape[1] <= 15:
        threshold = 0.55 * max(vmax, 1e-6)
        for row_idx in range(weight_matrix.shape[0]):
            for col_idx in range(weight_matrix.shape[1]):
                value = weight_matrix.iat[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                if uses_signed_weights:
                    color = "white" if abs(value) > threshold else "black"
                    label = f"{value:+.2f}"
                else:
                    color = "white" if value > threshold else "black"
                    label = f"{value:.2f}"
                ax_heat.text(col_idx, row_idx, label, ha="center", va="center", fontsize=7.1, color=color)
    fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04, label="Coefficient" if uses_signed_weights else "Importance")

    target_title = target_title_lookup.get(target_name, TARGET_SPECS[target_name]["title"])
    importance_title = "Coefficient-based feature importance" if uses_signed_weights else f"{model_family_title} feature importance"
    fig.suptitle(
        f"{target_title} | {feature_run_title}\n{importance_title}",
        fontsize=13,
    )
    out_path = save_figure(fig, target_name=target_name, feature_run_key=feature_run_key, suffix="feature_importance")
    if out_path is not None:
        print(f"Saved {out_path}")
    plt.show()
    plt.close(fig)
    return out_path


overview_df = build_feature_space_overview(transfer_metrics_df)
ood_auroc_summary_table_df = (
    overview_df.loc[:, [
        "feature_run_title",
        "feature_run_key",
        "target_short_label",
        "mean_ood_auroc",
        "min_ood_auroc",
    ]]
    .rename(
        columns={
            "feature_run_title": "feature_run",
            "feature_run_key": "feature_run_key",
            "target_short_label": "target",
        }
    )
    .sort_values(["target", "mean_ood_auroc", "min_ood_auroc"], ascending=[True, False, False])
    .reset_index(drop=True)
)


# %% [markdown]
# ## OOD AUROC Summary
#
# Compact table across saved feature runs and targets.

# %%
display(ood_auroc_summary_table_df)


# %% [markdown]
# ## Overview Tables
#
# The tables below summarize the saved transfer metrics for each target and
# saved `(feature_space, feature_size)` run. If a directory exists under
# `model_selection/<target>/<feature_space>/<size_label>` but does not have rows
# in `all_transfer_metrics.csv`, it is listed as "without transfer rows" and is
# skipped for the matrix/confusion plots.

# %%
for target_name in TARGET_SPECS:
    md(f"### {target_title_lookup.get(target_name, TARGET_SPECS[target_name]['title'])}")
    display(
        overview_df.loc[overview_df["target_name"].eq(target_name), [
            "feature_run_key",
            "feature_run_title",
            "feature_family_group",
            "feature_space_attention_subset_title",
            "selected_feature_count",
            "mean_val_accuracy",
            "mean_ood_accuracy",
            "min_ood_accuracy",
            "mean_ood_balanced_accuracy",
            "mean_ood_auroc",
            "min_ood_auroc",
        ]].reset_index(drop=True)
    )
    missing_dirs = missing_transfer_directories.get(target_name, [])
    if missing_dirs:
        print(
            "Skipping directories without saved per-environment transfer rows:",
            ", ".join(missing_dirs),
        )


# %% [markdown]
# ## Per-Run Reports
#
# For every target and saved feature run with transfer rows, the notebook shows:
# - a transfer AUROC matrix
# - a summed OOD confusion matrix
# - saved feature-weight plots
# - the corresponding top-feature summary table

# %%
for target_name in TARGET_SPECS:
    target_title = target_title_lookup.get(target_name, TARGET_SPECS[target_name]["title"])
    md(f"# {target_title}")

    target_runs_df = overview_df.loc[overview_df["target_name"].eq(target_name)].copy()
    target_runs_df = target_runs_df.sort_values(["mean_ood_auroc", "mean_val_auroc"], ascending=[False, False]).reset_index(drop=True)

    for row in target_runs_df.itertuples(index=False):
        md(f"## {row.feature_run_title}  \n`{row.feature_run_key}`")

        display(
            overview_df.loc[
                overview_df["target_name"].eq(target_name)
                & overview_df["feature_run_key"].eq(row.feature_run_key)
            ].reset_index(drop=True)
        )
        display(
            build_source_env_summary(
                transfer_metrics_df,
                target_name=target_name,
                feature_space=str(row.feature_space),
                feature_size_label=str(row.feature_size_label),
            )
        )

        auroc_matrix_df = build_transfer_matrix(
            transfer_metrics_df,
            target_name=target_name,
            feature_space=str(row.feature_space),
            feature_size_label=str(row.feature_size_label),
            metric="auroc",
        )
        ood_confusion_df = build_ood_confusion_matrix(
            transfer_metrics_df,
            target_name=target_name,
            feature_space=str(row.feature_space),
            feature_size_label=str(row.feature_size_label),
        )
        plot_auroc_and_confusion(
            auroc_matrix_df,
            ood_confusion_df,
            target_name=target_name,
            feature_run_key=str(row.feature_run_key),
            feature_run_title=str(row.feature_run_title),
        )

        feature_long_df, feature_summary_df, weight_kind, model_family_title = load_feature_importance_data(
            model_selection_df,
            target_name=target_name,
            feature_space=str(row.feature_space),
            feature_size_label=str(row.feature_size_label),
        )
        plot_feature_importances(
            feature_long_df,
            feature_summary_df,
            target_name=target_name,
            feature_run_key=str(row.feature_run_key),
            feature_run_title=str(row.feature_run_title),
            model_family_title=model_family_title,
            weight_kind=weight_kind,
            top_n=TOP_N_FEATURES,
        )

        display(
            feature_summary_df.loc[:, [
                "feature",
                "mean_weight",
                "mean_abs_weight",
                "selected_in_sources",
                "global_rank",
                "selected_rank",
                "consistency_score",
                "sign_direction",
                "feature_root",
                "family",
                "metric_name",
                "metric_group",
                "attention_feature_group",
            ]].head(TOP_N_FEATURES).reset_index(drop=True)
        )


# %% [markdown]
# ## Figure Export
#
# If `OOD_MAIN3_PRECOMPUTED_EXPORT=1`, the notebook writes PNGs to:
# `OUTPUT_ROOT / "precomputed_interpretable_plots"`

# %%
if EXPORT_FIGURES:
    print(f"Figures were saved under: {FIGURE_ROOT}")
else:
    print("Figure export disabled.")
