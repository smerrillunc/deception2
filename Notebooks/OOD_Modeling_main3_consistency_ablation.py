#!/usr/bin/env python3
# %% [markdown]
# # OOD Modeling Main 3: Fast Cross-Environment Consistency + Activation Baselines
#
# This notebook-style script keeps the `train on one environment / test on all`
# evaluation pattern, but makes the activation baselines use the raw `.h5`
# sentence-end activations instead of the scalar parquet activation summaries.
#
# Main choices in this version:
# - only predict `delta > 0.3` and `delta < -0.3`
# - rank attention features globally by cross-environment consistency and keep the top 256
# - sweep feature sizes `[32, 64, 128, 256]`
# - use fixed `C=0.1` by default for faster iteration
# - activation baselines use `.h5` final-layer sentence activations:
#   1. raw z-standardized final-sentence activations
#   2. PCA-k of z-standardized final-sentence activations
#   3. PCA-k of z-standardized (final - previous) activations
#   4. PCA-k of z-standardized (final - mean(previous up to 4)) activations
# - attention + activation uses k ranked attention features plus k PCA dimensions
# - report explicit selected-feature tables and coefficient tables for every trained model
# - summarize confusion matrices in raw prediction counts with target-aware labels

# %%
from __future__ import annotations

import gc
import importlib
import math
import os
import re
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from IPython.display import display
except Exception:
    def display(value: Any) -> None:
        print(value)

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

SOURCE_PATH = Path(
    os.environ.get(
        "OOD_MAIN3_COMPANION_SOURCE_PATH",
        __file__ if "__file__" in globals() else str(Path.cwd() / "OOD_Modeling_main3_consistency_ablation.py"),
    )
).resolve()
NOTEBOOK_ROOT = Path(
    os.environ.get(
        "OOD_MAIN3_COMPANION_NOTEBOOK_ROOT",
        str(SOURCE_PATH.parent),
    )
).resolve()
ROOT_DIR = Path(
    os.environ.get(
        "OOD_MAIN3_COMPANION_REPO_ROOT",
        str(NOTEBOOK_ROOT.parent),
    )
).resolve()
SRC_DIR = ROOT_DIR / "src"
for search_root in (NOTEBOOK_ROOT, SRC_DIR):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import ood_modeling_main_lib as oml
import deception_prefix_feature_and_activation_extractor as extractor

oml = importlib.reload(oml)
slugify = oml.slugify
choose_decision_threshold = oml.choose_decision_threshold
summarize_score_metrics = oml.summarize_score_metrics
safe_metric_mean = oml.safe_metric_mean
safe_metric_min = oml.safe_metric_min
safe_metric_std = oml.safe_metric_std
build_common_layer_roots = oml.build_common_layer_roots
COMMITMENT_NON_FEATURE_COLUMNS = set(oml.COMMITMENT_NON_FEATURE_COLUMNS)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 240)


def maybe_tqdm(iterable: Iterable[Any], *, desc: str, total: int | None = None, disable: bool = False):
    if disable or _tqdm is None:
        return iterable
    return _tqdm(iterable, desc=desc, total=total)


def env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    return float(raw)


def env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return tuple(float(value) for value in default)
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(float(part) for part in parts)


def env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return tuple(int(value) for value in default)
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(int(part) for part in parts)


# %%
DATASET_ROOT = Path(
    os.environ.get(
        "OOD_MAIN3_COMPANION_DATASET_ROOT",
        str(ROOT_DIR / "DatasetMain"),
    )
).resolve()
MODEL_DIRNAME = os.environ.get(
    "OOD_MAIN3_COMPANION_MODEL_NAME",
    os.environ.get("OOD_MAIN3_MODEL_NAME", "DeepSeek-R1-Distill-Qwen-7B"),
)

ENV_SPECS = OrderedDict(
    [
        ("AdvisorAudit", "advisor_audit"),
        ("BS", "bs"),
        ("CarSales", "car_sales"),
        ("Gridworld", "gridworld"),
        ("Interview", "interview"),
    ]
)
ENV_ORDER = list(ENV_SPECS.keys())

FEATURE_FILENAME = "prefix_deception_features.parquet.tmp"
ACTIVATION_FILENAME = "prefix_deception_activations.h5"
RANDOM_SEED = env_int("OOD_MAIN3_COMPANION_SEED", env_int("OOD_MAIN3_SEED", 42))
VAL_SIZE = env_float("OOD_MAIN3_COMPANION_VAL_SIZE", env_float("OOD_MAIN3_VAL_SIZE", 0.20))
DELTA_THRESHOLD = env_float("OOD_MAIN3_COMPANION_DELTA_THRESHOLD", env_float("OOD_MAIN3_DELTA_THRESHOLD", 0.30))
FEATURE_SIZE_GRID = env_int_tuple("OOD_MAIN3_COMPANION_FEATURE_SIZES", (32, 64, 128, 256))
ATTENTION_TOP_K = env_int("OOD_MAIN3_COMPANION_ATTENTION_TOP_K", max(FEATURE_SIZE_GRID))
C_GRID = env_float_tuple("OOD_MAIN3_COMPANION_C_GRID", (0.1,))
PER_ROOT_LIMIT = env_int("OOD_MAIN3_COMPANION_PER_ROOT_LIMIT", 4)
ROOT_BATCH_SIZE = env_int("OOD_MAIN3_COMPANION_ROOT_BATCH_SIZE", 8)
ACTIVATION_PCA_DIM = int(max(FEATURE_SIZE_GRID))
DECISION_THRESHOLD_MODE = os.environ.get("OOD_MAIN3_COMPANION_DECISION_THRESHOLD_MODE", "train_balanced_accuracy")
MODEL_SELECTION_OBJECTIVE = os.environ.get(
    "OOD_MAIN3_COMPANION_MODEL_SELECTION_OBJECTIVE",
    "mean_ood_auroc_oracle",
)
FORCE_REBUILD_REDUCTIONS = os.environ.get("OOD_MAIN3_COMPANION_FORCE_REBUILD", "0") == "1"
DISABLE_TQDM = os.environ.get("OOD_MAIN3_COMPANION_DISABLE_TQDM", "0") == "1"
TOP_FEATURES_TO_SHOW = env_int("OOD_MAIN3_COMPANION_TOP_FEATURES_TO_SHOW", 20)

OUTPUT_ROOT = Path(
    os.environ.get(
        "OOD_MAIN3_COMPANION_OUTPUT_ROOT",
        str(NOTEBOOK_ROOT / f"OOD_Modeling_main3_consistency_ablation_outputs__{slugify(MODEL_DIRNAME)}"),
    )
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ATTENTION_METRIC_NAMES = tuple(extractor.ATTENTION_METRIC_NAMES)
HEAD_SUMMARY_NAMES = tuple(extractor.HEAD_SUMMARY_NAMES)
BAND_NAMES = ("early", "mid", "late")
BAND_STAT_NAMES = ("mean", "min", "max", "std")

config_df = pd.DataFrame(
    [
        {"setting": "source_path", "value": str(SOURCE_PATH)},
        {"setting": "notebook_root", "value": str(NOTEBOOK_ROOT)},
        {"setting": "repo_root", "value": str(ROOT_DIR)},
        {"setting": "model_dirname", "value": MODEL_DIRNAME},
        {"setting": "dataset_root", "value": str(DATASET_ROOT)},
        {"setting": "output_root", "value": str(OUTPUT_ROOT)},
        {"setting": "env_order", "value": ", ".join(ENV_ORDER)},
        {"setting": "val_size", "value": VAL_SIZE},
        {"setting": "delta_threshold", "value": DELTA_THRESHOLD},
        {"setting": "attention_top_k", "value": ATTENTION_TOP_K},
        {"setting": "feature_size_grid", "value": ", ".join(str(value) for value in FEATURE_SIZE_GRID)},
        {"setting": "c_grid", "value": ", ".join(f"{value:g}" for value in C_GRID)},
        {"setting": "per_root_limit", "value": PER_ROOT_LIMIT},
        {"setting": "root_batch_size", "value": ROOT_BATCH_SIZE},
        {"setting": "activation_pca_dim", "value": ACTIVATION_PCA_DIM},
        {"setting": "model_selection_objective", "value": MODEL_SELECTION_OBJECTIVE},
        {"setting": "force_rebuild_reductions", "value": FORCE_REBUILD_REDUCTIONS},
        {"setting": "disable_tqdm", "value": DISABLE_TQDM},
        {"setting": "top_features_to_show", "value": TOP_FEATURES_TO_SHOW},
    ]
)
display(config_df)
config_df.to_csv(OUTPUT_ROOT / "config.csv", index=False)


TARGET_SPECS = OrderedDict(
    [
        (
            "delta_pos_gt_0_3",
            {
                "title": "delta_deception_rate > 0.3",
                "label_fn": lambda delta: delta > DELTA_THRESHOLD,
                "negative_label": f"<= {DELTA_THRESHOLD:.1f}",
                "positive_label": f"> {DELTA_THRESHOLD:.1f}",
            },
        ),
        (
            "delta_neg_lt_neg_0_3",
            {
                "title": "delta_deception_rate < -0.3",
                "label_fn": lambda delta: delta < -DELTA_THRESHOLD,
                "negative_label": f">= -{DELTA_THRESHOLD:.1f}",
                "positive_label": f"< -{DELTA_THRESHOLD:.1f}",
            },
        ),
    ]
)


@dataclass(frozen=True)
class FeatureSpaceSpec:
    name: str
    title: str
    family_title: str
    uses_attention: bool
    activation_variant: str | None
    activation_use_pca: bool


FEATURE_SPACES = OrderedDict(
    [
        (
            "attention_only",
            FeatureSpaceSpec(
                name="attention_only",
                title="Attention only",
                family_title="attention_only",
                uses_attention=True,
                activation_variant=None,
                activation_use_pca=False,
            ),
        ),
        (
            "activation_pca_final",
            FeatureSpaceSpec(
                name="activation_pca_final",
                title="Activation only: PCA final",
                family_title="activation_only",
                uses_attention=False,
                activation_variant="final_sentence",
                activation_use_pca=True,
            ),
        ),
        (
            "activation_pca_delta_last2",
            FeatureSpaceSpec(
                name="activation_pca_delta_last2",
                title="Activation only: PCA final - previous",
                family_title="activation_only",
                uses_attention=False,
                activation_variant="delta_last2",
                activation_use_pca=True,
            ),
        ),
        (
            "activation_pca_delta_prev4mean",
            FeatureSpaceSpec(
                name="activation_pca_delta_prev4mean",
                title="Activation only: PCA final - mean(prev 4)",
                family_title="activation_only",
                uses_attention=False,
                activation_variant="delta_prev4mean",
                activation_use_pca=True,
            ),
        ),
        (
            "baseline_raw_final",
            FeatureSpaceSpec(
                name="baseline_raw_final",
                title="Baseline: raw final",
                family_title="baseline",
                uses_attention=False,
                activation_variant="final_sentence",
                activation_use_pca=False,
            ),
        ),
        (
            "attention_plus_activation_pca_final",
            FeatureSpaceSpec(
                name="attention_plus_activation_pca_final",
                title="Attention + PCA final",
                family_title="attention_plus_activation",
                uses_attention=True,
                activation_variant="final_sentence",
                activation_use_pca=True,
            ),
        ),
        (
            "attention_plus_activation_pca_delta_last2",
            FeatureSpaceSpec(
                name="attention_plus_activation_pca_delta_last2",
                title="Attention + PCA final - previous",
                family_title="attention_plus_activation",
                uses_attention=True,
                activation_variant="delta_last2",
                activation_use_pca=True,
            ),
        ),
        (
            "attention_plus_activation_pca_delta_prev4mean",
            FeatureSpaceSpec(
                name="attention_plus_activation_pca_delta_prev4mean",
                title="Attention + PCA final - mean(prev 4)",
                family_title="attention_plus_activation",
                uses_attention=True,
                activation_variant="delta_prev4mean",
                activation_use_pca=True,
            ),
        ),
    ]
)

FAMILY_PANEL_ORDER = (
    "attention_only",
    "activation_only",
    "attention_plus_activation",
    "baseline",
)

FAMILY_DISPLAY_TITLES = {
    "attention_only": "Attention only",
    "activation_only": "Activation only",
    "attention_plus_activation": "Attention + activation",
    "baseline": "Baseline",
}


def build_dataset_file_map(model_dirname: str) -> OrderedDict[str, dict[str, Path]]:
    file_map: OrderedDict[str, dict[str, Path]] = OrderedDict()
    for env_name, env_dir in ENV_SPECS.items():
        feature_path = DATASET_ROOT / env_dir / model_dirname / FEATURE_FILENAME
        activation_path = DATASET_ROOT / env_dir / model_dirname / ACTIVATION_FILENAME
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature parquet for {env_name}: {feature_path}")
        if not activation_path.exists():
            raise FileNotFoundError(f"Missing activation h5 for {env_name}: {activation_path}")
        file_map[env_name] = {
            "feature_path": feature_path,
            "activation_path": activation_path,
        }
    return file_map


DATASET_FILE_MAP = build_dataset_file_map(MODEL_DIRNAME)


# %%
def annotate_prefix_metadata(df: pd.DataFrame, *, env_name: str) -> pd.DataFrame:
    out = df.copy()
    out["env_name"] = env_name
    out["example_id"] = out["example_id"].astype(str)
    out["sentence_idx"] = pd.to_numeric(out["sentence_idx"], errors="coerce")
    out["deception_rate"] = pd.to_numeric(out["deception_rate"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["example_id", "sentence_idx", "deception_rate"]).copy()
    out["sentence_idx"] = out["sentence_idx"].astype(int)
    out["deception_rate"] = out["deception_rate"].astype(np.float32)
    out = out.sort_values(["example_id", "sentence_idx", "row_idx"], kind="mergesort").reset_index(drop=True)
    out["prev_deception_rate"] = (
        out.groupby("example_id", sort=False)["deception_rate"].shift(1).astype(np.float32)
    )
    out["delta_deception_rate"] = (out["deception_rate"] - out["prev_deception_rate"]).astype(np.float32)
    valid_delta = out["delta_deception_rate"].notna()
    for target_name, target_spec in TARGET_SPECS.items():
        out[f"label__{target_name}"] = np.where(
            valid_delta,
            np.asarray(
                target_spec["label_fn"](out["delta_deception_rate"].to_numpy(dtype=np.float32, copy=False)),
                dtype=np.int8,
            ),
            np.nan,
        )
    return out


def build_example_split_map(example_ids: pd.Series, *, seed: int, val_size: float) -> dict[str, str]:
    unique_examples = pd.Series(example_ids.astype(str).unique(), dtype="string")
    if unique_examples.shape[0] < 2:
        raise ValueError("Need at least 2 unique example_id values to build train / val splits.")
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(unique_examples.shape[0])
    val_count = int(round(float(val_size) * unique_examples.shape[0]))
    val_count = max(1, min(unique_examples.shape[0] - 1, val_count))
    val_examples = set(unique_examples.iloc[order[:val_count]].astype(str).tolist())
    return {
        str(example_id): ("val" if str(example_id) in val_examples else "train")
        for example_id in unique_examples.astype(str).tolist()
    }


def split_summary_row(df: pd.DataFrame, *, env_name: str) -> dict[str, Any]:
    train_df = df.loc[df["split"].eq("train")].copy()
    val_df = df.loc[df["split"].eq("val")].copy()
    row: dict[str, Any] = {
        "env_name": env_name,
        "rows": int(len(df)),
        "examples": int(df["example_id"].nunique()),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_delta_rows": int(train_df["delta_deception_rate"].notna().sum()),
        "val_delta_rows": int(val_df["delta_deception_rate"].notna().sum()),
    }
    for target_name in TARGET_SPECS:
        row[f"train_pos_rate__{target_name}"] = float(pd.to_numeric(train_df[f"label__{target_name}"], errors="coerce").mean())
        row[f"val_pos_rate__{target_name}"] = float(pd.to_numeric(val_df[f"label__{target_name}"], errors="coerce").mean())
    return row


def load_feature_metadata(feature_path: Path, env_name: str) -> pd.DataFrame:
    df = pd.read_parquet(feature_path, columns=["example_id", "sentence_idx", "deception_rate"]).copy()
    df["row_idx"] = np.arange(len(df), dtype=np.int64)
    return annotate_prefix_metadata(df, env_name=env_name)


def load_activation_metadata(activation_path: Path, env_name: str) -> pd.DataFrame:
    with h5py.File(activation_path, "r") as f:
        example_ids = pd.Series(f["example_id"].asstr()[:], dtype="string")
        sentence_idx = pd.Series(np.asarray(f["sentence_idx"][:], dtype=np.int64))
        deception_rate = pd.Series(np.asarray(f["deception_rate"][:], dtype=np.float32))
    df = pd.DataFrame(
        {
            "example_id": example_ids,
            "sentence_idx": sentence_idx,
            "deception_rate": deception_rate,
            "row_idx": np.arange(len(example_ids), dtype=np.int64),
        }
    )
    return annotate_prefix_metadata(df, env_name=env_name)


@dataclass
class ActivationEnvStore:
    env_name: str
    activation_path: Path
    metadata_df: pd.DataFrame

    def load_matrix(self, row_indices: np.ndarray, *, variant: str) -> np.ndarray:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        sort_order = np.argsort(row_indices, kind="mergesort")
        sorted_indices = row_indices[sort_order]
        unsort = np.empty_like(sort_order)
        unsort[sort_order] = np.arange(sort_order.shape[0], dtype=np.int64)

        with h5py.File(self.activation_path, "r") as f:
            activations = np.asarray(f["activations"][sorted_indices], dtype=np.float32)
            activation_mask = np.asarray(f["activation_mask"][sorted_indices], dtype=bool)

        current = activations[:, 0, :]
        if variant == "final_sentence":
            feature_matrix = current
        elif variant == "delta_last2":
            prev = activations[:, 1, :]
            prev_valid = activation_mask[:, 1]
            feature_matrix = current - np.where(prev_valid[:, None], prev, 0.0)
        elif variant == "delta_prev4mean":
            prev_vectors = activations[:, 1:5, :]
            prev_mask = activation_mask[:, 1:5]
            prev_count = prev_mask.sum(axis=1, keepdims=True)
            prev_sum = (prev_vectors * prev_mask[:, :, None]).sum(axis=1)
            prev_mean = np.divide(
                prev_sum,
                np.maximum(prev_count, 1),
                out=np.zeros_like(current),
                where=prev_count > 0,
            )
            feature_matrix = current - prev_mean
        else:
            raise ValueError(f"Unsupported activation variant: {variant!r}")

        return np.asarray(feature_matrix[unsort], dtype=np.float32)


feature_metadata_by_env: OrderedDict[str, pd.DataFrame] = OrderedDict()
activation_metadata_by_env: OrderedDict[str, pd.DataFrame] = OrderedDict()
activation_stores: OrderedDict[str, ActivationEnvStore] = OrderedDict()
split_summary_rows: list[dict[str, Any]] = []

for env_name, env_paths in maybe_tqdm(
    DATASET_FILE_MAP.items(),
    desc="Load metadata",
    total=len(DATASET_FILE_MAP),
    disable=DISABLE_TQDM,
):
    feature_metadata_df = load_feature_metadata(env_paths["feature_path"], env_name)
    split_map = build_example_split_map(feature_metadata_df["example_id"], seed=RANDOM_SEED, val_size=VAL_SIZE)
    feature_metadata_df["split"] = feature_metadata_df["example_id"].map(split_map).astype("string")
    feature_metadata_by_env[env_name] = feature_metadata_df

    activation_metadata_df = load_activation_metadata(env_paths["activation_path"], env_name)
    activation_metadata_df["split"] = activation_metadata_df["example_id"].map(split_map).astype("string")
    if activation_metadata_df["split"].isna().any():
        raise ValueError(f"{env_name} activation metadata contains example IDs absent from the feature split map.")
    activation_metadata_by_env[env_name] = activation_metadata_df
    activation_stores[env_name] = ActivationEnvStore(
        env_name=env_name,
        activation_path=env_paths["activation_path"],
        metadata_df=activation_metadata_df,
    )

    split_summary_rows.append(split_summary_row(feature_metadata_df, env_name=env_name))

split_summary_df = pd.DataFrame(split_summary_rows).sort_values("env_name").reset_index(drop=True)
display(split_summary_df)
split_summary_df.to_csv(OUTPUT_ROOT / "split_summary.csv", index=False)

sample_activation_store = next(iter(activation_stores.values()))
sample_activation_row_idx = sample_activation_store.metadata_df["row_idx"].iloc[:1].to_numpy(dtype=np.int64, copy=False)
ACTIVATION_HIDDEN_DIM = int(
    sample_activation_store.load_matrix(sample_activation_row_idx, variant="final_sentence").shape[1]
)


split_cache_by_env: dict[str, dict[str, Any]] = {}
for env_name, metadata_df in feature_metadata_by_env.items():
    valid_delta = metadata_df["delta_deception_rate"].notna().to_numpy(dtype=bool, copy=False)
    train_mask = metadata_df["split"].eq("train").to_numpy(dtype=bool, copy=False) & valid_delta
    val_mask = metadata_df["split"].eq("val").to_numpy(dtype=bool, copy=False) & valid_delta
    bundle: dict[str, Any] = {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "train_row_idx": metadata_df.loc[train_mask, "row_idx"].to_numpy(dtype=np.int64, copy=False),
        "val_row_idx": metadata_df.loc[val_mask, "row_idx"].to_numpy(dtype=np.int64, copy=False),
    }
    for target_name in TARGET_SPECS:
        bundle[f"y_train__{target_name}"] = metadata_df.loc[train_mask, f"label__{target_name}"].to_numpy(dtype=np.int8, copy=False)
        bundle[f"y_val__{target_name}"] = metadata_df.loc[val_mask, f"label__{target_name}"].to_numpy(dtype=np.int8, copy=False)
    split_cache_by_env[env_name] = bundle


# %%
ATTN_ROOT_RE = re.compile(
    r"^(?P<metric>"
    + "|".join(re.escape(metric) for metric in ATTENTION_METRIC_NAMES)
    + r")_(?P<head_summary>"
    + "|".join(re.escape(name) for name in HEAD_SUMMARY_NAMES)
    + r")$"
)


def split_layer_positions(layer_count: int) -> dict[str, np.ndarray]:
    if layer_count < 1:
        raise ValueError(f"Expected at least one layer, got {layer_count}.")
    parts = np.array_split(np.arange(layer_count, dtype=np.int64), len(BAND_NAMES))
    return {band_name: indices for band_name, indices in zip(BAND_NAMES, parts, strict=True)}


def reduce_layer_block(feature_array: np.ndarray) -> dict[str, np.ndarray]:
    feature_array = np.asarray(feature_array, dtype=np.float32)
    band_indices = split_layer_positions(int(feature_array.shape[1]))
    out: dict[str, np.ndarray] = {}
    for band_name, indices in band_indices.items():
        band_values = feature_array[:, indices]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[f"{band_name}__mean"] = np.nanmean(band_values, axis=1, dtype=np.float32)
            out[f"{band_name}__min"] = np.nanmin(band_values, axis=1)
            out[f"{band_name}__max"] = np.nanmax(band_values, axis=1)
            out[f"{band_name}__std"] = np.nanstd(band_values, axis=1, dtype=np.float32)
    return out


layer_root_map, _ = build_common_layer_roots(
    OrderedDict((env_name, env_paths["feature_path"]) for env_name, env_paths in DATASET_FILE_MAP.items())
)
attention_layer_roots = OrderedDict(
    (root_name, columns)
    for root_name, columns in layer_root_map.items()
    if ATTN_ROOT_RE.fullmatch(root_name) is not None
)
if not attention_layer_roots:
    raise ValueError("No shared attention layer roots found across environments.")


def build_attention_reduction_lookup() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root_name, columns in attention_layer_roots.items():
        match = ATTN_ROOT_RE.fullmatch(root_name)
        if match is None:
            continue
        layer_count = len(columns)
        for band_name in BAND_NAMES:
            for stat_name in BAND_STAT_NAMES:
                rows.append(
                    {
                        "feature": f"{root_name}__band_{band_name}__{stat_name}",
                        "feature_root": root_name,
                        "metric_name": match.group("metric"),
                        "head_summary": match.group("head_summary"),
                        "band": band_name,
                        "band_stat": stat_name,
                        "layer_count": int(layer_count),
                        "family": "attention",
                    }
                )
    return pd.DataFrame(rows)


attention_reduction_lookup_df = build_attention_reduction_lookup()
display(attention_reduction_lookup_df.head(20))
attention_reduction_lookup_df.to_csv(OUTPUT_ROOT / "attention_reduction_lookup.csv", index=False)


def build_attention_reduced_env_frame(
    env_name: str,
    *,
    feature_path: Path,
    metadata_df: pd.DataFrame,
    root_batch_size: int,
) -> pd.DataFrame:
    row_ids = metadata_df["row_idx"].to_numpy(dtype=np.int64, copy=False)
    arrays: dict[str, np.ndarray] = {}
    ordered_roots = list(attention_layer_roots.items())
    iterator = maybe_tqdm(
        range(0, len(ordered_roots), int(root_batch_size)),
        desc=f"Reduce attention:{env_name}",
        total=int(math.ceil(len(ordered_roots) / int(root_batch_size))),
        disable=DISABLE_TQDM,
    )
    for start_idx in iterator:
        batch_items = ordered_roots[start_idx : start_idx + int(root_batch_size)]
        batch_columns = [column_name for _, columns in batch_items for column_name in columns]
        batch_df = pd.read_parquet(feature_path, columns=batch_columns)
        batch_df = batch_df.iloc[row_ids].reset_index(drop=True)
        for root_name, columns in batch_items:
            feature_array = (
                batch_df.loc[:, columns]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .to_numpy(dtype=np.float32, copy=False)
            )
            reduced = reduce_layer_block(feature_array)
            for key, values in reduced.items():
                arrays[f"{root_name}__band_{key}"] = np.asarray(values, dtype=np.float32)
        del batch_df
        gc.collect()

    return pd.concat([metadata_df.reset_index(drop=True), pd.DataFrame(arrays)], axis=1)


def feature_columns_for_env(env_df: pd.DataFrame) -> list[str]:
    reserved = {
        "env_name",
        "example_id",
        "sentence_idx",
        "deception_rate",
        "row_idx",
        "prev_deception_rate",
        "delta_deception_rate",
        "split",
        *(f"label__{target_name}" for target_name in TARGET_SPECS),
    }
    return [column for column in env_df.columns if column not in reserved]


attention_envs: OrderedDict[str, pd.DataFrame] = OrderedDict()
for env_name, env_paths in maybe_tqdm(
    DATASET_FILE_MAP.items(),
    desc="Build env frames",
    total=len(DATASET_FILE_MAP),
    disable=DISABLE_TQDM,
):
    attention_cache_path = CACHE_DIR / f"{slugify(env_name)}__attention_reduced.parquet"
    if attention_cache_path.exists() and not FORCE_REBUILD_REDUCTIONS:
        attention_df = pd.read_parquet(attention_cache_path)
    else:
        attention_df = build_attention_reduced_env_frame(
            env_name,
            feature_path=env_paths["feature_path"],
            metadata_df=feature_metadata_by_env[env_name],
            root_batch_size=int(ROOT_BATCH_SIZE),
        )
        attention_df.to_parquet(attention_cache_path, index=False)
    attention_envs[env_name] = attention_df

attention_feature_names = feature_columns_for_env(next(iter(attention_envs.values())))
feature_space_summary_df = pd.DataFrame(
    [
        {"setting": "attention_layer_root_count", "value": len(attention_layer_roots)},
        {"setting": "attention_reduced_feature_count", "value": len(attention_feature_names)},
        {"setting": "activation_hidden_dim", "value": ACTIVATION_HIDDEN_DIM},
        {"setting": "activation_pca_dim", "value": ACTIVATION_PCA_DIM},
        {"setting": "feature_size_grid", "value": ",".join(str(value) for value in FEATURE_SIZE_GRID)},
    ]
)
display(feature_space_summary_df)
feature_space_summary_df.to_csv(OUTPUT_ROOT / "feature_space_summary.csv", index=False)


# %%
def compute_standardized_effects(x_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
    x_array = np.asarray(x_array, dtype=np.float32)
    y_array = np.asarray(y_array, dtype=np.int8)
    pos_mask = y_array == 1
    neg_mask = y_array == 0
    if not pos_mask.any() or not neg_mask.any():
        return np.full(x_array.shape[1], np.nan, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(x_array, axis=0, dtype=np.float32)
        std = np.nanstd(x_array, axis=0, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std[~np.isfinite(std) | (np.abs(std) < 1e-8)] = np.nan

    z_array = (x_array - mean[None, :]) / std[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pos_mean = np.nanmean(z_array[pos_mask], axis=0, dtype=np.float32)
        neg_mean = np.nanmean(z_array[neg_mask], axis=0, dtype=np.float32)
    return np.asarray(pos_mean - neg_mean, dtype=np.float32)


def build_consistency_ranking(
    env_frames: OrderedDict[str, pd.DataFrame],
    *,
    feature_names: list[str],
    feature_lookup_df: pd.DataFrame,
    target_name: str,
) -> pd.DataFrame:
    effect_rows: list[pd.Series] = []
    target_col = f"label__{target_name}"

    for env_name, env_df in env_frames.items():
        train_df = env_df.loc[env_df["split"].eq("train") & env_df[target_col].notna()].copy()
        y_train = train_df[target_col].to_numpy(dtype=np.int8, copy=False)
        if np.unique(y_train).size < 2:
            raise ValueError(f"{env_name} train split does not contain both classes for {target_name}.")
        x_array = (
            train_df.loc[:, feature_names]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=np.float32, copy=False)
        )
        effect_rows.append(pd.Series(compute_standardized_effects(x_array, y_train), index=feature_names, name=env_name, dtype=np.float32))

    effects_df = pd.DataFrame(effect_rows).T.reset_index().rename(columns={"index": "feature"})
    effects_df = feature_lookup_df.merge(effects_df, on="feature", how="left", validate="one_to_one")
    effects_df = effects_df.rename(columns={env_name: f"{slugify(env_name)}_effect" for env_name in ENV_ORDER})

    effect_cols = [f"{slugify(env_name)}_effect" for env_name in ENV_ORDER]
    for effect_col in effect_cols:
        effects_df[f"{effect_col}_abs"] = effects_df[effect_col].abs()

    effect_array = effects_df[effect_cols].to_numpy(dtype=np.float32, copy=False)
    abs_array = effects_df[[f"{col}_abs" for col in effect_cols]].to_numpy(dtype=np.float32, copy=False)
    positive_all = np.all(effect_array > 0.0, axis=1)
    negative_all = np.all(effect_array < 0.0, axis=1)
    effects_df["same_sign_all"] = positive_all | negative_all
    effects_df["sign_direction"] = np.select([positive_all, negative_all], ["positive", "negative"], default="mixed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        effects_df["min_abs_effect"] = np.nanmin(abs_array, axis=1)
        effects_df["mean_abs_effect"] = np.nanmean(abs_array, axis=1)
        effects_df["median_abs_effect"] = np.nanmedian(abs_array, axis=1)
        effects_df["std_effect"] = np.nanstd(effect_array, axis=1)

    effects_df["consistency_score"] = (
        effects_df["min_abs_effect"]
        + (0.35 * effects_df["mean_abs_effect"])
        - (0.25 * effects_df["std_effect"])
    )
    effects_df = effects_df.replace([np.inf, -np.inf], np.nan)
    effects_df = effects_df.dropna(subset=["min_abs_effect", "mean_abs_effect", "std_effect"]).copy()
    effects_df = effects_df.sort_values(
        ["same_sign_all", "consistency_score", "min_abs_effect", "mean_abs_effect", "std_effect"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    effects_df["global_rank"] = np.arange(1, len(effects_df) + 1, dtype=int)
    effects_df["target_name"] = target_name
    return effects_df


def select_ranked_feature_pool(
    ranking_df: pd.DataFrame,
    *,
    max_features: int,
    per_root_limit: int,
) -> list[str]:
    selected: list[str] = []
    root_counts: dict[str, int] = {}
    for row in ranking_df.itertuples(index=False):
        root_name = str(row.feature_root)
        if root_counts.get(root_name, 0) >= int(per_root_limit):
            continue
        selected.append(str(row.feature))
        root_counts[root_name] = root_counts.get(root_name, 0) + 1
        if len(selected) >= int(max_features):
            break
    return selected


def make_activation_lookup(
    *,
    space_name: str,
    variant: str,
    use_pca: bool,
    hidden_dim: int,
    pca_dim: int,
) -> pd.DataFrame:
    if use_pca:
        feature_names = [f"{space_name}__pc_{idx + 1:03d}" for idx in range(int(pca_dim))]
        metric_name = f"{variant}__pca"
    else:
        feature_names = [f"{space_name}__dim_{idx:04d}" for idx in range(int(hidden_dim))]
        metric_name = f"{variant}__raw"
    return pd.DataFrame(
        [
            {
                "feature": feature_name,
                "feature_root": space_name,
                "metric_name": metric_name,
                "head_summary": "",
                "band": "",
                "band_stat": "",
                "layer_count": pd.NA,
                "family": "activation",
            }
            for feature_name in feature_names
        ]
    )


def make_activation_ranking_df(
    *,
    target_name: str,
    feature_lookup_df: pd.DataFrame,
    start_rank: int = 1,
) -> pd.DataFrame:
    out = feature_lookup_df.copy().reset_index(drop=True)
    for env_name in ENV_ORDER:
        effect_col = f"{slugify(env_name)}_effect"
        out[effect_col] = np.nan
        out[f"{effect_col}_abs"] = np.nan
    out["same_sign_all"] = pd.NA
    out["sign_direction"] = "not_ranked"
    out["min_abs_effect"] = np.nan
    out["mean_abs_effect"] = np.nan
    out["median_abs_effect"] = np.nan
    out["std_effect"] = np.nan
    out["consistency_score"] = np.nan
    out["global_rank"] = np.arange(int(start_rank), int(start_rank) + len(out), dtype=int)
    out["target_name"] = target_name
    return out


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float32)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float32)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite, dtype=np.float32))


def finite_min(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def finite_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite, dtype=np.float32))


@dataclass
class FittedBinaryModel:
    estimator: Pipeline
    chosen_c: float
    decision_threshold: float
    validation_metrics: dict[str, Any]


def build_estimator(*, c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=4000,
                    solver="liblinear",
                    random_state=int(RANDOM_SEED),
                ),
            ),
        ]
    )


def fit_candidate_classifiers(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[list[FittedBinaryModel], pd.DataFrame]:
    candidate_rows: list[dict[str, Any]] = []
    fitted_models: list[FittedBinaryModel] = []

    for c_value in C_GRID:
        estimator = build_estimator(c_value=float(c_value))
        estimator.fit(x_train, y_train)
        val_scores = estimator.predict_proba(x_val)[:, 1].astype(np.float32)
        decision_threshold = choose_decision_threshold(
            y_val,
            val_scores,
            default_threshold=0.5,
            mode=DECISION_THRESHOLD_MODE,
        )
        metrics = summarize_score_metrics(y_val, val_scores, decision_threshold=decision_threshold)
        candidate_rows.append(
            {
                "candidate_c": float(c_value),
                "decision_threshold": float(decision_threshold),
                **metrics,
            }
        )
        fitted_models.append(
            FittedBinaryModel(
                estimator=estimator,
                chosen_c=float(c_value),
                decision_threshold=float(decision_threshold),
                validation_metrics=metrics,
            )
        )

    if not fitted_models:
        raise RuntimeError("fit_candidate_classifiers failed to produce any models.")
    return fitted_models, pd.DataFrame(candidate_rows)


def extract_standardized_coefficients(
    fitted_model: FittedBinaryModel,
    *,
    feature_names: list[str],
    target_name: str,
    feature_space: str,
    train_env: str,
) -> pd.DataFrame:
    model = fitted_model.estimator.named_steps["model"]
    coef = np.asarray(model.coef_, dtype=np.float32).reshape(-1)
    out = pd.DataFrame(
        {
            "target_name": target_name,
            "feature_space": feature_space,
            "train_env": train_env,
            "feature": feature_names,
            "coefficient": coef,
        }
    )
    out["abs_coefficient"] = out["coefficient"].abs()
    return out


def build_model_selection_key(
    *,
    objective: str,
    oracle_mean_ood_auroc: float,
    oracle_mean_ood_average_precision: float,
    source_val_auroc: float,
    source_val_average_precision: float,
    source_val_balanced_accuracy: float,
    feature_count: int,
    chosen_c: float,
) -> tuple[float, ...]:
    if objective == "mean_ood_auroc_oracle":
        return (
            oracle_mean_ood_auroc if np.isfinite(oracle_mean_ood_auroc) else float("-inf"),
            oracle_mean_ood_average_precision if np.isfinite(oracle_mean_ood_average_precision) else float("-inf"),
            source_val_auroc if np.isfinite(source_val_auroc) else float("-inf"),
            source_val_average_precision if np.isfinite(source_val_average_precision) else float("-inf"),
            -int(feature_count),
            -float(chosen_c),
        )
    if objective == "source_val_auroc":
        return (
            source_val_auroc if np.isfinite(source_val_auroc) else float("-inf"),
            source_val_average_precision if np.isfinite(source_val_average_precision) else float("-inf"),
            source_val_balanced_accuracy if np.isfinite(source_val_balanced_accuracy) else float("-inf"),
            -int(feature_count),
            -float(chosen_c),
        )
    raise ValueError(
        "Unsupported MODEL_SELECTION_OBJECTIVE="
        f"{objective!r}. Expected one of ['mean_ood_auroc_oracle', 'source_val_auroc']."
    )


@dataclass
class ActivationMatrixBundle:
    matrices_by_env: dict[str, dict[str, np.ndarray]]
    feature_names: list[str]
    feature_lookup_df: pd.DataFrame
    effective_pca_dim: int | None


def build_attention_matrix_cache(selected_features: list[str]) -> dict[str, dict[str, np.ndarray]]:
    env_cache: dict[str, dict[str, np.ndarray]] = {}
    for env_name in ENV_ORDER:
        env_df = attention_envs[env_name]
        split_bundle = split_cache_by_env[env_name]
        pooled = (
            env_df.loc[:, selected_features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=np.float32, copy=False)
        )
        env_cache[env_name] = {
            "train": np.asarray(pooled[split_bundle["train_mask"]], dtype=np.float32),
            "val": np.asarray(pooled[split_bundle["val_mask"]], dtype=np.float32),
        }
        del pooled
        gc.collect()
    return env_cache


def build_activation_matrix_bundle(
    *,
    train_env: str,
    feature_space_name: str,
    feature_space: FeatureSpaceSpec,
) -> ActivationMatrixBundle:
    if feature_space.activation_variant is None:
        raise ValueError(f"{feature_space_name} does not define an activation variant.")

    raw_train = activation_stores[train_env].load_matrix(
        split_cache_by_env[train_env]["train_row_idx"],
        variant=str(feature_space.activation_variant),
    )
    if raw_train.size == 0:
        raise ValueError(f"{train_env} has no activation rows for {feature_space_name}.")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(raw_train)
    train_scaled = scaler.fit_transform(train_imputed)

    pca: PCA | None = None
    effective_pca_dim: int | None = None
    if feature_space.activation_use_pca:
        effective_pca_dim = int(min(int(ACTIVATION_PCA_DIM), train_scaled.shape[0] - 1, train_scaled.shape[1]))
        if effective_pca_dim < 1:
            raise ValueError(
                f"Effective PCA dim < 1 for {feature_space_name} / {train_env}. "
                f"train_rows={train_scaled.shape[0]}, hidden_dim={train_scaled.shape[1]}"
            )
        pca = PCA(
            n_components=int(effective_pca_dim),
            random_state=int(RANDOM_SEED),
            svd_solver="randomized",
        )
        pca.fit(train_scaled)

    matrices_by_env: dict[str, dict[str, np.ndarray]] = {}
    for env_name in ENV_ORDER:
        matrices_by_env[env_name] = {}
        for split_name in ("train", "val"):
            row_idx = split_cache_by_env[env_name][f"{split_name}_row_idx"]
            raw_matrix = activation_stores[env_name].load_matrix(
                row_idx,
                variant=str(feature_space.activation_variant),
            )
            transformed = scaler.transform(imputer.transform(raw_matrix))
            if pca is not None:
                transformed = pca.transform(transformed)
            matrices_by_env[env_name][split_name] = np.asarray(transformed, dtype=np.float32)

    feature_lookup_df = make_activation_lookup(
        space_name=feature_space_name,
        variant=str(feature_space.activation_variant),
        use_pca=bool(feature_space.activation_use_pca),
        hidden_dim=int(train_scaled.shape[1]),
        pca_dim=int(effective_pca_dim if effective_pca_dim is not None else ACTIVATION_PCA_DIM),
    )
    feature_names = feature_lookup_df["feature"].astype(str).tolist()
    return ActivationMatrixBundle(
        matrices_by_env=matrices_by_env,
        feature_names=feature_names,
        feature_lookup_df=feature_lookup_df,
        effective_pca_dim=effective_pca_dim,
    )


attention_lookup = attention_reduction_lookup_df.loc[
    attention_reduction_lookup_df["feature"].isin(attention_feature_names)
].copy()
activation_lookup_by_space: dict[str, pd.DataFrame] = {}
for feature_space_name, feature_space in FEATURE_SPACES.items():
    if feature_space.activation_variant is None:
        continue
    activation_lookup_by_space[feature_space_name] = make_activation_lookup(
        space_name=feature_space_name,
        variant=str(feature_space.activation_variant),
        use_pca=bool(feature_space.activation_use_pca),
        hidden_dim=int(ACTIVATION_HIDDEN_DIM),
        pca_dim=int(ACTIVATION_PCA_DIM),
    )

ranking_frames: list[pd.DataFrame] = []
ranking_pools: dict[tuple[str, str], list[str]] = {}
ranking_tables_by_key: dict[tuple[str, str], pd.DataFrame] = {}
attention_pools_by_target: dict[str, list[str]] = {}

for target_name in maybe_tqdm(list(TARGET_SPECS.keys()), desc="Rank targets", total=len(TARGET_SPECS), disable=DISABLE_TQDM):
    attention_ranking_df = build_consistency_ranking(
        attention_envs,
        feature_names=attention_feature_names,
        feature_lookup_df=attention_lookup,
        target_name=target_name,
    )
    attention_pool = select_ranked_feature_pool(
        attention_ranking_df,
        max_features=int(ATTENTION_TOP_K),
        per_root_limit=int(PER_ROOT_LIMIT),
    )
    attention_pools_by_target[target_name] = attention_pool

    for feature_space_name, feature_space in FEATURE_SPACES.items():
        if feature_space.uses_attention and feature_space.activation_variant is None:
            ranking_df = attention_ranking_df.copy()
            feature_pool = list(attention_pool)
        elif (not feature_space.uses_attention) and feature_space.activation_variant is not None:
            activation_lookup_df = activation_lookup_by_space[feature_space_name].copy()
            ranking_df = make_activation_ranking_df(
                target_name=target_name,
                feature_lookup_df=activation_lookup_df,
            )
            feature_pool = activation_lookup_df["feature"].astype(str).tolist()
        else:
            activation_lookup_df = activation_lookup_by_space[feature_space_name].copy()
            attention_part = attention_ranking_df.copy()
            activation_part = make_activation_ranking_df(
                target_name=target_name,
                feature_lookup_df=activation_lookup_df,
                start_rank=int(attention_part["global_rank"].max()) + 1,
            )
            ranking_df = pd.concat([attention_part, activation_part], ignore_index=True)
            feature_pool = list(attention_pool) + activation_lookup_df["feature"].astype(str).tolist()

        ranking_tables_by_key[(target_name, feature_space_name)] = ranking_df.copy()
        ranking_pools[(target_name, feature_space_name)] = feature_pool
        ranking_frame = ranking_df.copy()
        ranking_frame["feature_space"] = feature_space_name
        ranking_frames.append(ranking_frame)

ranking_df_all = pd.concat(ranking_frames, ignore_index=True)
ranking_df_all.to_csv(OUTPUT_ROOT / "global_consistency_rankings.csv", index=False)
display(ranking_df_all.head(20))

for (target_name, feature_space_name), feature_pool in ranking_pools.items():
    rank_dir = OUTPUT_ROOT / "rankings" / target_name
    rank_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feature": feature_pool,
            "pool_rank": np.arange(1, len(feature_pool) + 1, dtype=int),
        }
    ).to_csv(rank_dir / f"{feature_space_name}__feature_pool.csv", index=False)

attention_matrix_cache_by_target: dict[str, dict[str, dict[str, np.ndarray]]] = {}
for target_name in maybe_tqdm(
    list(TARGET_SPECS.keys()),
    desc="Cache attention",
    total=len(TARGET_SPECS),
    disable=DISABLE_TQDM,
):
    attention_matrix_cache_by_target[target_name] = build_attention_matrix_cache(attention_pools_by_target[target_name])


# %%
def feature_size_to_label(feature_size: int | None) -> str:
    return "raw_final" if feature_size is None else f"k{int(feature_size):03d}"


def feature_size_options_for_space(feature_space: FeatureSpaceSpec) -> tuple[int | None, ...]:
    if feature_space.family_title == "baseline":
        return (None,)
    return tuple(int(value) for value in FEATURE_SIZE_GRID)


def assemble_split_matrix(
    env_cache: dict[str, np.ndarray],
    *,
    split_name: str,
    feature_space: FeatureSpaceSpec,
    attention_dim: int,
    activation_dim: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if feature_space.uses_attention:
        parts.append(np.asarray(env_cache[f"{split_name}_attention_pool"][:, :attention_dim], dtype=np.float32))
    if feature_space.activation_variant is not None:
        activation_pool = np.asarray(env_cache[f"{split_name}_activation_pool"], dtype=np.float32)
        if feature_space.activation_use_pca:
            parts.append(np.asarray(activation_pool[:, :activation_dim], dtype=np.float32))
        else:
            parts.append(activation_pool)
    if not parts:
        raise ValueError("assemble_split_matrix received no feature parts.")
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)


all_transfer_metric_rows: list[dict[str, Any]] = []
all_model_selection_rows: list[dict[str, Any]] = []
all_coefficient_frames: list[pd.DataFrame] = []
activation_bundle_cache: dict[tuple[str, str], ActivationMatrixBundle] = {}

for target_name in maybe_tqdm(list(TARGET_SPECS.keys()), desc="Model targets", total=len(TARGET_SPECS), disable=DISABLE_TQDM):
    target_title = str(TARGET_SPECS[target_name]["title"])
    attention_feature_pool = list(attention_pools_by_target[target_name])
    for feature_space_name, feature_space in maybe_tqdm(
        FEATURE_SPACES.items(),
        desc=f"Model spaces:{target_name}",
        total=len(FEATURE_SPACES),
        disable=DISABLE_TQDM,
    ):
        ranking_df = ranking_tables_by_key[(target_name, feature_space_name)].copy()
        ranking_meta_df = ranking_df.loc[:, [column for column in ranking_df.columns if column != "feature_space"]].drop_duplicates(
            subset=["feature"],
            keep="first",
        )
        size_options = feature_size_options_for_space(feature_space)

        for train_env in maybe_tqdm(
            ENV_ORDER,
            desc=f"Train envs:{target_name}:{feature_space_name}",
            total=len(ENV_ORDER),
            disable=DISABLE_TQDM,
        ):
            activation_bundle: ActivationMatrixBundle | None = None
            if feature_space.activation_variant is not None:
                bundle_key = (train_env, feature_space_name)
                activation_bundle = activation_bundle_cache.get(bundle_key)
                if activation_bundle is None:
                    activation_bundle = build_activation_matrix_bundle(
                        train_env=train_env,
                        feature_space_name=feature_space_name,
                        feature_space=feature_space,
                    )
                    activation_bundle_cache[bundle_key] = activation_bundle

            env_pool_cache: dict[str, dict[str, np.ndarray]] = {}
            for env_name in ENV_ORDER:
                y_train = np.asarray(split_cache_by_env[env_name][f"y_train__{target_name}"], dtype=np.int8)
                y_val = np.asarray(split_cache_by_env[env_name][f"y_val__{target_name}"], dtype=np.int8)
                if np.unique(y_train).size < 2:
                    raise ValueError(f"{env_name} train split does not contain both classes for {target_name}.")
                if np.unique(y_val).size < 2:
                    raise ValueError(f"{env_name} val split does not contain both classes for {target_name}.")

                env_entry: dict[str, np.ndarray] = {
                    "y_train": y_train,
                    "y_val": y_val,
                }
                if feature_space.uses_attention:
                    env_entry["train_attention_pool"] = attention_matrix_cache_by_target[target_name][env_name]["train"]
                    env_entry["val_attention_pool"] = attention_matrix_cache_by_target[target_name][env_name]["val"]
                if activation_bundle is not None:
                    env_entry["train_activation_pool"] = activation_bundle.matrices_by_env[env_name]["train"]
                    env_entry["val_activation_pool"] = activation_bundle.matrices_by_env[env_name]["val"]
                env_pool_cache[env_name] = env_entry

            for feature_size in size_options:
                size_label = feature_size_to_label(feature_size)
                attention_dim = 0
                activation_dim = 0
                current_feature_names: list[str] = []

                if feature_space.uses_attention:
                    requested_attention = len(attention_feature_pool) if feature_size is None else int(feature_size)
                    attention_dim = int(min(len(attention_feature_pool), requested_attention))
                    current_feature_names.extend(attention_feature_pool[:attention_dim])

                if activation_bundle is not None:
                    if feature_space.activation_use_pca:
                        requested_activation = len(activation_bundle.feature_names) if feature_size is None else int(feature_size)
                        activation_dim = int(min(len(activation_bundle.feature_names), requested_activation))
                        current_feature_names.extend(activation_bundle.feature_names[:activation_dim])
                    else:
                        activation_dim = int(len(activation_bundle.feature_names))
                        current_feature_names.extend(activation_bundle.feature_names)

                if not current_feature_names:
                    raise ValueError(
                        f"No features constructed for {target_name} / {feature_space_name} / {train_env} / {size_label}."
                    )

                train_cache = env_pool_cache[train_env]
                x_train = assemble_split_matrix(
                    train_cache,
                    split_name="train",
                    feature_space=feature_space,
                    attention_dim=attention_dim,
                    activation_dim=activation_dim,
                )
                x_val = assemble_split_matrix(
                    train_cache,
                    split_name="val",
                    feature_space=feature_space,
                    attention_dim=attention_dim,
                    activation_dim=activation_dim,
                )
                fitted_models, candidate_df = fit_candidate_classifiers(
                    x_train,
                    train_cache["y_train"],
                    x_val,
                    train_cache["y_val"],
                )

                best_model: FittedBinaryModel | None = None
                best_key: tuple[float, ...] | None = None
                best_candidate_oracle_metrics: dict[str, float] | None = None
                oracle_rows: list[dict[str, Any]] = []

                for fitted_model in fitted_models:
                    oracle_aurocs: list[float] = []
                    oracle_average_precisions: list[float] = []
                    for test_env in ENV_ORDER:
                        if test_env == train_env:
                            continue
                        eval_cache = env_pool_cache[test_env]
                        x_eval = assemble_split_matrix(
                            eval_cache,
                            split_name="val",
                            feature_space=feature_space,
                            attention_dim=attention_dim,
                            activation_dim=activation_dim,
                        )
                        eval_scores = fitted_model.estimator.predict_proba(x_eval)[:, 1].astype(np.float32)
                        oracle_aurocs.append(safe_auroc(eval_cache["y_val"], eval_scores))
                        oracle_average_precisions.append(safe_average_precision(eval_cache["y_val"], eval_scores))
                    oracle_metrics = {
                        "oracle_mean_ood_auroc": finite_mean(oracle_aurocs),
                        "oracle_min_ood_auroc": finite_min(oracle_aurocs),
                        "oracle_std_ood_auroc": finite_std(oracle_aurocs),
                        "oracle_mean_ood_average_precision": finite_mean(oracle_average_precisions),
                    }
                    oracle_rows.append({"candidate_c": float(fitted_model.chosen_c), **oracle_metrics})
                    candidate_key = build_model_selection_key(
                        objective=MODEL_SELECTION_OBJECTIVE,
                        oracle_mean_ood_auroc=oracle_metrics["oracle_mean_ood_auroc"],
                        oracle_mean_ood_average_precision=oracle_metrics["oracle_mean_ood_average_precision"],
                        source_val_auroc=float(fitted_model.validation_metrics["auroc"]),
                        source_val_average_precision=float(fitted_model.validation_metrics["average_precision"]),
                        source_val_balanced_accuracy=float(fitted_model.validation_metrics["balanced_accuracy"]),
                        feature_count=len(current_feature_names),
                        chosen_c=float(fitted_model.chosen_c),
                    )
                    if best_key is None or candidate_key > best_key:
                        best_key = candidate_key
                        best_model = fitted_model
                        best_candidate_oracle_metrics = oracle_metrics

                if best_model is None or best_candidate_oracle_metrics is None:
                    raise RuntimeError(f"Failed to select model for {target_name} / {feature_space_name} / {train_env} / {size_label}.")

                selection_dir = OUTPUT_ROOT / "model_selection" / target_name / feature_space_name / size_label
                selection_dir.mkdir(parents=True, exist_ok=True)
                candidate_path = selection_dir / f"{slugify(train_env)}__candidates.csv"
                candidate_df = candidate_df.merge(
                    pd.DataFrame(oracle_rows),
                    on="candidate_c",
                    how="left",
                    validate="one_to_one",
                )
                candidate_df["target_name"] = target_name
                candidate_df["feature_space"] = feature_space_name
                candidate_df["feature_space_title"] = feature_space.title
                candidate_df["feature_family_group"] = feature_space.family_title
                candidate_df["train_env"] = train_env
                candidate_df["feature_size"] = pd.NA if feature_size is None else int(feature_size)
                candidate_df["feature_size_label"] = size_label
                candidate_df["attention_feature_count"] = int(attention_dim)
                candidate_df["activation_feature_count"] = int(activation_dim)
                candidate_df["selected_feature_count"] = len(current_feature_names)
                candidate_df["effective_activation_pca_dim"] = (
                    pd.NA if activation_bundle is None or activation_bundle.effective_pca_dim is None else int(activation_bundle.effective_pca_dim)
                )
                candidate_df.to_csv(candidate_path, index=False)

                selected_path = selection_dir / f"{slugify(train_env)}__selected_features.csv"
                coefficients_path = selection_dir / f"{slugify(train_env)}__coefficients.csv"

                selected_df = pd.DataFrame({"feature": current_feature_names})
                if not selected_df.empty:
                    selected_df = selected_df.merge(
                        ranking_meta_df,
                        on="feature",
                        how="left",
                        validate="one_to_one",
                    )
                selected_df["target_name"] = target_name
                selected_df["feature_space"] = feature_space_name
                selected_df["feature_space_title"] = feature_space.title
                selected_df["feature_family_group"] = feature_space.family_title
                selected_df["train_env"] = train_env
                selected_df["feature_size"] = pd.NA if feature_size is None else int(feature_size)
                selected_df["feature_size_label"] = size_label
                selected_df["attention_feature_count"] = int(attention_dim)
                selected_df["activation_feature_count"] = int(activation_dim)
                selected_df["selected_feature_count"] = len(current_feature_names)
                selected_df.to_csv(selected_path, index=False)

                coefficient_df = extract_standardized_coefficients(
                    best_model,
                    feature_names=current_feature_names,
                    target_name=target_name,
                    feature_space=feature_space_name,
                    train_env=train_env,
                )
                coefficient_df["feature_space_title"] = feature_space.title
                coefficient_df["feature_family_group"] = feature_space.family_title
                coefficient_df["feature_size"] = pd.NA if feature_size is None else int(feature_size)
                coefficient_df["feature_size_label"] = size_label
                coefficient_df["attention_feature_count"] = int(attention_dim)
                coefficient_df["activation_feature_count"] = int(activation_dim)
                coefficient_df["selected_feature_count"] = len(current_feature_names)
                coefficient_df.to_csv(coefficients_path, index=False)
                all_coefficient_frames.append(coefficient_df)

                all_model_selection_rows.append(
                    {
                        "target_name": target_name,
                        "target_title": target_title,
                        "feature_space": feature_space_name,
                        "feature_space_title": feature_space.title,
                        "feature_family_group": feature_space.family_title,
                        "train_env": train_env,
                        "feature_size": pd.NA if feature_size is None else int(feature_size),
                        "feature_size_label": size_label,
                        "attention_feature_count": int(attention_dim),
                        "activation_feature_count": int(activation_dim),
                        "selected_feature_count": len(current_feature_names),
                        "chosen_c": float(best_model.chosen_c),
                        "decision_threshold": float(best_model.decision_threshold),
                        "effective_activation_pca_dim": (
                            pd.NA if activation_bundle is None or activation_bundle.effective_pca_dim is None else int(activation_bundle.effective_pca_dim)
                        ),
                        "model_selection_objective": MODEL_SELECTION_OBJECTIVE,
                        "oracle_mean_ood_auroc_selected": float(best_candidate_oracle_metrics["oracle_mean_ood_auroc"]),
                        "oracle_min_ood_auroc_selected": float(best_candidate_oracle_metrics["oracle_min_ood_auroc"]),
                        "oracle_std_ood_auroc_selected": float(best_candidate_oracle_metrics["oracle_std_ood_auroc"]),
                        "oracle_mean_ood_average_precision_selected": float(best_candidate_oracle_metrics["oracle_mean_ood_average_precision"]),
                        "candidate_path": str(candidate_path),
                        "selected_features_path": str(selected_path),
                        "coefficients_path": str(coefficients_path),
                        **best_model.validation_metrics,
                    }
                )

                for test_env in ENV_ORDER:
                    eval_cache = env_pool_cache[test_env]
                    x_eval = assemble_split_matrix(
                        eval_cache,
                        split_name="val",
                        feature_space=feature_space,
                        attention_dim=attention_dim,
                        activation_dim=activation_dim,
                    )
                    eval_scores = best_model.estimator.predict_proba(x_eval)[:, 1].astype(np.float32)
                    eval_metrics = summarize_score_metrics(
                        eval_cache["y_val"],
                        eval_scores,
                        decision_threshold=float(best_model.decision_threshold),
                    )
                    all_transfer_metric_rows.append(
                        {
                            "target_name": target_name,
                            "target_title": target_title,
                            "feature_space": feature_space_name,
                            "feature_space_title": feature_space.title,
                            "feature_family_group": feature_space.family_title,
                            "train_env": train_env,
                            "test_env": test_env,
                            "eval_role": "val" if train_env == test_env else "ood",
                            "feature_size": pd.NA if feature_size is None else int(feature_size),
                            "feature_size_label": size_label,
                            "attention_feature_count": int(attention_dim),
                            "activation_feature_count": int(activation_dim),
                            "selected_feature_count": len(current_feature_names),
                            "chosen_c": float(best_model.chosen_c),
                            "decision_threshold": float(best_model.decision_threshold),
                            "effective_activation_pca_dim": (
                                pd.NA if activation_bundle is None or activation_bundle.effective_pca_dim is None else int(activation_bundle.effective_pca_dim)
                            ),
                            "model_selection_objective": MODEL_SELECTION_OBJECTIVE,
                            "oracle_mean_ood_auroc_selected": float(best_candidate_oracle_metrics["oracle_mean_ood_auroc"]),
                            "selected_features_path": str(selected_path),
                            "coefficients_path": str(coefficients_path),
                            **eval_metrics,
                        }
                    )

                gc.collect()

all_transfer_metrics_df = pd.DataFrame(all_transfer_metric_rows)
all_model_selection_df = pd.DataFrame(all_model_selection_rows)
all_coefficients_df = pd.concat(all_coefficient_frames, ignore_index=True) if all_coefficient_frames else pd.DataFrame()
all_transfer_metrics_df.to_csv(OUTPUT_ROOT / "all_transfer_metrics.csv", index=False)
all_model_selection_df.to_csv(OUTPUT_ROOT / "all_model_selection.csv", index=False)
all_coefficients_df.to_csv(OUTPUT_ROOT / "all_coefficients.csv", index=False)


# %%
def summarize_transfer_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "target_name",
        "target_title",
        "feature_space",
        "feature_space_title",
        "feature_family_group",
        "feature_size",
        "feature_size_label",
    ]
    for group_values, group_df in metrics_df.groupby(group_cols, dropna=False, sort=False):
        target_name, target_title, feature_space, feature_space_title, feature_family_group, feature_size, feature_size_label = group_values
        diagonal_df = group_df.loc[group_df["eval_role"].eq("val")].copy()
        ood_df = group_df.loc[group_df["eval_role"].eq("ood")].copy()
        meta = group_df.iloc[0]
        summary_rows.append(
            {
                "target_name": target_name,
                "target_title": target_title,
                "feature_space": feature_space,
                "feature_space_title": feature_space_title,
                "feature_family_group": feature_family_group,
                "feature_size": feature_size,
                "feature_size_label": feature_size_label,
                "attention_feature_count": int(meta["attention_feature_count"]),
                "activation_feature_count": int(meta["activation_feature_count"]),
                "selected_feature_count": int(meta["selected_feature_count"]),
                "effective_activation_pca_dim": meta.get("effective_activation_pca_dim", pd.NA),
                "mean_val_auroc": safe_metric_mean(diagonal_df["auroc"]),
                "min_val_auroc": safe_metric_min(diagonal_df["auroc"]),
                "mean_ood_auroc": safe_metric_mean(ood_df["auroc"]),
                "min_ood_auroc": safe_metric_min(ood_df["auroc"]),
                "std_ood_auroc": safe_metric_std(ood_df["auroc"]),
                "mean_val_average_precision": safe_metric_mean(diagonal_df["average_precision"]),
                "mean_ood_average_precision": safe_metric_mean(ood_df["average_precision"]),
                "mean_val_balanced_accuracy": safe_metric_mean(diagonal_df["balanced_accuracy"]),
                "mean_ood_balanced_accuracy": safe_metric_mean(ood_df["balanced_accuracy"]),
            }
        )
    if not summary_rows:
        return pd.DataFrame()
    out = pd.DataFrame(summary_rows)
    out["_feature_size_sort"] = pd.to_numeric(out["feature_size"], errors="coerce").fillna(-1).astype(int)
    out = out.sort_values(
        ["target_name", "_feature_size_sort", "mean_ood_auroc", "min_ood_auroc", "mean_val_auroc"],
        ascending=[True, True, False, False, False],
    ).drop(columns="_feature_size_sort")
    return out.reset_index(drop=True)


def summarize_train_env_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "target_name",
        "target_title",
        "feature_space",
        "feature_space_title",
        "feature_family_group",
        "feature_size",
        "feature_size_label",
        "train_env",
    ]
    for group_values, group_df in metrics_df.groupby(group_cols, dropna=False, sort=False):
        (
            target_name,
            target_title,
            feature_space,
            feature_space_title,
            feature_family_group,
            feature_size,
            feature_size_label,
            train_env,
        ) = group_values
        diagonal_df = group_df.loc[group_df["eval_role"].eq("val")].copy()
        ood_df = group_df.loc[group_df["eval_role"].eq("ood")].copy()
        selection_meta = group_df.iloc[0]
        summary_rows.append(
            {
                "target_name": target_name,
                "target_title": target_title,
                "feature_space": feature_space,
                "feature_space_title": feature_space_title,
                "feature_family_group": feature_family_group,
                "feature_size": feature_size,
                "feature_size_label": feature_size_label,
                "train_env": train_env,
                "model_selection_objective": selection_meta.get("model_selection_objective", MODEL_SELECTION_OBJECTIVE),
                "source_val_auroc": safe_metric_mean(diagonal_df["auroc"]),
                "source_val_average_precision": safe_metric_mean(diagonal_df["average_precision"]),
                "mean_ood_auroc": safe_metric_mean(ood_df["auroc"]),
                "min_ood_auroc": safe_metric_min(ood_df["auroc"]),
                "std_ood_auroc": safe_metric_std(ood_df["auroc"]),
                "mean_ood_average_precision": safe_metric_mean(ood_df["average_precision"]),
                "mean_ood_balanced_accuracy": safe_metric_mean(ood_df["balanced_accuracy"]),
                "attention_feature_count": int(selection_meta["attention_feature_count"]),
                "activation_feature_count": int(selection_meta["activation_feature_count"]),
                "selected_feature_count": int(selection_meta["selected_feature_count"]),
                "chosen_c": float(selection_meta["chosen_c"]),
                "decision_threshold": float(selection_meta["decision_threshold"]),
                "effective_activation_pca_dim": selection_meta.get("effective_activation_pca_dim", pd.NA),
                "oracle_mean_ood_auroc_selected": float(selection_meta.get("oracle_mean_ood_auroc_selected", float("nan"))),
                "selected_features_path": str(selection_meta["selected_features_path"]),
                "coefficients_path": str(selection_meta.get("coefficients_path", "")),
            }
        )
    if not summary_rows:
        return pd.DataFrame()
    out = pd.DataFrame(summary_rows)
    out["_feature_size_sort"] = pd.to_numeric(out["feature_size"], errors="coerce").fillna(-1).astype(int)
    out = out.sort_values(
        ["target_name", "_feature_size_sort", "mean_ood_auroc", "min_ood_auroc", "source_val_auroc"],
        ascending=[True, True, False, False, False],
    ).drop(columns="_feature_size_sort")
    return out.reset_index(drop=True)


def summarize_confusion_counts(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "target_name",
        "target_title",
        "feature_space",
        "feature_space_title",
        "feature_family_group",
        "feature_size",
        "feature_size_label",
        "eval_role",
    ]
    for group_values, group_df in metrics_df.groupby(group_cols, dropna=False, sort=False):
        (
            target_name,
            target_title,
            feature_space,
            feature_space_title,
            feature_family_group,
            feature_size,
            feature_size_label,
            eval_role,
        ) = group_values
        meta = group_df.iloc[0]
        summary_rows.append(
            {
                "target_name": target_name,
                "target_title": target_title,
                "feature_space": feature_space,
                "feature_space_title": feature_space_title,
                "feature_family_group": feature_family_group,
                "feature_size": feature_size,
                "feature_size_label": feature_size_label,
                "eval_role": eval_role,
                "attention_feature_count": int(meta["attention_feature_count"]),
                "activation_feature_count": int(meta["activation_feature_count"]),
                "selected_feature_count": int(meta["selected_feature_count"]),
                "sum_tn": int(group_df["tn"].sum()),
                "sum_fp": int(group_df["fp"].sum()),
                "sum_fn": int(group_df["fn"].sum()),
                "sum_tp": int(group_df["tp"].sum()),
                "n_pairs": int(len(group_df)),
            }
        )
    if not summary_rows:
        return pd.DataFrame()
    out = pd.DataFrame(summary_rows)
    out["_feature_size_sort"] = pd.to_numeric(out["feature_size"], errors="coerce").fillna(-1).astype(int)
    out = out.sort_values(
        ["target_name", "_feature_size_sort", "eval_role", "feature_space"],
        ascending=[True, True, True, True],
    ).drop(columns="_feature_size_sort")
    return out.reset_index(drop=True)


def summarize_coefficients(coefficients_df: pd.DataFrame, ranking_df: pd.DataFrame) -> pd.DataFrame:
    if coefficients_df.empty:
        return pd.DataFrame()
    coefficient_summary_df = (
        coefficients_df.groupby(["target_name", "feature_space", "feature_size", "feature_size_label", "feature"], as_index=False)
        .agg(
            mean_coefficient=("coefficient", "mean"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
            selected_in_sources=("train_env", "nunique"),
        )
        .sort_values(
            ["target_name", "feature_space", "feature_size_label", "mean_abs_coefficient"],
            ascending=[True, True, True, False],
        )
        .reset_index(drop=True)
    )
    ranking_cols = [
        "target_name",
        "feature_space",
        "feature",
        "feature_root",
        "family",
        "metric_name",
        "head_summary",
        "band",
        "band_stat",
        "global_rank",
        "same_sign_all",
        "sign_direction",
        "consistency_score",
        "min_abs_effect",
        "mean_abs_effect",
        "median_abs_effect",
        "std_effect",
    ]
    ranking_summary_df = ranking_df.loc[:, ranking_cols].drop_duplicates(
        subset=["target_name", "feature_space", "feature"],
        keep="first",
    )
    return coefficient_summary_df.merge(
        ranking_summary_df,
        on=["target_name", "feature_space", "feature"],
        how="left",
        validate="many_to_one",
    )


def build_family_panel_selection(transfer_summary_df: pd.DataFrame) -> pd.DataFrame:
    family_sort = {family: idx for idx, family in enumerate(FAMILY_PANEL_ORDER)}
    rows: list[dict[str, Any]] = []
    for target_name, target_spec in TARGET_SPECS.items():
        target_title = str(target_spec["title"])
        for requested_feature_size in FEATURE_SIZE_GRID:
            for family in FAMILY_PANEL_ORDER:
                subset = transfer_summary_df.loc[
                    transfer_summary_df["target_name"].eq(target_name)
                    & transfer_summary_df["feature_family_group"].eq(family)
                ].copy()
                if family != "baseline":
                    subset = subset.loc[pd.to_numeric(subset["feature_size"], errors="coerce").eq(int(requested_feature_size))]
                if subset.empty:
                    continue
                subset = subset.sort_values(
                    ["mean_ood_auroc", "min_ood_auroc", "mean_val_auroc", "selected_feature_count"],
                    ascending=[False, False, False, True],
                ).reset_index(drop=True)
                best = subset.iloc[0]
                rows.append(
                    {
                        "target_name": target_name,
                        "target_title": target_title,
                        "requested_feature_size": int(requested_feature_size),
                        "requested_feature_size_label": feature_size_to_label(int(requested_feature_size)),
                        "feature_family_group": family,
                        "selected_feature_space": str(best["feature_space"]),
                        "selected_feature_space_title": str(best["feature_space_title"]),
                        "source_feature_size": best["feature_size"],
                        "source_feature_size_label": str(best["feature_size_label"]),
                        "attention_feature_count": int(best["attention_feature_count"]),
                        "activation_feature_count": int(best["activation_feature_count"]),
                        "selected_feature_count": int(best["selected_feature_count"]),
                        "effective_activation_pca_dim": best.get("effective_activation_pca_dim", pd.NA),
                        "mean_val_auroc": float(best["mean_val_auroc"]),
                        "mean_ood_auroc": float(best["mean_ood_auroc"]),
                        "min_ood_auroc": float(best["min_ood_auroc"]),
                        "std_ood_auroc": float(best["std_ood_auroc"]),
                        "mean_ood_average_precision": float(best["mean_ood_average_precision"]),
                    }
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["_family_sort"] = out["feature_family_group"].map(family_sort).fillna(len(family_sort)).astype(int)
    out = out.sort_values(
        ["target_name", "requested_feature_size", "_family_sort", "mean_ood_auroc"],
        ascending=[True, True, True, False],
    ).drop(columns="_family_sort")
    return out.reset_index(drop=True)


def build_best_family_models(
    train_env_model_summary_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel_row in panel_selection_df.itertuples(index=False):
        subset = train_env_model_summary_df.loc[
            train_env_model_summary_df["target_name"].eq(panel_row.target_name)
            & train_env_model_summary_df["feature_space"].eq(panel_row.selected_feature_space)
            & train_env_model_summary_df["feature_size_label"].eq(panel_row.source_feature_size_label)
        ].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["mean_ood_auroc", "min_ood_auroc", "source_val_auroc"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        best = subset.iloc[0]
        rows.append(
            {
                "target_name": panel_row.target_name,
                "target_title": panel_row.target_title,
                "requested_feature_size": int(panel_row.requested_feature_size),
                "requested_feature_size_label": panel_row.requested_feature_size_label,
                "feature_family_group": panel_row.feature_family_group,
                "feature_space": str(best["feature_space"]),
                "feature_space_title": str(best["feature_space_title"]),
                "feature_size": best["feature_size"],
                "feature_size_label": str(best["feature_size_label"]),
                "train_env": str(best["train_env"]),
                "source_val_auroc": float(best["source_val_auroc"]),
                "mean_ood_auroc": float(best["mean_ood_auroc"]),
                "min_ood_auroc": float(best["min_ood_auroc"]),
                "std_ood_auroc": float(best["std_ood_auroc"]),
                "mean_ood_average_precision": float(best["mean_ood_average_precision"]),
                "mean_ood_balanced_accuracy": float(best["mean_ood_balanced_accuracy"]),
                "attention_feature_count": int(best["attention_feature_count"]),
                "activation_feature_count": int(best["activation_feature_count"]),
                "selected_feature_count": int(best["selected_feature_count"]),
                "chosen_c": float(best["chosen_c"]),
                "decision_threshold": float(best["decision_threshold"]),
                "effective_activation_pca_dim": best.get("effective_activation_pca_dim", pd.NA),
                "selected_features_path": str(best["selected_features_path"]),
                "coefficients_path": str(best["coefficients_path"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    family_sort = {family: idx for idx, family in enumerate(FAMILY_PANEL_ORDER)}
    out["_family_sort"] = out["feature_family_group"].map(family_sort).fillna(len(family_sort)).astype(int)
    out = out.sort_values(
        ["target_name", "requested_feature_size", "_family_sort", "mean_ood_auroc"],
        ascending=[True, True, True, False],
    ).drop(columns="_family_sort")
    return out.reset_index(drop=True)


def build_selected_panel_confusion_summary(
    metrics_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
    *,
    eval_role: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel_row in panel_selection_df.itertuples(index=False):
        subset = metrics_df.loc[
            metrics_df["target_name"].eq(panel_row.target_name)
            & metrics_df["feature_space"].eq(panel_row.selected_feature_space)
            & metrics_df["feature_size_label"].eq(panel_row.source_feature_size_label)
            & metrics_df["eval_role"].eq(eval_role)
        ].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "target_name": panel_row.target_name,
                "target_title": panel_row.target_title,
                "requested_feature_size": int(panel_row.requested_feature_size),
                "requested_feature_size_label": panel_row.requested_feature_size_label,
                "feature_family_group": panel_row.feature_family_group,
                "feature_space": panel_row.selected_feature_space,
                "feature_space_title": panel_row.selected_feature_space_title,
                "feature_size_label": panel_row.source_feature_size_label,
                "eval_role": eval_role,
                "sum_tn": int(subset["tn"].sum()),
                "sum_fp": int(subset["fp"].sum()),
                "sum_fn": int(subset["fn"].sum()),
                "sum_tp": int(subset["tp"].sum()),
                "n_pairs": int(len(subset)),
            }
        )
    if not rows:
        return pd.DataFrame()
    family_sort = {family: idx for idx, family in enumerate(FAMILY_PANEL_ORDER)}
    out = pd.DataFrame(rows)
    out["_family_sort"] = out["feature_family_group"].map(family_sort).fillna(len(family_sort)).astype(int)
    out = out.sort_values(
        ["target_name", "requested_feature_size", "_family_sort"],
        ascending=[True, True, True],
    ).drop(columns="_family_sort")
    return out.reset_index(drop=True)


transfer_summary_df = summarize_transfer_metrics(all_transfer_metrics_df)
train_env_model_summary_df = summarize_train_env_models(all_transfer_metrics_df)
confusion_summary_df = summarize_confusion_counts(all_transfer_metrics_df)
coefficient_summary_df = summarize_coefficients(all_coefficients_df, ranking_df_all)
family_panel_selection_df = build_family_panel_selection(transfer_summary_df)
best_family_models_df = build_best_family_models(train_env_model_summary_df, family_panel_selection_df)
selected_panel_confusion_ood_df = build_selected_panel_confusion_summary(
    all_transfer_metrics_df,
    family_panel_selection_df,
    eval_role="ood",
)

top_feature_tables: list[pd.DataFrame] = []
top_feature_dir = OUTPUT_ROOT / "best_features"
top_feature_dir.mkdir(parents=True, exist_ok=True)
for row in best_family_models_df.itertuples(index=False):
    feature_table = (
        all_coefficients_df.loc[
            all_coefficients_df["target_name"].eq(row.target_name)
            & all_coefficients_df["feature_space"].eq(row.feature_space)
            & all_coefficients_df["feature_size_label"].eq(row.feature_size_label)
            & all_coefficients_df["train_env"].eq(row.train_env)
        ]
        .merge(
            ranking_df_all.loc[
                ranking_df_all["target_name"].eq(row.target_name)
                & ranking_df_all["feature_space"].eq(row.feature_space)
            ].drop(columns=["target_name", "feature_space"], errors="ignore"),
            on="feature",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["abs_coefficient", "global_rank"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    if feature_table.empty:
        continue
    feature_table["requested_feature_size"] = int(row.requested_feature_size)
    feature_table["requested_feature_size_label"] = row.requested_feature_size_label
    feature_table["feature_family_group"] = row.feature_family_group
    feature_table.insert(0, "importance_rank", np.arange(1, len(feature_table) + 1, dtype=int))
    top_feature_df = feature_table.head(int(TOP_FEATURES_TO_SHOW)).copy()
    top_feature_tables.append(top_feature_df)
    out_name = f"{row.target_name}__{row.feature_family_group}__{row.requested_feature_size_label}__top_features.csv"
    top_feature_df.to_csv(top_feature_dir / out_name, index=False)

top_features_for_best_models_df = pd.concat(top_feature_tables, ignore_index=True) if top_feature_tables else pd.DataFrame()

display(transfer_summary_df)
display(train_env_model_summary_df)
display(confusion_summary_df)
display(family_panel_selection_df)
display(best_family_models_df)
display(top_features_for_best_models_df)

transfer_summary_df.to_csv(OUTPUT_ROOT / "transfer_summary.csv", index=False)
train_env_model_summary_df.to_csv(OUTPUT_ROOT / "train_env_model_summary.csv", index=False)
confusion_summary_df.to_csv(OUTPUT_ROOT / "confusion_summary.csv", index=False)
coefficient_summary_df.to_csv(OUTPUT_ROOT / "coefficient_summary.csv", index=False)
family_panel_selection_df.to_csv(OUTPUT_ROOT / "best_feature_space_by_target_size_family.csv", index=False)
best_family_models_df.to_csv(OUTPUT_ROOT / "best_model_by_target_size_family.csv", index=False)
selected_panel_confusion_ood_df.to_csv(OUTPUT_ROOT / "selected_panel_confusion_ood.csv", index=False)
top_features_for_best_models_df.to_csv(OUTPUT_ROOT / "top_features_for_best_models.csv", index=False)


# %%
def target_confusion_labels(target_name: str) -> tuple[str, str]:
    target_spec = TARGET_SPECS[target_name]
    return str(target_spec["negative_label"]), str(target_spec["positive_label"])


def build_transfer_matrix_for_selection(
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
    matrix = pd.DataFrame(index=ENV_ORDER, columns=ENV_ORDER, dtype=float)
    for row in subset.itertuples(index=False):
        matrix.loc[str(row.train_env), str(row.test_env)] = float(row.auroc) if pd.notna(row.auroc) else float("nan")
    return matrix


def build_confusion_matrix_df(
    confusion_df: pd.DataFrame,
    *,
    target_name: str,
    requested_feature_size: int,
    feature_family_group: str,
) -> pd.DataFrame | None:
    subset = confusion_df.loc[
        confusion_df["target_name"].eq(target_name)
        & confusion_df["requested_feature_size"].eq(int(requested_feature_size))
        & confusion_df["feature_family_group"].eq(feature_family_group)
    ].copy()
    if subset.empty:
        return None
    row = subset.iloc[0]
    negative_label, positive_label = target_confusion_labels(target_name)
    return pd.DataFrame(
        [
            [int(row["sum_tn"]), int(row["sum_fp"])],
            [int(row["sum_fn"]), int(row["sum_tp"])],
        ],
        index=[f"Actual {negative_label}", f"Actual {positive_label}"],
        columns=[f"Pred {negative_label}", f"Pred {positive_label}"],
    )


def export_selected_family_panel_tables(
    metrics_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
) -> None:
    export_root = OUTPUT_ROOT / "selected_family_panel_tables"
    export_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for panel_row in panel_selection_df.itertuples(index=False):
        panel_dir = export_root / panel_row.target_name / str(panel_row.requested_feature_size_label)
        panel_dir.mkdir(parents=True, exist_ok=True)

        matrix_df = build_transfer_matrix_for_selection(
            metrics_df,
            target_name=str(panel_row.target_name),
            feature_space=str(panel_row.selected_feature_space),
            feature_size_label=str(panel_row.source_feature_size_label),
        )
        matrix_path = panel_dir / f"{panel_row.feature_family_group}__auroc_matrix.csv"
        matrix_df.to_csv(matrix_path, index=True)

        confusion_matrix_df = build_confusion_matrix_df(
            confusion_df,
            target_name=str(panel_row.target_name),
            requested_feature_size=int(panel_row.requested_feature_size),
            feature_family_group=str(panel_row.feature_family_group),
        )
        confusion_path = panel_dir / f"{panel_row.feature_family_group}__ood_confusion_counts.csv"
        if confusion_matrix_df is not None:
            confusion_matrix_df.to_csv(confusion_path, index=True)

        manifest_rows.append(
            {
                "target_name": panel_row.target_name,
                "target_title": panel_row.target_title,
                "requested_feature_size": int(panel_row.requested_feature_size),
                "requested_feature_size_label": panel_row.requested_feature_size_label,
                "feature_family_group": panel_row.feature_family_group,
                "selected_feature_space": panel_row.selected_feature_space,
                "selected_feature_space_title": panel_row.selected_feature_space_title,
                "source_feature_size_label": panel_row.source_feature_size_label,
                "attention_feature_count": int(panel_row.attention_feature_count),
                "activation_feature_count": int(panel_row.activation_feature_count),
                "selected_feature_count": int(panel_row.selected_feature_count),
                "auroc_matrix_path": str(matrix_path),
                "ood_confusion_counts_path": str(confusion_path if confusion_matrix_df is not None else ""),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(export_root / "panel_table_manifest.csv", index=False)


def plot_feature_size_family_transfer_panels(
    metrics_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
    *,
    target_name: str,
    requested_feature_size: int,
) -> None:
    selection_subset = (
        panel_selection_df.loc[
            panel_selection_df["target_name"].eq(target_name)
            & panel_selection_df["requested_feature_size"].eq(int(requested_feature_size))
        ]
        .set_index("feature_family_group")
        .reindex(list(FAMILY_PANEL_ORDER))
        .reset_index()
    )
    matrices: dict[str, pd.DataFrame] = {}
    finite_values_list: list[np.ndarray] = []
    for row in selection_subset.itertuples(index=False):
        if pd.isna(row.selected_feature_space):
            continue
        matrix_df = build_transfer_matrix_for_selection(
            metrics_df,
            target_name=target_name,
            feature_space=str(row.selected_feature_space),
            feature_size_label=str(row.source_feature_size_label),
        )
        matrices[str(row.feature_family_group)] = matrix_df
        finite_values_list.append(matrix_df.to_numpy(dtype=float).ravel())

    finite_values = np.concatenate(finite_values_list) if finite_values_list else np.array([], dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    vmin = float(np.min(finite_values)) if finite_values.size else 0.0
    vmax = float(np.max(finite_values)) if finite_values.size else 1.0
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 2)
    image = None

    for idx, family in enumerate(FAMILY_PANEL_ORDER):
        ax = axes.flat[idx]
        row_df = selection_subset.loc[selection_subset["feature_family_group"].eq(family)]
        if row_df.empty or family not in matrices:
            ax.axis("off")
            continue
        row = row_df.iloc[0]
        matrix_df = matrices[family].reindex(index=ENV_ORDER, columns=ENV_ORDER)
        matrix = np.ma.masked_invalid(matrix_df.to_numpy(dtype=float))
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(ENV_ORDER)))
        ax.set_xticklabels(ENV_ORDER, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(ENV_ORDER)))
        ax.set_yticklabels(ENV_ORDER)
        ax.set_xlabel("Test env")
        ax.set_ylabel("Train env")
        title = (
            f"{FAMILY_DISPLAY_TITLES.get(family, str(row['feature_family_group']))}\n"
            f"{row['selected_feature_space_title']}\n"
            f"attn={int(row['attention_feature_count'])}, act={int(row['activation_feature_count'])}"
        )
        ax.set_title(title, fontsize=10.0)
        midpoint = (vmin + vmax) / 2.0 if np.isfinite(vmin) and np.isfinite(vmax) else 0.5
        for r in range(matrix_df.shape[0]):
            for c in range(matrix_df.shape[1]):
                value = matrix_df.iat[r, c]
                text = "nan" if not np.isfinite(value) else f"{value:.3f}"
                text_color = "white" if np.isfinite(value) and value < midpoint else "black"
                ax.text(c, r, text, ha="center", va="center", color=text_color, fontsize=8.0)

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="AUROC")

    fig.suptitle(
        f"{TARGET_SPECS[target_name]['title']} | feature size {requested_feature_size}\n"
        "(diagonal = source validation AUROC, off-diagonal = OOD AUROC)",
        fontsize=14,
    )
    out_path = OUTPUT_ROOT / f"{target_name}__k{int(requested_feature_size):03d}__family_transfer_heatmaps.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.show()


def plot_feature_size_family_confusion_panels(
    confusion_df: pd.DataFrame,
    *,
    target_name: str,
    requested_feature_size: int,
) -> None:
    subset = (
        confusion_df.loc[
            confusion_df["target_name"].eq(target_name)
            & confusion_df["requested_feature_size"].eq(int(requested_feature_size))
        ]
        .set_index("feature_family_group")
        .reindex(list(FAMILY_PANEL_ORDER))
        .reset_index()
    )
    negative_label, positive_label = target_confusion_labels(target_name)
    vmax = 1.0
    for row in subset.itertuples(index=False):
        if pd.isna(row.sum_tn):
            continue
        vmax = max(vmax, float(max(row.sum_tn, row.sum_fp, row.sum_fn, row.sum_tp)))

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.6), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 2)
    image = None

    for idx, family in enumerate(FAMILY_PANEL_ORDER):
        ax = axes.flat[idx]
        row_df = subset.loc[subset["feature_family_group"].eq(family)]
        if row_df.empty or row_df[["sum_tn", "sum_fp", "sum_fn", "sum_tp"]].isna().all(axis=None):
            ax.axis("off")
            continue
        row = row_df.iloc[0]
        matrix = np.array(
            [
                [float(row["sum_tn"]), float(row["sum_fp"])],
                [float(row["sum_fn"]), float(row["sum_tp"])],
            ],
            dtype=float,
        )
        image = ax.imshow(matrix, vmin=0.0, vmax=vmax, cmap=plt.cm.Blues)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Pred {negative_label}", f"Pred {positive_label}"], rotation=15, ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"Actual {negative_label}", f"Actual {positive_label}"])
        ax.set_title(
            f"{FAMILY_DISPLAY_TITLES.get(family, str(row['feature_family_group']))}\n"
            f"{row['feature_space_title']}\n"
            f"n_pairs={int(row['n_pairs'])}",
            fontsize=10.0,
        )
        midpoint = vmax / 2.0
        for r in range(2):
            for c in range(2):
                value = matrix[r, c]
                ax.text(
                    c,
                    r,
                    f"{int(round(value))}",
                    ha="center",
                    va="center",
                    color="white" if value > midpoint else "black",
                    fontsize=9.0,
                )

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Count")
    fig.suptitle(
        f"Summed OOD raw confusion counts for {TARGET_SPECS[target_name]['title']} | feature size {requested_feature_size}",
        fontsize=14,
    )
    out_path = OUTPUT_ROOT / f"{target_name}__k{int(requested_feature_size):03d}__ood_family_confusion_counts.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.show()


export_selected_family_panel_tables(
    all_transfer_metrics_df,
    family_panel_selection_df,
    selected_panel_confusion_ood_df,
)


for target_name in TARGET_SPECS:
    print(f"\nTarget: {TARGET_SPECS[target_name]['title']}")
    for requested_feature_size in FEATURE_SIZE_GRID:
        print(f"\nFeature size: {requested_feature_size}")
        selection_view = family_panel_selection_df.loc[
            family_panel_selection_df["target_name"].eq(target_name)
            & family_panel_selection_df["requested_feature_size"].eq(int(requested_feature_size))
        ]
        display(selection_view)
        plot_feature_size_family_transfer_panels(
            all_transfer_metrics_df,
            family_panel_selection_df,
            target_name=target_name,
            requested_feature_size=int(requested_feature_size),
        )
        plot_feature_size_family_confusion_panels(
            selected_panel_confusion_ood_df,
            target_name=target_name,
            requested_feature_size=int(requested_feature_size),
        )

print(f"Model selection objective: {MODEL_SELECTION_OBJECTIVE}")
print(f"Fixed C grid: {C_GRID}")
print("\nBest feature-space variant by target, feature size, and family:")
display(family_panel_selection_df)
print("\nBest train-env model by target, feature size, and family:")
display(best_family_models_df)
print(f"\nTop {TOP_FEATURES_TO_SHOW} features for the best model in each target/size/family panel:")
display(top_features_for_best_models_df)
print("\nOutputs written to:")
print(OUTPUT_ROOT)
