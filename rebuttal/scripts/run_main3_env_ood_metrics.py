#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import re
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # noqa: BLE001
    XGBClassifier = None

THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
REBUTTAL_ROOT = SCRIPT_DIR.parent
REPO_ROOT = REBUTTAL_ROOT.parent
NOTEBOOK_ROOT = REPO_ROOT / "Notebooks"
SRC_ROOT = REPO_ROOT / "src"

for search_root in (SCRIPT_DIR, NOTEBOOK_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import deception_prefix_feature_and_activation_extractor as extractor
import deception_prefix_text_structural_baseline_extractor as baseline_extractor
import ood_modeling_main_lib as oml
from commitment_rebuttal_lib import (
    calibration_curve_frame,
    fpr_at_fixed_recalls,
    safe_binary_metrics,
)

slugify = oml.slugify
choose_decision_threshold = oml.choose_decision_threshold
summarize_score_metrics = oml.summarize_score_metrics
safe_metric_mean = oml.safe_metric_mean
safe_metric_min = oml.safe_metric_min
safe_metric_std = oml.safe_metric_std
build_common_layer_roots = oml.build_common_layer_roots
classify_feature_family = oml.classify_feature_family

DEFAULT_DATASET_ROOT = REPO_ROOT / "DatasetMain"
DEFAULT_RESULTS_ROOT = REBUTTAL_ROOT / "results" / "OOD_Modeling_main3_env_ood_metrics_qwen14b_xgb_pca_128"
DEFAULT_FEATURE_SPACES = (
    "baseline_tfidf_last_sentence_text",
    "baseline_tfidf_prefix_text",
    "attention_only",
    "attention_grounding_only",
    "attention_concentration_only",
    "attention_grounding_transition_only",
    "attention_concentration_transition_only",
    "activation_pca_final",
    "activation_pca_delta_last2",
    "activation_pca_delta_prev4mean",
    "attention_plus_activation_pca_final",
    "attention_plus_activation_pca_delta_last2",
    "attention_plus_activation_pca_delta_prev4mean",
    "baseline_raw_final",
)
DEFAULT_MODEL_PRESET = "qwen14b"
DEFAULT_SCENARIOS = ("holdout_env_ood",)
DEFAULT_FEATURE_SIZES = (128,)
DEFAULT_FIXED_RECALL_LEVELS = (0.5, 0.8, 0.9, 0.95)

ATTENTION_METRIC_NAMES = tuple(extractor.ATTENTION_METRIC_NAMES)
HEAD_SUMMARY_NAMES = tuple(extractor.HEAD_SUMMARY_NAMES)
ATTENTION_TRANSITION_PREFIXES = tuple(extractor.TRANSITION_PREFIXES)
BAND_NAMES = ("early", "mid", "late")
BAND_STAT_NAMES = ("mean", "min", "max", "std")
GROUNDING_ATTENTION_METRIC_NAMES = (
    "current_vs_prior",
    "current_vs_prev",
    "current_vs_prev3",
    "recent_vs_early",
    "prev_share_of_prior",
    "current_share_total",
)
CONCENTRATION_ATTENTION_METRIC_NAMES = (
    "entropy_prior",
    "entropy_full",
    "top1_prior",
    "top5_prior",
    "herfindahl_prior",
    "effective_support_prior",
)
ATTENTION_METRIC_GROUP_BY_NAME = {
    **{metric_name: "grounding" for metric_name in GROUNDING_ATTENTION_METRIC_NAMES},
    **{metric_name: "concentration" for metric_name in CONCENTRATION_ATTENTION_METRIC_NAMES},
}

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
MODEL_SPECS = OrderedDict(
    [
        (
            "gptoss20b",
            {
                "dirname": "gpt-oss-20b",
                "display": "GPT-OSS-20B",
                "aliases": ("gptoss20b", "gpt-oss-20b", "gpt_oss_20b", "gptoss"),
            },
        ),
        (
            "llama8b",
            {
                "dirname": "DeepSeek-R1-Distill-Llama-8B",
                "display": "Llama-8B",
                "aliases": ("llama8b", "llama-8b", "deepseek-r1-distill-llama-8b"),
            },
        ),
        (
            "qwen7b",
            {
                "dirname": "DeepSeek-R1-Distill-Qwen-7B",
                "display": "Qwen-7B",
                "aliases": ("qwen7b", "qwen-7b", "deepseek-r1-distill-qwen-7b"),
            },
        ),
        (
            "qwen14b",
            {
                "dirname": "DeepSeek-R1-Distill-Qwen-14B",
                "display": "Qwen-14B",
                "aliases": ("qwen14b", "qwen-14b", "deepseek-r1-distill-qwen-14b"),
            },
        ),
    ]
)
MODEL_KEY_BY_ALIAS = {
    alias: model_key
    for model_key, spec in MODEL_SPECS.items()
    for alias in spec["aliases"]
}

FEATURE_FILENAME_CANDIDATES = (
    "prefix_deception_features.parquet",
    "prefix_deception_features.parquet.tmp",
)
ACTIVATION_FILENAME = "prefix_deception_activations.h5"
STRUCTURAL_BASELINE_FILENAME = getattr(
    baseline_extractor,
    "DEFAULT_OUTPUT_NAME",
    "commitment_text_structural_baselines.parquet",
)
TFIDF_CACHE_DIRNAME = getattr(
    baseline_extractor,
    "DEFAULT_TFIDF_CACHE_DIRNAME",
    "commitment_text_baseline_tfidf_cache",
)
SCENARIO_TITLES = OrderedDict(
    [
        ("single_source_ood", "Train on 1 environment; evaluate OOD on the other 4"),
        ("holdout_env_ood", "Train on 4 environments; evaluate OOD on the held-out environment"),
    ]
)

ATTN_ROOT_RE = re.compile(
    r"^(?:(?P<transition_prefix>"
    + "|".join(re.escape(prefix) for prefix in ATTENTION_TRANSITION_PREFIXES)
    + r")_)?(?P<metric>"
    + "|".join(re.escape(metric) for metric in ATTENTION_METRIC_NAMES)
    + r")_(?P<head_summary>"
    + "|".join(re.escape(name) for name in HEAD_SUMMARY_NAMES)
    + r")$"
)


@dataclass(frozen=True)
class FeatureSpaceSpec:
    name: str
    title: str
    family_title: str
    uses_attention: bool
    attention_subset_key: str | None
    activation_variant: str | None
    activation_use_pca: bool
    baseline_variant: str | None = None
    baseline_text_field: str | None = None


@dataclass(frozen=True)
class BundleSpec:
    env_name: str
    env_dir: str
    model_key: str
    model_dirname: str
    model_display: str
    feature_path: Path | None
    activation_path: Path | None
    structural_baseline_path: Path
    tfidf_cache_dir: Path

    @property
    def bundle_key(self) -> str:
        return self.env_name


@dataclass
class ActivationStore:
    bundle_key: str
    activation_path: Path
    hidden_dim: int
    metadata_df: pd.DataFrame

    def load_matrix(self, row_indices: np.ndarray, *, variant: str, hidden_dim: int | None = None) -> np.ndarray:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.size == 0:
            width = int(hidden_dim if hidden_dim is not None else self.hidden_dim)
            return np.zeros((0, width), dtype=np.float32)

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

        output = np.asarray(feature_matrix[unsort], dtype=np.float32)
        if hidden_dim is not None:
            output = output[:, : int(hidden_dim)]
        return output


@dataclass
class BaselineMatrixBundle:
    matrices_by_bundle: dict[str, dict[str, Any]]
    feature_names: list[str]
    feature_lookup_df: pd.DataFrame
    alignment_mode: str


@dataclass
class ActivationMatrixBundle:
    matrices_by_bundle: dict[str, dict[str, np.ndarray]]
    feature_names: list[str]
    feature_lookup_df: pd.DataFrame
    effective_pca_dim: int | None
    common_hidden_dim: int
    alignment_mode: str


@dataclass
class FittedBinaryModel:
    estimator: Pipeline
    candidate_key: str
    candidate_label: str
    candidate_complexity: float
    candidate_params: dict[str, Any]
    chosen_c: float | None
    candidate_max_depth: int | None
    decision_threshold: float
    validation_metrics: dict[str, Any]
    xgb_best_iteration: int | None = None
    xgb_best_score: float | None = None


@dataclass(frozen=True)
class AttentionSubsetSpec:
    key: str
    title: str
    metric_names: tuple[str, ...]
    transition_mode: str


ATTENTION_SUBSETS = OrderedDict(
    [
        (
            "all_attention",
            AttentionSubsetSpec(
                key="all_attention",
                title="All attention features",
                metric_names=ATTENTION_METRIC_NAMES,
                transition_mode="all",
            ),
        ),
        (
            "grounding_only",
            AttentionSubsetSpec(
                key="grounding_only",
                title="Grounding only",
                metric_names=GROUNDING_ATTENTION_METRIC_NAMES,
                transition_mode="base_only",
            ),
        ),
        (
            "concentration_only",
            AttentionSubsetSpec(
                key="concentration_only",
                title="Concentration only",
                metric_names=CONCENTRATION_ATTENTION_METRIC_NAMES,
                transition_mode="base_only",
            ),
        ),
        (
            "grounding_transition_only",
            AttentionSubsetSpec(
                key="grounding_transition_only",
                title="Grounding transitions only",
                metric_names=GROUNDING_ATTENTION_METRIC_NAMES,
                transition_mode="transition_only",
            ),
        ),
        (
            "concentration_transition_only",
            AttentionSubsetSpec(
                key="concentration_transition_only",
                title="Concentration transitions only",
                metric_names=CONCENTRATION_ATTENTION_METRIC_NAMES,
                transition_mode="transition_only",
            ),
        ),
    ]
)


FEATURE_SPACES = OrderedDict(
    [
        (
            "baseline_tfidf_last_sentence_text",
            FeatureSpaceSpec(
                name="baseline_tfidf_last_sentence_text",
                title="Baseline: TF-IDF last sentence",
                family_title="tfidf_baseline",
                uses_attention=False,
                attention_subset_key=None,
                activation_variant=None,
                activation_use_pca=False,
                baseline_variant="tfidf",
                baseline_text_field="last_sentence_text",
            ),
        ),
        (
            "baseline_tfidf_prefix_text",
            FeatureSpaceSpec(
                name="baseline_tfidf_prefix_text",
                title="Baseline: TF-IDF prefix",
                family_title="tfidf_baseline",
                uses_attention=False,
                attention_subset_key=None,
                activation_variant=None,
                activation_use_pca=False,
                baseline_variant="tfidf",
                baseline_text_field="prefix_text",
            ),
        ),
        (
            "attention_only",
            FeatureSpaceSpec(
                name="attention_only",
                title="Attention only",
                family_title="attention_only",
                uses_attention=True,
                attention_subset_key="all_attention",
                activation_variant=None,
                activation_use_pca=False,
            ),
        ),
        (
            "attention_grounding_only",
            FeatureSpaceSpec(
                name="attention_grounding_only",
                title="Attention only: grounding",
                family_title="attention_only",
                uses_attention=True,
                attention_subset_key="grounding_only",
                activation_variant=None,
                activation_use_pca=False,
            ),
        ),
        (
            "attention_concentration_only",
            FeatureSpaceSpec(
                name="attention_concentration_only",
                title="Attention only: concentration",
                family_title="attention_only",
                uses_attention=True,
                attention_subset_key="concentration_only",
                activation_variant=None,
                activation_use_pca=False,
            ),
        ),
        (
            "attention_grounding_transition_only",
            FeatureSpaceSpec(
                name="attention_grounding_transition_only",
                title="Attention only: grounding transition",
                family_title="attention_only",
                uses_attention=True,
                attention_subset_key="grounding_transition_only",
                activation_variant=None,
                activation_use_pca=False,
            ),
        ),
        (
            "attention_concentration_transition_only",
            FeatureSpaceSpec(
                name="attention_concentration_transition_only",
                title="Attention only: concentration transition",
                family_title="attention_only",
                uses_attention=True,
                attention_subset_key="concentration_transition_only",
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
                attention_subset_key=None,
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
                attention_subset_key=None,
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
                attention_subset_key=None,
                activation_variant="delta_prev4mean",
                activation_use_pca=True,
            ),
        ),
        (
            "attention_plus_activation_pca_final",
            FeatureSpaceSpec(
                name="attention_plus_activation_pca_final",
                title="Attention + PCA final",
                family_title="attention_plus_activation",
                uses_attention=True,
                attention_subset_key="all_attention",
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
                attention_subset_key="all_attention",
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
                attention_subset_key="all_attention",
                activation_variant="delta_prev4mean",
                activation_use_pca=True,
            ),
        ),
        (
            "baseline_raw_final",
            FeatureSpaceSpec(
                name="baseline_raw_final",
                title="Baseline: raw final activation",
                family_title="baseline",
                uses_attention=False,
                attention_subset_key=None,
                activation_variant="final_sentence",
                activation_use_pca=False,
            ),
        ),
    ]
)

FAMILY_PANEL_ORDER = (
    "tfidf_baseline",
    "attention_only",
    "activation_only",
    "attention_plus_activation",
    "baseline",
)
TARGET_SPECS = OrderedDict(
    [
        (
            "delta_pos_gt_0_3",
            {
                "title": "delta_deception_rate > 0.3",
                "label_fn": lambda delta, threshold: delta > threshold,
                "negative_label": "<= 0.3",
                "positive_label": "> 0.3",
            },
        ),
        (
            "delta_neg_lt_neg_0_3",
            {
                "title": "delta_deception_rate < -0.3",
                "label_fn": lambda delta, threshold: delta < -threshold,
                "negative_label": ">= -0.3",
                "positive_label": "< -0.3",
            },
        ),
    ]
)


def maybe_tqdm(iterable: Iterable[Any], *, desc: str, total: int | None = None, disable: bool = False):
    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None
    if disable or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total)


def parse_csv_list(raw: str | Sequence[str]) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [str(value).strip() for value in raw if str(value).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_int_csv(raw: str | Sequence[int]) -> list[int]:
    if isinstance(raw, (list, tuple)):
        return [int(value) for value in raw]
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def parse_float_csv(raw: str | Sequence[float]) -> list[float]:
    if isinstance(raw, (list, tuple)):
        return [float(value) for value in raw]
    return [float(part.strip()) for part in str(raw).split(",") if part.strip()]


def normalize_model_key(raw_value: str) -> str:
    clean = slugify(str(raw_value))
    if clean in MODEL_SPECS:
        return clean
    if clean in MODEL_KEY_BY_ALIAS:
        return MODEL_KEY_BY_ALIAS[clean]
    raise ValueError(
        f"Unsupported model preset {raw_value!r}. "
        f"Expected one of: {', '.join(MODEL_SPECS)}"
    )


def normalize_model_family(raw_value: str) -> str:
    value = slugify(str(raw_value or "xgb"))
    if value in {"xgb", "xgboost"}:
        return "xgboost"
    if value in {"logreg", "logistic", "logistic_regression", "lr"}:
        return "logreg"
    raise ValueError(f"Unsupported model family: {raw_value!r}")


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_parquet_file(path: Path) -> None:
    pq.ParquetFile(path)


def resolve_feature_path_optional(dataset_dir: Path) -> Path | None:
    candidate_paths = [(dataset_dir / filename).resolve() for filename in FEATURE_FILENAME_CANDIDATES]
    existing_paths = [path for path in candidate_paths if path.exists()]
    for path in existing_paths:
        try:
            validate_parquet_file(path)
            return path
        except Exception:
            continue
    return None


def resolve_feature_path(dataset_dir: Path, *, bundle_key: str) -> Path:
    path = resolve_feature_path_optional(dataset_dir)
    if path is None:
        checked = ", ".join(str((dataset_dir / filename).resolve()) for filename in FEATURE_FILENAME_CANDIDATES)
        raise FileNotFoundError(f"Missing readable feature parquet for {bundle_key}: checked {checked}")
    return path


def load_parquet_with_optional_columns(
    path: Path,
    *,
    required_columns: list[str],
    optional_columns: list[str],
) -> pd.DataFrame:
    requested_columns = list(required_columns) + list(optional_columns)
    try:
        return pd.read_parquet(path, columns=requested_columns).copy()
    except Exception:
        df = pd.read_parquet(path, columns=required_columns).copy()
        for column_name in optional_columns:
            try:
                optional_df = pd.read_parquet(path, columns=[column_name]).copy()
            except Exception:
                continue
            if column_name in optional_df.columns:
                df[column_name] = optional_df[column_name]
        return df


def count_alpha_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+", str(text)))


def annotate_prefix_metadata(
    df: pd.DataFrame,
    *,
    env_name: str,
    model_display: str,
    delta_threshold: float,
    min_num_valid: int,
    min_sentence_alpha_words: int,
    exclude_multiline_sentences: bool,
) -> pd.DataFrame:
    out = df.copy()
    out["env_name"] = env_name
    out["model_display"] = model_display
    out["example_id"] = out["example_id"].astype(str)
    out["sentence_idx"] = pd.to_numeric(out["sentence_idx"], errors="coerce")
    out["deception_rate"] = pd.to_numeric(out["deception_rate"], errors="coerce")
    has_num_valid = "num_valid" in out.columns
    has_sentence_text = "sentence_text" in out.columns
    if has_num_valid:
        out["num_valid"] = pd.to_numeric(out["num_valid"], errors="coerce")
    else:
        out["num_valid"] = np.nan
    if has_sentence_text:
        out["sentence_text"] = out["sentence_text"].fillna("").astype(str)
    else:
        out["sentence_text"] = ""
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["example_id", "sentence_idx", "deception_rate"]).copy()
    out["sentence_idx"] = out["sentence_idx"].astype(int)
    out["deception_rate"] = out["deception_rate"].astype(np.float32)
    out = out.sort_values(["example_id", "sentence_idx", "row_idx"], kind="mergesort").reset_index(drop=True)
    out["prev_deception_rate"] = out.groupby("example_id", sort=False)["deception_rate"].shift(1).astype(np.float32)
    out["prev_num_valid"] = out.groupby("example_id", sort=False)["num_valid"].shift(1)
    out["delta_deception_rate"] = (out["deception_rate"] - out["prev_deception_rate"]).astype(np.float32)

    out["sentence_alpha_word_count"] = out["sentence_text"].map(count_alpha_words).astype(np.int32)
    usable_sentence_mask = out["sentence_text"].astype(str).str.strip().ne("").to_numpy(dtype=bool, copy=False)
    if exclude_multiline_sentences:
        usable_sentence_mask &= ~out["sentence_text"].astype(str).str.contains("\n", regex=False).to_numpy(
            dtype=bool,
            copy=False,
        )
    if int(min_sentence_alpha_words) > 0:
        usable_sentence_mask &= out["sentence_alpha_word_count"].ge(int(min_sentence_alpha_words)).to_numpy(
            dtype=bool,
            copy=False,
        )

    if has_num_valid and int(min_num_valid) > 0:
        enough_num_valid_mask = (
            out["num_valid"].ge(int(min_num_valid)) & out["prev_num_valid"].ge(int(min_num_valid))
        ).fillna(False).to_numpy(dtype=bool, copy=False)
    else:
        enough_num_valid_mask = np.ones(len(out), dtype=bool)
    out["passes_commitment_pair_filters"] = usable_sentence_mask & enough_num_valid_mask

    valid_delta = out["delta_deception_rate"].notna()
    delta_array = out["delta_deception_rate"].to_numpy(dtype=np.float32, copy=False)
    for target_name, target_spec in TARGET_SPECS.items():
        labels = np.asarray(target_spec["label_fn"](delta_array, float(delta_threshold)), dtype=np.int8)
        out[f"label__{target_name}"] = np.where(valid_delta, labels, np.nan)
    return out


def build_example_split_map(example_ids: pd.Series, *, seed: int, val_size: float) -> dict[str, str]:
    unique_examples = pd.Series(example_ids.astype(str).unique(), dtype="string")
    if unique_examples.shape[0] < 2:
        raise ValueError("Need at least 2 unique example IDs to build train/val splits.")
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(unique_examples.shape[0])
    val_count = int(round(float(val_size) * unique_examples.shape[0]))
    val_count = max(1, min(unique_examples.shape[0] - 1, val_count))
    val_examples = set(unique_examples.iloc[order[:val_count]].astype(str).tolist())
    return {
        str(example_id): ("val" if str(example_id) in val_examples else "train")
        for example_id in unique_examples.astype(str).tolist()
    }


def load_feature_metadata(
    feature_path: Path,
    *,
    env_name: str,
    model_display: str,
    delta_threshold: float,
    min_num_valid: int,
    min_sentence_alpha_words: int,
    exclude_multiline_sentences: bool,
) -> pd.DataFrame:
    df = load_parquet_with_optional_columns(
        feature_path,
        required_columns=["example_id", "sentence_idx", "deception_rate"],
        optional_columns=["sentence_text", "num_valid"],
    )
    df["row_idx"] = np.arange(len(df), dtype=np.int64)
    return annotate_prefix_metadata(
        df,
        env_name=env_name,
        model_display=model_display,
        delta_threshold=delta_threshold,
        min_num_valid=min_num_valid,
        min_sentence_alpha_words=min_sentence_alpha_words,
        exclude_multiline_sentences=exclude_multiline_sentences,
    )


def load_activation_metadata(
    activation_path: Path,
    *,
    env_name: str,
    model_display: str,
    delta_threshold: float,
    min_num_valid: int,
    min_sentence_alpha_words: int,
    exclude_multiline_sentences: bool,
) -> tuple[pd.DataFrame, int]:
    with h5py.File(activation_path, "r") as f:
        example_ids = pd.Series(f["example_id"].asstr()[:], dtype="string")
        sentence_idx = pd.Series(np.asarray(f["sentence_idx"][:], dtype=np.int64))
        deception_rate = pd.Series(np.asarray(f["deception_rate"][:], dtype=np.float32))
        hidden_dim = int(f["activations"].shape[2])
    df = pd.DataFrame(
        {
            "example_id": example_ids,
            "sentence_idx": sentence_idx,
            "deception_rate": deception_rate,
            "row_idx": np.arange(len(example_ids), dtype=np.int64),
        }
    )
    annotated = annotate_prefix_metadata(
        df,
        env_name=env_name,
        model_display=model_display,
        delta_threshold=delta_threshold,
        min_num_valid=min_num_valid,
        min_sentence_alpha_words=min_sentence_alpha_words,
        exclude_multiline_sentences=exclude_multiline_sentences,
    )
    return annotated, hidden_dim


def load_structural_metadata(structural_path: Path, *, env_name: str, model_display: str) -> pd.DataFrame:
    df = pd.read_parquet(structural_path, columns=["example_id", "sentence_idx"]).copy()
    df["env_name"] = env_name
    df["model_display"] = model_display
    df["example_id"] = df["example_id"].astype(str)
    df["sentence_idx"] = pd.to_numeric(df["sentence_idx"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["example_id", "sentence_idx"]).copy()
    df["sentence_idx"] = df["sentence_idx"].astype(int)
    df["row_idx"] = np.arange(len(df), dtype=np.int64)
    return df.reset_index(drop=True)


def split_summary_row(df: pd.DataFrame, *, bundle_key: str) -> dict[str, Any]:
    train_df = df.loc[df["split"].eq("train")].copy()
    val_df = df.loc[df["split"].eq("val")].copy()
    train_modeled_df = train_df.loc[
        train_df["delta_deception_rate"].notna() & train_df["passes_commitment_pair_filters"].astype(bool)
    ].copy()
    val_modeled_df = val_df.loc[
        val_df["delta_deception_rate"].notna() & val_df["passes_commitment_pair_filters"].astype(bool)
    ].copy()
    row: dict[str, Any] = {
        "bundle_key": bundle_key,
        "rows": int(len(df)),
        "examples": int(df["example_id"].nunique()),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_modeled_rows": int(len(train_modeled_df)),
        "val_modeled_rows": int(len(val_modeled_df)),
    }
    for target_name in TARGET_SPECS:
        row[f"train_pos_rate__{target_name}"] = float(
            pd.to_numeric(train_modeled_df[f"label__{target_name}"], errors="coerce").mean()
        )
        row[f"val_pos_rate__{target_name}"] = float(
            pd.to_numeric(val_modeled_df[f"label__{target_name}"], errors="coerce").mean()
        )
    return row


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


def build_attention_reduction_lookup(attention_layer_roots: OrderedDict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root_name, columns in attention_layer_roots.items():
        match = ATTN_ROOT_RE.fullmatch(root_name)
        if match is None:
            continue
        layer_count = len(columns)
        metric_name = str(match.group("metric"))
        transition_prefix = str(match.group("transition_prefix") or "")
        is_transition = transition_prefix != ""
        metric_group = ATTENTION_METRIC_GROUP_BY_NAME[metric_name]
        attention_feature_group = f"{metric_group}_transition" if is_transition else metric_group
        for band_name in BAND_NAMES:
            for stat_name in BAND_STAT_NAMES:
                rows.append(
                    {
                        "feature": f"{root_name}__band_{band_name}__{stat_name}",
                        "feature_root": root_name,
                        "transition_prefix": transition_prefix,
                        "is_transition": bool(is_transition),
                        "metric_name": metric_name,
                        "metric_group": metric_group,
                        "attention_feature_group": attention_feature_group,
                        "head_summary": match.group("head_summary"),
                        "band": band_name,
                        "band_stat": stat_name,
                        "layer_count": int(layer_count),
                        "family": attention_feature_group,
                    }
                )
    return pd.DataFrame(rows)


def build_attention_reduced_bundle_frame(
    *,
    bundle_key: str,
    feature_path: Path,
    metadata_df: pd.DataFrame,
    attention_layer_roots: OrderedDict[str, list[str]],
    root_batch_size: int,
    disable_tqdm: bool,
) -> pd.DataFrame:
    row_ids = metadata_df["row_idx"].to_numpy(dtype=np.int64, copy=False)
    arrays: dict[str, np.ndarray] = {}
    ordered_roots = list(attention_layer_roots.items())
    iterator = maybe_tqdm(
        range(0, len(ordered_roots), int(root_batch_size)),
        desc=f"Reduce attention:{bundle_key}",
        total=int(math.ceil(len(ordered_roots) / int(root_batch_size))),
        disable=disable_tqdm,
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


def feature_columns_for_attention_frame(df: pd.DataFrame) -> list[str]:
    reserved = {
        "env_name",
        "model_display",
        "example_id",
        "sentence_idx",
        "sentence_text",
        "sentence_alpha_word_count",
        "deception_rate",
        "num_valid",
        "prev_num_valid",
        "row_idx",
        "prev_deception_rate",
        "delta_deception_rate",
        "passes_commitment_pair_filters",
        "split",
        *(f"label__{target_name}" for target_name in TARGET_SPECS),
    }
    return [column for column in df.columns if column not in reserved]


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
        effect_rows.append(
            pd.Series(
                compute_standardized_effects(x_array, y_train),
                index=feature_names,
                name=env_name,
                dtype=np.float32,
            )
        )
    effects_df = pd.DataFrame(effect_rows).T.reset_index().rename(columns={"index": "feature"})
    effects_df = feature_lookup_df.merge(effects_df, on="feature", how="left", validate="one_to_one")
    effect_cols = []
    for env_name in env_frames:
        effect_col = f"{slugify(env_name)}_effect"
        effects_df = effects_df.rename(columns={env_name: effect_col})
        effect_cols.append(effect_col)
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
        effects_df["min_abs_effect"] + (0.35 * effects_df["mean_abs_effect"]) - (0.25 * effects_df["std_effect"])
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


def select_attention_pool(
    ranking_df: pd.DataFrame,
    *,
    fallback_top_k: int,
) -> tuple[list[str], str]:
    filtered_df = ranking_df.loc[ranking_df["same_sign_all"].eq(True)].copy().reset_index(drop=True)
    if not filtered_df.empty:
        return filtered_df["feature"].astype(str).tolist(), "same_sign_all"
    top_df = ranking_df.head(int(max(1, fallback_top_k))).copy()
    if top_df.empty:
        raise ValueError("Attention ranking produced no usable features.")
    return top_df["feature"].astype(str).tolist(), "top_consistency_fallback"


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
                "transition_prefix": "",
                "is_transition": False,
                "metric_name": metric_name,
                "metric_group": "",
                "attention_feature_group": "",
                "head_summary": "",
                "band": "",
                "band_stat": "",
                "layer_count": pd.NA,
                "family": "activation",
            }
            for feature_name in feature_names
        ]
    )


def make_generic_feature_lookup(
    *,
    space_name: str,
    feature_names: list[str],
    feature_root: str,
    metric_name: str,
    family_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": str(feature_name),
                "feature_root": feature_root,
                "transition_prefix": "",
                "is_transition": False,
                "metric_name": metric_name,
                "metric_group": family_name,
                "attention_feature_group": family_name,
                "head_summary": "",
                "band": "",
                "band_stat": "",
                "layer_count": pd.NA,
                "family": family_name,
            }
            for feature_name in feature_names
        ]
    )


def choose_shared_feature_names(feature_names_by_bundle: OrderedDict[str, list[str]]) -> tuple[list[str], str]:
    if not feature_names_by_bundle:
        return [], "empty"
    ordered_keys = list(feature_names_by_bundle.keys())
    first_key = ordered_keys[0]
    common_names = set(feature_names_by_bundle[first_key])
    for bundle_key in ordered_keys[1:]:
        common_names &= set(feature_names_by_bundle[bundle_key])
    if common_names:
        ordered_common = [
            str(feature_name)
            for feature_name in feature_names_by_bundle[first_key]
            if str(feature_name) in common_names
        ]
        return ordered_common, "intersection"
    seen: set[str] = set()
    ordered_union: list[str] = []
    for bundle_key in ordered_keys:
        for feature_name in feature_names_by_bundle[bundle_key]:
            feature_name = str(feature_name)
            if feature_name in seen:
                continue
            seen.add(feature_name)
            ordered_union.append(feature_name)
    return ordered_union, "union_fallback"


def parse_tfidf_artifact_meta(meta_path: Path) -> dict[str, Any]:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def locate_tfidf_artifact(cache_dir: Path, text_field: str) -> dict[str, Path] | None:
    if not cache_dir.is_dir():
        return None
    candidate_rows: list[dict[str, Any]] = []
    for meta_path in sorted(cache_dir.glob("*.json")):
        try:
            meta = parse_tfidf_artifact_meta(meta_path)
        except Exception:
            continue
        if str(meta.get("feature_type", "")).strip().lower() != "tfidf":
            continue
        if str(meta.get("text_field", "")).strip() != str(text_field):
            continue
        matrix_path = meta_path.with_suffix(".npz")
        feature_names_path = meta_path.with_name(f"{meta_path.stem}__feature_names.npy")
        if not matrix_path.exists() or not feature_names_path.exists():
            continue
        vectorizer_params = meta.get("vectorizer_params", {})
        candidate_rows.append(
            {
                "meta_path": meta_path,
                "matrix_path": matrix_path,
                "feature_names_path": feature_names_path,
                "is_default_config": (
                    int(vectorizer_params.get("max_features", -1))
                    == int(getattr(baseline_extractor, "DEFAULT_TFIDF_MAX_FEATURES", 20000))
                    and tuple(vectorizer_params.get("ngram_range", []))
                    == (
                        int(getattr(baseline_extractor, "DEFAULT_TFIDF_MIN_NGRAM", 1)),
                        int(getattr(baseline_extractor, "DEFAULT_TFIDF_MAX_NGRAM", 2)),
                    )
                    and bool(vectorizer_params.get("lowercase", True))
                    and bool(vectorizer_params.get("sublinear_tf", True))
                    and vectorizer_params.get("stop_words", None) is None
                ),
                "mtime_ns": meta_path.stat().st_mtime_ns,
            }
        )
    if not candidate_rows:
        return None
    candidate_rows.sort(key=lambda row: (bool(row["is_default_config"]), int(row["mtime_ns"])), reverse=True)
    selected = candidate_rows[0]
    return {
        "meta_path": selected["meta_path"],
        "matrix_path": selected["matrix_path"],
        "feature_names_path": selected["feature_names_path"],
    }


def align_matrix_to_feature_order(
    matrix: sp.spmatrix,
    env_feature_names: list[str],
    target_feature_names: list[str],
) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    env_feature_to_idx = {str(feature_name): idx for idx, feature_name in enumerate(env_feature_names)}
    source_columns: list[int] = []
    target_columns: list[int] = []
    for target_idx, feature_name in enumerate(target_feature_names):
        source_idx = env_feature_to_idx.get(str(feature_name))
        if source_idx is None:
            continue
        source_columns.append(int(source_idx))
        target_columns.append(int(target_idx))
    if not target_feature_names:
        return sp.csr_matrix((matrix.shape[0], 0), dtype=np.float32)
    if not source_columns:
        return sp.csr_matrix((matrix.shape[0], len(target_feature_names)), dtype=np.float32)
    subset = matrix[:, np.asarray(source_columns, dtype=np.int64)]
    subset_coo = subset.tocoo()
    remapped_columns = np.asarray(target_columns, dtype=np.int64)[subset_coo.col]
    return sp.csr_matrix(
        (subset_coo.data.astype(np.float32, copy=False), (subset_coo.row, remapped_columns)),
        shape=(matrix.shape[0], len(target_feature_names)),
        dtype=np.float32,
    )


def gather_sparse_rows_with_missing(matrix: sp.spmatrix, row_idx: np.ndarray) -> sp.csr_matrix:
    row_idx = np.asarray(row_idx, dtype=np.int64)
    matrix = matrix.tocsr().astype(np.float32)
    if row_idx.size == 0:
        return sp.csr_matrix((0, matrix.shape[1]), dtype=np.float32)
    valid_positions = np.flatnonzero(row_idx >= 0)
    if valid_positions.size == row_idx.size:
        return matrix[row_idx].tocsr()
    if valid_positions.size == 0:
        return sp.csr_matrix((row_idx.size, matrix.shape[1]), dtype=np.float32)
    valid_rows = matrix[row_idx[valid_positions]].tocoo()
    remapped_row = valid_positions[valid_rows.row]
    return sp.csr_matrix(
        (valid_rows.data.astype(np.float32, copy=False), (remapped_row, valid_rows.col)),
        shape=(row_idx.size, matrix.shape[1]),
        dtype=np.float32,
    )


def build_tfidf_matrix_bundle(
    *,
    text_field: str,
    space_name: str,
    bundle_specs: dict[str, BundleSpec],
    structural_metadata_by_bundle: dict[str, pd.DataFrame],
    split_cache_by_bundle: dict[str, dict[str, Any]],
) -> BaselineMatrixBundle:
    artifact_paths_by_bundle: OrderedDict[str, dict[str, Path]] = OrderedDict()
    feature_names_by_bundle: OrderedDict[str, list[str]] = OrderedDict()
    for bundle_key, bundle_spec in bundle_specs.items():
        artifact_paths = locate_tfidf_artifact(bundle_spec.tfidf_cache_dir, text_field)
        if artifact_paths is None:
            raise FileNotFoundError(
                f"Could not find TF-IDF cache artifact for {bundle_key} / {text_field} under {bundle_spec.tfidf_cache_dir}"
            )
        meta = parse_tfidf_artifact_meta(artifact_paths["meta_path"])
        structural_metadata_df = structural_metadata_by_bundle.get(bundle_key)
        if structural_metadata_df is None:
            raise FileNotFoundError(f"{bundle_key} is missing companion structural metadata for TF-IDF alignment.")
        structural_key_df = pd.read_parquet(
            bundle_spec.structural_baseline_path,
            columns=["example_id", "sentence_idx"],
        ).copy()
        expected_fingerprint = baseline_extractor.dataset_row_fingerprint(structural_key_df)
        if str(meta.get("fingerprint", "")) != str(expected_fingerprint):
            raise ValueError(
                f"{bundle_key} TF-IDF fingerprint mismatch for {text_field}. "
                f"expected={expected_fingerprint}, found={meta.get('fingerprint', '')}"
            )
        if int(meta.get("num_rows", len(structural_metadata_df))) != int(len(structural_metadata_df)):
            raise ValueError(
                f"{bundle_key} TF-IDF row-count mismatch for {text_field}. "
                f"expected={len(structural_metadata_df)}, found={meta.get('num_rows')}"
            )
        feature_names = np.load(artifact_paths["feature_names_path"], allow_pickle=False).astype(str).tolist()
        artifact_paths_by_bundle[bundle_key] = artifact_paths
        feature_names_by_bundle[bundle_key] = feature_names
    shared_feature_names, shared_mode = choose_shared_feature_names(feature_names_by_bundle)
    if not shared_feature_names:
        raise ValueError(f"No shared TF-IDF features available for text field {text_field!r}.")
    feature_lookup_df = make_generic_feature_lookup(
        space_name=space_name,
        feature_names=shared_feature_names,
        feature_root=space_name,
        metric_name=f"tfidf__{text_field}__{shared_mode}",
        family_name="tfidf",
    )
    matrices_by_bundle: dict[str, dict[str, sp.csr_matrix]] = {}
    for bundle_key in bundle_specs:
        split_bundle = split_cache_by_bundle[bundle_key]
        artifact_paths = artifact_paths_by_bundle[bundle_key]
        bundle_feature_names = feature_names_by_bundle[bundle_key]
        matrix = sp.load_npz(artifact_paths["matrix_path"]).tocsr().astype(np.float32)
        aligned_matrix = align_matrix_to_feature_order(matrix, bundle_feature_names, shared_feature_names)
        matrices_by_bundle[bundle_key] = {
            "train": gather_sparse_rows_with_missing(aligned_matrix, split_bundle["train_structural_row_idx"]),
            "val": gather_sparse_rows_with_missing(aligned_matrix, split_bundle["val_structural_row_idx"]),
        }
    return BaselineMatrixBundle(
        matrices_by_bundle=matrices_by_bundle,
        feature_names=shared_feature_names,
        feature_lookup_df=feature_lookup_df,
        alignment_mode=shared_mode,
    )


def build_model_candidate_specs(
    y_train: np.ndarray,
    *,
    model_family: str,
    seed: int,
    logreg_c: float,
    xgb_max_depth: int,
    xgb_n_estimators: int,
    xgb_learning_rate: float,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
    xgb_reg_lambda: float,
    xgb_min_child_weight: float,
    xgb_gamma: float,
    xgb_n_jobs: int,
    xgb_eval_metric: str,
) -> list[dict[str, Any]]:
    y_train = np.asarray(y_train, dtype=np.int8)
    if model_family == "logreg":
        return [
            {
                "candidate_key": f"c={float(logreg_c):g}",
                "candidate_label": f"C={float(logreg_c):g}",
                "candidate_complexity": float(logreg_c),
                "chosen_c": float(logreg_c),
                "candidate_max_depth": np.nan,
                "candidate_params": {"C": float(logreg_c)},
            }
        ]
    if model_family == "xgboost":
        pos_count = int((y_train == 1).sum())
        neg_count = int((y_train == 0).sum())
        scale_pos_weight = float(neg_count / max(pos_count, 1))
        return [
            {
                "candidate_key": f"max_depth={int(xgb_max_depth)}",
                "candidate_label": f"max_depth={int(xgb_max_depth)}",
                "candidate_complexity": float(xgb_max_depth),
                "chosen_c": np.nan,
                "candidate_max_depth": int(xgb_max_depth),
                "candidate_params": {
                    "objective": "binary:logistic",
                    "n_estimators": int(xgb_n_estimators),
                    "max_depth": int(xgb_max_depth),
                    "learning_rate": float(xgb_learning_rate),
                    "subsample": float(xgb_subsample),
                    "colsample_bytree": float(xgb_colsample_bytree),
                    "reg_lambda": float(xgb_reg_lambda),
                    "min_child_weight": float(xgb_min_child_weight),
                    "gamma": float(xgb_gamma),
                    "scale_pos_weight": scale_pos_weight,
                    "tree_method": "hist",
                    "eval_metric": str(xgb_eval_metric),
                    "random_state": int(seed),
                    "n_jobs": int(xgb_n_jobs),
                    "verbosity": 0,
                },
            }
        ]
    raise ValueError(f"Unsupported model family: {model_family!r}")


def build_estimator(*, candidate_params: dict[str, Any], input_is_sparse: bool, model_family: str, seed: int) -> Pipeline:
    if model_family == "logreg":
        model = LogisticRegression(
            C=float(candidate_params["C"]),
            class_weight="balanced",
            max_iter=4000,
            solver="liblinear",
            random_state=int(seed),
        )
        if input_is_sparse:
            return Pipeline([("model", model)])
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])
    if model_family == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is unavailable, but model_family='xgboost' was requested.")
        model = XGBClassifier(**candidate_params)
        if input_is_sparse:
            return Pipeline([("model", model)])
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    raise ValueError(f"Unsupported model family: {model_family!r}")


def serialize_candidate_params(candidate_params: dict[str, Any]) -> str:
    return json.dumps(candidate_params, sort_keys=True)


def extract_xgb_training_diagnostics(estimator: Pipeline) -> tuple[int | None, float | None]:
    model = estimator.named_steps["model"]
    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    try:
        best_iteration = None if best_iteration is None else int(best_iteration)
    except Exception:  # noqa: BLE001
        best_iteration = None
    try:
        best_score = None if best_score is None else float(best_score)
    except Exception:  # noqa: BLE001
        best_score = None
    return best_iteration, best_score


def fit_xgboost_estimator(
    *,
    x_train: Any,
    y_train: np.ndarray,
    x_val: Any,
    y_val: np.ndarray,
    candidate_params: dict[str, Any],
    input_is_sparse: bool,
    xgb_early_stopping_rounds: int,
) -> Pipeline:
    use_early_stopping = (
        int(xgb_early_stopping_rounds) > 0
        and len(y_val) > 0
        and np.unique(np.asarray(y_val, dtype=np.int8)).size >= 2
    )
    if input_is_sparse:
        model = XGBClassifier(**candidate_params)
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if use_early_stopping:
            fit_kwargs["eval_set"] = [(x_val, y_val)]
            try:
                model.fit(x_train, y_train, early_stopping_rounds=int(xgb_early_stopping_rounds), **fit_kwargs)
            except TypeError:
                model = XGBClassifier(**candidate_params, early_stopping_rounds=int(xgb_early_stopping_rounds))
                model.fit(x_train, y_train, **fit_kwargs)
        else:
            model.fit(x_train, y_train, **fit_kwargs)
        return Pipeline([("model", model)])

    imputer = SimpleImputer(strategy="median")
    x_train_fit = imputer.fit_transform(x_train)
    x_val_fit = imputer.transform(x_val)
    model = XGBClassifier(**candidate_params)
    fit_kwargs = {"verbose": False}
    if use_early_stopping:
        fit_kwargs["eval_set"] = [(x_val_fit, y_val)]
        try:
            model.fit(x_train_fit, y_train, early_stopping_rounds=int(xgb_early_stopping_rounds), **fit_kwargs)
        except TypeError:
            model = XGBClassifier(**candidate_params, early_stopping_rounds=int(xgb_early_stopping_rounds))
            model.fit(x_train_fit, y_train, **fit_kwargs)
    else:
        model.fit(x_train_fit, y_train, **fit_kwargs)
    return Pipeline([("imputer", imputer), ("model", model)])


def fit_candidate_classifiers(
    x_train: Any,
    y_train: np.ndarray,
    x_val: Any,
    y_val: np.ndarray,
    *,
    model_family: str,
    seed: int,
    logreg_c: float,
    xgb_max_depth: int,
    xgb_n_estimators: int,
    xgb_learning_rate: float,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
    xgb_reg_lambda: float,
    xgb_min_child_weight: float,
    xgb_gamma: float,
    xgb_n_jobs: int,
    xgb_eval_metric: str,
    xgb_early_stopping_rounds: int,
    decision_threshold_mode: str,
) -> tuple[list[FittedBinaryModel], pd.DataFrame]:
    candidate_rows: list[dict[str, Any]] = []
    fitted_models: list[FittedBinaryModel] = []
    input_is_sparse = bool(sp.issparse(x_train) or sp.issparse(x_val))
    for candidate_spec in build_model_candidate_specs(
        y_train,
        model_family=model_family,
        seed=seed,
        logreg_c=logreg_c,
        xgb_max_depth=xgb_max_depth,
        xgb_n_estimators=xgb_n_estimators,
        xgb_learning_rate=xgb_learning_rate,
        xgb_subsample=xgb_subsample,
        xgb_colsample_bytree=xgb_colsample_bytree,
        xgb_reg_lambda=xgb_reg_lambda,
        xgb_min_child_weight=xgb_min_child_weight,
        xgb_gamma=xgb_gamma,
        xgb_n_jobs=xgb_n_jobs,
        xgb_eval_metric=xgb_eval_metric,
    ):
        if model_family == "xgboost":
            estimator = fit_xgboost_estimator(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                candidate_params=dict(candidate_spec["candidate_params"]),
                input_is_sparse=input_is_sparse,
                xgb_early_stopping_rounds=int(xgb_early_stopping_rounds),
            )
        else:
            estimator = build_estimator(
                candidate_params=dict(candidate_spec["candidate_params"]),
                input_is_sparse=input_is_sparse,
                model_family=model_family,
                seed=seed,
            )
            estimator.fit(x_train, y_train)
        val_scores = estimator.predict_proba(x_val)[:, 1].astype(np.float32)
        xgb_best_iteration, xgb_best_score = extract_xgb_training_diagnostics(estimator)
        decision_threshold = choose_decision_threshold(
            y_val,
            val_scores,
            default_threshold=0.5,
            mode=decision_threshold_mode,
        )
        metrics = summarize_score_metrics(y_val, val_scores, decision_threshold=decision_threshold)
        metrics_extra = safe_binary_metrics(y_val, val_scores)
        metrics["pr_auc"] = float(metrics_extra["pr_auc"])
        metrics["brier"] = float(metrics_extra["brier"])
        metrics["positive_count"] = float(metrics_extra["positive_count"])
        metrics["negative_count"] = float(metrics_extra["negative_count"])
        metrics["base_rate"] = float(metrics_extra["base_rate"])
        candidate_rows.append(
            {
                "model_family": model_family,
                "candidate_key": str(candidate_spec["candidate_key"]),
                "candidate_label": str(candidate_spec["candidate_label"]),
                "candidate_complexity": float(candidate_spec["candidate_complexity"]),
                "candidate_params_json": serialize_candidate_params(candidate_spec["candidate_params"]),
                "candidate_c": float(candidate_spec["chosen_c"]) if pd.notna(candidate_spec["chosen_c"]) else np.nan,
                "candidate_max_depth": int(candidate_spec["candidate_max_depth"]) if pd.notna(candidate_spec["candidate_max_depth"]) else np.nan,
                "decision_threshold": float(decision_threshold),
                "xgb_best_iteration": pd.NA if xgb_best_iteration is None else int(xgb_best_iteration),
                "xgb_best_score": pd.NA if xgb_best_score is None else float(xgb_best_score),
                **metrics,
            }
        )
        fitted_models.append(
            FittedBinaryModel(
                estimator=estimator,
                candidate_key=str(candidate_spec["candidate_key"]),
                candidate_label=str(candidate_spec["candidate_label"]),
                candidate_complexity=float(candidate_spec["candidate_complexity"]),
                candidate_params=dict(candidate_spec["candidate_params"]),
                chosen_c=float(candidate_spec["chosen_c"]) if pd.notna(candidate_spec["chosen_c"]) else None,
                candidate_max_depth=int(candidate_spec["candidate_max_depth"]) if pd.notna(candidate_spec["candidate_max_depth"]) else None,
                decision_threshold=float(decision_threshold),
                validation_metrics=metrics,
                xgb_best_iteration=xgb_best_iteration,
                xgb_best_score=xgb_best_score,
            )
        )
    if not fitted_models:
        raise RuntimeError("fit_candidate_classifiers produced no models.")
    return fitted_models, pd.DataFrame(candidate_rows)


def extract_feature_weights(
    fitted_model: FittedBinaryModel,
    *,
    feature_names: list[str],
    target_name: str,
    feature_space: str,
    train_model: str,
) -> pd.DataFrame:
    model = fitted_model.estimator.named_steps["model"]
    if isinstance(model, LogisticRegression):
        feature_weight = np.asarray(model.coef_, dtype=np.float32).reshape(-1)
        weight_kind = "coefficient"
    else:
        feature_weight = np.asarray(model.feature_importances_, dtype=np.float32).reshape(-1)
        weight_kind = "gain_importance"
    if feature_weight.shape[0] != len(feature_names):
        raise ValueError(
            f"Feature-weight length mismatch: expected {len(feature_names)}, got {feature_weight.shape[0]}"
        )
    out = pd.DataFrame(
        {
            "target_name": target_name,
            "feature_space": feature_space,
            "train_model": train_model,
            "feature": feature_names,
            "feature_weight_kind": weight_kind,
            "feature_weight": feature_weight,
        }
    )
    out["abs_feature_weight"] = out["feature_weight"].abs()
    out["coefficient"] = out["feature_weight"]
    out["abs_coefficient"] = out["abs_feature_weight"]
    return out


def finite_mean(values: Sequence[float]) -> float:
    values_array = np.asarray(values, dtype=float)
    finite = values_array[np.isfinite(values_array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def finite_min(values: Sequence[float]) -> float:
    values_array = np.asarray(values, dtype=float)
    finite = values_array[np.isfinite(values_array)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def optional_float(value: Any) -> float | Any:
    if value is None or pd.isna(value):
        return pd.NA
    return float(value)


def build_model_selection_key(
    *,
    objective: str,
    oracle_mean_ood_auroc: float,
    oracle_mean_ood_pr_auc: float,
    source_val_auroc: float,
    source_val_pr_auc: float,
    source_val_balanced_accuracy: float,
    feature_count: int,
    candidate_complexity: float,
) -> tuple[float, ...]:
    if objective == "mean_ood_auroc_oracle":
        return (
            oracle_mean_ood_auroc if np.isfinite(oracle_mean_ood_auroc) else float("-inf"),
            oracle_mean_ood_pr_auc if np.isfinite(oracle_mean_ood_pr_auc) else float("-inf"),
            source_val_auroc if np.isfinite(source_val_auroc) else float("-inf"),
            source_val_pr_auc if np.isfinite(source_val_pr_auc) else float("-inf"),
            -int(feature_count),
            -float(candidate_complexity),
        )
    if objective == "source_val_auroc":
        return (
            source_val_auroc if np.isfinite(source_val_auroc) else float("-inf"),
            source_val_pr_auc if np.isfinite(source_val_pr_auc) else float("-inf"),
            source_val_balanced_accuracy if np.isfinite(source_val_balanced_accuracy) else float("-inf"),
            -int(feature_count),
            -float(candidate_complexity),
        )
    raise ValueError(f"Unsupported model selection objective: {objective!r}")


def feature_size_to_label(feature_size: int) -> str:
    return f"k{int(feature_size):03d}"


def selected_feature_size_label(feature_space: FeatureSpaceSpec, feature_size: int | None) -> str:
    if feature_space.family_title == "baseline":
        if feature_space.baseline_variant == "tfidf":
            return f"tfidf_{slugify(str(feature_space.baseline_text_field or 'text'))}"
        if feature_space.activation_variant == "final_sentence" and not feature_space.activation_use_pca:
            return "raw_final"
        return slugify(feature_space.name)
    if feature_space.family_title == "tfidf_baseline":
        return f"tfidf_{slugify(str(feature_space.baseline_text_field or 'text'))}"
    if feature_space.family_title == "attention_only":
        return "all_attention"
    if feature_size is None:
        return "all_features"
    return feature_size_to_label(int(feature_size))


def feature_size_options_for_space(feature_space: FeatureSpaceSpec, feature_size_grid: Sequence[int]) -> tuple[int | None, ...]:
    if feature_space.family_title in {"baseline", "tfidf_baseline", "attention_only"}:
        return (None,)
    return tuple(int(value) for value in feature_size_grid)


def concatenate_split_matrices(parts: list[Any]) -> Any:
    if not parts:
        raise ValueError("concatenate_split_matrices received no parts.")
    if len(parts) == 1:
        return parts[0]
    if any(sp.issparse(part) for part in parts):
        sparse_parts = [
            part.tocsr() if sp.issparse(part) else sp.csr_matrix(np.asarray(part, dtype=np.float32))
            for part in parts
        ]
        return sp.hstack(sparse_parts, format="csr")
    return np.concatenate(parts, axis=1)


def concatenate_model_bundle_matrices(
    bundle_keys: Sequence[str],
    bundle_matrices: dict[str, dict[str, Any]],
    *,
    split_name: str,
) -> Any:
    parts = [bundle_matrices[bundle_key][split_name] for bundle_key in bundle_keys]
    if any(sp.issparse(part) for part in parts):
        sparse_parts = [
            part.tocsr() if sp.issparse(part) else sp.csr_matrix(np.asarray(part, dtype=np.float32))
            for part in parts
        ]
        return sp.vstack(sparse_parts, format="csr")
    return np.concatenate([np.asarray(part, dtype=np.float32) for part in parts], axis=0)


def concatenate_model_targets(
    bundle_keys: Sequence[str],
    split_cache_by_bundle: dict[str, dict[str, Any]],
    *,
    split_name: str,
    target_name: str,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(split_cache_by_bundle[bundle_key][f"y_{split_name}__{target_name}"], dtype=np.int8)
            for bundle_key in bundle_keys
        ],
        axis=0,
    )


def concatenate_model_metadata(
    bundle_keys: Sequence[str],
    feature_metadata_by_bundle: dict[str, pd.DataFrame],
    split_cache_by_bundle: dict[str, dict[str, Any]],
    *,
    split_name: str,
    target_name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for bundle_key in bundle_keys:
        bundle_df = feature_metadata_by_bundle[bundle_key]
        split_mask = split_cache_by_bundle[bundle_key][f"{split_name}_mask"]
        target_col = f"label__{target_name}"
        subset = bundle_df.loc[split_mask, ["env_name", "model_display", "example_id", "sentence_idx", target_col]].copy()
        subset = subset.rename(columns={target_col: "label"})
        frames.append(subset)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["env_name", "model_display", "example_id", "sentence_idx", "label"]
    )


def normalize_scenario_key(raw_value: str) -> str:
    value = slugify(str(raw_value or "single_source_ood"))
    alias_map = {
        "single_source_ood": "single_source_ood",
        "train_one_eval_all": "single_source_ood",
        "single_env_ood": "single_source_ood",
        "one_to_all": "single_source_ood",
        "holdout_env_ood": "holdout_env_ood",
        "train_four_holdout_one": "holdout_env_ood",
        "leave_one_env_out": "holdout_env_ood",
        "four_to_one": "holdout_env_ood",
    }
    if value in alias_map:
        return alias_map[value]
    raise ValueError(
        f"Unsupported scenario {raw_value!r}. "
        "Expected one of ['single_source_ood', 'holdout_env_ood']."
    )


@dataclass(frozen=True)
class ExperimentRunSpec:
    scenario_name: str
    scenario_title: str
    train_env_label: str
    source_envs: tuple[str, ...]
    source_envs_label: str
    ood_envs: tuple[str, ...]
    heldout_env: str | None


def build_experiment_run_specs(selected_scenarios: Sequence[str]) -> tuple[list[ExperimentRunSpec], dict[str, list[str]]]:
    run_specs: list[ExperimentRunSpec] = []
    train_axis_labels_by_scenario: dict[str, list[str]] = {}
    for scenario_name in selected_scenarios:
        scenario_title = str(SCENARIO_TITLES[scenario_name])
        scenario_labels: list[str] = []
        if scenario_name == "single_source_ood":
            for env_name in ENV_ORDER:
                scenario_labels.append(env_name)
                run_specs.append(
                    ExperimentRunSpec(
                        scenario_name=scenario_name,
                        scenario_title=scenario_title,
                        train_env_label=env_name,
                        source_envs=(env_name,),
                        source_envs_label=env_name,
                        ood_envs=tuple(other_env for other_env in ENV_ORDER if other_env != env_name),
                        heldout_env=None,
                    )
                )
            train_axis_labels_by_scenario[scenario_name] = scenario_labels
            continue
        if scenario_name == "holdout_env_ood":
            for heldout_env in ENV_ORDER:
                train_env_label = f"All except {heldout_env}"
                source_envs = tuple(env_name for env_name in ENV_ORDER if env_name != heldout_env)
                scenario_labels.append(train_env_label)
                run_specs.append(
                    ExperimentRunSpec(
                        scenario_name=scenario_name,
                        scenario_title=scenario_title,
                        train_env_label=train_env_label,
                        source_envs=source_envs,
                        source_envs_label=", ".join(source_envs),
                        ood_envs=(heldout_env,),
                        heldout_env=heldout_env,
                    )
                )
            train_axis_labels_by_scenario[scenario_name] = scenario_labels
            continue
        raise ValueError(f"Unsupported scenario_name={scenario_name!r}")
    return run_specs, train_axis_labels_by_scenario


def assemble_env_split_matrix(
    env_cache: dict[str, Any],
    *,
    split_name: str,
    feature_space: FeatureSpaceSpec,
    attention_dim: int,
    activation_dim: int,
) -> Any:
    parts: list[Any] = []
    if feature_space.uses_attention:
        parts.append(np.asarray(env_cache[f"{split_name}_attention_pool"][:, :attention_dim], dtype=np.float32))
    if feature_space.activation_variant is not None:
        activation_pool = np.asarray(env_cache[f"{split_name}_activation_pool"], dtype=np.float32)
        if feature_space.activation_use_pca:
            parts.append(np.asarray(activation_pool[:, :activation_dim], dtype=np.float32))
        else:
            parts.append(activation_pool)
    if feature_space.baseline_variant is not None:
        baseline_pool = env_cache[f"{split_name}_baseline_pool"]
        if sp.issparse(baseline_pool):
            parts.append(baseline_pool.tocsr())
        else:
            parts.append(np.asarray(baseline_pool, dtype=np.float32))
    if not parts:
        raise ValueError("assemble_env_split_matrix received no feature parts.")
    if len(parts) == 1:
        return parts[0]
    if any(sp.issparse(part) for part in parts):
        sparse_parts = [
            part.tocsr() if sp.issparse(part) else sp.csr_matrix(np.asarray(part, dtype=np.float32))
            for part in parts
        ]
        return sp.hstack(sparse_parts, format="csr")
    return np.concatenate(parts, axis=1)


def concatenate_source_split_matrices_env(
    env_pool_cache: dict[str, dict[str, Any]],
    *,
    source_envs: tuple[str, ...],
    split_name: str,
    feature_space: FeatureSpaceSpec,
    attention_dim: int,
    activation_dim: int,
    target_name: str,
) -> tuple[Any, np.ndarray]:
    x_parts: list[Any] = []
    y_parts: list[np.ndarray] = []
    for env_name in source_envs:
        env_cache = env_pool_cache[env_name]
        x_parts.append(
            assemble_env_split_matrix(
                env_cache,
                split_name=split_name,
                feature_space=feature_space,
                attention_dim=attention_dim,
                activation_dim=activation_dim,
            )
        )
        y_parts.append(np.asarray(env_cache[f"y_{split_name}__{target_name}"], dtype=np.int8))
    if not x_parts or not y_parts:
        raise ValueError(f"No split parts available for source_envs={source_envs} / split_name={split_name}.")
    if any(sp.issparse(part) for part in x_parts):
        sparse_parts = [
            part.tocsr() if sp.issparse(part) else sp.csr_matrix(np.asarray(part, dtype=np.float32))
            for part in x_parts
        ]
        x_out = sp.vstack(sparse_parts, format="csr")
    else:
        x_out = np.concatenate(x_parts, axis=0)
    return x_out, np.concatenate(y_parts, axis=0)


def evaluate_env_predictions(
    *,
    run_spec: ExperimentRunSpec,
    feature_space_name: str,
    feature_space_title: str,
    feature_family_group: str,
    feature_space_attention_subset_key: str,
    feature_space_attention_subset_title: str,
    target_name: str,
    target_title: str,
    test_env: str,
    eval_role: str,
    feature_size: int | None,
    feature_size_label: str,
    attention_feature_count: int,
    activation_feature_count: int,
    selected_feature_count: int,
    effective_activation_pca_dim: int | None,
    alignment_detail: str,
    eval_df: pd.DataFrame,
    y_score: np.ndarray,
    decision_threshold: float,
    calibration_bins: int,
    fixed_recall_levels: Sequence[float],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y_true = eval_df["label"].to_numpy(dtype=np.int8, copy=False)
    metrics = summarize_score_metrics(y_true, y_score, decision_threshold=decision_threshold)
    metrics_extra = safe_binary_metrics(y_true, y_score)
    average_precision = metrics.get("average_precision", pd.NA)
    if pd.isna(average_precision):
        average_precision = float(metrics_extra["pr_auc"])
    metric_row = {
        "scenario_name": run_spec.scenario_name,
        "scenario_title": run_spec.scenario_title,
        "target_name": target_name,
        "target_title": target_title,
        "feature_space": feature_space_name,
        "feature_space_title": feature_space_title,
        "feature_family_group": feature_family_group,
        "feature_space_attention_subset_key": feature_space_attention_subset_key,
        "feature_space_attention_subset_title": feature_space_attention_subset_title,
        "train_env": run_spec.train_env_label,
        "source_envs": run_spec.source_envs_label,
        "source_env_count": int(len(run_spec.source_envs)),
        "heldout_env": run_spec.heldout_env if run_spec.heldout_env is not None else pd.NA,
        "test_env": test_env,
        "eval_role": eval_role,
        "feature_size": pd.NA if feature_size is None else int(feature_size),
        "feature_size_label": feature_size_label,
        "attention_feature_count": int(attention_feature_count),
        "activation_feature_count": int(activation_feature_count),
        "selected_feature_count": int(selected_feature_count),
        "effective_activation_pca_dim": pd.NA if effective_activation_pca_dim is None else int(effective_activation_pca_dim),
        "alignment_detail": alignment_detail,
        "n_examples": int((eval_df["env_name"].astype(str) + "::" + eval_df["example_id"].astype(str)).nunique()),
        **metrics,
        "average_precision": float(average_precision),
        "pr_auc": float(metrics_extra["pr_auc"]),
        "brier": float(metrics_extra["brier"]),
        "positive_count": float(metrics_extra["positive_count"]),
        "negative_count": float(metrics_extra["negative_count"]),
        "base_rate": float(metrics_extra["base_rate"]),
    }
    calibration_df = calibration_curve_frame(
        y_true=y_true,
        y_score=y_score,
        n_bins=int(calibration_bins),
    )
    if not calibration_df.empty:
        calibration_df["scenario_name"] = run_spec.scenario_name
        calibration_df["scenario_title"] = run_spec.scenario_title
        calibration_df["target_name"] = target_name
        calibration_df["feature_space"] = feature_space_name
        calibration_df["feature_space_title"] = feature_space_title
        calibration_df["feature_family_group"] = feature_family_group
        calibration_df["train_env"] = run_spec.train_env_label
        calibration_df["source_envs"] = run_spec.source_envs_label
        calibration_df["heldout_env"] = run_spec.heldout_env if run_spec.heldout_env is not None else pd.NA
        calibration_df["test_env"] = test_env
        calibration_df["eval_role"] = eval_role
        calibration_df["feature_size_label"] = feature_size_label
    fpr_df = fpr_at_fixed_recalls(
        y_true=y_true,
        y_score=y_score,
        recall_levels=fixed_recall_levels,
    )
    if not fpr_df.empty:
        fpr_df["scenario_name"] = run_spec.scenario_name
        fpr_df["scenario_title"] = run_spec.scenario_title
        fpr_df["target_name"] = target_name
        fpr_df["feature_space"] = feature_space_name
        fpr_df["feature_space_title"] = feature_space_title
        fpr_df["feature_family_group"] = feature_family_group
        fpr_df["train_env"] = run_spec.train_env_label
        fpr_df["source_envs"] = run_spec.source_envs_label
        fpr_df["heldout_env"] = run_spec.heldout_env if run_spec.heldout_env is not None else pd.NA
        fpr_df["test_env"] = test_env
        fpr_df["eval_role"] = eval_role
        fpr_df["feature_size_label"] = feature_size_label
    return metric_row, calibration_df, fpr_df


def summarize_transfer_metrics_env(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "scenario_name",
        "scenario_title",
        "target_name",
        "target_title",
        "feature_space",
        "feature_space_title",
        "feature_family_group",
        "feature_size",
        "feature_size_label",
    ]
    for group_values, group_df in metrics_df.groupby(group_cols, dropna=False, sort=False):
        (
            scenario_name,
            scenario_title,
            target_name,
            target_title,
            feature_space,
            feature_space_title,
            feature_family_group,
            feature_size,
            feature_size_label,
        ) = group_values
        diagonal_df = group_df.loc[group_df["eval_role"].eq("val")].copy()
        ood_df = group_df.loc[group_df["eval_role"].eq("ood")].copy()
        meta = group_df.iloc[0]
        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_title": scenario_title,
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
                "alignment_detail": str(meta.get("alignment_detail", "")),
                "n_source_val_envs": int(diagonal_df["test_env"].nunique()),
                "n_ood_envs": int(ood_df["test_env"].nunique()),
                "mean_val_auroc": safe_metric_mean(diagonal_df["auroc"]),
                "min_val_auroc": safe_metric_min(diagonal_df["auroc"]),
                "mean_ood_auroc": safe_metric_mean(ood_df["auroc"]),
                "min_ood_auroc": safe_metric_min(ood_df["auroc"]),
                "std_ood_auroc": safe_metric_std(ood_df["auroc"]),
                "mean_val_average_precision": safe_metric_mean(diagonal_df["average_precision"]),
                "mean_ood_average_precision": safe_metric_mean(ood_df["average_precision"]),
                "mean_val_pr_auc": safe_metric_mean(diagonal_df["pr_auc"]),
                "mean_ood_pr_auc": safe_metric_mean(ood_df["pr_auc"]),
                "mean_val_brier": safe_metric_mean(diagonal_df["brier"]),
                "mean_ood_brier": safe_metric_mean(ood_df["brier"]),
                "mean_val_balanced_accuracy": safe_metric_mean(diagonal_df["balanced_accuracy"]),
                "mean_ood_balanced_accuracy": safe_metric_mean(ood_df["balanced_accuracy"]),
            }
        )
    if not summary_rows:
        return pd.DataFrame()
    out = pd.DataFrame(summary_rows)
    out["_feature_size_sort"] = pd.to_numeric(out["feature_size"], errors="coerce").fillna(-1).astype(int)
    out = out.sort_values(
        ["scenario_name", "target_name", "_feature_size_sort", "mean_ood_auroc", "mean_ood_pr_auc", "mean_val_auroc"],
        ascending=[True, True, True, False, False, False],
    ).drop(columns="_feature_size_sort")
    return out.reset_index(drop=True)


def summarize_train_env_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "scenario_name",
        "scenario_title",
        "target_name",
        "target_title",
        "feature_space",
        "feature_space_title",
        "feature_family_group",
        "feature_size",
        "feature_size_label",
        "train_env",
        "source_envs",
        "source_env_count",
        "heldout_env",
    ]
    for group_values, group_df in metrics_df.groupby(group_cols, dropna=False, sort=False):
        (
            scenario_name,
            scenario_title,
            target_name,
            target_title,
            feature_space,
            feature_space_title,
            feature_family_group,
            feature_size,
            feature_size_label,
            train_env,
            source_envs,
            source_env_count,
            heldout_env,
        ) = group_values
        diagonal_df = group_df.loc[group_df["eval_role"].eq("val")].copy()
        ood_df = group_df.loc[group_df["eval_role"].eq("ood")].copy()
        meta = group_df.iloc[0]
        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_title": scenario_title,
                "target_name": target_name,
                "target_title": target_title,
                "feature_space": feature_space,
                "feature_space_title": feature_space_title,
                "feature_family_group": feature_family_group,
                "feature_size": feature_size,
                "feature_size_label": feature_size_label,
                "train_env": train_env,
                "source_envs": source_envs,
                "source_env_count": int(source_env_count),
                "heldout_env": heldout_env,
                "source_val_auroc": safe_metric_mean(diagonal_df["auroc"]),
                "source_val_average_precision": safe_metric_mean(diagonal_df["average_precision"]),
                "source_val_pr_auc": safe_metric_mean(diagonal_df["pr_auc"]),
                "source_val_balanced_accuracy": safe_metric_mean(diagonal_df["balanced_accuracy"]),
                "source_val_brier": safe_metric_mean(diagonal_df["brier"]),
                "mean_ood_auroc": safe_metric_mean(ood_df["auroc"]),
                "min_ood_auroc": safe_metric_min(ood_df["auroc"]),
                "std_ood_auroc": safe_metric_std(ood_df["auroc"]),
                "mean_ood_average_precision": safe_metric_mean(ood_df["average_precision"]),
                "mean_ood_pr_auc": safe_metric_mean(ood_df["pr_auc"]),
                "mean_ood_balanced_accuracy": safe_metric_mean(ood_df["balanced_accuracy"]),
                "mean_ood_brier": safe_metric_mean(ood_df["brier"]),
                "n_source_val_envs": int(diagonal_df["test_env"].nunique()),
                "n_ood_envs": int(ood_df["test_env"].nunique()),
                "attention_feature_count": int(meta["attention_feature_count"]),
                "activation_feature_count": int(meta["activation_feature_count"]),
                "selected_feature_count": int(meta["selected_feature_count"]),
                "effective_activation_pca_dim": meta.get("effective_activation_pca_dim", pd.NA),
                "alignment_detail": str(meta.get("alignment_detail", "")),
                "chosen_c": optional_float(meta.get("chosen_c", pd.NA)),
                "chosen_max_depth": optional_float(meta.get("chosen_max_depth", pd.NA)),
                "decision_threshold": float(meta["decision_threshold"]),
                "selected_features_path": str(meta["selected_features_path"]),
                "coefficients_path": str(meta.get("coefficients_path", "")),
                "model_artifact_path": str(meta.get("model_artifact_path", "")),
            }
        )
    if not summary_rows:
        return pd.DataFrame()
    out = pd.DataFrame(summary_rows)
    out["_feature_size_sort"] = pd.to_numeric(out["feature_size"], errors="coerce").fillna(-1).astype(int)
    out = out.sort_values(
        ["scenario_name", "target_name", "_feature_size_sort", "mean_ood_auroc", "mean_ood_pr_auc", "source_val_auroc"],
        ascending=[True, True, True, False, False, False],
    ).drop(columns="_feature_size_sort")
    return out.reset_index(drop=True)


def summarize_target_env_breakdown(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    group_cols = [
        "scenario_name",
        "scenario_title",
        "target_name",
        "target_title",
        "feature_space",
        "feature_space_title",
        "feature_family_group",
        "feature_size",
        "feature_size_label",
        "eval_role",
        "test_env",
    ]
    rows: list[dict[str, Any]] = []
    for group_values, group_df in metrics_df.groupby(group_cols, dropna=False, sort=False):
        (
            scenario_name,
            scenario_title,
            target_name,
            target_title,
            feature_space,
            feature_space_title,
            feature_family_group,
            feature_size,
            feature_size_label,
            eval_role,
            test_env,
        ) = group_values
        rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_title": scenario_title,
                "target_name": target_name,
                "target_title": target_title,
                "feature_space": feature_space,
                "feature_space_title": feature_space_title,
                "feature_family_group": feature_family_group,
                "feature_size": feature_size,
                "feature_size_label": feature_size_label,
                "eval_role": eval_role,
                "test_env": test_env,
                "n_train_env_runs": int(group_df["train_env"].nunique()),
                "mean_auroc": safe_metric_mean(group_df["auroc"]),
                "mean_average_precision": safe_metric_mean(group_df["average_precision"]),
                "mean_pr_auc": safe_metric_mean(group_df["pr_auc"]),
                "mean_brier": safe_metric_mean(group_df["brier"]),
                "mean_balanced_accuracy": safe_metric_mean(group_df["balanced_accuracy"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_confusion_counts_env(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "scenario_name",
        "scenario_title",
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
            scenario_name,
            scenario_title,
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
                "scenario_name": scenario_name,
                "scenario_title": scenario_title,
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
    return pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()


def family_uses_requested_feature_size_env(feature_family_group: str) -> bool:
    return feature_family_group in {"activation_only", "attention_plus_activation"}


def build_family_panel_selection_env(transfer_summary_df: pd.DataFrame, feature_size_grid: Sequence[int]) -> pd.DataFrame:
    if transfer_summary_df.empty:
        return pd.DataFrame()
    family_sort = {family: idx for idx, family in enumerate(FAMILY_PANEL_ORDER)}
    rows: list[dict[str, Any]] = []
    for scenario_name, scenario_title in SCENARIO_TITLES.items():
        if transfer_summary_df["scenario_name"].eq(scenario_name).sum() == 0:
            continue
        for target_name, target_spec in TARGET_SPECS.items():
            target_title = str(target_spec["title"])
            for requested_feature_size in feature_size_grid:
                for family in FAMILY_PANEL_ORDER:
                    subset = transfer_summary_df.loc[
                        transfer_summary_df["scenario_name"].eq(scenario_name)
                        & transfer_summary_df["target_name"].eq(target_name)
                        & transfer_summary_df["feature_family_group"].eq(family)
                    ].copy()
                    if family_uses_requested_feature_size_env(family):
                        subset = subset.loc[
                            pd.to_numeric(subset["feature_size"], errors="coerce").eq(int(requested_feature_size))
                        ]
                    if subset.empty:
                        continue
                    subset = subset.sort_values(
                        ["mean_ood_auroc", "mean_ood_pr_auc", "mean_val_auroc", "selected_feature_count"],
                        ascending=[False, False, False, True],
                    ).reset_index(drop=True)
                    best = subset.iloc[0]
                    rows.append(
                        {
                            "scenario_name": scenario_name,
                            "scenario_title": scenario_title,
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
                            "alignment_detail": str(best.get("alignment_detail", "")),
                            "mean_val_auroc": float(best["mean_val_auroc"]),
                            "mean_ood_auroc": float(best["mean_ood_auroc"]),
                            "min_ood_auroc": float(best["min_ood_auroc"]),
                            "std_ood_auroc": float(best["std_ood_auroc"]),
                            "mean_ood_average_precision": float(best["mean_ood_average_precision"]),
                            "mean_ood_pr_auc": float(best["mean_ood_pr_auc"]),
                            "mean_ood_brier": float(best["mean_ood_brier"]),
                        }
                    )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["_family_sort"] = out["feature_family_group"].map(family_sort).fillna(len(family_sort)).astype(int)
    out = out.sort_values(
        ["scenario_name", "target_name", "requested_feature_size", "_family_sort", "mean_ood_auroc"],
        ascending=[True, True, True, True, False],
    ).drop(columns="_family_sort")
    return out.reset_index(drop=True)


def build_best_family_models_env(
    train_env_summary_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel_row in panel_selection_df.itertuples(index=False):
        subset = train_env_summary_df.loc[
            train_env_summary_df["scenario_name"].eq(panel_row.scenario_name)
            & train_env_summary_df["target_name"].eq(panel_row.target_name)
            & train_env_summary_df["feature_space"].eq(panel_row.selected_feature_space)
            & train_env_summary_df["feature_size_label"].eq(panel_row.source_feature_size_label)
        ].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["mean_ood_auroc", "mean_ood_pr_auc", "source_val_auroc"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        best = subset.iloc[0]
        rows.append(
            {
                "scenario_name": panel_row.scenario_name,
                "scenario_title": panel_row.scenario_title,
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
                "source_envs": str(best["source_envs"]),
                "source_env_count": int(best["source_env_count"]),
                "heldout_env": best["heldout_env"],
                "source_val_auroc": float(best["source_val_auroc"]),
                "mean_ood_auroc": float(best["mean_ood_auroc"]),
                "min_ood_auroc": float(best["min_ood_auroc"]),
                "std_ood_auroc": float(best["std_ood_auroc"]),
                "mean_ood_average_precision": float(best["mean_ood_average_precision"]),
                "mean_ood_pr_auc": float(best["mean_ood_pr_auc"]),
                "mean_ood_balanced_accuracy": float(best["mean_ood_balanced_accuracy"]),
                "mean_ood_brier": float(best["mean_ood_brier"]),
                "attention_feature_count": int(best["attention_feature_count"]),
                "activation_feature_count": int(best["activation_feature_count"]),
                "selected_feature_count": int(best["selected_feature_count"]),
                "chosen_c": optional_float(best["chosen_c"]),
                "chosen_max_depth": optional_float(best["chosen_max_depth"]),
                "decision_threshold": float(best["decision_threshold"]),
                "effective_activation_pca_dim": best.get("effective_activation_pca_dim", pd.NA),
                "alignment_detail": str(best.get("alignment_detail", "")),
                "selected_features_path": str(best["selected_features_path"]),
                "coefficients_path": str(best["coefficients_path"]),
                "model_artifact_path": str(best.get("model_artifact_path", "")),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_selected_panel_confusion_summary_env(
    metrics_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
    *,
    eval_role: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel_row in panel_selection_df.itertuples(index=False):
        subset = metrics_df.loc[
            metrics_df["scenario_name"].eq(panel_row.scenario_name)
            & metrics_df["target_name"].eq(panel_row.target_name)
            & metrics_df["feature_space"].eq(panel_row.selected_feature_space)
            & metrics_df["feature_size_label"].eq(panel_row.source_feature_size_label)
            & metrics_df["eval_role"].eq(eval_role)
        ].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "scenario_name": panel_row.scenario_name,
                "scenario_title": panel_row.scenario_title,
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
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_transfer_matrix_for_selection_env(
    metrics_df: pd.DataFrame,
    *,
    scenario_name: str,
    target_name: str,
    feature_space: str,
    feature_size_label: str,
    metric_column: str,
    train_axis_labels_by_scenario: dict[str, list[str]],
) -> pd.DataFrame:
    subset = metrics_df.loc[
        metrics_df["scenario_name"].eq(scenario_name)
        & metrics_df["target_name"].eq(target_name)
        & metrics_df["feature_space"].eq(feature_space)
        & metrics_df["feature_size_label"].eq(feature_size_label)
    ].copy()
    row_labels = train_axis_labels_by_scenario[str(scenario_name)]
    matrix = pd.DataFrame(index=row_labels, columns=ENV_ORDER, dtype=float)
    for row in subset.itertuples(index=False):
        value = getattr(row, metric_column)
        matrix.loc[str(row.train_env), str(row.test_env)] = float(value) if pd.notna(value) else float("nan")
    return matrix


def build_confusion_matrix_df_env(
    confusion_df: pd.DataFrame,
    *,
    scenario_name: str,
    target_name: str,
    requested_feature_size: int,
    feature_family_group: str,
    eval_role: str,
) -> pd.DataFrame | None:
    subset = confusion_df.loc[
        confusion_df["scenario_name"].eq(scenario_name)
        & confusion_df["target_name"].eq(target_name)
        & confusion_df["requested_feature_size"].eq(int(requested_feature_size))
        & confusion_df["feature_family_group"].eq(feature_family_group)
        & confusion_df["eval_role"].eq(eval_role)
    ].copy()
    if subset.empty:
        return None
    row = subset.iloc[0]
    negative_label = TARGET_SPECS[target_name]["negative_label"]
    positive_label = TARGET_SPECS[target_name]["positive_label"]
    return pd.DataFrame(
        [
            [int(row["sum_tn"]), int(row["sum_fp"])],
            [int(row["sum_fn"]), int(row["sum_tp"])],
        ],
        index=[f"Actual {negative_label}", f"Actual {positive_label}"],
        columns=[f"Pred {negative_label}", f"Pred {positive_label}"],
    )


def export_selected_family_panel_tables_env(
    output_root: Path,
    metrics_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    train_axis_labels_by_scenario: dict[str, list[str]],
) -> None:
    export_root = output_root / "selected_family_panel_tables"
    export_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for panel_row in panel_selection_df.itertuples(index=False):
        panel_dir = export_root / str(panel_row.scenario_name) / panel_row.target_name / str(panel_row.requested_feature_size_label)
        panel_dir.mkdir(parents=True, exist_ok=True)
        auroc_matrix_df = build_transfer_matrix_for_selection_env(
            metrics_df,
            scenario_name=str(panel_row.scenario_name),
            target_name=str(panel_row.target_name),
            feature_space=str(panel_row.selected_feature_space),
            feature_size_label=str(panel_row.source_feature_size_label),
            metric_column="auroc",
            train_axis_labels_by_scenario=train_axis_labels_by_scenario,
        )
        auroc_matrix_path = panel_dir / f"{panel_row.feature_family_group}__auroc_matrix.csv"
        auroc_matrix_df.to_csv(auroc_matrix_path, index=True)
        pr_auc_matrix_df = build_transfer_matrix_for_selection_env(
            metrics_df,
            scenario_name=str(panel_row.scenario_name),
            target_name=str(panel_row.target_name),
            feature_space=str(panel_row.selected_feature_space),
            feature_size_label=str(panel_row.source_feature_size_label),
            metric_column="pr_auc",
            train_axis_labels_by_scenario=train_axis_labels_by_scenario,
        )
        pr_auc_matrix_path = panel_dir / f"{panel_row.feature_family_group}__pr_auc_matrix.csv"
        pr_auc_matrix_df.to_csv(pr_auc_matrix_path, index=True)
        val_confusion_matrix_df = build_confusion_matrix_df_env(
            confusion_df,
            scenario_name=str(panel_row.scenario_name),
            target_name=str(panel_row.target_name),
            requested_feature_size=int(panel_row.requested_feature_size),
            feature_family_group=str(panel_row.feature_family_group),
            eval_role="val",
        )
        val_confusion_path = panel_dir / f"{panel_row.feature_family_group}__val_confusion_counts.csv"
        if val_confusion_matrix_df is not None:
            val_confusion_matrix_df.to_csv(val_confusion_path, index=True)
        ood_confusion_matrix_df = build_confusion_matrix_df_env(
            confusion_df,
            scenario_name=str(panel_row.scenario_name),
            target_name=str(panel_row.target_name),
            requested_feature_size=int(panel_row.requested_feature_size),
            feature_family_group=str(panel_row.feature_family_group),
            eval_role="ood",
        )
        ood_confusion_path = panel_dir / f"{panel_row.feature_family_group}__ood_confusion_counts.csv"
        if ood_confusion_matrix_df is not None:
            ood_confusion_matrix_df.to_csv(ood_confusion_path, index=True)
        manifest_rows.append(
            {
                "scenario_name": panel_row.scenario_name,
                "scenario_title": panel_row.scenario_title,
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
                "auroc_matrix_path": str(auroc_matrix_path),
                "pr_auc_matrix_path": str(pr_auc_matrix_path),
                "val_confusion_path": str(val_confusion_path),
                "ood_confusion_path": str(ood_confusion_path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(export_root / "panel_table_manifest.csv", index=False)


def evaluate_predictions(
    *,
    feature_space_name: str,
    feature_space_title: str,
    feature_family_group: str,
    target_name: str,
    target_title: str,
    source_model: str,
    test_model: str,
    eval_role: str,
    feature_size: int | None,
    feature_size_label: str,
    attention_feature_count: int,
    activation_feature_count: int,
    selected_feature_count: int,
    effective_activation_pca_dim: int | None,
    alignment_detail: str,
    eval_df: pd.DataFrame,
    y_score: np.ndarray,
    decision_threshold: float,
    calibration_bins: int,
    fixed_recall_levels: Sequence[float],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y_true = eval_df["label"].to_numpy(dtype=np.int8, copy=False)
    metrics = summarize_score_metrics(y_true, y_score, decision_threshold=decision_threshold)
    metrics_extra = safe_binary_metrics(y_true, y_score)
    metric_row = {
        "scenario_name": SCENARIO_NAME,
        "scenario_title": SCENARIO_TITLE,
        "target_name": target_name,
        "target_title": target_title,
        "feature_space": feature_space_name,
        "feature_space_title": feature_space_title,
        "feature_family_group": feature_family_group,
        "train_model": source_model,
        "source_models": source_model,
        "source_model_count": 1,
        "test_model": test_model,
        "eval_role": eval_role,
        "feature_size": pd.NA if feature_size is None else int(feature_size),
        "feature_size_label": feature_size_label,
        "attention_feature_count": int(attention_feature_count),
        "activation_feature_count": int(activation_feature_count),
        "selected_feature_count": int(selected_feature_count),
        "effective_activation_pca_dim": pd.NA if effective_activation_pca_dim is None else int(effective_activation_pca_dim),
        "alignment_detail": alignment_detail,
        "n_examples": int((eval_df["env_name"].astype(str) + "::" + eval_df["example_id"].astype(str)).nunique()),
        **metrics,
        "pr_auc": float(metrics_extra["pr_auc"]),
        "brier": float(metrics_extra["brier"]),
        "positive_count": float(metrics_extra["positive_count"]),
        "negative_count": float(metrics_extra["negative_count"]),
        "base_rate": float(metrics_extra["base_rate"]),
    }
    calibration_df = calibration_curve_frame(
        y_true=y_true,
        y_score=y_score,
        n_bins=int(calibration_bins),
    )
    if not calibration_df.empty:
        calibration_df["scenario_name"] = SCENARIO_NAME
        calibration_df["target_name"] = target_name
        calibration_df["feature_space"] = feature_space_name
        calibration_df["feature_space_title"] = feature_space_title
        calibration_df["feature_family_group"] = feature_family_group
        calibration_df["train_model"] = source_model
        calibration_df["test_model"] = test_model
        calibration_df["eval_role"] = eval_role
        calibration_df["feature_size_label"] = feature_size_label
    fpr_df = fpr_at_fixed_recalls(
        y_true=y_true,
        y_score=y_score,
        recall_levels=fixed_recall_levels,
    )
    if not fpr_df.empty:
        fpr_df["scenario_name"] = SCENARIO_NAME
        fpr_df["target_name"] = target_name
        fpr_df["feature_space"] = feature_space_name
        fpr_df["feature_space_title"] = feature_space_title
        fpr_df["feature_family_group"] = feature_family_group
        fpr_df["train_model"] = source_model
        fpr_df["test_model"] = test_model
        fpr_df["eval_role"] = eval_role
        fpr_df["feature_size_label"] = feature_size_label
    return metric_row, calibration_df, fpr_df


def summarize_transfer_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "scenario_name",
        "scenario_title",
        "target_name",
        "target_title",
        "feature_space",
        "feature_space_title",
        "feature_family_group",
        "feature_size",
        "feature_size_label",
    ]
    for group_values, group_df in metrics_df.groupby(group_cols, dropna=False, sort=False):
        (
            scenario_name,
            scenario_title,
            target_name,
            target_title,
            feature_space,
            feature_space_title,
            feature_family_group,
            feature_size,
            feature_size_label,
        ) = group_values
        diagonal_df = group_df.loc[group_df["eval_role"].eq("val")].copy()
        ood_df = group_df.loc[group_df["eval_role"].eq("ood")].copy()
        meta = group_df.iloc[0]
        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_title": scenario_title,
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
                "alignment_detail": str(meta.get("alignment_detail", "")),
                "n_source_val_models": int(diagonal_df["test_model"].nunique()),
                "n_ood_models": int(ood_df["test_model"].nunique()),
                "mean_val_auroc": safe_metric_mean(diagonal_df["auroc"]),
                "min_val_auroc": safe_metric_min(diagonal_df["auroc"]),
                "mean_ood_auroc": safe_metric_mean(ood_df["auroc"]),
                "min_ood_auroc": safe_metric_min(ood_df["auroc"]),
                "std_ood_auroc": safe_metric_std(ood_df["auroc"]),
                "mean_val_average_precision": safe_metric_mean(diagonal_df["average_precision"]),
                "mean_ood_average_precision": safe_metric_mean(ood_df["average_precision"]),
                "mean_val_pr_auc": safe_metric_mean(diagonal_df["pr_auc"]),
                "mean_ood_pr_auc": safe_metric_mean(ood_df["pr_auc"]),
                "mean_val_brier": safe_metric_mean(diagonal_df["brier"]),
                "mean_ood_brier": safe_metric_mean(ood_df["brier"]),
                "mean_val_balanced_accuracy": safe_metric_mean(diagonal_df["balanced_accuracy"]),
                "mean_ood_balanced_accuracy": safe_metric_mean(ood_df["balanced_accuracy"]),
            }
        )
    if not summary_rows:
        return pd.DataFrame()
    out = pd.DataFrame(summary_rows)
    out["_feature_size_sort"] = pd.to_numeric(out["feature_size"], errors="coerce").fillna(-1).astype(int)
    out = out.sort_values(
        ["scenario_name", "target_name", "_feature_size_sort", "mean_ood_auroc", "mean_ood_pr_auc", "mean_val_auroc"],
        ascending=[True, True, True, False, False, False],
    ).drop(columns="_feature_size_sort")
    return out.reset_index(drop=True)


def summarize_train_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "scenario_name",
        "scenario_title",
        "target_name",
        "target_title",
        "feature_space",
        "feature_space_title",
        "feature_family_group",
        "feature_size",
        "feature_size_label",
        "train_model",
        "source_models",
        "source_model_count",
    ]
    for group_values, group_df in metrics_df.groupby(group_cols, dropna=False, sort=False):
        (
            scenario_name,
            scenario_title,
            target_name,
            target_title,
            feature_space,
            feature_space_title,
            feature_family_group,
            feature_size,
            feature_size_label,
            train_model,
            source_models,
            source_model_count,
        ) = group_values
        diagonal_df = group_df.loc[group_df["eval_role"].eq("val")].copy()
        ood_df = group_df.loc[group_df["eval_role"].eq("ood")].copy()
        meta = group_df.iloc[0]
        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_title": scenario_title,
                "target_name": target_name,
                "target_title": target_title,
                "feature_space": feature_space,
                "feature_space_title": feature_space_title,
                "feature_family_group": feature_family_group,
                "feature_size": feature_size,
                "feature_size_label": feature_size_label,
                "train_model": train_model,
                "source_models": source_models,
                "source_model_count": int(source_model_count),
                "source_val_auroc": safe_metric_mean(diagonal_df["auroc"]),
                "source_val_average_precision": safe_metric_mean(diagonal_df["average_precision"]),
                "source_val_pr_auc": safe_metric_mean(diagonal_df["pr_auc"]),
                "source_val_balanced_accuracy": safe_metric_mean(diagonal_df["balanced_accuracy"]),
                "source_val_brier": safe_metric_mean(diagonal_df["brier"]),
                "mean_ood_auroc": safe_metric_mean(ood_df["auroc"]),
                "min_ood_auroc": safe_metric_min(ood_df["auroc"]),
                "std_ood_auroc": safe_metric_std(ood_df["auroc"]),
                "mean_ood_average_precision": safe_metric_mean(ood_df["average_precision"]),
                "mean_ood_pr_auc": safe_metric_mean(ood_df["pr_auc"]),
                "mean_ood_balanced_accuracy": safe_metric_mean(ood_df["balanced_accuracy"]),
                "mean_ood_brier": safe_metric_mean(ood_df["brier"]),
                "n_ood_models": int(ood_df["test_model"].nunique()),
                "attention_feature_count": int(meta["attention_feature_count"]),
                "activation_feature_count": int(meta["activation_feature_count"]),
                "selected_feature_count": int(meta["selected_feature_count"]),
                "effective_activation_pca_dim": meta.get("effective_activation_pca_dim", pd.NA),
                "alignment_detail": str(meta.get("alignment_detail", "")),
                "chosen_c": optional_float(meta.get("chosen_c", pd.NA)),
                "chosen_max_depth": optional_float(meta.get("chosen_max_depth", pd.NA)),
                "decision_threshold": float(meta["decision_threshold"]),
                "selected_features_path": str(meta["selected_features_path"]),
                "coefficients_path": str(meta.get("coefficients_path", "")),
            }
        )
    if not summary_rows:
        return pd.DataFrame()
    out = pd.DataFrame(summary_rows)
    out["_feature_size_sort"] = pd.to_numeric(out["feature_size"], errors="coerce").fillna(-1).astype(int)
    out = out.sort_values(
        ["scenario_name", "target_name", "_feature_size_sort", "mean_ood_auroc", "mean_ood_pr_auc", "source_val_auroc"],
        ascending=[True, True, True, False, False, False],
    ).drop(columns="_feature_size_sort")
    return out.reset_index(drop=True)


def summarize_confusion_counts(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "scenario_name",
        "scenario_title",
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
            scenario_name,
            scenario_title,
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
                "scenario_name": scenario_name,
                "scenario_title": scenario_title,
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
    return pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()


def family_uses_requested_feature_size(feature_family_group: str) -> bool:
    return feature_family_group in {"activation_only", "attention_plus_activation"}


def build_family_panel_selection(transfer_summary_df: pd.DataFrame, feature_size_grid: Sequence[int]) -> pd.DataFrame:
    family_sort = {family: idx for idx, family in enumerate(FAMILY_PANEL_ORDER)}
    rows: list[dict[str, Any]] = []
    for target_name, target_spec in TARGET_SPECS.items():
        target_title = str(target_spec["title"])
        for requested_feature_size in feature_size_grid:
            for family in FAMILY_PANEL_ORDER:
                subset = transfer_summary_df.loc[
                    transfer_summary_df["scenario_name"].eq(SCENARIO_NAME)
                    & transfer_summary_df["target_name"].eq(target_name)
                    & transfer_summary_df["feature_family_group"].eq(family)
                ].copy()
                if family_uses_requested_feature_size(family):
                    subset = subset.loc[
                        pd.to_numeric(subset["feature_size"], errors="coerce").eq(int(requested_feature_size))
                    ]
                if subset.empty:
                    continue
                subset = subset.sort_values(
                    ["mean_ood_auroc", "mean_ood_pr_auc", "mean_val_auroc", "selected_feature_count"],
                    ascending=[False, False, False, True],
                ).reset_index(drop=True)
                best = subset.iloc[0]
                rows.append(
                    {
                        "scenario_name": SCENARIO_NAME,
                        "scenario_title": SCENARIO_TITLE,
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
                        "alignment_detail": str(best.get("alignment_detail", "")),
                        "mean_val_auroc": float(best["mean_val_auroc"]),
                        "mean_ood_auroc": float(best["mean_ood_auroc"]),
                        "min_ood_auroc": float(best["min_ood_auroc"]),
                        "std_ood_auroc": float(best["std_ood_auroc"]),
                        "mean_ood_average_precision": float(best["mean_ood_average_precision"]),
                        "mean_ood_pr_auc": float(best["mean_ood_pr_auc"]),
                        "mean_ood_brier": float(best["mean_ood_brier"]),
                    }
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["_family_sort"] = out["feature_family_group"].map(family_sort).fillna(len(family_sort)).astype(int)
    out = out.sort_values(
        ["scenario_name", "target_name", "requested_feature_size", "_family_sort", "mean_ood_auroc"],
        ascending=[True, True, True, True, False],
    ).drop(columns="_family_sort")
    return out.reset_index(drop=True)


def build_best_family_models(
    train_model_summary_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel_row in panel_selection_df.itertuples(index=False):
        subset = train_model_summary_df.loc[
            train_model_summary_df["scenario_name"].eq(panel_row.scenario_name)
            & train_model_summary_df["target_name"].eq(panel_row.target_name)
            & train_model_summary_df["feature_space"].eq(panel_row.selected_feature_space)
            & train_model_summary_df["feature_size_label"].eq(panel_row.source_feature_size_label)
        ].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["mean_ood_auroc", "mean_ood_pr_auc", "source_val_auroc"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        best = subset.iloc[0]
        rows.append(
            {
                "scenario_name": panel_row.scenario_name,
                "scenario_title": panel_row.scenario_title,
                "target_name": panel_row.target_name,
                "target_title": panel_row.target_title,
                "requested_feature_size": int(panel_row.requested_feature_size),
                "requested_feature_size_label": panel_row.requested_feature_size_label,
                "feature_family_group": panel_row.feature_family_group,
                "feature_space": str(best["feature_space"]),
                "feature_space_title": str(best["feature_space_title"]),
                "feature_size": best["feature_size"],
                "feature_size_label": str(best["feature_size_label"]),
                "train_model": str(best["train_model"]),
                "source_models": str(best["source_models"]),
                "source_model_count": int(best["source_model_count"]),
                "source_val_auroc": float(best["source_val_auroc"]),
                "mean_ood_auroc": float(best["mean_ood_auroc"]),
                "min_ood_auroc": float(best["min_ood_auroc"]),
                "std_ood_auroc": float(best["std_ood_auroc"]),
                "mean_ood_average_precision": float(best["mean_ood_average_precision"]),
                "mean_ood_pr_auc": float(best["mean_ood_pr_auc"]),
                "mean_ood_balanced_accuracy": float(best["mean_ood_balanced_accuracy"]),
                "mean_ood_brier": float(best["mean_ood_brier"]),
                "attention_feature_count": int(best["attention_feature_count"]),
                "activation_feature_count": int(best["activation_feature_count"]),
                "selected_feature_count": int(best["selected_feature_count"]),
                "chosen_c": optional_float(best["chosen_c"]),
                "chosen_max_depth": optional_float(best["chosen_max_depth"]),
                "decision_threshold": float(best["decision_threshold"]),
                "effective_activation_pca_dim": best.get("effective_activation_pca_dim", pd.NA),
                "alignment_detail": str(best.get("alignment_detail", "")),
                "selected_features_path": str(best["selected_features_path"]),
                "coefficients_path": str(best["coefficients_path"]),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_selected_panel_confusion_summary(
    metrics_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
    *,
    eval_role: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel_row in panel_selection_df.itertuples(index=False):
        subset = metrics_df.loc[
            metrics_df["scenario_name"].eq(panel_row.scenario_name)
            & metrics_df["target_name"].eq(panel_row.target_name)
            & metrics_df["feature_space"].eq(panel_row.selected_feature_space)
            & metrics_df["feature_size_label"].eq(panel_row.source_feature_size_label)
            & metrics_df["eval_role"].eq(eval_role)
        ].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "scenario_name": panel_row.scenario_name,
                "scenario_title": panel_row.scenario_title,
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
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_transfer_matrix_for_selection(
    metrics_df: pd.DataFrame,
    *,
    target_name: str,
    feature_space: str,
    feature_size_label: str,
    metric_column: str,
) -> pd.DataFrame:
    subset = metrics_df.loc[
        metrics_df["scenario_name"].eq(SCENARIO_NAME)
        & metrics_df["target_name"].eq(target_name)
        & metrics_df["feature_space"].eq(feature_space)
        & metrics_df["feature_size_label"].eq(feature_size_label)
    ].copy()
    matrix = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    for row in subset.itertuples(index=False):
        matrix.loc[str(row.train_model), str(row.test_model)] = float(getattr(row, metric_column))
    return matrix


def build_confusion_matrix_df(
    confusion_df: pd.DataFrame,
    *,
    target_name: str,
    requested_feature_size: int,
    feature_family_group: str,
    eval_role: str,
) -> pd.DataFrame | None:
    subset = confusion_df.loc[
        confusion_df["scenario_name"].eq(SCENARIO_NAME)
        & confusion_df["target_name"].eq(target_name)
        & confusion_df["requested_feature_size"].eq(int(requested_feature_size))
        & confusion_df["feature_family_group"].eq(feature_family_group)
        & confusion_df["eval_role"].eq(eval_role)
    ].copy()
    if subset.empty:
        return None
    row = subset.iloc[0]
    negative_label = TARGET_SPECS[target_name]["negative_label"]
    positive_label = TARGET_SPECS[target_name]["positive_label"]
    return pd.DataFrame(
        [
            [int(row["sum_tn"]), int(row["sum_fp"])],
            [int(row["sum_fn"]), int(row["sum_tp"])],
        ],
        index=[f"Actual {negative_label}", f"Actual {positive_label}"],
        columns=[f"Pred {negative_label}", f"Pred {positive_label}"],
    )


def export_selected_family_panel_tables(
    output_root: Path,
    metrics_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
) -> None:
    export_root = output_root / "selected_family_panel_tables"
    export_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for panel_row in panel_selection_df.itertuples(index=False):
        panel_dir = export_root / str(panel_row.scenario_name) / panel_row.target_name / str(panel_row.requested_feature_size_label)
        panel_dir.mkdir(parents=True, exist_ok=True)
        auroc_matrix_df = build_transfer_matrix_for_selection(
            metrics_df,
            target_name=str(panel_row.target_name),
            feature_space=str(panel_row.selected_feature_space),
            feature_size_label=str(panel_row.source_feature_size_label),
            metric_column="auroc",
        )
        auroc_matrix_path = panel_dir / f"{panel_row.feature_family_group}__auroc_matrix.csv"
        auroc_matrix_df.to_csv(auroc_matrix_path, index=True)
        pr_auc_matrix_df = build_transfer_matrix_for_selection(
            metrics_df,
            target_name=str(panel_row.target_name),
            feature_space=str(panel_row.selected_feature_space),
            feature_size_label=str(panel_row.source_feature_size_label),
            metric_column="pr_auc",
        )
        pr_auc_matrix_path = panel_dir / f"{panel_row.feature_family_group}__pr_auc_matrix.csv"
        pr_auc_matrix_df.to_csv(pr_auc_matrix_path, index=True)
        val_confusion_matrix_df = build_confusion_matrix_df(
            confusion_df,
            target_name=str(panel_row.target_name),
            requested_feature_size=int(panel_row.requested_feature_size),
            feature_family_group=str(panel_row.feature_family_group),
            eval_role="val",
        )
        val_confusion_path = panel_dir / f"{panel_row.feature_family_group}__val_confusion_counts.csv"
        if val_confusion_matrix_df is not None:
            val_confusion_matrix_df.to_csv(val_confusion_path, index=True)
        ood_confusion_matrix_df = build_confusion_matrix_df(
            confusion_df,
            target_name=str(panel_row.target_name),
            requested_feature_size=int(panel_row.requested_feature_size),
            feature_family_group=str(panel_row.feature_family_group),
            eval_role="ood",
        )
        ood_confusion_path = panel_dir / f"{panel_row.feature_family_group}__ood_confusion_counts.csv"
        if ood_confusion_matrix_df is not None:
            ood_confusion_matrix_df.to_csv(ood_confusion_path, index=True)
        manifest_rows.append(
            {
                "scenario_name": panel_row.scenario_name,
                "scenario_title": panel_row.scenario_title,
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
                "auroc_matrix_path": str(auroc_matrix_path),
                "pr_auc_matrix_path": str(pr_auc_matrix_path),
                "val_confusion_path": str(val_confusion_path),
                "ood_confusion_path": str(ood_confusion_path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(export_root / "panel_table_manifest.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Main3-style environment OOD predictors for one model, export rebuttal metrics, "
            "and summarize per-target-environment results."
        )
    )
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--model-preset", type=str, default=DEFAULT_MODEL_PRESET)
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--train-env-labels", type=str, default="")
    parser.add_argument("--feature-spaces", type=str, default=",".join(DEFAULT_FEATURE_SPACES))
    parser.add_argument("--feature-sizes", type=str, default=",".join(str(value) for value in DEFAULT_FEATURE_SIZES))
    parser.add_argument("--model-family", type=str, default="xgb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--delta-threshold", type=float, default=0.30)
    parser.add_argument("--min-num-valid", type=int, default=11)
    parser.add_argument("--min-sentence-alpha-words", type=int, default=4)
    parser.add_argument("--exclude-multiline-sentences", action="store_true", default=False)
    parser.add_argument("--root-batch-size", type=int, default=8)
    parser.add_argument("--fallback-attention-top-k", type=int, default=128)
    parser.add_argument("--decision-threshold-mode", type=str, default="train_balanced_accuracy")
    parser.add_argument("--model-selection-objective", type=str, default="mean_ood_auroc_oracle")
    parser.add_argument("--activation-alignment-mode", type=str, choices=("truncate_to_min_hidden_dim", "require_equal_hidden_dim"), default="truncate_to_min_hidden_dim")
    parser.add_argument("--logreg-c", type=float, default=0.1)
    parser.add_argument("--xgb-max-depth", type=int, default=5)
    parser.add_argument("--xgb-n-estimators", type=int, default=200)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-gamma", type=float, default=0.0)
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--xgb-eval-metric", type=str, default="aucpr")
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=30)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--fixed-recall-levels", type=str, default=",".join(str(value) for value in DEFAULT_FIXED_RECALL_LEVELS))
    parser.add_argument("--top-features-to-show", type=int, default=20)
    parser.add_argument("--force-rebuild-reductions", action="store_true", default=False)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = ensure_dir(args.output_root)
    cache_dir = ensure_dir(output_root / "cache")
    model_key = normalize_model_key(args.model_preset)
    model_spec = MODEL_SPECS[model_key]
    selected_scenarios = [normalize_scenario_key(value) for value in parse_csv_list(args.scenarios)]
    selected_scenarios = list(dict.fromkeys(selected_scenarios))
    if not selected_scenarios:
        raise ValueError("At least one scenario must be provided via --scenarios.")
    selected_train_env_labels = parse_csv_list(args.train_env_labels)
    feature_space_names = parse_csv_list(args.feature_spaces)
    feature_sizes = parse_int_csv(args.feature_sizes)
    fixed_recall_levels = parse_float_csv(args.fixed_recall_levels)
    model_family = normalize_model_family(args.model_family)

    selected_feature_spaces: OrderedDict[str, FeatureSpaceSpec] = OrderedDict()
    for feature_space_name in feature_space_names:
        if feature_space_name not in FEATURE_SPACES:
            raise ValueError(f"Unsupported feature space {feature_space_name!r}.")
        selected_feature_spaces[feature_space_name] = FEATURE_SPACES[feature_space_name]

    bundle_specs: dict[str, BundleSpec] = {}
    inventory_rows: list[dict[str, Any]] = []
    for env_name, env_dir in ENV_SPECS.items():
        dataset_dir = dataset_root / env_dir / str(model_spec["dirname"])
        feature_path = resolve_feature_path_optional(dataset_dir)
        activation_path = (dataset_dir / ACTIVATION_FILENAME).resolve()
        if not activation_path.exists():
            activation_path = None
        structural_baseline_path = (dataset_dir / STRUCTURAL_BASELINE_FILENAME).resolve()
        tfidf_cache_dir = (dataset_dir / TFIDF_CACHE_DIRNAME).resolve()
        bundle_spec = BundleSpec(
            env_name=env_name,
            env_dir=env_dir,
            model_key=model_key,
            model_dirname=str(model_spec["dirname"]),
            model_display=str(model_spec["display"]),
            feature_path=feature_path,
            activation_path=activation_path,
            structural_baseline_path=structural_baseline_path,
            tfidf_cache_dir=tfidf_cache_dir,
        )
        bundle_specs[bundle_spec.bundle_key] = bundle_spec
        inventory_rows.append(
            {
                "bundle_key": bundle_spec.bundle_key,
                "env_name": env_name,
                "model_key": model_key,
                "model_display": model_spec["display"],
                "dataset_dir": str(dataset_dir),
                "feature_path": "" if feature_path is None else str(feature_path),
                "activation_path": "" if activation_path is None else str(activation_path),
                "structural_baseline_path": str(structural_baseline_path),
                "tfidf_cache_dir": str(tfidf_cache_dir),
            }
        )
    inventory_df = pd.DataFrame(inventory_rows).sort_values(["env_name"]).reset_index(drop=True)
    inventory_df.to_csv(output_root / "bundle_inventory.csv", index=False)

    uses_attention = any(feature_space.uses_attention for feature_space in selected_feature_spaces.values())
    uses_activation = any(feature_space.activation_variant is not None for feature_space in selected_feature_spaces.values())
    uses_tfidf = any(feature_space.baseline_variant == "tfidf" for feature_space in selected_feature_spaces.values())
    if uses_attention:
        missing_attention = [bundle_key for bundle_key, bundle_spec in bundle_specs.items() if bundle_spec.feature_path is None]
        if missing_attention:
            raise FileNotFoundError(f"Attention feature spaces require readable feature parquet files for: {', '.join(missing_attention)}")
    if uses_activation:
        missing_activation = [bundle_key for bundle_key, bundle_spec in bundle_specs.items() if bundle_spec.activation_path is None]
        if missing_activation:
            raise FileNotFoundError(f"Activation feature spaces require activation HDF5 files for: {', '.join(missing_activation)}")
    if uses_tfidf:
        missing_structural = [bundle_key for bundle_key, bundle_spec in bundle_specs.items() if not bundle_spec.structural_baseline_path.exists()]
        if missing_structural:
            raise FileNotFoundError(f"TF-IDF baselines require structural baseline parquet files for: {', '.join(missing_structural)}")

    config_df = pd.DataFrame(
        [
            {"setting": "dataset_root", "value": str(dataset_root)},
            {"setting": "output_root", "value": str(output_root)},
            {"setting": "model_family", "value": model_family},
            {"setting": "model_key", "value": model_key},
            {"setting": "model_display", "value": str(model_spec["display"])},
            {"setting": "model_dirname", "value": str(model_spec["dirname"])},
            {"setting": "scenarios", "value": ", ".join(selected_scenarios)},
            {"setting": "train_env_labels_filter", "value": ", ".join(selected_train_env_labels)},
            {"setting": "env_order", "value": ", ".join(ENV_ORDER)},
            {"setting": "feature_spaces", "value": ", ".join(selected_feature_spaces.keys())},
            {"setting": "feature_sizes", "value": ", ".join(str(value) for value in feature_sizes)},
            {"setting": "delta_threshold", "value": args.delta_threshold},
            {"setting": "val_size", "value": args.val_size},
            {"setting": "activation_alignment_mode", "value": args.activation_alignment_mode},
            {"setting": "decision_threshold_mode", "value": args.decision_threshold_mode},
            {"setting": "model_selection_objective", "value": args.model_selection_objective},
            {"setting": "xgb_n_estimators", "value": args.xgb_n_estimators},
            {"setting": "xgb_eval_metric", "value": args.xgb_eval_metric},
            {"setting": "xgb_early_stopping_rounds", "value": args.xgb_early_stopping_rounds},
            {"setting": "fixed_recall_levels", "value": ", ".join(str(value) for value in fixed_recall_levels)},
        ]
    )
    config_df.to_csv(output_root / "config.csv", index=False)

    feature_metadata_by_bundle: dict[str, pd.DataFrame] = {}
    structural_metadata_by_bundle: dict[str, pd.DataFrame] = {}
    activation_stores: dict[str, ActivationStore] = {}
    split_cache_by_bundle: dict[str, dict[str, Any]] = {}
    split_summary_rows: list[dict[str, Any]] = []
    hidden_dims: list[int] = []

    for bundle_key, bundle_spec in maybe_tqdm(bundle_specs.items(), desc="Load metadata", total=len(bundle_specs), disable=args.disable_tqdm):
        if bundle_spec.feature_path is None:
            raise FileNotFoundError(f"Missing feature path for {bundle_key}.")
        feature_metadata_df = load_feature_metadata(
            bundle_spec.feature_path,
            env_name=bundle_spec.env_name,
            model_display=bundle_spec.model_display,
            delta_threshold=float(args.delta_threshold),
            min_num_valid=int(args.min_num_valid),
            min_sentence_alpha_words=int(args.min_sentence_alpha_words),
            exclude_multiline_sentences=bool(args.exclude_multiline_sentences),
        )
        split_map = build_example_split_map(feature_metadata_df["example_id"], seed=int(args.seed), val_size=float(args.val_size))
        feature_metadata_df["split"] = feature_metadata_df["example_id"].map(split_map).astype("string")
        feature_metadata_by_bundle[bundle_key] = feature_metadata_df

        if bundle_spec.activation_path is not None:
            activation_metadata_df, hidden_dim = load_activation_metadata(
                bundle_spec.activation_path,
                env_name=bundle_spec.env_name,
                model_display=bundle_spec.model_display,
                delta_threshold=float(args.delta_threshold),
                min_num_valid=int(args.min_num_valid),
                min_sentence_alpha_words=int(args.min_sentence_alpha_words),
                exclude_multiline_sentences=bool(args.exclude_multiline_sentences),
            )
            activation_metadata_df["split"] = activation_metadata_df["example_id"].map(split_map).astype("string")
            if activation_metadata_df["split"].isna().any():
                raise ValueError(f"{bundle_key} activation metadata contains example IDs absent from the feature split map.")
            activation_stores[bundle_key] = ActivationStore(
                bundle_key=bundle_key,
                activation_path=bundle_spec.activation_path,
                hidden_dim=int(hidden_dim),
                metadata_df=activation_metadata_df,
            )
            hidden_dims.append(int(hidden_dim))

        if uses_tfidf:
            structural_metadata_df = load_structural_metadata(
                bundle_spec.structural_baseline_path,
                env_name=bundle_spec.env_name,
                model_display=bundle_spec.model_display,
            )
            structural_metadata_df["split"] = structural_metadata_df["example_id"].map(split_map).astype("string")
            structural_metadata_by_bundle[bundle_key] = structural_metadata_df

        split_summary_rows.append(split_summary_row(feature_metadata_df, bundle_key=bundle_key))

    split_summary_df = pd.DataFrame(split_summary_rows).sort_values("bundle_key").reset_index(drop=True)
    split_summary_df.to_csv(output_root / "split_summary.csv", index=False)

    for bundle_key, feature_metadata_df in feature_metadata_by_bundle.items():
        valid_delta = feature_metadata_df["delta_deception_rate"].notna().to_numpy(dtype=bool, copy=False)
        modeled_mask = feature_metadata_df["passes_commitment_pair_filters"].to_numpy(dtype=bool, copy=False)
        train_mask = feature_metadata_df["split"].eq("train").to_numpy(dtype=bool, copy=False) & valid_delta & modeled_mask
        val_mask = feature_metadata_df["split"].eq("val").to_numpy(dtype=bool, copy=False) & valid_delta & modeled_mask
        bundle: dict[str, Any] = {
            "train_mask": train_mask,
            "val_mask": val_mask,
            "train_row_idx": feature_metadata_df.loc[train_mask, "row_idx"].to_numpy(dtype=np.int64, copy=False),
            "val_row_idx": feature_metadata_df.loc[val_mask, "row_idx"].to_numpy(dtype=np.int64, copy=False),
        }
        for target_name in TARGET_SPECS:
            bundle[f"y_train__{target_name}"] = feature_metadata_df.loc[train_mask, f"label__{target_name}"].to_numpy(dtype=np.int8, copy=False)
            bundle[f"y_val__{target_name}"] = feature_metadata_df.loc[val_mask, f"label__{target_name}"].to_numpy(dtype=np.int8, copy=False)
        if uses_tfidf:
            structural_metadata_df = structural_metadata_by_bundle[bundle_key]
            feature_keys_df = feature_metadata_df.loc[:, ["example_id", "sentence_idx"]].copy()
            feature_keys_df["feature_row_idx"] = feature_metadata_df["row_idx"].to_numpy(dtype=np.int64, copy=False)
            structural_keys_df = structural_metadata_df.loc[:, ["example_id", "sentence_idx"]].copy()
            structural_keys_df["structural_row_idx"] = structural_metadata_df["row_idx"].to_numpy(dtype=np.int64, copy=False)
            aligned_key_df = feature_keys_df.merge(
                structural_keys_df,
                on=["example_id", "sentence_idx"],
                how="left",
                validate="one_to_one",
            ).sort_values("feature_row_idx", kind="mergesort").reset_index(drop=True)
            structural_row_idx = aligned_key_df["structural_row_idx"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
            bundle["train_structural_row_idx"] = structural_row_idx[train_mask]
            bundle["val_structural_row_idx"] = structural_row_idx[val_mask]
        split_cache_by_bundle[bundle_key] = bundle

    feature_space_catalog_df = pd.DataFrame(
        [
            {
                "feature_space": feature_space.name,
                "feature_space_title": feature_space.title,
                "feature_family_group": feature_space.family_title,
                "uses_attention": feature_space.uses_attention,
                "attention_subset_key": feature_space.attention_subset_key or "",
                "attention_subset_title": (
                    ATTENTION_SUBSETS[str(feature_space.attention_subset_key)].title
                    if feature_space.attention_subset_key is not None
                    else ""
                ),
                "activation_variant": feature_space.activation_variant or "",
                "activation_use_pca": feature_space.activation_use_pca,
                "baseline_variant": feature_space.baseline_variant or "",
                "baseline_text_field": feature_space.baseline_text_field or "",
            }
            for feature_space in selected_feature_spaces.values()
        ]
    )
    feature_space_catalog_df.to_csv(output_root / "feature_space_catalog.csv", index=False)

    activation_common_hidden_dim = min(hidden_dims) if hidden_dims else 0
    if uses_activation and args.activation_alignment_mode == "require_equal_hidden_dim" and len(set(hidden_dims)) > 1:
        raise ValueError(
            "Activation alignment mode 'require_equal_hidden_dim' was requested, but hidden dims differ across models: "
            + ", ".join(str(value) for value in sorted(set(hidden_dims)))
        )

    attention_lookup_df = pd.DataFrame()
    attention_feature_names: list[str] = []
    attention_frames_by_bundle: dict[str, pd.DataFrame] = {}
    if uses_attention:
        layer_root_map, _ = build_common_layer_roots(
            OrderedDict(
                (bundle_key, bundle_spec.feature_path)
                for bundle_key, bundle_spec in bundle_specs.items()
                if bundle_spec.feature_path is not None
            )
        )
        attention_layer_roots = OrderedDict(
            (root_name, columns)
            for root_name, columns in layer_root_map.items()
            if ATTN_ROOT_RE.fullmatch(root_name) is not None
        )
        if not attention_layer_roots:
            raise ValueError("No shared attention layer roots found across the selected bundles.")
        attention_lookup_df = build_attention_reduction_lookup(attention_layer_roots)
        attention_lookup_df.to_csv(output_root / "attention_reduction_lookup.csv", index=False)
        expected_reduced_columns = set(attention_lookup_df["feature"].astype(str).tolist())
        for bundle_key, bundle_spec in maybe_tqdm(bundle_specs.items(), desc="Build attention frames", total=len(bundle_specs), disable=args.disable_tqdm):
            if bundle_spec.feature_path is None:
                continue
            cache_path = cache_dir / f"{slugify(bundle_key)}__attention_reduced.parquet"
            rebuild = bool(args.force_rebuild_reductions)
            if cache_path.exists() and not rebuild:
                cached_df = pd.read_parquet(cache_path)
                missing_columns = sorted(expected_reduced_columns - set(cached_df.columns.astype(str).tolist()))
                if missing_columns:
                    rebuild = True
            if not cache_path.exists() or rebuild:
                reduced_df = build_attention_reduced_bundle_frame(
                    bundle_key=bundle_key,
                    feature_path=bundle_spec.feature_path,
                    metadata_df=feature_metadata_by_bundle[bundle_key],
                    attention_layer_roots=attention_layer_roots,
                    root_batch_size=int(args.root_batch_size),
                    disable_tqdm=bool(args.disable_tqdm),
                )
                reduced_df.to_parquet(cache_path, index=False)
            attention_frames_by_bundle[bundle_key] = pd.read_parquet(cache_path)
        attention_feature_names = feature_columns_for_attention_frame(next(iter(attention_frames_by_bundle.values())))
        attention_lookup_df = attention_lookup_df.loc[
            attention_lookup_df["feature"].isin(attention_feature_names)
        ].copy().reset_index(drop=True)

    tfidf_bundle_by_space: dict[str, BaselineMatrixBundle] = {}
    if uses_tfidf:
        for feature_space_name, feature_space in selected_feature_spaces.items():
            if feature_space.baseline_variant != "tfidf":
                continue
            tfidf_bundle_by_space[feature_space_name] = build_tfidf_matrix_bundle(
                text_field=str(feature_space.baseline_text_field),
                space_name=feature_space_name,
                bundle_specs=bundle_specs,
                structural_metadata_by_bundle=structural_metadata_by_bundle,
                split_cache_by_bundle=split_cache_by_bundle,
            )

    used_attention_subset_keys = [
        subset_key
        for subset_key in ATTENTION_SUBSETS
        if any(
            feature_space.uses_attention and feature_space.attention_subset_key == subset_key
            for feature_space in selected_feature_spaces.values()
        )
    ]
    attention_rankings_by_target_subset: dict[tuple[str, str], pd.DataFrame] = {}
    attention_pools_by_target_subset: dict[tuple[str, str], tuple[list[str], str]] = {}
    attention_matrix_cache_by_target_subset: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    if uses_attention:
        rankings_dir = ensure_dir(output_root / "rankings")
        env_frames = OrderedDict((env_name, attention_frames_by_bundle[env_name]) for env_name in ENV_ORDER)
        for target_name in TARGET_SPECS:
            target_dir = ensure_dir(rankings_dir / target_name)
            for subset_key in used_attention_subset_keys:
                subset_spec = ATTENTION_SUBSETS[subset_key]
                subset_lookup_df = attention_lookup_df.loc[
                    attention_lookup_df["metric_name"].isin(list(subset_spec.metric_names))
                ].copy()
                if subset_spec.transition_mode == "base_only":
                    subset_lookup_df = subset_lookup_df.loc[subset_lookup_df["is_transition"].eq(False)].copy()
                elif subset_spec.transition_mode == "transition_only":
                    subset_lookup_df = subset_lookup_df.loc[subset_lookup_df["is_transition"].eq(True)].copy()
                subset_feature_names = [
                    feature_name
                    for feature_name in attention_feature_names
                    if feature_name in set(subset_lookup_df["feature"].astype(str).tolist())
                ]
                if not subset_feature_names:
                    raise ValueError(f"No attention features matched subset {subset_key!r} for {target_name}.")
                ranking_df = build_consistency_ranking(
                    env_frames,
                    feature_names=subset_feature_names,
                    feature_lookup_df=subset_lookup_df,
                    target_name=target_name,
                )
                pool_features, selection_mode = select_attention_pool(
                    ranking_df,
                    fallback_top_k=int(args.fallback_attention_top_k),
                )
                ranking_df["selected_for_pool"] = ranking_df["feature"].astype(str).isin(set(pool_features))
                ranking_df["selection_mode"] = selection_mode
                ranking_df["attention_subset_key"] = subset_key
                ranking_df["attention_subset_title"] = subset_spec.title
                attention_rankings_by_target_subset[(target_name, subset_key)] = ranking_df
                attention_pools_by_target_subset[(target_name, subset_key)] = (pool_features, selection_mode)
                ranking_df.to_csv(target_dir / f"{subset_key}__attention_ranking.csv", index=False)
                cache: dict[str, dict[str, np.ndarray]] = {}
                for bundle_key, frame_df in attention_frames_by_bundle.items():
                    split_bundle = split_cache_by_bundle[bundle_key]
                    pooled = (
                        frame_df.loc[:, pool_features]
                        .apply(pd.to_numeric, errors="coerce")
                        .replace([np.inf, -np.inf], np.nan)
                        .to_numpy(dtype=np.float32, copy=False)
                    )
                    cache[bundle_key] = {
                        "train": np.asarray(pooled[split_bundle["train_mask"]], dtype=np.float32),
                        "val": np.asarray(pooled[split_bundle["val_mask"]], dtype=np.float32),
                    }
                attention_matrix_cache_by_target_subset[(target_name, subset_key)] = cache

    experiment_run_specs, train_axis_labels_by_scenario = build_experiment_run_specs(selected_scenarios)
    if selected_train_env_labels:
        selected_train_env_labels_set = set(selected_train_env_labels)
        experiment_run_specs = [
            run_spec
            for run_spec in experiment_run_specs
            if run_spec.train_env_label in selected_train_env_labels_set
        ]
        train_axis_labels_by_scenario = {
            scenario_name: [
                train_env_label
                for train_env_label in labels
                if train_env_label in selected_train_env_labels_set
            ]
            for scenario_name, labels in train_axis_labels_by_scenario.items()
        }
    if not experiment_run_specs:
        raise ValueError(
            "No experiment run specs remain after filtering. "
            f"scenarios={selected_scenarios}, train_env_labels={selected_train_env_labels}"
        )

    all_transfer_rows: list[dict[str, Any]] = []
    all_model_selection_rows: list[dict[str, Any]] = []
    all_calibration_frames: list[pd.DataFrame] = []
    all_fpr_frames: list[pd.DataFrame] = []
    all_coefficient_frames: list[pd.DataFrame] = []
    activation_bundle_cache: dict[tuple[str, str, str, int | None], ActivationMatrixBundle] = {}

    for target_name in maybe_tqdm(list(TARGET_SPECS.keys()), desc="Targets", total=len(TARGET_SPECS), disable=args.disable_tqdm):
        target_title = str(TARGET_SPECS[target_name]["title"])
        for feature_space_name, feature_space in maybe_tqdm(
            selected_feature_spaces.items(),
            desc=f"Spaces:{target_name}",
            total=len(selected_feature_spaces),
            disable=args.disable_tqdm,
        ):
            feature_space_attention_subset_key = str(feature_space.attention_subset_key or "")
            feature_space_attention_subset_title = (
                ATTENTION_SUBSETS[str(feature_space.attention_subset_key)].title
                if feature_space.attention_subset_key is not None
                else ""
            )
            ranking_meta_df = pd.DataFrame()
            attention_pool_features: list[str] = []
            attention_selection_mode = ""
            if feature_space.uses_attention:
                assert feature_space.attention_subset_key is not None
                ranking_meta_df = attention_rankings_by_target_subset[(target_name, str(feature_space.attention_subset_key))].copy()
                attention_pool_features, attention_selection_mode = attention_pools_by_target_subset[
                    (target_name, str(feature_space.attention_subset_key))
                ]
            baseline_bundle = tfidf_bundle_by_space.get(feature_space_name)

            for run_spec in maybe_tqdm(
                experiment_run_specs,
                desc=f"Runs:{target_name}:{feature_space_name}",
                total=len(experiment_run_specs),
                disable=args.disable_tqdm,
            ):
                for feature_size in feature_size_options_for_space(feature_space, feature_sizes):
                    activation_bundle: ActivationMatrixBundle | None = None
                    effective_activation_pca_dim: int | None = None
                    attention_feature_count = 0
                    activation_feature_count = 0
                    attention_dim = 0
                    activation_dim = 0
                    current_feature_names: list[str] = []
                    alignment_detail_parts: list[str] = []

                    if feature_space.uses_attention:
                        attention_dim = int(len(attention_pool_features))
                        attention_feature_count = int(len(attention_pool_features))
                        current_feature_names.extend(attention_pool_features)
                        alignment_detail_parts.append(f"attention_pool={attention_selection_mode}")

                    if feature_space.activation_variant is not None:
                        activation_cache_key = (
                            run_spec.scenario_name,
                            run_spec.train_env_label,
                            feature_space_name,
                            feature_size if feature_space.activation_use_pca else None,
                        )
                        activation_bundle = activation_bundle_cache.get(activation_cache_key)
                        if activation_bundle is None:
                            if activation_common_hidden_dim < 1:
                                raise ValueError("No activation hidden dimensions were discovered.")
                            raw_train_parts: list[np.ndarray] = []
                            for env_name in run_spec.source_envs:
                                raw_train_parts.append(
                                    activation_stores[env_name].load_matrix(
                                        split_cache_by_bundle[env_name]["train_row_idx"],
                                        variant=str(feature_space.activation_variant),
                                        hidden_dim=int(activation_common_hidden_dim),
                                    )
                                )
                            raw_train = np.concatenate(raw_train_parts, axis=0)
                            imputer = SimpleImputer(strategy="median")
                            scaler = StandardScaler()
                            train_scaled = scaler.fit_transform(imputer.fit_transform(raw_train))
                            pca: PCA | None = None
                            effective_dim: int | None = None
                            if feature_space.activation_use_pca:
                                requested_dim = int(feature_size if feature_size is not None else activation_common_hidden_dim)
                                effective_dim = int(
                                    min(requested_dim, activation_common_hidden_dim, max(1, train_scaled.shape[0] - 1))
                                )
                                pca = PCA(
                                    n_components=int(effective_dim),
                                    random_state=int(args.seed),
                                    svd_solver="randomized",
                                )
                                pca.fit(train_scaled)
                            matrices_by_bundle: dict[str, dict[str, np.ndarray]] = {}
                            for env_name in ENV_ORDER:
                                matrices_by_bundle[env_name] = {}
                                for split_name in ("train", "val"):
                                    row_idx = split_cache_by_bundle[env_name][f"{split_name}_row_idx"]
                                    raw_matrix = activation_stores[env_name].load_matrix(
                                        row_idx,
                                        variant=str(feature_space.activation_variant),
                                        hidden_dim=int(activation_common_hidden_dim),
                                    )
                                    transformed = scaler.transform(imputer.transform(raw_matrix))
                                    if pca is not None:
                                        transformed = pca.transform(transformed)
                                    matrices_by_bundle[env_name][split_name] = np.asarray(transformed, dtype=np.float32)
                            feature_lookup_df = make_activation_lookup(
                                space_name=feature_space_name,
                                variant=str(feature_space.activation_variant),
                                use_pca=bool(feature_space.activation_use_pca),
                                hidden_dim=int(activation_common_hidden_dim),
                                pca_dim=int(effective_dim if effective_dim is not None else activation_common_hidden_dim),
                            )
                            activation_bundle = ActivationMatrixBundle(
                                matrices_by_bundle=matrices_by_bundle,
                                feature_names=feature_lookup_df["feature"].astype(str).tolist(),
                                feature_lookup_df=feature_lookup_df,
                                effective_pca_dim=effective_dim,
                                common_hidden_dim=int(activation_common_hidden_dim),
                                alignment_mode=str(args.activation_alignment_mode),
                            )
                            activation_bundle_cache[activation_cache_key] = activation_bundle
                        if feature_space.activation_use_pca:
                            activation_dim = int(
                                len(activation_bundle.feature_names)
                                if feature_size is None
                                else min(len(activation_bundle.feature_names), int(feature_size))
                            )
                        else:
                            activation_dim = int(len(activation_bundle.feature_names))
                        activation_feature_count = int(activation_dim)
                        effective_activation_pca_dim = activation_bundle.effective_pca_dim
                        current_feature_names.extend(activation_bundle.feature_names[:activation_dim])
                        alignment_detail_parts.append(
                            f"activation={activation_bundle.alignment_mode}:hidden={activation_bundle.common_hidden_dim}"
                        )

                    if baseline_bundle is not None:
                        current_feature_names.extend(baseline_bundle.feature_names)
                        alignment_detail_parts.append(f"tfidf={baseline_bundle.alignment_mode}")

                    if not current_feature_names:
                        raise ValueError(
                            f"No features constructed for {target_name} / {feature_space_name} / {run_spec.train_env_label} / {feature_size}."
                        )

                    env_pool_cache: dict[str, dict[str, Any]] = {}
                    for env_name in ENV_ORDER:
                        y_train_env = np.asarray(split_cache_by_bundle[env_name][f"y_train__{target_name}"], dtype=np.int8)
                        y_val_env = np.asarray(split_cache_by_bundle[env_name][f"y_val__{target_name}"], dtype=np.int8)
                        if np.unique(y_train_env).size < 2:
                            raise ValueError(f"{env_name} train split does not contain both classes for {target_name}.")
                        if np.unique(y_val_env).size < 2:
                            raise ValueError(f"{env_name} val split does not contain both classes for {target_name}.")
                        env_entry: dict[str, Any] = {
                            f"y_train__{target_name}": y_train_env,
                            f"y_val__{target_name}": y_val_env,
                        }
                        if feature_space.uses_attention:
                            env_entry["train_attention_pool"] = attention_matrix_cache_by_target_subset[
                                (target_name, str(feature_space.attention_subset_key))
                            ][env_name]["train"]
                            env_entry["val_attention_pool"] = attention_matrix_cache_by_target_subset[
                                (target_name, str(feature_space.attention_subset_key))
                            ][env_name]["val"]
                        if activation_bundle is not None:
                            env_entry["train_activation_pool"] = activation_bundle.matrices_by_bundle[env_name]["train"]
                            env_entry["val_activation_pool"] = activation_bundle.matrices_by_bundle[env_name]["val"]
                        if baseline_bundle is not None:
                            env_entry["train_baseline_pool"] = baseline_bundle.matrices_by_bundle[env_name]["train"]
                            env_entry["val_baseline_pool"] = baseline_bundle.matrices_by_bundle[env_name]["val"]
                        env_pool_cache[env_name] = env_entry

                    x_train, y_train = concatenate_source_split_matrices_env(
                        env_pool_cache,
                        source_envs=run_spec.source_envs,
                        split_name="train",
                        feature_space=feature_space,
                        attention_dim=attention_dim,
                        activation_dim=activation_dim,
                        target_name=target_name,
                    )
                    x_val, y_val = concatenate_source_split_matrices_env(
                        env_pool_cache,
                        source_envs=run_spec.source_envs,
                        split_name="val",
                        feature_space=feature_space,
                        attention_dim=attention_dim,
                        activation_dim=activation_dim,
                        target_name=target_name,
                    )
                    fitted_models, candidate_df = fit_candidate_classifiers(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        model_family=model_family,
                        seed=int(args.seed),
                        logreg_c=float(args.logreg_c),
                        xgb_max_depth=int(args.xgb_max_depth),
                        xgb_n_estimators=int(args.xgb_n_estimators),
                        xgb_learning_rate=float(args.xgb_learning_rate),
                        xgb_subsample=float(args.xgb_subsample),
                        xgb_colsample_bytree=float(args.xgb_colsample_bytree),
                        xgb_reg_lambda=float(args.xgb_reg_lambda),
                        xgb_min_child_weight=float(args.xgb_min_child_weight),
                        xgb_gamma=float(args.xgb_gamma),
                        xgb_n_jobs=int(args.xgb_n_jobs),
                        xgb_eval_metric=str(args.xgb_eval_metric),
                        xgb_early_stopping_rounds=int(args.xgb_early_stopping_rounds),
                        decision_threshold_mode=str(args.decision_threshold_mode),
                    )
                    size_label = selected_feature_size_label(feature_space, feature_size)
                    selection_dir = ensure_dir(
                        output_root / "model_selection" / run_spec.scenario_name / target_name / feature_space_name / size_label
                    )
                    oracle_rows: list[dict[str, Any]] = []
                    best_model: FittedBinaryModel | None = None
                    best_key: tuple[float, ...] | None = None
                    best_oracle_metrics: dict[str, float] | None = None
                    for fitted_model in fitted_models:
                        oracle_aurocs: list[float] = []
                        oracle_pr_aucs: list[float] = []
                        for test_env in run_spec.ood_envs:
                            eval_cache = env_pool_cache[test_env]
                            x_eval = assemble_env_split_matrix(
                                eval_cache,
                                split_name="val",
                                feature_space=feature_space,
                                attention_dim=attention_dim,
                                activation_dim=activation_dim,
                            )
                            eval_scores = fitted_model.estimator.predict_proba(x_eval)[:, 1].astype(np.float32)
                            metrics_extra = safe_binary_metrics(eval_cache[f"y_val__{target_name}"], eval_scores)
                            oracle_aurocs.append(float(metrics_extra["auroc"]))
                            oracle_pr_aucs.append(float(metrics_extra["pr_auc"]))
                        oracle_metrics = {
                            "oracle_mean_ood_auroc": finite_mean(oracle_aurocs),
                            "oracle_min_ood_auroc": finite_min(oracle_aurocs),
                            "oracle_std_ood_auroc": safe_metric_std(pd.Series(oracle_aurocs, dtype=float)),
                            "oracle_mean_ood_pr_auc": finite_mean(oracle_pr_aucs),
                        }
                        oracle_rows.append({"candidate_key": fitted_model.candidate_key, **oracle_metrics})
                        candidate_key = build_model_selection_key(
                            objective=str(args.model_selection_objective),
                            oracle_mean_ood_auroc=oracle_metrics["oracle_mean_ood_auroc"],
                            oracle_mean_ood_pr_auc=oracle_metrics["oracle_mean_ood_pr_auc"],
                            source_val_auroc=float(fitted_model.validation_metrics["auroc"]),
                            source_val_pr_auc=float(fitted_model.validation_metrics["pr_auc"]),
                            source_val_balanced_accuracy=float(fitted_model.validation_metrics["balanced_accuracy"]),
                            feature_count=len(current_feature_names),
                            candidate_complexity=float(fitted_model.candidate_complexity),
                        )
                        if best_key is None or candidate_key > best_key:
                            best_key = candidate_key
                            best_model = fitted_model
                            best_oracle_metrics = oracle_metrics

                    if best_model is None or best_oracle_metrics is None:
                        raise RuntimeError(
                            f"Failed to select a model for {target_name} / {feature_space_name} / {run_spec.train_env_label}."
                        )

                    candidate_path = selection_dir / f"{slugify(run_spec.train_env_label)}__candidates.csv"
                    candidate_df = candidate_df.merge(
                        pd.DataFrame(oracle_rows),
                        on="candidate_key",
                        how="left",
                        validate="one_to_one",
                    )
                    candidate_df["scenario_name"] = run_spec.scenario_name
                    candidate_df["scenario_title"] = run_spec.scenario_title
                    candidate_df["target_name"] = target_name
                    candidate_df["feature_space"] = feature_space_name
                    candidate_df["feature_space_title"] = feature_space.title
                    candidate_df["feature_family_group"] = feature_space.family_title
                    candidate_df["feature_space_attention_subset_key"] = feature_space_attention_subset_key
                    candidate_df["feature_space_attention_subset_title"] = feature_space_attention_subset_title
                    candidate_df["train_env"] = run_spec.train_env_label
                    candidate_df["source_envs"] = run_spec.source_envs_label
                    candidate_df["source_env_count"] = int(len(run_spec.source_envs))
                    candidate_df["heldout_env"] = run_spec.heldout_env if run_spec.heldout_env is not None else pd.NA
                    candidate_df["ood_env_count"] = int(len(run_spec.ood_envs))
                    candidate_df["feature_size"] = pd.NA if feature_size is None else int(feature_size)
                    candidate_df["feature_size_label"] = size_label
                    candidate_df["attention_feature_count"] = int(attention_feature_count)
                    candidate_df["activation_feature_count"] = int(activation_feature_count)
                    candidate_df["selected_feature_count"] = int(len(current_feature_names))
                    candidate_df["alignment_detail"] = "; ".join(alignment_detail_parts)
                    candidate_df["effective_activation_pca_dim"] = pd.NA if effective_activation_pca_dim is None else int(effective_activation_pca_dim)
                    candidate_df.to_csv(candidate_path, index=False)

                    selected_path = selection_dir / f"{slugify(run_spec.train_env_label)}__selected_features.csv"
                    coefficients_path = selection_dir / f"{slugify(run_spec.train_env_label)}__coefficients.csv"
                    model_artifact_path = selection_dir / f"{slugify(run_spec.train_env_label)}__model.pkl"
                    selection_summary_path = selection_dir / f"{slugify(run_spec.train_env_label)}__selection_summary.csv"
                    transfer_metrics_path = selection_dir / f"{slugify(run_spec.train_env_label)}__transfer_metrics.csv"
                    selected_df = pd.DataFrame(
                        {
                            "feature": current_feature_names,
                            "selected_rank": np.arange(1, len(current_feature_names) + 1, dtype=int),
                        }
                    )
                    if not ranking_meta_df.empty:
                        selected_df = selected_df.merge(
                            ranking_meta_df.drop_duplicates(subset=["feature"], keep="first"),
                            on="feature",
                            how="left",
                        )
                    selected_df["scenario_name"] = run_spec.scenario_name
                    selected_df["scenario_title"] = run_spec.scenario_title
                    selected_df["target_name"] = target_name
                    selected_df["feature_space"] = feature_space_name
                    selected_df["feature_space_title"] = feature_space.title
                    selected_df["feature_family_group"] = feature_space.family_title
                    selected_df["feature_space_attention_subset_key"] = feature_space_attention_subset_key
                    selected_df["feature_space_attention_subset_title"] = feature_space_attention_subset_title
                    selected_df["train_env"] = run_spec.train_env_label
                    selected_df["source_envs"] = run_spec.source_envs_label
                    selected_df["source_env_count"] = int(len(run_spec.source_envs))
                    selected_df["heldout_env"] = run_spec.heldout_env if run_spec.heldout_env is not None else pd.NA
                    selected_df["feature_size"] = pd.NA if feature_size is None else int(feature_size)
                    selected_df["feature_size_label"] = size_label
                    selected_df["attention_feature_count"] = int(attention_feature_count)
                    selected_df["activation_feature_count"] = int(activation_feature_count)
                    selected_df["selected_feature_count"] = int(len(current_feature_names))
                    selected_df.to_csv(selected_path, index=False)

                    coefficient_df = extract_feature_weights(
                        best_model,
                        feature_names=current_feature_names,
                        target_name=target_name,
                        feature_space=feature_space_name,
                        train_model=run_spec.train_env_label,
                    ).rename(columns={"train_model": "train_env"})
                    coefficient_df["scenario_name"] = run_spec.scenario_name
                    coefficient_df["scenario_title"] = run_spec.scenario_title
                    coefficient_df["feature_space_title"] = feature_space.title
                    coefficient_df["feature_family_group"] = feature_space.family_title
                    coefficient_df["feature_space_attention_subset_key"] = feature_space_attention_subset_key
                    coefficient_df["feature_space_attention_subset_title"] = feature_space_attention_subset_title
                    coefficient_df["source_envs"] = run_spec.source_envs_label
                    coefficient_df["source_env_count"] = int(len(run_spec.source_envs))
                    coefficient_df["heldout_env"] = run_spec.heldout_env if run_spec.heldout_env is not None else pd.NA
                    coefficient_df["feature_size"] = pd.NA if feature_size is None else int(feature_size)
                    coefficient_df["feature_size_label"] = size_label
                    coefficient_df["attention_feature_count"] = int(attention_feature_count)
                    coefficient_df["activation_feature_count"] = int(activation_feature_count)
                    coefficient_df["selected_feature_count"] = int(len(current_feature_names))
                    coefficient_df["alignment_detail"] = "; ".join(alignment_detail_parts)
                    coefficient_df.to_csv(coefficients_path, index=False)
                    all_coefficient_frames.append(coefficient_df)

                    with model_artifact_path.open("wb") as f:
                        pickle.dump(
                            {
                                "estimator": best_model.estimator,
                                "decision_threshold": float(best_model.decision_threshold),
                                "candidate_key": best_model.candidate_key,
                                "candidate_label": best_model.candidate_label,
                                "candidate_params": dict(best_model.candidate_params),
                                "feature_names": list(current_feature_names),
                                "target_name": target_name,
                                "feature_space": feature_space_name,
                                "feature_size_label": size_label,
                                "train_env": run_spec.train_env_label,
                                "source_envs": run_spec.source_envs,
                                "heldout_env": run_spec.heldout_env,
                            },
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )

                    selection_summary_row = {
                        "scenario_name": run_spec.scenario_name,
                        "scenario_title": run_spec.scenario_title,
                        "target_name": target_name,
                        "target_title": target_title,
                        "feature_space": feature_space_name,
                        "feature_space_title": feature_space.title,
                        "feature_family_group": feature_space.family_title,
                        "feature_space_attention_subset_key": feature_space_attention_subset_key,
                        "feature_space_attention_subset_title": feature_space_attention_subset_title,
                        "train_env": run_spec.train_env_label,
                        "source_envs": run_spec.source_envs_label,
                        "source_env_count": int(len(run_spec.source_envs)),
                        "heldout_env": run_spec.heldout_env if run_spec.heldout_env is not None else pd.NA,
                        "ood_env_count": int(len(run_spec.ood_envs)),
                        "feature_size": pd.NA if feature_size is None else int(feature_size),
                        "feature_size_label": size_label,
                        "attention_feature_count": int(attention_feature_count),
                        "activation_feature_count": int(activation_feature_count),
                        "selected_feature_count": int(len(current_feature_names)),
                        "alignment_detail": "; ".join(alignment_detail_parts),
                        "candidate_key": best_model.candidate_key,
                        "candidate_label": best_model.candidate_label,
                        "candidate_complexity": float(best_model.candidate_complexity),
                        "candidate_params_json": serialize_candidate_params(best_model.candidate_params),
                        "chosen_c": pd.NA if best_model.chosen_c is None else float(best_model.chosen_c),
                        "chosen_max_depth": pd.NA if best_model.candidate_max_depth is None else int(best_model.candidate_max_depth),
                        "decision_threshold": float(best_model.decision_threshold),
                        "xgb_best_iteration": pd.NA if best_model.xgb_best_iteration is None else int(best_model.xgb_best_iteration),
                        "xgb_best_score": pd.NA if best_model.xgb_best_score is None else float(best_model.xgb_best_score),
                        "effective_activation_pca_dim": pd.NA if effective_activation_pca_dim is None else int(effective_activation_pca_dim),
                        "model_selection_objective": args.model_selection_objective,
                        "oracle_mean_ood_auroc_selected": float(best_oracle_metrics["oracle_mean_ood_auroc"]),
                        "oracle_min_ood_auroc_selected": float(best_oracle_metrics["oracle_min_ood_auroc"]),
                        "oracle_std_ood_auroc_selected": float(best_oracle_metrics["oracle_std_ood_auroc"]),
                        "oracle_mean_ood_pr_auc_selected": float(best_oracle_metrics["oracle_mean_ood_pr_auc"]),
                        "model_artifact_path": str(model_artifact_path),
                        "selected_features_path": str(selected_path),
                        "coefficients_path": str(coefficients_path),
                        "selection_summary_path": str(selection_summary_path),
                        "transfer_metrics_path": str(transfer_metrics_path),
                        **best_model.validation_metrics,
                    }
                    pd.DataFrame([selection_summary_row]).to_csv(selection_summary_path, index=False)
                    all_model_selection_rows.append(selection_summary_row)

                    current_transfer_rows: list[dict[str, Any]] = []
                    current_calibration_frames: list[pd.DataFrame] = []
                    current_fpr_frames: list[pd.DataFrame] = []
                    for eval_role, eval_envs in (("val", run_spec.source_envs), ("ood", run_spec.ood_envs)):
                        for test_env in eval_envs:
                            eval_cache = env_pool_cache[test_env]
                            x_eval = assemble_env_split_matrix(
                                eval_cache,
                                split_name="val",
                                feature_space=feature_space,
                                attention_dim=attention_dim,
                                activation_dim=activation_dim,
                            )
                            eval_scores = best_model.estimator.predict_proba(x_eval)[:, 1].astype(np.float32)
                            eval_df = concatenate_model_metadata(
                                [test_env],
                                feature_metadata_by_bundle,
                                split_cache_by_bundle,
                                split_name="val",
                                target_name=target_name,
                            )
                            metric_row, calibration_df, fpr_df = evaluate_env_predictions(
                                run_spec=run_spec,
                                feature_space_name=feature_space_name,
                                feature_space_title=feature_space.title,
                                feature_family_group=feature_space.family_title,
                                feature_space_attention_subset_key=feature_space_attention_subset_key,
                                feature_space_attention_subset_title=feature_space_attention_subset_title,
                                target_name=target_name,
                                target_title=target_title,
                                test_env=test_env,
                                eval_role=eval_role,
                                feature_size=feature_size,
                                feature_size_label=size_label,
                                attention_feature_count=int(attention_feature_count),
                                activation_feature_count=int(activation_feature_count),
                                selected_feature_count=int(len(current_feature_names)),
                                effective_activation_pca_dim=effective_activation_pca_dim,
                                alignment_detail="; ".join(alignment_detail_parts),
                                eval_df=eval_df,
                                y_score=eval_scores,
                                decision_threshold=float(best_model.decision_threshold),
                                calibration_bins=int(args.calibration_bins),
                                fixed_recall_levels=fixed_recall_levels,
                            )
                            metric_row.update(
                                {
                                    "candidate_key": best_model.candidate_key,
                                    "candidate_label": best_model.candidate_label,
                                    "candidate_complexity": float(best_model.candidate_complexity),
                                    "candidate_params_json": serialize_candidate_params(best_model.candidate_params),
                                    "chosen_c": pd.NA if best_model.chosen_c is None else float(best_model.chosen_c),
                                    "chosen_max_depth": pd.NA if best_model.candidate_max_depth is None else int(best_model.candidate_max_depth),
                                    "decision_threshold": float(best_model.decision_threshold),
                                    "xgb_best_iteration": pd.NA if best_model.xgb_best_iteration is None else int(best_model.xgb_best_iteration),
                                    "xgb_best_score": pd.NA if best_model.xgb_best_score is None else float(best_model.xgb_best_score),
                                    "model_selection_objective": args.model_selection_objective,
                                    "model_artifact_path": str(model_artifact_path),
                                    "selected_features_path": str(selected_path),
                                    "coefficients_path": str(coefficients_path),
                                    "transfer_metrics_path": str(transfer_metrics_path),
                                }
                            )
                            current_transfer_rows.append(metric_row)
                            if not calibration_df.empty:
                                current_calibration_frames.append(calibration_df)
                            if not fpr_df.empty:
                                current_fpr_frames.append(fpr_df)
                    pd.DataFrame(current_transfer_rows).to_csv(transfer_metrics_path, index=False)
                    all_transfer_rows.extend(current_transfer_rows)
                    all_calibration_frames.extend(current_calibration_frames)
                    all_fpr_frames.extend(current_fpr_frames)

    all_transfer_metrics_df = pd.DataFrame(all_transfer_rows)
    all_model_selection_df = pd.DataFrame(all_model_selection_rows)
    all_coefficients_df = pd.concat(all_coefficient_frames, ignore_index=True) if all_coefficient_frames else pd.DataFrame()
    all_calibration_df = pd.concat(all_calibration_frames, ignore_index=True) if all_calibration_frames else pd.DataFrame()
    all_fpr_df = pd.concat(all_fpr_frames, ignore_index=True) if all_fpr_frames else pd.DataFrame()

    all_transfer_metrics_df.to_csv(output_root / "all_transfer_metrics.csv", index=False)
    all_model_selection_df.to_csv(output_root / "all_model_selection.csv", index=False)
    all_coefficients_df.to_csv(output_root / "all_coefficients.csv", index=False)
    all_calibration_df.to_csv(output_root / "all_calibration_curves.csv", index=False)
    all_fpr_df.to_csv(output_root / "all_fpr_at_recall.csv", index=False)

    transfer_summary_df = summarize_transfer_metrics_env(all_transfer_metrics_df)
    train_env_model_summary_df = summarize_train_env_models(all_transfer_metrics_df)
    target_env_breakdown_df = summarize_target_env_breakdown(all_transfer_metrics_df)
    confusion_summary_df = summarize_confusion_counts_env(all_transfer_metrics_df)
    family_panel_selection_df = build_family_panel_selection_env(transfer_summary_df, feature_sizes)
    best_family_models_df = build_best_family_models_env(train_env_model_summary_df, family_panel_selection_df)
    selected_panel_confusion_val_df = build_selected_panel_confusion_summary_env(
        all_transfer_metrics_df,
        family_panel_selection_df,
        eval_role="val",
    )
    selected_panel_confusion_ood_df = build_selected_panel_confusion_summary_env(
        all_transfer_metrics_df,
        family_panel_selection_df,
        eval_role="ood",
    )
    selected_panel_confusion_df = pd.concat(
        [selected_panel_confusion_val_df, selected_panel_confusion_ood_df],
        ignore_index=True,
    ) if not selected_panel_confusion_val_df.empty or not selected_panel_confusion_ood_df.empty else pd.DataFrame()

    transfer_summary_df.to_csv(output_root / "transfer_summary.csv", index=False)
    train_env_model_summary_df.to_csv(output_root / "train_env_model_summary.csv", index=False)
    target_env_breakdown_df.to_csv(output_root / "target_env_breakdown_summary.csv", index=False)
    confusion_summary_df.to_csv(output_root / "confusion_summary.csv", index=False)
    family_panel_selection_df.to_csv(output_root / "best_feature_space_by_target_size_family.csv", index=False)
    best_family_models_df.to_csv(output_root / "best_model_by_target_size_family.csv", index=False)
    selected_panel_confusion_val_df.to_csv(output_root / "selected_panel_confusion_val.csv", index=False)
    selected_panel_confusion_ood_df.to_csv(output_root / "selected_panel_confusion_ood.csv", index=False)
    selected_panel_confusion_df.to_csv(output_root / "selected_panel_confusion_all.csv", index=False)
    export_selected_family_panel_tables_env(
        output_root,
        all_transfer_metrics_df,
        family_panel_selection_df,
        selected_panel_confusion_df,
        train_axis_labels_by_scenario,
    )

    top_feature_tables: list[pd.DataFrame] = []
    top_feature_dir = ensure_dir(output_root / "best_features")
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
        feature_table.insert(0, "importance_rank", np.arange(1, len(feature_table) + 1, dtype=int))
        top_feature_df = feature_table.head(int(args.top_features_to_show)).copy()
        top_feature_tables.append(top_feature_df)
        scenario_dir = ensure_dir(top_feature_dir / str(row.scenario_name))
        out_name = f"{row.target_name}__{row.feature_family_group}__{row.requested_feature_size_label}__top_features.csv"
        top_feature_df.to_csv(scenario_dir / out_name, index=False)
    top_features_for_best_models_df = pd.concat(top_feature_tables, ignore_index=True) if top_feature_tables else pd.DataFrame()
    top_features_for_best_models_df.to_csv(output_root / "top_features_for_best_models.csv", index=False)

    print(f"Output root: {output_root}")
    print(f"Transfer rows: {len(all_transfer_metrics_df):,}")
    print(f"Model selections: {len(all_model_selection_df):,}")
    print(f"Calibration rows: {len(all_calibration_df):,}")
    print(f"FPR rows: {len(all_fpr_df):,}")
    print(f"Target-env breakdown rows: {len(target_env_breakdown_df):,}")


if __name__ == "__main__":
    main()
