#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit


THIS_FILE = Path(__file__).resolve()
REBUTTAL_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import attention_features as attention_features
import deception_prefix_text_structural_baseline_extractor as structural_extractor
import prefix_target_baselines as prefix_target_baselines

try:
    import deception_prefix_feature_and_activation_extractor as main_feature_extractor
    MAIN_FEATURE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    main_feature_extractor = None  # type: ignore[assignment]
    MAIN_FEATURE_IMPORT_ERROR = exc


DEFAULT_DATASET_ROOT = REPO_ROOT / "DatasetMainCompressed"
DEFAULT_RESULTS_ROOT = REBUTTAL_ROOT / "results"
DEFAULT_RUN_NAME = "commitment_threshold_sweep_v1"
DEFAULT_ENVIRONMENTS = ("advisor_audit", "bs", "car_sales", "gridworld", "interview")
DEFAULT_MODEL_BUNDLE_NAMES = (
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "gpt-oss-20b",
)
DEFAULT_TAU_VALUES = (0.3, 0.4, 0.5)
DEFAULT_LABEL_KINDS = ("deceptive",)
DEFAULT_FIXED_RECALL_LEVELS = (0.5, 0.8, 0.9, 0.95)
DEFAULT_SPLIT_SEED = 42
DEFAULT_TRAIN_SIZE = 0.7
DEFAULT_VAL_SIZE = 0.15
DEFAULT_TEST_SIZE = 0.15
DEFAULT_MIN_VALID = 10

MODEL_ID_BY_BUNDLE = {
    "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}

ENV_DISPLAY_BY_NAME = {
    "advisor_audit": "AdvisorAudit",
    "bs": "BS",
    "car_sales": "CarSales",
    "gridworld": "Gridworld",
    "interview": "Interview",
}

MODEL_DISPLAY_BY_BUNDLE = {
    "DeepSeek-R1-Distill-Llama-8B": "Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B": "Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-7B": "Qwen-7B",
    "gpt-oss-20b": "GPT-OSS-20B",
}

MAIN_FEATURE_METADATA_COLUMNS = set(getattr(main_feature_extractor, "METADATA_COLUMNS", ()))

# These columns are target leakage for rebuttal classifiers because they directly use
# localization outputs that define the commitment labels.
STRUCTURAL_LEAKAGE_COLUMNS = {
    "deception_rate",
    "prev_deception_rate",
    "next_deception_rate",
    "delta_deception_rate",
    "abs_delta_deception_rate",
    "target_value",
    "deceptive_commitment_sentence_idx",
    "truthful_commitment_sentence_idx",
    "example_commitment_sentence_idx",
    "has_deceptive_commitment",
    "has_truthful_commitment",
    "has_example_commitment",
    "is_deceptive_commitment_juncture",
    "is_truthful_commitment_juncture",
    "is_commitment_juncture",
    "is_usable_example",
    "example_label",
    "example_label_source",
    "commitment_direction",
}

POSITION_ONLY_FEATURE_COLUMNS = (
    "sentence_idx",
    "sentence_idx_1based",
    "trace_length",
    "prefix_sentence_count",
    "sentences_remaining",
    "normalized_position",
    "reverse_normalized_position",
    "has_previous_sentence",
    "has_next_sentence",
)

LENGTH_ONLY_FEATURE_COLUMNS = (
    "current_sentence_char_count",
    "current_sentence_word_count",
    "current_sentence_token_count",
    "previous_sentence_char_count",
    "previous_sentence_word_count",
    "previous_sentence_token_count",
    "prefix_char_count",
    "prefix_word_count",
    "prefix_token_count",
    "sentence_char_delta",
    "sentence_word_delta",
    "sentence_token_delta",
)

STRUCTURAL_NONSEMANTIC_FEATURE_COLUMNS = POSITION_ONLY_FEATURE_COLUMNS + LENGTH_ONLY_FEATURE_COLUMNS + (
    "num_valid",
    "num_truthful",
    "start_token",
    "end_token",
    "token_count",
    "context_token_count",
    "prompt_token_count",
    "raw_text_context_token_count",
    "available_token_count",
    "prior_all_token_count",
    "recent_token_count",
    "early_token_count",
    "raw_start",
    "raw_end",
    "full_start",
    "full_end",
)

TEXT_COLUMNS = ("sentence_text", "last_sentence_text", "prefix_text", "full_prefix_text", "prompt")


@dataclass(frozen=True)
class BundleSpec:
    env_name: str
    env_display: str
    model_bundle_name: str
    model_display: str
    model_id: str
    dataset_dir: Path
    localization_dir: Path


def ensure_dir(path: Path | str) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run_root_for_name(run_name: str, results_root: Path | str = DEFAULT_RESULTS_ROOT) -> Path:
    return ensure_dir(Path(results_root) / str(run_name))


def resolve_local_hf_snapshot(model_id_or_path: str | Path) -> str:
    source = Path(str(model_id_or_path)).expanduser()
    if source.exists():
        return str(source.resolve())
    try:
        return snapshot_download(repo_id=str(model_id_or_path), local_files_only=True)
    except Exception:
        return str(model_id_or_path)


def bundle_dir_for_kind(run_root: Path | str, bundle: BundleSpec, kind: str) -> Path:
    return ensure_dir(Path(run_root) / "bundles" / f"{bundle.env_name}__{bundle.model_bundle_name}" / str(kind))


def training_root(run_root: Path | str) -> Path:
    return ensure_dir(Path(run_root) / "training")


def tau_to_token(tau: float) -> str:
    return str(float(tau)).replace(".", "p")


def parse_csv_list(raw_value: str | Sequence[str] | None) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple)):
        return [str(value).strip() for value in raw_value if str(value).strip()]
    return [part.strip() for part in str(raw_value).split(",") if part.strip()]


def parse_tau_values(raw_value: str | Sequence[float] | None) -> list[float]:
    if raw_value is None:
        return list(DEFAULT_TAU_VALUES)
    if isinstance(raw_value, (list, tuple)):
        return [float(value) for value in raw_value]
    return [float(part.strip()) for part in str(raw_value).split(",") if part.strip()]


def parse_recall_levels(raw_value: str | Sequence[float] | None) -> list[float]:
    if raw_value is None:
        return list(DEFAULT_FIXED_RECALL_LEVELS)
    if isinstance(raw_value, (list, tuple)):
        return [float(value) for value in raw_value]
    return [float(part.strip()) for part in str(raw_value).split(",") if part.strip()]


def label_column_name(label_kind: str, tau: float) -> str:
    return f"label_{str(label_kind).strip().lower()}_tau_{tau_to_token(tau)}"


def commitment_index_column_name(label_kind: str, tau: float) -> str:
    return f"commitment_idx_{str(label_kind).strip().lower()}_tau_{tau_to_token(tau)}"


def usable_example_column_name(label_kind: str) -> str:
    return f"usable_{str(label_kind).strip().lower()}_example"


def bundle_specs(
    *,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    environments: Sequence[str] | None = None,
    model_bundle_names: Sequence[str] | None = None,
) -> list[BundleSpec]:
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    requested_envs = tuple(environments or DEFAULT_ENVIRONMENTS)
    requested_models = tuple(model_bundle_names or DEFAULT_MODEL_BUNDLE_NAMES)
    out: list[BundleSpec] = []
    for env_name in requested_envs:
        env_dir = dataset_root_path / str(env_name)
        if not env_dir.exists():
            continue
        for model_bundle_name in requested_models:
            dataset_dir = env_dir / str(model_bundle_name)
            localization_dir = dataset_dir / "localization"
            if not localization_dir.exists():
                continue
            out.append(
                BundleSpec(
                    env_name=str(env_name),
                    env_display=ENV_DISPLAY_BY_NAME.get(str(env_name), str(env_name)),
                    model_bundle_name=str(model_bundle_name),
                    model_display=MODEL_DISPLAY_BY_BUNDLE.get(str(model_bundle_name), str(model_bundle_name)),
                    model_id=MODEL_ID_BY_BUNDLE.get(str(model_bundle_name), str(model_bundle_name)),
                    dataset_dir=dataset_dir,
                    localization_dir=localization_dir,
                )
            )
    return out


def iter_localization_paths(
    localization_dir: Path | str,
    *,
    max_examples: int = 0,
    shard_id: int = 0,
    num_shards: int = 1,
) -> list[Path]:
    attention_features.validate_shard_args(shard_id=int(shard_id), num_shards=int(num_shards))
    localization_dir_path = Path(localization_dir).expanduser().resolve()
    paths = sorted(localization_dir_path.glob("*.json")) + sorted(localization_dir_path.glob("*.json.gz"))
    paths = sorted(paths)
    if max_examples > 0:
        paths = paths[: int(max_examples)]
    if int(num_shards) == 1:
        return paths
    return paths[int(shard_id) :: int(num_shards)]


def load_localization_payload(path: Path | str) -> dict[str, Any]:
    resolved = Path(path)
    if resolved.suffix == ".gz":
        with gzip.open(resolved, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(resolved.read_text(encoding="utf-8"))


def empty_explicit_label_map() -> dict[str, tuple[str, str]]:
    return {}


def infer_example_label(example: dict[str, Any]) -> tuple[Optional[str], str]:
    return structural_extractor.infer_example_label(
        example,
        explicit_label_map=empty_explicit_label_map(),
    )


def compute_commitment_indices(
    aligned_sentence_df: pd.DataFrame,
    *,
    tau: float,
    min_valid: int = DEFAULT_MIN_VALID,
) -> dict[str, Optional[int]]:
    df = aligned_sentence_df.sort_values("sentence_idx").reset_index(drop=True).copy()
    deceptive_idx: Optional[int] = None
    truthful_idx: Optional[int] = None
    prev_rate = float("nan")
    prev_valid = float("nan")
    for row in df.itertuples():
        current_rate = pd.to_numeric(pd.Series([row.deception_rate]), errors="coerce").iloc[0]
        current_valid = pd.to_numeric(pd.Series([row.num_valid]), errors="coerce").iloc[0]
        both_valid = (
            pd.notna(prev_rate)
            and pd.notna(current_rate)
            and pd.notna(prev_valid)
            and pd.notna(current_valid)
            and float(prev_valid) > float(min_valid)
            and float(current_valid) > float(min_valid)
        )
        if both_valid:
            delta = float(current_rate) - float(prev_rate)
            if deceptive_idx is None and delta > float(tau):
                deceptive_idx = int(row.sentence_idx)
            if truthful_idx is None and delta < -float(tau):
                truthful_idx = int(row.sentence_idx)
        prev_rate = current_rate
        prev_valid = current_valid
    return {
        "deceptive": deceptive_idx,
        "truthful": truthful_idx,
    }


def valid_binary_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.isin([0.0, 1.0])


def row_level_split_assignments(
    rows_df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    train_size: float = DEFAULT_TRAIN_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
) -> pd.DataFrame:
    total = float(train_size + val_size + test_size)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train/val/test sizes must sum to 1.0, got {total}")
    split_rows: list[pd.DataFrame] = []
    for (env_name, model_bundle_name), subset_df in rows_df.groupby(["env_name", "model_bundle_name"], sort=False):
        groups = subset_df["example_id"].astype(str).drop_duplicates().reset_index(drop=True)
        if len(groups) < 3:
            raise ValueError(
                f"Need at least 3 unique examples for grouped splits in {env_name} / {model_bundle_name}; "
                f"got {len(groups)}."
            )
        unique_df = pd.DataFrame({"example_id": groups})
        temp_size = float(val_size + test_size)
        outer = GroupShuffleSplit(n_splits=1, test_size=temp_size, random_state=int(seed))
        train_idx, temp_idx = next(outer.split(unique_df, groups=unique_df["example_id"]))
        train_examples = set(unique_df.iloc[train_idx]["example_id"].astype(str))
        temp_examples = unique_df.iloc[temp_idx]["example_id"].astype(str).reset_index(drop=True)
        rel_test_size = float(test_size / temp_size)
        inner = GroupShuffleSplit(n_splits=1, test_size=rel_test_size, random_state=int(seed) + 1)
        val_idx, test_idx = next(inner.split(temp_examples.to_frame(name="example_id"), groups=temp_examples))
        val_examples = set(temp_examples.iloc[val_idx].astype(str))
        test_examples = set(temp_examples.iloc[test_idx].astype(str))

        bundle_df = subset_df.loc[:, ["env_name", "model_bundle_name", "example_id"]].drop_duplicates().copy()
        bundle_df["example_id"] = bundle_df["example_id"].astype(str)
        bundle_df["split"] = "train"
        bundle_df.loc[bundle_df["example_id"].isin(val_examples), "split"] = "val"
        bundle_df.loc[bundle_df["example_id"].isin(test_examples), "split"] = "test"
        bundle_df.loc[bundle_df["example_id"].isin(train_examples), "split"] = "train"
        split_rows.append(bundle_df)
    split_df = pd.concat(split_rows, ignore_index=True)
    return split_df


def attach_splits(rows_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    out = rows_df.merge(
        split_df,
        on=["env_name", "model_bundle_name", "example_id"],
        how="left",
        validate="many_to_one",
    )
    if out["split"].isna().any():
        raise ValueError("Missing split assignment for some rows.")
    return out


def choose_present_columns(df: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [column for column in candidates if column in df.columns]


def numeric_frame(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    out = df.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def safe_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    out: dict[str, float] = {
        "positive_count": float(int(y_true.sum())),
        "negative_count": float(int((1 - y_true).sum())),
        "base_rate": float(y_true.mean()) if y_true.size else float("nan"),
        "auroc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float("nan"),
    }
    if y_true.size == 0:
        return out
    try:
        out["brier"] = float(brier_score_loss(y_true, y_score))
    except Exception:
        out["brier"] = float("nan")
    if np.unique(y_true).size == 2:
        try:
            out["auroc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["auroc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            out["pr_auc"] = float("nan")
    return out


def calibration_curve_frame(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int,
    strategy: str = "quantile",
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0:
        return pd.DataFrame(columns=["bin_idx", "mean_pred", "frac_pos", "count"])
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true,
            y_score,
            n_bins=int(n_bins),
            strategy=str(strategy),
        )
    except Exception:
        return pd.DataFrame(columns=["bin_idx", "mean_pred", "frac_pos", "count"])

    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(y_score, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_ids = pd.cut(y_score, bins=edges, labels=False, include_lowest=True, duplicates="drop")
    count_series = pd.Series(bin_ids).value_counts().sort_index()
    rows: list[dict[str, Any]] = []
    for idx, (pred, frac) in enumerate(zip(mean_pred, frac_pos)):
        rows.append(
            {
                "bin_idx": int(idx),
                "mean_pred": float(pred),
                "frac_pos": float(frac),
                "count": int(count_series.get(idx, 0)),
            }
        )
    return pd.DataFrame(rows)


def fpr_at_fixed_recalls(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_levels: Sequence[float],
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    if y_true.size == 0 or np.unique(y_true).size < 2:
        for recall_level in recall_levels:
            rows.append(
                {
                    "recall_target": float(recall_level),
                    "fpr": float("nan"),
                    "threshold": float("nan"),
                }
            )
        return pd.DataFrame(rows)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    for recall_level in recall_levels:
        mask = tpr >= float(recall_level)
        if not np.any(mask):
            rows.append(
                {
                    "recall_target": float(recall_level),
                    "fpr": float("nan"),
                    "threshold": float("nan"),
                }
            )
            continue
        candidate_idx = int(np.argmin(fpr[mask]))
        selected_idx = np.where(mask)[0][candidate_idx]
        rows.append(
            {
                "recall_target": float(recall_level),
                "fpr": float(fpr[selected_idx]),
                "threshold": float(thresholds[selected_idx]) if np.isfinite(thresholds[selected_idx]) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main_feature_numeric_columns(feature_df: pd.DataFrame) -> list[str]:
    excluded = set(MAIN_FEATURE_METADATA_COLUMNS) | {"env_name", "env_display", "model_bundle_name", "model_display", "model_id"}
    return [column for column in feature_df.columns if column not in excluded]


def structural_feature_columns(rows_df: pd.DataFrame, baseline_name: str) -> list[str]:
    if baseline_name == "position_only":
        candidates = POSITION_ONLY_FEATURE_COLUMNS
    elif baseline_name == "length_only":
        candidates = LENGTH_ONLY_FEATURE_COLUMNS
    elif baseline_name == "structural_nonsemantic":
        candidates = STRUCTURAL_NONSEMANTIC_FEATURE_COLUMNS
    else:
        raise ValueError(f"Unknown structural baseline: {baseline_name}")
    columns = choose_present_columns(rows_df, candidates)
    return [column for column in columns if column not in STRUCTURAL_LEAKAGE_COLUMNS]


def summarize_bundle_rows(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "row_count": int(len(df)),
        "example_count": int(df["example_id"].astype(str).nunique()) if "example_id" in df.columns else 0,
    }
    if "split" in df.columns:
        split_counts = df["split"].astype(str).value_counts().to_dict()
        for key, value in split_counts.items():
            out[f"split_{key}_rows"] = int(value)
    return out
