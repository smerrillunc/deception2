#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from localization_fulltrace_rebuttal_lib import (
    DEFAULT_RESULTS_ROOT,
    DEFAULT_RUN_NAME,
    example_output_filename,
    read_json,
    relpath_from_repo,
    resolve_repo_path,
    run_root,
    write_csv,
    write_json,
)

BUNDLE_GROUP_COLUMNS = ["env_name", "env_display", "model_bundle_name", "model_display"]
MODEL_GROUP_COLUMNS = ["model_bundle_name", "model_display", "model_id"]
SUMMARY_METRIC_COLUMNS = [
    "num_examples",
    "deceptive_examples",
    "truthful_examples",
    "mean_adaptive_probe_fraction",
    "boundary_exact_rate",
    "boundary_within_one_rate",
    "bracket_contains_boundary_rate",
    "peak_exact_rate",
    "peak_within_one_rate",
    "mean_peak_probe_recall_within_one",
    "multi_peak_fraction",
    "gradual_fraction",
    "sharp_or_other_fraction",
    "prompt_match_rate",
    "raw_text_match_rate",
]
PER_EXAMPLE_COLUMNS = [
    "bundle_key",
    "env_name",
    "env_display",
    "model_bundle_name",
    "model_display",
    "model_id",
    "example_id",
    "deceptive",
    "label_name",
    "sentence_count",
    "prompt_match",
    "raw_text_match",
    "adaptive_probe_count",
    "adaptive_probe_fraction",
    "full_boundary_sentence_end_idx",
    "adaptive_left_sentence_end_idx",
    "adaptive_right_sentence_end_idx",
    "boundary_exact",
    "boundary_within_one",
    "adaptive_bracket_contains_full_boundary",
    "boundary_abs_error",
    "adaptive_peak_sentence_idx",
    "peak_exact",
    "peak_within_one",
    "adaptive_exact_peak_probe_recall",
    "adaptive_within_one_peak_probe_recall",
    "adaptive_all_peaks_covered_within_one",
    "full_peak_sentence_idx",
    "full_peak_deception_rate",
    "full_prominent_peak_count",
    "full_prominent_peak_sentence_indices",
    "full_total_positive_rise",
    "full_max_positive_jump",
    "full_positive_jump_count",
    "full_jump_concentration",
    "trace_shape_label",
    "is_multi_peak",
    "is_gradual",
    "adaptive_output_relpath",
    "full_output_relpath",
]
CURVE_POINT_COLUMNS = [
    "bundle_key",
    "example_id",
    "method",
    "sentence_idx",
    "sentence_number",
    "sentence_end_idx",
    "deception_rate",
    "num_valid",
    "num_truthful",
    "sentence_text",
    "is_probed_by_adaptive",
    "env_name",
    "model_bundle_name",
    "label_name",
    "full_boundary_sentence_end_idx",
    "adaptive_right_sentence_end_idx",
    "trace_shape_label",
]
SHAPE_PREVALENCE_COLUMNS = BUNDLE_GROUP_COLUMNS + [
    "trace_shape_label",
    "num_examples",
    "total_examples",
    "fraction",
]
CASE_STUDY_COLUMNS = PER_EXAMPLE_COLUMNS + ["case_category"]
COMPLETION_COLUMNS = [
    "bundle_key",
    "env_name",
    "env_display",
    "model_bundle_name",
    "model_display",
    "model_id",
    "expected_examples",
    "completed_dataset_adaptive_examples",
    "completed_adaptive_examples",
    "completed_full_examples",
    "completed_paired_examples",
]
MODEL_COMPLETION_COLUMNS = [
    "model_bundle_name",
    "model_display",
    "model_id",
    "expected_examples",
    "completed_dataset_adaptive_examples",
    "completed_adaptive_examples",
    "completed_full_examples",
    "completed_paired_examples",
]


@dataclass(frozen=True)
class ShapeThresholds:
    boundary_threshold: float
    peak_min_value: float
    peak_prominence: float
    min_peak_separation: int
    gradual_total_rise_threshold: float
    gradual_max_jump_threshold: float
    gradual_step_threshold: float
    gradual_jump_concentration_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze matched adaptive/full localization rebuttal runs and quantify "
            "trace shapes plus adaptive-vs-full agreement."
        )
    )
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--peak-min-value", type=float, default=0.5)
    parser.add_argument("--peak-prominence", type=float, default=0.15)
    parser.add_argument("--min-peak-separation", type=int, default=2)
    parser.add_argument("--gradual-total-rise-threshold", type=float, default=0.3)
    parser.add_argument("--gradual-max-jump-threshold", type=float, default=0.25)
    parser.add_argument("--gradual-step-threshold", type=float, default=0.05)
    parser.add_argument("--gradual-jump-concentration-threshold", type=float, default=0.65)
    return parser.parse_args()


def shape_thresholds_from_args(args: argparse.Namespace) -> ShapeThresholds:
    return ShapeThresholds(
        boundary_threshold=float(args.boundary_threshold),
        peak_min_value=float(args.peak_min_value),
        peak_prominence=float(args.peak_prominence),
        min_peak_separation=max(1, int(args.min_peak_separation)),
        gradual_total_rise_threshold=float(args.gradual_total_rise_threshold),
        gradual_max_jump_threshold=float(args.gradual_max_jump_threshold),
        gradual_step_threshold=float(args.gradual_step_threshold),
        gradual_jump_concentration_threshold=float(args.gradual_jump_concentration_threshold),
    )


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def safe_sum(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.sum())


def read_localization_payload(path: Path) -> dict[str, Any]:
    file_path = Path(path)
    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(file_path.read_text(encoding="utf-8"))


def local_peak_candidates(values: np.ndarray) -> list[int]:
    n = int(values.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [0]

    peaks: list[int] = []
    start = 0
    while start < n:
        end = start
        while end + 1 < n and float(values[end + 1]) == float(values[start]):
            end += 1
        left_value = float(values[start - 1]) if start > 0 else float("-inf")
        right_value = float(values[end + 1]) if end + 1 < n else float("-inf")
        plateau_value = float(values[start])
        if plateau_value >= left_value and plateau_value > right_value:
            peaks.append((start + end) // 2)
        start = end + 1

    if not peaks:
        peaks = [int(np.argmax(values))]
    return peaks


def approximate_peak_prominence(values: np.ndarray, peak_idx: int) -> float:
    peak_idx = int(peak_idx)
    peak_value = float(values[peak_idx])
    left_min = float(np.min(values[: peak_idx + 1])) if peak_idx > 0 else peak_value
    right_min = float(np.min(values[peak_idx:])) if peak_idx + 1 < len(values) else peak_value
    return float(peak_value - max(left_min, right_min))


def prominent_peaks(values: np.ndarray, *, thresholds: ShapeThresholds) -> list[int]:
    if values.size == 0:
        return []
    raw_candidates = local_peak_candidates(values)
    scored: list[tuple[float, float, int]] = []
    for peak_idx in raw_candidates:
        prominence = approximate_peak_prominence(values, peak_idx)
        peak_value = float(values[peak_idx])
        if peak_value < thresholds.peak_min_value or prominence < thresholds.peak_prominence:
            continue
        scored.append((peak_value, prominence, int(peak_idx)))
    scored.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)

    kept: list[int] = []
    for _, _, peak_idx in scored:
        if any(abs(int(peak_idx) - int(existing)) < thresholds.min_peak_separation for existing in kept):
            continue
        kept.append(int(peak_idx))
    return sorted(kept)


def shape_metrics(values: np.ndarray, *, thresholds: ShapeThresholds) -> dict[str, Any]:
    if values.size == 0:
        return {
            "full_peak_sentence_idx": math.nan,
            "full_peak_deception_rate": math.nan,
            "full_prominent_peak_count": 0,
            "full_prominent_peak_sentence_indices": "",
            "full_total_positive_rise": 0.0,
            "full_max_positive_jump": 0.0,
            "full_positive_jump_count": 0,
            "full_jump_concentration": math.nan,
            "trace_shape_label": "unknown",
            "is_multi_peak": False,
            "is_gradual": False,
        }

    diffs = np.diff(values) if values.size > 1 else np.asarray([], dtype=np.float64)
    positive_diffs = np.maximum(diffs, 0.0)
    total_positive_rise = float(np.sum(positive_diffs)) if positive_diffs.size else 0.0
    max_positive_jump = float(np.max(positive_diffs)) if positive_diffs.size else 0.0
    positive_jump_count = int(np.sum(positive_diffs >= thresholds.gradual_step_threshold)) if positive_diffs.size else 0
    jump_concentration = (
        float(max_positive_jump / total_positive_rise)
        if total_positive_rise > 0.0
        else math.nan
    )
    full_peak_sentence_idx = int(np.argmax(values))
    full_peak_deception_rate = float(np.max(values))
    peak_indices = prominent_peaks(values, thresholds=thresholds)
    is_multi_peak = len(peak_indices) >= 2
    is_gradual = bool(
        (not is_multi_peak)
        and (total_positive_rise >= thresholds.gradual_total_rise_threshold)
        and (max_positive_jump < thresholds.gradual_max_jump_threshold)
        and (positive_jump_count >= 3)
        and (not math.isnan(jump_concentration))
        and (jump_concentration < thresholds.gradual_jump_concentration_threshold)
    )
    if is_multi_peak:
        shape_label = "multi_peak"
    elif is_gradual:
        shape_label = "gradual"
    else:
        shape_label = "sharp_or_other"

    return {
        "full_peak_sentence_idx": int(full_peak_sentence_idx),
        "full_peak_deception_rate": float(full_peak_deception_rate),
        "full_prominent_peak_count": int(len(peak_indices)),
        "full_prominent_peak_sentence_indices": ",".join(str(int(idx) + 1) for idx in peak_indices),
        "full_total_positive_rise": float(total_positive_rise),
        "full_max_positive_jump": float(max_positive_jump),
        "full_positive_jump_count": int(positive_jump_count),
        "full_jump_concentration": float(jump_concentration) if not math.isnan(jump_concentration) else math.nan,
        "trace_shape_label": shape_label,
        "is_multi_peak": bool(is_multi_peak),
        "is_gradual": bool(is_gradual),
    }


def earliest_boundary_end_idx(
    sentence_end_indices: np.ndarray,
    deception_rates: np.ndarray,
    num_valid: np.ndarray,
    *,
    threshold: float,
    min_valid: int,
) -> int | None:
    for end_idx, rate, valid in zip(sentence_end_indices, deception_rates, num_valid):
        if int(valid) < int(min_valid):
            continue
        if float(rate) >= float(threshold):
            return int(end_idx)
    return None


def full_history_frame(payload: dict[str, Any], *, bundle_key: str, example_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in payload.get("history", []) or []:
        sentence_idx = int(entry.get("sentence_idx", 0))
        rows.append(
            {
                "bundle_key": bundle_key,
                "example_id": example_id,
                "method": "full",
                "sentence_idx": sentence_idx,
                "sentence_number": sentence_idx + 1,
                "sentence_end_idx": sentence_idx + 1,
                "deception_rate": float(entry.get("deception_rate", math.nan)),
                "num_valid": float(entry.get("num_valid", math.nan)),
                "num_truthful": float(entry.get("num_truthful", math.nan)),
                "sentence_text": str(entry.get("sentence_text") or ""),
            }
        )
    return pd.DataFrame(rows).sort_values("sentence_idx").reset_index(drop=True)


def adaptive_history_frame(payload: dict[str, Any], *, bundle_key: str, example_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in payload.get("history", []) or []:
        sentence_end_idx = int(entry.get("sentence_end_idx", 0))
        sentence_idx = max(0, sentence_end_idx - 1)
        rows.append(
            {
                "bundle_key": bundle_key,
                "example_id": example_id,
                "method": "adaptive",
                "sentence_idx": sentence_idx,
                "sentence_number": sentence_idx + 1,
                "sentence_end_idx": sentence_end_idx,
                "deception_rate": float(entry.get("deception_rate", math.nan)),
                "num_valid": float(entry.get("num_valid", math.nan)),
                "num_truthful": float(entry.get("num_truthful", math.nan)),
                "sentence_text": str(entry.get("sentence_text") or ""),
            }
        )
    return pd.DataFrame(rows).sort_values("sentence_idx").reset_index(drop=True)


def compare_records(
    *,
    selection_row: dict[str, Any],
    full_payload: dict[str, Any],
    adaptive_payload: dict[str, Any],
    thresholds: ShapeThresholds,
    min_valid: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    bundle_key = str(selection_row["bundle_key"])
    example_id = str(selection_row["example_id"])
    full_df = full_history_frame(full_payload, bundle_key=bundle_key, example_id=example_id)
    adaptive_df = adaptive_history_frame(adaptive_payload, bundle_key=bundle_key, example_id=example_id)
    if full_df.empty or adaptive_df.empty:
        raise ValueError(f"Missing history rows for paired example {bundle_key} / {example_id}")

    full_df["is_probed_by_adaptive"] = full_df["sentence_idx"].isin(set(adaptive_df["sentence_idx"].astype(int)))
    adaptive_df["is_probed_by_adaptive"] = True

    full_values = full_df["deception_rate"].to_numpy(dtype=np.float64, copy=False)
    full_end_indices = full_df["sentence_end_idx"].to_numpy(dtype=np.int64, copy=False)
    full_num_valid = full_df["num_valid"].to_numpy(dtype=np.float64, copy=False)
    full_boundary_end_idx = earliest_boundary_end_idx(
        full_end_indices,
        full_values,
        full_num_valid,
        threshold=thresholds.boundary_threshold,
        min_valid=int(min_valid),
    )

    adaptive_values = adaptive_df["deception_rate"].to_numpy(dtype=np.float64, copy=False)
    adaptive_peak_sentence_idx = int(adaptive_df.loc[adaptive_df["deception_rate"].astype(float).idxmax(), "sentence_idx"])
    adaptive_probed_sentence_idx = sorted(set(int(value) for value in adaptive_df["sentence_idx"].tolist()))

    peak_metrics = shape_metrics(full_values, thresholds=thresholds)
    full_peak_sentence_idx = int(peak_metrics["full_peak_sentence_idx"]) if not math.isnan(float(peak_metrics["full_peak_sentence_idx"])) else None

    prominent_peak_indices = [
        int(value) - 1
        for value in str(peak_metrics["full_prominent_peak_sentence_indices"]).split(",")
        if str(value).strip()
    ]
    exact_peak_hits = [
        1.0 if any(int(probed_idx) == int(peak_idx) for probed_idx in adaptive_probed_sentence_idx) else 0.0
        for peak_idx in prominent_peak_indices
    ]
    within_one_peak_hits = [
        1.0 if any(abs(int(probed_idx) - int(peak_idx)) <= 1 for probed_idx in adaptive_probed_sentence_idx) else 0.0
        for peak_idx in prominent_peak_indices
    ]

    adaptive_right_end_idx = adaptive_payload.get("right_sentence_end_idx")
    adaptive_left_end_idx = adaptive_payload.get("left_sentence_end_idx")
    if adaptive_right_end_idx is not None:
        adaptive_right_end_idx = int(adaptive_right_end_idx)
    if adaptive_left_end_idx is not None:
        adaptive_left_end_idx = int(adaptive_left_end_idx)

    if full_boundary_end_idx is None and adaptive_right_end_idx is None:
        boundary_exact = 1.0
        boundary_within_one = 1.0
        bracket_contains_boundary = 1.0
        boundary_abs_error = 0.0
    elif full_boundary_end_idx is None or adaptive_right_end_idx is None:
        boundary_exact = 0.0
        boundary_within_one = 0.0
        bracket_contains_boundary = 0.0
        boundary_abs_error = math.nan
    else:
        boundary_abs_error = float(abs(int(full_boundary_end_idx) - int(adaptive_right_end_idx)))
        boundary_exact = float(int(full_boundary_end_idx) == int(adaptive_right_end_idx))
        boundary_within_one = float(abs(int(full_boundary_end_idx) - int(adaptive_right_end_idx)) <= 1)
        bracket_contains_boundary = float(
            adaptive_left_end_idx is not None
            and int(adaptive_left_end_idx) <= int(full_boundary_end_idx) <= int(adaptive_right_end_idx)
        )

    prompt_match = str(full_payload.get("prompt") or "") == str(adaptive_payload.get("prompt") or "")
    raw_text_match = str(full_payload.get("raw_text") or "") == str(adaptive_payload.get("raw_text") or "")

    metrics_row = {
        "bundle_key": bundle_key,
        "env_name": str(selection_row["env_name"]),
        "env_display": str(selection_row["env_display"]),
        "model_bundle_name": str(selection_row["model_bundle_name"]),
        "model_display": str(selection_row["model_display"]),
        "model_id": str(selection_row["model_id"]),
        "example_id": example_id,
        "deceptive": bool(selection_row["deceptive"]),
        "label_name": str(selection_row["label_name"]),
        "sentence_count": int(selection_row["sentence_count"]),
        "prompt_match": bool(prompt_match),
        "raw_text_match": bool(raw_text_match),
        "adaptive_probe_count": int(len(adaptive_df)),
        "adaptive_probe_fraction": float(len(adaptive_df) / len(full_df)),
        "full_boundary_sentence_end_idx": full_boundary_end_idx if full_boundary_end_idx is not None else math.nan,
        "adaptive_left_sentence_end_idx": adaptive_left_end_idx if adaptive_left_end_idx is not None else math.nan,
        "adaptive_right_sentence_end_idx": adaptive_right_end_idx if adaptive_right_end_idx is not None else math.nan,
        "boundary_exact": float(boundary_exact),
        "boundary_within_one": float(boundary_within_one),
        "adaptive_bracket_contains_full_boundary": float(bracket_contains_boundary),
        "boundary_abs_error": float(boundary_abs_error) if not math.isnan(boundary_abs_error) else math.nan,
        "adaptive_peak_sentence_idx": int(adaptive_peak_sentence_idx),
        "peak_exact": float(full_peak_sentence_idx is not None and int(adaptive_peak_sentence_idx) == int(full_peak_sentence_idx)),
        "peak_within_one": float(full_peak_sentence_idx is not None and abs(int(adaptive_peak_sentence_idx) - int(full_peak_sentence_idx)) <= 1),
        "adaptive_exact_peak_probe_recall": float(np.mean(exact_peak_hits)) if exact_peak_hits else math.nan,
        "adaptive_within_one_peak_probe_recall": float(np.mean(within_one_peak_hits)) if within_one_peak_hits else math.nan,
        "adaptive_all_peaks_covered_within_one": float(all(bool(value) for value in within_one_peak_hits)) if within_one_peak_hits else math.nan,
        **peak_metrics,
    }

    curve_points = pd.concat([full_df, adaptive_df], ignore_index=True, sort=False)
    curve_points["env_name"] = str(selection_row["env_name"])
    curve_points["model_bundle_name"] = str(selection_row["model_bundle_name"])
    curve_points["label_name"] = str(selection_row["label_name"])
    curve_points["full_boundary_sentence_end_idx"] = metrics_row["full_boundary_sentence_end_idx"]
    curve_points["adaptive_right_sentence_end_idx"] = metrics_row["adaptive_right_sentence_end_idx"]
    curve_points["trace_shape_label"] = str(metrics_row["trace_shape_label"])
    return metrics_row, curve_points


def grouped_summary(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg_kwargs = dict(
        num_examples=("example_id", "count"),
        deceptive_examples=("deceptive", lambda values: int(np.sum(values))),
        truthful_examples=("deceptive", lambda values: int(len(values) - np.sum(values))),
        mean_adaptive_probe_fraction=("adaptive_probe_fraction", safe_mean),
        boundary_exact_rate=("boundary_exact", safe_mean),
        boundary_within_one_rate=("boundary_within_one", safe_mean),
        bracket_contains_boundary_rate=("adaptive_bracket_contains_full_boundary", safe_mean),
        peak_exact_rate=("peak_exact", safe_mean),
        peak_within_one_rate=("peak_within_one", safe_mean),
        mean_peak_probe_recall_within_one=("adaptive_within_one_peak_probe_recall", safe_mean),
        multi_peak_fraction=("is_multi_peak", safe_mean),
        gradual_fraction=("is_gradual", safe_mean),
        sharp_or_other_fraction=("trace_shape_label", lambda values: float(np.mean(pd.Series(values).astype(str).eq("sharp_or_other")))),
        prompt_match_rate=("prompt_match", safe_mean),
        raw_text_match_rate=("raw_text_match", safe_mean),
    )
    if not group_columns:
        row = {
            "num_examples": int(df["example_id"].count()),
            "deceptive_examples": int(np.sum(df["deceptive"])),
            "truthful_examples": int(len(df) - np.sum(df["deceptive"])),
            "mean_adaptive_probe_fraction": safe_mean(df["adaptive_probe_fraction"]),
            "boundary_exact_rate": safe_mean(df["boundary_exact"]),
            "boundary_within_one_rate": safe_mean(df["boundary_within_one"]),
            "bracket_contains_boundary_rate": safe_mean(df["adaptive_bracket_contains_full_boundary"]),
            "peak_exact_rate": safe_mean(df["peak_exact"]),
            "peak_within_one_rate": safe_mean(df["peak_within_one"]),
            "mean_peak_probe_recall_within_one": safe_mean(df["adaptive_within_one_peak_probe_recall"]),
            "multi_peak_fraction": safe_mean(df["is_multi_peak"]),
            "gradual_fraction": safe_mean(df["is_gradual"]),
            "sharp_or_other_fraction": float(np.mean(df["trace_shape_label"].astype(str).eq("sharp_or_other"))),
            "prompt_match_rate": safe_mean(df["prompt_match"]),
            "raw_text_match_rate": safe_mean(df["raw_text_match"]),
        }
        return pd.DataFrame([row])
    grouped = (
        df.groupby(group_columns, as_index=False)
        .agg(**agg_kwargs)
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return grouped


def shape_prevalence_table(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    counts = (
        df.groupby(group_columns + ["trace_shape_label"], as_index=False)
        .agg(num_examples=("example_id", "count"))
    )
    totals = (
        df.groupby(group_columns, as_index=False)
        .agg(total_examples=("example_id", "count"))
    )
    merged = counts.merge(totals, on=group_columns, how="left", validate="many_to_one")
    merged["fraction"] = merged["num_examples"] / merged["total_examples"]
    return merged.sort_values(group_columns + ["trace_shape_label"]).reset_index(drop=True)


def plot_bundle_probe_fraction(bundle_summary_df: pd.DataFrame, out_path: Path) -> None:
    if bundle_summary_df.empty:
        return
    plot_df = bundle_summary_df.copy()
    plot_df["bundle_label"] = plot_df["env_display"].astype(str) + "\n" + plot_df["model_display"].astype(str)
    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    ax.bar(
        plot_df["bundle_label"],
        plot_df["mean_adaptive_probe_fraction"],
        color="#5B7C99",
    )
    ax.set_ylabel("Mean dataset-adaptive probe fraction")
    ax.set_title("Dataset adaptive probe coverage relative to exhaustive localization")
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bundle_agreement(bundle_summary_df: pd.DataFrame, out_path: Path) -> None:
    if bundle_summary_df.empty:
        return
    plot_df = bundle_summary_df.copy()
    plot_df["bundle_label"] = plot_df["env_display"].astype(str) + "\n" + plot_df["model_display"].astype(str)
    x = np.arange(len(plot_df))
    width = 0.32
    fig, ax = plt.subplots(figsize=(14.5, 5.5), constrained_layout=True)
    ax.bar(x - width / 2.0, plot_df["peak_within_one_rate"], width=width, label="Peak within one", color="#3D6E70")
    ax.bar(x + width / 2.0, plot_df["boundary_within_one_rate"], width=width, label="Boundary within one", color="#C9774D")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["bundle_label"], rotation=60)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Agreement rate")
    ax.set_title("Dataset adaptive vs exhaustive agreement by bundle")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_shape_prevalence(shape_df: pd.DataFrame, out_path: Path) -> None:
    if shape_df.empty:
        return
    pivot = (
        shape_df.pivot_table(
            index=["env_display", "model_display"],
            columns="trace_shape_label",
            values="fraction",
            fill_value=0.0,
        )
        .reset_index()
    )
    bundle_labels = pivot["env_display"].astype(str) + "\n" + pivot["model_display"].astype(str)
    categories = ["multi_peak", "gradual", "sharp_or_other"]
    colors = {
        "multi_peak": "#2B6CB0",
        "gradual": "#38A169",
        "sharp_or_other": "#D69E2E",
    }
    bottoms = np.zeros(len(pivot), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(14.5, 5.8), constrained_layout=True)
    for category in categories:
        values = pivot[category].to_numpy(dtype=np.float64) if category in pivot.columns else np.zeros(len(pivot), dtype=np.float64)
        ax.bar(bundle_labels, values, bottom=bottoms, label=category.replace("_", " "), color=colors[category])
        bottoms += values
    ax.set_ylabel("Fraction of exhaustive traces")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Gradual and multi-peak trace prevalence by bundle")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pick_case_studies(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    cases: list[pd.DataFrame] = []

    multi_peak_cases = (
        metrics_df.loc[metrics_df["trace_shape_label"].astype(str).eq("multi_peak")]
        .sort_values(["full_prominent_peak_count", "full_peak_deception_rate", "example_id"], ascending=[False, False, True])
        .head(2)
    )
    if not multi_peak_cases.empty:
        multi_peak_cases = multi_peak_cases.copy()
        multi_peak_cases["case_category"] = "multi_peak"
        cases.append(multi_peak_cases)

    gradual_cases = (
        metrics_df.loc[metrics_df["trace_shape_label"].astype(str).eq("gradual")]
        .sort_values(["full_positive_jump_count", "full_total_positive_rise", "example_id"], ascending=[False, False, True])
        .head(2)
    )
    if not gradual_cases.empty:
        gradual_cases = gradual_cases.copy()
        gradual_cases["case_category"] = "gradual"
        cases.append(gradual_cases)

    disagreement_cases = (
        metrics_df.sort_values(
            ["boundary_abs_error", "peak_within_one", "adaptive_probe_fraction", "example_id"],
            ascending=[False, True, True, True],
        )
        .head(2)
    )
    if not disagreement_cases.empty:
        disagreement_cases = disagreement_cases.copy()
        disagreement_cases["case_category"] = "adaptive_disagreement"
        cases.append(disagreement_cases)

    if not cases:
        return pd.DataFrame()
    combined = pd.concat(cases, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["bundle_key", "example_id"], keep="first").reset_index(drop=True)
    return combined


def plot_case_study(example_metrics: dict[str, Any], curve_df: pd.DataFrame, out_path: Path) -> None:
    subset = curve_df.loc[
        curve_df["bundle_key"].astype(str).eq(str(example_metrics["bundle_key"]))
        & curve_df["example_id"].astype(str).eq(str(example_metrics["example_id"]))
    ].copy()
    if subset.empty:
        return

    full_df = subset.loc[subset["method"].astype(str).eq("full")].sort_values("sentence_idx")
    adaptive_df = subset.loc[subset["method"].astype(str).eq("adaptive")].sort_values("sentence_idx")
    if full_df.empty:
        return

    x_full = full_df["sentence_number"].to_numpy(dtype=int)
    y_full = full_df["deception_rate"].to_numpy(dtype=float)
    x_adapt = adaptive_df["sentence_number"].to_numpy(dtype=int)
    y_adapt = adaptive_df["deception_rate"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.plot(x_full, y_full, marker="o", linewidth=2.4, color="black", label="Full")
    ax.scatter(x_adapt, y_adapt, s=70, color="#C05621", label="Dataset adaptive probes", zorder=4)

    peak_string = str(example_metrics.get("full_prominent_peak_sentence_indices") or "")
    for peak_text in [value for value in peak_string.split(",") if value.strip()]:
        peak_num = int(peak_text)
        ax.axvline(peak_num, color="#3182CE", linewidth=1.1, alpha=0.35)

    adaptive_right = example_metrics.get("adaptive_right_sentence_end_idx")
    if pd.notna(adaptive_right):
        ax.axvline(int(adaptive_right), color="#DD6B20", linestyle="--", linewidth=1.4, alpha=0.75, label="Dataset adaptive right boundary")

    full_boundary = example_metrics.get("full_boundary_sentence_end_idx")
    if pd.notna(full_boundary):
        ax.axvline(int(full_boundary), color="#2F855A", linestyle=":", linewidth=1.6, alpha=0.85, label="Full boundary")

    ax.set_title(
        f"{example_metrics['case_category']}: {example_metrics['env_display']} / {example_metrics['model_display']}\n"
        f"{example_metrics['example_id']} | shape={example_metrics['trace_shape_label']}"
    )
    ax.set_xlabel("Sentence index")
    ax.set_ylabel("Deception rate")
    ax.set_xticks(x_full)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary_markdown(
    *,
    completion_summary: dict[str, Any],
    overall_summary_df: pd.DataFrame,
    case_studies_df: pd.DataFrame,
) -> str:
    lines = [
        "# Localization Dataset-Adaptive-vs-Full Summary",
        "",
        f"- Expected paired examples: {int(completion_summary['expected_examples'])}",
        f"- Completed dataset adaptive examples: {int(completion_summary['completed_dataset_adaptive_examples'])}",
        f"- Completed full examples: {int(completion_summary['completed_full_examples'])}",
        f"- Completed paired examples: {int(completion_summary['completed_paired_examples'])}",
    ]
    if not overall_summary_df.empty:
        row = overall_summary_df.iloc[0]
        lines.extend(
            [
                "",
                "## Overall",
                f"- Mean adaptive probe fraction: {float(row['mean_adaptive_probe_fraction']):.3f}",
                f"- Peak within-one agreement: {float(row['peak_within_one_rate']):.3f}",
                f"- Boundary within-one agreement: {float(row['boundary_within_one_rate']):.3f}",
                f"- Multi-peak trace fraction: {float(row['multi_peak_fraction']):.3f}",
                f"- Gradual trace fraction: {float(row['gradual_fraction']):.3f}",
            ]
        )
    if not case_studies_df.empty:
        lines.extend(["", "## Case Studies"])
        for _, row in case_studies_df.iterrows():
            lines.append(
                f"- {row['case_category']}: {row['env_display']} / {row['model_display']} / {row['example_id']}"
            )
    return "\n".join(lines) + "\n"


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty and not list(df.columns):
        return pd.DataFrame(columns=columns)
    missing = [column for column in columns if column not in df.columns]
    for column in missing:
        df[column] = pd.NA
    return df.loc[:, columns]


def completion_summary_by_model(bundle_completion_df: pd.DataFrame) -> pd.DataFrame:
    if bundle_completion_df.empty:
        return pd.DataFrame(columns=MODEL_COMPLETION_COLUMNS)
    model_df = (
        bundle_completion_df.groupby(MODEL_GROUP_COLUMNS, as_index=False)
        .agg(
            expected_examples=("expected_examples", "sum"),
            completed_dataset_adaptive_examples=("completed_dataset_adaptive_examples", "sum"),
            completed_adaptive_examples=("completed_adaptive_examples", "sum"),
            completed_full_examples=("completed_full_examples", "sum"),
            completed_paired_examples=("completed_paired_examples", "sum"),
        )
        .sort_values(["model_display"])
        .reset_index(drop=True)
    )
    return ensure_columns(model_df, MODEL_COMPLETION_COLUMNS)


def main() -> None:
    args = parse_args()
    thresholds = shape_thresholds_from_args(args)
    run_root_path = run_root(args.run_name, results_root=args.results_root)
    analysis_root = run_root_path / "analysis"
    figures_root = analysis_root / "figures"
    analysis_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    config = read_json(run_root_path / "config.json", default={}) or {}
    selected_examples_df = pd.read_csv(run_root_path / "selected_examples.csv")
    full_manifest_df = pd.read_csv(run_root_path / "run_manifest_full.csv")

    full_manifest_by_bundle = {
        str(row["bundle_key"]): row.to_dict()
        for _, row in full_manifest_df.iterrows()
    }

    bundle_completion_rows: list[dict[str, Any]] = []
    per_example_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []

    expected_examples = int(len(selected_examples_df))
    completed_dataset_adaptive_examples = 0
    completed_full_examples = 0
    completed_paired_examples = 0

    for bundle_key, bundle_df in selected_examples_df.groupby("bundle_key", sort=True):
        bundle_key = str(bundle_key)
        full_spec = full_manifest_by_bundle.get(bundle_key)
        if full_spec is None:
            continue

        full_out_dir = resolve_repo_path(str(full_spec["out_dir_relpath"]))
        bundle_dataset_adaptive_completed = 0
        bundle_full_completed = 0
        bundle_paired_completed = 0

        for _, selection_row in bundle_df.iterrows():
            selection = selection_row.to_dict()
            example_id = str(selection["example_id"])
            adaptive_relpath = str(selection.get("source_localization_relpath") or "").strip()
            adaptive_path = (
                resolve_repo_path(adaptive_relpath)
                if adaptive_relpath
                else None
            )
            full_path = full_out_dir / example_output_filename(example_id)
            adaptive_exists = adaptive_path is not None and adaptive_path.exists()
            full_exists = full_path.exists()
            bundle_dataset_adaptive_completed += int(adaptive_exists)
            bundle_full_completed += int(full_exists)
            completed_dataset_adaptive_examples += int(adaptive_exists)
            completed_full_examples += int(full_exists)
            if not adaptive_exists or not full_exists:
                continue

            adaptive_payload = read_localization_payload(adaptive_path)
            full_payload = read_localization_payload(full_path)
            metrics_row, curve_df = compare_records(
                selection_row=selection,
                full_payload=full_payload,
                adaptive_payload=adaptive_payload,
                thresholds=thresholds,
                min_valid=int(config.get("localization_args", {}).get("min_valid", 3)),
            )
            metrics_row["adaptive_output_relpath"] = adaptive_relpath
            metrics_row["full_output_relpath"] = relpath_from_repo(full_path)
            per_example_rows.append(metrics_row)
            curve_frames.append(curve_df)
            bundle_paired_completed += 1
            completed_paired_examples += 1

        bundle_completion_rows.append(
            {
                "bundle_key": bundle_key,
                "env_name": str(bundle_df["env_name"].iloc[0]),
                "env_display": str(bundle_df["env_display"].iloc[0]),
                "model_bundle_name": str(bundle_df["model_bundle_name"].iloc[0]),
                "model_display": str(bundle_df["model_display"].iloc[0]),
                "model_id": str(bundle_df["model_id"].iloc[0]),
                "expected_examples": int(len(bundle_df)),
                "completed_dataset_adaptive_examples": int(bundle_dataset_adaptive_completed),
                "completed_adaptive_examples": int(bundle_dataset_adaptive_completed),
                "completed_full_examples": int(bundle_full_completed),
                "completed_paired_examples": int(bundle_paired_completed),
            }
        )

    completion_summary = {
        "run_name": args.run_name,
        "expected_examples": int(expected_examples),
        "completed_dataset_adaptive_examples": int(completed_dataset_adaptive_examples),
        "completed_adaptive_examples": int(completed_dataset_adaptive_examples),
        "completed_full_examples": int(completed_full_examples),
        "completed_paired_examples": int(completed_paired_examples),
        "adaptive_reference_source": "selected_examples.csv[source_localization_relpath]",
        "shape_thresholds": asdict(thresholds),
        "multi_peak_definition": {
            "rule": "At least two prominent local peaks in the exhaustive deception-rate trace.",
            "peak_min_value": float(thresholds.peak_min_value),
            "peak_prominence": float(thresholds.peak_prominence),
            "min_peak_separation": int(thresholds.min_peak_separation),
        },
        "gradual_definition": {
            "rule": "Not multi-peak, with sustained rise and no single dominant jump.",
            "gradual_total_rise_threshold": float(thresholds.gradual_total_rise_threshold),
            "gradual_max_jump_threshold": float(thresholds.gradual_max_jump_threshold),
            "gradual_step_threshold": float(thresholds.gradual_step_threshold),
            "gradual_jump_concentration_threshold": float(thresholds.gradual_jump_concentration_threshold),
        },
        "created_at_analysis": config.get("created_at_utc", ""),
        "analysis_completed_at": pd.Timestamp.utcnow().isoformat(),
    }

    metrics_df = pd.DataFrame(per_example_rows, columns=PER_EXAMPLE_COLUMNS)
    curve_points_df = (
        pd.concat(curve_frames, ignore_index=True, sort=False)
        if curve_frames
        else pd.DataFrame(columns=CURVE_POINT_COLUMNS)
    )
    curve_points_df = ensure_columns(curve_points_df, CURVE_POINT_COLUMNS)
    bundle_completion_df = (
        pd.DataFrame(bundle_completion_rows, columns=COMPLETION_COLUMNS)
        .sort_values(["env_display", "model_display"])
        .reset_index(drop=True)
    )
    bundle_completion_df = ensure_columns(bundle_completion_df, COMPLETION_COLUMNS)
    model_completion_df = completion_summary_by_model(bundle_completion_df)
    bundle_summary_df = grouped_summary(metrics_df, ["env_name", "env_display", "model_bundle_name", "model_display"])
    bundle_summary_df = ensure_columns(bundle_summary_df, BUNDLE_GROUP_COLUMNS + SUMMARY_METRIC_COLUMNS)
    model_summary_df = grouped_summary(metrics_df, MODEL_GROUP_COLUMNS)
    model_summary_df = ensure_columns(model_summary_df, MODEL_GROUP_COLUMNS + SUMMARY_METRIC_COLUMNS)
    overall_summary_df = grouped_summary(metrics_df, [])
    overall_summary_df = ensure_columns(overall_summary_df, SUMMARY_METRIC_COLUMNS)
    if not overall_summary_df.empty:
        overall_summary_df.insert(0, "summary_scope", "overall")
    else:
        overall_summary_df = pd.DataFrame(columns=["summary_scope", *SUMMARY_METRIC_COLUMNS])
    shape_df = shape_prevalence_table(metrics_df, ["env_name", "env_display", "model_bundle_name", "model_display"])
    shape_df = ensure_columns(shape_df, SHAPE_PREVALENCE_COLUMNS)
    model_shape_df = shape_prevalence_table(metrics_df, MODEL_GROUP_COLUMNS)
    model_shape_df = ensure_columns(model_shape_df, MODEL_GROUP_COLUMNS + ["trace_shape_label", "num_examples", "total_examples", "fraction"])
    case_studies_df = pick_case_studies(metrics_df)
    case_studies_df = ensure_columns(case_studies_df, CASE_STUDY_COLUMNS)

    write_json(analysis_root / "completion_summary.json", completion_summary)
    write_csv(analysis_root / "bundle_completion_summary.csv", bundle_completion_rows)
    model_completion_df.to_csv(analysis_root / "model_completion_summary.csv", index=False)
    write_csv(analysis_root / "per_example_metrics.csv", per_example_rows, fieldnames=PER_EXAMPLE_COLUMNS)
    curve_points_df.to_csv(analysis_root / "curve_points.csv", index=False)
    bundle_summary_df.to_csv(analysis_root / "adaptive_vs_full_summary_by_bundle.csv", index=False)
    model_summary_df.to_csv(analysis_root / "adaptive_vs_full_summary_by_model.csv", index=False)
    overall_summary_df.to_csv(analysis_root / "adaptive_vs_full_summary_overall.csv", index=False)
    shape_df.to_csv(analysis_root / "trace_shape_prevalence_by_bundle.csv", index=False)
    model_shape_df.to_csv(analysis_root / "trace_shape_prevalence_by_model.csv", index=False)
    case_studies_df.to_csv(analysis_root / "case_studies.csv", index=False)

    plot_bundle_probe_fraction(bundle_summary_df, figures_root / "adaptive_probe_fraction_by_bundle.png")
    plot_bundle_agreement(bundle_summary_df, figures_root / "adaptive_vs_full_agreement_by_bundle.png")
    plot_shape_prevalence(shape_df, figures_root / "trace_shape_prevalence_by_bundle.png")
    for _, case_row in case_studies_df.iterrows():
        slug = f"{case_row['case_category']}__{str(case_row['bundle_key']).replace('/', '_')}__{str(case_row['example_id']).replace('/', '_')}"
        plot_case_study(case_row.to_dict(), curve_points_df, figures_root / f"{slug}.png")

    summary_md = build_summary_markdown(
        completion_summary=completion_summary,
        overall_summary_df=overall_summary_df,
        case_studies_df=case_studies_df,
    )
    (analysis_root / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Analysis root: {analysis_root}")
    print(f"Expected examples: {expected_examples}")
    print(f"Completed paired examples: {completed_paired_examples}")
    if not bundle_summary_df.empty:
        print("Bundle summary rows:", len(bundle_summary_df))


if __name__ == "__main__":
    main()
