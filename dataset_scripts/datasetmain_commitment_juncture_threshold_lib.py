from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[no-redef]
        return iterable

import datasetmain_commitment_juncture_prevalence_lib as cj


DATASETMAIN_ROOT = cj.DATASETMAIN_ROOT
TAU_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_MIN_VALID = 10
DEFAULT_SOURCE_KIND = "localization_json"

EXAMPLE_KEY_COLUMNS = ["env_name", "model_name", "example_id"]

_PREFIX_BASE_COLUMNS = [
    "example_id",
    "sentence_idx",
    "sentence_idx_inclusive",
    "sentence_text",
    "deception_rate",
    "num_valid",
    "num_truthful",
]


def _normalize_prefix_df(
    df: pd.DataFrame,
    *,
    env_name: str,
    model_name: str,
    bundle_dir: Path,
    source_kind: str,
    source_path: Path | None,
    include_sentence_text: bool,
) -> pd.DataFrame:
    out = df.copy()
    if "sentence_idx" not in out.columns and "sentence_idx_inclusive" in out.columns:
        out["sentence_idx"] = out["sentence_idx_inclusive"]
    for column in ["sentence_text", "num_valid", "num_truthful"]:
        if column not in out.columns:
            out[column] = np.nan if column != "sentence_text" else None
    if not include_sentence_text and "sentence_text" in out.columns:
        out = out.drop(columns=["sentence_text"])

    out["example_id"] = out["example_id"].astype(str)
    out["sentence_idx"] = pd.to_numeric(out["sentence_idx"], errors="coerce").fillna(-1).astype("int32")
    out["deception_rate"] = pd.to_numeric(out["deception_rate"], errors="coerce").astype(float)
    out["num_valid"] = pd.to_numeric(out["num_valid"], errors="coerce").astype(float)
    out["num_truthful"] = pd.to_numeric(out["num_truthful"], errors="coerce").astype(float)
    out["env_name"] = env_name
    out["env_display"] = cj.canonical_env_display(env_name)
    out["model_name"] = model_name
    out["model_display"] = cj.canonical_model_display(model_name)
    out["bundle_dir"] = str(bundle_dir)
    out["source_kind"] = source_kind
    out["source_path"] = str(source_path) if source_path is not None else None

    desired_columns = [
        "env_name",
        "env_display",
        "model_name",
        "model_display",
        "example_id",
        "sentence_idx",
        "deception_rate",
        "num_valid",
        "num_truthful",
        "bundle_dir",
        "source_kind",
        "source_path",
    ]
    if include_sentence_text and "sentence_text" in out.columns:
        desired_columns.insert(6, "sentence_text")

    out = out.loc[:, [column for column in desired_columns if column in out.columns]]
    out = out.sort_values(["env_display", "model_display", "example_id", "sentence_idx"], kind="mergesort")
    out = out.reset_index(drop=True)
    return out


def _readable_prefix_parquet(bundle_dir: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    parquet_errors: list[dict[str, Any]] = []
    for candidate_name in cj.PREFIX_PARQUET_CANDIDATES:
        candidate = bundle_dir / candidate_name
        if not candidate.exists():
            continue
        try:
            pq.ParquetFile(candidate)
            return candidate, parquet_errors
        except Exception as exc:
            parquet_errors.append(
                {
                    "bundle_dir": str(bundle_dir),
                    "path": str(candidate),
                    "source_kind": "parquet",
                    "error": repr(exc),
                }
            )
    return None, parquet_errors


def _load_prefix_rows_from_json(
    bundle_dir: Path,
    *,
    env_name: str,
    model_name: str,
    include_sentence_text: bool,
    max_json_files_per_bundle: int | None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    json_paths = cj._localization_paths(bundle_dir, max_json_files_per_bundle=max_json_files_per_bundle)
    path_iterable = json_paths
    if show_progress:
        path_iterable = tqdm(
            json_paths,
            total=len(json_paths),
            desc=progress_desc or f"{cj.canonical_model_display(model_name)} / {cj.canonical_env_display(env_name)}",
            unit="file",
            leave=False,
        )
    for path in path_iterable:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            example_id = str(payload.get("example_id") or path.stem)
            history = payload.get("history") or []
            for fallback_idx, entry in enumerate(history):
                sentence_idx = entry.get("sentence_idx_inclusive", entry.get("sentence_idx", fallback_idx))
                row: dict[str, Any] = {
                    "example_id": example_id,
                    "sentence_idx": sentence_idx,
                    "deception_rate": entry.get("deception_rate"),
                    "num_valid": entry.get("num_valid"),
                    "num_truthful": entry.get("num_truthful"),
                }
                if include_sentence_text:
                    row["sentence_text"] = entry.get("sentence_text")
                rows.append(row)
        except Exception as exc:
            parse_errors.append(
                {
                    "env_name": env_name,
                    "env_display": cj.canonical_env_display(env_name),
                    "model_name": model_name,
                    "model_display": cj.canonical_model_display(model_name),
                    "bundle_dir": str(bundle_dir),
                    "path": str(path),
                    "source_kind": "localization_json",
                    "error": repr(exc),
                }
            )

    prefix_df = pd.DataFrame(rows)
    if prefix_df.empty:
        return prefix_df, pd.DataFrame(parse_errors)
    return (
        _normalize_prefix_df(
            prefix_df,
            env_name=env_name,
            model_name=model_name,
            bundle_dir=bundle_dir,
            source_kind="localization_json",
            source_path=None,
            include_sentence_text=include_sentence_text,
        ),
        pd.DataFrame(parse_errors),
    )


def _load_prefix_rows_for_bundle(
    bundle_dir: Path,
    *,
    env_name: str,
    model_name: str,
    include_sentence_text: bool,
    max_json_files_per_bundle: int | None,
    show_file_progress: bool = False,
    progress_desc: str | None = None,
    preferred_source: str = DEFAULT_SOURCE_KIND,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    preferred_source_key = str(preferred_source or DEFAULT_SOURCE_KIND).strip().lower()
    if preferred_source_key not in {"localization_json", "parquet", "auto"}:
        raise ValueError(
            f"Unsupported preferred_source={preferred_source!r}. "
            "Choose from {'localization_json', 'parquet', 'auto'}."
        )

    def load_from_parquet() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
        parquet_path, parquet_error_rows = _readable_prefix_parquet(bundle_dir)
        if parquet_path is None:
            return None
        schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        read_columns = [column for column in _PREFIX_BASE_COLUMNS if column in schema_names]
        prefix_df = pd.read_parquet(parquet_path, columns=read_columns)
        prefix_df = _normalize_prefix_df(
            prefix_df,
            env_name=env_name,
            model_name=model_name,
            bundle_dir=bundle_dir,
            source_kind="parquet",
            source_path=parquet_path,
            include_sentence_text=include_sentence_text,
        )
        parse_error_df = pd.DataFrame(parquet_error_rows)
        inventory_row = {
            "env_name": env_name,
            "env_display": cj.canonical_env_display(env_name),
            "model_name": model_name,
            "model_display": cj.canonical_model_display(model_name),
            "bundle_dir": str(bundle_dir),
            "source_kind": "parquet",
            "source_path": str(parquet_path),
            "json_file_count": len(cj._localization_paths(bundle_dir, max_json_files_per_bundle=max_json_files_per_bundle)),
            "loaded_rows": int(len(prefix_df)),
            "loaded_examples": int(prefix_df["example_id"].nunique()) if not prefix_df.empty else 0,
        }
        return prefix_df, parse_error_df, inventory_row

    def load_from_json(*, prepend_parquet_errors: list[dict[str, Any]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        prefix_df, parse_error_df = _load_prefix_rows_from_json(
            bundle_dir,
            env_name=env_name,
            model_name=model_name,
            include_sentence_text=include_sentence_text,
            max_json_files_per_bundle=max_json_files_per_bundle,
            show_progress=show_file_progress,
            progress_desc=progress_desc,
        )
        if prepend_parquet_errors:
            parse_error_df = pd.concat([pd.DataFrame(prepend_parquet_errors), parse_error_df], ignore_index=True)
        inventory_row = {
            "env_name": env_name,
            "env_display": cj.canonical_env_display(env_name),
            "model_name": model_name,
            "model_display": cj.canonical_model_display(model_name),
            "bundle_dir": str(bundle_dir),
            "source_kind": "localization_json",
            "source_path": None,
            "json_file_count": len(cj._localization_paths(bundle_dir, max_json_files_per_bundle=max_json_files_per_bundle)),
            "loaded_rows": int(len(prefix_df)),
            "loaded_examples": int(prefix_df["example_id"].nunique()) if not prefix_df.empty else 0,
        }
        return prefix_df, parse_error_df, inventory_row

    if preferred_source_key == "localization_json":
        return load_from_json()

    parquet_path, parquet_error_rows = _readable_prefix_parquet(bundle_dir)
    if preferred_source_key == "parquet":
        if parquet_path is not None:
            return load_from_parquet()
        return load_from_json(prepend_parquet_errors=parquet_error_rows)

    if parquet_path is not None:
        return load_from_parquet()
    return load_from_json(prepend_parquet_errors=parquet_error_rows)


def load_datasetmain_threshold_prefix_df(
    root: Path | str = DATASETMAIN_ROOT,
    *,
    include_sentence_text: bool = False,
    max_json_files_per_bundle: int | None = None,
    preferred_source: str = DEFAULT_SOURCE_KIND,
    show_progress: bool = False,
    progress_level: str = "bundle",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    progress_level_key = str(progress_level or "bundle").strip().lower()
    if progress_level_key not in {"bundle", "file"}:
        raise ValueError(f"Unsupported progress_level={progress_level!r}. Choose from {'bundle', 'file'}.")

    inventory_rows: list[dict[str, Any]] = []
    prefix_frames: list[pd.DataFrame] = []
    parse_error_frames: list[pd.DataFrame] = []

    bundle_rows = cj._bundle_dirs(root)
    bundle_iterable = bundle_rows
    if show_progress:
        bundle_iterable = tqdm(
            bundle_rows,
            total=len(bundle_rows),
            desc="DatasetMain bundles",
            unit="bundle",
        )

    for env_name, model_name, bundle_dir in bundle_iterable:
        prefix_df, parse_error_df, inventory_row = _load_prefix_rows_for_bundle(
            bundle_dir,
            env_name=env_name,
            model_name=model_name,
            include_sentence_text=include_sentence_text,
            max_json_files_per_bundle=max_json_files_per_bundle,
            show_file_progress=show_progress and progress_level_key == "file",
            progress_desc=f"{cj.canonical_model_display(model_name)} / {cj.canonical_env_display(env_name)}",
            preferred_source=preferred_source,
        )
        inventory_rows.append(inventory_row)
        if not prefix_df.empty:
            prefix_frames.append(prefix_df)
        if not parse_error_df.empty:
            parse_error_frames.append(parse_error_df)

    inventory_df = pd.DataFrame(inventory_rows)
    prefix_df = pd.concat(prefix_frames, ignore_index=True) if prefix_frames else pd.DataFrame()
    parse_error_df = pd.concat(parse_error_frames, ignore_index=True) if parse_error_frames else pd.DataFrame()

    if not inventory_df.empty:
        if "model_display" in inventory_df.columns:
            inventory_df["_model_sort"] = inventory_df["model_display"].map(cj._model_sort_key)
        if "env_display" in inventory_df.columns:
            inventory_df["_env_sort"] = inventory_df["env_display"].map(cj._env_sort_key)
        inventory_df = inventory_df.sort_values(["_model_sort", "_env_sort"]).drop(
            columns=["_model_sort", "_env_sort"],
            errors="ignore",
        )
        inventory_df = inventory_df.reset_index(drop=True)

    if not prefix_df.empty:
        prefix_df = prefix_df.sort_values(
            ["env_display", "model_display", "example_id", "sentence_idx"],
            kind="mergesort",
        ).reset_index(drop=True)

    return inventory_df, prefix_df, parse_error_df


def build_consecutive_pair_df(
    prefix_df: pd.DataFrame,
    *,
    min_valid: int = DEFAULT_MIN_VALID,
) -> pd.DataFrame:
    if prefix_df.empty:
        return pd.DataFrame()

    trace_df = prefix_df.copy()
    if "sentence_text" not in trace_df.columns:
        trace_df["sentence_text"] = None
    trace_df["sentence_idx"] = pd.to_numeric(trace_df["sentence_idx"], errors="coerce").fillna(-1).astype("int32")
    trace_df["deception_rate"] = pd.to_numeric(trace_df["deception_rate"], errors="coerce").astype(float)
    trace_df["num_valid"] = pd.to_numeric(trace_df["num_valid"], errors="coerce").astype(float)

    group_columns = ["env_name", "model_name", "example_id"]
    sort_columns = ["env_display", "model_display", "example_id", "sentence_idx"]
    trace_df = trace_df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    grouped = trace_df.groupby(group_columns, observed=True, sort=False)

    trace_df["trace_step"] = grouped.cumcount().astype("int32")
    trace_df["trace_length"] = grouped["sentence_idx"].transform("size").astype("int32")
    trace_df["prev_sentence_idx"] = grouped["sentence_idx"].shift(1)
    trace_df["prev_sentence_text"] = grouped["sentence_text"].shift(1)
    trace_df["prev_deception_rate"] = grouped["deception_rate"].shift(1)
    trace_df["prev_num_valid"] = grouped["num_valid"].shift(1)
    trace_df["delta_deception_rate"] = trace_df["deception_rate"] - trace_df["prev_deception_rate"]

    pair_df = trace_df.loc[trace_df["trace_step"].gt(0)].copy()
    pair_df["pair_is_valid"] = (
        pair_df["prev_deception_rate"].notna()
        & pair_df["deception_rate"].notna()
        & pair_df["prev_num_valid"].gt(float(min_valid))
        & pair_df["num_valid"].gt(float(min_valid))
    )
    pair_df["abs_delta_deception_rate"] = pair_df["delta_deception_rate"].abs()
    pair_df["polarity"] = np.where(
        pair_df["delta_deception_rate"].ge(0.0),
        "toward deception",
        "toward truthfulness",
    )
    pair_df["post_location_fraction"] = (pair_df["trace_step"] + 1) / pair_df["trace_length"]

    rename_map = {
        "sentence_idx": "post_sentence_idx",
        "sentence_text": "post_sentence_text",
        "deception_rate": "post_deception_rate",
        "num_valid": "post_num_valid",
        "prev_sentence_idx": "pre_sentence_idx",
        "prev_sentence_text": "pre_sentence_text",
        "prev_deception_rate": "pre_deception_rate",
        "prev_num_valid": "pre_num_valid",
    }
    pair_df = pair_df.rename(columns=rename_map)

    desired_columns = [
        "env_name",
        "env_display",
        "model_name",
        "model_display",
        "example_id",
        "trace_length",
        "trace_step",
        "pre_sentence_idx",
        "post_sentence_idx",
        "pre_sentence_text",
        "post_sentence_text",
        "pre_deception_rate",
        "post_deception_rate",
        "pre_num_valid",
        "post_num_valid",
        "delta_deception_rate",
        "abs_delta_deception_rate",
        "polarity",
        "pair_is_valid",
        "post_location_fraction",
        "bundle_dir",
        "source_kind",
        "source_path",
    ]
    out = pair_df.loc[:, [column for column in desired_columns if column in pair_df.columns]].copy()
    out = out.sort_values(
        ["env_display", "model_display", "example_id", "trace_step"],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def select_threshold_events(
    pair_df: pd.DataFrame,
    *,
    tau: float,
    polarity: str,
) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df.copy()
    valid_df = pair_df.loc[pair_df["pair_is_valid"].fillna(False)].copy()
    tau_value = float(tau)
    polarity_key = str(polarity).strip().lower()

    if polarity_key in {"positive", "toward deception", "deception"}:
        mask = valid_df["delta_deception_rate"].gt(tau_value)
        event_polarity = "toward deception"
    elif polarity_key in {"negative", "toward truthfulness", "truthfulness"}:
        mask = valid_df["delta_deception_rate"].lt(-tau_value)
        event_polarity = "toward truthfulness"
    elif polarity_key in {"both", "absolute", "either"}:
        mask = valid_df["abs_delta_deception_rate"].gt(tau_value)
        event_polarity = "both"
    else:
        raise ValueError(f"Unsupported polarity={polarity!r}")

    out = valid_df.loc[mask].copy()
    out["tau"] = tau_value
    out["event_polarity"] = event_polarity if event_polarity != "both" else out["polarity"].astype(str)
    return out.reset_index(drop=True)


def _example_coverage(
    valid_subset_df: pd.DataFrame,
    event_subset_df: pd.DataFrame,
) -> tuple[int, int, float]:
    valid_example_count = int(valid_subset_df.loc[:, EXAMPLE_KEY_COLUMNS].drop_duplicates().shape[0])
    event_example_count = int(event_subset_df.loc[:, EXAMPLE_KEY_COLUMNS].drop_duplicates().shape[0])
    coverage = float(event_example_count / valid_example_count) if valid_example_count else float("nan")
    return valid_example_count, event_example_count, coverage


def _summary_row_for_threshold(
    valid_subset_df: pd.DataFrame,
    event_subset_df: pd.DataFrame,
    *,
    tau: float,
    polarity: str,
) -> dict[str, Any]:
    valid_example_count, event_example_count, coverage = _example_coverage(valid_subset_df, event_subset_df)
    valid_pair_count = int(len(valid_subset_df))
    event_pair_count = int(len(event_subset_df))
    pair_coverage = float(event_pair_count / valid_pair_count) if valid_pair_count else float("nan")

    return {
        "tau": float(tau),
        "polarity": str(polarity),
        "valid_example_count": valid_example_count,
        "event_example_count": event_example_count,
        "coverage": coverage,
        "valid_pair_count": valid_pair_count,
        "event_pair_count": event_pair_count,
        "pair_coverage": pair_coverage,
        "mean_delta": float(event_subset_df["delta_deception_rate"].mean()) if not event_subset_df.empty else float("nan"),
        "mean_abs_delta": float(event_subset_df["abs_delta_deception_rate"].mean()) if not event_subset_df.empty else float("nan"),
        "mean_pre_rate": float(event_subset_df["pre_deception_rate"].mean()) if not event_subset_df.empty else float("nan"),
        "mean_post_rate": float(event_subset_df["post_deception_rate"].mean()) if not event_subset_df.empty else float("nan"),
        "mean_post_location": float(event_subset_df["post_location_fraction"].mean()) if not event_subset_df.empty else float("nan"),
    }


def summarize_threshold_sweep(
    pair_df: pd.DataFrame,
    *,
    tau_values: Iterable[float] = TAU_VALUES,
    polarity: str = "positive",
    groupby_columns: list[str] | None = None,
) -> pd.DataFrame:
    if pair_df.empty:
        columns = list(groupby_columns or []) + [
            "tau",
            "polarity",
            "valid_example_count",
            "event_example_count",
            "coverage",
            "valid_pair_count",
            "event_pair_count",
            "pair_coverage",
            "mean_delta",
            "mean_abs_delta",
            "mean_pre_rate",
            "mean_post_rate",
            "mean_post_location",
        ]
        return pd.DataFrame(columns=columns)

    tau_list = [float(value) for value in tau_values]
    groupby_columns = list(groupby_columns or [])
    valid_df = pair_df.loc[pair_df["pair_is_valid"].fillna(False)].copy()
    if valid_df.empty:
        return summarize_threshold_sweep(
            pd.DataFrame(columns=pair_df.columns),
            tau_values=tau_list,
            polarity=polarity,
            groupby_columns=groupby_columns,
        )

    grouped_rows: list[dict[str, Any]] = []
    if groupby_columns:
        grouped_iterable = valid_df.groupby(groupby_columns, dropna=False, observed=True, sort=False)
    else:
        grouped_iterable = [((), valid_df)]

    for group_keys, group_df in grouped_iterable:
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        group_row = {column: value for column, value in zip(groupby_columns, group_keys, strict=True)}
        for tau in tau_list:
            event_df = select_threshold_events(group_df, tau=tau, polarity=polarity)
            row = dict(group_row)
            row.update(_summary_row_for_threshold(group_df, event_df, tau=tau, polarity=polarity))
            grouped_rows.append(row)

    summary_df = pd.DataFrame(grouped_rows)
    if summary_df.empty:
        return summary_df

    if "model_display" in summary_df.columns:
        summary_df["_model_sort"] = summary_df["model_display"].map(cj._model_sort_key)
    if "env_display" in summary_df.columns:
        summary_df["_env_sort"] = summary_df["env_display"].map(cj._env_sort_key)
    summary_df = summary_df.sort_values(
        [column for column in ["_model_sort", "_env_sort", "tau"] if column in summary_df.columns],
        kind="mergesort",
    ).drop(columns=["_model_sort", "_env_sort"], errors="ignore")
    return summary_df.reset_index(drop=True)


def format_threshold_summary_table(
    summary_df: pd.DataFrame,
    *,
    include_group_columns: bool = True,
    include_counts: bool = True,
) -> pd.DataFrame:
    if summary_df.empty:
        columns: list[str] = []
        if include_group_columns:
            if "model_display" in summary_df.columns:
                columns.append("Model")
            if "env_display" in summary_df.columns:
                columns.append("Environment")
        columns.extend(["Threshold", "Coverage", "Mean Delta", "Pre-rate", "Post-rate"])
        if include_counts:
            columns.extend(["Examples", "Pairs"])
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame()
    if include_group_columns:
        if "model_display" in summary_df.columns:
            out["Model"] = summary_df["model_display"].astype(str)
        if "env_display" in summary_df.columns:
            out["Environment"] = summary_df["env_display"].astype(str)
    out["Threshold"] = pd.to_numeric(summary_df["tau"], errors="coerce")
    out["Coverage"] = pd.to_numeric(summary_df["coverage"], errors="coerce")
    out["Mean Delta"] = pd.to_numeric(summary_df["mean_delta"], errors="coerce")
    out["Pre-rate"] = pd.to_numeric(summary_df["mean_pre_rate"], errors="coerce")
    out["Post-rate"] = pd.to_numeric(summary_df["mean_post_rate"], errors="coerce")
    if include_counts:
        out["Examples"] = pd.to_numeric(summary_df["event_example_count"], errors="coerce")
        out["Pairs"] = pd.to_numeric(summary_df["event_pair_count"], errors="coerce")
    return out
