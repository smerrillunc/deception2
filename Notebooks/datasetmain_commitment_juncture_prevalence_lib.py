from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATASETMAIN_ROOT = Path("/playpen-ssd/smerrill/deception2/DatasetMain")
DELTA_DECEPTION_THRESHOLD = 0.3
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_NUM_RESAMPLES = 1000
BOOTSTRAP_MAX_CHUNK_ELEMENTS = 1_000_000

PREFIX_PARQUET_CANDIDATES = (
    "prefix_deception_features.parquet.tmp",
    "prefix_deception_features.parquet",
)
LOCALIZATION_DIRNAME = "localization"
LOCALIZATION_GLOB = "sentence_localization_*.json"
EXAMPLES_JSONL_NAME = "examples.jsonl"

ENV_DISPLAY_OVERRIDES = {
    "advisor_audit": "AdvisorAudit",
    "advisor-audit": "AdvisorAudit",
    "bs": "BS",
    "car_sales": "CarSales",
    "car-sales": "CarSales",
    "gridworld": "Gridworld",
    "interview": "Interview",
}
ENV_DISPLAY_ORDER = ["AdvisorAudit", "BS", "CarSales", "Gridworld", "Interview"]

MODEL_DISPLAY_OVERRIDES = {
    "deepseek-r1-distill-llama-8b": "Llama-8B",
    "llama-8b": "Llama-8B",
    "deepseek-r1-distill-qwen-7b": "Qwen-7B",
    "qwen-7b": "Qwen-7B",
    "deepseek-r1-distill-qwen-14b": "Qwen-14B",
    "qwen-14b": "Qwen-14B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "gptoss20b": "GPT-OSS-20B",
}
MODEL_DISPLAY_ORDER = ["GPT-OSS-20B", "Llama-8B", "Qwen-7B", "Qwen-14B"]


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _clean_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def canonical_env_display(env_name: Any) -> str:
    text = str(env_name or "").strip()
    if not text:
        return ""
    key = _clean_key(text)
    if key in ENV_DISPLAY_OVERRIDES:
        return ENV_DISPLAY_OVERRIDES[key]
    return text


def canonical_model_display(model_name: Any) -> str:
    text = str(model_name or "").strip()
    if not text:
        return ""
    key = _clean_key(text)
    if key in MODEL_DISPLAY_OVERRIDES:
        return MODEL_DISPLAY_OVERRIDES[key]
    return text


def _env_sort_key(value: Any) -> tuple[int, str]:
    display = canonical_env_display(value)
    try:
        return (ENV_DISPLAY_ORDER.index(display), display)
    except ValueError:
        return (len(ENV_DISPLAY_ORDER), display)


def _model_sort_key(value: Any) -> tuple[int, str]:
    display = canonical_model_display(value)
    try:
        return (MODEL_DISPLAY_ORDER.index(display), display)
    except ValueError:
        return (len(MODEL_DISPLAY_ORDER), display)


def _bundle_dirs(root: Path | str) -> list[tuple[str, str, Path]]:
    root_path = Path(root)
    bundle_rows: list[tuple[str, str, Path]] = []
    if not root_path.exists():
        return bundle_rows
    for env_dir in sorted((path for path in root_path.iterdir() if path.is_dir()), key=lambda path: path.name):
        for model_dir in sorted((path for path in env_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
            bundle_rows.append((env_dir.name, model_dir.name, model_dir))
    return bundle_rows


def _find_prefix_parquet(bundle_dir: Path) -> Path | None:
    for candidate_name in PREFIX_PARQUET_CANDIDATES:
        candidate = bundle_dir / candidate_name
        if candidate.exists():
            return candidate
    return None


def _localization_paths(bundle_dir: Path, max_json_files_per_bundle: int | None = None) -> list[Path]:
    loc_dir = bundle_dir / LOCALIZATION_DIRNAME
    if not loc_dir.exists():
        return []
    paths = sorted(loc_dir.glob(LOCALIZATION_GLOB))
    if max_json_files_per_bundle is not None:
        return paths[: max_json_files_per_bundle]
    return paths


def _history_rows_from_payload(
    payload: dict[str, Any],
    *,
    env_name: str,
    model_name: str,
    bundle_dir: Path,
    source_path: Path,
    include_sentence_text: bool,
) -> list[dict[str, Any]]:
    example_id = str(payload.get("example_id") or source_path.stem)
    history = payload.get("history") or []
    rows: list[dict[str, Any]] = []
    for fallback_idx, entry in enumerate(history):
        sentence_idx = entry.get("sentence_idx_inclusive", entry.get("sentence_idx", fallback_idx))
        try:
            sentence_idx_value = int(sentence_idx)
        except Exception:
            sentence_idx_value = fallback_idx
        row: dict[str, Any] = {
            "env_name": env_name,
            "env_display": canonical_env_display(env_name),
            "model_name": model_name,
            "model_display": canonical_model_display(model_name),
            "example_id": example_id,
            "sentence_idx": sentence_idx_value,
            "deception_rate": entry.get("deception_rate"),
            "bundle_dir": str(bundle_dir),
            "source_kind": "localization_json",
            "source_path": str(source_path),
        }
        if include_sentence_text:
            row["sentence_text"] = entry.get("sentence_text")
        rows.append(row)
    return rows


def _compact_prefix_df(df: pd.DataFrame, *, include_sentence_text: bool) -> pd.DataFrame:
    out = df.copy()
    if not include_sentence_text and "sentence_text" in out.columns:
        out = out.drop(columns=["sentence_text"])
    if "sentence_idx" in out.columns:
        out["sentence_idx"] = pd.to_numeric(out["sentence_idx"], errors="coerce").fillna(-1).astype("int32")
    if "deception_rate" in out.columns:
        out["deception_rate"] = pd.to_numeric(out["deception_rate"], errors="coerce").astype("float32")
    for column in ["env_name", "env_display", "model_name", "model_display", "source_kind"]:
        if column in out.columns:
            out[column] = out[column].astype("category")
    return out.sort_values(["env_display", "model_display", "example_id", "sentence_idx"]).reset_index(drop=True)


def load_datasetmain_prefix_df(
    root: Path | str = DATASETMAIN_ROOT,
    *,
    include_sentence_text: bool = True,
    max_json_files_per_bundle: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory_rows: list[dict[str, Any]] = []
    prefix_frames: list[pd.DataFrame] = []
    parse_errors: list[dict[str, Any]] = []

    for env_name, model_name, bundle_dir in _bundle_dirs(root):
        parquet_path = _find_prefix_parquet(bundle_dir)
        if parquet_path is not None:
            df = pd.read_parquet(parquet_path)
            if "sentence_idx" not in df.columns and "sentence_idx_inclusive" in df.columns:
                df["sentence_idx"] = df["sentence_idx_inclusive"]
            out = df.loc[:, [column for column in ["example_id", "sentence_idx", "deception_rate", "sentence_text"] if column in df.columns]].copy()
            out["env_name"] = env_name
            out["env_display"] = canonical_env_display(env_name)
            out["model_name"] = model_name
            out["model_display"] = canonical_model_display(model_name)
            out["bundle_dir"] = str(bundle_dir)
            out["source_kind"] = "parquet"
            out["source_path"] = str(parquet_path)
            prefix_frames.append(_compact_prefix_df(out, include_sentence_text=include_sentence_text))
            inventory_rows.append(
                {
                    "env_name": env_name,
                    "env_display": canonical_env_display(env_name),
                    "model_name": model_name,
                    "model_display": canonical_model_display(model_name),
                    "bundle_dir": str(bundle_dir),
                    "source_kind": "parquet",
                    "json_file_count": 0,
                    "loaded_examples": int(out["example_id"].astype(str).nunique()),
                }
            )
            continue

        json_rows: list[dict[str, Any]] = []
        json_paths = _localization_paths(bundle_dir, max_json_files_per_bundle=max_json_files_per_bundle)
        for path in json_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                json_rows.extend(
                    _history_rows_from_payload(
                        payload,
                        env_name=env_name,
                        model_name=model_name,
                        bundle_dir=bundle_dir,
                        source_path=path,
                        include_sentence_text=include_sentence_text,
                    )
                )
            except Exception as exc:
                parse_errors.append(
                    {
                        "env_name": env_name,
                        "model_name": model_name,
                        "path": str(path),
                        "error": repr(exc),
                    }
                )
        if json_rows:
            bundle_df = _compact_prefix_df(pd.DataFrame(json_rows), include_sentence_text=include_sentence_text)
            prefix_frames.append(bundle_df)
            inventory_rows.append(
                {
                    "env_name": env_name,
                    "env_display": canonical_env_display(env_name),
                    "model_name": model_name,
                    "model_display": canonical_model_display(model_name),
                    "bundle_dir": str(bundle_dir),
                    "source_kind": "localization_json",
                    "json_file_count": len(json_paths),
                    "loaded_examples": int(bundle_df["example_id"].astype(str).nunique()),
                }
            )

    inventory_df = pd.DataFrame(inventory_rows)
    prefix_df = pd.concat(prefix_frames, ignore_index=True) if prefix_frames else pd.DataFrame()
    parse_error_df = pd.DataFrame(parse_errors)

    if not inventory_df.empty:
        inventory_df = inventory_df.sort_values(["env_display", "model_display"]).reset_index(drop=True)
    if not prefix_df.empty:
        prefix_df = _compact_prefix_df(prefix_df, include_sentence_text=include_sentence_text)
    return inventory_df, prefix_df, parse_error_df


def build_commitment_tables(
    prefix_df: pd.DataFrame,
    *,
    delta_threshold: float = DELTA_DECEPTION_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prefix_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    trace_df = prefix_df.sort_values(["env_display", "model_display", "example_id", "sentence_idx"]).copy()
    trace_df["prev_deception_rate"] = trace_df.groupby(["env_display", "model_display", "example_id"], observed=True)["deception_rate"].shift(1)
    trace_df["delta_deception_rate"] = trace_df["deception_rate"] - trace_df["prev_deception_rate"]
    trace_df["trace_length"] = trace_df.groupby(["env_display", "model_display", "example_id"], observed=True)["sentence_idx"].transform("max") + 1

    event_mask = trace_df["delta_deception_rate"].gt(delta_threshold) | trace_df["delta_deception_rate"].lt(-delta_threshold)
    event_df = trace_df.loc[event_mask, [
        "env_name",
        "env_display",
        "model_name",
        "model_display",
        "example_id",
        "sentence_idx",
        "delta_deception_rate",
        "trace_length",
    ]].copy()
    event_df["polarity"] = np.where(
        event_df["delta_deception_rate"].ge(0),
        "toward deception",
        "toward truthfulness",
    )
    event_df["location_fraction"] = (event_df["sentence_idx"] + 1) / event_df["trace_length"]
    event_df = event_df.sort_values(["env_display", "model_display", "example_id", "sentence_idx"]).reset_index(drop=True)

    positive_first = (
        event_df.loc[event_df["polarity"].eq("toward deception")]
        .drop_duplicates(["env_display", "model_display", "example_id"], keep="first")
        .rename(
            columns={
                "sentence_idx": "deceptive_commitment_sentence_idx",
                "location_fraction": "deceptive_commitment_location",
            }
        )
        .loc[:, ["env_display", "model_display", "example_id", "deceptive_commitment_sentence_idx", "deceptive_commitment_location"]]
    )
    negative_first = (
        event_df.loc[event_df["polarity"].eq("toward truthfulness")]
        .drop_duplicates(["env_display", "model_display", "example_id"], keep="first")
        .rename(
            columns={
                "sentence_idx": "truthful_commitment_sentence_idx",
                "location_fraction": "truthful_commitment_location",
            }
        )
        .loc[:, ["env_display", "model_display", "example_id", "truthful_commitment_sentence_idx", "truthful_commitment_location"]]
    )
    summary_df = (
        trace_df.loc[:, ["env_name", "env_display", "model_name", "model_display", "example_id", "trace_length"]]
        .drop_duplicates(["env_display", "model_display", "example_id"])
        .merge(positive_first, on=["env_display", "model_display", "example_id"], how="left")
        .merge(negative_first, on=["env_display", "model_display", "example_id"], how="left")
        .sort_values(["env_display", "model_display", "example_id"])
        .reset_index(drop=True)
    )
    return summary_df, event_df


def select_gallery_examples(
    event_df: pd.DataFrame,
    *,
    polarity: str,
    one_per_env: bool = True,
) -> pd.DataFrame:
    if event_df.empty:
        return event_df.copy()
    out = event_df.loc[event_df["polarity"].astype(str).eq(str(polarity))].copy()
    out = out.sort_values(["env_display", "model_display", "example_id", "sentence_idx"])
    if one_per_env:
        out = out.drop_duplicates(["env_display"], keep="first")
    return out.reset_index(drop=True)


def load_example_trace(
    root: Path | str,
    env_name: str,
    model_name: str,
    example_id: str,
) -> pd.DataFrame:
    bundle_dir = Path(root) / str(env_name) / str(model_name)
    parquet_path = _find_prefix_parquet(bundle_dir)
    if parquet_path is not None:
        df = pd.read_parquet(parquet_path)
        if "sentence_idx" not in df.columns and "sentence_idx_inclusive" in df.columns:
            df["sentence_idx"] = df["sentence_idx_inclusive"]
        out = df.loc[df["example_id"].astype(str).eq(str(example_id)), [column for column in ["example_id", "sentence_idx", "sentence_text", "deception_rate"] if column in df.columns]].copy()
        if not out.empty:
            out["sentence_idx"] = pd.to_numeric(out["sentence_idx"], errors="coerce").fillna(-1).astype("int32")
            out["deception_rate"] = pd.to_numeric(out["deception_rate"], errors="coerce").astype("float32")
            return out.sort_values("sentence_idx").reset_index(drop=True)

    loc_dir = bundle_dir / LOCALIZATION_DIRNAME
    preferred_path = loc_dir / f"sentence_localization_{example_id}.json"
    candidate_paths = [preferred_path] if preferred_path.exists() else []
    if not candidate_paths:
        candidate_paths = sorted(loc_dir.glob(f"*{example_id.replace('/', '*')}*.json"))
    if not candidate_paths:
        return pd.DataFrame(columns=["example_id", "sentence_idx", "sentence_text", "deception_rate"])
    payload = json.loads(candidate_paths[0].read_text(encoding="utf-8"))
    out = pd.DataFrame(
        _history_rows_from_payload(
            payload,
            env_name=env_name,
            model_name=model_name,
            bundle_dir=bundle_dir,
            source_path=candidate_paths[0],
            include_sentence_text=True,
        )
    )
    if out.empty:
        return pd.DataFrame(columns=["example_id", "sentence_idx", "sentence_text", "deception_rate"])
    out["sentence_idx"] = pd.to_numeric(out["sentence_idx"], errors="coerce").fillna(-1).astype("int32")
    out["deception_rate"] = pd.to_numeric(out["deception_rate"], errors="coerce").astype("float32")
    return out.loc[:, ["example_id", "sentence_idx", "sentence_text", "deception_rate"]].sort_values("sentence_idx").reset_index(drop=True)


def attach_gallery_sentence_text(gallery_df: pd.DataFrame, *, root: Path | str = DATASETMAIN_ROOT) -> pd.DataFrame:
    if gallery_df.empty:
        return gallery_df.copy()
    out = gallery_df.copy()
    cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    sentence_texts: list[str | None] = []
    for row in out.itertuples(index=False):
        key = (str(row.env_name), str(row.model_name), str(row.example_id))
        if key not in cache:
            cache[key] = load_example_trace(root, key[0], key[1], key[2])
        trace_df = cache[key]
        sentence_lookup = trace_df.loc[trace_df["sentence_idx"].astype(int).eq(int(row.sentence_idx)), "sentence_text"]
        sentence_texts.append(str(sentence_lookup.iloc[0]) if not sentence_lookup.empty else None)
    out["sentence_text"] = sentence_texts
    return out


def _extract_example_label_from_generation_block(full_score: dict[str, Any]) -> str | None:
    generation_votes: list[bool] = []
    for generation in full_score.get("generations") or []:
        deceptive = _coerce_bool(generation.get("deceptive"))
        if deceptive is not None:
            generation_votes.append(deceptive)
            continue
        truthful = _coerce_bool(generation.get("is_truthful"))
        if truthful is not None:
            generation_votes.append(not truthful)
    if generation_votes:
        deceptive_fraction = float(np.mean(generation_votes))
        if deceptive_fraction > 0.5:
            return "deceptive"
        if deceptive_fraction < 0.5:
            return "truthful"
    deceptive = _coerce_bool(full_score.get("deceptive"))
    if deceptive is not None:
        return "deceptive" if deceptive else "truthful"
    truthful = _coerce_bool(full_score.get("is_truthful"))
    if truthful is not None:
        return "truthful" if truthful else "deceptive"
    rate = pd.to_numeric(pd.Series([full_score.get("deception_rate")]), errors="coerce").iloc[0]
    if pd.notna(rate):
        return "deceptive" if float(rate) > 0.5 else "truthful"
    return None


def _summarize_localization_example_payload(
    payload: dict[str, Any],
    *,
    default_example_id: str,
    delta_threshold: float = DELTA_DECEPTION_THRESHOLD,
) -> dict[str, Any]:
    example_id = str(payload.get("example_id") or default_example_id)
    history = payload.get("history") or []
    sorted_history = []
    for fallback_idx, entry in enumerate(history):
        sentence_idx = entry.get("sentence_idx_inclusive", entry.get("sentence_idx", fallback_idx))
        try:
            sentence_idx_value = int(sentence_idx)
        except Exception:
            sentence_idx_value = fallback_idx
        sorted_history.append(
            {
                "sentence_idx": sentence_idx_value,
                "deception_rate": pd.to_numeric(pd.Series([entry.get("deception_rate")]), errors="coerce").iloc[0],
            }
        )
    sorted_history.sort(key=lambda item: item["sentence_idx"])
    trace_length = len(sorted_history)

    deceptive_commitment_sentence_idx = np.nan
    truthful_commitment_sentence_idx = np.nan
    deceptive_commitment_location = np.nan
    truthful_commitment_location = np.nan
    previous_rate = np.nan
    for entry in sorted_history:
        current_rate = entry["deception_rate"]
        if pd.notna(previous_rate) and pd.notna(current_rate):
            delta = float(current_rate) - float(previous_rate)
            if math.isnan(deceptive_commitment_location) and delta > delta_threshold:
                deceptive_commitment_sentence_idx = entry["sentence_idx"]
                deceptive_commitment_location = (entry["sentence_idx"] + 1) / trace_length if trace_length else np.nan
            if math.isnan(truthful_commitment_location) and delta < -delta_threshold:
                truthful_commitment_sentence_idx = entry["sentence_idx"]
                truthful_commitment_location = (entry["sentence_idx"] + 1) / trace_length if trace_length else np.nan
        previous_rate = current_rate

    full_score = payload.get("full_score") or {}
    example_label = _extract_example_label_from_generation_block(full_score)
    if example_label is None:
        example_label = _extract_example_label_from_generation_block(sorted_history[-1] if sorted_history else {})

    return {
        "example_id": example_id,
        "trace_length": trace_length,
        "full_deception_rate": pd.to_numeric(pd.Series([full_score.get("deception_rate")]), errors="coerce").iloc[0],
        "example_label": example_label,
        "is_usable_example": example_label in {"deceptive", "truthful"},
        "deceptive_commitment_sentence_idx": deceptive_commitment_sentence_idx,
        "deceptive_commitment_location": deceptive_commitment_location,
        "deceptive_commitment_example": pd.notna(deceptive_commitment_location),
        "truthful_commitment_sentence_idx": truthful_commitment_sentence_idx,
        "truthful_commitment_location": truthful_commitment_location,
        "truthful_commitment_example": pd.notna(truthful_commitment_location),
    }


def _parse_localization_example_file_worker(
    path: str,
    *,
    env_name: str,
    model_name: str,
    delta_threshold: float = DELTA_DECEPTION_THRESHOLD,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = _summarize_localization_example_payload(
        payload,
        default_example_id=Path(path).stem,
        delta_threshold=delta_threshold,
    )
    summary["env_name"] = env_name
    summary["env_display"] = canonical_env_display(env_name)
    summary["model_name"] = model_name
    summary["model_display"] = canonical_model_display(model_name)
    summary["source_path"] = str(path)
    return summary


def _load_examples_label_map(bundle_dir_str: str | Path) -> dict[str, str]:
    bundle_dir = Path(bundle_dir_str)
    path = bundle_dir / EXAMPLES_JSONL_NAME
    label_map: dict[str, str] = {}
    if not path.exists():
        return label_map
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            example_id = row.get("example_id")
            if example_id is None:
                continue
            deceptive = _coerce_bool(row.get("deceptive"))
            if deceptive is not None:
                label_map[str(example_id)] = "deceptive" if deceptive else "truthful"
                continue
            truthful = _coerce_bool(row.get("is_truthful"))
            if truthful is not None:
                label_map[str(example_id)] = "truthful" if truthful else "deceptive"
    return label_map


def _annotate_localization_example_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    for column in [
        "deceptive_commitment_location",
        "truthful_commitment_location",
        "full_deception_rate",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    for column in [
        "trace_length",
        "deceptive_commitment_sentence_idx",
        "truthful_commitment_sentence_idx",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["env_name", "env_display", "model_name", "model_display", "example_label"]:
        if column in out.columns:
            out[column] = out[column].astype("category")
    return out


def _finalize_localization_example_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = _annotate_localization_example_df(df)
    return out.sort_values(["model_display", "env_display", "example_id"]).reset_index(drop=True)


def read_bundle_localization_example_summaries(
    bundle_dir: Path | str,
    *,
    env_name: str,
    model_name: str,
    max_json_files_per_bundle: int | None = None,
    delta_threshold: float = DELTA_DECEPTION_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bundle_path = Path(bundle_dir)
    json_paths = _localization_paths(bundle_path, max_json_files_per_bundle=max_json_files_per_bundle)
    examples_label_map = _load_examples_label_map(bundle_path)

    rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for path in json_paths:
        try:
            row = _parse_localization_example_file_worker(
                str(path),
                env_name=env_name,
                model_name=model_name,
                delta_threshold=delta_threshold,
            )
            explicit_label = examples_label_map.get(str(row["example_id"]))
            if explicit_label is not None:
                row["example_label"] = explicit_label
                row["is_usable_example"] = True
            rows.append(row)
        except Exception as exc:
            parse_errors.append(
                {
                    "env_name": env_name,
                    "env_display": canonical_env_display(env_name),
                    "model_name": model_name,
                    "model_display": canonical_model_display(model_name),
                    "path": str(path),
                    "error": repr(exc),
                }
            )

    example_df = _finalize_localization_example_df(pd.DataFrame(rows))
    parse_error_df = pd.DataFrame(parse_errors)
    inventory_row = {
        "env_name": env_name,
        "env_display": canonical_env_display(env_name),
        "model_name": model_name,
        "model_display": canonical_model_display(model_name),
        "bundle_dir": str(bundle_path),
        "json_file_count": len(json_paths),
        "loaded_examples": int(len(example_df)),
        "usable_examples": int(example_df["is_usable_example"].fillna(False).sum()) if not example_df.empty else 0,
        "unusable_examples": int((~example_df["is_usable_example"].fillna(False)).sum()) if not example_df.empty else 0,
    }
    return example_df, parse_error_df, inventory_row


def build_localization_only_inventory(
    root: Path | str = DATASETMAIN_ROOT,
    *,
    max_json_files_per_bundle: int | None = None,
) -> pd.DataFrame:
    inventory_rows: list[dict[str, Any]] = []
    for env_name, model_name, bundle_dir in _bundle_dirs(root):
        inventory_rows.append(
            {
                "env_name": env_name,
                "env_display": canonical_env_display(env_name),
                "model_name": model_name,
                "model_display": canonical_model_display(model_name),
                "bundle_dir": str(bundle_dir),
                "json_file_count": len(_localization_paths(bundle_dir, max_json_files_per_bundle=max_json_files_per_bundle)),
                "has_examples_jsonl": (bundle_dir / EXAMPLES_JSONL_NAME).exists(),
            }
        )
    inventory_df = pd.DataFrame(inventory_rows)
    if inventory_df.empty:
        return inventory_df
    return inventory_df.sort_values(["model_display", "env_display"]).reset_index(drop=True)


def load_datasetmain_localization_example_df(
    root: Path | str = DATASETMAIN_ROOT,
    *,
    max_json_files_per_bundle: int | None = None,
    delta_threshold: float = DELTA_DECEPTION_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory_rows: list[dict[str, Any]] = []
    example_frames: list[pd.DataFrame] = []
    parse_error_frames: list[pd.DataFrame] = []

    for env_name, model_name, bundle_dir in _bundle_dirs(root):
        example_df, parse_error_df, inventory_row = read_bundle_localization_example_summaries(
            bundle_dir,
            env_name=env_name,
            model_name=model_name,
            max_json_files_per_bundle=max_json_files_per_bundle,
            delta_threshold=delta_threshold,
        )
        inventory_rows.append(inventory_row)
        if not example_df.empty:
            example_frames.append(example_df)
        if not parse_error_df.empty:
            parse_error_frames.append(parse_error_df)

    inventory_df = pd.DataFrame(inventory_rows)
    if not inventory_df.empty:
        inventory_df = inventory_df.sort_values(["model_display", "env_display"]).reset_index(drop=True)
    example_df = _finalize_localization_example_df(pd.concat(example_frames, ignore_index=True)) if example_frames else pd.DataFrame()
    parse_error_df = pd.concat(parse_error_frames, ignore_index=True) if parse_error_frames else pd.DataFrame()
    return inventory_df, example_df, parse_error_df


def _standard_error(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float).dropna()
    if len(series) <= 1:
        return 0.0 if len(series) == 1 else float("nan")
    return float(series.std(ddof=1) / math.sqrt(len(series)))


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _bootstrap_mean_ci(
    values: Iterable[float],
    *,
    key: Any,
    num_resamples: int = BOOTSTRAP_NUM_RESAMPLES,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    series = pd.Series(list(values), dtype=float).dropna()
    if series.empty:
        return float("nan"), float("nan")
    if len(series) == 1:
        value = float(series.iloc[0])
        return value, value

    array = series.to_numpy(dtype=float)
    n = len(array)
    chunk_size = max(1, min(num_resamples, BOOTSTRAP_MAX_CHUNK_ELEMENTS // max(n, 1)))
    rng = np.random.default_rng(_stable_seed(key))
    means: list[np.ndarray] = []
    remaining = int(num_resamples)
    while remaining > 0:
        current_chunk = min(chunk_size, remaining)
        draw = rng.choice(array, size=(current_chunk, n), replace=True)
        means.append(draw.mean(axis=1))
        remaining -= current_chunk
    boot_means = np.concatenate(means)
    alpha = 1.0 - float(confidence_level)
    lower, upper = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def _commitment_stats_for_subset(
    subset_df: pd.DataFrame,
    *,
    label_name: str,
    bootstrap_location_ci_key: Any | None = None,
    bootstrap_num_resamples: int = BOOTSTRAP_NUM_RESAMPLES,
    bootstrap_confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    label_subset = subset_df.loc[subset_df["example_label"].astype(str).eq(label_name)].copy()
    label_prefix = "deceptive" if label_name == "deceptive" else "truthful"
    location_column = f"{label_prefix}_commitment_location"
    commitment_flag_column = f"{label_prefix}_commitment_example"

    total_examples = int(len(label_subset))
    commitment_subset = label_subset.loc[label_subset[commitment_flag_column].fillna(False)].copy()
    commitment_examples = int(len(commitment_subset))
    fraction = float(commitment_examples / total_examples) if total_examples else float("nan")

    location_values = pd.to_numeric(commitment_subset[location_column], errors="coerce").dropna()
    location_mean = float(location_values.mean()) if not location_values.empty else float("nan")
    location_se = _standard_error(location_values.tolist())
    location_ci_lower = float("nan")
    location_ci_upper = float("nan")
    if bootstrap_location_ci_key is not None and not location_values.empty:
        location_ci_lower, location_ci_upper = _bootstrap_mean_ci(
            location_values.tolist(),
            key=(bootstrap_location_ci_key, label_name),
            num_resamples=bootstrap_num_resamples,
            confidence_level=bootstrap_confidence_level,
        )

    return {
        f"{label_prefix}_examples": total_examples,
        f"{label_prefix}_commitment_examples": commitment_examples,
        f"{label_prefix}_commitment_example_fraction": fraction,
        f"{label_prefix}_commitment_example_location_mean": location_mean,
        f"{label_prefix}_commitment_example_location_se": location_se,
        f"{label_prefix}_commitment_example_location_ci_lower": location_ci_lower,
        f"{label_prefix}_commitment_example_location_ci_upper": location_ci_upper,
    }


def build_commitment_example_statistics(
    example_df: pd.DataFrame,
    groupby_columns: list[str],
    *,
    bootstrap_location_ci: bool = False,
    bootstrap_num_resamples: int = BOOTSTRAP_NUM_RESAMPLES,
    bootstrap_confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    if example_df.empty:
        return pd.DataFrame(columns=groupby_columns)

    usable_df = example_df.loc[example_df["is_usable_example"].fillna(False)].copy()
    if usable_df.empty:
        return pd.DataFrame(columns=groupby_columns)

    grouped_rows: list[dict[str, Any]] = []
    for group_keys, group_df in usable_df.groupby(groupby_columns, dropna=False, observed=True):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row = {column: value for column, value in zip(groupby_columns, group_keys, strict=True)}
        bootstrap_key = tuple(group_keys) if bootstrap_location_ci else None
        row.update(
            _commitment_stats_for_subset(
                group_df,
                label_name="deceptive",
                bootstrap_location_ci_key=bootstrap_key,
                bootstrap_num_resamples=bootstrap_num_resamples,
                bootstrap_confidence_level=bootstrap_confidence_level,
            )
        )
        row.update(
            _commitment_stats_for_subset(
                group_df,
                label_name="truthful",
                bootstrap_location_ci_key=bootstrap_key,
                bootstrap_num_resamples=bootstrap_num_resamples,
                bootstrap_confidence_level=bootstrap_confidence_level,
            )
        )
        grouped_rows.append(row)

    stats_df = pd.DataFrame(grouped_rows)
    if "model_display" in stats_df.columns:
        stats_df["_model_sort"] = stats_df["model_display"].map(_model_sort_key)
    if "env_display" in stats_df.columns:
        stats_df["_env_sort"] = stats_df["env_display"].map(_env_sort_key)
    sort_columns = [column for column in ["_model_sort", "_env_sort"] if column in stats_df.columns]
    if sort_columns:
        stats_df = stats_df.sort_values(sort_columns)
    stats_df = stats_df.drop(columns=["_model_sort", "_env_sort"], errors="ignore")
    return stats_df.reset_index(drop=True)


def _format_location_with_se(mean_value: Any, se_value: Any) -> str:
    mean_numeric = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
    if pd.isna(mean_numeric):
        return ""
    se_numeric = pd.to_numeric(pd.Series([se_value]), errors="coerce").iloc[0]
    if pd.isna(se_numeric):
        return f"{100 * float(mean_numeric):.1f}%"
    return f"{100 * float(mean_numeric):.1f}% +/- {100 * float(se_numeric):.1f}%"


def _format_location_with_ci(mean_value: Any, lower_value: Any, upper_value: Any) -> str:
    mean_numeric = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
    if pd.isna(mean_numeric):
        return ""
    lower_numeric = pd.to_numeric(pd.Series([lower_value]), errors="coerce").iloc[0]
    upper_numeric = pd.to_numeric(pd.Series([upper_value]), errors="coerce").iloc[0]
    if pd.isna(lower_numeric) or pd.isna(upper_numeric):
        return f"{100 * float(mean_numeric):.1f}%"
    return (
        f"{100 * float(mean_numeric):.1f}% "
        f"[{100 * float(lower_numeric):.1f}%, {100 * float(upper_numeric):.1f}%]"
    )


def make_commitment_paper_table(
    stats_df: pd.DataFrame,
    *,
    location_interval_style: str = "se",
) -> pd.DataFrame:
    if stats_df.empty:
        columns = ["Model"]
        if "env_display" in stats_df.columns:
            columns.append("Environment")
        columns.extend(
            [
                "Deceptive Examples",
                "Deceptive Commitment Examples",
                "Deceptive Commitment Example Fraction",
                "Deceptive Commitment Example Location",
                "Truthful Examples",
                "Truthful Commitment Examples",
                "Truthful Commitment Example Fraction",
                "Truthful Commitment Example Location",
            ]
        )
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame()
    if "model_display" in stats_df.columns:
        out["Model"] = stats_df["model_display"].astype(str)
    if "env_display" in stats_df.columns:
        out["Environment"] = stats_df["env_display"].astype(str)
    out["Deceptive Examples"] = stats_df["deceptive_examples"].astype(int)
    out["Deceptive Commitment Examples"] = stats_df["deceptive_commitment_examples"].astype(int)
    out["Deceptive Commitment Example Fraction"] = pd.to_numeric(
        stats_df["deceptive_commitment_example_fraction"],
        errors="coerce",
    )
    if location_interval_style == "bootstrap_ci":
        out["Deceptive Commitment Example Location"] = [
            _format_location_with_ci(mean_value, lower_value, upper_value)
            for mean_value, lower_value, upper_value in zip(
                stats_df["deceptive_commitment_example_location_mean"],
                stats_df["deceptive_commitment_example_location_ci_lower"],
                stats_df["deceptive_commitment_example_location_ci_upper"],
                strict=False,
            )
        ]
    else:
        out["Deceptive Commitment Example Location"] = [
            _format_location_with_se(mean_value, se_value)
            for mean_value, se_value in zip(
                stats_df["deceptive_commitment_example_location_mean"],
                stats_df["deceptive_commitment_example_location_se"],
                strict=False,
            )
        ]

    out["Truthful Examples"] = stats_df["truthful_examples"].astype(int)
    out["Truthful Commitment Examples"] = stats_df["truthful_commitment_examples"].astype(int)
    out["Truthful Commitment Example Fraction"] = pd.to_numeric(
        stats_df["truthful_commitment_example_fraction"],
        errors="coerce",
    )
    if location_interval_style == "bootstrap_ci":
        out["Truthful Commitment Example Location"] = [
            _format_location_with_ci(mean_value, lower_value, upper_value)
            for mean_value, lower_value, upper_value in zip(
                stats_df["truthful_commitment_example_location_mean"],
                stats_df["truthful_commitment_example_location_ci_lower"],
                stats_df["truthful_commitment_example_location_ci_upper"],
                strict=False,
            )
        ]
    else:
        out["Truthful Commitment Example Location"] = [
            _format_location_with_se(mean_value, se_value)
            for mean_value, se_value in zip(
                stats_df["truthful_commitment_example_location_mean"],
                stats_df["truthful_commitment_example_location_se"],
                strict=False,
            )
        ]

    desired_columns = ["Model"]
    if "Environment" in out.columns:
        desired_columns.append("Environment")
    desired_columns.extend(
        [
            "Deceptive Examples",
            "Deceptive Commitment Examples",
            "Deceptive Commitment Example Fraction",
            "Deceptive Commitment Example Location",
            "Truthful Examples",
            "Truthful Commitment Examples",
            "Truthful Commitment Example Fraction",
            "Truthful Commitment Example Location",
        ]
    )
    return out.loc[:, desired_columns]
