import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binomtest
from sentence_pipeline import split_sentence_spans


DEFAULT_RESULTS_ROOT = Path("/playpen-ssd/smerrill/deception2/BS/Results/SentencePipeline")
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_SENTENCES_FILENAME = "sentences.jsonl"
DEFAULT_TAGS_FILENAME = "tags.jsonl"
DEFAULT_TAXONOMY_PATH = Path("/playpen-ssd/smerrill/deception2/BS/config/sentence_taxonomy.json")


# -----------------------------
# Utilities
# -----------------------------

def bootstrap_deception_ci(
    generations: List[Dict],
    *,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    truth_values = [
        g.get("is_truthful")
        for g in generations
        if g.get("is_truthful") is not None
    ]
    if not truth_values:
        return np.nan, np.nan

    deceptions = np.array([0.0 if t else 1.0 for t in truth_values], dtype=float)
    if n_boot <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    samples = rng.choice(deceptions, size=(n_boot, len(deceptions)), replace=True)
    rates = samples.mean(axis=1)
    lo, hi = np.quantile(rates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _escape_html_attr(text: str) -> str:
    return (
        _escape_html(text)
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_example_id(example_id: object) -> str:
    if not isinstance(example_id, str):
        return ""
    return example_id.strip()


def _infer_sentence_idx_from_sentence_id(sentence_id: object) -> Optional[int]:
    if not isinstance(sentence_id, str):
        return None
    m = re.search(r"/sent_(\d+)$", sentence_id.strip())
    if not m:
        return None
    return _safe_int(m.group(1))


def normalize_tag_label(label_name: object, taxonomy_labels: Optional[List[str]]) -> str:
    if not isinstance(label_name, str) or not label_name.strip():
        return "other"

    raw = label_name.strip()
    if not taxonomy_labels:
        return raw

    canonical_by_lower = {lbl.lower(): lbl for lbl in taxonomy_labels}

    # Exact (case-insensitive).
    exact = canonical_by_lower.get(raw.lower())
    if exact is not None:
        return exact

    # Prefix before ":", "-", or "(".
    head = re.split(r"[:\-\(]", raw, maxsplit=1)[0].strip().lower()
    if head in canonical_by_lower:
        return canonical_by_lower[head]

    # Substring fallback for noisy labels.
    raw_low = raw.lower()
    for low, canon in canonical_by_lower.items():
        if low in raw_low:
            return canon

    return "other"


def get_tag_display_label(tag_payload: Optional[Dict], *, missing: str = "no_tag") -> str:
    if not isinstance(tag_payload, dict):
        return missing

    for key in ("label_norm", "label_name", "label_raw"):
        value = tag_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "other"


def normalize_history(history: List[Dict]) -> List[Dict]:
    normalized = []
    for probe in history:
        out = dict(probe)

        sent_end = out.get("sentence_end_idx")
        sent_idx = out.get("sentence_idx_inclusive")
        if sent_idx is None:
            sent_idx = out.get("sentence_idx")
        if sent_end is None and sent_idx is not None:
            sent_end = int(sent_idx) + 1
        if sent_end is not None:
            sent_idx = int(sent_end) - 1 if int(sent_end) > 0 else None

        out["sentence_end_idx"] = int(sent_end) if sent_end is not None else None
        out["sentence_idx"] = int(sent_idx) if sent_idx is not None else None

        char_span = out.get("char_span")
        if isinstance(char_span, list):
            char_span = tuple(char_span)
        if isinstance(char_span, tuple) and len(char_span) == 2:
            out["char_span"] = (int(char_span[0]), int(char_span[1]))
            out["char_start"] = int(char_span[0])
            out["char_end"] = int(char_span[1])
        else:
            out["char_span"] = None
            out["char_start"] = None
            out["char_end"] = None

        normalized.append(out)

    return normalized


def flatten_history(history: List[Dict], raw_text: str) -> pd.DataFrame:
    rows = []
    for step_id, probe in enumerate(history):
        generations = probe.get("generations") or []
        for sample_id, gen in enumerate(generations):
            rows.append({
                "step_id": step_id,
                "sentence_end_idx": probe.get("sentence_end_idx"),
                "sentence_idx": probe.get("sentence_idx"),
                "sample_id": sample_id,
                "deception_rate_step": probe.get("deception_rate"),
                "num_truthful_step": probe.get("num_truthful"),
                "num_valid_step": probe.get("num_valid"),
                "gen_text": gen.get("gen_text"),
                "action": gen.get("action"),
                "is_truthful": gen.get("is_truthful"),
                "parse_error": gen.get("parse_error"),
                "parsed": gen.get("parsed"),
                "sentence_text": probe.get("sentence_text"),
                "char_span": probe.get("char_span"),
                "raw_text": raw_text,
            })
    return pd.DataFrame(rows)


def build_stats(history: List[Dict], *, n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES) -> pd.DataFrame:
    rows = []
    for step_id, probe in enumerate(history):
        sent_end = probe.get("sentence_end_idx")
        if sent_end is None:
            continue
        num_true = int(probe.get("num_truthful") or 0)
        num_valid = int(probe.get("num_valid") or 0)
        dec_rate = probe.get("deception_rate")
        if dec_rate is None and num_valid > 0:
            dec_rate = 1.0 - (num_true / num_valid)
        seed = probe.get("seed")
        if seed is None:
            seed = step_id + 1
        generations = probe.get("generations") or []
        ci_low, ci_high = bootstrap_deception_ci(
            generations,
            alpha=0.05,
            n_boot=n_boot,
            seed=int(seed),
        )
        p_value = binomtest(num_true, num_valid, p=0.5).pvalue if num_valid > 0 else 1.0

        rows.append({
            "step_id": step_id,
            "sentence_end_idx": sent_end,
            "sentence_idx": probe.get("sentence_idx"),
            "deception_rate": dec_rate,
            "num_truthful": num_true,
            "num_valid": num_valid,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("sentence_end_idx").reset_index(drop=True)
    return df


def plot_sentence_localization(
    df_stats: pd.DataFrame,
    deceptive_sentence_idx: Optional[int] = None,
    right_sentence_end_idx: Optional[int] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(8, 4))

    x = df_stats["sentence_idx"]
    y = df_stats["deception_rate"]

    ax.plot(x, y, color="#666666", linewidth=2, label="Deception rate")

    if deceptive_sentence_idx is not None:
        after_mask = x > deceptive_sentence_idx
        before_mask = ~after_mask
    else:
        after_mask = pd.Series([False] * len(df_stats))
        before_mask = ~after_mask

    ax.scatter(x[before_mask], y[before_mask], color="#1f77b4", s=40, label="Before/at deceptive")
    if after_mask.any():
        ax.scatter(x[after_mask], y[after_mask], color="#2ca02c", s=40, label="After deceptive")

    if df_stats["ci_low"].notna().any():
        ax.fill_between(
            x,
            df_stats["ci_low"],
            df_stats["ci_high"],
            alpha=0.2,
            label="95% bootstrap CI",
        )

    ax.axhline(0.5, linestyle="--", linewidth=2, label="50% threshold")

    if right_sentence_end_idx is not None:
        ax.axvline(
            int(right_sentence_end_idx) - 1,
            linestyle="-.",
            linewidth=2,
            label=f"Earliest >= 0.5 @ {int(right_sentence_end_idx) - 1}",
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Sentence index")
    ax.set_ylabel("Deception rate")
    if len(df_stats):
        ax.set_title(f"Sentence-level deception localization\nmin p = {df_stats['p_value'].min():.1e}")
    else:
        ax.set_title("Sentence-level deception localization")
    ax.grid(alpha=0.3)
    ax.legend()

    return fig, df_stats


def render_highlighted_sentences(
    raw_text: str,
    sentence_spans: List[Dict],
    selected_idx: Optional[int],
    deceptive_idx: Optional[int],
    selected_span: Optional[Tuple[int, int]] = None,
    sentence_tags: Optional[Dict[int, Dict]] = None,
) -> None:
    if not raw_text:
        st.info("No raw text available.")
        return

    if not sentence_spans:
        html = (
            "<div style=\"background-color:white; padding:8px; line-height:1.5; "
            "white-space:pre-wrap; font-family:serif;\">"
            f"{_escape_html(raw_text)}"
            "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)
        return

    parts: List[str] = []
    last = 0
    sel_start = None
    sel_end = None
    if selected_span and len(selected_span) == 2:
        sel_start, sel_end = selected_span
        if sel_start is not None and sel_end is not None and sel_end <= sel_start:
            sel_start, sel_end = None, None

    def _with_style(text: str, *, bg_color: Optional[str], tooltip: Optional[str]) -> str:
        if not text:
            return ""
        escaped = _escape_html(text)
        inner = escaped
        if bg_color:
            inner = f"<mark style='background-color:{bg_color};'>{escaped}</mark>"
        if tooltip:
            return f"<span title=\"{_escape_html_attr(tooltip)}\">{inner}</span>"
        return inner

    for idx, span in enumerate(sentence_spans):
        start = span.get("start")
        end = span.get("end")
        if start is None or end is None:
            continue
        if start > last:
            parts.append(_escape_html(raw_text[last:start]))

        text = raw_text[start:end]
        span_idx = span.get("sentence_idx")
        if span_idx is None:
            span_idx = idx
        span_idx = _safe_int(span_idx)
        if span_idx is None:
            span_idx = idx
        base_green = deceptive_idx is not None and span_idx > deceptive_idx

        tooltip_parts = [f"Sentence {span_idx}"]
        tag = sentence_tags.get(span_idx) if sentence_tags else None
        if isinstance(tag, dict):
            label_name = get_tag_display_label(tag, missing="")
            confidence = _safe_float(tag.get("confidence"))
            if label_name:
                tooltip_parts.append(f"Tag: {label_name}")
            if confidence is not None and np.isfinite(confidence):
                tooltip_parts.append(f"Confidence: {confidence:.2f}")
        tooltip = " | ".join(tooltip_parts)

        if sel_start is not None and sel_end is not None and not (sel_end <= start or sel_start >= end):
            overlap_start = max(start, sel_start)
            overlap_end = min(end, sel_end)
            before = raw_text[start:overlap_start]
            middle = raw_text[overlap_start:overlap_end]
            after = raw_text[overlap_end:end]
            if base_green:
                if before:
                    parts.append(_with_style(before, bg_color="#c8f7c5", tooltip=tooltip))
                parts.append(_with_style(middle, bg_color="#ffe08a", tooltip=tooltip))
                if after:
                    parts.append(_with_style(after, bg_color="#c8f7c5", tooltip=tooltip))
            else:
                parts.append(_with_style(before, bg_color=None, tooltip=tooltip))
                parts.append(_with_style(middle, bg_color="#ffe08a", tooltip=tooltip))
                parts.append(_with_style(after, bg_color=None, tooltip=tooltip))
        elif selected_idx is not None and span_idx == selected_idx:
            parts.append(_with_style(text, bg_color="#ffe08a", tooltip=tooltip))
        elif base_green:
            parts.append(_with_style(text, bg_color="#c8f7c5", tooltip=tooltip))
        else:
            parts.append(_with_style(text, bg_color=None, tooltip=tooltip))
        last = end

    if last < len(raw_text):
        parts.append(_escape_html(raw_text[last:]))

    html = (
        "<div style=\"background-color:white; padding:8px; line-height:1.5; "
        "white-space:pre-wrap; font-family:serif;\">"
        + "".join(parts)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_prefix_generation_basic(prefix_text: str, gen_text: str) -> None:
    html = (
        "<div style='background-color:white; padding:5px; line-height:1.5; "
        "white-space:pre-wrap; font-family:monospace'>"
        f"<span style='color:blue'>{_escape_html(prefix_text)}</span>"
        f"<span style='color:green'>{_escape_html(gen_text)}</span>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _append_colored_segment(
    html_parts: List[str],
    full_text: str,
    start: int,
    end: int,
    color: str,
) -> None:
    if start >= end:
        return
    html_parts.append(f"<span style='color:{color}'>{_escape_html(full_text[start:end])}</span>")


def render_prefix_generation_with_sentence_indices(
    prefix_text: str,
    gen_text: str,
    selected_sentence_idx: Optional[int],
    sentence_labels: Optional[Dict[int, str]] = None,
) -> None:
    full_text = prefix_text + gen_text
    if not full_text:
        st.info("No text available for rendering.")
        return

    spans = [
        s for s in split_sentence_spans(full_text)
        if s.get("start") is not None and s.get("end") is not None
    ]
    spans.sort(key=lambda s: s.get("start", 0))
    if not spans:
        render_prefix_generation_basic(prefix_text, gen_text)
        return

    if selected_sentence_idx is None and prefix_text:
        prefix_spans = [
            s for s in split_sentence_spans(prefix_text)
            if s.get("start") is not None and s.get("end") is not None
        ]
        prefix_spans.sort(key=lambda s: s.get("start", 0))
        if prefix_spans:
            selected_sentence_idx = len(prefix_spans) - 1

    if selected_sentence_idx is not None and spans:
        selected_sentence_idx = int(selected_sentence_idx)
        if selected_sentence_idx < 0:
            selected_sentence_idx = None
        else:
            selected_sentence_idx = max(0, min(selected_sentence_idx, len(spans) - 1))

    html_parts: List[str] = []
    last = 0

    def color_for_idx(idx: int) -> str:
        if selected_sentence_idx is None:
            return "black"
        if idx < selected_sentence_idx:
            return "blue"
        if idx > selected_sentence_idx:
            return "green"
        return "black"

    for idx, span in enumerate(spans):
        start = span.get("start")
        end = span.get("end")
        color = color_for_idx(idx)
        if start > last:
            _append_colored_segment(html_parts, full_text, last, start, color)
        _append_colored_segment(html_parts, full_text, start, end, color)
        label = None
        if sentence_labels:
            label = sentence_labels.get(idx)
        marker = f"{idx + 1}"
        if label:
            marker = f"{marker} [{label}]."
        html_parts.append(
            f"<sup style='font-size:0.7em;color:{color}'>{_escape_html(marker)}</sup> "
        )
        last = end

    if last < len(full_text):
        tail_color = color_for_idx(len(spans) - 1) if spans else "black"
        _append_colored_segment(html_parts, full_text, last, len(full_text), tail_color)

    html = (
        "<div style='background-color:white; padding:5px; line-height:1.5; "
        "white-space:pre-wrap; font-family:monospace'>"
        + "".join(html_parts)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def strip_code_fences(text: str) -> str:
    if not text:
        return text
    return text.replace("```", "")


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


@st.cache_data(show_spinner=False)
def load_sentences_index(
    path: Path,
    include_example_ids: Optional[Tuple[str, ...]] = None,
) -> Dict[str, List[Dict]]:
    include_set = set(include_example_ids) if include_example_ids else None
    index: Dict[str, List[Dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ex_id = normalize_example_id(rec.get("example_id"))
            if not ex_id:
                continue
            if include_set is not None and ex_id not in include_set:
                continue

            sent_idx = _safe_int(rec.get("sentence_idx"))
            if sent_idx is None:
                continue

            index.setdefault(ex_id, []).append({
                "sentence_idx": sent_idx,
                "sentence_id": rec.get("sentence_id"),
                "sentence_text": rec.get("sentence_text", ""),
                "start": _safe_int(rec.get("start")),
                "end": _safe_int(rec.get("end")),
            })

    for ex_id in list(index.keys()):
        index[ex_id] = sorted(index[ex_id], key=lambda r: r.get("sentence_idx", 0))
    return index


@st.cache_data(show_spinner=False)
def load_taxonomy_labels(path: Path) -> List[str]:
    labels: List[str] = []
    if not path.exists():
        return labels

    obj = json.loads(path.read_text())
    for item in obj.get("labels", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            labels.append(name.strip())
    return labels


@st.cache_data(show_spinner=False)
def load_tags_index(
    path: Path,
    taxonomy_labels: Optional[List[str]] = None,
    include_example_ids: Optional[Tuple[str, ...]] = None,
) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, int], Dict[str, int]]:
    include_set = set(include_example_ids) if include_example_ids else None
    index: Dict[str, Dict[str, Dict]] = {}
    raw_label_counter: Counter = Counter()
    norm_label_counter: Counter = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ex_id = normalize_example_id(rec.get("example_id"))
            if not ex_id:
                continue
            if include_set is not None and ex_id not in include_set:
                continue

            sentence_id = rec.get("sentence_id")
            sentence_idx = _safe_int(rec.get("sentence_idx"))
            if sentence_idx is None:
                sentence_idx = _infer_sentence_idx_from_sentence_id(sentence_id)

            raw_label = rec.get("label_name")
            norm_label = normalize_tag_label(raw_label, taxonomy_labels)
            raw_label_key = raw_label if isinstance(raw_label, str) else str(raw_label)
            raw_label_counter[raw_label_key] += 1
            norm_label_counter[norm_label] += 1

            tag_payload = {
                "label_raw": raw_label,
                "label_norm": norm_label,
                "label_name": norm_label,
                "label_id": rec.get("label_id"),
                "confidence": _safe_float(rec.get("confidence")),
                "sentence_id": sentence_id,
                "sentence_idx": sentence_idx,
            }

            bucket = index.setdefault(ex_id, {"by_sentence_id": {}, "by_sentence_idx": {}})
            if sentence_id:
                bucket["by_sentence_id"][str(sentence_id)] = tag_payload
            if sentence_idx is not None:
                bucket["by_sentence_idx"][int(sentence_idx)] = tag_payload

    return index, dict(raw_label_counter), dict(norm_label_counter)


@st.cache_data(show_spinner=False)
def collect_localization_example_ids_from_json_files(json_paths: Tuple[str, ...]) -> Tuple[str, ...]:
    example_ids = set()
    for path_str in json_paths:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        ex_id = normalize_example_id(rec.get("example_id"))
        if ex_id:
            example_ids.add(ex_id)
    return tuple(sorted(example_ids))


@st.cache_data(show_spinner=False)
def collect_localization_example_ids_from_jsonl(path: Path) -> Tuple[str, ...]:
    example_ids = set()
    for rec in load_jsonl(path):
        ex_id = normalize_example_id(rec.get("example_id"))
        if ex_id:
            example_ids.add(ex_id)
    return tuple(sorted(example_ids))


def collect_localization_example_ids_from_records(records: List[Dict]) -> Tuple[str, ...]:
    example_ids = set()
    for rec in records:
        ex_id = normalize_example_id(rec.get("example_id"))
        if ex_id:
            example_ids.add(ex_id)
    return tuple(sorted(example_ids))


def build_sentence_spans(
    raw_text: str,
    example_id: str,
    sentences_index: Optional[Dict[str, List[Dict]]],
) -> Tuple[List[Dict], Dict[int, Dict]]:
    spans: List[Dict] = []
    if sentences_index and example_id and example_id in sentences_index:
        for rec in sentences_index.get(example_id, []):
            sentence_idx = _safe_int(rec.get("sentence_idx"))
            start = _safe_int(rec.get("start"))
            end = _safe_int(rec.get("end"))
            if sentence_idx is None or start is None or end is None or end <= start:
                continue

            text = rec.get("sentence_text")
            if (not text) and raw_text and start >= 0 and end <= len(raw_text):
                text = raw_text[start:end]

            spans.append({
                "sentence_idx": sentence_idx,
                "start": start,
                "end": end,
                "text": text,
                "sentence_id": rec.get("sentence_id"),
            })

    if not spans:
        for idx, span in enumerate(split_sentence_spans(raw_text)):
            start = _safe_int(span.get("start"))
            end = _safe_int(span.get("end"))
            if start is None or end is None or end <= start:
                continue
            spans.append({
                "sentence_idx": idx,
                "start": start,
                "end": end,
                "text": span.get("text"),
                "sentence_id": None,
            })

    spans = [s for s in spans if s.get("start") is not None and s.get("end") is not None]
    spans.sort(key=lambda s: (s.get("start", 0), s.get("sentence_idx", 0)))
    span_map = {int(s.get("sentence_idx")): s for s in spans if s.get("sentence_idx") is not None}
    return spans, span_map


def resolve_sentence_span(
    sentence_idx: Optional[int],
    sentence_span_map: Dict[int, Dict],
    probe: Optional[Dict] = None,
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    if sentence_idx is None or sentence_idx < 0:
        return None, None, None

    idx = int(sentence_idx)
    if sentence_span_map:
        span = sentence_span_map.get(idx)
        if span:
            return span.get("text"), span.get("start"), span.get("end")
        return None, None, None

    if probe is None:
        return None, None, None

    span = probe.get("char_span")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return probe.get("sentence_text"), int(span[0]), int(span[1])
    return probe.get("sentence_text"), None, None


def compute_deceptive_sentence_idx(
    right_sentence_end_idx: Optional[int],
    df_stats: pd.DataFrame,
) -> Optional[int]:
    if right_sentence_end_idx is not None:
        idx = int(right_sentence_end_idx) - 1
        if idx >= 0:
            return idx

    if len(df_stats) == 0:
        return None

    candidates = df_stats[
        (df_stats["deception_rate"].notna())
        & (df_stats["num_valid"] > 0)
        & (df_stats["deception_rate"] >= 0.5)
    ]
    if len(candidates) == 0:
        return None
    return int(candidates.sort_values("sentence_idx").iloc[0]["sentence_idx"])


def has_localization_outputs(path: Path) -> bool:
    if (path / "localization").is_dir():
        return True
    if any(path.glob("localization*.jsonl")):
        return True
    return False


def discover_localization_dirs(version_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    if has_localization_outputs(version_dir):
        candidates.append(version_dir)
    for child in sorted([p for p in version_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        if has_localization_outputs(child):
            candidates.append(child)
    return candidates


def build_sentence_tags_by_idx(
    example_id: str,
    tags_index: Optional[Dict[str, Dict[str, Dict]]],
    sentences_index: Optional[Dict[str, List[Dict]]] = None,
) -> Dict[int, Dict]:
    norm_example_id = normalize_example_id(example_id)
    if not norm_example_id or not tags_index or not sentences_index:
        return {}
    if norm_example_id not in sentences_index:
        return {}

    example_tags = tags_index.get(norm_example_id)
    if not example_tags:
        return {}

    by_sentence_id = example_tags.get("by_sentence_id", {})
    by_sentence_idx = example_tags.get("by_sentence_idx", {})
    out: Dict[int, Dict] = {}

    # Match notebook logic exactly: iterate sentence rows and join by sentence_id,
    # then fallback to sentence_idx when needed.
    sentence_rows = sorted(
        sentences_index.get(norm_example_id, []),
        key=lambda r: (_safe_int(r.get("sentence_idx")) if _safe_int(r.get("sentence_idx")) is not None else 10**9),
    )

    for rec in sentence_rows:
        sentence_idx = _safe_int(rec.get("sentence_idx"))
        if sentence_idx is None:
            continue

        tag = None
        sentence_id = rec.get("sentence_id")
        if sentence_id:
            tag = by_sentence_id.get(str(sentence_id))
        if tag is None:
            tag = by_sentence_idx.get(sentence_idx)
        if tag is not None:
            out[sentence_idx] = tag

    return out


def sentence_selector_label(
    sentence_idx: int,
    sentence_span_map: Dict[int, Dict],
    sentence_tags_by_idx: Dict[int, Dict],
) -> str:
    idx = _safe_int(sentence_idx)
    if idx is None:
        return str(sentence_idx)

    span = sentence_span_map.get(idx, {})
    sentence_text = span.get("text") if isinstance(span, dict) else ""
    if not isinstance(sentence_text, str):
        sentence_text = ""
    sentence_text = " ".join(sentence_text.split())
    if len(sentence_text) > 120:
        sentence_text = sentence_text[:117] + "..."

    tag = sentence_tags_by_idx.get(idx)
    tag_label = get_tag_display_label(tag, missing="no_tag")

    if sentence_text:
        return f"{idx}: {sentence_text} [{tag_label}]"
    return f"{idx} [{tag_label}]"


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Sentence Localization Explorer", layout="wide")
st.title("Sentence-level Deception Localization Dashboard")

st.sidebar.header("Results")

root_input = st.sidebar.text_input("Results root", value=str(DEFAULT_RESULTS_ROOT))
results_root = Path(root_input)
if not results_root.exists():
    st.sidebar.error("Results root not found.")
    st.stop()

versions = sorted([p for p in results_root.iterdir() if p.is_dir()])
if not versions:
    st.sidebar.error("No result versions found.")
    st.stop()

selected_version = st.sidebar.selectbox(
    "Results version",
    versions,
    format_func=lambda p: p.name,
)

candidate_data_dirs = discover_localization_dirs(selected_version)
if not candidate_data_dirs:
    st.sidebar.error("No localization outputs found under the selected version.")
    st.stop()

if len(candidate_data_dirs) == 1:
    data_dir = candidate_data_dirs[0]
else:
    data_dir = st.sidebar.selectbox(
        "Localization path / model",
        candidate_data_dirs,
        format_func=lambda p: p.name if p != selected_version else f"{p.name} (root)",
    )
st.sidebar.caption(f"Using data dir: {data_dir}")

json_dir = data_dir / "localization"
json_files = sorted(json_dir.glob("sentence_localization_*.json")) if json_dir.exists() else []
jsonl_files = sorted(data_dir.glob("localization*.jsonl"))
sentences_path = data_dir / DEFAULT_SENTENCES_FILENAME
tags_path = data_dir / DEFAULT_TAGS_FILENAME
taxonomy_path = DEFAULT_TAXONOMY_PATH
source_kind = st.sidebar.radio("Load from", ["Per-example JSON", "JSONL"], index=0)

result = None
result_path = None
loc_example_ids: Tuple[str, ...] = tuple()

if source_kind == "Per-example JSON":
    if not json_files:
        st.sidebar.error("No per-example JSON files found.")
        st.stop()
    filter_text = st.sidebar.text_input("Filter filename", value="")
    filtered_files = [p for p in json_files if filter_text in p.name] if filter_text else json_files
    if not filtered_files:
        st.sidebar.error("No files match the filter.")
        st.stop()
    result_path = st.sidebar.selectbox(
        "Result file",
        filtered_files,
        format_func=lambda p: p.name,
    )
    result = load_json(result_path)
    loc_example_ids = collect_localization_example_ids_from_json_files(tuple(str(p) for p in json_files))
else:
    if not jsonl_files:
        st.sidebar.error("No JSONL files found.")
        st.stop()
    jsonl_path = st.sidebar.selectbox(
        "JSONL file",
        jsonl_files,
        format_func=lambda p: p.name,
    )
    records = load_jsonl(jsonl_path)
    filter_text = st.sidebar.text_input("Filter example_id", value="")
    filtered = [r for r in records if filter_text in (r.get("example_id") or "")] if filter_text else records
    if not filtered:
        st.sidebar.error("No JSONL records match the filter.")
        st.stop()
    idx_options = list(range(len(filtered)))
    selected_idx = st.sidebar.selectbox(
        "Example",
        idx_options,
        format_func=lambda i: filtered[i].get("example_id") or f"record {i}",
    )
    result = filtered[selected_idx]
    loc_example_ids = collect_localization_example_ids_from_jsonl(jsonl_path)


if result is None:
    st.stop()

taxonomy_labels: List[str] = []
if taxonomy_path.exists():
    taxonomy_labels = load_taxonomy_labels(taxonomy_path)
else:
    st.sidebar.caption(f"Taxonomy file not found: {taxonomy_path}")

include_example_ids = loc_example_ids if loc_example_ids else None

sentences_index = None
if sentences_path.exists():
    sentences_index = load_sentences_index(sentences_path, include_example_ids=include_example_ids)

tags_index = None
raw_label_counts: Dict[str, int] = {}
norm_label_counts: Dict[str, int] = {}
if tags_path.exists():
    tags_index, raw_label_counts, norm_label_counts = load_tags_index(
        tags_path,
        taxonomy_labels=taxonomy_labels,
        include_example_ids=include_example_ids,
    )
else:
    st.sidebar.caption(f"No {DEFAULT_TAGS_FILENAME} found in selected data dir.")

with st.sidebar.expander("Tag-link diagnostics", expanded=False):
    st.markdown(f"- Localization examples: {len(loc_example_ids)}")
    st.markdown(f"- Examples with sentences: {len(sentences_index) if sentences_index else 0}")
    st.markdown(f"- Examples with tags: {len(tags_index) if tags_index else 0}")
    if norm_label_counts:
        norm_top = pd.Series(norm_label_counts).sort_values(ascending=False).head(15)
        st.markdown("- Top normalized labels:")
        st.dataframe(norm_top.to_frame("count"), use_container_width=True)

raw_text = result.get("raw_text") or ""
history = result.get("history") or []
if not history:
    st.error("No history found in this result.")
    st.stop()

history_norm = normalize_history(history)
df_plot = flatten_history(history_norm, raw_text)
df_stats = build_stats(history_norm)

example_id = normalize_example_id(result.get("example_id") or "")
truthful_rank = result.get("truthful_rank")
right_sentence_end_idx = result.get("right_sentence_end_idx")
deceptive_sentence_idx = compute_deceptive_sentence_idx(right_sentence_end_idx, df_stats)

sentence_spans, sentence_span_map = build_sentence_spans(raw_text, example_id, sentences_index)
sentence_tags_by_idx = build_sentence_tags_by_idx(
    example_id,
    tags_index,
    sentences_index=sentences_index,
)
tag_coverage = f"{len(sentence_tags_by_idx)}/{len(sentence_spans)}" if sentence_spans else "0/0"
has_example_tags = len(sentence_tags_by_idx) > 0

st.subheader("Result Summary")
summary_lines = []
if example_id:
    summary_lines.append(f"Example ID: {example_id}")
if truthful_rank is not None:
    summary_lines.append(f"Truthful rank: {truthful_rank}")
summary_lines.append(f"Probes: {len(history_norm)}")
if deceptive_sentence_idx is not None:
    summary_lines.append(f"Deceptive sentence idx: {deceptive_sentence_idx}")
if result_path:
    summary_lines.append(f"File: {result_path.name}")
summary_lines.append(f"Data directory: {data_dir}")
summary_lines.append(f"Tag coverage (matched/total sentences): {tag_coverage}")
if summary_lines:
    st.markdown("\n".join([f"- {line}" for line in summary_lines]))

with st.expander("Raw prompt"):
    prompt = result.get("prompt") or ""
    if prompt:
        st.text(prompt)
    else:
        st.info("No prompt stored in this result.")

# -----------------------------
# Plot
# -----------------------------
st.subheader("Deception Rate vs Sentence Index")

if len(df_stats) > 0:
    fig1, df_stats = plot_sentence_localization(
        df_stats,
        deceptive_sentence_idx=deceptive_sentence_idx,
        right_sentence_end_idx=right_sentence_end_idx,
    )
    st.pyplot(fig1)
    with st.expander("Show probe statistics"):
        st.dataframe(df_stats, use_container_width=True)
else:
    st.info("No stats available to plot.")


# -----------------------------
# Sentence Selector
# -----------------------------
st.subheader("Sentence Selector")

available_idxs = sorted(int(i) for i in df_stats["sentence_idx"].dropna().unique()) if len(df_stats) else []
if not available_idxs:
    st.info("No sentence indices available for selection.")
    st.stop()

selected_sentence_idx = st.selectbox(
    "Sentence index",
    available_idxs,
    format_func=lambda i: sentence_selector_label(i, sentence_span_map, sentence_tags_by_idx),
)

probe_rows = [
    (i, p)
    for i, p in enumerate(history_norm)
    if p.get("sentence_idx") == selected_sentence_idx
    or p.get("sentence_end_idx") == selected_sentence_idx + 1
]
if not probe_rows:
    st.info("No probe found for this sentence end index.")
    st.stop()

probe = probe_rows[0][1]
if len(probe_rows) > 1:
    step_options = [i for i, _ in probe_rows]
    selected_step = st.selectbox("Probe step", step_options, format_func=lambda i: f"step {i}")
    probe = dict(history_norm[selected_step])

resolved_sentence_text, resolved_start, resolved_end = resolve_sentence_span(
    selected_sentence_idx,
    sentence_span_map,
    probe=probe,
)
resolved_sentence_idx = int(selected_sentence_idx)

if sentence_spans and resolved_sentence_idx is not None and resolved_sentence_idx not in sentence_span_map:
    st.warning(
        "Sentence index is missing from the sentence spans for this example. "
        "Your stored results may have been generated with a different splitter."
    )

st.markdown(
    f"Sentence idx: {resolved_sentence_idx} | Deception rate: {probe.get('deception_rate')} | "
    f"Valid samples: {probe.get('num_valid')}"
)

selected_tag = sentence_tags_by_idx.get(resolved_sentence_idx)
if selected_tag:
    tag_label = get_tag_display_label(selected_tag, missing="other")
    tag_conf = selected_tag.get("confidence")
    conf_str = f"{tag_conf:.2f}" if tag_conf is not None and np.isfinite(tag_conf) else "n/a"
    st.markdown(f"Tag label: `{tag_label}` | Confidence: `{conf_str}`")
elif tags_index is None or not has_example_tags:
    st.markdown("Tag label: `no_tag` | Confidence: `n/a` (no tag match for selected example/model)")
else:
    st.markdown("Tag label: `other` | Confidence: `n/a`")

probe_sentence_text = probe.get("sentence_text") or ""

sentence_text = probe_sentence_text or resolved_sentence_text or ""
char_start = resolved_start
char_end = resolved_end

if (not sentence_text) and char_start is None:
    st.info("No sentence text for this probe.")

selected_span = (char_start, char_end) if char_start is not None and char_end is not None else None


# -----------------------------
# Sample Selection
# -----------------------------
st.subheader("Sample Selector")

current_tag_label = "other"
if tags_index is None or not has_example_tags:
    current_tag_label = "no_tag"
if selected_tag:
    current_tag_label = get_tag_display_label(selected_tag, missing="other")
if sentence_text:
    st.markdown(f"Sentence: {sentence_text} [{current_tag_label}]")

if "truthful_sel" not in st.session_state:
    st.session_state.truthful_sel = "None"
if "deceptive_sel" not in st.session_state:
    st.session_state.deceptive_sel = "None"


def on_truthful_change():
    st.session_state.deceptive_sel = "None"


def on_deceptive_change():
    st.session_state.truthful_sel = "None"


subset = df_plot[
    (df_plot["sentence_idx"] == selected_sentence_idx)
    | (df_plot["sentence_end_idx"] == selected_sentence_idx + 1)
]
if subset.empty:
    st.info("No generations available for this probe.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.markdown("Truthful samples")
    truthful = subset[subset["is_truthful"] == True]
    truthful_opts = {
        f"Truthful generation {r.sample_id}": r.sample_id
        for _, r in truthful.iterrows()
    }
    st.selectbox(
        "Select truthful sample",
        ["None"] + list(truthful_opts.keys()),
        key="truthful_sel",
        on_change=on_truthful_change,
    )

with col2:
    st.markdown("Deceptive samples")
    deceptive = subset[subset["is_truthful"] == False]
    deceptive_opts = {
        f"Deceptive generation {r.sample_id}": r.sample_id
        for _, r in deceptive.iterrows()
    }
    st.selectbox(
        "Select deceptive sample",
        ["None"] + list(deceptive_opts.keys()),
        key="deceptive_sel",
        on_change=on_deceptive_change,
    )


# -----------------------------
# Generation Viewer
# -----------------------------
st.subheader("Generation Viewer")

selected_sample_id = None
if st.session_state.truthful_sel != "None":
    selected_sample_id = truthful_opts[st.session_state.truthful_sel]
elif st.session_state.deceptive_sel != "None":
    selected_sample_id = deceptive_opts[st.session_state.deceptive_sel]

if selected_sample_id is None:
    st.info("Select a sample above to view the generation.")
    st.stop()

row = subset[subset["sample_id"] == selected_sample_id].iloc[0]
gen_text = strip_code_fences(row.get("gen_text") or "")

prefix_mode = st.radio(
    "Prefix view",
    ["Full prefix to sentence end", "Sentence only"],
    index=0,
    help="Choose whether the prefix includes all text up to the sentence end, or only the sentence itself.",
)

if prefix_mode == "Sentence only":
    prefix_text = sentence_text or ""
    # In sentence-only mode, align the right edge of the prefix to the selected sentence idx.
    prefix_global_start_idx = resolved_sentence_idx if prefix_text else None
else:
    if char_end is not None:
        prefix_text = raw_text[:char_end]
    else:
        prefix_text = sentence_text or ""
    # Full-prefix mode starts from sentence 0.
    prefix_global_start_idx = 0 if prefix_text else None

# Label only the last sentence in the rendered prefix so we never tag green generation text.
prefix_spans_for_render = [
    s for s in split_sentence_spans(prefix_text)
    if s.get("start") is not None and s.get("end") is not None
]
prefix_spans_for_render.sort(key=lambda s: s.get("start", 0))
display_selected_idx = (len(prefix_spans_for_render) - 1) if prefix_spans_for_render else None

generation_sentence_labels: Dict[int, str] = {}
if prefix_spans_for_render and prefix_global_start_idx is not None:
    # Map every prefix-local sentence idx to a global sentence idx, then attach labels.
    # For sentence-only mode with splitter drift, anchor the last local sentence to resolved idx.
    if prefix_mode == "Sentence only":
        global_offset = int(resolved_sentence_idx) - (len(prefix_spans_for_render) - 1)
    else:
        global_offset = int(prefix_global_start_idx)

    for local_idx in range(len(prefix_spans_for_render)):
        global_idx = global_offset + local_idx
        tag_payload = sentence_tags_by_idx.get(global_idx)
        if tag_payload:
            generation_sentence_labels[local_idx] = get_tag_display_label(tag_payload, missing="other")
        elif tags_index is None or not has_example_tags:
            generation_sentence_labels[local_idx] = "no_tag"
        else:
            generation_sentence_labels[local_idx] = "other"

st.markdown(
    f"Action: {row.get('action')} | Truthful: {row.get('is_truthful')} | Parse error: {row.get('parse_error')}"
)
st.markdown(f"Sentence of interest: `S_{resolved_sentence_idx + 1} [{current_tag_label}]`")

render_prefix_generation_with_sentence_indices(
    prefix_text,
    gen_text,
    display_selected_idx,
    sentence_labels=generation_sentence_labels,
)
