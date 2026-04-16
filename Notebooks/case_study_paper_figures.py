#!/usr/bin/env python3
from __future__ import annotations

import gc
import math
import os
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for the paper-figure generator.") from exc


NOTEBOOK_ROOT = Path("/playpen-ssd/smerrill/deception2/Notebooks")
CASE_STUDY_NOTEBOOK_PATH = NOTEBOOK_ROOT / "bs_gpt_oss_20b_case_study.ipynb"
OUTPUT_ROOT = NOTEBOOK_ROOT / "bs_gpt_oss_20b_paper_figures_outputs"
BS_SPIKE_SUMMARY_PATH = NOTEBOOK_ROOT / "bs_feature_motivation_outputs__top12" / "bs_spike_vs_previous_feature_summary.csv"
BS_PREV3_BRIDGE_CACHE_PATH = OUTPUT_ROOT / "bs_prev3_bridge_summary.csv"
EXEC_CELL_INDICES = [2, 4, 6, 8, 10]
DEFAULT_POSTERIOR_SAMPLES = 4000
DEFAULT_POSTERIOR_SEED = 13

ENV_NAME = "bs"
DEFAULT_CHERRY_PICK = {
    "example_id": "2026-02-06/gpu_2/state_809/sample_0",
    "spike_boundary_id": "bs::2026-02-06/gpu_2/state_809/sample_0::p24",
    "prior_boundary_id": "bs::2026-02-06/gpu_2/state_809/sample_0::p23",
}

PANEL_COLORS = {
    "prior": "#4e79a7",
    "spike": "#d1495b",
    "neutral": "#5f6368",
}

TEXT_SIZES = {
    "figure_title": 15,
    "axes_title": 13,
    "axes_label": 12,
    "tick": 11,
    "legend": 10,
    "annotation": 9.5,
    "sentence_title": 10.2,
    "sentence_title_small": 9.8,
    "sentence_title_tight": 9.4,
    "sentence_body": 9.5,
    "sentence_body_small": 9.0,
    "sentence_body_tight": 8.5,
}

SENTENCE_BOX_LAYOUT_CANDIDATES = [
    {
        "wrap_width": 70,
        "body_fontsize": TEXT_SIZES["sentence_body"],
        "title_fontsize": TEXT_SIZES["sentence_title"],
    },
    {
        "wrap_width": 70,
        "body_fontsize": TEXT_SIZES["sentence_body_small"],
        "title_fontsize": TEXT_SIZES["sentence_title_small"],
    },
    {
        "wrap_width": 70,
        "body_fontsize": TEXT_SIZES["sentence_body_tight"],
        "title_fontsize": TEXT_SIZES["sentence_title_tight"],
    },
    {
        "wrap_width": 70,
        "body_fontsize": 7.8,
        "title_fontsize": 8.8,
    },
]

FIGURE_SIZES = {
    "text_plus_plot": (10.8, 5.5),
    "three_panel": (15.8, 5.4),
    "two_by_two": (13.8, 8.6),
    "two_panel": (13.8, 5.4),
    "aggregate_two_panel": (14.0, 5.8),
}

GRID_ALPHA = 0.22

LEGEND_STYLE = {
    "frameon": True,
    "facecolor": "white",
    "edgecolor": "#dddddd",
}

LAYOUT_SPECS = {
    "context": {"method": "tight_layout", "rect": [0, 0, 1, 0.98]},
    "three_panel": {
        "method": "subplots_adjust",
        "top": 0.84,
        "bottom": 0.13,
        "left": 0.08,
        "right": 0.985,
        "wspace": 0.24,
    },
    "two_by_two": {"method": "tight_layout", "rect": [0, 0.04, 1, 0.96]},
    "two_panel": {
        "method": "subplots_adjust",
        "top": 0.84,
        "bottom": 0.14,
        "left": 0.08,
        "right": 0.985,
        "wspace": 0.18,
    },
    "aggregate_two_panel": {
        "method": "subplots_adjust",
        "top": 0.84,
        "bottom": 0.14,
        "left": 0.08,
        "right": 0.985,
        "wspace": 0.22,
    },
}

_CACHED_CASE_STUDY_NAMESPACE: dict | None = None


def apply_paper_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.family": "DejaVu Sans",
            "axes.titlesize": TEXT_SIZES["axes_title"],
            "axes.titleweight": "semibold",
            "axes.labelsize": TEXT_SIZES["axes_label"],
            "xtick.labelsize": TEXT_SIZES["tick"],
            "ytick.labelsize": TEXT_SIZES["tick"],
            "legend.fontsize": TEXT_SIZES["legend"],
            "figure.titlesize": TEXT_SIZES["figure_title"],
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_case_study_namespace(*, force_reload: bool = False) -> dict:
    global _CACHED_CASE_STUDY_NAMESPACE
    if _CACHED_CASE_STUDY_NAMESPACE is not None and not force_reload:
        return _CACHED_CASE_STUDY_NAMESPACE

    notebook = nbformat.read(CASE_STUDY_NOTEBOOK_PATH, as_version=4)
    namespace: dict = {}
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        from pandas.errors import PerformanceWarning

        warnings.filterwarnings("ignore", category=PerformanceWarning)
    except Exception:
        pass
    for cell_idx in EXEC_CELL_INDICES:
        exec(compile(notebook.cells[cell_idx].source, f"cell-{cell_idx}", "exec"), namespace)
    _CACHED_CASE_STUDY_NAMESPACE = namespace
    return namespace


def reset_runtime_cache(*, drop_model: bool = False, drop_namespace: bool = False) -> None:
    global _CACHED_CASE_STUDY_NAMESPACE
    namespace = _CACHED_CASE_STUDY_NAMESPACE
    if namespace is not None:
        cleanup_fn = namespace.get("cleanup_cached_example")
        if callable(cleanup_fn):
            cleanup_fn()
        if drop_model:
            reset_fn = namespace.get("reset_shared_model")
            if callable(reset_fn):
                reset_fn()
    if drop_model or drop_namespace:
        _CACHED_CASE_STUDY_NAMESPACE = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _preview_text(text: str, *, width: int = 84) -> str:
    clean = " ".join(str(text).split())
    return textwrap.fill(clean, width=width)


def _safe_late_start(num_layers: int, late_layers: int = 8) -> int:
    return max(0, int(num_layers) - int(late_layers))


def _feature_curve(feature_view: dict, stem: str) -> np.ndarray:
    num_layers = int(feature_view["num_layers"])
    row = feature_view["selected_feature_row"]
    return np.asarray([float(row.get(f"{stem}_l{layer_idx}", np.nan)) for layer_idx in range(num_layers)], dtype=float)


def _late_mean(values: np.ndarray, *, late_start: int) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr[late_start:]))


def _sentence_lookup(sentence_df: pd.DataFrame) -> dict[int, pd.Series]:
    ordered_df = sentence_df.sort_values("sentence_idx").reset_index(drop=True)
    return {int(row["sentence_idx"]): row for _, row in ordered_df.iterrows()}


def _activation_curve(activation_df: pd.DataFrame, stem: str) -> np.ndarray:
    return pd.to_numeric(activation_df[stem], errors="coerce").to_numpy(dtype=float)


def _safe_nan_summary(values: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmean(finite)), float(np.nanstd(finite, ddof=1)) if finite.size > 1 else 0.0, float(np.nanmax(finite))


def _commitment_box_title(row) -> str:
    return f"S{int(row.sentence_idx)} | counterfactual deception rate={float(row.deception_rate):.3f}"


def _text_bbox_fits(box_bbox: Bbox, text_bbox: Bbox, *, inset_px: float = 6.0) -> bool:
    return (
        text_bbox.x0 >= box_bbox.x0 + inset_px
        and text_bbox.x1 <= box_bbox.x1 - inset_px
        and text_bbox.y0 >= box_bbox.y0 + inset_px
        and text_bbox.y1 <= box_bbox.y1 - inset_px
    )


def _fit_commitment_context_boxes(
    fig,
    ax_text,
    context_rows,
    sentence_lookup: dict[int, pd.Series],
    *,
    box_x: float,
    box_w: float,
    top_margin: float,
    bottom_margin: float,
    gap: float,
) -> tuple[list[tuple], float, float, float]:
    n_blocks = len(context_rows)
    block_height = (top_margin - bottom_margin - gap * (n_blocks - 1)) / n_blocks
    fig.canvas.draw()

    fallback_result: tuple[list[tuple], float, float, float] | None = None
    for candidate in SENTENCE_BOX_LAYOUT_CANDIDATES:
        wrapped_blocks = []
        temp_artists = []
        measured_boxes = []

        for i, row in enumerate(context_rows):
            sentence_idx = int(row.sentence_idx)
            wrapped_body = _preview_text(
                str(sentence_lookup[sentence_idx]["sentence_text"]),
                width=int(candidate["wrap_width"]),
            )
            wrapped_blocks.append((row, wrapped_body))

            y_top = top_margin - i * (block_height + gap)
            y = y_top - block_height
            title = _commitment_box_title(row)

            title_artist = ax_text.text(
                box_x + 0.02,
                y + block_height - 0.03,
                title,
                transform=ax_text.transAxes,
                fontsize=float(candidate["title_fontsize"]),
                fontweight="semibold",
                color="#1f1f1f",
                va="top",
                ha="left",
                alpha=0.0,
            )
            body_artist = ax_text.text(
                box_x + 0.02,
                y + block_height - 0.085,
                wrapped_body,
                transform=ax_text.transAxes,
                fontsize=float(candidate["body_fontsize"]),
                color="#333333",
                va="top",
                ha="left",
                linespacing=1.25,
                alpha=0.0,
            )
            temp_artists.extend([title_artist, body_artist])
            measured_boxes.append(
                (
                    ax_text.transAxes.transform_bbox(Bbox.from_bounds(box_x, y, box_w, block_height)),
                    title_artist,
                    body_artist,
                )
            )

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fits_all_boxes = True
        for box_bbox, title_artist, body_artist in measured_boxes:
            title_bbox = title_artist.get_window_extent(renderer=renderer)
            body_bbox = body_artist.get_window_extent(renderer=renderer)
            if not _text_bbox_fits(box_bbox, title_bbox):
                fits_all_boxes = False
                break
            if not _text_bbox_fits(box_bbox, body_bbox):
                fits_all_boxes = False
                break
            if body_bbox.y1 >= title_bbox.y0 - 6.0:
                fits_all_boxes = False
                break

        for artist in temp_artists:
            artist.remove()

        fallback_result = (
            wrapped_blocks,
            float(candidate["body_fontsize"]),
            float(candidate["title_fontsize"]),
            block_height,
        )
        if fits_all_boxes:
            return fallback_result

    if fallback_result is None:
        raise RuntimeError("Could not build sentence-box layout candidates for the commitment context figure.")
    return fallback_result


def style_axis(
    ax,
    *,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    grid_axis: str = "both",
    title_pad: float = 8.0,
) -> None:
    ax.set_title(title, pad=title_pad)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(axis=grid_axis, alpha=GRID_ALPHA)


def style_legend(ax, *, loc: str) -> None:
    ax.legend(loc=loc, **LEGEND_STYLE)


def finalize_figure(fig, layout_key: str) -> None:
    layout_spec = LAYOUT_SPECS[layout_key].copy()
    method = layout_spec.pop("method")
    if method == "tight_layout":
        fig.tight_layout(**layout_spec)
        return
    if method == "subplots_adjust":
        fig.subplots_adjust(**layout_spec)
        return
    raise ValueError(f"Unknown layout method: {method}")


def build_current_vs_prev3_attention_curve_df(base_bundle: dict, boundary_row: pd.Series) -> pd.DataFrame:
    sentence_df = base_bundle["sentence_df"].sort_values("sentence_idx").reset_index(drop=True)
    selected_sentence_idx = int(boundary_row["sentence_idx"])
    selected_matches = sentence_df.loc[sentence_df["sentence_idx"].eq(selected_sentence_idx)]
    if selected_matches.empty:
        raise KeyError(f"Could not find sentence_idx={selected_sentence_idx} in the example sentence table.")

    selected_row = selected_matches.iloc[0]
    selected_position = int(selected_matches.index[0])
    prev3_rows = sentence_df.iloc[max(0, selected_position - 3) : selected_position].copy().reset_index(drop=True)
    if prev3_rows.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "current_vs_prev3_mean",
                "current_vs_prev3_std",
                "current_vs_prev3_max",
                "selected_sentence_idx",
                "prev3_sentence_count",
                "prev3_token_count",
            ]
        )

    q_idx = int(selected_row["end_token"])
    current_start = int(selected_row["start_token"])
    current_end = int(selected_row["end_token"])
    prev3_start = int(prev3_rows.iloc[0]["start_token"])
    prev3_end = int(prev3_rows.iloc[-1]["end_token"])
    prev3_token_count = int(prev3_end - prev3_start + 1)

    rows = []
    for layer_idx, layer_attn in enumerate(base_bundle["attentions"]):
        query_attn = layer_attn[0, :, q_idx, : current_end + 1].detach().to(torch.float32)
        current_mass = query_attn[:, current_start : current_end + 1].sum(dim=1)
        prev3_mass = query_attn[:, prev3_start : prev3_end + 1].sum(dim=1)
        total_mass = current_mass + prev3_mass
        share = torch.full_like(current_mass, float("nan"), dtype=torch.float32)
        valid = total_mass > 0
        share[valid] = torch.clamp(current_mass[valid] / total_mass[valid], min=0.0, max=1.0)
        mean_value, std_value, max_value = _safe_nan_summary(share.detach().cpu().numpy())
        rows.append(
            {
                "layer": int(layer_idx),
                "current_vs_prev3_mean": mean_value,
                "current_vs_prev3_std": std_value,
                "current_vs_prev3_max": max_value,
                "selected_sentence_idx": selected_sentence_idx,
                "prev3_sentence_count": int(len(prev3_rows)),
                "prev3_token_count": prev3_token_count,
            }
        )
    return pd.DataFrame(rows)


def build_mean_prior_sentence_activation_similarity_df(base_bundle: dict, boundary_row: pd.Series) -> pd.DataFrame:
    sentence_df = base_bundle["sentence_df"].sort_values("sentence_idx").reset_index(drop=True)
    hidden_states = base_bundle["hidden_states"]
    selected_sentence_idx = int(boundary_row["sentence_idx"])
    selected_matches = sentence_df.loc[sentence_df["sentence_idx"].eq(selected_sentence_idx)]
    if selected_matches.empty:
        raise KeyError(f"Could not find sentence_idx={selected_sentence_idx} in the example sentence table.")

    selected_row = selected_matches.iloc[0]
    previous_rows = sentence_df.loc[sentence_df["sentence_idx"].lt(selected_sentence_idx)].copy()
    if previous_rows.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "mean_cosine_similarity",
                "std_cosine_similarity",
                "sem_cosine_similarity",
                "num_prior_sentences",
                "selected_sentence_idx",
            ]
        )

    selected_end_token = int(selected_row["end_token"])
    prior_end_tokens = [int(value) for value in previous_rows["end_token"].tolist()]

    rows = []
    for layer_idx, hidden_state in enumerate(hidden_states):
        layer_hidden = hidden_state[0]
        selected_vec = layer_hidden[selected_end_token].detach().to(torch.float32)
        token_index_tensor = torch.tensor(prior_end_tokens, device=layer_hidden.device, dtype=torch.long)
        prior_matrix = layer_hidden.index_select(0, token_index_tensor).detach().to(torch.float32)

        selected_norm = selected_vec.norm()
        prior_norms = prior_matrix.norm(dim=1)
        denom = prior_norms * selected_norm
        valid_mask = torch.isfinite(denom) & denom.gt(0)
        valid_count = int(valid_mask.sum().item())

        if valid_count > 0:
            valid_prior = prior_matrix[valid_mask]
            valid_denom = denom[valid_mask]
            cosine_values = (valid_prior @ selected_vec) / valid_denom
            mean_value = float(cosine_values.mean().item())
            if valid_count > 1:
                std_value = float(cosine_values.std(unbiased=True).item())
                sem_value = float(std_value / math.sqrt(valid_count))
            else:
                std_value = 0.0
                sem_value = 0.0
        else:
            mean_value = float("nan")
            std_value = float("nan")
            sem_value = float("nan")

        rows.append(
            {
                "layer": int(layer_idx),
                "mean_cosine_similarity": mean_value,
                "std_cosine_similarity": std_value,
                "sem_cosine_similarity": sem_value,
                "num_prior_sentences": valid_count,
                "selected_sentence_idx": selected_sentence_idx,
            }
        )

    return pd.DataFrame(rows)


def build_final_layer_prior_sentence_similarity_df(base_bundle: dict, boundary_row: pd.Series) -> pd.DataFrame:
    sentence_df = base_bundle["sentence_df"].sort_values("sentence_idx").reset_index(drop=True)
    hidden_states = base_bundle["hidden_states"]
    if not hidden_states:
        return pd.DataFrame(
            columns=[
                "layer",
                "selected_sentence_idx",
                "prior_sentence_idx",
                "sentence_distance",
                "cosine_similarity",
                "prior_sentence_text",
            ]
        )

    selected_sentence_idx = int(boundary_row["sentence_idx"])
    selected_matches = sentence_df.loc[sentence_df["sentence_idx"].eq(selected_sentence_idx)]
    if selected_matches.empty:
        raise KeyError(f"Could not find sentence_idx={selected_sentence_idx} in the example sentence table.")

    selected_row = selected_matches.iloc[0]
    previous_rows = sentence_df.loc[sentence_df["sentence_idx"].lt(selected_sentence_idx)].copy()
    if previous_rows.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "selected_sentence_idx",
                "prior_sentence_idx",
                "sentence_distance",
                "cosine_similarity",
                "prior_sentence_text",
            ]
        )

    final_layer_idx = int(len(hidden_states) - 1)
    final_hidden = hidden_states[final_layer_idx][0]
    selected_vec = final_hidden[int(selected_row["end_token"])].detach().to(torch.float32)
    selected_norm = selected_vec.norm()

    rows = []
    for prior_row in previous_rows.itertuples():
        prior_vec = final_hidden[int(prior_row.end_token)].detach().to(torch.float32)
        denom = selected_norm * prior_vec.norm()
        if float(denom.item()) > 0:
            cosine_value = float(torch.dot(selected_vec, prior_vec).item() / denom.item())
        else:
            cosine_value = float("nan")
        rows.append(
            {
                "layer": final_layer_idx,
                "selected_sentence_idx": selected_sentence_idx,
                "prior_sentence_idx": int(prior_row.sentence_idx),
                "sentence_distance": int(selected_sentence_idx - int(prior_row.sentence_idx)),
                "cosine_similarity": cosine_value,
                "prior_sentence_text": str(prior_row.sentence_text),
            }
        )
    return pd.DataFrame(rows)


def _resolve_prior_boundary_id(boundary_lookup: pd.DataFrame, *, example_id: str, spike_boundary_id: str) -> str:
    spike_row = boundary_lookup.loc[spike_boundary_id]
    example_rows = boundary_lookup.loc[boundary_lookup["example_id"].eq(example_id)]
    if example_rows.empty:
        raise KeyError(f"Could not find any boundaries for example_id={example_id!r}")
    prior_match = example_rows.loc[example_rows["prefix_idx"].eq(int(spike_row["prev_prefix_idx"]))]
    if prior_match.empty:
        raise KeyError(
            f"Could not infer prior boundary for example_id={example_id!r} and spike_boundary_id={spike_boundary_id!r}"
        )
    return str(prior_match.index[0])


def load_case(
    *,
    example_id: str,
    spike_boundary_id: str,
    prior_boundary_id: str | None = None,
    env_name: str = ENV_NAME,
) -> dict:
    apply_paper_style()
    namespace = load_case_study_namespace()
    requested_device = os.environ.get("BS_PAPER_FIGURES_DEVICE")
    if requested_device:
        namespace["MODEL_DEVICE"] = str(requested_device)
    env_state = namespace["get_env_state"](env_name)
    try:
        base_bundle = namespace["get_example_analysis_bundle"](env_name, example_id)
    except torch.OutOfMemoryError:
        if str(namespace.get("MODEL_DEVICE", "")).lower() == "cpu":
            raise
        if "reset_shared_model" in namespace:
            namespace["reset_shared_model"]()
        namespace["MODEL_DEVICE"] = "cpu"
        base_bundle = namespace["get_example_analysis_bundle"](env_name, example_id)
    boundary_lookup = env_state["boundary_lookup_df"]
    spike_row = boundary_lookup.loc[spike_boundary_id]
    if prior_boundary_id is None:
        prior_boundary_id = _resolve_prior_boundary_id(
            boundary_lookup,
            example_id=example_id,
            spike_boundary_id=spike_boundary_id,
        )
    prior_row = boundary_lookup.loc[prior_boundary_id]

    spike_attention_view = namespace["build_attention_views"](
        base_bundle,
        spike_row,
        query_mode="last_token_selected_sentence",
    )
    prior_attention_view = namespace["build_attention_views"](
        base_bundle,
        prior_row,
        query_mode="last_token_selected_sentence",
    )
    spike_feature_view = namespace["build_feature_metric_views"](base_bundle, spike_row)
    prior_feature_view = namespace["build_feature_metric_views"](base_bundle, prior_row)
    spike_activation_df = namespace["build_sentence_end_activation_curve_df"](base_bundle, spike_row)
    prior_activation_df = namespace["build_sentence_end_activation_curve_df"](base_bundle, prior_row)
    spike_prev3_attention_df = build_current_vs_prev3_attention_curve_df(base_bundle, spike_row)
    prior_prev3_attention_df = build_current_vs_prev3_attention_curve_df(base_bundle, prior_row)
    spike_activation_similarity_df = build_mean_prior_sentence_activation_similarity_df(base_bundle, spike_row)
    prior_activation_similarity_df = build_mean_prior_sentence_activation_similarity_df(base_bundle, prior_row)
    spike_final_layer_similarity_df = build_final_layer_prior_sentence_similarity_df(base_bundle, spike_row)
    prior_final_layer_similarity_df = build_final_layer_prior_sentence_similarity_df(base_bundle, prior_row)

    sentence_df = base_bundle["sentence_df"].sort_values("sentence_idx").reset_index(drop=True).copy()
    local_context_df = namespace["build_context_window_df"](sentence_df, int(spike_row["sentence_idx"])).copy()

    late_start = _safe_late_start(int(base_bundle["num_layers"]))

    case = {
        "namespace": namespace,
        "env_name": env_name,
        "example_id": example_id,
        "base_bundle": base_bundle,
        "sentence_df": sentence_df,
        "sentence_lookup": _sentence_lookup(sentence_df),
        "spike_row": spike_row,
        "prior_row": prior_row,
        "spike_attention_view": spike_attention_view,
        "prior_attention_view": prior_attention_view,
        "spike_feature_view": spike_feature_view,
        "prior_feature_view": prior_feature_view,
        "spike_activation_df": spike_activation_df,
        "prior_activation_df": prior_activation_df,
        "spike_prev3_attention_df": spike_prev3_attention_df,
        "prior_prev3_attention_df": prior_prev3_attention_df,
        "spike_activation_similarity_df": spike_activation_similarity_df,
        "prior_activation_similarity_df": prior_activation_similarity_df,
        "spike_final_layer_similarity_df": spike_final_layer_similarity_df,
        "prior_final_layer_similarity_df": prior_final_layer_similarity_df,
        "local_context_df": local_context_df,
        "num_layers": int(base_bundle["num_layers"]),
        "late_start": late_start,
    }
    case["sentence_table"] = build_sentence_table(case)
    case["late_metric_summary"] = build_late_metric_summary(case)
    return case


def load_cherry_picked_case() -> dict:
    return load_case(**DEFAULT_CHERRY_PICK)


def build_sentence_table(case: dict) -> pd.DataFrame:
    sentence_df = case["sentence_df"]
    spike_sentence_idx = int(case["spike_row"]["sentence_idx"])
    prior_sentence_idx = int(case["prior_row"]["sentence_idx"])
    window_df = sentence_df.loc[
        sentence_df["sentence_idx"].between(spike_sentence_idx - 2, spike_sentence_idx + 1)
    ].copy()
    if window_df.empty:
        return pd.DataFrame()
    role_map = {
        spike_sentence_idx - 2: "sentence i-2",
        prior_sentence_idx: "sentence i-1",
        spike_sentence_idx: "sentence i",
        spike_sentence_idx + 1: "sentence i+1",
    }
    window_df["role"] = window_df["sentence_idx"].map(role_map).fillna("context")
    window_df["deception_rate"] = pd.to_numeric(window_df["deception_rate"], errors="coerce")
    return window_df[["role", "sentence_idx", "deception_rate", "sentence_text"]].reset_index(drop=True)


def build_late_metric_summary(case: dict) -> pd.DataFrame:
    late_start = int(case["late_start"])
    metric_specs = [
        ("current_vs_prev3_mean", "current vs previous 3 (attention)", "prev3_attention"),
        ("current_vs_prev_mean", "current vs previous", "feature"),
        ("delta_current_vs_prev_mean", "delta current vs previous", "feature"),
        ("cos_cur_mean3", "cos(current, mean prev 3)", "activation"),
        ("mean_cosine_similarity", "cos(current, all previous sentences)", "activation_similarity"),
        ("entropy_prior_mean", "prior entropy", "feature"),
        ("delta_entropy_prior_mean", "prior delta entropy", "feature"),
        ("top5_prior_mean", "prior top-5 share", "feature"),
        ("herfindahl_prior_mean", "prior herfindahl", "feature"),
    ]
    rows = []
    for stem, label, source in metric_specs:
        if source == "feature":
            prior_values = _feature_curve(case["prior_feature_view"], stem)
            spike_values = _feature_curve(case["spike_feature_view"], stem)
        elif source == "prev3_attention":
            prior_values = pd.to_numeric(
                case["prior_prev3_attention_df"][stem], errors="coerce"
            ).to_numpy(dtype=float)
            spike_values = pd.to_numeric(
                case["spike_prev3_attention_df"][stem], errors="coerce"
            ).to_numpy(dtype=float)
        elif source == "activation":
            prior_values = pd.to_numeric(case["prior_activation_df"][stem], errors="coerce").to_numpy(dtype=float)
            spike_values = pd.to_numeric(case["spike_activation_df"][stem], errors="coerce").to_numpy(dtype=float)
        else:
            prior_values = pd.to_numeric(
                case["prior_activation_similarity_df"][stem], errors="coerce"
            ).to_numpy(dtype=float)
            spike_values = pd.to_numeric(
                case["spike_activation_similarity_df"][stem], errors="coerce"
            ).to_numpy(dtype=float)
        prior_late = _late_mean(prior_values, late_start=late_start)
        spike_late = _late_mean(spike_values, late_start=late_start)
        rows.append(
            {
                "metric": label,
                "prior_late_mean": prior_late,
                "spike_late_mean": spike_late,
                "spike_minus_prior": float(spike_late - prior_late),
            }
        )
    return pd.DataFrame(rows)


def load_bs_spike_activation_summary() -> pd.DataFrame:
    if not BS_SPIKE_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing BS activation summary CSV: {BS_SPIKE_SUMMARY_PATH}")
    summary_df = pd.read_csv(BS_SPIKE_SUMMARY_PATH).copy()
    required_columns = [
        "example_id",
        "spike_boundary_id",
        "prior_boundary_id",
    ]
    missing = [column for column in required_columns if column not in summary_df.columns]
    if missing:
        raise KeyError(f"Missing columns in {BS_SPIKE_SUMMARY_PATH}: {missing}")
    return summary_df


def build_bs_prev3_bridge_summary(*, cache_path: Path = BS_PREV3_BRIDGE_CACHE_PATH, force_recompute: bool = False) -> pd.DataFrame:
    required_columns = [
        "example_id",
        "spike_boundary_id",
        "prior_boundary_id",
        "prior_attention_value",
        "spike_attention_value",
        "diff_attention_value",
        "prior_activation_value",
        "spike_activation_value",
        "diff_activation_value",
    ]
    if cache_path.exists() and not force_recompute:
        cached_df = pd.read_csv(cache_path)
        if all(column in cached_df.columns for column in required_columns):
            return cached_df

    source_df = load_bs_spike_activation_summary()[
        ["example_id", "spike_boundary_id", "prior_boundary_id"]
    ].drop_duplicates().reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for row in source_df.itertuples():
        case = load_case(
            example_id=str(row.example_id),
            spike_boundary_id=str(row.spike_boundary_id),
            prior_boundary_id=str(row.prior_boundary_id),
        )
        late_start = int(case["late_start"])
        prior_attention = pd.to_numeric(
            case["prior_prev3_attention_df"]["current_vs_prev3_mean"], errors="coerce"
        ).to_numpy(dtype=float)
        spike_attention = pd.to_numeric(
            case["spike_prev3_attention_df"]["current_vs_prev3_mean"], errors="coerce"
        ).to_numpy(dtype=float)
        prior_activation = _activation_curve(case["prior_activation_df"], "cos_cur_mean3")
        spike_activation = _activation_curve(case["spike_activation_df"], "cos_cur_mean3")
        prior_attention_late = _late_mean(prior_attention, late_start=late_start)
        spike_attention_late = _late_mean(spike_attention, late_start=late_start)
        prior_activation_late = _late_mean(prior_activation, late_start=late_start)
        spike_activation_late = _late_mean(spike_activation, late_start=late_start)
        rows.append(
            {
                "example_id": str(row.example_id),
                "spike_boundary_id": str(row.spike_boundary_id),
                "prior_boundary_id": str(row.prior_boundary_id),
                "prior_attention_value": prior_attention_late,
                "spike_attention_value": spike_attention_late,
                "diff_attention_value": float(spike_attention_late - prior_attention_late),
                "prior_activation_value": prior_activation_late,
                "spike_activation_value": spike_activation_late,
                "diff_activation_value": float(spike_activation_late - prior_activation_late),
            }
        )

    summary_df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(cache_path, index=False)
    return summary_df


def _beta_rate_draws(
    rate: float,
    num_valid: float,
    *,
    rng: np.random.Generator,
    num_samples: int = DEFAULT_POSTERIOR_SAMPLES,
) -> np.ndarray:
    safe_rate = float(np.clip(float(rate), 0.0, 1.0))
    if not np.isfinite(num_valid) or float(num_valid) <= 0:
        return np.full(int(num_samples), safe_rate, dtype=float)
    safe_n = max(1, int(round(float(num_valid))))
    success_count = int(round(safe_rate * safe_n))
    success_count = max(0, min(success_count, safe_n))
    return rng.beta(1.0 + success_count, 1.0 + safe_n - success_count, size=int(num_samples))


def build_counterfactual_uncertainty_df(
    case: dict,
    *,
    num_samples: int = DEFAULT_POSTERIOR_SAMPLES,
    seed: int = DEFAULT_POSTERIOR_SEED,
) -> pd.DataFrame:
    sentence_df = case["sentence_df"].sort_values("sentence_idx").reset_index(drop=True).copy()
    sentence_df["deception_rate"] = pd.to_numeric(sentence_df["deception_rate"], errors="coerce")
    sentence_df["num_valid"] = pd.to_numeric(sentence_df["num_valid"], errors="coerce")
    spike_sentence_idx = int(case["spike_row"]["sentence_idx"])
    plot_df = sentence_df.loc[
        sentence_df["sentence_idx"].between(spike_sentence_idx - 2, spike_sentence_idx + 1)
    ].copy()
    if plot_df.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(int(seed))
    rate_draws = {}
    rate_se = {}
    for row in sentence_df.itertuples():
        draws = _beta_rate_draws(
            float(row.deception_rate),
            float(row.num_valid) if np.isfinite(row.num_valid) else float("nan"),
            rng=rng,
            num_samples=num_samples,
        )
        rate_draws[int(row.sentence_idx)] = draws
        rate_se[int(row.sentence_idx)] = float(np.std(draws, ddof=1))

    records = []
    for row in plot_df.itertuples():
        sentence_idx = int(row.sentence_idx)
        previous_rows = sentence_df.loc[sentence_df["sentence_idx"].lt(sentence_idx)]
        if previous_rows.empty:
            delta_value = float("nan")
            delta_se = float("nan")
        else:
            prev_sentence_idx = int(previous_rows.iloc[-1]["sentence_idx"])
            delta_value = float(row.deception_rate - previous_rows.iloc[-1]["deception_rate"])
            delta_draws = rate_draws[sentence_idx] - rate_draws[prev_sentence_idx]
            delta_se = float(np.std(delta_draws, ddof=1))
        relative_position = sentence_idx - spike_sentence_idx
        relative_label = {
            -2: "sentence i-2",
            -1: "sentence i-1",
            0: "sentence i",
            1: "sentence i+1",
        }.get(relative_position, f"S{sentence_idx}")
        records.append(
            {
                "sentence_idx": sentence_idx,
                "relative_position": relative_position,
                "relative_label": relative_label,
                "deception_rate": float(row.deception_rate),
                "deception_rate_se": rate_se.get(sentence_idx, float("nan")),
                "delta_counterfactual_deception": delta_value,
                "delta_counterfactual_deception_se": delta_se,
                "sentence_text": row.sentence_text,
            }
        )
    return pd.DataFrame(records).sort_values("sentence_idx").reset_index(drop=True)


def build_absolute_sentence_attention_df(case: dict, *, boundary_key: str) -> pd.DataFrame:
    if boundary_key not in {"prior", "spike"}:
        raise KeyError(f"Unknown boundary key: {boundary_key}")
    boundary_row = case["prior_row"] if boundary_key == "prior" else case["spike_row"]
    base_bundle = case["base_bundle"]
    namespace = case["namespace"]
    query_view = namespace["resolve_query_token"](base_bundle, boundary_row, query_mode="last_token_selected_sentence")
    query_token_idx = int(query_view["query_token_idx"])
    selected_sentence_idx = int(boundary_row["sentence_idx"])

    sentence_df = case["sentence_df"].sort_values("sentence_idx").reset_index(drop=True)
    current_match = sentence_df.loc[sentence_df["sentence_idx"].eq(selected_sentence_idx)]
    if current_match.empty:
        return pd.DataFrame()
    current_row = current_match.iloc[0]

    previous_rows = sentence_df.loc[sentence_df["sentence_idx"].lt(selected_sentence_idx)]
    previous_row = previous_rows.iloc[-1] if not previous_rows.empty else None

    role_specs = [
        {
            "row_key": "current_sentence",
            "sentence_label": "current\nsentence",
            "sentence_idx": int(current_row["sentence_idx"]),
            "start_token": int(current_row["start_token"]),
            "end_token": int(current_row["end_token"]),
            "display_order": 0,
        }
    ]

    if previous_row is not None:
        role_specs.append(
            {
                "row_key": "previous_sentence",
                "sentence_label": "previous\nsentence",
                "sentence_idx": int(previous_row["sentence_idx"]),
                "start_token": int(previous_row["start_token"]),
                "end_token": int(previous_row["end_token"]),
                "display_order": 1,
            }
        )
        prior_end = int(previous_row["start_token"]) - 1
    else:
        role_specs.append(
            {
                "row_key": "previous_sentence",
                "sentence_label": "previous\nsentence",
                "sentence_idx": -1,
                "start_token": None,
                "end_token": None,
                "display_order": 1,
            }
        )
        prior_end = int(current_row["start_token"]) - 1

    role_specs.append(
        {
            "row_key": "all_prior_context",
            "sentence_label": "all prior\ncontext",
            "sentence_idx": -1,
            "start_token": 0,
            "end_token": prior_end,
            "display_order": 2,
        }
    )

    attention_rows = []
    for layer_idx, layer_attn in enumerate(base_bundle["attentions"]):
        query_attn = layer_attn[0, :, query_token_idx, :].detach().to(torch.float32)
        for spec in role_specs:
            start_token = spec["start_token"]
            end_token = spec["end_token"]
            available = (
                start_token is not None
                and end_token is not None
                and int(end_token) >= int(start_token)
                and int(end_token) <= query_token_idx
            )
            if available:
                token_idx = torch.arange(int(start_token), int(end_token) + 1, device=query_attn.device, dtype=torch.long)
                mass = query_attn.index_select(-1, token_idx).sum(dim=-1)
                mean_attn = float(mass.mean().item())
            else:
                mean_attn = float("nan")
            attention_rows.append(
                {
                    "layer": int(layer_idx),
                    "row_key": str(spec["row_key"]),
                    "sentence_label": str(spec["sentence_label"]),
                    "sentence_idx": int(spec["sentence_idx"]),
                    "mean_attn": mean_attn,
                    "available": available,
                    "display_order": int(spec["display_order"]),
                }
            )
    return pd.DataFrame(attention_rows)


def _attention_matrix(attention_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    if attention_df.empty:
        return pd.DataFrame(), [], {}
    ordered_keys = (
        attention_df[["display_order", "row_key"]]
        .drop_duplicates()
        .sort_values("display_order")["row_key"]
        .tolist()
    )
    label_map = (
        attention_df[["row_key", "sentence_label"]]
        .drop_duplicates("row_key")
        .set_index("row_key")["sentence_label"]
        .to_dict()
    )
    matrix_df = (
        attention_df.pivot_table(index="row_key", columns="layer", values="mean_attn", aggfunc="mean")
        .reindex(ordered_keys)
        .sort_index(axis=1)
    )
    return matrix_df, ordered_keys, label_map


def save_figure(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return path


def plot_commitment_context_figure(case: dict):
    apply_paper_style()
    context_df = build_counterfactual_uncertainty_df(case)
    prior_sentence_idx = int(case["prior_row"]["sentence_idx"])
    spike_sentence_idx = int(case["spike_row"]["sentence_idx"])
    sentence_lookup = case["sentence_lookup"]
    context_rows = list(context_df.itertuples())

    fig = plt.figure(figsize=FIGURE_SIZES["text_plus_plot"])
    grid = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.45], wspace=0.22)

    ax_line = fig.add_subplot(grid[0, 0])
    ax_text = fig.add_subplot(grid[0, 1])
    ax_text.axis("off")

    # --- Left panel: counterfactual deception rate ---
    ax_line.errorbar(
        context_df["sentence_idx"],
        context_df["deception_rate"],
        yerr=context_df["deception_rate_se"],
        color=PANEL_COLORS["neutral"],
        linewidth=2.0,
        marker="o",
        capsize=4,
        ecolor="#666666",
    )

    for row in context_df.itertuples():
        color = "#b5b5b5"
        size = 55
        if int(row.sentence_idx) == prior_sentence_idx:
            color = PANEL_COLORS["prior"]
            size = 85
        elif int(row.sentence_idx) == spike_sentence_idx:
            color = PANEL_COLORS["spike"]
            size = 95
        ax_line.scatter(row.sentence_idx, row.deception_rate, color=color, s=size, zorder=3)

    ax_line.axvline(
        spike_sentence_idx,
        color=PANEL_COLORS["spike"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
    )
    style_axis(
        ax_line,
        title="Counterfactual Deception Rate",
        xlabel="Sentence index",
        ylabel="Counterfactual deception rate",
    )
    x_vals = context_df["sentence_idx"].astype(int).tolist()
    ax_line.set_xticks(x_vals)
    ax_line.set_xticklabels([str(x) for x in x_vals])

    # --- Right panel: 4 stacked sentence boxes ---
    ax_text.set_title("Local sentence context", pad=10)
    finalize_figure(fig, "context")

    top_margin = 0.96
    bottom_margin = 0.04
    gap = 0.025
    box_x = 0.02
    box_w = 0.96
    wrapped_blocks, body_fontsize, title_fontsize, block_height = _fit_commitment_context_boxes(
        fig,
        ax_text,
        context_rows,
        sentence_lookup,
        box_x=box_x,
        box_w=box_w,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        gap=gap,
    )

    for i, (row, wrapped_body) in enumerate(wrapped_blocks):
        sentence_idx = int(row.sentence_idx)

        edge = "#9c9c9c"
        if sentence_idx == prior_sentence_idx:
            edge = PANEL_COLORS["prior"]
        elif sentence_idx == spike_sentence_idx:
            edge = PANEL_COLORS["spike"]

        y_top = top_margin - i * (block_height + gap)
        y = y_top - block_height

        rect = plt.Rectangle(
            (box_x, y),
            box_w,
            block_height,
            transform=ax_text.transAxes,
            facecolor="white",
            edgecolor=edge,
            linewidth=1.8,
        )
        ax_text.add_patch(rect)

        title = _commitment_box_title(row)

        ax_text.text(
            box_x + 0.02,
            y + block_height - 0.03,
            title,
            transform=ax_text.transAxes,
            fontsize=title_fontsize,
            fontweight="semibold",
            color="#1f1f1f",
            va="top",
            ha="left",
        )

        ax_text.text(
            box_x + 0.02,
            y + block_height - 0.085,
            wrapped_body,
            transform=ax_text.transAxes,
            fontsize=body_fontsize,
            color="#333333",
            va="top",
            ha="left",
            linespacing=1.25,
        )

    return fig



def plot_absolute_sentence_attention_figure(case: dict):
    apply_paper_style()
    prior_df = build_absolute_sentence_attention_df(case, boundary_key="prior")
    spike_df = build_absolute_sentence_attention_df(case, boundary_key="spike")
    prior_matrix, ordered_keys, label_map = _attention_matrix(prior_df)
    spike_matrix, _, _ = _attention_matrix(spike_df)
    if prior_matrix.empty or spike_matrix.empty:
        raise RuntimeError("Could not construct local absolute-sentence attention heatmaps.")
    prior_matrix = prior_matrix.reindex(ordered_keys)
    spike_matrix = spike_matrix.reindex(ordered_keys)
    diff_matrix = spike_matrix - prior_matrix
    unavailable_mask = prior_matrix.isna() | spike_matrix.isna()
    diff_matrix = diff_matrix.mask(unavailable_mask)

    shared_max = float(
        np.nanmax(
            [
                prior_matrix.to_numpy(dtype=float),
                spike_matrix.to_numpy(dtype=float),
            ]
        )
    )
    diff_values = diff_matrix.to_numpy(dtype=float)
    diff_abs = float(np.nanmax(np.abs(diff_values))) if np.isfinite(diff_values).any() else 0.0

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["three_panel"], sharey=True)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#ececec")
    diff_cmap = plt.cm.coolwarm.copy()
    diff_cmap.set_bad("#ececec")

    panels = [
        (
            axes[0],
            prior_matrix,
            f"Pre-spike sentence\nattention mass | S{int(case['prior_row']['sentence_idx'])}",
            cmap,
            0.0,
            shared_max,
            "Attention mass",
        ),
        (
            axes[1],
            spike_matrix,
            f"Spike sentence\nattention mass | S{int(case['spike_row']['sentence_idx'])}",
            cmap,
            0.0,
            shared_max,
            "Attention mass",
        ),
        (
            axes[2],
            diff_matrix,
            "Spike - pre-spike\nattention mass",
            diff_cmap,
            -diff_abs,
            diff_abs,
            "Delta attention mass",
        ),
    ]

    for ax, matrix_df, title, color_map, vmin, vmax, colorbar_label in panels:
        image = ax.imshow(
            matrix_df.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
            cmap=color_map,
            vmin=vmin,
            vmax=vmax,
        )
        style_axis(ax, title=title, xlabel="Layer", grid=False, title_pad=10)
        ax.set_yticks(np.arange(len(matrix_df.index)))
        ax.set_yticklabels([label_map.get(row_key, row_key) for row_key in matrix_df.index.tolist()])
        tick_step = max(1, int(math.ceil(matrix_df.shape[1] / 10)))
        tick_positions = list(range(0, matrix_df.shape[1], tick_step))
        if tick_positions[-1] != matrix_df.shape[1] - 1:
            tick_positions.append(matrix_df.shape[1] - 1)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(matrix_df.columns[idx]) for idx in tick_positions])
        ax.tick_params(axis="x", labelrotation=0)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.012)
        colorbar.set_label(colorbar_label)
    axes[0].set_ylabel("Attended context")
    finalize_figure(fig, "three_panel")
    return fig


def plot_probe_feature_story_figure(case: dict):
    apply_paper_style()
    late_start = int(case["late_start"])
    curve_specs = [
        ("current_vs_prev_mean", "Selected-sentence attention share", "feature"),
        ("delta_current_vs_prev_mean", "Change in selected-sentence attention share", "feature"),
        ("entropy_prior_mean", "Prior-context attention entropy", "feature"),
        ("delta_entropy_prior_mean", "Change in prior-context attention entropy", "feature"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES["two_by_two"], sharex=False)
    axes = axes.ravel()
    layers = np.arange(case["num_layers"])

    for ax, (stem, title, source) in zip(axes, curve_specs):
        if source == "feature":
            prior_values = _feature_curve(case["prior_feature_view"], stem)
            spike_values = _feature_curve(case["spike_feature_view"], stem)
        else:
            prior_values = pd.to_numeric(case["prior_activation_df"][stem], errors="coerce").to_numpy(dtype=float)
            spike_values = pd.to_numeric(case["spike_activation_df"][stem], errors="coerce").to_numpy(dtype=float)
        ax.axvspan(late_start, case["num_layers"] - 1, color="#f3efe8", alpha=0.8, zorder=0)
        ax.plot(layers, prior_values, color=PANEL_COLORS["prior"], linewidth=2.0, marker="o", markersize=3.8, label="sentence i-1")
        ax.plot(layers, spike_values, color=PANEL_COLORS["spike"], linewidth=2.0, marker="o", markersize=3.8, label="spike sentence")
        style_axis(ax, title=title, xlabel="Layer", ylabel="Feature value")
        prior_late = _late_mean(prior_values, late_start=late_start)
        spike_late = _late_mean(spike_values, late_start=late_start)
        ax.text(
            0.03,
            0.96,
            f"late mean\nprior={prior_late:0.3f}\nspike={spike_late:0.3f}\nΔ={spike_late - prior_late:+0.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=TEXT_SIZES["annotation"],
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#dddddd"},
        )


    style_legend(axes[0], loc="lower left")

    finalize_figure(fig, "two_by_two")
    return fig


def plot_activation_similarity_figure(case: dict):
    apply_paper_style()
    late_start = int(case["late_start"])
    num_layers = int(case["num_layers"])
    layers = np.arange(num_layers, dtype=float)

    prior_attention = pd.to_numeric(
        case["prior_prev3_attention_df"]["current_vs_prev3_mean"], errors="coerce"
    ).to_numpy(dtype=float)
    spike_attention = pd.to_numeric(
        case["spike_prev3_attention_df"]["current_vs_prev3_mean"], errors="coerce"
    ).to_numpy(dtype=float)
    prior_activation = _activation_curve(case["prior_activation_df"], "cos_cur_mean3")
    spike_activation = _activation_curve(case["spike_activation_df"], "cos_cur_mean3")

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["two_panel"], sharex=True)
    axes[0].axvspan(late_start, num_layers - 1, color="#f3efe8", alpha=0.82, zorder=0)
    axes[1].axvspan(late_start, num_layers - 1, color="#f3efe8", alpha=0.82, zorder=0)

    axes[0].plot(
        layers,
        prior_attention,
        color=PANEL_COLORS["prior"],
        linewidth=2.4,
        marker="o",
        markersize=4.0,
        markevery=max(1, num_layers // 12),
        label="pre-spike sentence",
    )
    axes[0].plot(
        layers,
        spike_attention,
        color=PANEL_COLORS["spike"],
        linewidth=2.4,
        marker="o",
        markersize=4.0,
        markevery=max(1, num_layers // 12),
        label="spike sentence",
    )
    axes[0].fill_between(
        layers,
        prior_attention,
        spike_attention,
        where=np.isfinite(prior_attention) & np.isfinite(spike_attention) & (spike_attention >= prior_attention),
        color="#86b88f",
        alpha=0.16,
        interpolate=True,
    )
    style_axis(
        axes[0],
        title="Attention feature",
        xlabel="Layer",
        ylabel="Current-sentence attention share",
    )
    style_legend(axes[0], loc="lower right")

    axes[1].plot(
        layers,
        prior_activation,
        color=PANEL_COLORS["prior"],
        linewidth=2.4,
        marker="o",
        markersize=4.0,
        markevery=max(1, num_layers // 12),
        label="pre-spike sentence",
    )
    axes[1].plot(
        layers,
        spike_activation,
        color=PANEL_COLORS["spike"],
        linewidth=2.4,
        marker="o",
        markersize=4.0,
        markevery=max(1, num_layers // 12),
        label="spike sentence",
    )
    axes[1].fill_between(
        layers,
        prior_activation,
        spike_activation,
        where=np.isfinite(prior_activation) & np.isfinite(spike_activation) & (spike_activation >= prior_activation),
        color="#86b88f",
        alpha=0.16,
        interpolate=True,
    )
    style_axis(
        axes[1],
        title="Activation feature",
        xlabel="Layer",
        ylabel="Activation alignment with recent context",
    )
    style_legend(axes[1], loc="lower right")

    tick_step = max(1, int(math.ceil(num_layers / 10)))
    tick_positions = list(range(0, num_layers, tick_step))
    if tick_positions[-1] != num_layers - 1:
        tick_positions.append(num_layers - 1)
    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(value) for value in tick_positions])
    """
    attention_summary = "\n".join(
        [
            f"Late-layer mean ({late_start}-{num_layers - 1})",
            f"pre-spike = {_late_mean(prior_attention, late_start=late_start):0.3f}",
            f"spike = {_late_mean(spike_attention, late_start=late_start):0.3f}",
            f"delta = {_late_mean(spike_attention - prior_attention, late_start=late_start):+0.3f}",
        ]
    )
    activation_summary = "\n".join(
        [
            f"Late-layer mean ({late_start}-{num_layers - 1})",
            f"pre-spike = {_late_mean(prior_activation, late_start=late_start):0.3f}",
            f"spike = {_late_mean(spike_activation, late_start=late_start):0.3f}",
            f"delta = {_late_mean(spike_activation - prior_activation, late_start=late_start):+0.3f}",
        ]
    )
    axes[0].text(
        0.02,
        0.98,
        attention_summary,
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#dddddd"},
    )
    axes[1].text(
        0.02,
        0.98,
        activation_summary,
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#dddddd"},
    )
    """
    #fig.suptitle("The spike shifts toward previous-3 context in both attention and activations", y=0.98)
    finalize_figure(fig, "two_panel")
    return fig


def plot_activation_similarity_aggregate_figure(case: dict):
    apply_paper_style()
    summary_df = build_bs_prev3_bridge_summary().copy()
    summary_df = summary_df.dropna(
        subset=[
            "prior_attention_value",
            "spike_attention_value",
            "diff_attention_value",
            "prior_activation_value",
            "spike_activation_value",
            "diff_activation_value",
        ]
    ).sort_values("diff_activation_value").reset_index(drop=True)
    if summary_df.empty:
        raise RuntimeError("Could not load any aggregate BS activation similarity rows.")

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["aggregate_two_panel"], sharex=True)
    selected_spike_boundary = str(case["spike_row"].name) if case is not None and "spike_row" in case else None

    panel_specs = [
        (
            axes[0],
            "prior_attention_value",
            "spike_attention_value",
            "diff_attention_value",
            "Attention feature",
            "Current-sentence attention share",
        ),
        (
            axes[1],
            "prior_activation_value",
            "spike_activation_value",
            "diff_activation_value",
            "Activation feature",
            "Activation alignment with recent context",
        ),
    ]
    x_prior = np.zeros(len(summary_df), dtype=float)
    x_spike = np.ones(len(summary_df), dtype=float)

    for ax, prior_col, spike_col, delta_col, title, ylabel in panel_specs:
        for row_idx, row in summary_df.iterrows():
            is_selected = selected_spike_boundary is not None and str(row["spike_boundary_id"]) == selected_spike_boundary
            line_color = "#455a64" if is_selected else "#90a4ae"
            line_width = 2.4 if is_selected else 1.4
            line_alpha = 0.95 if is_selected else (0.35 if float(row[delta_col]) >= 0 else 0.2)
            ax.plot(
                [x_prior[row_idx], x_spike[row_idx]],
                [float(row[prior_col]), float(row[spike_col])],
                color=line_color,
                linewidth=line_width,
                alpha=line_alpha,
                zorder=1,
            )

        ax.scatter(
            x_prior,
            summary_df[prior_col],
            color=PANEL_COLORS["prior"],
            s=42,
            alpha=0.9,
            label="pre-spike sentence",
            zorder=3,
        )
        ax.scatter(
            x_spike,
            summary_df[spike_col],
            color=PANEL_COLORS["spike"],
            s=42,
            alpha=0.9,
            label="spike sentence",
            zorder=3,
        )

        prior_mean = float(summary_df[prior_col].mean())
        spike_mean = float(summary_df[spike_col].mean())
        prior_sem = float(summary_df[prior_col].std(ddof=1) / math.sqrt(len(summary_df))) if len(summary_df) > 1 else 0.0
        spike_sem = float(summary_df[spike_col].std(ddof=1) / math.sqrt(len(summary_df))) if len(summary_df) > 1 else 0.0
        ax.errorbar(
            [0, 1],
            [prior_mean, spike_mean],
            yerr=[prior_sem, spike_sem],
            color="#222222",
            linewidth=2.6,
            capsize=5,
            zorder=4,
        )
        ax.scatter([0, 1], [prior_mean, spike_mean], color="#222222", s=80, zorder=5)

        positive_fraction = float((summary_df[delta_col] > 0).mean())
        median_delta = float(summary_df[delta_col].median())
        mean_delta = float(summary_df[delta_col].mean())
        """
        ax.text(
            0.03,
            0.97,
            "\n".join(
                [
                    f"n = {len(summary_df)} BS spike examples",
                    f"positive fraction = {positive_fraction:0.0%}",
                    f"mean delta = {mean_delta:+0.3f}",
                    f"median delta = {median_delta:+0.3f}",
                ]
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.6,
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#dddddd"},
        )
        """
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pre-spike", "spike"])
        style_axis(ax, title=title, ylabel=ylabel, grid_axis="y")
        
    style_legend(axes[1], loc="lower right")
    #fig.suptitle("Across BS examples, the spike strengthens both previous-3 attention and previous-3 alignment", y=0.98)
    finalize_figure(fig, "aggregate_two_panel")
    return fig


def generate_figure_package(case: dict | None = None, *, output_dir: Path = OUTPUT_ROOT) -> dict:
    if case is None:
        case = load_cherry_picked_case()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentence_table_path = output_dir / "case_sentence_table.csv"
    late_metric_summary_path = output_dir / "late_metric_summary.csv"
    case["sentence_table"].to_csv(sentence_table_path, index=False)
    case["late_metric_summary"].to_csv(late_metric_summary_path, index=False)

    context_path = save_figure(plot_commitment_context_figure(case), output_dir / "figure1_commitment_context.png")
    attention_path = save_figure(plot_absolute_sentence_attention_figure(case), output_dir / "figure2_attention_reallocation.png")
    feature_path = save_figure(plot_probe_feature_story_figure(case), output_dir / "figure3_probe_feature_story.png")
    activation_similarity_path = save_figure(
        plot_activation_similarity_figure(case),
        output_dir / "figure4_prev3_bridge.png",
    )
    activation_aggregate_path = save_figure(
        plot_activation_similarity_aggregate_figure(case),
        output_dir / "figure5_prev3_aggregate.png",
    )

    return {
        "case": case,
        "output_dir": output_dir,
        "sentence_table_path": sentence_table_path,
        "late_metric_summary_path": late_metric_summary_path,
        "context_path": context_path,
        "attention_path": attention_path,
        "feature_path": feature_path,
        "activation_similarity_path": activation_similarity_path,
        "activation_aggregate_path": activation_aggregate_path,
    }


def main() -> None:
    outputs = generate_figure_package()
    print(f"Saved paper figures to {outputs['output_dir']}")
    print(f"Context figure: {outputs['context_path']}")
    print(f"Attention figure: {outputs['attention_path']}")
    print(f"Feature figure: {outputs['feature_path']}")
    print(f"Activation similarity figure: {outputs['activation_similarity_path']}")
    print(f"Activation aggregate figure: {outputs['activation_aggregate_path']}")
    print(f"Sentence table: {outputs['sentence_table_path']}")
    print(f"Metric summary: {outputs['late_metric_summary_path']}")


if __name__ == "__main__":
    main()
