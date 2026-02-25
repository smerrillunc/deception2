#!/usr/bin/env python3
"""
Build reduced temporal features with consistent naming:
  - before_{feature_name}_{stat}
  - at_{feature_name}
  - after_{feature_name}_{stat}

Design goals:
  1) Keep a reduced set of high-signal features across structural/lexical,
     entropy-logit, activation, and attention families.
  2) Emit a consistent temporal schema for all families.
  3) Optionally run raw extraction and temporal reduction in one script.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from raw_corr_sweep_runner import ExtractConfig, run_extract
except Exception:
    ExtractConfig = None  # type: ignore[assignment]
    run_extract = None  # type: ignore[assignment]


DEFAULT_STRUCTURAL_BASE = [
    "word_count",
    "char_count",
    "avg_word_len",
    "upper_ratio",
    "digit_count",
    "negation_count",
    "punct_qmark",
    "punct_exclaim",
]

DEFAULT_LEXICAL_BASE = [
    "deceptive_word_frac",
    "hedge_word_frac",
    "certainty_word_frac",
    "negation_word_frac",
    "justification_word_frac",
    "self_reference_word_frac",
    "contradiction_word_frac",
]

# Reduced uncertainty/confidence set commonly used for LM reliability signals:
# surprisal/NLL, entropy, confidence (pmax), and logit margin/dispersion.
DEFAULT_ENTROPY_BASE = [
    "tok_nll_mean",
    "tok_entropy_mean",
    "tok_entropy_topk_renorm_mean",
    "tok_margin_logit_mean",
    "tok_pmax_mean",
    "tok_logit_std_mean",
    "tok_entropy_delta_mean",
    "tok_entropy_posdiff_mean",
]

DEFAULT_ACTIVATION_BASE = [
    "act_m1_l2_mean",
    "act_m2_l2_mean",
    "act_m4_l2_mean",
    "act_m1_absmean_mean",
    "act_m2_absmean_mean",
    "act_pair_m2_m1_cos",
    "act_traj_energy_slope",
    "act_traj_energy_std",
]

DEFAULT_ATTENTION_BASE = [
    "attn_rawmean__d1__in_long_mass",
    "attn_rawmean__d1__out_long_mass",
    "attn_rawmean__d1__anchor_ratio",
    "attn_rawmean__d2__in_long_entropy",
    "attn_rawmean__d2__out_long_entropy",
    "attn_roll__d1__in_long_mass",
    "attn_roll__d1__out_long_mass",
    "attn_roll__d1__anchor_ratio",
    "attn_roll__d2__in_long_entropy",
    "attn_roll__d2__out_long_entropy",
]

FAMILY_ORDER = ["structural", "lexical", "entropy", "activation", "attention"]
DEFAULT_SENTENCE_LEVEL_STATS = ("mean", "max", "min", "std")
DEFAULT_TEMPORAL_AGG_STATS = ("mean", "max", "min", "std")
VALID_STATS = {"mean", "max", "min", "std"}


def _parse_cols_arg(raw: str | None, default: Sequence[str]) -> List[str]:
    if raw is None:
        return list(default)
    cols = [x.strip() for x in raw.split(",") if x.strip()]
    return cols


def _parse_stats_arg(raw: str, name: str) -> Tuple[str, ...]:
    stats = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not stats:
        raise ValueError(f"{name} must include at least one stat.")
    bad = [s for s in stats if s not in VALID_STATS]
    if bad:
        raise ValueError(f"{name} has invalid stats {bad}. Allowed: {sorted(VALID_STATS)}")
    return stats


def _unique(seq: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


_ACT_LAYER_RE = re.compile(r"^act_(m\d+|p\d+)_[a-z0-9_]+$")


def _activation_triplet_tags(df: pd.DataFrame) -> Tuple[List[str], Dict[str, object]]:
    tags: List[str] = []
    for c in df.columns:
        m = _ACT_LAYER_RE.match(c)
        if m:
            tags.append(m.group(1))
    tags = _unique(tags)

    p_nums = sorted(int(t[1:]) for t in tags if t.startswith("p"))
    m_nums = sorted((int(t[1:]) for t in tags if t.startswith("m")), reverse=True)

    ordered: List[str]
    source: str
    notes: str
    if p_nums:
        ordered = [f"p{n}" for n in p_nums]
        source = "positive_indices"
        notes = "Interpreted as direct hidden-state indices (p*)."
    elif m_nums:
        # mN means N layers from last. Larger N is earlier in depth.
        ordered = [f"m{n}" for n in m_nums]
        source = "negative_offsets"
        notes = "No p* tags found; used m* tags as earliest/mid/latest proxies."
    else:
        return [], {
            "ordered_tags": [],
            "selected_triplet_tags": [],
            "source": "none",
            "notes": "No activation layer tags found in input columns.",
        }

    first = ordered[0]
    # Lower-mid choice keeps the middle anchor from drifting too close to "last"
    # when the number of available layer tags is even.
    mid = ordered[(len(ordered) - 1) // 2]
    last = ordered[-1]
    selected = _unique([first, mid, last])
    meta = {
        "ordered_tags": ordered,
        "selected_triplet_tags": selected,
        "source": source,
        "notes": notes,
    }
    return selected, meta


def _derive_activation_base_columns(df: pd.DataFrame) -> Tuple[List[str], Dict[str, object]]:
    selected_tags, tag_meta = _activation_triplet_tags(df)
    cols: List[str] = []

    per_layer_suffixes = ("l2_mean", "absmean_mean", "sparse_mean")
    for tag in selected_tags:
        for suffix in per_layer_suffixes:
            c = f"act_{tag}_{suffix}"
            if c in df.columns:
                cols.append(c)

    # Pairwise drift between chosen depth anchors (if present).
    for a, b in zip(selected_tags[:-1], selected_tags[1:]):
        for met in ("cos", "l2diff", "ratio"):
            c = f"act_pair_{a}_{b}_{met}"
            if c in df.columns:
                cols.append(c)

    # Keep trajectory features if available.
    for c in (
        "act_traj_energy_slope",
        "act_traj_energy_curv",
        "act_traj_energy_range",
        "act_traj_energy_last_first",
        "act_traj_energy_std",
    ):
        if c in df.columns:
            cols.append(c)

    cols = _unique(cols)
    if not cols:
        cols = [c for c in DEFAULT_ACTIVATION_BASE if c in df.columns]
        tag_meta = dict(tag_meta)
        tag_meta["notes"] = (
            str(tag_meta.get("notes", "")) + " | Fell back to DEFAULT_ACTIVATION_BASE."
        ).strip(" |")

    return cols, tag_meta


def _parse_int_tuple(raw: str, name: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError(f"{name} must include at least one integer.")
    return vals


def _activation_offsets_layer0_mid_last(
    model_name: str,
    *,
    trust_remote_code: bool,
) -> Tuple[Tuple[int, ...], Dict[str, object]]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    n_hidden = getattr(cfg, "num_hidden_layers", None)
    if not isinstance(n_hidden, int) or n_hidden <= 0:
        raise ValueError(
            f"Could not read num_hidden_layers from config for {model_name}. "
            "Use --activation-layer-mode explicit."
        )

    # hidden_states[0] is embeddings for most decoder models; first block output is index 1.
    layer0_idx = 1
    mid_idx = max(1, n_hidden // 2)
    last_idx = n_hidden
    offsets = tuple(_unique([layer0_idx, mid_idx, last_idx]))
    meta = {
        "num_hidden_layers": int(n_hidden),
        "layer0_index_used": int(layer0_idx),
        "mid_index_used": int(mid_idx),
        "last_index_used": int(last_idx),
        "activation_layer_offsets_used": offsets,
    }
    return offsets, meta


def _rolling_stat_grouped(
    source: pd.Series,
    groups: pd.Series,
    *,
    window: int,
    stat: str,
    reverse: bool = False,
) -> pd.Series:
    if window <= 0:
        return pd.Series(np.nan, index=source.index, dtype=float)

    s = source
    g = groups
    if reverse:
        s = s.iloc[::-1]
        g = g.iloc[::-1]

    min_periods = 2 if stat == "std" else 1
    rolling = s.groupby(g, sort=False).rolling(window=window, min_periods=min_periods)
    if stat == "mean":
        out = rolling.mean().reset_index(level=0, drop=True)
    elif stat == "min":
        out = rolling.min().reset_index(level=0, drop=True)
    elif stat == "max":
        out = rolling.max().reset_index(level=0, drop=True)
    elif stat == "std":
        out = rolling.std().reset_index(level=0, drop=True)
    else:
        raise ValueError(f"Unsupported rolling stat: {stat}")

    if reverse:
        out = out.iloc[::-1]
    return out.reindex(source.index)


_STAT_SUFFIX_RE = re.compile(r"^(.*)_(mean|max|min|std)$")


def _base_name_from_col(col: str) -> Tuple[str, str | None]:
    m = _STAT_SUFFIX_RE.match(col)
    if not m:
        return col, None
    return m.group(1), m.group(2)


def _build_sentence_level_feature_frame(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    sentence_level_stats: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build sentence-level stat features per base feature so every base has
    *_mean/*_max/*_min/*_std representation.
    """
    out = pd.DataFrame(index=df.index)
    built_cols: List[str] = []
    seen = set()

    for col in cols:
        if col not in df.columns:
            continue

        base_name, detected_stat = _base_name_from_col(col)
        src_raw = pd.to_numeric(df[col], errors="coerce")

        for st in sentence_level_stats:
            out_col = f"{base_name}_{st}"
            if out_col in seen:
                continue

            sibling = f"{base_name}_{st}"
            if sibling in df.columns:
                src = pd.to_numeric(df[sibling], errors="coerce")
            elif detected_stat == st:
                src = src_raw
            elif st in ("mean", "max", "min"):
                # For scalar sentence-level features, reuse the scalar value.
                src = src_raw
            else:
                # std fallback for scalar sentence-level features.
                src = pd.Series(0.0, index=df.index, dtype=float)

            out[out_col] = src.astype(float)
            built_cols.append(out_col)
            seen.add(out_col)

    return out, built_cols


def _temporalize_column(
    df: pd.DataFrame,
    col: str,
    *,
    before_window: int,
    after_window: int,
    include_diff: bool,
    temporal_agg_stats: Sequence[str],
) -> pd.DataFrame:
    vals = pd.to_numeric(df[col], errors="coerce")
    groups = df["example_id"].astype(str)
    grp = vals.groupby(groups, sort=False)

    before_src = grp.shift(1)
    after_src = grp.shift(-1)

    out = pd.DataFrame(index=df.index)
    out[f"at_{col}"] = vals.astype(float)

    for stat in temporal_agg_stats:
        out[f"before_{col}_{stat}"] = _rolling_stat_grouped(
            before_src,
            groups,
            window=before_window,
            stat=stat,
            reverse=False,
        )
        out[f"after_{col}_{stat}"] = _rolling_stat_grouped(
            after_src,
            groups,
            window=after_window,
            stat=stat,
            reverse=True,
        )

    if include_diff:
        out[f"before_{col}_mean_minus_at"] = out[f"before_{col}_mean"] - out[f"at_{col}"]
        out[f"after_{col}_mean_minus_at"] = out[f"after_{col}_mean"] - out[f"at_{col}"]
        out[f"after_{col}_mean_minus_before_mean"] = (
            out[f"after_{col}_mean"] - out[f"before_{col}_mean"]
        )

    return out


def _resolve_bases(
    df: pd.DataFrame,
    requested: Mapping[str, Sequence[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    available: Dict[str, List[str]] = {}
    missing: Dict[str, List[str]] = {}
    for fam, cols in requested.items():
        have = [c for c in cols if c in df.columns]
        miss = [c for c in cols if c not in df.columns]
        available[fam] = have
        missing[fam] = miss
    return available, missing


def _build_feature_sets(generated_by_family: Mapping[str, Sequence[str]]) -> OrderedDict:
    structural = list(generated_by_family.get("structural", []))
    lexical = list(generated_by_family.get("lexical", []))
    entropy = list(generated_by_family.get("entropy", []))
    activation = list(generated_by_family.get("activation", []))
    attention = list(generated_by_family.get("attention", []))

    struct_lex = _unique(structural + lexical)
    set1 = _unique(struct_lex + entropy)
    set2 = _unique(set1 + activation)
    set3 = _unique(set2 + attention)

    # Keep names close to prior notebooks.
    feature_sets = OrderedDict(
        {
            "baseline_struct": structural,
            "baseline_struct_lex": struct_lex,
            "set1_struct_lex_entropy": set1,
            "set2_struct_lex_entropy_activation": set2,
            "set3_struct_lex_entropy_activation_attention": set3,
        }
    )
    return feature_sets


def build_temporal_reduced_features(
    df: pd.DataFrame,
    *,
    base_by_family: Mapping[str, Sequence[str]],
    before_window: int,
    after_window: int,
    include_diff: bool,
    sentence_level_stats: Sequence[str],
    temporal_agg_stats: Sequence[str],
) -> Tuple[pd.DataFrame, OrderedDict, Dict[str, List[str]], Dict[str, List[str]]]:
    if "example_id" not in df.columns:
        raise ValueError("Input DataFrame must contain example_id.")
    if "sentence_idx" not in df.columns:
        raise ValueError("Input DataFrame must contain sentence_idx.")

    work = df.copy()
    work["example_id"] = work["example_id"].astype(str)
    work["sentence_idx"] = pd.to_numeric(work["sentence_idx"], errors="coerce")
    work = work[np.isfinite(work["sentence_idx"])].copy()
    work["sentence_idx"] = work["sentence_idx"].astype(int)
    work = work.sort_values(["example_id", "sentence_idx"], kind="stable").reset_index(drop=True)

    available, missing = _resolve_bases(work, base_by_family)

    meta_cols = [
        c
        for c in ["example_id", "sentence_idx", "deception_rate", "sentence_position", "total_sentences"]
        if c in work.columns
    ]

    out_frames = [work[meta_cols].copy()]
    generated_by_family: Dict[str, List[str]] = {fam: [] for fam in FAMILY_ORDER}

    for fam in FAMILY_ORDER:
        cols = available.get(fam, [])
        sentence_df, sentence_cols = _build_sentence_level_feature_frame(
            work,
            cols,
            sentence_level_stats=sentence_level_stats,
        )
        fam_df = pd.concat([work[["example_id"]], sentence_df], axis=1)
        for col in sentence_cols:
            tdf = _temporalize_column(
                fam_df,
                col,
                before_window=before_window,
                after_window=after_window,
                include_diff=include_diff,
                temporal_agg_stats=temporal_agg_stats,
            )
            out_frames.append(tdf)
            generated_by_family[fam].extend(tdf.columns.tolist())

    out_df = pd.concat(out_frames, axis=1)
    feature_sets = _build_feature_sets(generated_by_family)
    return out_df, feature_sets, generated_by_family, missing


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def _build_base_by_family_from_args(
    args: argparse.Namespace,
    df: pd.DataFrame,
) -> Tuple[Dict[str, List[str]], Dict[str, object]]:
    structural = _parse_cols_arg(args.structural_cols, DEFAULT_STRUCTURAL_BASE)
    lexical = _parse_cols_arg(args.lexical_cols, DEFAULT_LEXICAL_BASE)
    entropy = _parse_cols_arg(args.entropy_cols, DEFAULT_ENTROPY_BASE)
    attention = _parse_cols_arg(args.attention_cols, DEFAULT_ATTENTION_BASE)

    if args.activation_cols is not None:
        activation = _parse_cols_arg(args.activation_cols, DEFAULT_ACTIVATION_BASE)
        activation_meta: Dict[str, object] = {
            "activation_base_source": "explicit_cols",
            "selected_triplet_tags": [],
            "notes": "Activation columns provided by --activation-cols.",
        }
    else:
        activation, tag_meta = _derive_activation_base_columns(df)
        activation_meta = {
            "activation_base_source": "derived_layer0_mid_last",
            **tag_meta,
        }

    base_by_family = {
        "structural": structural,
        "lexical": lexical,
        "entropy": entropy,
        "activation": activation,
        "attention": attention,
    }
    return base_by_family, activation_meta


def _add_temporal_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--before-window", type=int, default=3)
    ap.add_argument("--after-window", type=int, default=3)
    ap.add_argument("--sentence-level-stats", type=str, default="mean,max,min,std")
    ap.add_argument("--temporal-agg-stats", type=str, default="mean,max,min,std")
    ap.add_argument("--no-diff", action="store_true", default=False)
    ap.add_argument("--structural-cols", type=str, default=None)
    ap.add_argument("--lexical-cols", type=str, default=None)
    ap.add_argument("--entropy-cols", type=str, default=None)
    ap.add_argument("--activation-cols", type=str, default=None)
    ap.add_argument("--attention-cols", type=str, default=None)
    ap.add_argument("--feature-sets-out", type=Path, default=None)
    ap.add_argument("--manifest-out", type=Path, default=None)


def _add_extract_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--examples-path", type=Path, required=True)
    ap.add_argument("--sentences-path", type=Path, required=True)
    ap.add_argument("--localization-path", type=Path, required=True)
    ap.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--num-examples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-localized", action="store_true", default=True)
    ap.add_argument("--max-tokens", type=int, default=10000)
    ap.add_argument("--topk-vocab", type=int, default=32)
    ap.add_argument("--min-dist-list", type=str, default="1,2,4")
    ap.add_argument("--attention-layer-offsets", type=str, default="-1,-2,-3")
    ap.add_argument("--activation-layer-offsets", type=str, default="-1,-2,-4,-8")
    ap.add_argument(
        "--activation-layer-mode",
        type=str,
        choices=("layer0_mid_last", "explicit"),
        default="layer0_mid_last",
        help="How to choose activation layers during raw extraction in extract-build mode.",
    )
    ap.add_argument("--activation-sparsity-eps", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--token-oom-backoff", type=float, default=0.5)
    ap.add_argument("--token-min-tokens", type=int, default=256)
    ap.add_argument("--raw-out-path", type=Path, default=None)


def _run_build(args: argparse.Namespace) -> None:
    in_path: Path = args.input_features
    out_path: Path = args.out_path

    print(f"[build] loading input features: {in_path}", flush=True)
    df = pd.read_parquet(in_path)
    base_by_family, activation_meta = _build_base_by_family_from_args(args, df=df)
    include_diff = not args.no_diff
    sentence_level_stats = _parse_stats_arg(args.sentence_level_stats, "sentence-level-stats")
    temporal_agg_stats = _parse_stats_arg(args.temporal_agg_stats, "temporal-agg-stats")

    out_df, feature_sets, generated_by_family, missing = build_temporal_reduced_features(
        df,
        base_by_family=base_by_family,
        before_window=args.before_window,
        after_window=args.after_window,
        include_diff=include_diff,
        sentence_level_stats=sentence_level_stats,
        temporal_agg_stats=temporal_agg_stats,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[build] wrote temporal features: {out_path} shape={out_df.shape}", flush=True)

    feature_sets_out = args.feature_sets_out or out_path.with_name(out_path.stem + "_feature_sets.json")
    manifest_out = args.manifest_out or out_path.with_name(out_path.stem + "_manifest.json")

    _write_json(feature_sets_out, feature_sets)
    manifest = {
        "input_features": str(in_path),
        "output_features": str(out_path),
        "before_window": int(args.before_window),
        "after_window": int(args.after_window),
        "include_diff": bool(include_diff),
        "sentence_level_stats": list(sentence_level_stats),
        "temporal_agg_stats": list(temporal_agg_stats),
        "requested_base_features": base_by_family,
        "missing_base_features": missing,
        "activation_selection": activation_meta,
        "generated_feature_counts": {k: len(v) for k, v in generated_by_family.items()},
        "generated_feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "schema": {
            "at": "at_{feature}",
            "before": "before_{feature}_{stat}",
            "after": "after_{feature}_{stat}",
            "differences": [
                "before_{feature}_mean_minus_at",
                "after_{feature}_mean_minus_at",
                "after_{feature}_mean_minus_before_mean",
            ],
        },
    }
    _write_json(manifest_out, manifest)
    print(f"[build] wrote feature sets: {feature_sets_out}", flush=True)
    print(f"[build] wrote manifest: {manifest_out}", flush=True)


def _run_extract_build(args: argparse.Namespace) -> None:
    if ExtractConfig is None or run_extract is None:
        raise RuntimeError(
            "raw_corr_sweep_runner import failed. extract-build mode requires "
            "ExtractConfig/run_extract from raw_corr_sweep_runner.py."
        )

    raw_out_path: Path
    if args.raw_out_path is not None:
        raw_out_path = args.raw_out_path
        raw_out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="temporal_reduced_raw_"))
        raw_out_path = tmpdir / "raw_features.parquet"

    min_dist_list = _parse_int_tuple(args.min_dist_list, "min-dist-list")
    layer_offsets = _parse_int_tuple(args.attention_layer_offsets, "attention-layer-offsets")
    act_layer_meta: Dict[str, object] = {}
    if args.activation_layer_mode == "layer0_mid_last":
        try:
            act_layer_offsets, act_layer_meta = _activation_offsets_layer0_mid_last(
                args.model_name,
                trust_remote_code=args.trust_remote_code,
            )
            print(
                "[extract-build] activation layers (layer0/mid/last) "
                f"offsets={act_layer_offsets} meta={act_layer_meta}",
                flush=True,
            )
        except Exception as e:
            print(
                "[extract-build] warning: layer0_mid_last resolution failed; "
                f"falling back to explicit offsets '{args.activation_layer_offsets}'. "
                f"reason={type(e).__name__}: {e}",
                flush=True,
            )
            act_layer_offsets = _parse_int_tuple(args.activation_layer_offsets, "activation-layer-offsets")
            act_layer_meta = {
                "activation_layer_mode": "fallback_explicit",
                "activation_layer_offsets_used": act_layer_offsets,
            }
    else:
        act_layer_offsets = _parse_int_tuple(args.activation_layer_offsets, "activation-layer-offsets")
        act_layer_meta = {
            "activation_layer_mode": "explicit",
            "activation_layer_offsets_used": act_layer_offsets,
        }

    cfg = ExtractConfig(
        examples_path=args.examples_path,
        sentences_path=args.sentences_path,
        localization_path=args.localization_path,
        out_path=raw_out_path,
        model_name=args.model_name,
        num_examples=args.num_examples,
        seed=args.seed,
        only_localized=args.only_localized,
        max_tokens=args.max_tokens,
        topk_vocab=args.topk_vocab,
        min_dist_list=min_dist_list,
        attention_layer_offsets=layer_offsets,
        activation_layer_offsets=act_layer_offsets,
        activation_sparsity_eps=args.activation_sparsity_eps,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        progress_every=args.progress_every,
        token_oom_backoff=args.token_oom_backoff,
        token_min_tokens=args.token_min_tokens,
    )

    print("[extract-build] running raw extraction...", flush=True)
    raw_df = run_extract(cfg)
    print(f"[extract-build] raw extraction complete shape={raw_df.shape}", flush=True)

    # Reuse build path from in-memory DataFrame.
    base_by_family, activation_meta = _build_base_by_family_from_args(args, df=raw_df)
    include_diff = not args.no_diff
    sentence_level_stats = _parse_stats_arg(args.sentence_level_stats, "sentence-level-stats")
    temporal_agg_stats = _parse_stats_arg(args.temporal_agg_stats, "temporal-agg-stats")
    out_df, feature_sets, generated_by_family, missing = build_temporal_reduced_features(
        raw_df,
        base_by_family=base_by_family,
        before_window=args.before_window,
        after_window=args.after_window,
        include_diff=include_diff,
        sentence_level_stats=sentence_level_stats,
        temporal_agg_stats=temporal_agg_stats,
    )

    out_path: Path = args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[extract-build] wrote temporal features: {out_path} shape={out_df.shape}", flush=True)

    feature_sets_out = args.feature_sets_out or out_path.with_name(out_path.stem + "_feature_sets.json")
    manifest_out = args.manifest_out or out_path.with_name(out_path.stem + "_manifest.json")
    _write_json(feature_sets_out, feature_sets)
    manifest = {
        "raw_output_path": str(raw_out_path),
        "output_features": str(out_path),
        "before_window": int(args.before_window),
        "after_window": int(args.after_window),
        "include_diff": bool(include_diff),
        "sentence_level_stats": list(sentence_level_stats),
        "temporal_agg_stats": list(temporal_agg_stats),
        "requested_base_features": base_by_family,
        "missing_base_features": missing,
        "activation_extraction": act_layer_meta,
        "activation_selection": activation_meta,
        "generated_feature_counts": {k: len(v) for k, v in generated_by_family.items()},
        "generated_feature_sets": {k: len(v) for k, v in feature_sets.items()},
    }
    _write_json(manifest_out, manifest)
    print(f"[extract-build] wrote feature sets: {feature_sets_out}", flush=True)
    print(f"[extract-build] wrote manifest: {manifest_out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_build = sub.add_parser("build")
    ap_build.add_argument("--input-features", type=Path, required=True)
    ap_build.add_argument("--out-path", type=Path, required=True)
    _add_temporal_args(ap_build)

    ap_extract_build = sub.add_parser("extract-build")
    ap_extract_build.add_argument("--out-path", type=Path, required=True)
    _add_extract_args(ap_extract_build)
    _add_temporal_args(ap_extract_build)

    args = ap.parse_args()
    if args.mode == "build":
        _run_build(args)
        return
    _run_extract_build(args)


if __name__ == "__main__":
    main()
