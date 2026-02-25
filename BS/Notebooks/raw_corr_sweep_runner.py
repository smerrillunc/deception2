#!/usr/bin/env python3
"""
Raw feature mining + correlation strategy sweep.

Design goals:
- Rebuild features directly from raw examples/sentences/localization files.
- Do not read prior cached feature tables.
- Mine structural, lexical, token-logit, and attention-derived per-sentence features.
- Evaluate many transformation strategies by correlation stability (grouped splits).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Ensure BS/src is importable when running from arbitrary cwd.
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from sentence_pipeline import read_jsonl
from extract_sentence_features import _map_tokens_to_sentences, load_localization_history, text_features


WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


LEXICONS: Dict[str, set[str]] = {
    "deceptive_word_frac": {
        "maybe",
        "perhaps",
        "actually",
        "honestly",
        "trust",
        "promise",
        "believe",
        "pretend",
        "fake",
        "bluff",
        "lie",
        "deceive",
        "trick",
    },
    "hedge_word_frac": {
        "maybe",
        "perhaps",
        "probably",
        "possibly",
        "likely",
        "seems",
        "appears",
        "might",
        "could",
    },
    "certainty_word_frac": {
        "definitely",
        "certainly",
        "always",
        "never",
        "must",
        "sure",
        "clearly",
    },
    "negation_word_frac": {
        "no",
        "not",
        "never",
        "none",
        "nothing",
        "neither",
        "nor",
        "without",
    },
    "justification_word_frac": {
        "because",
        "since",
        "therefore",
        "thus",
        "hence",
        "so",
        "reason",
        "why",
    },
    "self_reference_word_frac": {
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "we",
        "our",
        "ours",
    },
    "contradiction_word_frac": {
        "but",
        "however",
        "though",
        "although",
        "yet",
        "instead",
        "otherwise",
        "except",
    },
}


def _safe_entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p + 1e-12)).sum())


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    s = x.sum()
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    cum = np.cumsum(xs)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def _lexicon_fraction(text: str, lexicon: set[str]) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    toks = WORD_RE.findall(text.lower())
    if not toks:
        return 0.0
    hits = sum(1 for w in toks if w in lexicon)
    return float(hits / len(toks))


def _rolling_nan_mean_var(arr: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.size
    out_mean = np.full(n, np.nan, dtype=np.float64)
    out_var = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out_mean, out_var
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        vals = arr[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out_mean[i] = float(vals.mean())
        out_var[i] = float(vals.var())
    return out_mean, out_var


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return np.nan
    xx = x[m]
    yy = y[m]
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return np.nan
    xr = pd.Series(x[m]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y[m]).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def _split_group(groups: np.ndarray, seed: int, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(groups.shape[0])
    tr, te = next(gss.split(idx, groups=groups.astype(str)))
    return tr, te


def _sentence_attention_matrix(attn_tt: "torch.Tensor", token_sent: "torch.Tensor", n_sent: int) -> "torch.Tensor":
    import torch

    valid = token_sent >= 0
    if valid.sum().item() == 0:
        return torch.zeros((n_sent, n_sent), device=attn_tt.device, dtype=torch.float32)

    valid_idx = torch.nonzero(valid, as_tuple=True)[0]
    sent_ids = token_sent[valid]

    counts = torch.bincount(sent_ids, minlength=n_sent).to(torch.float32)
    counts_safe = counts.clone()
    counts_safe[counts_safe == 0] = 1.0

    a_valid = attn_tt[:, valid_idx].to(torch.float32)
    v = torch.zeros((attn_tt.shape[0], n_sent), device=attn_tt.device, dtype=torch.float32)
    v.index_add_(1, sent_ids, a_valid)
    v = v / counts_safe

    vq = v[valid_idx]
    m = torch.zeros((n_sent, n_sent), device=attn_tt.device, dtype=torch.float32)
    m.index_add_(0, sent_ids, vq)
    m = m / counts_safe.view(-1, 1)
    return m


def _attention_rollout_safe(attn_layers: Sequence["torch.Tensor"], add_residual: bool = True) -> "torch.Tensor":
    import torch

    t = attn_layers[0].shape[-1]
    r = torch.eye(t, device=attn_layers[0].device, dtype=torch.float32)
    i = torch.eye(t, device=attn_layers[0].device, dtype=torch.float32)
    eps = 1e-12

    for a in attn_layers:
        abar = a.to(torch.float32).mean(dim=0)
        abar = torch.nan_to_num(abar, nan=0.0, posinf=0.0, neginf=0.0)
        if add_residual:
            abar = abar + i
        rowsum = abar.sum(dim=-1, keepdim=True)
        bad = rowsum.squeeze(-1) <= eps
        if bad.any():
            abar[bad] = 1.0 / t
            rowsum = abar.sum(dim=-1, keepdim=True)
        abar = abar / (rowsum + eps)
        r = abar @ r
        r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    r = r / (r.sum(dim=-1, keepdim=True) + eps)
    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _mine_sentence_matrix(m: np.ndarray, min_dist: int, topk: int = 3) -> Dict[str, np.ndarray]:
    """
    m[q, k] = attention mass from query sentence q to key sentence k.
    """
    n = m.shape[0]
    out = {}

    in_mass = np.zeros(n, dtype=np.float64)
    in_mean = np.zeros(n, dtype=np.float64)
    in_max = np.zeros(n, dtype=np.float64)
    in_entropy = np.zeros(n, dtype=np.float64)
    in_gini = np.zeros(n, dtype=np.float64)
    in_topk_share = np.zeros(n, dtype=np.float64)
    in_dist_mu = np.zeros(n, dtype=np.float64)
    in_dist_var = np.zeros(n, dtype=np.float64)

    out_mass = np.zeros(n, dtype=np.float64)
    out_mean = np.zeros(n, dtype=np.float64)
    out_max = np.zeros(n, dtype=np.float64)
    out_entropy = np.zeros(n, dtype=np.float64)
    out_gini = np.zeros(n, dtype=np.float64)
    out_topk_share = np.zeros(n, dtype=np.float64)
    out_dist_mu = np.zeros(n, dtype=np.float64)
    out_dist_var = np.zeros(n, dtype=np.float64)

    anchor_mass = np.zeros(n, dtype=np.float64)
    anchor_ratio = np.zeros(n, dtype=np.float64)

    for j in range(n):
        q_idx = np.arange(j + min_dist, n, dtype=int)
        if q_idx.size > 0:
            vals = m[q_idx, j]
            vals = np.clip(vals, 0.0, None)
            s = vals.sum()
            in_mass[j] = float(s)
            if vals.size > 0:
                in_mean[j] = float(vals.mean())
                in_max[j] = float(vals.max())
                if s > 0:
                    p = vals / s
                    in_entropy[j] = _safe_entropy_from_probs(p)
                    in_gini[j] = _gini(vals)
                    kk = min(topk, vals.size)
                    in_topk_share[j] = float(np.sort(vals)[-kk:].sum() / s) if kk > 0 else 0.0
                    dist = (q_idx - j).astype(np.float64)
                    in_dist_mu[j] = float((p * dist).sum())
                    in_dist_var[j] = float((p * (dist - in_dist_mu[j]) ** 2).sum())
            anchor_mass[j] = float(m[j, j]) if np.isfinite(m[j, j]) else 0.0
            denom = float(anchor_mass[j] + s + 1e-12)
            anchor_ratio[j] = float(anchor_mass[j] / denom)

        k_idx = np.arange(0, j - min_dist + 1, dtype=int)
        if k_idx.size > 0:
            vals = m[j, k_idx]
            vals = np.clip(vals, 0.0, None)
            s = vals.sum()
            out_mass[j] = float(s)
            if vals.size > 0:
                out_mean[j] = float(vals.mean())
                out_max[j] = float(vals.max())
                if s > 0:
                    p = vals / s
                    out_entropy[j] = _safe_entropy_from_probs(p)
                    out_gini[j] = _gini(vals)
                    kk = min(topk, vals.size)
                    out_topk_share[j] = float(np.sort(vals)[-kk:].sum() / s) if kk > 0 else 0.0
                    dist = (j - k_idx).astype(np.float64)
                    out_dist_mu[j] = float((p * dist).sum())
                    out_dist_var[j] = float((p * (dist - out_dist_mu[j]) ** 2).sum())

    out["in_long_mass"] = in_mass
    out["in_long_mean"] = in_mean
    out["in_long_max"] = in_max
    out["in_long_entropy"] = in_entropy
    out["in_long_gini"] = in_gini
    out["in_long_topk_share"] = in_topk_share
    out["in_long_dist_mu"] = in_dist_mu
    out["in_long_dist_var"] = in_dist_var

    out["out_long_mass"] = out_mass
    out["out_long_mean"] = out_mean
    out["out_long_max"] = out_max
    out["out_long_entropy"] = out_entropy
    out["out_long_gini"] = out_gini
    out["out_long_topk_share"] = out_topk_share
    out["out_long_dist_mu"] = out_dist_mu
    out["out_long_dist_var"] = out_dist_var

    out["anchor_mass"] = anchor_mass
    out["anchor_ratio"] = anchor_ratio
    out["entropy_in_minus_out"] = in_entropy - out_entropy
    out["distmu_in_minus_out"] = in_dist_mu - out_dist_mu
    return out


def _aggregate_tok_by_sentence(values: np.ndarray, sent_idx: np.ndarray, n_sent: int, stat: str) -> np.ndarray:
    out = np.full(n_sent, np.nan, dtype=np.float64)
    for s in range(n_sent):
        m = sent_idx == s
        v = values[m]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        if stat == "mean":
            out[s] = float(v.mean())
        elif stat == "max":
            out[s] = float(v.max())
        elif stat == "min":
            out[s] = float(v.min())
        elif stat == "std":
            out[s] = float(v.std())
        elif stat == "sum":
            out[s] = float(v.sum())
        elif stat == "p90":
            out[s] = float(np.quantile(v, 0.90))
        elif stat == "p95":
            out[s] = float(np.quantile(v, 0.95))
    return out


def _compute_token_and_attention_features(
    raw_text: str,
    sentence_spans: List[Dict[str, Any]],
    *,
    tokenizer,
    model,
    device: str,
    max_tokens: int,
    topk_vocab: int,
    min_dist_list: Sequence[int],
    attention_layer_offsets: Sequence[int],
    activation_layer_offsets: Sequence[int],
    activation_sparsity_eps: float,
) -> Dict[int, Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    if not raw_text or not sentence_spans:
        return {}

    enc = tokenizer(
        raw_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True if max_tokens > 0 else False,
        max_length=max_tokens if max_tokens > 0 else None,
    )

    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()
    t = int(input_ids.shape[0])
    n_sent = len(sentence_spans)
    if t < 4 or n_sent == 0:
        return {}

    tok_to_sent = np.asarray(_map_tokens_to_sentences(offsets, sentence_spans), dtype=int)
    if tok_to_sent.size != t:
        return {}

    model_ids = input_ids.unsqueeze(0).to(device)
    with torch.inference_mode():
        autocast_ctx = torch.amp.autocast("cuda", enabled=False) if str(device).startswith("cuda") else nullcontext()
        with autocast_ctx:
            out = model(
                model_ids,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
            )

    logits = getattr(out, "logits", None)
    if logits is None:
        return {}
    logits = logits[0].to(torch.float32)  # [T, V]
    attn_layers = getattr(out, "attentions", None)
    hidden_states = getattr(out, "hidden_states", None) or ()
    if logits.shape[0] != t:
        return {}

    # Token-level metrics aligned to token position.
    nll_tok = np.full(t, np.nan, dtype=np.float64)
    ent_tok = np.full(t, np.nan, dtype=np.float64)
    ent_topk_tok = np.full(t, np.nan, dtype=np.float64)
    margin_tok = np.full(t, np.nan, dtype=np.float64)
    pmax_tok = np.full(t, np.nan, dtype=np.float64)
    logit_std_tok = np.full(t, np.nan, dtype=np.float64)

    logits_next = logits[:-1, :]  # predict token at position i+1
    targets = input_ids[1:].to(device)
    nll = F.cross_entropy(logits_next, targets, reduction="none").detach().cpu().numpy()
    nll_tok[1:] = nll

    logz = torch.logsumexp(logits_next, dim=-1, keepdim=True)
    k = min(max(2, topk_vocab), logits_next.shape[-1])
    topv, _ = torch.topk(logits_next, k=k, dim=-1)

    p_top = torch.exp(topv - logz)
    p_top_sum = torch.clamp(p_top.sum(dim=-1), min=1e-9, max=1.0)
    p_other = torch.clamp(1.0 - p_top_sum, min=1e-12)

    ent_approx = -(p_top * (topv - logz)).sum(dim=-1) - p_other * torch.log(p_other)
    ent_tok[1:] = ent_approx.detach().cpu().numpy()

    p_top_renorm = p_top / p_top_sum.unsqueeze(-1)
    ent_topk = -(p_top_renorm * torch.log(torch.clamp(p_top_renorm, min=1e-12))).sum(dim=-1)
    ent_topk_tok[1:] = ent_topk.detach().cpu().numpy()

    top2, _ = torch.topk(logits_next, k=2, dim=-1)
    margin = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy()
    margin_tok[1:] = margin

    pmax = torch.exp(top2[:, 0:1] - logz).squeeze(-1).detach().cpu().numpy()
    pmax_tok[1:] = pmax

    logit_std = logits_next.std(dim=-1).detach().cpu().numpy()
    logit_std_tok[1:] = logit_std

    ent_delta = np.full(t, np.nan, dtype=np.float64)
    ent_posdiff = np.full(t, np.nan, dtype=np.float64)
    if t >= 3:
        d = ent_tok[2:] - ent_tok[1:-1]
        ent_delta[2:] = d
        ent_posdiff[2:] = np.maximum(d, 0.0)

    roll_mean, roll_var = _rolling_nan_mean_var(ent_tok, radius=3)

    # Map token metrics to sentence aggregates.
    rows: Dict[int, Dict[str, Any]] = {i: {} for i in range(n_sent)}
    mapped = tok_to_sent >= 0
    token_sent = tok_to_sent.copy()
    token_sent[~mapped] = -1

    tok_count = np.bincount(token_sent[mapped], minlength=n_sent).astype(int)
    tok_count_frac = tok_count.astype(np.float64) / max(int(mapped.sum()), 1)

    for s in range(n_sent):
        rows[s]["tok_count"] = int(tok_count[s])
        rows[s]["tok_count_frac"] = float(tok_count_frac[s])

    metrics = {
        "tok_nll": nll_tok,
        "tok_entropy": ent_tok,
        "tok_entropy_topk_renorm": ent_topk_tok,
        "tok_margin_logit": margin_tok,
        "tok_pmax": pmax_tok,
        "tok_logit_std": logit_std_tok,
        "tok_entropy_delta": ent_delta,
        "tok_entropy_posdiff": ent_posdiff,
        "tok_entropy_roll_mean": roll_mean,
        "tok_entropy_roll_var": roll_var,
    }

    for name, arr in metrics.items():
        for stat in ("mean", "max", "min", "std"):
            vals = _aggregate_tok_by_sentence(arr, token_sent, n_sent, stat)
            col = f"{name}_{stat}"
            for s in range(n_sent):
                if np.isfinite(vals[s]):
                    rows[s][col] = float(vals[s])

    # Activation-level features from hidden states.
    if hidden_states:
        hs_len = len(hidden_states)
        act_layers: List[Tuple[str, int, "torch.Tensor"]] = []
        seen_idx = set()
        for off in activation_layer_offsets:
            idx = hs_len + off if off < 0 else off
            if 0 <= idx < hs_len and idx not in seen_idx:
                seen_idx.add(idx)
                tag = f"m{abs(int(off))}" if off < 0 else f"p{int(off)}"
                act_layers.append((tag, idx, hidden_states[idx][0].to(torch.float32)))

        if act_layers:
            valid_mask = token_sent >= 0
            valid_idx_t = torch.nonzero(torch.tensor(valid_mask, device=device), as_tuple=True)[0]
            sent_ids_t = torch.tensor(token_sent[valid_mask], device=device, dtype=torch.long)
            counts = torch.bincount(sent_ids_t, minlength=n_sent).to(torch.float32)
            counts_safe = torch.clamp(counts, min=1.0)

            sent_emb_by_layer: List[Tuple[str, int, np.ndarray]] = []
            for tag, idx, h in act_layers:
                # Token-level activation statistics.
                tok_l2 = torch.linalg.norm(h, ord=2, dim=-1).detach().cpu().numpy()
                tok_abs = torch.mean(torch.abs(h), dim=-1).detach().cpu().numpy()
                tok_sparse = torch.mean((torch.abs(h) <= float(activation_sparsity_eps)).to(torch.float32), dim=-1).detach().cpu().numpy()

                for name, arr in (
                    (f"act_{tag}_l2", tok_l2),
                    (f"act_{tag}_absmean", tok_abs),
                    (f"act_{tag}_sparse", tok_sparse),
                ):
                    for stat in ("mean", "max", "std", "min"):
                        vals = _aggregate_tok_by_sentence(arr, token_sent, n_sent, stat)
                        col = f"{name}_{stat}"
                        for s in range(n_sent):
                            if np.isfinite(vals[s]):
                                rows[s][col] = float(vals[s])

                # Sentence embedding per layer for drift/trajectory features.
                sum_emb = torch.zeros((n_sent, h.shape[-1]), device=device, dtype=torch.float32)
                if valid_idx_t.numel() > 0:
                    sum_emb.index_add_(0, sent_ids_t, h[valid_idx_t])
                emb = sum_emb / counts_safe.view(-1, 1)
                emb_np = emb.detach().cpu().numpy()
                sent_emb_by_layer.append((tag, idx, emb_np))

            # Pairwise layer drift features using consecutive layers in depth order.
            sent_emb_by_layer.sort(key=lambda x: x[1])
            for i_pair in range(1, len(sent_emb_by_layer)):
                tag_a, idx_a, ea = sent_emb_by_layer[i_pair - 1]
                tag_b, idx_b, eb = sent_emb_by_layer[i_pair]
                na = np.linalg.norm(ea, axis=1)
                nb = np.linalg.norm(eb, axis=1)
                dot = np.sum(ea * eb, axis=1)
                cos = dot / (na * nb + 1e-8)
                l2d = np.linalg.norm(eb - ea, axis=1)
                ratio = nb / (na + 1e-8)
                for s in range(n_sent):
                    rows[s][f"act_pair_{tag_a}_{tag_b}_cos"] = float(cos[s])
                    rows[s][f"act_pair_{tag_a}_{tag_b}_l2diff"] = float(l2d[s])
                    rows[s][f"act_pair_{tag_a}_{tag_b}_ratio"] = float(ratio[s])

            # Depth-trajectory features on sentence embedding energies.
            if len(sent_emb_by_layer) >= 2:
                layer_pos = np.asarray([idx for _, idx, _ in sent_emb_by_layer], dtype=np.float64)
                if np.std(layer_pos) < 1e-8:
                    layer_pos = np.arange(len(sent_emb_by_layer), dtype=np.float64)
                layer_pos = (layer_pos - layer_pos.min()) / (layer_pos.max() - layer_pos.min() + 1e-8)

                energies = np.stack([np.linalg.norm(e, axis=1) for _, _, e in sent_emb_by_layer], axis=0)  # [L, S]
                for s in range(n_sent):
                    ys = energies[:, s]
                    if not np.isfinite(ys).all():
                        continue
                    try:
                        slope = float(np.polyfit(layer_pos, ys, 1)[0])
                    except Exception:
                        slope = np.nan
                    curv = float(np.mean(np.diff(ys, n=2))) if ys.size >= 3 else np.nan
                    rows[s]["act_traj_energy_slope"] = slope
                    rows[s]["act_traj_energy_curv"] = curv
                    rows[s]["act_traj_energy_range"] = float(np.max(ys) - np.min(ys))
                    rows[s]["act_traj_energy_last_first"] = float(ys[-1] - ys[0])
                    rows[s]["act_traj_energy_std"] = float(np.std(ys))

    # Attention features.
    if not attn_layers:
        return rows

    layer_ids = []
    l_total = len(attn_layers)
    for off in attention_layer_offsets:
        idx = l_total + off if off < 0 else off
        if 0 <= idx < l_total:
            layer_ids.append(idx)
    if not layer_ids:
        layer_ids = [l_total - 1]

    token_sent_t = torch.tensor(token_sent, device=device, dtype=torch.long)

    # Raw mean/max from last selected layer.
    last = attn_layers[layer_ids[-1]][0].to(torch.float32)  # [H,T,T]
    m_rawmean = _sentence_attention_matrix(last.mean(dim=0), token_sent_t, n_sent).detach().cpu().numpy()
    m_rawmax = _sentence_attention_matrix(last.max(dim=0).values, token_sent_t, n_sent).detach().cpu().numpy()

    # Rollout across selected layers.
    roll_layers = [attn_layers[i][0].detach() for i in layer_ids]
    m_roll = _sentence_attention_matrix(_attention_rollout_safe(roll_layers, add_residual=True), token_sent_t, n_sent)
    m_roll = m_roll.detach().cpu().numpy()

    mats = {
        "attn_rawmean": m_rawmean,
        "attn_rawmax": m_rawmax,
        "attn_roll": m_roll,
    }

    for chan, mm in mats.items():
        mm = np.nan_to_num(mm, nan=0.0, posinf=0.0, neginf=0.0)
        for d in min_dist_list:
            mined = _mine_sentence_matrix(mm, min_dist=int(d), topk=3)
            for feat, arr in mined.items():
                col = f"{chan}__d{int(d)}__{feat}"
                for s in range(n_sent):
                    rows[s][col] = float(arr[s])

    return rows


@dataclass
class ExtractConfig:
    examples_path: Path
    sentences_path: Path
    localization_path: Path
    out_path: Path
    model_name: str
    num_examples: int
    seed: int
    only_localized: bool
    max_tokens: int
    topk_vocab: int
    min_dist_list: Tuple[int, ...]
    attention_layer_offsets: Tuple[int, ...]
    activation_layer_offsets: Tuple[int, ...]
    activation_sparsity_eps: float
    device: str
    trust_remote_code: bool
    progress_every: int
    token_oom_backoff: float = 0.5
    token_min_tokens: int = 256


def run_extract(cfg: ExtractConfig) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if not (0.0 < float(cfg.token_oom_backoff) < 1.0):
        raise ValueError("token_oom_backoff must be in (0, 1).")
    if int(cfg.token_min_tokens) <= 0:
        raise ValueError("token_min_tokens must be positive.")

    print(f"[extract] loading raw inputs from {cfg.examples_path.parent}", flush=True)
    examples = [x for x in read_jsonl(cfg.examples_path) if x.get("example_id")]
    sentences = [x for x in read_jsonl(cfg.sentences_path) if x.get("example_id")]
    loc_map = load_localization_history(str(cfg.localization_path))

    by_example: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sentences:
        by_example[s["example_id"]].append(s)
    for ex_id in by_example:
        by_example[ex_id].sort(key=lambda r: int(r.get("sentence_idx", 0)))

    ex_by_id = {e["example_id"]: e for e in examples if e["example_id"] in by_example}
    example_ids = list(ex_by_id.keys())
    if cfg.num_examples > 0 and cfg.num_examples < len(example_ids):
        example_ids = random.sample(example_ids, cfg.num_examples)
    example_ids = sorted(example_ids)
    print(f"[extract] selected examples={len(example_ids)}", flush=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "7")

    print(f"[extract] loading model/tokenizer: {cfg.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
        trust_remote_code=cfg.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=cfg.trust_remote_code,
    )
    model = model.to(dtype=torch.float32).to(cfg.device)
    model.eval()
    param_dtypes = {p.dtype for p in model.parameters()}
    tok_fp32_enforced = bool(param_dtypes == {torch.float32})
    if not tok_fp32_enforced:
        raise RuntimeError(f"Strict FP32 enforcement failed. Found model parameter dtypes: {sorted(str(x) for x in param_dtypes)}")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    runtime_max_tokens = int(cfg.max_tokens)

    for i, ex_id in enumerate(example_ids, start=1):
        ex = ex_by_id.get(ex_id)
        sents = by_example.get(ex_id, [])
        if not ex or not sents:
            skipped += 1
            continue

        loc_hist = loc_map.get(ex_id, {})
        rates = {int(k): (v.get("deception_rate") if isinstance(v, dict) else None) for k, v in loc_hist.items()}
        if cfg.only_localized and not rates:
            skipped += 1
            continue

        raw_text = ex.get("action_reasoning") or ex.get("action_raw_text") or ""
        sentence_spans = [{"start": s.get("start"), "end": s.get("end")} for s in sents]

        mined = {}
        cur_max_tokens = int(runtime_max_tokens)
        tok_max_tokens_used = int(runtime_max_tokens)
        tok_oom_retries = 0
        try:
            while True:
                try:
                    mined = _compute_token_and_attention_features(
                        raw_text,
                        sentence_spans,
                        tokenizer=tokenizer,
                        model=model,
                        device=cfg.device,
                        max_tokens=cur_max_tokens,
                        topk_vocab=cfg.topk_vocab,
                        min_dist_list=cfg.min_dist_list,
                        attention_layer_offsets=cfg.attention_layer_offsets,
                        activation_layer_offsets=cfg.activation_layer_offsets,
                        activation_sparsity_eps=cfg.activation_sparsity_eps,
                    )
                    tok_max_tokens_used = int(cur_max_tokens)
                    runtime_max_tokens = int(cur_max_tokens)
                    break
                except RuntimeError as oom_err:
                    msg = str(oom_err).lower()
                    is_oom = ("out of memory" in msg) or ("cuda error: out of memory" in msg) or ("cuda out of memory" in msg)
                    can_retry = (
                        cfg.device.startswith("cuda")
                        and is_oom
                        and cur_max_tokens > 0
                        and cur_max_tokens > int(cfg.token_min_tokens)
                    )
                    if not can_retry:
                        raise
                    tok_oom_retries += 1
                    next_max = max(int(cfg.token_min_tokens), int(cur_max_tokens * float(cfg.token_oom_backoff)))
                    if next_max >= cur_max_tokens:
                        next_max = cur_max_tokens - 1
                    if next_max < int(cfg.token_min_tokens):
                        raise
                    print(
                        f"[extract] OOM ex={ex_id} retry={tok_oom_retries} max_tokens {cur_max_tokens} -> {next_max}",
                        flush=True,
                    )
                    cur_max_tokens = next_max
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        except Exception as e:
            skipped += 1
            if i % cfg.progress_every == 0:
                print(f"[extract] warning ex={ex_id} failed: {type(e).__name__}", flush=True)
            continue

        total_sent = len(sents)
        for s in sents:
            sidx = int(s.get("sentence_idx", 0))
            rate = rates.get(sidx)
            if cfg.only_localized and rate is None:
                continue

            text = s.get("sentence_text", "") or ""
            row: Dict[str, Any] = {
                "example_id": ex_id,
                "sentence_idx": sidx,
                "sentence_position": (sidx / max(total_sent - 1, 1)),
                "total_sentences": total_sent,
                "deception_rate": rate,
                "tok_max_tokens_used": int(tok_max_tokens_used),
                "tok_oom_retries": int(tok_oom_retries),
                "tok_fp32_enforced": bool(tok_fp32_enforced),
            }
            row.update(text_features(text))
            for name, lex in LEXICONS.items():
                row[name] = _lexicon_fraction(text, lex)

            if mined and sidx in mined:
                row.update(mined[sidx])

            rows.append(row)

        if i % cfg.progress_every == 0:
            print(
                f"[extract] progress {i}/{len(example_ids)} rows={len(rows)} skipped={skipped}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows extracted. Check localization coverage / inputs.")

    if "tok_count" in df.columns:
        tok_count = pd.to_numeric(df["tok_count"], errors="coerce").fillna(0.0)
        tokenless_frac = float((tok_count <= 0.0).mean())
        print(
            f"[extract] token coverage: tokenless_sentence_frac={tokenless_frac:.4f} | "
            f"median_tok_count={float(tok_count.median()):.2f}",
            flush=True,
        )

    # Within-example zscore for attention columns.
    attn_cols = [c for c in df.columns if c.startswith("attn_")]
    z_data: Dict[str, pd.Series] = {}
    for c in attn_cols:
        g = df.groupby("example_id")[c]
        mu = g.transform("mean")
        sd = g.transform("std")
        z_data[c + "__z"] = (pd.to_numeric(df[c], errors="coerce") - mu) / (sd + 1e-8)
    if z_data:
        df = pd.concat([df, pd.DataFrame(z_data, index=df.index)], axis=1)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.out_path, index=False)
    print(f"[extract] wrote {cfg.out_path} shape={df.shape}", flush=True)
    return df


def _rank_within_example(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    grp = df.groupby("example_id", sort=False)
    for c in cols:
        out[c + "__rk"] = grp[c].rank(method="average", pct=True)
    return out


def _z_within_example(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    grp = df.groupby("example_id", sort=False)
    for c in cols:
        mu = grp[c].transform("mean")
        sd = grp[c].transform("std")
        out[c + "__z2"] = (df[c] - mu) / (sd + 1e-8)
    return out


def _delta_prev(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    grp = df.groupby("example_id", sort=False)
    for c in cols:
        prev = grp[c].shift(1)
        d = df[c] - prev
        out[c + "__d1"] = d
        out[c + "__absd1"] = d.abs()
    return out


def _rolling_prev(df: pd.DataFrame, cols: List[str], k: int) -> pd.DataFrame:
    out_data: Dict[str, pd.Series] = {}
    grp = df.groupby("example_id", sort=False)
    for c in cols:
        shifted = grp[c].shift(1)
        out_data[f"{c}__prev{k}__mean"] = shifted.groupby(df["example_id"]).rolling(k, min_periods=1).mean().reset_index(level=0, drop=True)
        out_data[f"{c}__prev{k}__max"] = shifted.groupby(df["example_id"]).rolling(k, min_periods=1).max().reset_index(level=0, drop=True)
        out_data[f"{c}__prev{k}__min"] = shifted.groupby(df["example_id"]).rolling(k, min_periods=1).min().reset_index(level=0, drop=True)
        out_data[f"{c}__prev{k}__std"] = shifted.groupby(df["example_id"]).rolling(k, min_periods=2).std().reset_index(level=0, drop=True)
    if not out_data:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(out_data, index=df.index)


def _future_next(df: pd.DataFrame, cols: List[str], k: int) -> pd.DataFrame:
    out_data: Dict[str, pd.Series] = {}
    grp = df.groupby("example_id", sort=False)
    for c in cols:
        nxt = grp[c].shift(-1)
        out_data[f"{c}__next{k}__mean"] = (
            nxt.iloc[::-1].groupby(df["example_id"].iloc[::-1]).rolling(k, min_periods=1).mean().reset_index(level=0, drop=True).iloc[::-1]
        )
    if not out_data:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(out_data, index=df.index)


def _interaction_bundle(df: pd.DataFrame, attn_cols: List[str], tok_cols: List[str], n_pairs: int = 10) -> pd.DataFrame:
    out_data: Dict[str, pd.Series] = {}
    if not attn_cols or not tok_cols:
        return pd.DataFrame(index=df.index)
    attn_pick = attn_cols[: min(len(attn_cols), n_pairs)]
    tok_pick = tok_cols[: min(len(tok_cols), n_pairs)]
    for a in attn_pick:
        for t in tok_pick:
            out_data[f"int__{a}__x__{t}"] = df[a] * df[t]
    if not out_data:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(out_data, index=df.index)


def _sanitize_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan)


def _build_position_design(df: pd.DataFrame, knots: Sequence[float]) -> np.ndarray:
    p = pd.to_numeric(df["sentence_position"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)
    ts = pd.to_numeric(df["total_sentences"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    log_ts = np.log1p(np.clip(ts, 0.0, None))

    cols = [
        np.ones_like(p),
        p,
        p**2,
        p**3,
        log_ts,
        p * log_ts,
        (p**2) * log_ts,
    ]
    for k in knots:
        kk = float(k)
        if kk <= 0.0 or kk >= 1.0:
            continue
        cols.append(np.maximum(0.0, p - kk))
    z = np.stack(cols, axis=1)
    return z


def _residualize_matrix(x: np.ndarray, z: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    # x: [n, m], z: [n, d]
    xt = np.asarray(x, dtype=np.float64)
    zt = np.asarray(z, dtype=np.float64)

    # Fill NaNs in x by column means for stable linear projection.
    col_mean = np.nanmean(xt, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    inds = np.where(~np.isfinite(xt))
    xt2 = xt.copy()
    xt2[inds] = np.take(col_mean, inds[1])

    ztz = zt.T @ zt
    d = ztz.shape[0]
    ztz = ztz + ridge * np.eye(d, dtype=np.float64)
    beta = np.linalg.solve(ztz, zt.T @ xt2)
    resid = xt2 - zt @ beta
    return resid


def _residualize_vector(y: np.ndarray, z: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    yt = np.asarray(y, dtype=np.float64)
    yfill = yt.copy()
    if not np.isfinite(yfill).all():
        mu = np.nanmean(yfill)
        if not np.isfinite(mu):
            mu = 0.0
        yfill[~np.isfinite(yfill)] = mu
    ztz = z.T @ z + ridge * np.eye(z.shape[1], dtype=np.float64)
    beta = np.linalg.solve(ztz, z.T @ yfill)
    return yfill - z @ beta


def _corr_table(df: pd.DataFrame, cols: List[str], y_col: str = "deception_rate") -> pd.DataFrame:
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    rows = []
    xdf = _sanitize_numeric(df, cols)
    for c in cols:
        x = xdf[c].to_numpy(dtype=float)
        p = _safe_corr(x, y)
        s = _safe_spearman(x, y)
        n = int((np.isfinite(x) & np.isfinite(y)).sum())
        rows.append(
            {
                "feature": c,
                "pearson": p,
                "abs_pearson": abs(p) if np.isfinite(p) else np.nan,
                "spearman": s,
                "abs_spearman": abs(s) if np.isfinite(s) else np.nan,
                "n": n,
            }
        )
    out = pd.DataFrame(rows).sort_values("abs_pearson", ascending=False)
    return out.reset_index(drop=True)


@dataclass
class SweepContext:
    df: pd.DataFrame
    structural_cols: List[str]
    lexical_cols: List[str]
    token_cols: List[str]
    activation_cols: List[str]
    attn_cols: List[str]
    top_token: List[str]
    top_activation: List[str]
    top_attn: List[str]
    top_lex: List[str]


def _build_sweep_context(df: pd.DataFrame, *, drop_position_features: bool = False) -> SweepContext:
    structural_cols = [
        c
        for c in [
            "sentence_position",
            "total_sentences",
            "char_count",
            "word_count",
            "digit_count",
            "upper_ratio",
            "avg_word_len",
            "negation_count",
            "punct_period",
            "punct_comma",
            "punct_qmark",
            "punct_exclaim",
            "punct_colon",
            "punct_semicolon",
        ]
        if c in df.columns
    ]
    if drop_position_features:
        structural_cols = [c for c in structural_cols if c not in {"sentence_position", "sentence_idx"}]
    lexical_cols = [c for c in df.columns if c.endswith("_word_frac")]
    token_cols = [c for c in df.columns if c.startswith("tok_")]
    activation_cols = [c for c in df.columns if c.startswith("act_")]
    attn_cols = [c for c in df.columns if c.startswith("attn_")]

    base = df.copy()
    y = pd.to_numeric(base["deception_rate"], errors="coerce").to_numpy(dtype=float)

    def top_by_corr(cols: List[str], k: int) -> List[str]:
        if not cols:
            return []
        xdf = _sanitize_numeric(base, cols)
        scores = []
        for c in cols:
            p = _safe_corr(xdf[c].to_numpy(dtype=float), y)
            scores.append((c, abs(p) if np.isfinite(p) else -1.0))
        scores.sort(key=lambda t: t[1], reverse=True)
        return [c for c, s in scores[:k] if s >= 0]

    top_token = top_by_corr(token_cols, k=16)
    top_activation = top_by_corr(activation_cols, k=16)
    top_attn = top_by_corr(attn_cols, k=16)
    top_lex = top_by_corr(lexical_cols + structural_cols, k=12)

    return SweepContext(
        df=df,
        structural_cols=structural_cols,
        lexical_cols=lexical_cols,
        token_cols=token_cols,
        activation_cols=activation_cols,
        attn_cols=attn_cols,
        top_token=top_token,
        top_activation=top_activation,
        top_attn=top_attn,
        top_lex=top_lex,
    )


def _strategy_columns(name: str, ctx: SweepContext) -> Tuple[pd.DataFrame, List[str]]:
    df = ctx.df
    base_sl = list(dict.fromkeys(ctx.structural_cols + ctx.lexical_cols))
    base_tok = list(dict.fromkeys(ctx.top_token if ctx.top_token else ctx.token_cols[:24]))
    base_act = list(dict.fromkeys(ctx.top_activation if ctx.top_activation else ctx.activation_cols[:24]))
    base_dyn = list(dict.fromkeys(base_tok + base_act))
    base_attn = list(dict.fromkeys(ctx.top_attn if ctx.top_attn else ctx.attn_cols[:24]))

    if name == "raw_struct_lex":
        cols = base_sl
        return df, cols
    if name == "raw_token":
        cols = base_dyn
        return df, cols
    if name == "raw_attention":
        cols = base_attn
        return df, cols
    if name == "raw_token_attention":
        cols = list(dict.fromkeys(base_dyn + base_attn + ctx.top_lex))
        return df, cols
    if name == "z_token_attention":
        extra = _z_within_example(df, list(dict.fromkeys(base_dyn + base_attn)))
        out = pd.concat([df, extra], axis=1)
        return out, list(extra.columns)
    if name == "rank_token_attention":
        extra = _rank_within_example(df, list(dict.fromkeys(base_dyn + base_attn)))
        out = pd.concat([df, extra], axis=1)
        return out, list(extra.columns)
    if name == "delta_token_attention":
        extra = _delta_prev(df, list(dict.fromkeys(base_dyn + base_attn)))
        out = pd.concat([df, extra], axis=1)
        return out, list(extra.columns)
    if name == "prev3_token_attention":
        extra = _rolling_prev(df, list(dict.fromkeys(base_dyn + base_attn)), k=3)
        out = pd.concat([df, extra], axis=1)
        return out, list(extra.columns)
    if name == "prev5_token_attention":
        extra = _rolling_prev(df, list(dict.fromkeys(base_dyn + base_attn)), k=5)
        out = pd.concat([df, extra], axis=1)
        return out, list(extra.columns)
    if name == "future3_attention":
        extra = _future_next(df, base_attn, k=3)
        out = pd.concat([df, extra], axis=1)
        return out, list(extra.columns)
    if name == "interaction_attn_tok":
        extra = _interaction_bundle(df, base_attn, base_dyn, n_pairs=8)
        out = pd.concat([df, extra], axis=1)
        return out, list(extra.columns)
    if name == "hybrid_all":
        zf = _z_within_example(df, list(dict.fromkeys(base_dyn + base_attn)))
        df1 = pd.concat([df, zf], axis=1)
        df1_delta = _delta_prev(df1, list(dict.fromkeys(base_dyn + base_attn)))
        df1_prev3 = _rolling_prev(df1, list(dict.fromkeys(base_dyn + base_attn)), k=3)
        out = pd.concat([df1, df1_delta, df1_prev3], axis=1)
        cols = list(dict.fromkeys(base_sl + list(zf.columns) + list(df1_delta.columns) + list(df1_prev3.columns)))
        return out, cols

    raise ValueError(f"Unknown strategy: {name}")


def _evaluate_strategy(
    name: str,
    ctx: SweepContext,
    seeds: List[int],
    topk: int,
    *,
    deconfound_position: bool = False,
    deconfound_knots: Sequence[float] = (),
    residualize_target: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dfx, cols = _strategy_columns(name, ctx)
    if not cols:
        return pd.DataFrame(), {"strategy": name, "n_features": 0}

    y = pd.to_numeric(dfx["deception_rate"], errors="coerce").to_numpy(dtype=float)
    groups = dfx["example_id"].astype(str).to_numpy()
    x = _sanitize_numeric(dfx, cols)
    x_mat = x.to_numpy(dtype=np.float64)
    y_used = y.copy()

    pos_diag_mean = np.nan
    pos_diag_max = np.nan
    y_pos_abs_corr = np.nan
    if deconfound_position:
        z = _build_position_design(dfx, knots=deconfound_knots)
        x_mat = _residualize_matrix(x_mat, z)
        if residualize_target:
            y_used = _residualize_vector(y_used, z)
        # Diagnostics: residual association with sentence_position should be tiny.
        pos = pd.to_numeric(dfx["sentence_position"], errors="coerce").to_numpy(dtype=float)
        pos_corrs = []
        for j in range(x_mat.shape[1]):
            c = _safe_corr(x_mat[:, j], pos)
            if np.isfinite(c):
                pos_corrs.append(abs(c))
        if pos_corrs:
            pos_diag_mean = float(np.mean(pos_corrs))
            pos_diag_max = float(np.max(pos_corrs))
        yc = _safe_corr(y_used, pos)
        y_pos_abs_corr = abs(yc) if np.isfinite(yc) else np.nan
    x = pd.DataFrame(x_mat, columns=cols, index=dfx.index)

    corr_src = pd.concat([pd.DataFrame({"deception_rate": y_used}, index=dfx.index), x], axis=1)
    corr_full = _corr_table(corr_src, cols)
    corr_full["strategy"] = name

    seed_scores = []
    for sd in seeds:
        tr, te = _split_group(groups, seed=sd, test_size=0.2)
        ytr = y_used[tr]
        yte = y_used[te]

        tr_scores = []
        te_scores = []
        for c in cols:
            xv = x[c].to_numpy(dtype=float)
            p_tr = _safe_corr(xv[tr], ytr)
            p_te = _safe_corr(xv[te], yte)
            tr_scores.append((c, abs(p_tr) if np.isfinite(p_tr) else -1.0))
            te_scores.append((c, abs(p_te) if np.isfinite(p_te) else np.nan))

        tr_scores.sort(key=lambda t: t[1], reverse=True)
        pick = [c for c, _ in tr_scores[: min(topk, len(tr_scores))]]
        te_map = {c: s for c, s in te_scores}
        picked_te = [te_map[c] for c in pick if np.isfinite(te_map.get(c, np.nan))]
        mean_abs_te = float(np.mean(picked_te)) if picked_te else np.nan
        best_abs_te = float(np.max(picked_te)) if picked_te else np.nan
        seed_scores.append(
            {
                "seed": sd,
                "mean_abs_test_corr_topk": mean_abs_te,
                "best_abs_test_corr_topk": best_abs_te,
            }
        )

    summary = {
        "strategy": name,
        "n_features": len(cols),
        "top_feature_full": corr_full.iloc[0]["feature"] if len(corr_full) else None,
        "top_feature_full_abs_pearson": float(corr_full.iloc[0]["abs_pearson"]) if len(corr_full) else np.nan,
        "mean_abs_test_corr_topk_mean": float(np.nanmean([r["mean_abs_test_corr_topk"] for r in seed_scores])) if seed_scores else np.nan,
        "mean_abs_test_corr_topk_std": float(np.nanstd([r["mean_abs_test_corr_topk"] for r in seed_scores])) if seed_scores else np.nan,
        "best_abs_test_corr_topk_mean": float(np.nanmean([r["best_abs_test_corr_topk"] for r in seed_scores])) if seed_scores else np.nan,
        "seeds": ",".join(str(s) for s in seeds),
        "deconfound_position": bool(deconfound_position),
        "residualize_target": bool(residualize_target),
        "mean_abs_feature_pos_corr": pos_diag_mean,
        "max_abs_feature_pos_corr": pos_diag_max,
        "abs_target_pos_corr": y_pos_abs_corr,
    }
    return corr_full, summary


ALL_STRATEGIES = [
    "raw_struct_lex",
    "raw_token",
    "raw_attention",
    "raw_token_attention",
    "z_token_attention",
    "rank_token_attention",
    "delta_token_attention",
    "prev3_token_attention",
    "prev5_token_attention",
    "future3_attention",
    "interaction_attn_tok",
    "hybrid_all",
]


def run_sweep(
    data_path: Path,
    out_dir: Path,
    tag: str,
    strategies: List[str],
    seeds: List[int],
    topk: int,
    *,
    deconfound_position: bool = False,
    deconfound_knots: Sequence[float] = (),
    residualize_target: bool = False,
) -> None:
    print(f"[sweep] loading data: {data_path}", flush=True)
    df = pd.read_parquet(data_path)
    df = df.copy()
    df["deception_rate"] = pd.to_numeric(df["deception_rate"], errors="coerce")
    df = df[np.isfinite(df["deception_rate"])].reset_index(drop=True)
    print(f"[sweep] rows={len(df)} examples={df['example_id'].nunique()} cols={df.shape[1]}", flush=True)
    if deconfound_position:
        print(
            f"[sweep] positional deconfounding enabled | residualize_target={residualize_target} | knots={list(deconfound_knots)}",
            flush=True,
        )

    ctx = _build_sweep_context(df, drop_position_features=deconfound_position)
    detail_frames = []
    summaries = []
    for s in strategies:
        print(f"[sweep] strategy={s}", flush=True)
        detail, summary = _evaluate_strategy(
            s,
            ctx,
            seeds=seeds,
            topk=topk,
            deconfound_position=deconfound_position,
            deconfound_knots=deconfound_knots,
            residualize_target=residualize_target,
        )
        if len(detail):
            detail_frames.append(detail)
        summaries.append(summary)
        print(
            f"  -> top={summary.get('top_feature_full')} | abs_corr={summary.get('top_feature_full_abs_pearson'):.6f} | "
            f"mean_abs_test_topk={summary.get('mean_abs_test_corr_topk_mean'):.6f}",
            flush=True,
        )

    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summaries).sort_values("mean_abs_test_corr_topk_mean", ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"corr_sweep_{tag}_detail.csv"
    summary_path = out_dir / f"corr_sweep_{tag}_summary.csv"
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n[sweep summary]", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"[saved] {detail_path}", flush=True)
    print(f"[saved] {summary_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_extract = sub.add_parser("extract")
    ap_extract.add_argument("--examples-path", type=Path, required=True)
    ap_extract.add_argument("--sentences-path", type=Path, required=True)
    ap_extract.add_argument("--localization-path", type=Path, required=True)
    ap_extract.add_argument("--out-path", type=Path, required=True)
    ap_extract.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap_extract.add_argument("--num-examples", type=int, default=600)
    ap_extract.add_argument("--seed", type=int, default=0)
    ap_extract.add_argument("--only-localized", action="store_true", default=True)
    ap_extract.add_argument("--max-tokens", type=int, default=10000)
    ap_extract.add_argument("--topk-vocab", type=int, default=32)
    ap_extract.add_argument("--min-dist-list", type=str, default="1,2,4")
    ap_extract.add_argument("--attention-layer-offsets", type=str, default="-1,-2,-3")
    ap_extract.add_argument("--activation-layer-offsets", type=str, default="-1,-2,-4,-8")
    ap_extract.add_argument("--activation-sparsity-eps", type=float, default=0.01)
    ap_extract.add_argument("--device", type=str, default="cuda")
    ap_extract.add_argument("--trust-remote-code", action="store_true", default=True)
    ap_extract.add_argument("--progress-every", type=int, default=25)
    ap_extract.add_argument("--token-oom-backoff", type=float, default=0.5)
    ap_extract.add_argument("--token-min-tokens", type=int, default=256)

    ap_sweep = sub.add_parser("sweep")
    ap_sweep.add_argument("--data-path", type=Path, required=True)
    ap_sweep.add_argument("--out-dir", type=Path, required=True)
    ap_sweep.add_argument("--tag", type=str, required=True)
    ap_sweep.add_argument("--strategies", type=str, default="all")
    ap_sweep.add_argument("--seeds", type=str, default="42,52,62")
    ap_sweep.add_argument("--topk", type=int, default=20)
    ap_sweep.add_argument("--deconfound-position", action="store_true", default=False)
    ap_sweep.add_argument("--deconfound-knots", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap_sweep.add_argument("--residualize-target", action="store_true", default=False)

    args = ap.parse_args()

    if args.mode == "extract":
        min_dist_list = tuple(int(x.strip()) for x in args.min_dist_list.split(",") if x.strip())
        layer_offsets = tuple(int(x.strip()) for x in args.attention_layer_offsets.split(",") if x.strip())
        act_layer_offsets = tuple(int(x.strip()) for x in args.activation_layer_offsets.split(",") if x.strip())
        cfg = ExtractConfig(
            examples_path=args.examples_path,
            sentences_path=args.sentences_path,
            localization_path=args.localization_path,
            out_path=args.out_path,
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
        run_extract(cfg)
        return

    strategies = ALL_STRATEGIES if args.strategies == "all" else [s.strip() for s in args.strategies.split(",") if s.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    knots = tuple(float(x.strip()) for x in args.deconfound_knots.split(",") if x.strip())
    run_sweep(
        data_path=args.data_path,
        out_dir=args.out_dir,
        tag=args.tag,
        strategies=strategies,
        seeds=seeds,
        topk=args.topk,
        deconfound_position=args.deconfound_position,
        deconfound_knots=knots,
        residualize_target=args.residualize_target,
    )


if __name__ == "__main__":
    main()
