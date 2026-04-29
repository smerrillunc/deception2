#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_CONFIG_ROOT", "/tmp/vllm")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
NOTEBOOK_ROOT = ROOT_DIR / "Notebooks"

for search_root in (SCRIPT_DIR, NOTEBOOK_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from sentence_pipeline import split_sentence_spans
from localization_sampling_budget_ablation import (
    DATASET_ROOT,
    ENV_SPECS,
    bootstrap_mean_ci,
    build_candidate_pool,
    cache_filename_for_example,
    extract_raw_text,
    load_prepare_messages_for_model,
    load_sentence_localization_batch,
    maybe_tqdm,
    parse_envs,
    prepare_example_runtime,
    select_examples_round_robin,
    slugify,
)

DEFAULT_SIMILARITY_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one integer in {text!r}.")
    return values


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one float in {text!r}.")
    return values


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def extract_prefix_text(*, raw_text: str, sentences: list[dict[str, Any]], prefix_sentence_idx: int) -> str:
    prefix_sentence_idx = int(prefix_sentence_idx)
    prefix_sentence_idx = max(0, min(len(sentences) - 1, prefix_sentence_idx))
    end_char = int(sentences[prefix_sentence_idx]["end"])
    return raw_text[:end_char]


def choose_prefix_sentence_idx(*, sentence_count: int, strategy: str) -> int:
    if sentence_count < 2:
        return 0
    if strategy == "first":
        return 0
    if strategy == "midpoint":
        return min(sentence_count - 2, max(0, (sentence_count - 1) // 2))
    if strategy == "last_minus_one":
        return max(0, sentence_count - 2)
    raise ValueError(f"Unknown prefix strategy: {strategy}")


def render_prompt_for_prefix(
    *,
    slb_module: Any,
    tokenizer: Any,
    prompt_text: str,
    prompt_messages: list[dict[str, Any]] | None,
    prefix_text: str,
) -> str:
    return slb_module._render_prefix_prompt(
        tokenizer,
        prompt_text,
        prompt_messages,
        prefix_text,
    )


def extract_reasoning_text_from_generation(
    gen_text: str,
    *,
    model_name: str,
    get_reasoning_model_output_fn: Any,
    extract_json_with_reasoning_fn: Any,
) -> str:
    text = str(gen_text or "")
    if not text.strip():
        return ""
    for parser in (
        lambda value: get_reasoning_model_output_fn(value, model_name=model_name),
        extract_json_with_reasoning_fn,
    ):
        try:
            parsed = parser(text)
            reasoning = parsed.get("reasoning") if isinstance(parsed, dict) else None
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()
        except Exception:
            pass

    markers = ("```json", "{", "\n{")
    cutoff = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            cutoff = min(cutoff, idx)
    return text[:cutoff].strip()


def extract_first_sentence(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    sentences = split_sentence_spans(normalized)
    if sentences:
        first_sentence = str(sentences[0].get("text") or "").strip()
        if first_sentence:
            return first_sentence
    first_line = normalized.splitlines()[0].strip()
    return first_line or normalized


def char_ngram_counter(text: str, *, min_n: int = 3, max_n: int = 5) -> Counter[str]:
    text = normalize_text(text)
    if not text:
        return Counter()
    padded = f" {text} "
    grams: Counter[str] = Counter()
    for n in range(int(min_n), int(max_n) + 1):
        if len(padded) < n:
            continue
        for idx in range(len(padded) - n + 1):
            grams[padded[idx : idx + n]] += 1
    return grams


def mean_pairwise_char_ngram_cosine(texts: list[str]) -> float:
    normalized_texts = [normalize_text(text) for text in texts if normalize_text(text)]
    if len(normalized_texts) < 2:
        return float("nan")

    counters = [char_ngram_counter(text) for text in normalized_texts]
    norms = [math.sqrt(sum(value * value for value in counter.values())) for counter in counters]
    sims: list[float] = []
    for idx_a in range(len(counters)):
        for idx_b in range(idx_a + 1, len(counters)):
            if norms[idx_a] == 0.0 or norms[idx_b] == 0.0:
                continue
            small_idx, large_idx = (idx_a, idx_b) if len(counters[idx_a]) <= len(counters[idx_b]) else (idx_b, idx_a)
            dot = sum(
                float(value) * float(counters[large_idx].get(key, 0.0))
                for key, value in counters[small_idx].items()
            )
            sims.append(dot / (norms[small_idx] * norms[large_idx]))
    if not sims:
        return float("nan")
    return float(np.mean(np.asarray(sims, dtype=np.float32), dtype=np.float32))


def mean_pairwise_embedding_cosine(embeddings: np.ndarray) -> float:
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return float("nan")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    valid_mask = np.squeeze(norms > 0.0, axis=1)
    if int(np.sum(valid_mask)) < 2:
        return float("nan")
    valid_embeddings = embeddings[valid_mask]
    valid_norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
    valid_embeddings = valid_embeddings / np.clip(valid_norms, 1e-12, None)
    sim_matrix = valid_embeddings @ valid_embeddings.T
    tri_upper = np.triu_indices(sim_matrix.shape[0], k=1)
    if tri_upper[0].size == 0:
        return float("nan")
    return float(np.mean(sim_matrix[tri_upper], dtype=np.float32))


class SentenceEmbeddingSimilarity:
    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 64,
        max_length: int = 128,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = str(model_name)
        requested_device = str(device).strip().lower() or "cpu"
        if requested_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"
        else:
            resolved_device = requested_device
        self.device = torch.device(resolved_device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(8, int(max_length))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=bool(local_files_only),
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            local_files_only=bool(local_files_only),
        )
        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)

    def _mean_pool(self, token_embeddings: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-12)
        return pooled / denom

    def encode(self, texts: list[str]) -> np.ndarray:
        normalized_texts = [normalize_text(text) for text in texts]
        nonempty_rows = [
            (row_idx, text)
            for row_idx, text in enumerate(normalized_texts)
            if text
        ]
        if not nonempty_rows:
            vector_dim = self.hidden_size if self.hidden_size > 0 else 1
            return np.zeros((len(texts), vector_dim), dtype=np.float32)

        vectors_by_row: dict[int, np.ndarray] = {}
        vector_dim: int | None = None
        with self._torch.inference_mode():
            for start in range(0, len(nonempty_rows), self.batch_size):
                batch_rows = nonempty_rows[start : start + self.batch_size]
                batch_indices = [int(row_idx) for row_idx, _ in batch_rows]
                batch_texts = [text for _, text in batch_rows]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                }
                outputs = self.model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                batch_embeddings = pooled.detach().cpu().to(self._torch.float32).numpy()
                vector_dim = int(batch_embeddings.shape[1])
                for batch_idx, row_idx in enumerate(batch_indices):
                    vectors_by_row[int(row_idx)] = batch_embeddings[batch_idx]

        final_dim = vector_dim if vector_dim is not None else (self.hidden_size if self.hidden_size > 0 else 1)
        embeddings = np.zeros((len(texts), final_dim), dtype=np.float32)
        for row_idx, vector in vectors_by_row.items():
            embeddings[int(row_idx)] = vector
        return embeddings

    def mean_pairwise_similarity(self, texts: list[str]) -> float:
        if len(texts) < 2:
            return float("nan")
        embeddings = self.encode(texts)
        return mean_pairwise_embedding_cosine(embeddings)


def unique_fraction(texts: list[str]) -> float:
    normalized = [normalize_text(text) for text in texts if normalize_text(text)]
    if not normalized:
        return float("nan")
    return float(len(set(normalized)) / len(normalized))


def mean_token_count(texts: list[str], *, tokenizer: Any) -> float:
    counts: list[int] = []
    for text in texts:
        if not str(text or "").strip():
            continue
        try:
            token_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
            counts.append(int(len(token_ids)))
        except Exception:
            counts.append(int(len(str(text).split())))
    if not counts:
        return float("nan")
    return float(np.mean(np.asarray(counts, dtype=np.float32), dtype=np.float32))


def sample_raw_continuations(
    *,
    llm: Any,
    sampling_params_cls: Any,
    rendered_prompt: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    seed: int,
) -> list[str]:
    sampling_params = sampling_params_cls(
        n=int(n_samples),
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        max_tokens=int(max_new_tokens),
        seed=int(seed),
    )
    outputs = llm.generate(prompts=[rendered_prompt], sampling_params=sampling_params)
    generations: list[str] = []
    for output in outputs:
        for sample_output in output.outputs:
            generations.append(str(sample_output.text))
    return generations


def combo_key(temperature: float, top_p: float, repetition_penalty: float) -> str:
    return f"temp_{temperature:.1f}__top_p_{top_p:.1f}__rep_{repetition_penalty:.1f}"


def build_combo_grid(
    *,
    temperatures: tuple[float, ...],
    top_ps: tuple[float, ...],
    repetition_penalties: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for temperature in temperatures:
        for top_p in top_ps:
            for repetition_penalty in repetition_penalties:
                rows.append(
                    {
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "repetition_penalty": float(repetition_penalty),
                        "combo_key": combo_key(float(temperature), float(top_p), float(repetition_penalty)),
                    }
                )
    return rows


def summarize_combo_generations(
    *,
    generations: list[str],
    tokenizer: Any,
    similarity_encoder: SentenceEmbeddingSimilarity,
    model_name: str,
    get_reasoning_model_output_fn: Any,
    extract_json_with_reasoning_fn: Any,
    keep_generation_details: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    reasoning_texts: list[str] = []
    first_sentences: list[str] = []
    generation_rows: list[dict[str, Any]] = []

    for gen_idx, gen_text in enumerate(generations):
        reasoning_text = extract_reasoning_text_from_generation(
            gen_text,
            model_name=model_name,
            get_reasoning_model_output_fn=get_reasoning_model_output_fn,
            extract_json_with_reasoning_fn=extract_json_with_reasoning_fn,
        )
        first_sentence = extract_first_sentence(reasoning_text)
        reasoning_texts.append(reasoning_text)
        first_sentences.append(first_sentence)
        if keep_generation_details:
            generation_rows.append(
                {
                    "generation_idx": int(gen_idx),
                    "reasoning_text": reasoning_text,
                    "first_sentence": first_sentence,
                    "full_generation_text": gen_text,
                }
            )

    summary = {
        "avg_reasoning_tokens": mean_token_count(reasoning_texts, tokenizer=tokenizer),
        "avg_first_sentence_tokens": mean_token_count(first_sentences, tokenizer=tokenizer),
        "avg_first_sentence_embedding_similarity": similarity_encoder.mean_pairwise_similarity(first_sentences),
        "avg_first_sentence_lexical_similarity": mean_pairwise_char_ngram_cosine(first_sentences),
        "unique_first_sentence_fraction": unique_fraction(first_sentences),
        "nonempty_first_sentence_count": int(sum(bool(normalize_text(text)) for text in first_sentences)),
    }
    summary["avg_first_sentence_similarity"] = float(summary["avg_first_sentence_embedding_similarity"])
    return summary, generation_rows, first_sentences


def recover_first_sentences(combo_result: dict[str, Any]) -> list[str] | None:
    first_sentences = combo_result.get("first_sentences")
    if isinstance(first_sentences, list):
        return [str(value or "") for value in first_sentences]
    generation_rows = combo_result.get("generation_rows")
    if isinstance(generation_rows, list):
        recovered = [str((row or {}).get("first_sentence") or "") for row in generation_rows]
        combo_result["first_sentences"] = recovered
        return recovered
    return None


def ensure_similarity_metrics(
    combo_result: dict[str, Any],
    *,
    similarity_encoder: SentenceEmbeddingSimilarity,
) -> bool:
    updated = False
    first_sentences = recover_first_sentences(combo_result)

    if "avg_first_sentence_lexical_similarity" not in combo_result:
        if first_sentences is not None:
            combo_result["avg_first_sentence_lexical_similarity"] = mean_pairwise_char_ngram_cosine(first_sentences)
            updated = True
        elif "avg_first_sentence_similarity" in combo_result:
            combo_result["avg_first_sentence_lexical_similarity"] = float(combo_result["avg_first_sentence_similarity"])
            updated = True

    if "avg_first_sentence_embedding_similarity" not in combo_result:
        if first_sentences is None:
            return False
        combo_result["avg_first_sentence_embedding_similarity"] = similarity_encoder.mean_pairwise_similarity(first_sentences)
        updated = True

    if (
        "avg_first_sentence_similarity" not in combo_result
        or not math.isfinite(float(combo_result["avg_first_sentence_similarity"]))
        or abs(
            float(combo_result["avg_first_sentence_similarity"])
            - float(combo_result["avg_first_sentence_embedding_similarity"])
        ) > 1e-9
    ):
        combo_result["avg_first_sentence_similarity"] = float(combo_result["avg_first_sentence_embedding_similarity"])
        updated = True

    combo_result["_updated_similarity_metrics"] = bool(updated)
    return True


def enrich_prefix_payload_similarity(
    prefix_payload: dict[str, Any],
    *,
    similarity_encoder: SentenceEmbeddingSimilarity,
) -> tuple[bool, bool]:
    updated = False
    for combo_result in prefix_payload.get("combo_results", []):
        ok = ensure_similarity_metrics(
            combo_result,
            similarity_encoder=similarity_encoder,
        )
        if not ok:
            return False, updated
        updated = updated or bool(combo_result.pop("_updated_similarity_metrics", False))
    return True, updated


def build_heatmap_matrix(
    combo_summary_df: pd.DataFrame,
    *,
    metric_col: str,
    repetition_penalty: float,
    temperatures: tuple[float, ...],
    top_ps: tuple[float, ...],
) -> pd.DataFrame:
    subset = combo_summary_df.loc[
        combo_summary_df["repetition_penalty"].eq(float(repetition_penalty))
    ].copy()
    matrix = (
        subset.pivot(index="temperature", columns="top_p", values=metric_col)
        .reindex(index=list(temperatures), columns=list(top_ps))
    )
    return matrix


def plot_heatmap(
    matrix_df: pd.DataFrame,
    *,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    matrix = matrix_df.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(matrix_df.shape[1]))
    ax.set_xticklabels([f"{float(value):.1f}" for value in matrix_df.columns])
    ax.set_yticks(np.arange(matrix_df.shape[0]))
    ax.set_yticklabels([f"{float(value):.1f}" for value in matrix_df.index])
    ax.set_xlabel("top_p")
    ax.set_ylabel("temperature")
    ax.set_title(title)
    for row_idx in range(matrix_df.shape[0]):
        for col_idx in range(matrix_df.shape[1]):
            value = matrix_df.iat[row_idx, col_idx]
            text = "nan" if not np.isfinite(value) else f"{value:.3f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color="white" if np.isfinite(value) and value < np.nanmean(matrix) else "black", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03, label=cbar_label)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_vs_tokens(combo_summary_df: pd.DataFrame, *, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    scatter = ax.scatter(
        combo_summary_df["avg_reasoning_tokens_mean"],
        combo_summary_df["avg_first_sentence_embedding_similarity_mean"],
        c=combo_summary_df["temperature"],
        s=140,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.6,
    )
    for row in combo_summary_df.itertuples(index=False):
        ax.annotate(
            f"T{row.temperature:.1f}\nP{row.top_p:.1f}\nR{row.repetition_penalty:.1f}",
            (
                float(row.avg_reasoning_tokens_mean),
                float(row.avg_first_sentence_embedding_similarity_mean),
            ),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8.2,
        )
    ax.set_xlabel("Average reasoning tokens")
    ax.set_ylabel("Average first-sentence embedding cosine")
    ax.set_title("Verbosity-diversity tradeoff across decoding settings")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.03, label="temperature")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_paper_summary_markdown(
    *,
    combo_summary_df: pd.DataFrame,
    selected_prefixes_df: pd.DataFrame,
    prefix_strategy: str,
    similarity_model_name: str,
) -> str:
    lines: list[str] = []
    lines.append("# Decoding Diversity Ablation Summary")
    lines.append("")
    lines.append(f"- Prefixes analyzed: {int(selected_prefixes_df.shape[0])}")
    lines.append(f"- Prefix strategy: {prefix_strategy}")
    lines.append(f"- Primary similarity metric: mean pairwise cosine of first-sentence embeddings from `{similarity_model_name}`")
    lines.append("")
    ranked_diverse = combo_summary_df.sort_values(
        ["avg_first_sentence_embedding_similarity_mean", "avg_reasoning_tokens_mean"],
        ascending=[True, True],
    ).reset_index(drop=True)
    ranked_short = combo_summary_df.sort_values(
        ["avg_reasoning_tokens_mean", "avg_first_sentence_embedding_similarity_mean"],
        ascending=[True, True],
    ).reset_index(drop=True)
    if not ranked_diverse.empty:
        row = ranked_diverse.iloc[0]
        lines.append("## Most diverse setting")
        lines.append(
            f"- temp={row['temperature']:.1f}, top_p={row['top_p']:.1f}, rep_pen={row['repetition_penalty']:.1f}"
        )
        lines.append(
            f"- avg first-sentence embedding cosine={row['avg_first_sentence_embedding_similarity_mean']:.4f}"
        )
        lines.append(
            f"- avg reasoning tokens={row['avg_reasoning_tokens_mean']:.2f}"
        )
        lines.append(
            f"- avg first-sentence lexical cosine={row['avg_first_sentence_lexical_similarity_mean']:.4f}"
        )
    if not ranked_short.empty:
        row = ranked_short.iloc[0]
        lines.append("")
        lines.append("## Shortest continuations")
        lines.append(
            f"- temp={row['temperature']:.1f}, top_p={row['top_p']:.1f}, rep_pen={row['repetition_penalty']:.1f}"
        )
        lines.append(
            f"- avg reasoning tokens={row['avg_reasoning_tokens_mean']:.2f}"
        )
        lines.append(
            f"- avg first-sentence embedding cosine={row['avg_first_sentence_embedding_similarity_mean']:.4f}"
        )
        lines.append(
            f"- avg first-sentence lexical cosine={row['avg_first_sentence_lexical_similarity_mean']:.4f}"
        )
    return "\n".join(lines) + "\n"


def maybe_load_existing_selected_prefixes(output_root: Path) -> pd.DataFrame | None:
    selected_prefixes_path = output_root / "selected_prefixes.csv"
    if not selected_prefixes_path.exists():
        return None
    selected_prefixes_df = pd.read_csv(selected_prefixes_path).copy()
    if selected_prefixes_df.empty:
        return None
    return selected_prefixes_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Decoding diversity ablation for localization continuations.")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--model-dirname", type=str, default="DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--envs", nargs="*", default=list(ENV_SPECS.keys()))
    parser.add_argument("--selected-examples-csv", type=str, default="")
    parser.add_argument("--text-field", type=str, default="action_reasoning")
    parser.add_argument("--num-prefixes", type=int, default=100)
    parser.add_argument("--max-sentences", type=int, default=20)
    parser.add_argument("--prefix-strategy", type=str, choices=["first", "midpoint", "last_minus_one"], default="midpoint")
    parser.add_argument("--temperatures", type=str, default="0.5,0.7,0.9")
    parser.add_argument("--top-ps", type=str, default="0.5,0.7,0.9")
    parser.add_argument("--repetition-penalties", type=str, default="1.1,1.2")
    parser.add_argument("--n-generations", type=int, default=100)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--selection-seed", type=int, default=1234)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument("--similarity-model-name", type=str, default=DEFAULT_SIMILARITY_MODEL_NAME)
    parser.add_argument("--similarity-device", type=str, choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--similarity-batch-size", type=int, default=128)
    parser.add_argument("--similarity-max-length", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(NOTEBOOK_ROOT / "datasetmain_localization_decoding_diversity_outputs"),
    )
    parser.add_argument("--selection-only", action="store_true", default=False)
    parser.add_argument("--analysis-only", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--save-generation-details", action="store_true", default=False)
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    args = parser.parse_args(argv)

    env_names = parse_envs(args.envs)
    temperatures = parse_float_list(args.temperatures)
    top_ps = parse_float_list(args.top_ps)
    repetition_penalties = parse_float_list(args.repetition_penalties)
    combo_grid = build_combo_grid(
        temperatures=temperatures,
        top_ps=top_ps,
        repetition_penalties=repetition_penalties,
    )

    if int(args.n_generations) < 2:
        raise ValueError("--n-generations must be >= 2.")
    if int(args.num_prefixes) < 1:
        raise ValueError("--num-prefixes must be >= 1.")

    dataset_root = Path(args.dataset_root)
    run_tag = args.run_tag.strip() or (
        f"{slugify(args.model_dirname)}__prefixes_{int(args.num_prefixes)}__{slugify(args.prefix_strategy)}"
    )
    output_root = Path(args.output_root) / run_tag
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = output_root / "per_prefix_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    config_df = pd.DataFrame(
        [
            {"setting": "dataset_root", "value": str(dataset_root)},
            {"setting": "model_dirname", "value": args.model_dirname},
            {"setting": "envs", "value": ", ".join(env_names)},
            {"setting": "num_prefixes", "value": int(args.num_prefixes)},
            {"setting": "max_sentences", "value": int(args.max_sentences)},
            {"setting": "prefix_strategy", "value": args.prefix_strategy},
            {"setting": "n_generations", "value": int(args.n_generations)},
            {"setting": "temperatures", "value": ", ".join(f"{value:.1f}" for value in temperatures)},
            {"setting": "top_ps", "value": ", ".join(f"{value:.1f}" for value in top_ps)},
            {"setting": "repetition_penalties", "value": ", ".join(f"{value:.1f}" for value in repetition_penalties)},
            {"setting": "similarity_model_name", "value": args.similarity_model_name},
            {"setting": "similarity_device", "value": args.similarity_device},
            {"setting": "similarity_batch_size", "value": int(args.similarity_batch_size)},
            {"setting": "similarity_max_length", "value": int(args.similarity_max_length)},
            {"setting": "selection_only", "value": bool(args.selection_only)},
            {"setting": "analysis_only", "value": bool(args.analysis_only)},
            {"setting": "output_root", "value": str(output_root)},
        ]
    )
    config_df.to_csv(output_root / "config.csv", index=False)

    selected_prefixes_df: pd.DataFrame | None = None
    candidate_lookup: dict[str, Any] = {}
    if args.analysis_only and not args.selected_examples_csv.strip():
        selected_prefixes_df = maybe_load_existing_selected_prefixes(output_root)

    if selected_prefixes_df is None:
        candidate_df, candidate_lookup, candidate_count_df = build_candidate_pool(
            dataset_root,
            env_names=env_names,
            model_dirname=args.model_dirname,
            text_field=args.text_field,
            deceptive_only=True,
            max_sentences=int(args.max_sentences),
            disable_tqdm=bool(args.disable_tqdm),
        )
        if candidate_df.empty:
            raise ValueError("No short deceptive candidates found.")
        candidate_df = candidate_df.loc[candidate_df["sentence_count"].ge(2)].copy()
        if candidate_df.empty:
            raise ValueError("No candidates with at least 2 reasoning sentences found.")
        candidate_df.to_csv(output_root / "candidate_examples.csv", index=False)
        candidate_count_df.to_csv(output_root / "candidate_counts_by_env.csv", index=False)

        if args.selected_examples_csv.strip():
            selected_examples_df = pd.read_csv(args.selected_examples_csv).copy()
            selected_examples_df = selected_examples_df.loc[
                selected_examples_df["example_key"].isin(candidate_df["example_key"])
            ].copy()
            if selected_examples_df.empty:
                raise ValueError(
                    f"No rows from {args.selected_examples_csv} matched the current candidate pool."
                )
            selected_examples_df = select_examples_round_robin(
                selected_examples_df,
                env_names=env_names,
                num_examples=int(args.num_prefixes),
                selection_seed=int(args.selection_seed),
                case_study_example_id=None,
            )
        else:
            selected_examples_df = select_examples_round_robin(
                candidate_df,
                env_names=env_names,
                num_examples=int(args.num_prefixes),
                selection_seed=int(args.selection_seed),
                case_study_example_id=None,
            )

        selected_prefix_rows: list[dict[str, Any]] = []
        selection_iterator = maybe_tqdm(
            list(selected_examples_df.to_dict(orient="records")),
            desc="Selected prefixes",
            total=len(selected_examples_df),
            disable=bool(args.disable_tqdm),
        )
        for row in selection_iterator:
            example = candidate_lookup[str(row["example_key"])]
            raw_text_full = extract_raw_text(example, args.text_field)
            if not raw_text_full:
                continue
            sentences = split_sentence_spans(raw_text_full)
            if len(sentences) < 2:
                continue
            prefix_sentence_idx = choose_prefix_sentence_idx(
                sentence_count=len(sentences),
                strategy=args.prefix_strategy,
            )
            prefix_text = extract_prefix_text(
                raw_text=raw_text_full,
                sentences=sentences,
                prefix_sentence_idx=prefix_sentence_idx,
            )
            selected_prefix_rows.append(
                {
                    **row,
                    "prefix_sentence_idx": int(prefix_sentence_idx),
                    "prefix_sentence_number": int(prefix_sentence_idx) + 1,
                    "prefix_sentence_text": str(sentences[prefix_sentence_idx]["text"]),
                    "remaining_sentence_count": int(len(sentences) - (prefix_sentence_idx + 1)),
                    "prefix_char_count": int(len(prefix_text)),
                }
            )

        selected_prefixes_df = pd.DataFrame(selected_prefix_rows)
        if selected_prefixes_df.empty:
            raise ValueError("No valid prefixes were selected.")
        selected_prefixes_df = selected_prefixes_df.sort_values(["env_name", "sentence_count", "example_id"]).reset_index(drop=True)
        selected_prefixes_df.to_csv(output_root / "selected_prefixes.csv", index=False)
        selection_count_df = (
            selected_prefixes_df.groupby("env_name", as_index=False)
            .agg(selected_prefixes=("example_id", "count"))
            .sort_values("env_name")
            .reset_index(drop=True)
        )
        selection_count_df.to_csv(output_root / "selection_counts_by_env.csv", index=False)
    else:
        selected_prefixes_df = selected_prefixes_df.sort_values(["env_name", "sentence_count", "example_id"]).reset_index(drop=True)

    selected_prefixes_df.to_csv(output_root / "selected_prefixes.csv", index=False)
    selection_count_df = (
        selected_prefixes_df.groupby("env_name", as_index=False)
        .agg(selected_prefixes=("example_id", "count"))
        .sort_values("env_name")
        .reset_index(drop=True)
    )
    selection_count_df.to_csv(output_root / "selection_counts_by_env.csv", index=False)

    if args.selection_only:
        print(f"Selection written to {output_root / 'selected_prefixes.csv'}")
        print(f"Selection counts written to {output_root / 'selection_counts_by_env.csv'}")
        return

    similarity_encoder = SentenceEmbeddingSimilarity(
        model_name=args.similarity_model_name,
        device=args.similarity_device,
        batch_size=int(args.similarity_batch_size),
        max_length=int(args.similarity_max_length),
        local_files_only=False,
    )

    combo_df = pd.DataFrame(combo_grid)
    combo_df.to_csv(output_root / "combo_grid.csv", index=False)

    per_prefix_rows: list[dict[str, Any]] = []
    missing_cache_keys: list[str] = []
    stale_cache_keys: list[str] = []

    if not args.analysis_only:
        import torch
        from vllm import LLM, SamplingParams

        slb_module = load_sentence_localization_batch()
        prepare_messages_for_model_fn = load_prepare_messages_for_model()
        from utils import extract_json_with_reasoning, get_reasoning_model_output

        visible_gpu_count = max(1, int(torch.cuda.device_count()))
        if int(args.tensor_parallel_size) > visible_gpu_count:
            raise ValueError(
                f"--tensor-parallel-size={args.tensor_parallel_size} exceeds visible GPU count={visible_gpu_count}."
            )

        resolved_model_name = args.model_name.strip()
        if not resolved_model_name:
            inferred = (
                selected_prefixes_df["meta_model_name"]
                .dropna()
                .astype(str)
                .loc[lambda s: s.str.len() > 0]
            )
            if inferred.empty:
                raise ValueError("Could not infer model name; pass --model-name explicitly.")
            resolved_model_name = str(inferred.iloc[0])

        llm = LLM(
            model=resolved_model_name,
            max_model_len=int(args.max_new_tokens),
            seed=1,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            tensor_parallel_size=int(args.tensor_parallel_size),
        )
        tokenizer = llm.get_tokenizer()

        prefix_iterator = maybe_tqdm(
            list(selected_prefixes_df.to_dict(orient="records")),
            desc="Prefixes",
            total=len(selected_prefixes_df),
            disable=bool(args.disable_tqdm),
        )
        for prefix_idx, prefix_row in enumerate(prefix_iterator):
            example = candidate_lookup[str(prefix_row["example_key"])]
            runtime = prepare_example_runtime(
                example,
                slb_module=slb_module,
                prepare_messages_for_model_fn=prepare_messages_for_model_fn,
                model_name=resolved_model_name,
                text_field=args.text_field,
                tokenizer=tokenizer,
            )
            prefix_text = extract_prefix_text(
                raw_text=str(runtime["raw_text"]),
                sentences=list(runtime["sentences"]),
                prefix_sentence_idx=int(prefix_row["prefix_sentence_idx"]),
            )
            rendered_prompt = render_prompt_for_prefix(
                slb_module=slb_module,
                tokenizer=tokenizer,
                prompt_text=str(runtime["prompt_text"]),
                prompt_messages=runtime["prepared_messages"],
                prefix_text=prefix_text,
            )

            cache_path = cache_dir / cache_filename_for_example(str(prefix_row["env_name"]), str(prefix_row["example_id"]))
            prefix_payload: dict[str, Any] | None = None
            if cache_path.exists() and not args.overwrite:
                prefix_payload = json.loads(cache_path.read_text(encoding="utf-8"))
                cache_ok, cache_updated = enrich_prefix_payload_similarity(
                    prefix_payload,
                    similarity_encoder=similarity_encoder,
                )
                if cache_ok:
                    if cache_updated:
                        cache_path.write_text(json.dumps(prefix_payload, indent=2), encoding="utf-8")
                else:
                    prefix_payload = None
            if prefix_payload is None:
                prefix_payload = {
                    "example_key": str(prefix_row["example_key"]),
                    "example_id": str(prefix_row["example_id"]),
                    "env_name": str(prefix_row["env_name"]),
                    "prefix_sentence_idx": int(prefix_row["prefix_sentence_idx"]),
                    "prefix_sentence_number": int(prefix_row["prefix_sentence_number"]),
                    "prefix_sentence_text": str(prefix_row["prefix_sentence_text"]),
                    "combo_results": [],
                }
                combo_iterator = maybe_tqdm(
                    combo_grid,
                    desc=f"Combos:{prefix_row['env_name']}:{prefix_row['example_id']}",
                    total=len(combo_grid),
                    disable=bool(args.disable_tqdm),
                    leave=False,
                )
                for combo_order_idx, combo in enumerate(combo_iterator):
                    generations = sample_raw_continuations(
                        llm=llm,
                        sampling_params_cls=SamplingParams,
                        rendered_prompt=rendered_prompt,
                        n_samples=int(args.n_generations),
                        temperature=float(combo["temperature"]),
                        top_p=float(combo["top_p"]),
                        repetition_penalty=float(combo["repetition_penalty"]),
                        max_new_tokens=int(args.max_new_tokens),
                        seed=int(args.base_seed) + (prefix_idx * 10_000) + (combo_order_idx * 100),
                    )
                    summary, generation_rows, first_sentences = summarize_combo_generations(
                        generations=generations,
                        tokenizer=tokenizer,
                        similarity_encoder=similarity_encoder,
                        model_name=resolved_model_name,
                        get_reasoning_model_output_fn=get_reasoning_model_output,
                        extract_json_with_reasoning_fn=extract_json_with_reasoning,
                        keep_generation_details=bool(args.save_generation_details),
                    )
                    combo_payload = {
                        **combo,
                        "n_generations": int(args.n_generations),
                        **summary,
                        "first_sentences": first_sentences,
                    }
                    if args.save_generation_details:
                        combo_payload["generation_rows"] = generation_rows
                    prefix_payload["combo_results"].append(combo_payload)
                cache_path.write_text(json.dumps(prefix_payload, indent=2), encoding="utf-8")

            for combo_result in prefix_payload.get("combo_results", []):
                per_prefix_rows.append(
                    {
                        "example_key": str(prefix_payload["example_key"]),
                        "example_id": str(prefix_payload["example_id"]),
                        "env_name": str(prefix_payload["env_name"]),
                        "prefix_sentence_idx": int(prefix_payload["prefix_sentence_idx"]),
                        "prefix_sentence_number": int(prefix_payload["prefix_sentence_number"]),
                        "prefix_sentence_text": str(prefix_payload["prefix_sentence_text"]),
                        "temperature": float(combo_result["temperature"]),
                        "top_p": float(combo_result["top_p"]),
                        "repetition_penalty": float(combo_result["repetition_penalty"]),
                        "combo_key": str(combo_result["combo_key"]),
                        "n_generations": int(combo_result["n_generations"]),
                        "avg_reasoning_tokens": float(combo_result["avg_reasoning_tokens"]),
                        "avg_first_sentence_tokens": float(combo_result["avg_first_sentence_tokens"]),
                        "avg_first_sentence_similarity": float(combo_result["avg_first_sentence_embedding_similarity"]),
                        "avg_first_sentence_embedding_similarity": float(combo_result["avg_first_sentence_embedding_similarity"]),
                        "avg_first_sentence_lexical_similarity": float(combo_result["avg_first_sentence_lexical_similarity"]),
                        "unique_first_sentence_fraction": float(combo_result["unique_first_sentence_fraction"]),
                        "nonempty_first_sentence_count": int(combo_result["nonempty_first_sentence_count"]),
                    }
                )
    else:
        resolved_model_name = args.model_name.strip()
        analysis_iterator = maybe_tqdm(
            list(selected_prefixes_df.to_dict(orient="records")),
            desc="Cached prefixes",
            total=len(selected_prefixes_df),
            disable=bool(args.disable_tqdm),
        )
        for prefix_row in analysis_iterator:
            cache_path = cache_dir / cache_filename_for_example(str(prefix_row["env_name"]), str(prefix_row["example_id"]))
            if not cache_path.exists():
                missing_cache_keys.append(str(prefix_row["example_key"]))
                continue
            prefix_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_ok, cache_updated = enrich_prefix_payload_similarity(
                prefix_payload,
                similarity_encoder=similarity_encoder,
            )
            if not cache_ok:
                stale_cache_keys.append(str(prefix_row["example_key"]))
                continue
            if cache_updated:
                cache_path.write_text(json.dumps(prefix_payload, indent=2), encoding="utf-8")
            for combo_result in prefix_payload.get("combo_results", []):
                per_prefix_rows.append(
                    {
                        "example_key": str(prefix_payload["example_key"]),
                        "example_id": str(prefix_payload["example_id"]),
                        "env_name": str(prefix_payload["env_name"]),
                        "prefix_sentence_idx": int(prefix_payload["prefix_sentence_idx"]),
                        "prefix_sentence_number": int(prefix_payload["prefix_sentence_number"]),
                        "prefix_sentence_text": str(prefix_payload["prefix_sentence_text"]),
                        "temperature": float(combo_result["temperature"]),
                        "top_p": float(combo_result["top_p"]),
                        "repetition_penalty": float(combo_result["repetition_penalty"]),
                        "combo_key": str(combo_result["combo_key"]),
                        "n_generations": int(combo_result["n_generations"]),
                        "avg_reasoning_tokens": float(combo_result["avg_reasoning_tokens"]),
                        "avg_first_sentence_tokens": float(combo_result["avg_first_sentence_tokens"]),
                        "avg_first_sentence_similarity": float(combo_result["avg_first_sentence_embedding_similarity"]),
                        "avg_first_sentence_embedding_similarity": float(combo_result["avg_first_sentence_embedding_similarity"]),
                        "avg_first_sentence_lexical_similarity": float(combo_result["avg_first_sentence_lexical_similarity"]),
                        "unique_first_sentence_fraction": float(combo_result["unique_first_sentence_fraction"]),
                        "nonempty_first_sentence_count": int(combo_result["nonempty_first_sentence_count"]),
                    }
                )
        if missing_cache_keys:
            raise FileNotFoundError(
                "analysis-only mode requires caches for all selected prefixes. Missing caches for:\n"
                + "\n".join(missing_cache_keys[:20])
            )
        if stale_cache_keys:
            raise ValueError(
                "analysis-only mode found outdated caches without stored first sentences for:\n"
                + "\n".join(stale_cache_keys[:20])
                + "\nRerun without --analysis-only (or with --overwrite) to regenerate those prefixes."
            )

    per_prefix_df = pd.DataFrame(per_prefix_rows)
    if per_prefix_df.empty:
        raise ValueError("No per-prefix results were collected.")
    per_prefix_df.to_csv(output_root / "per_prefix_combo_metrics.csv", index=False)

    combo_summary_rows: list[dict[str, Any]] = []
    combo_groups = list(
        per_prefix_df.groupby(
            ["temperature", "top_p", "repetition_penalty", "combo_key"], sort=False, dropna=False
        )
    )
    combo_summary_iterator = maybe_tqdm(
        combo_groups,
        desc="Summarizing combos",
        total=len(combo_groups),
        disable=bool(args.disable_tqdm),
    )
    for combo, combo_df_subset in combo_summary_iterator:
        temperature, top_p, repetition_penalty, combo_key_value = combo
        token_mean, token_lo, token_hi = bootstrap_mean_ci(
            combo_df_subset["avg_reasoning_tokens"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(1000 + int(round(float(temperature) * 100)) + int(round(float(top_p) * 10))),
            repeats=int(args.bootstrap_repeats),
        )
        emb_sim_mean, emb_sim_lo, emb_sim_hi = bootstrap_mean_ci(
            combo_df_subset["avg_first_sentence_embedding_similarity"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(2000 + int(round(float(temperature) * 100)) + int(round(float(top_p) * 10))),
            repeats=int(args.bootstrap_repeats),
        )
        lex_sim_mean, lex_sim_lo, lex_sim_hi = bootstrap_mean_ci(
            combo_df_subset["avg_first_sentence_lexical_similarity"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(2500 + int(round(float(temperature) * 100)) + int(round(float(top_p) * 10))),
            repeats=int(args.bootstrap_repeats),
        )
        unique_mean, unique_lo, unique_hi = bootstrap_mean_ci(
            combo_df_subset["unique_first_sentence_fraction"].to_numpy(dtype=np.float32, copy=False),
            rng=np.random.default_rng(3000 + int(round(float(temperature) * 100)) + int(round(float(top_p) * 10))),
            repeats=int(args.bootstrap_repeats),
        )
        combo_summary_rows.append(
            {
                "combo_key": str(combo_key_value),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "num_prefixes": int(combo_df_subset["example_key"].nunique()),
                "avg_reasoning_tokens_mean": token_mean,
                "avg_reasoning_tokens_ci_low": token_lo,
                "avg_reasoning_tokens_ci_high": token_hi,
                "avg_first_sentence_similarity_mean": emb_sim_mean,
                "avg_first_sentence_similarity_ci_low": emb_sim_lo,
                "avg_first_sentence_similarity_ci_high": emb_sim_hi,
                "avg_first_sentence_embedding_similarity_mean": emb_sim_mean,
                "avg_first_sentence_embedding_similarity_ci_low": emb_sim_lo,
                "avg_first_sentence_embedding_similarity_ci_high": emb_sim_hi,
                "avg_first_sentence_lexical_similarity_mean": lex_sim_mean,
                "avg_first_sentence_lexical_similarity_ci_low": lex_sim_lo,
                "avg_first_sentence_lexical_similarity_ci_high": lex_sim_hi,
                "unique_first_sentence_fraction_mean": unique_mean,
                "unique_first_sentence_fraction_ci_low": unique_lo,
                "unique_first_sentence_fraction_ci_high": unique_hi,
                "avg_first_sentence_tokens_mean": float(np.mean(combo_df_subset["avg_first_sentence_tokens"], dtype=np.float32)),
                "avg_nonempty_first_sentence_count": float(np.mean(combo_df_subset["nonempty_first_sentence_count"], dtype=np.float32)),
            }
        )
    combo_summary_df = pd.DataFrame(combo_summary_rows).sort_values(
        ["repetition_penalty", "temperature", "top_p"]
    ).reset_index(drop=True)
    combo_summary_df.to_csv(output_root / "combo_summary.csv", index=False)

    ranked_table_df = combo_summary_df.sort_values(
        ["avg_first_sentence_embedding_similarity_mean", "avg_reasoning_tokens_mean"],
        ascending=[True, True],
    ).reset_index(drop=True)
    ranked_table_df["rank_by_diversity"] = np.arange(1, len(ranked_table_df) + 1, dtype=int)
    ranked_table_df.to_csv(output_root / "combo_summary_ranked.csv", index=False)

    repetition_penalty_iterator = maybe_tqdm(
        list(repetition_penalties),
        desc="Plotting heatmaps",
        total=len(repetition_penalties),
        disable=bool(args.disable_tqdm),
        leave=False,
    )
    for repetition_penalty in repetition_penalty_iterator:
        token_matrix = build_heatmap_matrix(
            combo_summary_df,
            metric_col="avg_reasoning_tokens_mean",
            repetition_penalty=float(repetition_penalty),
            temperatures=temperatures,
            top_ps=top_ps,
        )
        plot_heatmap(
            token_matrix,
            title=f"Average reasoning tokens | repetition penalty={float(repetition_penalty):.1f}",
            cbar_label="Avg reasoning tokens",
            out_path=figures_dir / f"heatmap_reasoning_tokens__rep_{float(repetition_penalty):.1f}.png",
            cmap="YlOrRd",
        )
        sim_matrix = build_heatmap_matrix(
            combo_summary_df,
            metric_col="avg_first_sentence_embedding_similarity_mean",
            repetition_penalty=float(repetition_penalty),
            temperatures=temperatures,
            top_ps=top_ps,
        )
        plot_heatmap(
            sim_matrix,
            title=f"Average first-sentence embedding cosine | repetition penalty={float(repetition_penalty):.1f}",
            cbar_label="Avg embedding cosine",
            out_path=figures_dir / f"heatmap_first_sentence_embedding_similarity__rep_{float(repetition_penalty):.1f}.png",
            cmap="viridis",
        )

    plot_similarity_vs_tokens(
        combo_summary_df,
        out_path=figures_dir / "embedding_similarity_vs_reasoning_tokens.png",
    )

    summary_md = build_paper_summary_markdown(
        combo_summary_df=combo_summary_df,
        selected_prefixes_df=selected_prefixes_df,
        prefix_strategy=args.prefix_strategy,
        similarity_model_name=args.similarity_model_name,
    )
    (output_root / "paper_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Finished decoding diversity ablation for model_dirname={args.model_dirname}")
    print(f"Selected prefixes: {len(selected_prefixes_df)}")
    print(f"Output root: {output_root}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
