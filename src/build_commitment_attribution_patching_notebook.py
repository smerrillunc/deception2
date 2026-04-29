from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT_DIR / "Notebooks" / "Attribution_Patching_Demo.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # BS Commitment Attribution Patching

        This notebook adapts the original TransformerLens attribution patching demo to BS commitment
        junctures from:

        - `/playpen-ssd/smerrill/deception2/DatasetMain/bs/DeepSeek-R1-Distill-Qwen-7B/localization`

        The clean/corrupted setup is now built from matched branches sharing the same prefix `p`:

        - deceptive branch: `x_D = p + s_D`
        - truthful branch: `x_H = p + s_H`

        with commitment junctures filtered by `Delta_k = P_D(y_1:k) - P_D(y_1:k-1) > 0.3`.

        The IOI one-token logit-difference objective is replaced with a teacher-forced sentence metric:

        - `score_D = mean token logprob(s_D | p)`
        - `score_H = mean token logprob(s_H | p)`
        - `metric = score_D - score_H`

        Important: this notebook does **not** generate continuations during attribution. It only
        scores the existing commitment sentences. Counterfactual deception-rate generation is left
        for later behavioral validation.

        By default this notebook skips full attention-pattern caching. The BS prefixes are long
        enough that quadratic pattern tensors become the main memory bottleneck, so the default
        workflow focuses on residual-stream and head-vector attribution in the same TransformerLens
        style as the original demo.
        """
    ),
    code(
        """
        from __future__ import annotations

        import copy
        import gc
        import json
        import os
        import random
        import sys
        from functools import partial
        from pathlib import Path
        from typing import Any, Callable

        import einops
        import numpy as np
        import pandas as pd
        import plotly.io as pio
        import torch
        import torch.nn.functional as F
        from fancy_einsum import einsum
        from IPython.display import Markdown, display
        from transformers import AutoTokenizer

        try:
            import circuitsvis as cv
        except ImportError:
            cv = None

        import transformer_lens
        import transformer_lens.utilities as utils
        from transformer_lens import ActivationCache
        from transformer_lens.model_bridge import TransformerBridge
        import transformer_lens.patching as patching

        REPO_ROOT = Path("/playpen-ssd/smerrill/deception2")
        SRC_ROOT = REPO_ROOT / "src"
        if str(SRC_ROOT) not in sys.path:
            sys.path.insert(0, str(SRC_ROOT))

        from activation_patching import encode_text_for_model
        from activation_patching_debug import load_commitment_pairs

        try:
            from neel_plotly import line, imshow, scatter
        except ImportError:
            def line(*args, **kwargs):
                return None

            def imshow(*args, **kwargs):
                return None

            def scatter(*args, **kwargs):
                return None

        pio.renderers.default = "colab" if os.getenv("COLAB_RELEASE_TAG") else pio.renderers.default
        pd.options.display.max_colwidth = 200
        pd.options.display.max_columns = 200
        torch.set_grad_enabled(True)
        """
    ),
    code(
        """
        LOCALIZATION_DIR = Path(
            os.environ.get(
                "ATTR_PATCH_LOCALIZATION_DIR",
                str(REPO_ROOT / "DatasetMain" / "bs" / "DeepSeek-R1-Distill-Qwen-7B" / "localization"),
            )
        )
        LOCAL_MODEL_SNAPSHOT = Path(
            os.environ.get(
                "ATTR_PATCH_MODEL_SNAPSHOT",
                "/playpen-ssd/smerrill/huggingface/transformers/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60",
            )
        )
        MODEL_NAME_OR_PATH = str(
            LOCAL_MODEL_SNAPSHOT if LOCAL_MODEL_SNAPSHOT.exists() else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        )
        DTYPE_BY_NAME = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        DTYPE_NAME = os.environ.get("ATTR_PATCH_DTYPE", "bfloat16").strip().lower()
        if DTYPE_NAME not in DTYPE_BY_NAME:
            raise ValueError(f"Unsupported ATTR_PATCH_DTYPE={DTYPE_NAME!r}. Choose from {sorted(DTYPE_BY_NAME)}.")
        MODEL_DTYPE = DTYPE_BY_NAME[DTYPE_NAME]

        if torch.cuda.is_available():
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        visible_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        default_device = "cpu"
        if torch.cuda.is_available():
            if visible_cuda_devices:
                default_device = "cuda:0"
            elif torch.cuda.device_count() > 7:
                default_device = "cuda:7"
            else:
                default_device = f"cuda:{torch.cuda.current_device()}"
        DEVICE = os.environ.get("ATTR_PATCH_DEVICE", default_device)
        USE_ATTN_RESULT = os.environ.get("ATTR_PATCH_USE_ATTN_RESULT", "0") == "1"
        RUN_HEAD_OUT_ATTR = os.environ.get("ATTR_PATCH_RUN_HEAD_OUT_ATTR", "0") == "1"
        RUN_HEAD_QKV_ATTR = os.environ.get("ATTR_PATCH_RUN_HEAD_QKV_ATTR", "0") == "1"
        PROCESS_COMPATIBILITY_WEIGHTS = os.environ.get("ATTR_PATCH_PROCESS_COMPAT_WEIGHTS", "0") == "1"

        PAIR_COUNT = int(os.environ.get("ATTR_PATCH_PAIR_COUNT", "1"))
        PAIR_SEARCH_LIMIT = int(os.environ.get("ATTR_PATCH_PAIR_SEARCH_LIMIT", "256"))
        MIN_COMMITMENT_DELTA = float(os.environ.get("ATTR_PATCH_MIN_COMMITMENT_DELTA", "0.3"))
        MIN_COMMITMENT_DECEPTION_RATE = float(os.environ.get("ATTR_PATCH_MIN_COMMITMENT_DECEPTION_RATE", "0.0"))
        MIN_DONOR_CLARITY_SCORE = float(os.environ.get("ATTR_PATCH_MIN_DONOR_CLARITY_SCORE", "0.0"))
        MAX_INPUT_TOKENS = int(os.environ.get("ATTR_PATCH_MAX_INPUT_TOKENS", "10000"))
        ALIGNMENT_MODE = os.environ.get("ATTR_PATCH_ALIGNMENT_MODE", "equal_token_length")
        CACHE_ATTENTION_PATTERN = os.environ.get("ATTR_PATCH_CACHE_PATTERN", "0") == "1"
        RANDOM_SEED = int(os.environ.get("ATTR_PATCH_SEED", "17"))

        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            if str(DEVICE).startswith("cuda"):
                torch.cuda.set_device(torch.device(DEVICE))
            torch.cuda.manual_seed_all(RANDOM_SEED)

        device_help = (
            "If you want physical GPU 7, either set `ATTR_PATCH_DEVICE=cuda:7` before loading the model, "
            "or launch the kernel with `CUDA_VISIBLE_DEVICES=7` and keep `ATTR_PATCH_DEVICE=cuda:0`."
            if torch.cuda.is_available()
            else "CUDA is unavailable in this kernel."
        )

        display(
            Markdown(
                "\\n".join(
                    [
                        f"- `LOCALIZATION_DIR`: `{LOCALIZATION_DIR}`",
                        f"- `MODEL_NAME_OR_PATH`: `{MODEL_NAME_OR_PATH}`",
                        f"- `DEVICE`: `{DEVICE}`",
                        f"- `CUDA_VISIBLE_DEVICES`: `{visible_cuda_devices or '(unset)'}`",
                        f"- `MODEL_DTYPE`: `{DTYPE_NAME}`",
                        f"- `PAIR_COUNT`: `{PAIR_COUNT}`",
                        f"- `MIN_COMMITMENT_DELTA`: `{MIN_COMMITMENT_DELTA}`",
                        f"- `ALIGNMENT_MODE`: `{ALIGNMENT_MODE}`",
                        f"- `CACHE_ATTENTION_PATTERN`: `{CACHE_ATTENTION_PATTERN}`",
                        f"- `USE_ATTN_RESULT`: `{USE_ATTN_RESULT}`",
                        f"- `RUN_HEAD_OUT_ATTR`: `{RUN_HEAD_OUT_ATTR}`",
                        f"- `RUN_HEAD_QKV_ATTR`: `{RUN_HEAD_QKV_ATTR}`",
                        f"- `PROCESS_COMPATIBILITY_WEIGHTS`: `{PROCESS_COMPATIBILITY_WEIGHTS}`",
                        f"- `PYTORCH_CUDA_ALLOC_CONF`: `{os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(unset)')}`",
                        "",
                        device_help,
                    ]
                )
            )
        )
        """
    ),
    code(
        """
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"


        def token_len(text: str) -> int:
            return int(
                len(
                    tokenizer(
                        text,
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )["input_ids"]
                )
            )


        pairs_df = load_commitment_pairs(
            localization_dir=LOCALIZATION_DIR,
            pair_cache_path=REPO_ROOT / "Cache" / "activation_patching" / "bs_commitment_pairs_for_notebook.jsonl",
            pair_count=max(PAIR_COUNT, 32),
            pair_search_limit=PAIR_SEARCH_LIMIT,
            refresh_cache=False,
            min_commitment_delta=MIN_COMMITMENT_DELTA,
            min_commitment_deception_rate=MIN_COMMITMENT_DECEPTION_RATE,
            min_donor_clarity_score=MIN_DONOR_CLARITY_SCORE,
            disable_tqdm=False,
        ).copy()

        pairs_df["shared_prefix_token_len"] = pairs_df["shared_prefix_text"].map(token_len)
        pairs_df["deceptive_sentence_token_len"] = pairs_df["deceptive_commitment_sentence"].map(token_len)
        pairs_df["truthful_sentence_token_len"] = pairs_df["truthful_donor_sentence"].map(token_len)
        pairs_df["sentence_token_len_gap"] = (
            pairs_df["deceptive_sentence_token_len"] - pairs_df["truthful_sentence_token_len"]
        ).abs()
        pairs_df["same_sentence_token_len"] = (
            pairs_df["deceptive_sentence_token_len"] == pairs_df["truthful_sentence_token_len"]
        )

        if ALIGNMENT_MODE == "equal_token_length":
            aligned_pairs_df = pairs_df.loc[pairs_df["same_sentence_token_len"]].copy()
            if len(aligned_pairs_df) < PAIR_COUNT:
                display(
                    Markdown(
                        f"Only {len(aligned_pairs_df)} pairs had equal tokenized commitment lengths; "
                        "falling back to the smallest length gaps."
                    )
                )
                aligned_pairs_df = pairs_df.copy()
        else:
            aligned_pairs_df = pairs_df.copy()

        aligned_pairs_df = aligned_pairs_df.sort_values(
            [
                "sentence_token_len_gap",
                "commitment_delta",
                "deceptive_prefix_deception_rate",
                "donor_clarity_score",
            ],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
        pairs_df = aligned_pairs_df.head(PAIR_COUNT).reset_index(drop=True)
        pairs_df.insert(0, "notebook_pair_index", np.arange(len(pairs_df), dtype=int))

        display(
            pairs_df[
                [
                    "notebook_pair_index",
                    "example_id",
                    "commitment_delta",
                    "deceptive_prefix_deception_rate",
                    "deceptive_sentence_token_len",
                    "truthful_sentence_token_len",
                    "deceptive_commitment_sentence",
                    "truthful_donor_sentence",
                ]
            ]
        )
        """
    ),
    code(
        """
        model = TransformerBridge.boot_transformers(
            MODEL_NAME_OR_PATH,
            device=DEVICE,
            dtype=MODEL_DTYPE,
            trust_remote_code=True,
        )
        model.enable_compatibility_mode(no_processing=not PROCESS_COMPATIBILITY_WEIGHTS)
        model.set_use_attn_result(USE_ATTN_RESULT)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        first_param = next(model.parameters())
        cuda_summary_lines = []
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                marker = " <= current" if idx == torch.cuda.current_device() else ""
                cuda_summary_lines.append(f"- visible cuda:{idx}: `{torch.cuda.get_device_name(idx)}`{marker}")
            cuda_summary_lines.append(
                f"- allocated on current device: `{torch.cuda.memory_allocated(torch.cuda.current_device()) / 2**30:.2f} GiB`"
            )

        display(
            Markdown(
                "\\n".join(
                    [
                        f"Loaded TransformerLens model with `{model.cfg.n_layers}` layers and `{model.cfg.n_heads}` heads on `{model.cfg.device}`.",
                        f"- first parameter dtype: `{first_param.dtype}`",
                        (
                            "- compatibility-mode weight processing: "
                            f"`{'enabled' if PROCESS_COMPATIBILITY_WEIGHTS else 'disabled'}`"
                        ),
                        *cuda_summary_lines,
                    ]
                )
            )
        )
        """
    ),
    md(
        """
        ## Branch Construction

        For each selected pair we build two teacher-forced branches from the same shared prefix:

        - deceptive row: `p + s_D`
        - truthful row: `p + s_H`

        The clean batch is ordered `[D_0, H_0, D_1, H_1, ...]` and the corrupted batch swaps each
        pair to `[H_0, D_0, H_1, D_1, ...]`. This lets the metric stay pairwise inside a standard
        TransformerLens patching workflow.
        """
    ),
    code(
        """
        def encode_branch(prefix_text: str, full_text: str) -> dict[str, Any]:
            prefix_inputs = encode_text_for_model(tokenizer, prefix_text, max_input_tokens=MAX_INPUT_TOKENS)
            full_inputs = encode_text_for_model(tokenizer, full_text, max_input_tokens=MAX_INPUT_TOKENS)
            prefix_ids = prefix_inputs["input_ids"][0].detach().cpu()
            full_ids = full_inputs["input_ids"][0].detach().cpu()
            prefix_len = int(prefix_ids.shape[0])
            total_len = int(full_ids.shape[0])
            sentence_len = int(total_len - prefix_len)
            if sentence_len <= 0:
                raise ValueError("Commitment branch must add at least one token after the shared prefix.")
            if prefix_len > total_len or not torch.equal(full_ids[:prefix_len], prefix_ids):
                raise ValueError("Full branch tokenization does not begin with the shared prefix tokenization.")
            return {
                "prefix_len": prefix_len,
                "total_len": total_len,
                "sentence_len": sentence_len,
                "score_start_pos": prefix_len,
                "score_stop_pos": total_len,
                "scored_token_count": sentence_len,
                "full_ids": full_ids,
            }


        def pad_token_sequences(token_sequences: list[torch.Tensor], pad_token_id: int) -> torch.Tensor:
            max_len = max(int(seq.shape[0]) for seq in token_sequences)
            batch = torch.full((len(token_sequences), max_len), int(pad_token_id), dtype=torch.long)
            for row_idx, seq in enumerate(token_sequences):
                batch[row_idx, : int(seq.shape[0])] = seq
            return batch


        def make_row_meta(
            pair_row: dict[str, Any],
            branch_role: str,
            branch_text: str,
            sentence_text: str,
            encoded: dict[str, Any],
        ) -> dict[str, Any]:
            return {
                "pair_id": str(pair_row["pair_id"]),
                "example_id": str(pair_row["example_id"]),
                "pair_index": int(pair_row["notebook_pair_index"]),
                "branch_role": str(branch_role),
                "branch_text": str(branch_text),
                "sentence_text": str(sentence_text),
                "prefix_text": str(pair_row["shared_prefix_text"]),
                "prefix_len": int(encoded["prefix_len"]),
                "sentence_len": int(encoded["sentence_len"]),
                "score_start_pos": int(encoded["score_start_pos"]),
                "score_stop_pos": int(encoded["score_stop_pos"]),
                "scored_token_count": int(encoded["scored_token_count"]),
                "full_ids": encoded["full_ids"],
            }


        def build_paired_batches(pairs: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
            clean_rows: list[dict[str, Any]] = []
            corrupted_rows: list[dict[str, Any]] = []

            for pair_row in pairs.to_dict(orient="records"):
                deceptive_encoded = encode_branch(
                    prefix_text=str(pair_row["shared_prefix_text"]),
                    full_text=str(pair_row["deceptive_branch_text"]),
                )
                truthful_encoded = encode_branch(
                    prefix_text=str(pair_row["shared_prefix_text"]),
                    full_text=str(pair_row["truthful_branch_text"]),
                )
                if int(deceptive_encoded["prefix_len"]) != int(truthful_encoded["prefix_len"]):
                    raise ValueError("Prefix token lengths differ across deceptive/truthful branches.")

                deceptive_meta = make_row_meta(
                    pair_row,
                    branch_role="deceptive",
                    branch_text=str(pair_row["deceptive_branch_text"]),
                    sentence_text=str(pair_row["deceptive_commitment_sentence"]),
                    encoded=deceptive_encoded,
                )
                truthful_meta = make_row_meta(
                    pair_row,
                    branch_role="truthful",
                    branch_text=str(pair_row["truthful_branch_text"]),
                    sentence_text=str(pair_row["truthful_donor_sentence"]),
                    encoded=truthful_encoded,
                )

                clean_rows.extend([deceptive_meta, truthful_meta])
                corrupted_rows.extend([copy.deepcopy(truthful_meta), copy.deepcopy(deceptive_meta)])

            clean_tokens = pad_token_sequences([row["full_ids"] for row in clean_rows], tokenizer.pad_token_id)
            corrupted_tokens = pad_token_sequences([row["full_ids"] for row in corrupted_rows], tokenizer.pad_token_id)

            clean_batch = {
                "rows": clean_rows,
                "tokens": clean_tokens.to(model.cfg.device),
            }
            corrupted_batch = {
                "rows": corrupted_rows,
                "tokens": corrupted_tokens.to(model.cfg.device),
            }
            return clean_batch, corrupted_batch


        clean_batch, corrupted_batch = build_paired_batches(pairs_df)


        def batch_rows_to_df(batch: dict[str, Any]) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "row_idx": row_idx,
                        "pair_index": row["pair_index"],
                        "branch_role": row["branch_role"],
                        "prefix_len": row["prefix_len"],
                        "sentence_len": row["sentence_len"],
                        "score_start_pos": row["score_start_pos"],
                        "score_stop_pos": row["score_stop_pos"],
                        "sentence_text": row["sentence_text"],
                    }
                    for row_idx, row in enumerate(batch["rows"])
                ]
            )


        display(batch_rows_to_df(clean_batch).head(12))
        print("clean_tokens shape:", tuple(clean_batch["tokens"].shape))
        print("corrupted_tokens shape:", tuple(corrupted_batch["tokens"].shape))
        """
    ),
    code(
        """
        def mean_token_logprob_by_row(
            logits: torch.Tensor,
            token_tensor: torch.Tensor,
            row_meta: list[dict[str, Any]],
        ) -> torch.Tensor:
            if logits.ndim != 3:
                raise ValueError(f"Expected logits with shape [batch, pos, vocab], got {tuple(logits.shape)}")
            log_probs = F.log_softmax(logits.float(), dim=-1)
            values = []
            for row_idx, meta in enumerate(row_meta):
                start = int(meta["score_start_pos"])
                stop = int(meta["score_stop_pos"])
                if start <= 0 or stop <= start:
                    raise ValueError(f"Invalid span for row {row_idx}: start={start}, stop={stop}")
                step_logits = log_probs[row_idx, start - 1 : stop - 1, :]
                step_target_ids = token_tensor[row_idx, start:stop]
                token_log_probs = step_logits.gather(-1, step_target_ids.unsqueeze(-1)).squeeze(-1)
                values.append(token_log_probs.mean())
            return torch.stack(values, dim=0)


        def paired_margin_from_logits(logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
            row_scores = mean_token_logprob_by_row(logits, batch["tokens"], batch["rows"])
            return (row_scores[0::2] - row_scores[1::2]).mean()


        def make_metric(batch: dict[str, Any], clean_baseline: float, corrupted_baseline: float):
            def metric(logits: torch.Tensor) -> torch.Tensor:
                raw_margin = paired_margin_from_logits(logits, batch)
                return (raw_margin - corrupted_baseline) / (clean_baseline - corrupted_baseline)

            return metric
        """
    ),
    code(
        """
        with torch.inference_mode():
            clean_logits = model(clean_batch["tokens"])
        clean_margin = float(paired_margin_from_logits(clean_logits, clean_batch).item())
        clean_row_scores = mean_token_logprob_by_row(clean_logits, clean_batch["tokens"], clean_batch["rows"]).detach().cpu()

        del clean_logits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.inference_mode():
            corrupted_logits = model(corrupted_batch["tokens"])
        corrupted_margin = float(paired_margin_from_logits(corrupted_logits, corrupted_batch).item())
        del corrupted_logits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        clean_pair_scores_df = pairs_df[
            [
                "notebook_pair_index",
                "example_id",
                "commitment_delta",
                "deceptive_commitment_sentence",
                "truthful_donor_sentence",
            ]
        ].copy()
        clean_pair_scores_df["score_D"] = clean_row_scores[0::2].numpy()
        clean_pair_scores_df["score_H"] = clean_row_scores[1::2].numpy()
        clean_pair_scores_df["metric"] = clean_pair_scores_df["score_D"] - clean_pair_scores_df["score_H"]

        display(clean_pair_scores_df)
        print(f"Clean batch mean margin: {clean_margin:.4f}")
        print(f"Corrupted batch mean margin: {corrupted_margin:.4f}")

        clean_metric = make_metric(clean_batch, clean_margin, corrupted_margin)
        corrupted_metric = make_metric(corrupted_batch, clean_margin, corrupted_margin)

        print("Clean baseline -> 1.0000")
        print("Corrupted baseline -> 0.0000")
        """
    ),
    code(
        """
        PATCH_ACTIVATION_EXACT_NAMES = {
            "hook_embed",
            "hook_pos_embed",
        }
        PATCH_ACTIVATION_BASE_SUFFIXES = (
            "hook_resid_pre",
            "hook_resid_mid",
            "hook_resid_post",
            "hook_attn_out",
            "hook_mlp_out",
        )
        PATCH_ACTIVATION_HEAD_SUFFIXES = (
            "attn.hook_q",
            "attn.hook_k",
            "attn.hook_v",
            "attn.hook_z",
        )


        def filter_patch_activations(name: str) -> bool:
            if name in PATCH_ACTIVATION_EXACT_NAMES:
                return True
            if CACHE_ATTENTION_PATTERN and name.endswith("attn.hook_pattern"):
                return True
            if any(name.endswith(suffix) for suffix in PATCH_ACTIVATION_BASE_SUFFIXES):
                return True
            if RUN_HEAD_QKV_ATTR and any(name.endswith(suffix) for suffix in PATCH_ACTIVATION_HEAD_SUFFIXES):
                return True
            return False


        def _move_cache_tensors(cache: ActivationCache, device: torch.device) -> None:
            for key in list(cache.cache_dict.keys()):
                value = cache.cache_dict[key]
                if isinstance(value, torch.Tensor):
                    cache.cache_dict[key] = value.to(device)

            if "hook_pos_embed" in cache.cache_dict:
                pos_embed = cache.cache_dict["hook_pos_embed"]
                if isinstance(pos_embed, torch.Tensor) and pos_embed.shape[0] == 1:
                    for key, value in cache.cache_dict.items():
                        if key == "hook_pos_embed":
                            continue
                        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] > 1:
                            cache.cache_dict["hook_pos_embed"] = pos_embed.expand(value.shape[0], -1, -1).to(device)
                            break


        def _enable_grad_from_embed(act: torch.Tensor, hook: Any) -> torch.Tensor:
            return act.detach().requires_grad_(True)


        def get_cache_fwd_and_bwd(model: Any, tokens: torch.Tensor, metric: Callable[[torch.Tensor], torch.Tensor]):
            model.reset_hooks()
            grad_cache: dict[str, torch.Tensor] = {}
            cpu_device = torch.device("cpu")

            def backward_cache_hook(act: torch.Tensor, hook: Any):
                grad_cache[hook.name] = act.detach().to(cpu_device)

            model.zero_grad(set_to_none=True)
            try:
                model.add_hook("hook_embed", _enable_grad_from_embed, "fwd")
                model.add_hook(filter_patch_activations, backward_cache_hook, "bwd")
                with torch.enable_grad():
                    output, fwd_cache = model.run_with_cache(tokens, names_filter=filter_patch_activations)
                    value = metric(output)
                    value.backward()
            finally:
                model.reset_hooks()
                model.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for key, hook_point in model.hook_dict.items():
                if hook_point.name != key and filter_patch_activations(key):
                    if hook_point.name in grad_cache and key not in grad_cache:
                        grad_cache[key] = grad_cache[hook_point.name]

            _move_cache_tensors(fwd_cache, cpu_device)
            grad_act_cache = ActivationCache(grad_cache, model)
            _move_cache_tensors(grad_act_cache, cpu_device)
            return value.item(), fwd_cache, grad_act_cache
        """
    ),
    code(
        """
        clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
            model,
            clean_batch["tokens"],
            clean_metric,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
            model,
            corrupted_batch["tokens"],
            corrupted_metric,
        )

        print("Clean value:", clean_value)
        print("Corrupted value:", corrupted_value)
        print("Clean activations cached:", len(clean_cache))
        print("Corrupted activations cached:", len(corrupted_cache))
        print("Clean gradients cached:", len(clean_grad_cache))
        print("Corrupted gradients cached:", len(corrupted_grad_cache))

        assert abs(clean_value - 1.0) < 1e-5, f"Expected clean metric 1.0, got {clean_value}"
        assert abs(corrupted_value) < 1e-5, f"Expected corrupted metric 0.0, got {corrupted_value}"
        """
    ),
    md(
        """
        ## Memory Note

        This adapted notebook now defaults to a lower-memory path:

        - model weights load in `bfloat16`
        - TransformerLens compatibility-mode weight processing is off by default
        - `attn.hook_result` stays off unless you explicitly set `ATTR_PATCH_USE_ATTN_RESULT=1`
        - exact head-output attribution is skipped unless you set `ATTR_PATCH_RUN_HEAD_OUT_ATTR=1`
        - per-head Q/K/V/Z cache collection is skipped unless you set `ATTR_PATCH_RUN_HEAD_QKV_ATTR=1`
        - forward and backward caches are moved to CPU between attribution passes

        The important distinction is:

        - loading the HF model in `bfloat16` is cheap enough
        - compatibility-mode weight processing makes a separate processed `state_dict` copy and
          temporarily upcasts those copied tensors to `float32`, which is what can blow up VRAM

        Full attention-pattern attribution remains possible if you set:

        - `ATTR_PATCH_CACHE_PATTERN=1`

        but that becomes quadratic in sequence length and is usually the first thing to blow up on
        long BS reasoning traces.

        If you change `ATTR_PATCH_DEVICE` or `CUDA_VISIBLE_DEVICES`, restart the kernel before
        re-running the model-load cell so the new device routing actually takes effect.
        """
    ),
    code(
        """
        graph_str_tokens = model.to_str_tokens(clean_batch["tokens"][0])
        HEAD_NAMES = [
            f"L{layer}H{head}" for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)
        ]


        def attr_patch_residual(
            clean_cache: ActivationCache,
            corrupted_cache: ActivationCache,
            corrupted_grad_cache: ActivationCache,
        ):
            clean_residual, residual_labels = clean_cache.accumulated_resid(
                -1,
                incl_mid=True,
                return_labels=True,
            )
            corrupted_residual = corrupted_cache.accumulated_resid(-1, incl_mid=True, return_labels=False)
            corrupted_grad_residual = corrupted_grad_cache.accumulated_resid(-1, incl_mid=True, return_labels=False)
            residual_attr = einops.reduce(
                corrupted_grad_residual * (clean_residual - corrupted_residual),
                "component batch pos d_model -> component pos",
                "sum",
            )
            return residual_attr, residual_labels


        def attr_patch_layer_out(
            clean_cache: ActivationCache,
            corrupted_cache: ActivationCache,
            corrupted_grad_cache: ActivationCache,
        ):
            clean_layer_out, labels = clean_cache.decompose_resid(-1, return_labels=True)
            corrupted_layer_out = corrupted_cache.decompose_resid(-1, return_labels=False)
            corrupted_grad_layer_out = corrupted_grad_cache.decompose_resid(-1, return_labels=False)
            layer_out_attr = einops.reduce(
                corrupted_grad_layer_out * (clean_layer_out - corrupted_layer_out),
                "component batch pos d_model -> component pos",
                "sum",
            )
            return layer_out_attr, labels


        def attr_patch_head_out(
            clean_cache: ActivationCache,
            corrupted_cache: ActivationCache,
            corrupted_grad_cache: ActivationCache,
        ):
            clean_head_out = clean_cache.stack_head_results(-1, return_labels=False)
            corrupted_head_out = corrupted_cache.stack_head_results(-1, return_labels=False)
            corrupted_grad_head_out = corrupted_grad_cache.stack_head_results(-1, return_labels=False)
            head_out_attr = einops.reduce(
                corrupted_grad_head_out * (clean_head_out - corrupted_head_out),
                "component batch pos d_model -> component pos",
                "sum",
            )
            return head_out_attr, HEAD_NAMES


        residual_attr, residual_labels = attr_patch_residual(clean_cache, corrupted_cache, corrupted_grad_cache)
        layer_out_attr, layer_out_labels = attr_patch_layer_out(clean_cache, corrupted_cache, corrupted_grad_cache)

        imshow(
            residual_attr,
            y=residual_labels,
            yaxis="Component",
            xaxis="Position",
            x=[f"{token}_{idx}" for idx, token in enumerate(graph_str_tokens)],
            title="Residual attribution patching",
        )
        imshow(
            layer_out_attr,
            y=layer_out_labels,
            yaxis="Component",
            xaxis="Position",
            x=[f"{token}_{idx}" for idx, token in enumerate(graph_str_tokens)],
            title="Layer-output attribution patching",
        )
        if RUN_HEAD_OUT_ATTR:
            head_out_attr, _ = attr_patch_head_out(clean_cache, corrupted_cache, corrupted_grad_cache)
            imshow(
                einops.reduce(
                    head_out_attr,
                    "(layer head) pos -> layer head",
                    "sum",
                    layer=model.cfg.n_layers,
                    head=model.cfg.n_heads,
                ),
                yaxis="Layer",
                xaxis="Head Index",
                title="Head-output attribution patching summed over position",
            )
        else:
            print(
                "Set ATTR_PATCH_RUN_HEAD_OUT_ATTR=1 to materialize per-head residual outputs. "
                "That view is noticeably more memory intensive on long BS traces."
            )
        """
    ),
    code(
        """
        attribution_cache_dict = {}
        shared_keys = (
            set(corrupted_grad_cache.cache_dict.keys())
            & set(clean_cache.cache_dict.keys())
            & set(corrupted_cache.cache_dict.keys())
        )
        for key in shared_keys:
            if key.endswith("attn.hook_result"):
                continue
            attribution_cache_dict[key] = corrupted_grad_cache.cache_dict[key] * (
                clean_cache.cache_dict[key] - corrupted_cache.cache_dict[key]
            )
        attr_cache = ActivationCache(attribution_cache_dict, model)


        def get_attr_patch_block_every(attr_cache: ActivationCache) -> torch.Tensor:
            resid_pre_attr = einops.reduce(
                attr_cache.stack_activation("resid_pre"),
                "layer batch pos d_model -> layer pos",
                "sum",
            )
            attn_out_attr = einops.reduce(
                attr_cache.stack_activation("attn_out"),
                "layer batch pos d_model -> layer pos",
                "sum",
            )
            mlp_out_attr = einops.reduce(
                attr_cache.stack_activation("mlp_out"),
                "layer batch pos d_model -> layer pos",
                "sum",
            )
            return torch.stack([resid_pre_attr, attn_out_attr, mlp_out_attr], dim=0)


        def get_attr_patch_attn_head_all_pos_every(attr_cache: ActivationCache) -> torch.Tensor:
            head_out_all_pos_attr = einops.reduce(
                attr_cache.stack_activation("z"),
                "layer batch pos head_index d_head -> layer head_index",
                "sum",
            )
            head_q_all_pos_attr = einops.reduce(
                attr_cache.stack_activation("q"),
                "layer batch pos head_index d_head -> layer head_index",
                "sum",
            )
            head_k_all_pos_attr = einops.reduce(
                attr_cache.stack_activation("k"),
                "layer batch pos head_index d_head -> layer head_index",
                "sum",
            )
            head_v_all_pos_attr = einops.reduce(
                attr_cache.stack_activation("v"),
                "layer batch pos head_index d_head -> layer head_index",
                "sum",
            )
            return torch.stack(
                [
                    head_out_all_pos_attr,
                    head_q_all_pos_attr,
                    head_k_all_pos_attr,
                    head_v_all_pos_attr,
                ],
                dim=0,
            )


        def get_attr_patch_attn_head_by_pos_every(attr_cache: ActivationCache) -> torch.Tensor:
            head_out_by_pos_attr = einops.reduce(
                attr_cache.stack_activation("z"),
                "layer batch pos head_index d_head -> layer pos head_index",
                "sum",
            )
            head_q_by_pos_attr = einops.reduce(
                attr_cache.stack_activation("q"),
                "layer batch pos head_index d_head -> layer pos head_index",
                "sum",
            )
            head_k_by_pos_attr = einops.reduce(
                attr_cache.stack_activation("k"),
                "layer batch pos head_index d_head -> layer pos head_index",
                "sum",
            )
            head_v_by_pos_attr = einops.reduce(
                attr_cache.stack_activation("v"),
                "layer batch pos head_index d_head -> layer pos head_index",
                "sum",
            )
            return torch.stack(
                [
                    head_out_by_pos_attr,
                    head_q_by_pos_attr,
                    head_k_by_pos_attr,
                    head_v_by_pos_attr,
                ],
                dim=0,
            )


        every_block_attr_patch_result = get_attr_patch_block_every(attr_cache)

        imshow(
            every_block_attr_patch_result,
            facet_col=0,
            facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
            title="Attribution patching per block",
            xaxis="Position",
            yaxis="Layer",
            x=[f"{token}_{idx}" for idx, token in enumerate(graph_str_tokens)],
        )
        if RUN_HEAD_QKV_ATTR:
            every_head_all_pos_attr_patch_result = get_attr_patch_attn_head_all_pos_every(attr_cache)
            every_head_by_pos_attr_patch_result = get_attr_patch_attn_head_by_pos_every(attr_cache)
            imshow(
                every_head_all_pos_attr_patch_result,
                facet_col=0,
                facet_labels=["Output", "Query", "Key", "Value"],
                title="Attribution patching per head (all positions)",
                xaxis="Head",
                yaxis="Layer",
            )
            imshow(
                einops.rearrange(
                    every_head_by_pos_attr_patch_result,
                    "act_type layer pos head -> act_type (layer head) pos",
                ),
                facet_col=0,
                facet_labels=["Output", "Query", "Key", "Value"],
                title="Attribution patching per head (by position)",
                xaxis="Position",
                yaxis="Layer & Head",
                x=[f"{token}_{idx}" for idx, token in enumerate(graph_str_tokens)],
                y=HEAD_NAMES,
            )
        else:
            print(
                "Set ATTR_PATCH_RUN_HEAD_QKV_ATTR=1 to cache per-head Q/K/V/Z activations. "
                "Those tensors are one of the biggest memory costs on long BS prefixes."
            )
        """
    ),
    code(
        """
        RUN_ACTIVATION_PATCHING = os.environ.get("ATTR_PATCH_RUN_ACT_PATCH", "0") == "1"

        if RUN_ACTIVATION_PATCHING:
            clean_cache_for_patch = ActivationCache(
                {key: value.to(model.cfg.device) for key, value in clean_cache.cache_dict.items()},
                model,
            )
            every_block_act_patch_result = patching.get_act_patch_block_every(
                model,
                corrupted_batch["tokens"],
                clean_cache_for_patch,
                corrupted_metric,
            )
            imshow(
                every_block_act_patch_result,
                facet_col=0,
                facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
                title="Exact activation patching per block",
                xaxis="Position",
                yaxis="Layer",
                x=[f"{token}_{idx}" for idx, token in enumerate(graph_str_tokens)],
            )
        else:
            print("Set ATTR_PATCH_RUN_ACT_PATCH=1 to run exact activation patching comparisons.")
        """
    ),
    md(
        """
        ## Behavioral Validation Later

        This notebook intentionally stops at teacher-forced attribution patching on the matched
        commitment sentence pairs.

        - No continuation generation is used in the attribution metric.
        - The localization-derived counterfactual deception rates are only used to choose the
          commitment junctures and truthful donor sentences.
        - If you want later behavioral validation, use a separate continuation-generation pass
          after selecting circuits or patched sites.
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
)

NOTEBOOK_PATH.write_text(nbf.writes(notebook), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
