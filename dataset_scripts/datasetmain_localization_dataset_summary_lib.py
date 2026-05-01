from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import transformers
from transformers import AutoTokenizer

try:
    from transformers import LlamaTokenizerFast
except Exception:  # pragma: no cover - optional fallback import
    LlamaTokenizerFast = None  # type: ignore[assignment]

try:
    from transformers import Qwen2TokenizerFast
except Exception:  # pragma: no cover - optional fallback import
    Qwen2TokenizerFast = None  # type: ignore[assignment]

import datasetmain_commitment_juncture_prevalence_lib as cj

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETMAIN_ROOT = cj.DATASETMAIN_ROOT
DEFAULT_MAX_FILES_PER_BUNDLE: int | None = None
DEFAULT_NUM_WORKERS = min(20, max(1, (os.cpu_count() or 1)))
DEFAULT_TOKEN_COUNT_MODE = "generic"
DEFAULT_SHOW_PROGRESS = False
DEFAULT_PROGRESS_LEVEL = "bundle"
HF_CACHE_ROOT = Path(
    os.environ.get(
        "HF_CACHE_ROOT",
        str(REPO_ROOT.parent / "huggingface" / "hub"),
    )
).expanduser()

MODEL_TOKENIZER_CONFIGS: dict[str, dict[str, str]] = {
    "DeepSeek-R1-Distill-Qwen-7B": {
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "hf_cache_dir": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B",
        "family": "qwen2",
    },
    "Qwen-7B": {
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "hf_cache_dir": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B",
        "family": "qwen2",
    },
    "DeepSeek-R1-Distill-Qwen-14B": {
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "hf_cache_dir": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
        "family": "qwen2",
    },
    "Qwen-14B": {
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "hf_cache_dir": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
        "family": "qwen2",
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "hf_cache_dir": "models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B",
        "family": "llama",
    },
    "Llama-8B": {
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "hf_cache_dir": "models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B",
        "family": "llama",
    },
    "gpt-oss-20b": {
        "hf_repo": "openai/gpt-oss-20b",
        "hf_cache_dir": "models--openai--gpt-oss-20b",
        "family": "generic",
    },
    "GPT-OSS-20B": {
        "hf_repo": "openai/gpt-oss-20b",
        "hf_cache_dir": "models--openai--gpt-oss-20b",
        "family": "generic",
    },
}

MODEL_TYPE_TOKENIZER_CLASS_FALLBACKS: dict[str, str] = {
    "llama": "LlamaTokenizerFast",
    "qwen2": "Qwen2TokenizerFast",
    "gpt_oss": "PreTrainedTokenizerFast",
}

SUMMARY_COUNT_COLUMNS = [
    "file_count",
    "file_size_bytes_total",
    "reasoning_sentence_total",
    "reasoning_token_total",
    "reasoning_word_total",
    "localized_prefix_total",
    "prompt_sentence_total_unique",
    "prompt_token_total_unique",
    "prompt_word_total_unique",
    "prompt_sentence_total_expanded",
    "prompt_token_total_expanded",
    "prompt_word_total_expanded",
    "continuation_total",
    "continuation_sentence_total",
    "continuation_token_total",
    "continuation_word_total",
    "recovered_json_file_count",
    "skipped_json_file_count",
]

PARSE_ISSUE_COLUMNS = [
    "env_name",
    "env_display",
    "model_name",
    "model_display",
    "bundle_dir",
    "path",
    "source_kind",
    "issue_kind",
    "recovery_strategy",
    "error",
]

_TOKENIZER_CACHE: dict[str, Any] = {}
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", flags=re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+", flags=re.UNICODE)
JSON_START_PATTERN = re.compile(r"[{\[]")
SHARD_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class _RegexTokenizer:
    pad_token_id = 0
    eos_token = None

    def count_tokens(self, text: Any) -> int:
        return len(TOKEN_PATTERN.findall(str(text or '')))


def latest_snapshot_path(root: Path) -> Path | None:
    snapshot_root = root / "snapshots"
    if not snapshot_root.exists():
        return None
    snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
    return snapshots[-1] if snapshots else None


def resolve_model_tokenizer_config(model_name: str) -> dict[str, str] | None:
    candidate_keys = [
        str(model_name).strip(),
        cj.canonical_model_display(model_name),
    ]
    seen: set[str] = set()
    for candidate_key in candidate_keys:
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        model_cfg = MODEL_TOKENIZER_CONFIGS.get(candidate_key)
        if model_cfg is not None:
            return model_cfg
    return None


def resolve_tokenizer_name_or_path(
    model_name: str,
    *,
    hf_cache_root: Path | str = HF_CACHE_ROOT,
) -> str | None:
    model_cfg = resolve_model_tokenizer_config(model_name)
    if model_cfg is None:
        return None
    cached_snapshot = latest_snapshot_path(Path(hf_cache_root).expanduser().resolve() / model_cfg["hf_cache_dir"])
    if cached_snapshot is not None:
        return str(cached_snapshot)
    return str(model_cfg["hf_repo"])


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _resolve_tokenizer_class_name(name_or_path: str) -> str | None:
    path = Path(name_or_path)
    if not path.exists() or not path.is_dir():
        return None

    tokenizer_cfg = _read_json_if_exists(path / "tokenizer_config.json") or {}
    tokenizer_class_name = tokenizer_cfg.get("tokenizer_class")
    if isinstance(tokenizer_class_name, str) and tokenizer_class_name.strip():
        return tokenizer_class_name.strip()

    model_cfg = _read_json_if_exists(path / "config.json") or {}
    model_type = str(model_cfg.get("model_type") or "").strip()
    if not model_type:
        return None
    return MODEL_TYPE_TOKENIZER_CLASS_FALLBACKS.get(model_type)


def _load_tokenizer_from_declared_class(
    name_or_path: str,
    *,
    trust_remote_code: bool = True,
):
    tokenizer_class_name = _resolve_tokenizer_class_name(name_or_path)
    if tokenizer_class_name is None:
        raise ValueError(f"Could not infer tokenizer class for {name_or_path!r}.")

    tokenizer_cls = getattr(transformers, tokenizer_class_name, None)
    if tokenizer_cls is None:
        raise ValueError(
            f"Tokenizer class {tokenizer_class_name!r} is not available in transformers "
            f"{getattr(transformers, '__version__', 'unknown')}."
        )
    return tokenizer_cls.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)


def get_tokenizer(
    model_name: str,
    *,
    hf_cache_root: Path | str = HF_CACHE_ROOT,
):
    cache_key = f"{model_name}||{Path(hf_cache_root).expanduser().resolve()}"
    if cache_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[cache_key]

    tokenizer_name_or_path = resolve_tokenizer_name_or_path(model_name, hf_cache_root=hf_cache_root)
    model_cfg = resolve_model_tokenizer_config(model_name)
    if tokenizer_name_or_path is None or model_cfg is None:
        tokenizer = _RegexTokenizer()
        _TOKENIZER_CACHE[cache_key] = tokenizer
        return tokenizer

    load_attempts = [
        lambda: AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        ),
    ]
    family = str(model_cfg.get('family') or '').strip().lower()
    if family == 'llama' and LlamaTokenizerFast is not None:
        load_attempts.append(lambda: LlamaTokenizerFast.from_pretrained(tokenizer_name_or_path, local_files_only=True))
    if family == 'qwen2' and Qwen2TokenizerFast is not None:
        load_attempts.append(lambda: Qwen2TokenizerFast.from_pretrained(tokenizer_name_or_path, local_files_only=True))

    tokenizer = None
    for load_attempt in load_attempts:
        try:
            tokenizer = load_attempt()
            break
        except Exception:
            continue
    if tokenizer is None:
        try:
            tokenizer = _load_tokenizer_from_declared_class(tokenizer_name_or_path, trust_remote_code=True)
        except Exception:
            tokenizer = None
    if tokenizer is None:
        tokenizer = _RegexTokenizer()

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    _TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def count_text_tokens(
    text: Any,
    *,
    token_count_mode: str = DEFAULT_TOKEN_COUNT_MODE,
    model_name: str | None = None,
    hf_cache_root: Path | str = HF_CACHE_ROOT,
) -> int:
    text_value = str(text or "")
    mode = str(token_count_mode).strip().lower()
    if mode in {"regex", "generic"}:
        return len(TOKEN_PATTERN.findall(text_value))
    if mode == "hf":
        if model_name is None:
            raise ValueError('model_name is required when token_count_mode="hf".')
        tokenizer = get_tokenizer(model_name, hf_cache_root=hf_cache_root)
        if hasattr(tokenizer, "count_tokens"):
            return int(tokenizer.count_tokens(text_value))
        try:
            token_ids = tokenizer.encode(text_value, add_special_tokens=False)
        except Exception:
            token_ids = tokenizer(text_value, add_special_tokens=False)["input_ids"]
        return len(token_ids)
    raise ValueError(f'Unsupported token_count_mode={token_count_mode!r}. Choose from {{"regex", "generic", "hf"}}.')


def count_text_words(text: Any) -> int:
    return len(WORD_PATTERN.findall(str(text or "")))


def count_text_sentences(text: Any) -> int:
    cleaned = str(text or "").strip()
    if not cleaned:
        return 0
    parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(cleaned) if part.strip()]
    return max(1, len(parts))


def build_shard_slug(env_name: str, model_name: str) -> str:
    raw_slug = f"{env_name}__{model_name}"
    slug = SHARD_SLUG_PATTERN.sub("_", raw_slug).strip("._-")
    return slug or "shard"


def list_bundle_specs(
    root: Path | str = DATASETMAIN_ROOT,
    *,
    env_name: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for current_env_name, current_model_name, bundle_dir in cj._bundle_dirs(root):
        if env_name is not None and str(current_env_name) != str(env_name):
            continue
        if model_name is not None and str(current_model_name) != str(model_name):
            continue
        rows.append(
            {
                "env_name": str(current_env_name),
                "env_display": cj.canonical_env_display(current_env_name),
                "model_name": str(current_model_name),
                "model_display": cj.canonical_model_display(current_model_name),
                "bundle_dir": str(bundle_dir),
                "shard_slug": build_shard_slug(str(current_env_name), str(current_model_name)),
            }
        )
    return rows


def _is_incremental_extension_safe(previous_text: str, delta_text: str) -> bool:
    if not previous_text or not delta_text:
        return True
    prev_char = previous_text[-1]
    next_char = delta_text[0]
    if prev_char.isspace() or next_char.isspace():
        return True
    if (prev_char.isalnum() or prev_char == '_') and (next_char.isalnum() or next_char == '_'):
        return False
    if prev_char.isalnum() and next_char in "-'":
        return False
    if prev_char in "-'" and next_char.isalnum():
        return False
    return True


def _empty_bundle_summary(env_name: str, model_name: str, bundle_dir: Path) -> dict[str, Any]:
    return {
        "env_name": env_name,
        "env_display": cj.canonical_env_display(env_name),
        "model_name": model_name,
        "model_display": cj.canonical_model_display(model_name),
        "bundle_dir": str(bundle_dir),
        "file_count": 0,
        "file_size_bytes_total": 0,
        "reasoning_sentence_total": 0,
        "reasoning_token_total": 0,
        "reasoning_word_total": 0,
        "localized_prefix_total": 0,
        "prompt_sentence_total_unique": 0,
        "prompt_token_total_unique": 0,
        "prompt_word_total_unique": 0,
        "prompt_sentence_total_expanded": 0,
        "prompt_token_total_expanded": 0,
        "prompt_word_total_expanded": 0,
        "continuation_total": 0,
        "continuation_sentence_total": 0,
        "continuation_token_total": 0,
        "continuation_word_total": 0,
        "recovered_json_file_count": 0,
        "skipped_json_file_count": 0,
    }


def _extract_continuation_text(generation: dict[str, Any], *, prefix_text: str) -> str:
    gen_text = generation.get("gen_text")
    if gen_text is not None:
        return str(gen_text)
    full_generation_text = str(generation.get("full_generation_text") or "")
    if prefix_text and full_generation_text.startswith(prefix_text):
        return full_generation_text[len(prefix_text):]
    return full_generation_text


def _localization_paths_for_bundle(bundle_dir: Path, *, max_files: int | None) -> list[Path]:
    paths = sorted((bundle_dir / cj.LOCALIZATION_DIRNAME).glob(cj.LOCALIZATION_GLOB))
    if max_files is not None:
        return paths[: int(max_files)]
    return paths


def _validate_localization_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected localization payload to be a JSON object, got {type(payload).__name__}.")
    return payload


def _looks_like_localization_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(key in payload for key in ("history", "raw_text", "messages"))


def _next_json_start(raw_text: str, start_idx: int) -> int | None:
    match = JSON_START_PATTERN.search(raw_text, pos=max(0, int(start_idx)))
    if match is None:
        return None
    return int(match.start())


def _recover_localization_payload_from_text(raw_text: str) -> tuple[dict[str, Any], str] | None:
    decoder = json.JSONDecoder()
    candidate_starts: list[int] = []

    non_whitespace_match = re.search(r"\S", raw_text)
    if non_whitespace_match is not None:
        candidate_starts.append(int(non_whitespace_match.start()))

    first_json_start = _next_json_start(raw_text, 0)
    if first_json_start is not None:
        candidate_starts.append(first_json_start)

    seen_starts: set[int] = set()
    for initial_start in candidate_starts:
        if initial_start in seen_starts:
            continue
        seen_starts.add(initial_start)

        current_start = initial_start
        scanned_candidate_count = 0
        while current_start is not None and scanned_candidate_count < 8:
            try:
                candidate_payload, end_idx = decoder.raw_decode(raw_text, idx=current_start)
            except json.JSONDecodeError:
                current_start = _next_json_start(raw_text, current_start + 1)
                scanned_candidate_count += 1
                continue

            if _looks_like_localization_payload(candidate_payload):
                payload = _validate_localization_payload(candidate_payload)
                strategy = "raw_decode_first_object"
                if current_start != initial_start or scanned_candidate_count > 0:
                    strategy = "raw_decode_candidate_scan"
                if raw_text[end_idx:].strip():
                    strategy = f"{strategy}_with_trailing_extra"
                return payload, strategy

            current_start = _next_json_start(raw_text, end_idx)
            scanned_candidate_count += 1

    return None


def _join_recovery_steps(recovery_steps: list[str]) -> str | None:
    deduped: list[str] = []
    seen: set[str] = set()
    for step in recovery_steps:
        normalized = str(step).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    if not deduped:
        return None
    return "+".join(deduped)


def _join_error_messages(error_messages: list[str]) -> str:
    cleaned = [str(message).strip() for message in error_messages if str(message).strip()]
    if not cleaned:
        return "Unknown localization parse failure."
    return " | ".join(cleaned)


def load_localization_payload(path: Path | str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path_obj = Path(path)
    recovery_steps: list[str] = []
    error_messages: list[str] = []

    try:
        raw_bytes = path_obj.read_bytes()
    except Exception as exc:
        return None, {
            "issue_kind": "skipped",
            "recovery_strategy": None,
            "error": repr(exc),
        }

    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        recovery_steps.append("utf8_replace")
        error_messages.append(repr(exc))

    try:
        payload = _validate_localization_payload(json.loads(raw_text))
        if recovery_steps:
            return payload, {
                "issue_kind": "recovered",
                "recovery_strategy": _join_recovery_steps(recovery_steps),
                "error": _join_error_messages(error_messages),
            }
        return payload, None
    except Exception as exc:
        error_messages.append(repr(exc))

    recovered = _recover_localization_payload_from_text(raw_text)
    if recovered is not None:
        payload, recovery_strategy = recovered
        recovery_steps.append(recovery_strategy)
        return payload, {
            "issue_kind": "recovered",
            "recovery_strategy": _join_recovery_steps(recovery_steps),
            "error": _join_error_messages(error_messages),
        }

    return None, {
        "issue_kind": "skipped",
        "recovery_strategy": _join_recovery_steps(recovery_steps),
        "error": _join_error_messages(error_messages),
    }


def _build_parse_issue_row(
    *,
    env_name: str,
    model_name: str,
    bundle_dir: Path,
    path: Path,
    issue: dict[str, Any],
) -> dict[str, Any]:
    return {
        "env_name": env_name,
        "env_display": cj.canonical_env_display(env_name),
        "model_name": model_name,
        "model_display": cj.canonical_model_display(model_name),
        "bundle_dir": str(bundle_dir),
        "path": str(path),
        "source_kind": "localization_json",
        "issue_kind": issue.get("issue_kind"),
        "recovery_strategy": issue.get("recovery_strategy"),
        "error": issue.get("error"),
    }


def _summarize_localization_payload(
    payload: dict[str, Any],
    *,
    model_name: str,
    path: Path,
    token_count_mode: str,
    hf_cache_root: Path | str,
) -> dict[str, int]:
    history = payload.get("history") or []
    if not isinstance(history, list):
        history = []
    raw_text = str(payload.get("raw_text") or "")

    file_stats = {
        "file_count": 1,
        "file_size_bytes_total": int(path.stat().st_size),
        "reasoning_sentence_total": len(history),
        "reasoning_token_total": 0,
        "reasoning_word_total": 0,
        "localized_prefix_total": 0,
        "prompt_sentence_total_unique": 0,
        "prompt_token_total_unique": 0,
        "prompt_word_total_unique": 0,
        "prompt_sentence_total_expanded": 0,
        "prompt_token_total_expanded": 0,
        "prompt_word_total_expanded": 0,
        "continuation_total": 0,
        "continuation_sentence_total": 0,
        "continuation_token_total": 0,
        "continuation_word_total": 0,
    }

    last_prefix_text = ""
    last_prefix_token_count = 0
    last_prefix_word_count = 0

    for sentence_pos, row in enumerate(history, start=1):
        row_dict = row if isinstance(row, dict) else {}
        prefix_text = str(row_dict.get("prefix_text") or "")
        delta_text = prefix_text[len(last_prefix_text):] if prefix_text.startswith(last_prefix_text) else ""

        if delta_text and _is_incremental_extension_safe(last_prefix_text, delta_text):
            prompt_token_count = last_prefix_token_count + count_text_tokens(
                delta_text,
                token_count_mode=token_count_mode,
                model_name=model_name,
                hf_cache_root=hf_cache_root,
            )
            prompt_word_count = last_prefix_word_count + count_text_words(delta_text)
        else:
            prompt_token_count = count_text_tokens(
                prefix_text,
                token_count_mode=token_count_mode,
                model_name=model_name,
                hf_cache_root=hf_cache_root,
            )
            prompt_word_count = count_text_words(prefix_text)
        prompt_sentence_count = sentence_pos

        last_prefix_text = prefix_text
        last_prefix_token_count = prompt_token_count
        last_prefix_word_count = prompt_word_count

        file_stats["localized_prefix_total"] += 1
        file_stats["prompt_sentence_total_unique"] += prompt_sentence_count
        file_stats["prompt_token_total_unique"] += prompt_token_count
        file_stats["prompt_word_total_unique"] += prompt_word_count

        generations = row_dict.get("generations") or []
        if not isinstance(generations, list):
            generations = [generations]
        generation_count = len(generations)
        file_stats["continuation_total"] += generation_count
        file_stats["prompt_sentence_total_expanded"] += prompt_sentence_count * generation_count
        file_stats["prompt_token_total_expanded"] += prompt_token_count * generation_count
        file_stats["prompt_word_total_expanded"] += prompt_word_count * generation_count

        for generation in generations:
            generation_dict = generation if isinstance(generation, dict) else {}
            continuation_text = _extract_continuation_text(generation_dict, prefix_text=prefix_text)
            file_stats["continuation_sentence_total"] += count_text_sentences(continuation_text)
            file_stats["continuation_token_total"] += count_text_tokens(
                continuation_text,
                token_count_mode=token_count_mode,
                model_name=model_name,
                hf_cache_root=hf_cache_root,
            )
            file_stats["continuation_word_total"] += count_text_words(continuation_text)

    if history and raw_text == last_prefix_text:
        file_stats["reasoning_token_total"] = int(last_prefix_token_count)
        file_stats["reasoning_word_total"] = int(last_prefix_word_count)
    else:
        file_stats["reasoning_token_total"] = count_text_tokens(
            raw_text,
            token_count_mode=token_count_mode,
            model_name=model_name,
            hf_cache_root=hf_cache_root,
        )
        file_stats["reasoning_word_total"] = count_text_words(raw_text)

    return file_stats


def _summarize_localization_bundle_impl(
    bundle_dir: Path | str,
    *,
    env_name: str,
    model_name: str,
    max_files: int | None = DEFAULT_MAX_FILES_PER_BUNDLE,
    token_count_mode: str = DEFAULT_TOKEN_COUNT_MODE,
    hf_cache_root: Path | str = HF_CACHE_ROOT,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle_path = Path(bundle_dir)
    summary = _empty_bundle_summary(env_name, model_name, bundle_path)
    parse_issues: list[dict[str, Any]] = []

    paths = _localization_paths_for_bundle(bundle_path, max_files=max_files)
    iterator = paths
    if show_progress:
        iterator = tqdm(
            paths,
            total=len(paths),
            desc=progress_desc or f"{cj.canonical_env_display(env_name)} | {cj.canonical_model_display(model_name)}",
            unit="file",
        )

    for path in iterator:
        payload, issue = load_localization_payload(path)
        if payload is None:
            summary["skipped_json_file_count"] += 1
            if issue is not None:
                parse_issues.append(
                    _build_parse_issue_row(
                        env_name=env_name,
                        model_name=model_name,
                        bundle_dir=bundle_path,
                        path=path,
                        issue=issue,
                    )
                )
            continue

        try:
            file_stats = _summarize_localization_payload(
                payload,
                model_name=model_name,
                path=path,
                token_count_mode=token_count_mode,
                hf_cache_root=hf_cache_root,
            )
        except Exception as exc:
            summary["skipped_json_file_count"] += 1
            parse_issues.append(
                _build_parse_issue_row(
                    env_name=env_name,
                    model_name=model_name,
                    bundle_dir=bundle_path,
                    path=path,
                    issue={
                        "issue_kind": "skipped",
                        "recovery_strategy": None if issue is None else issue.get("recovery_strategy"),
                        "error": repr(exc),
                    },
                )
            )
            continue

        for key, value in file_stats.items():
            summary[key] += int(value)

        if issue is not None and str(issue.get("issue_kind")) == "recovered":
            summary["recovered_json_file_count"] += 1
            parse_issues.append(
                _build_parse_issue_row(
                    env_name=env_name,
                    model_name=model_name,
                    bundle_dir=bundle_path,
                    path=path,
                    issue=issue,
                )
            )

    return summary, parse_issues


def summarize_localization_bundle(
    bundle_dir: Path | str,
    *,
    env_name: str,
    model_name: str,
    max_files: int | None = DEFAULT_MAX_FILES_PER_BUNDLE,
    token_count_mode: str = DEFAULT_TOKEN_COUNT_MODE,
    hf_cache_root: Path | str = HF_CACHE_ROOT,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    summary, _ = _summarize_localization_bundle_impl(
        bundle_dir,
        env_name=env_name,
        model_name=model_name,
        max_files=max_files,
        token_count_mode=token_count_mode,
        hf_cache_root=hf_cache_root,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    return summary


def summarize_localization_bundle_with_issues(
    bundle_dir: Path | str,
    *,
    env_name: str,
    model_name: str,
    max_files: int | None = DEFAULT_MAX_FILES_PER_BUNDLE,
    token_count_mode: str = DEFAULT_TOKEN_COUNT_MODE,
    hf_cache_root: Path | str = HF_CACHE_ROOT,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary, parse_issues = _summarize_localization_bundle_impl(
        bundle_dir,
        env_name=env_name,
        model_name=model_name,
        max_files=max_files,
        token_count_mode=token_count_mode,
        hf_cache_root=hf_cache_root,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    return summary, pd.DataFrame(parse_issues, columns=PARSE_ISSUE_COLUMNS)


def _summarize_localization_bundle_worker(
    job: tuple[str, str, str, int | None, str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env_name, model_name, bundle_dir_str, max_files, token_count_mode, hf_cache_root_str = job
    return _summarize_localization_bundle_impl(
        bundle_dir_str,
        env_name=env_name,
        model_name=model_name,
        max_files=max_files,
        token_count_mode=token_count_mode,
        hf_cache_root=hf_cache_root_str,
        show_progress=False,
    )


def empty_parse_issue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PARSE_ISSUE_COLUMNS)


def build_bundle_summary_df(
    root: Path | str = DATASETMAIN_ROOT,
    *,
    max_files_per_bundle: int | None = DEFAULT_MAX_FILES_PER_BUNDLE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    token_count_mode: str = DEFAULT_TOKEN_COUNT_MODE,
    hf_cache_root: Path | str = HF_CACHE_ROOT,
    show_progress: bool = DEFAULT_SHOW_PROGRESS,
    progress_level: str = DEFAULT_PROGRESS_LEVEL,
    env_name: str | None = None,
    model_name: str | None = None,
    return_parse_issues: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    progress_level_key = str(progress_level).strip().lower()
    if progress_level_key not in {"bundle", "file"}:
        raise ValueError(f"Unsupported progress_level={progress_level!r}. Choose from {{'bundle', 'file'}}.")

    jobs = [
        (
            bundle_spec["env_name"],
            bundle_spec["model_name"],
            bundle_spec["bundle_dir"],
            max_files_per_bundle,
            str(token_count_mode),
            str(hf_cache_root),
        )
        for bundle_spec in list_bundle_specs(root, env_name=env_name, model_name=model_name)
    ]
    if not jobs:
        if return_parse_issues:
            return pd.DataFrame(), empty_parse_issue_df()
        return pd.DataFrame()

    results: list[tuple[dict[str, Any], list[dict[str, Any]]]]
    if int(num_workers) <= 1:
        if show_progress and progress_level_key == "file":
            results = [
                _summarize_localization_bundle_impl(
                    bundle_dir_str,
                    env_name=current_env_name,
                    model_name=current_model_name,
                    max_files=max_files_per_bundle,
                    token_count_mode=token_count_mode,
                    hf_cache_root=hf_cache_root,
                    show_progress=True,
                    progress_desc=(
                        f"{cj.canonical_env_display(current_env_name)} | "
                        f"{cj.canonical_model_display(current_model_name)}"
                    ),
                )
                for current_env_name, current_model_name, bundle_dir_str, _, _, _ in jobs
            ]
        else:
            iterator = jobs
            if show_progress:
                iterator = tqdm(jobs, total=len(jobs), desc="Summarizing bundles", unit="bundle")
            results = [_summarize_localization_bundle_worker(job) for job in iterator]
    else:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            mapped = executor.map(_summarize_localization_bundle_worker, jobs)
            if show_progress:
                mapped = tqdm(mapped, total=len(jobs), desc="Summarizing bundles", unit="bundle")
            results = list(mapped)

    rows = [summary for summary, _ in results]
    parse_issue_rows = [row for _, issue_rows in results for row in issue_rows]

    bundle_df = pd.DataFrame(rows)
    if bundle_df.empty:
        if return_parse_issues:
            return bundle_df, pd.DataFrame(parse_issue_rows, columns=PARSE_ISSUE_COLUMNS)
        return bundle_df

    bundle_df = _ensure_summary_count_columns(bundle_df)
    bundle_df["_model_sort"] = bundle_df["model_display"].map(cj._model_sort_key)
    bundle_df["_env_sort"] = bundle_df["env_display"].map(cj._env_sort_key)
    bundle_df = bundle_df.sort_values(["_model_sort", "_env_sort"]).drop(
        columns=["_model_sort", "_env_sort"],
        errors="ignore",
    )
    finalized_bundle_df = finalize_summary_df(bundle_df.reset_index(drop=True))
    if return_parse_issues:
        return finalized_bundle_df, pd.DataFrame(parse_issue_rows, columns=PARSE_ISSUE_COLUMNS)
    return finalized_bundle_df


def _ensure_summary_count_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    out = summary_df.copy()
    for column in SUMMARY_COUNT_COLUMNS:
        if column not in out.columns:
            out[column] = 0
    return out


def _aggregate_group_rows(group_df: pd.DataFrame) -> dict[str, Any]:
    prepared_df = _ensure_summary_count_columns(group_df)
    return {
        column: int(pd.to_numeric(prepared_df[column], errors="coerce").fillna(0).sum())
        for column in SUMMARY_COUNT_COLUMNS
    }


def _safe_divide(numerator: Any, denominator: Any) -> float:
    denom = float(denominator)
    if denom == 0.0:
        return float("nan")
    return float(numerator) / denom


def finalize_summary_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    out = _ensure_summary_count_columns(summary_df)
    out["avg_reasoning_sentences"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["reasoning_sentence_total"], out["file_count"], strict=False)
    ]
    out["avg_reasoning_tokens"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["reasoning_token_total"], out["file_count"], strict=False)
    ]
    out["avg_reasoning_words"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["reasoning_word_total"], out["file_count"], strict=False)
    ]
    out["avg_prefixes_localized"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["localized_prefix_total"], out["file_count"], strict=False)
    ]
    out["avg_localized_traces_per_reasoning_trace"] = out["avg_prefixes_localized"]
    out["avg_reasoning_tokens_per_sentence"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["reasoning_token_total"], out["reasoning_sentence_total"], strict=False)
    ]
    out["avg_reasoning_words_per_sentence"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["reasoning_word_total"], out["reasoning_sentence_total"], strict=False)
    ]
    out["avg_continuation_tokens"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["continuation_token_total"], out["continuation_total"], strict=False)
    ]
    out["avg_continuation_words"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["continuation_word_total"], out["continuation_total"], strict=False)
    ]
    out["avg_continuations_per_prefix"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["continuation_total"], out["localized_prefix_total"], strict=False)
    ]
    out["avg_continuations_per_reasoning_trace"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["continuation_total"], out["file_count"], strict=False)
    ]
    out["avg_file_size_mb"] = pd.to_numeric(out["file_size_bytes_total"], errors="coerce") / (1024 ** 2)
    out["file_size_tb"] = pd.to_numeric(out["file_size_bytes_total"], errors="coerce") / (10 ** 12)
    out["expanded_dataset_sentence_total"] = (
        pd.to_numeric(out["prompt_sentence_total_expanded"], errors="coerce").fillna(0)
        + pd.to_numeric(out["continuation_sentence_total"], errors="coerce").fillna(0)
    ).astype("int64")
    out["expanded_dataset_token_total"] = (
        pd.to_numeric(out["prompt_token_total_expanded"], errors="coerce").fillna(0)
        + pd.to_numeric(out["continuation_token_total"], errors="coerce").fillna(0)
    ).astype("int64")
    out["expanded_dataset_word_total"] = (
        pd.to_numeric(out["prompt_word_total_expanded"], errors="coerce").fillna(0)
        + pd.to_numeric(out["continuation_word_total"], errors="coerce").fillna(0)
    ).astype("int64")
    out["attempted_json_file_total"] = (
        pd.to_numeric(out["file_count"], errors="coerce").fillna(0)
        + pd.to_numeric(out["skipped_json_file_count"], errors="coerce").fillna(0)
    ).astype("int64")
    out["recovered_json_rate"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["recovered_json_file_count"], out["attempted_json_file_total"], strict=False)
    ]
    out["skipped_json_rate"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["skipped_json_file_count"], out["attempted_json_file_total"], strict=False)
    ]
    out["expanded_token_multiplier"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["expanded_dataset_token_total"], out["reasoning_token_total"], strict=False)
    ]
    out["expanded_word_multiplier"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["expanded_dataset_word_total"], out["reasoning_word_total"], strict=False)
    ]
    out["expanded_sentence_multiplier"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["expanded_dataset_sentence_total"], out["reasoning_sentence_total"], strict=False)
    ]
    out["prompt_token_share_of_expanded"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["prompt_token_total_expanded"], out["expanded_dataset_token_total"], strict=False)
    ]
    out["continuation_token_share_of_expanded"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["continuation_token_total"], out["expanded_dataset_token_total"], strict=False)
    ]
    out["avg_expanded_tokens_per_reasoning_trace"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["expanded_dataset_token_total"], out["file_count"], strict=False)
    ]
    out["avg_expanded_words_per_reasoning_trace"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["expanded_dataset_word_total"], out["file_count"], strict=False)
    ]
    out["avg_expanded_sentences_per_reasoning_trace"] = [
        _safe_divide(num, denom)
        for num, denom in zip(out["expanded_dataset_sentence_total"], out["file_count"], strict=False)
    ]
    return out


def summarize_groups(
    bundle_df: pd.DataFrame,
    groupby_columns: list[str],
) -> pd.DataFrame:
    if bundle_df.empty:
        return pd.DataFrame(columns=groupby_columns)

    rows: list[dict[str, Any]] = []
    for group_keys, group_df in bundle_df.groupby(groupby_columns, dropna=False, observed=True, sort=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        row = {column: value for column, value in zip(groupby_columns, group_keys, strict=True)}
        row.update(_aggregate_group_rows(group_df))
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if "model_display" in summary_df.columns:
        summary_df["_model_sort"] = summary_df["model_display"].map(cj._model_sort_key)
    if "env_display" in summary_df.columns:
        summary_df["_env_sort"] = summary_df["env_display"].map(cj._env_sort_key)
    sort_columns = [column for column in ["_model_sort", "_env_sort"] if column in summary_df.columns]
    if sort_columns:
        summary_df = summary_df.sort_values(sort_columns)
    summary_df = summary_df.drop(columns=["_model_sort", "_env_sort"], errors="ignore")
    return finalize_summary_df(summary_df.reset_index(drop=True))


def combine_bundle_summary_dfs(bundle_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty_frames = [frame.copy() for frame in bundle_dfs if frame is not None and not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame()

    combined_df = pd.concat(non_empty_frames, ignore_index=True, sort=False)
    combined_df = _ensure_summary_count_columns(combined_df)

    duplicate_key_columns = [column for column in ["env_name", "model_name", "bundle_dir"] if column in combined_df.columns]
    if len(duplicate_key_columns) == 3 and combined_df.duplicated(duplicate_key_columns).any():
        duplicate_records = combined_df.loc[
            combined_df.duplicated(duplicate_key_columns, keep=False),
            duplicate_key_columns,
        ].drop_duplicates()
        sample_records = duplicate_records.head(5).to_dict(orient="records")
        raise ValueError(
            "Duplicate bundle summaries found while combining shards. "
            f"Sample duplicates: {sample_records}"
        )

    if "model_display" in combined_df.columns:
        combined_df["_model_sort"] = combined_df["model_display"].map(cj._model_sort_key)
    if "env_display" in combined_df.columns:
        combined_df["_env_sort"] = combined_df["env_display"].map(cj._env_sort_key)
    sort_columns = [column for column in ["_model_sort", "_env_sort"] if column in combined_df.columns]
    if sort_columns:
        combined_df = combined_df.sort_values(sort_columns)
    combined_df = combined_df.drop(columns=["_model_sort", "_env_sort"], errors="ignore")
    return finalize_summary_df(combined_df.reset_index(drop=True))


def make_requested_summary_table(
    summary_df: pd.DataFrame,
    *,
    include_model: bool = True,
    include_environment: bool = False,
) -> pd.DataFrame:
    if summary_df.empty:
        columns: list[str] = []
        if include_model:
            columns.append("Model")
        if include_environment:
            columns.append("Environment")
        columns.extend(
            [
                "Avg. reasoning sent.",
                "Avg. reasoning tokens",
                "Avg. reasoning words",
                "Avg. words / reasoning sent.",
                "Avg. localized traces / reasoning trace",
                "Avg. continuation tokens",
            ]
        )
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame()
    if include_model and "model_display" in summary_df.columns:
        out["Model"] = summary_df["model_display"].astype(str)
    if include_environment and "env_display" in summary_df.columns:
        out["Environment"] = summary_df["env_display"].astype(str)
    out["Avg. reasoning sent."] = pd.to_numeric(summary_df["avg_reasoning_sentences"], errors="coerce")
    out["Avg. reasoning tokens"] = pd.to_numeric(summary_df["avg_reasoning_tokens"], errors="coerce")
    out["Avg. reasoning words"] = pd.to_numeric(summary_df["avg_reasoning_words"], errors="coerce")
    out["Avg. words / reasoning sent."] = pd.to_numeric(
        summary_df["avg_reasoning_words_per_sentence"],
        errors="coerce",
    )
    out["Avg. localized traces / reasoning trace"] = pd.to_numeric(
        summary_df["avg_localized_traces_per_reasoning_trace"],
        errors="coerce",
    )
    out["Avg. continuation tokens"] = pd.to_numeric(summary_df["avg_continuation_tokens"], errors="coerce")
    return out


def make_total_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["Metric", "Value"])

    totals = _aggregate_group_rows(summary_df)
    totals = finalize_summary_df(pd.DataFrame([totals])).iloc[0]

    rows = [
        ("Localization files", int(totals["file_count"])),
        ("Attempted localization JSONs", int(totals["attempted_json_file_total"])),
        ("Recovered localization JSONs", int(totals["recovered_json_file_count"])),
        ("Skipped broken localization JSONs", int(totals["skipped_json_file_count"])),
        ("Recovered localization JSON rate", float(totals["recovered_json_rate"])),
        ("Skipped broken localization JSON rate", float(totals["skipped_json_rate"])),
        ("Localized sentences", int(totals["localized_prefix_total"])),
        ("Continuations", int(totals["continuation_total"])),
        ("Unique reasoning sentences", int(totals["reasoning_sentence_total"])),
        ("Unique reasoning tokens", int(totals["reasoning_token_total"])),
        ("Unique reasoning words", int(totals["reasoning_word_total"])),
        ("Unique localized-prefix tokens", int(totals["prompt_token_total_unique"])),
        ("Unique localized-prefix words", int(totals["prompt_word_total_unique"])),
        ("Expanded prompt sentences", int(totals["prompt_sentence_total_expanded"])),
        ("Expanded prompt tokens", int(totals["prompt_token_total_expanded"])),
        ("Expanded prompt words", int(totals["prompt_word_total_expanded"])),
        ("Continuation sentences", int(totals["continuation_sentence_total"])),
        ("Continuation tokens", int(totals["continuation_token_total"])),
        ("Continuation words", int(totals["continuation_word_total"])),
        ("Total sentences in expanded dataset", int(totals["expanded_dataset_sentence_total"])),
        ("Total tokens in expanded dataset", int(totals["expanded_dataset_token_total"])),
        ("Total words in expanded dataset", int(totals["expanded_dataset_word_total"])),
        ("Expanded / reasoning token multiplier", float(totals["expanded_token_multiplier"])),
        ("Expanded / reasoning word multiplier", float(totals["expanded_word_multiplier"])),
        ("Expanded / reasoning sentence multiplier", float(totals["expanded_sentence_multiplier"])),
        ("Prompt token share of expanded dataset", float(totals["prompt_token_share_of_expanded"])),
        ("Continuation token share of expanded dataset", float(totals["continuation_token_share_of_expanded"])),
        ("Avg. reasoning sentences / trace", float(totals["avg_reasoning_sentences"])),
        ("Avg. reasoning tokens / trace", float(totals["avg_reasoning_tokens"])),
        ("Avg. reasoning words / trace", float(totals["avg_reasoning_words"])),
        ("Avg. reasoning words / sentence", float(totals["avg_reasoning_words_per_sentence"])),
        ("Avg. continuations / localized trace", float(totals["avg_continuations_per_prefix"])),
        ("Avg. continuations / reasoning trace", float(totals["avg_continuations_per_reasoning_trace"])),
        ("Avg. continuation tokens", float(totals["avg_continuation_tokens"])),
        ("Avg. continuation words", float(totals["avg_continuation_words"])),
        ("Localization file size (bytes)", int(totals["file_size_bytes_total"])),
        ("Localization file size (TB)", float(totals["file_size_tb"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _share_percent(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    total = float(numeric.fillna(0).sum())
    if total == 0.0:
        return pd.Series([float("nan")] * len(numeric), index=values.index, dtype="float64")
    return (numeric / total) * 100.0


def make_dataset_overview_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["Metric", "Value"])

    totals = finalize_summary_df(pd.DataFrame([_aggregate_group_rows(summary_df)])).iloc[0]
    rows = [
        ("Reasoning traces", int(totals["file_count"])),
        ("Attempted localization JSONs", int(totals["attempted_json_file_total"])),
        ("Recovered localization JSONs", int(totals["recovered_json_file_count"])),
        ("Skipped broken localization JSONs", int(totals["skipped_json_file_count"])),
        ("Recovery rate (%)", float(totals["recovered_json_rate"]) * 100.0),
        ("Skip rate (%)", float(totals["skipped_json_rate"]) * 100.0),
        ("Localized traces", int(totals["localized_prefix_total"])),
        ("Continuations", int(totals["continuation_total"])),
        ("Unique reasoning sentences", int(totals["reasoning_sentence_total"])),
        ("Unique reasoning tokens", int(totals["reasoning_token_total"])),
        ("Unique reasoning words", int(totals["reasoning_word_total"])),
        ("Expanded dataset sentences", int(totals["expanded_dataset_sentence_total"])),
        ("Expanded dataset tokens", int(totals["expanded_dataset_token_total"])),
        ("Expanded dataset words", int(totals["expanded_dataset_word_total"])),
        ("Expanded / reasoning sentence multiplier", float(totals["expanded_sentence_multiplier"])),
        ("Expanded / reasoning token multiplier", float(totals["expanded_token_multiplier"])),
        ("Expanded / reasoning word multiplier", float(totals["expanded_word_multiplier"])),
        ("Prompt token share of expanded dataset (%)", float(totals["prompt_token_share_of_expanded"]) * 100.0),
        (
            "Continuation token share of expanded dataset (%)",
            float(totals["continuation_token_share_of_expanded"]) * 100.0,
        ),
        ("Avg. reasoning sentences / trace", float(totals["avg_reasoning_sentences"])),
        ("Avg. reasoning tokens / trace", float(totals["avg_reasoning_tokens"])),
        ("Avg. reasoning words / trace", float(totals["avg_reasoning_words"])),
        ("Avg. reasoning tokens / sentence", float(totals["avg_reasoning_tokens_per_sentence"])),
        ("Avg. reasoning words / sentence", float(totals["avg_reasoning_words_per_sentence"])),
        ("Avg. localized traces / reasoning trace", float(totals["avg_localized_traces_per_reasoning_trace"])),
        ("Avg. continuations / localized trace", float(totals["avg_continuations_per_prefix"])),
        ("Avg. continuations / reasoning trace", float(totals["avg_continuations_per_reasoning_trace"])),
        ("Avg. continuation tokens", float(totals["avg_continuation_tokens"])),
        ("Avg. continuation words", float(totals["avg_continuation_words"])),
        ("Avg. expanded tokens / reasoning trace", float(totals["avg_expanded_tokens_per_reasoning_trace"])),
        ("Avg. expanded words / reasoning trace", float(totals["avg_expanded_words_per_reasoning_trace"])),
        (
            "Avg. expanded sentences / reasoning trace",
            float(totals["avg_expanded_sentences_per_reasoning_trace"]),
        ),
        ("Localization file size (TB)", float(totals["file_size_tb"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def make_paper_scale_table(
    summary_df: pd.DataFrame,
    *,
    include_model: bool = True,
    include_environment: bool = False,
) -> pd.DataFrame:
    if summary_df.empty:
        columns: list[str] = []
        if include_model:
            columns.append("Model")
        if include_environment:
            columns.append("Environment")
        columns.extend(
            [
                "Reasoning Traces",
                "Share of Traces (%)",
                "Localized Traces",
                "Continuations",
                "Expanded Dataset Tokens",
                "Share of Expanded Tokens (%)",
                "Expanded Dataset Words",
                "Expanded Dataset Sentences",
                "Avg. reasoning sent.",
                "Avg. reasoning tokens",
                "Avg. words / reasoning sent.",
                "Avg. continuations / trace",
                "Avg. continuation tokens",
                "Expanded token multiplier",
                "Recovery rate (%)",
                "Skip rate (%)",
                "File Size (TB)",
            ]
        )
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame()
    if include_model and "model_display" in summary_df.columns:
        out["Model"] = summary_df["model_display"].astype(str)
    if include_environment and "env_display" in summary_df.columns:
        out["Environment"] = summary_df["env_display"].astype(str)
    out["Reasoning Traces"] = pd.to_numeric(summary_df["file_count"], errors="coerce")
    out["Share of Traces (%)"] = _share_percent(summary_df["file_count"])
    out["Localized Traces"] = pd.to_numeric(summary_df["localized_prefix_total"], errors="coerce")
    out["Continuations"] = pd.to_numeric(summary_df["continuation_total"], errors="coerce")
    out["Expanded Dataset Tokens"] = pd.to_numeric(summary_df["expanded_dataset_token_total"], errors="coerce")
    out["Share of Expanded Tokens (%)"] = _share_percent(summary_df["expanded_dataset_token_total"])
    out["Expanded Dataset Words"] = pd.to_numeric(summary_df["expanded_dataset_word_total"], errors="coerce")
    out["Expanded Dataset Sentences"] = pd.to_numeric(summary_df["expanded_dataset_sentence_total"], errors="coerce")
    out["Avg. reasoning sent."] = pd.to_numeric(summary_df["avg_reasoning_sentences"], errors="coerce")
    out["Avg. reasoning tokens"] = pd.to_numeric(summary_df["avg_reasoning_tokens"], errors="coerce")
    out["Avg. words / reasoning sent."] = pd.to_numeric(
        summary_df["avg_reasoning_words_per_sentence"],
        errors="coerce",
    )
    out["Avg. continuations / trace"] = pd.to_numeric(
        summary_df["avg_continuations_per_reasoning_trace"],
        errors="coerce",
    )
    out["Avg. continuation tokens"] = pd.to_numeric(summary_df["avg_continuation_tokens"], errors="coerce")
    out["Expanded token multiplier"] = pd.to_numeric(summary_df["expanded_token_multiplier"], errors="coerce")
    out["Recovery rate (%)"] = pd.to_numeric(summary_df["recovered_json_rate"], errors="coerce") * 100.0
    out["Skip rate (%)"] = pd.to_numeric(summary_df["skipped_json_rate"], errors="coerce") * 100.0
    out["File Size (TB)"] = pd.to_numeric(summary_df["file_size_tb"], errors="coerce")
    return out


def summarize_parse_issues(parse_issue_df: pd.DataFrame) -> pd.DataFrame:
    if parse_issue_df.empty:
        return pd.DataFrame(
            columns=[
                "Model",
                "Environment",
                "Issue Kind",
                "Recovery Strategy",
                "Issue Count",
            ]
        )

    out = parse_issue_df.copy()
    for required_column in ["model_display", "env_display", "issue_kind", "recovery_strategy"]:
        if required_column not in out.columns:
            out[required_column] = None
    summary_df = (
        out.groupby(
            ["model_display", "env_display", "issue_kind", "recovery_strategy"],
            dropna=False,
            observed=True,
            sort=False,
        )
        .size()
        .reset_index(name="Issue Count")
    )
    return summary_df.rename(
        columns={
            "model_display": "Model",
            "env_display": "Environment",
            "issue_kind": "Issue Kind",
            "recovery_strategy": "Recovery Strategy",
        }
    )


def format_int(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return f"{int(numeric):,}"


def format_float(value: Any, *, digits: int = 1) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return f"{float(numeric):,.{digits}f}"
