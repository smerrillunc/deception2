from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
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
    'DeepSeek-R1-Distill-Qwen-7B': {
        'hf_repo': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'hf_cache_dir': 'models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B',
        'family': 'qwen2',
    },
    'Qwen-7B': {
        'hf_repo': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'hf_cache_dir': 'models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B',
        'family': 'qwen2',
    },
    'DeepSeek-R1-Distill-Qwen-14B': {
        'hf_repo': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        'hf_cache_dir': 'models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B',
        'family': 'qwen2',
    },
    'Qwen-14B': {
        'hf_repo': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        'hf_cache_dir': 'models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B',
        'family': 'qwen2',
    },
    'DeepSeek-R1-Distill-Llama-8B': {
        'hf_repo': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'hf_cache_dir': 'models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B',
        'family': 'llama',
    },
    'Llama-8B': {
        'hf_repo': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'hf_cache_dir': 'models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B',
        'family': 'llama',
    },
    'gpt-oss-20b': {
        'hf_repo': 'openai/gpt-oss-20b',
        'hf_cache_dir': 'models--openai--gpt-oss-20b',
        'family': 'generic',
    },
    'GPT-OSS-20B': {
        'hf_repo': 'openai/gpt-oss-20b',
        'hf_cache_dir': 'models--openai--gpt-oss-20b',
        'family': 'generic',
    },
}

_TOKENIZER_CACHE: dict[str, Any] = {}
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", flags=re.UNICODE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+", flags=re.UNICODE)


class _RegexTokenizer:
    pad_token_id = 0
    eos_token = None

    def count_tokens(self, text: Any) -> int:
        return len(TOKEN_PATTERN.findall(str(text or '')))


def latest_snapshot_path(root: Path) -> Path | None:
    snapshot_root = root / 'snapshots'
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
    cached_snapshot = latest_snapshot_path(Path(hf_cache_root).expanduser().resolve() / model_cfg['hf_cache_dir'])
    if cached_snapshot is not None:
        return str(cached_snapshot)
    return str(model_cfg['hf_repo'])


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
        tokenizer = _RegexTokenizer()

    if getattr(tokenizer, 'pad_token_id', None) is None and getattr(tokenizer, 'eos_token', None) is not None:
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
    text_value = str(text or '')
    mode = str(token_count_mode).strip().lower()
    if mode in {'regex', 'generic'}:
        return len(TOKEN_PATTERN.findall(text_value))
    if mode == 'hf':
        if model_name is None:
            raise ValueError('model_name is required when token_count_mode="hf".')
        tokenizer = get_tokenizer(model_name, hf_cache_root=hf_cache_root)
        if hasattr(tokenizer, 'count_tokens'):
            return int(tokenizer.count_tokens(text_value))
        token_ids = tokenizer.encode(text_value, add_special_tokens=False)
        return len(token_ids)
    raise ValueError(f'Unsupported token_count_mode={token_count_mode!r}. Choose from {{"regex", "generic", "hf"}}.')


def count_text_words(text: Any) -> int:
    return len(WORD_PATTERN.findall(str(text or '')))


def count_text_sentences(text: Any) -> int:
    cleaned = str(text or '').strip()
    if not cleaned:
        return 0
    parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(cleaned) if part.strip()]
    return max(1, len(parts))


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
        'env_name': env_name,
        'env_display': cj.canonical_env_display(env_name),
        'model_name': model_name,
        'model_display': cj.canonical_model_display(model_name),
        'bundle_dir': str(bundle_dir),
        'file_count': 0,
        'file_size_bytes_total': 0,
        'reasoning_sentence_total': 0,
        'reasoning_token_total': 0,
        'reasoning_word_total': 0,
        'localized_prefix_total': 0,
        'prompt_sentence_total_unique': 0,
        'prompt_token_total_unique': 0,
        'prompt_word_total_unique': 0,
        'prompt_sentence_total_expanded': 0,
        'prompt_token_total_expanded': 0,
        'prompt_word_total_expanded': 0,
        'continuation_total': 0,
        'continuation_sentence_total': 0,
        'continuation_token_total': 0,
        'continuation_word_total': 0,
    }


def _extract_continuation_text(generation: dict[str, Any], *, prefix_text: str) -> str:
    gen_text = generation.get('gen_text')
    if gen_text is not None:
        return str(gen_text)
    full_generation_text = str(generation.get('full_generation_text') or '')
    if prefix_text and full_generation_text.startswith(prefix_text):
        return full_generation_text[len(prefix_text):]
    return full_generation_text


def _localization_paths_for_bundle(bundle_dir: Path, *, max_files: int | None) -> list[Path]:
    paths = sorted((bundle_dir / cj.LOCALIZATION_DIRNAME).glob(cj.LOCALIZATION_GLOB))
    if max_files is not None:
        return paths[: int(max_files)]
    return paths


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
    bundle_path = Path(bundle_dir)
    summary = _empty_bundle_summary(env_name, model_name, bundle_path)

    paths = _localization_paths_for_bundle(bundle_path, max_files=max_files)
    iterator = paths
    if show_progress:
        iterator = tqdm(paths, total=len(paths), desc=progress_desc or f'{cj.canonical_env_display(env_name)} | {cj.canonical_model_display(model_name)}', unit='file')

    for path in iterator:
        payload_bytes = path.read_bytes()
        payload = json.loads(payload_bytes)
        history = payload.get('history') or []
        raw_text = str(payload.get('raw_text') or '')

        summary['file_count'] += 1
        summary['file_size_bytes_total'] += len(payload_bytes)
        summary['reasoning_sentence_total'] += len(history)

        last_prefix_text = ''
        last_prefix_token_count = 0
        last_prefix_word_count = 0

        for sentence_pos, row in enumerate(history, start=1):
            prefix_text = str(row.get('prefix_text') or '')
            if prefix_text.startswith(last_prefix_text):
                delta_text = prefix_text[len(last_prefix_text):]
            else:
                delta_text = ''

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

            summary['localized_prefix_total'] += 1
            summary['prompt_sentence_total_unique'] += prompt_sentence_count
            summary['prompt_token_total_unique'] += prompt_token_count
            summary['prompt_word_total_unique'] += prompt_word_count

            generations = row.get('generations') or []
            generation_count = len(generations)
            summary['continuation_total'] += generation_count
            summary['prompt_sentence_total_expanded'] += prompt_sentence_count * generation_count
            summary['prompt_token_total_expanded'] += prompt_token_count * generation_count
            summary['prompt_word_total_expanded'] += prompt_word_count * generation_count

            for generation in generations:
                continuation_text = _extract_continuation_text(generation, prefix_text=prefix_text)
                summary['continuation_sentence_total'] += count_text_sentences(continuation_text)
                summary['continuation_token_total'] += count_text_tokens(
                    continuation_text,
                    token_count_mode=token_count_mode,
                    model_name=model_name,
                    hf_cache_root=hf_cache_root,
                )
                summary['continuation_word_total'] += count_text_words(continuation_text)

        if history and raw_text == last_prefix_text:
            summary['reasoning_token_total'] += last_prefix_token_count
            summary['reasoning_word_total'] += last_prefix_word_count
        else:
            summary['reasoning_token_total'] += count_text_tokens(
                raw_text,
                token_count_mode=token_count_mode,
                model_name=model_name,
                hf_cache_root=hf_cache_root,
            )
            summary['reasoning_word_total'] += count_text_words(raw_text)

    return summary


def _summarize_localization_bundle_worker(job: tuple[str, str, str, int | None, str, str]) -> dict[str, Any]:
    env_name, model_name, bundle_dir_str, max_files, token_count_mode, hf_cache_root_str = job
    return summarize_localization_bundle(
        bundle_dir_str,
        env_name=env_name,
        model_name=model_name,
        max_files=max_files,
        token_count_mode=token_count_mode,
        hf_cache_root=hf_cache_root_str,
        show_progress=False,
    )


def build_bundle_summary_df(
    root: Path | str = DATASETMAIN_ROOT,
    *,
    max_files_per_bundle: int | None = DEFAULT_MAX_FILES_PER_BUNDLE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    token_count_mode: str = DEFAULT_TOKEN_COUNT_MODE,
    hf_cache_root: Path | str = HF_CACHE_ROOT,
    show_progress: bool = DEFAULT_SHOW_PROGRESS,
    progress_level: str = DEFAULT_PROGRESS_LEVEL,
) -> pd.DataFrame:
    progress_level_key = str(progress_level).strip().lower()
    if progress_level_key not in {'bundle', 'file'}:
        raise ValueError(f'Unsupported progress_level={progress_level!r}. Choose from {{"bundle", "file"}}.')

    jobs = [
        (env_name, model_name, str(bundle_dir), max_files_per_bundle, str(token_count_mode), str(hf_cache_root))
        for env_name, model_name, bundle_dir in cj._bundle_dirs(root)
    ]
    if not jobs:
        return pd.DataFrame()

    if int(num_workers) <= 1:
        if show_progress and progress_level_key == 'file':
            rows = [
                summarize_localization_bundle(
                    bundle_dir_str,
                    env_name=env_name,
                    model_name=model_name,
                    max_files=max_files_per_bundle,
                    token_count_mode=token_count_mode,
                    hf_cache_root=hf_cache_root,
                    show_progress=True,
                    progress_desc=f'{cj.canonical_env_display(env_name)} | {cj.canonical_model_display(model_name)}',
                )
                for env_name, model_name, bundle_dir_str, _, _, _ in jobs
            ]
        else:
            iterator = jobs
            if show_progress:
                iterator = tqdm(jobs, total=len(jobs), desc='Summarizing bundles', unit='bundle')
            rows = [_summarize_localization_bundle_worker(job) for job in iterator]
    else:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            mapped = executor.map(_summarize_localization_bundle_worker, jobs)
            if show_progress:
                mapped = tqdm(mapped, total=len(jobs), desc='Summarizing bundles', unit='bundle')
            rows = list(mapped)

    bundle_df = pd.DataFrame(rows)
    if bundle_df.empty:
        return bundle_df

    bundle_df['_model_sort'] = bundle_df['model_display'].map(cj._model_sort_key)
    bundle_df['_env_sort'] = bundle_df['env_display'].map(cj._env_sort_key)
    bundle_df = bundle_df.sort_values(['_model_sort', '_env_sort']).drop(
        columns=['_model_sort', '_env_sort'],
        errors='ignore',
    )
    return finalize_summary_df(bundle_df.reset_index(drop=True))


def _aggregate_group_rows(group_df: pd.DataFrame) -> dict[str, Any]:
    count_columns = [
        'file_count',
        'file_size_bytes_total',
        'reasoning_sentence_total',
        'reasoning_token_total',
        'reasoning_word_total',
        'localized_prefix_total',
        'prompt_sentence_total_unique',
        'prompt_token_total_unique',
        'prompt_word_total_unique',
        'prompt_sentence_total_expanded',
        'prompt_token_total_expanded',
        'prompt_word_total_expanded',
        'continuation_total',
        'continuation_sentence_total',
        'continuation_token_total',
        'continuation_word_total',
    ]
    row = {
        column: int(pd.to_numeric(group_df[column], errors='coerce').fillna(0).sum())
        for column in count_columns
    }
    return row


def _safe_divide(numerator: Any, denominator: Any) -> float:
    denom = float(denominator)
    if denom == 0.0:
        return float('nan')
    return float(numerator) / denom


def finalize_summary_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    out = summary_df.copy()
    out['avg_reasoning_sentences'] = [
        _safe_divide(num, denom)
        for num, denom in zip(out['reasoning_sentence_total'], out['file_count'], strict=False)
    ]
    out['avg_reasoning_tokens'] = [
        _safe_divide(num, denom)
        for num, denom in zip(out['reasoning_token_total'], out['file_count'], strict=False)
    ]
    out['avg_prefixes_localized'] = [
        _safe_divide(num, denom)
        for num, denom in zip(out['localized_prefix_total'], out['file_count'], strict=False)
    ]
    out['avg_continuation_tokens'] = [
        _safe_divide(num, denom)
        for num, denom in zip(out['continuation_token_total'], out['continuation_total'], strict=False)
    ]
    out['avg_continuations_per_prefix'] = [
        _safe_divide(num, denom)
        for num, denom in zip(out['continuation_total'], out['localized_prefix_total'], strict=False)
    ]
    out['avg_file_size_mb'] = pd.to_numeric(out['file_size_bytes_total'], errors='coerce') / (1024 ** 2)
    out['file_size_tb'] = pd.to_numeric(out['file_size_bytes_total'], errors='coerce') / (10 ** 12)
    out['expanded_dataset_sentence_total'] = (
        pd.to_numeric(out['prompt_sentence_total_expanded'], errors='coerce').fillna(0)
        + pd.to_numeric(out['continuation_sentence_total'], errors='coerce').fillna(0)
    ).astype('int64')
    out['expanded_dataset_token_total'] = (
        pd.to_numeric(out['prompt_token_total_expanded'], errors='coerce').fillna(0)
        + pd.to_numeric(out['continuation_token_total'], errors='coerce').fillna(0)
    ).astype('int64')
    out['expanded_dataset_word_total'] = (
        pd.to_numeric(out['prompt_word_total_expanded'], errors='coerce').fillna(0)
        + pd.to_numeric(out['continuation_word_total'], errors='coerce').fillna(0)
    ).astype('int64')
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
    if 'model_display' in summary_df.columns:
        summary_df['_model_sort'] = summary_df['model_display'].map(cj._model_sort_key)
    if 'env_display' in summary_df.columns:
        summary_df['_env_sort'] = summary_df['env_display'].map(cj._env_sort_key)
    sort_columns = [column for column in ['_model_sort', '_env_sort'] if column in summary_df.columns]
    if sort_columns:
        summary_df = summary_df.sort_values(sort_columns)
    summary_df = summary_df.drop(columns=['_model_sort', '_env_sort'], errors='ignore')
    return finalize_summary_df(summary_df.reset_index(drop=True))


def make_requested_summary_table(
    summary_df: pd.DataFrame,
    *,
    include_model: bool = True,
    include_environment: bool = False,
) -> pd.DataFrame:
    if summary_df.empty:
        columns: list[str] = []
        if include_model:
            columns.append('Model')
        if include_environment:
            columns.append('Environment')
        columns.extend(
            [
                'Avg. reasoning sent.',
                'Avg. reasoning tokens',
                'Avg. prefixes localized',
                'Avg. continuation tokens',
            ]
        )
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame()
    if include_model and 'model_display' in summary_df.columns:
        out['Model'] = summary_df['model_display'].astype(str)
    if include_environment and 'env_display' in summary_df.columns:
        out['Environment'] = summary_df['env_display'].astype(str)
    out['Avg. reasoning sent.'] = pd.to_numeric(summary_df['avg_reasoning_sentences'], errors='coerce')
    out['Avg. reasoning tokens'] = pd.to_numeric(summary_df['avg_reasoning_tokens'], errors='coerce')
    out['Avg. prefixes localized'] = pd.to_numeric(summary_df['avg_prefixes_localized'], errors='coerce')
    out['Avg. continuation tokens'] = pd.to_numeric(summary_df['avg_continuation_tokens'], errors='coerce')
    return out


def make_total_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=['Metric', 'Value'])

    totals = _aggregate_group_rows(summary_df)
    totals = finalize_summary_df(pd.DataFrame([totals])).iloc[0]

    rows = [
        ('Localization files', int(totals['file_count'])),
        ('Localized sentences', int(totals['localized_prefix_total'])),
        ('Continuations', int(totals['continuation_total'])),
        ('Unique reasoning sentences', int(totals['reasoning_sentence_total'])),
        ('Unique reasoning tokens', int(totals['reasoning_token_total'])),
        ('Unique reasoning words', int(totals['reasoning_word_total'])),
        ('Unique localized-prefix tokens', int(totals['prompt_token_total_unique'])),
        ('Unique localized-prefix words', int(totals['prompt_word_total_unique'])),
        ('Expanded prompt sentences', int(totals['prompt_sentence_total_expanded'])),
        ('Expanded prompt tokens', int(totals['prompt_token_total_expanded'])),
        ('Expanded prompt words', int(totals['prompt_word_total_expanded'])),
        ('Continuation sentences', int(totals['continuation_sentence_total'])),
        ('Continuation tokens', int(totals['continuation_token_total'])),
        ('Continuation words', int(totals['continuation_word_total'])),
        ('Total sentences in expanded dataset', int(totals['expanded_dataset_sentence_total'])),
        ('Total tokens in expanded dataset', int(totals['expanded_dataset_token_total'])),
        ('Total words in expanded dataset', int(totals['expanded_dataset_word_total'])),
        ('Localization file size (bytes)', int(totals['file_size_bytes_total'])),
        ('Localization file size (TB)', float(totals['file_size_tb'])),
    ]
    return pd.DataFrame(rows, columns=['Metric', 'Value'])


def format_int(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    if pd.isna(numeric):
        return ''
    return f'{int(numeric):,}'


def format_float(value: Any, *, digits: int = 1) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    if pd.isna(numeric):
        return ''
    return f'{float(numeric):,.{digits}f}'
