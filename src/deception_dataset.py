from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path, strict: bool) -> Iterator[tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError:
                if strict:
                    raise


def _extract_run_info(samples_path: Path, root_dir: Path) -> Dict[str, Optional[str]]:
    rel = samples_path.relative_to(root_dir)
    parts = rel.parts

    run_id = str(Path(*parts[:-1])) if len(parts) > 1 else samples_path.parent.name
    gpu = next((p for p in parts if p.startswith("gpu_")), None)
    run_date = next((p for p in parts if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p)), None)

    return {
        "run_id": run_id,
        "run_date": run_date,
        "gpu": gpu,
        "source_path": str(samples_path),
    }


def iter_deception_records(
    root_dir: str | Path,
    *,
    include_messages: bool = False,
    include_action: bool = False,
    flatten_action: bool = True,
    include_meta: bool = True,
    strict_json: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Iterate over all deception_samples.jsonl files under root_dir, yielding
    normalized records with run metadata attached.
    """
    root = Path(root_dir)
    samples_files = sorted(root.rglob("deception_samples.jsonl"))

    for samples_path in samples_files:
        run_info = _extract_run_info(samples_path, root)
        meta_path = samples_path.with_name("meta.json")
        meta = _load_json(meta_path) if include_meta and meta_path.exists() else {}

        for line_num, rec in _iter_jsonl(samples_path, strict_json):
            out: Dict[str, Any] = {}

            if include_meta and meta:
                for key, value in meta.items():
                    out[f"meta_{key}"] = value

            out.update(run_info)

            for key in (
                "state_id",
                "sample_idx",
                "seed",
                "deceptive",
                "game_id",
                "turn_idx",
                "phase",
                "current_rank",
                "active_player",
                "hand",
                "pile_size",
                "history_len",
                "prompt",
                "game_type",
                "truth_context",
            ):
                if key in rec:
                    out[key] = rec[key]

            if include_messages and "messages" in rec:
                out["messages"] = rec["messages"]

            action = rec.get("action") if isinstance(rec.get("action"), dict) else None
            if include_action and action is not None:
                out["action"] = action

            if flatten_action and action is not None:
                out["action_type"] = action.get("Action")
                out["cards_played"] = action.get("Cards_played")
                out["action_parse_fail"] = action.get("Parse_fail")
                if "reasoning" in action:
                    out["action_reasoning"] = action.get("reasoning")
                if "_raw_text" in action:
                    out["action_raw_text"] = action.get("_raw_text")
                if isinstance(action.get("Cards_played"), list):
                    out["cards_played_len"] = len(action["Cards_played"])

            state_id = out.get("state_id")
            sample_idx = out.get("sample_idx")
            if state_id is not None and sample_idx is not None:
                out["record_id"] = f"{run_info['run_id']}/state_{state_id}/sample_{sample_idx}"
            else:
                out["record_id"] = f"{run_info['run_id']}/line_{line_num}"

            yield out


def write_jsonl(records: List[Dict[str, Any]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def count_sentences(text: Any) -> int:
    if not isinstance(text, str):
        return 0
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return 0
    parts = [p for p in _SENTENCE_SPLIT_RE.split(cleaned) if p]
    return len(parts)


def _iter_text_fields(row: Dict[str, Any], fields: Sequence[str]) -> Iterator[str]:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            yield value
