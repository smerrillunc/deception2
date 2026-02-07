from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
SENTENCE_SPAN_RE = re.compile(r"[^.!?]+[.!?]?\s*")


def split_sentences(text: Any) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned = " ".join(text.strip().split())
    return [s for s in SENTENCE_SPLIT_RE.split(cleaned) if s]


def split_sentence_spans(text: Any) -> List[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return []
    spans = []
    for match in SENTENCE_SPAN_RE.finditer(text):
        span_text = match.group(0)
        if not span_text.strip():
            continue
        start, end = match.span()
        while end > start and text[end - 1].isspace():
            end -= 1
        spans.append({
            "start": start,
            "end": end,
            "text": text[start:end],
        })
    return spans


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


@dataclass
class TaxonomyLabel:
    id: str
    name: str
    description: str
    examples: Optional[List[str]] = None


@dataclass
class SentenceTaxonomy:
    name: str
    version: str
    labels: List[TaxonomyLabel]
    instructions: Optional[str] = None

    @staticmethod
    def from_json(path: str | Path) -> "SentenceTaxonomy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        labels = [
            TaxonomyLabel(
                id=item["id"],
                name=item["name"],
                description=item["description"],
                examples=item.get("examples"),
            )
            for item in data["labels"]
        ]
        return SentenceTaxonomy(
            name=data.get("name", "taxonomy"),
            version=data.get("version", "v1"),
            labels=labels,
            instructions=data.get("instructions"),
        )

    def to_prompt_block(self) -> str:
        lines = []
        lines.append(f"Taxonomy: {self.name} ({self.version})")
        for idx, label in enumerate(self.labels, 1):
            lines.append(f"{idx}. {label.name}: {label.description}")
            if label.examples:
                lines.append(f"Examples: {' | '.join(label.examples)}")
        if self.instructions:
            lines.append(f"Instructions: {self.instructions}")
        return "\n".join(lines)


def make_sentence_id(example_id: str, sentence_idx: int) -> str:
    return f"{example_id}/sent_{sentence_idx:04d}"


def build_sentence_records(
    examples: Iterable[Dict[str, Any]],
    *,
    text_field: str = "action_reasoning",
    example_id_field: str = "record_id",
    include_example_fields: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    include_example_fields = include_example_fields or []

    for ex in examples:
        example_id = ex.get(example_id_field) or ex.get("example_id")
        if not example_id:
            continue
        text = ex.get(text_field)
        sentences = split_sentence_spans(text)
        for idx, sent in enumerate(sentences):
            rec = {
                "sentence_id": make_sentence_id(example_id, idx),
                "example_id": example_id,
                "source_field": text_field,
                "sentence_idx": idx,
                "sentence_text": sent["text"],
                "start": sent["start"],
                "end": sent["end"],
            }
            for field in include_example_fields:
                if field in ex:
                    rec[field] = ex[field]
            yield rec


def build_sentence_prompt(sentence: str, taxonomy: SentenceTaxonomy) -> str:
    return (
        "Label the sentence with exactly one taxonomy category.\n"
        "Return JSON only: {\"label_id\": \"...\", \"label_name\": \"...\", \"confidence\": 0-1}\n\n"
        f"{taxonomy.to_prompt_block()}\n\n"
        f"Sentence:\n{sentence}"
    )


TaggerFn = Callable[[str, SentenceTaxonomy], Dict[str, Any]]


def tag_sentences(
    sentences: Iterable[Dict[str, Any]],
    taxonomy: SentenceTaxonomy,
    *,
    tagger: TaggerFn,
    sentence_text_field: str = "sentence_text",
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
    cache_key_field: str = "sentence_id",
) -> Iterator[Dict[str, Any]]:
    for rec in sentences:
        sentence = rec.get(sentence_text_field, "")
        cache_key = rec.get(cache_key_field)
        if cache is not None and cache_key in cache:
            tag = cache[cache_key]
        else:
            tag = tagger(sentence, taxonomy)
            if cache is not None and cache_key is not None:
                cache[cache_key] = tag
        yield {
            "sentence_id": rec["sentence_id"],
            "example_id": rec["example_id"],
            **tag,
        }


def build_localization_input(
    *,
    examples_path: str | Path,
    sentences_path: str | Path,
    tags_path: str | Path,
    out_path: str | Path,
    build_record_fn: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> None:
    """
    Join examples + sentences + tags and build localization input records.
    build_record_fn(example, sentence, tag) -> dict
    """
    examples = {ex["example_id"]: ex for ex in read_jsonl(examples_path)}
    sentences = {s["sentence_id"]: s for s in read_jsonl(sentences_path)}

    def _records():
        for tag in read_jsonl(tags_path):
            sentence = sentences.get(tag["sentence_id"])
            if not sentence:
                continue
            example = examples.get(sentence["example_id"])
            if not example:
                continue
            yield build_record_fn(example, sentence, tag)

    write_jsonl(_records(), out_path)
