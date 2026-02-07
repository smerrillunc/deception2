#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from sentence_pipeline import SentenceTaxonomy, build_sentence_prompt, read_jsonl, write_jsonl
from utils import extract_json_with_reasoning


def main(argv=None):
    parser = argparse.ArgumentParser(description="LLM tagger for sentence taxonomy.")
    parser.add_argument("--sentences_path", type=str, required=True)
    parser.add_argument("--taxonomy_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4-mini")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    taxonomy = SentenceTaxonomy.from_json(args.taxonomy_path)

    client = OpenAI(api_key=args.api_key)

    def _extract_output_text(response) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text
        # Fallback to manual extraction
        output = getattr(response, "output", []) or []
        parts = []
        for item in output:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    parts.append(getattr(content, "text", ""))
        return "\n".join(p for p in parts if p).strip()

    def _call_openai(prompt: str) -> Dict[str, Any]:
        last_error: Optional[str] = None
        for attempt in range(args.max_retries):
            try:
                response = client.responses.create(
                    model=args.model_name,
                    instructions="You are a sentence classifier. Output JSON only.",
                    input=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_output_tokens=args.max_tokens,
                )
                raw_text = _extract_output_text(response)
                parsed = extract_json_with_reasoning(raw_text)
                parsed["_raw_text"] = raw_text
                return parsed
            except Exception as exc:
                last_error = str(exc)
                time.sleep(min(2 ** attempt, 8))
        return {
            "Parse_fail": True,
            "error": last_error or "Unknown error",
            "_raw_text": "",
        }

    def _records():
        count = 0
        for rec in read_jsonl(args.sentences_path):
            if args.limit and count >= args.limit:
                break
            count += 1
            sentence = rec.get("sentence_text", "")
            prompt = build_sentence_prompt(sentence, taxonomy)
            parsed = _call_openai(prompt)

            out = {
                "sentence_id": rec["sentence_id"],
                "example_id": rec["example_id"],
                "model_name": args.model_name,
                "taxonomy_name": taxonomy.name,
                "taxonomy_version": taxonomy.version,
                "timestamp": time.time(),
            }

            if isinstance(parsed, dict) and parsed.get("Parse_fail"):
                out.update({
                    "parse_fail": True,
                    "error": parsed.get("error"),
                    "raw_text": parsed.get("_raw_text"),
                })
            else:
                out.update({
                    "label_id": parsed.get("label_id"),
                    "label_name": parsed.get("label_name"),
                    "confidence": parsed.get("confidence"),
                    "raw_text": parsed.get("_raw_text"),
                })

            yield out

    write_jsonl(_records(), args.out_path)
    print(f"Wrote tags: {args.out_path}")


if __name__ == "__main__":
    main()
