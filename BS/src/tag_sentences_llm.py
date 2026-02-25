#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, Optional

from openai import OpenAI

from sentence_pipeline import SentenceTaxonomy, build_sentence_prompt, read_jsonl, write_jsonl

try:
    from utils import extract_json_with_reasoning
except Exception:
    def extract_json_with_reasoning(text: str) -> dict:
        """
        Lightweight fallback parser so this script doesn't hard-require vLLM.
        """
        match = re.search(r"\{[\s\S]*?\}", text)
        if not match:
            raise ValueError("No JSON object found in text.")

        raw_json = match.group(0)
        cleaned_json = re.sub(r"#.*?$", "", raw_json, flags=re.MULTILINE)
        cleaned_json = re.sub(r"//.*?$", "", cleaned_json, flags=re.MULTILINE)
        cleaned_json = re.sub(r",\s*([\]}])", r"\1", cleaned_json)

        data = json.loads(cleaned_json)
        reasoning = (text[:match.start()] + text[match.end():]).strip()
        data["reasoning"] = reasoning
        return data

SYSTEM_PROMPT = "You are a sentence classifier. Output JSON only."


def _extract_responses_output_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    output = getattr(response, "output", []) or []
    parts = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                parts.append(getattr(content, "text", ""))
    return "\n".join(p for p in parts if p).strip()


def _extract_chat_output_text(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in {"text", "output_text"}:
                    parts.append(part.get("text", ""))
            elif getattr(part, "type", None) in {"text", "output_text"}:
                parts.append(getattr(part, "text", ""))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description="LLM tagger for sentence taxonomy.")
    parser.add_argument("--sentences_path", type=str, required=True)
    parser.add_argument("--taxonomy_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4-mini")
    parser.add_argument("--backend", choices=["openai", "vllm"], default="openai")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_mode", choices=["auto", "responses", "chat"], default="auto")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_dtype", type=str, default="auto")
    parser.add_argument("--vllm_batch_size", type=int, default=32)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args(argv)

    taxonomy = SentenceTaxonomy.from_json(args.taxonomy_path)
    client: Optional[OpenAI] = None
    llm = None
    SamplingParams = None

    if args.backend == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if args.base_url and not api_key:
            api_key = "EMPTY"
        if not api_key:
            raise ValueError(
                "Missing API key. Set --api_key or OPENAI_API_KEY. "
                "For local OpenAI-compatible servers, pass --base_url and omit key if your server accepts any key."
            )
        client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
        print(f"Using backend=openai model={args.model_name} api_mode={args.api_mode} base_url={args.base_url or 'default'}")
    else:
        try:
            from vllm import LLM, SamplingParams as _SamplingParams  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "backend=vllm requested, but `vllm` is not installed in this environment."
            ) from exc

        llm_kwargs = {
            "model": args.model_name,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.vllm_dtype,
            "trust_remote_code": args.trust_remote_code,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
        if args.max_model_len > 0:
            llm_kwargs["max_model_len"] = args.max_model_len
        llm = LLM(**llm_kwargs)
        SamplingParams = _SamplingParams
        print(
            "Using backend=vllm "
            f"model={args.model_name} tensor_parallel_size={args.tensor_parallel_size} "
            f"dtype={args.vllm_dtype}"
        )

    def _call_openai_raw(prompt: str) -> str:
        if client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        responses_error: Optional[str] = None
        if args.api_mode in {"responses", "auto"}:
            try:
                response = client.responses.create(
                    model=args.model_name,
                    instructions=SYSTEM_PROMPT,
                    input=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_output_tokens=args.max_tokens,
                )
                raw_text = _extract_responses_output_text(response)
                if raw_text:
                    return raw_text
                responses_error = "responses API returned empty text"
            except Exception as exc:
                if args.api_mode == "responses":
                    raise
                responses_error = f"responses API failed: {exc}"

        if args.api_mode == "responses":
            raise RuntimeError(responses_error or "responses API returned no text.")

        if args.api_mode in {"chat", "auto"}:
            chat_error: Optional[str] = None
            try:
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                raw_text = _extract_chat_output_text(response)
                if raw_text:
                    return raw_text
                chat_error = "chat.completions API returned empty text"
            except Exception as exc:
                if args.api_mode == "chat":
                    raise
                chat_error = f"chat.completions API failed: {exc}"

            errors = [e for e in [responses_error, chat_error] if e]
            raise RuntimeError("; ".join(errors) if errors else "No OpenAI API mode succeeded.")

        raise ValueError(f"Unsupported api_mode: {args.api_mode}")

    def _call_vllm_raw(prompt: str) -> str:
        if llm is None or SamplingParams is None:
            raise RuntimeError("vLLM backend is not initialized.")
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        outputs = llm.chat(messages=[messages], sampling_params=[sampling])
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no outputs.")
        raw_text = outputs[0].outputs[0].text
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise RuntimeError("vLLM returned empty text.")
        return raw_text.strip()

    def _parse_raw_text(raw_text: str) -> Dict[str, Any]:
        parsed = extract_json_with_reasoning(raw_text)
        parsed["_raw_text"] = raw_text
        print(parsed)
        return parsed

    def _call_vllm_batch(prompts: list[str]) -> list[Dict[str, Any]]:
        if llm is None or SamplingParams is None:
            raise RuntimeError("vLLM backend is not initialized.")
        if not prompts:
            return []

        results: list[Optional[Dict[str, Any]]] = [None] * len(prompts)
        pending = list(range(len(prompts)))
        last_errors: Dict[int, str] = {}
        last_raw_text: Dict[int, str] = {}

        for attempt in range(args.max_retries):
            if not pending:
                break

            try:
                msg_list = [
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompts[idx]},
                    ]
                    for idx in pending
                ]
                params_list = [
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        seed=attempt * 100000 + local_idx + 1,
                    )
                    for local_idx in range(len(pending))
                ]
                outputs = llm.chat(messages=msg_list, sampling_params=params_list)
            except Exception as exc:
                err = str(exc)
                print(err)
                for idx in pending:
                    last_errors[idx] = err
                time.sleep(min(2 ** attempt, 8))
                continue

            next_pending: list[int] = []
            for local_idx, idx in enumerate(pending):
                raw_text = ""
                try:
                    out = outputs[local_idx]
                    raw_text = out.outputs[0].text
                    if not isinstance(raw_text, str):
                        raw_text = str(raw_text)
                    raw_text = raw_text.strip()
                    if not raw_text:
                        raise ValueError("vLLM returned empty text.")
                    parsed = _parse_raw_text(raw_text)
                    results[idx] = parsed
                except Exception as exc:
                    err = str(exc)
                    print(err)
                    if raw_text:
                        last_raw_text[idx] = raw_text
                    last_errors[idx] = err
                    next_pending.append(idx)

            pending = next_pending
            if pending:
                time.sleep(min(2 ** attempt, 8))

        for idx, parsed in enumerate(results):
            if parsed is not None:
                continue
            results[idx] = {
                "Parse_fail": True,
                "error": last_errors.get(idx, "Unknown error"),
                "_raw_text": last_raw_text.get(idx, ""),
            }
        return [r for r in results if r is not None]

    def _call_model(prompt: str) -> Dict[str, Any]:
        last_error: Optional[str] = None
        for attempt in range(args.max_retries):
            try:
                if args.backend == "openai":
                    raw_text = _call_openai_raw(prompt)
                else:
                    raw_text = _call_vllm_raw(prompt)
                return _parse_raw_text(raw_text)
            except Exception as exc:
                print(exc)
                last_error = str(exc)
                time.sleep(min(2 ** attempt, 8))
        return {
            "Parse_fail": True,
            "error": last_error or "Unknown error",
            "_raw_text": "",
        }

    def _to_out_record(rec: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
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
        return out

    def _records():
        count = 0
        if args.backend == "vllm":
            batch_size = max(1, args.vllm_batch_size)
            batch: list[Dict[str, Any]] = []

            for rec in read_jsonl(args.sentences_path):
                if args.limit and count >= args.limit:
                    break
                count += 1
                batch.append(rec)

                if len(batch) < batch_size:
                    continue

                prompts = [build_sentence_prompt(r.get("sentence_text", ""), taxonomy) for r in batch]
                parsed_batch = _call_vllm_batch(prompts)
                for rec_item, parsed in zip(batch, parsed_batch):
                    print(parsed)
                    yield _to_out_record(rec_item, parsed)
                batch = []

            if batch:
                prompts = [build_sentence_prompt(r.get("sentence_text", ""), taxonomy) for r in batch]
                parsed_batch = _call_vllm_batch(prompts)
                for rec_item, parsed in zip(batch, parsed_batch):
                    print(parsed)
                    yield _to_out_record(rec_item, parsed)
            return

        for rec in read_jsonl(args.sentences_path):
            if args.limit and count >= args.limit:
                break
            count += 1
            sentence = rec.get("sentence_text", "")
            prompt = build_sentence_prompt(sentence, taxonomy)
            parsed = _call_model(prompt)
            yield _to_out_record(rec, parsed)

    write_jsonl(_records(), args.out_path)
    print(f"Wrote tags: {args.out_path}")


if __name__ == "__main__":
    main()
