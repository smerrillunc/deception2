import random, numpy as np, torch
import json, os, time
from pathlib import Path
import re
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.mistral_reasoning_parser import MistralReasoningParser
import json
import os
import tempfile
import random
import re
import logging
from vllm import SamplingParams

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(obj, f, default=_json_default, indent=2)

def append_jsonl(obj, path):
    """
    Append a single JSON object as a line in a JSONL file.
    Uses a lock file so multi-process writes block instead of clobbering.
    """
    ensure_dir(os.path.dirname(path))
    line = json.dumps(obj, default=_json_default) + "\n"

    lock_path = path + ".lock"
    lock_fd = None
    start = time.time()
    stale_after = 120.0  # seconds
    poll = 0.05

    while True:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > stale_after:
                    os.remove(lock_path)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(poll)

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except Exception:
                pass
            try:
                os.remove(lock_path)
            except Exception:
                pass

def _json_default(o):
    try:
        return o.__dict__
    except Exception:
        return str(o)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def load_model_and_tokenizer(model_name, max_seq_length=10000, device_map="auto"):
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fix_tokenizer=True,
        #offload_folder="/playpen-ssd/smerrill/offload", 
    ) 
    if 'llama' in model_name.lower():
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1",
        )
    return model, tokenizer



def extract_reasoning_and_text2(output, llm):
    """
    Extracts reasoning and remaining text from vLLM output using GPT-OSS or Mistral reasoning parser.

    Args:
        output: vLLM output object (e.g., output1[0])
        llm: vLLM tokenizer used for decoding

    Returns:
        dict: {
            "reasoning": str,  # cleaned reasoning text (all THINK blocks joined)
            "text": str        # remaining text
        }
    """

    tokenizer = llm.get_tokenizer()

    try:
        # Create the reasoning parser
        name = tokenizer.name_or_path.lower()
        if 'deepseek' in name:
            parser = DeepSeekR1ReasoningParser(tokenizer)
        elif 'qwen' in name or 'qwq' in name:
            parser = Qwen3ReasoningParser(tokenizer)
        elif 'gpt-oss' in name:
            parser = GptOssReasoningParser(tokenizer)
        elif 'phi' in name:
            # Match <think>...</think> including newlines
            think_match = re.search(r"<think>.*?</think>", output.outputs[0].text, flags=re.DOTALL)

            think = think_match.group(0) if think_match else ""
            
            # Remove the think block from the text
            remaining = re.sub(r"<think>.*?</think>", "", output.outputs[0].text, flags=re.DOTALL).strip()

            return {
                "reasoning": think,
                "text": remaining
            }

        else:
            raise ValueError("Unsupported model for reasoning extraction")
    except Exception:
        parser = MistralReasoningParser(tokenizer)

    # If parser is Mistral, use regex to extract [THINK] blocks
    if isinstance(parser, MistralReasoningParser):
        # Decode all tokens first
        full_text = tokenizer.decode(output.outputs[0].token_ids)

        # Extract all [THINK] blocks
        think_blocks = re.findall(r'\[THINK\](.*?)\[/THINK\]', full_text, re.DOTALL)
        reasoning_text = "\n".join([block.strip() for block in think_blocks])

        # Remove [THINK] blocks to get remaining text
        remaining_text = re.sub(r'\[THINK\].*?\[/THINK\]', '', full_text, flags=re.DOTALL).strip()

        return {
            "reasoning": reasoning_text,
            "text": remaining_text
        }

    # Otherwise, use token-based parser
    token_ids = output.outputs[0].token_ids

    # -----------------------------
    # 1) Find the position of the reasoning end marker
    # -----------------------------
    reasoning_end_idx = None
    for i in range(len(token_ids)):
        if parser.is_reasoning_end(token_ids[:i+1]):
            reasoning_end_idx = i + 1  # include this token
            break

    if reasoning_end_idx is None:
        reasoning_end_idx = len(token_ids)  # fallback: all tokens

    # -----------------------------
    # 2) Decode reasoning vs remaining text
    # -----------------------------
    reasoning_text = tokenizer.decode(token_ids[:reasoning_end_idx])
    remaining_text = tokenizer.decode(token_ids[reasoning_end_idx:])

    # -----------------------------
    # 3) Clean reasoning and remaining text
    # -----------------------------
    clean_reasoning = re.sub(r"<\|.*?\|>", "", reasoning_text).strip()
    clean_reasoning = re.sub(r'^analysis', '', clean_reasoning, flags=re.IGNORECASE).strip()
    clean_reasoning = re.sub(r'assistantfinal$', '', clean_reasoning, flags=re.IGNORECASE).strip()

    clean_text = re.sub(r"<\|.*?\|>", "", remaining_text).strip()

    # -----------------------------
    # 4) Return as JSON
    # -----------------------------
    return {
        "reasoning": clean_reasoning,
        "text": clean_text
    }

# -------------------------
# Utilities (based on your notebook)
# -------------------------
def extract_json_with_reasoning(text: str) -> dict:
    """
    Extract the first JSON object in the text, clean it, and attach remaining text as `reasoning`.
    Raises ValueError if no JSON object can be found or parsed.
    """
    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        raise ValueError("No JSON object found in text.")

    raw_json = match.group(0)

    # Remove comments
    cleaned_json = re.sub(r"#.*?$", "", raw_json, flags=re.MULTILINE)
    cleaned_json = re.sub(r"//.*?$", "", cleaned_json, flags=re.MULTILINE)

    # Remove trailing commas
    cleaned_json = re.sub(r",\s*([\]}])", r"\1", cleaned_json)

    try:
        data = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON after cleaning:\n{cleaned_json}") from e

    reasoning = (text[:match.start()] + text[match.end():]).strip()
    reasoning = re.sub(r"(JSON\s*Response\s*:?)$", "", reasoning, flags=re.IGNORECASE | re.MULTILINE).strip()
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)

    data["reasoning"] = reasoning
    return data

def get_reasoning_model_output(text: str) -> dict:
    """
    For reasoning models that include a <think>...</think> block, extract it and the output JSON.
    """
    think_match = re.search(r".*?</think>", text, flags=re.DOTALL)
    think = think_match.group(0) if think_match else ""
    remaining = re.sub(r".*?</think>", "", text, flags=re.DOTALL).strip()
    output_json = extract_json_with_reasoning(remaining)
    output_json.update({'reasoning': think})
    return output_json

def atomic_write_json(path: str, data):
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_", dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# -------------------------
# LLM interaction helper
# -------------------------
def get_model_output(llm, messages, is_reasoning_model: bool,
                     temperature: float, top_p: float, max_tokens: int,
                     repetition_penalty: float, num_responses: int, max_retries: int):
    """
    Produce parsed output(s) from an LLM. Returns either a dict (single) or list-of-dicts (multiple),
    plus the internal retry seed index used.
    Each parsed dict will contain either the parsed JSON fields or a 'Parse_fail' key with raw_text and error.
    """
    for attempt in range(max_retries):
        try:
            # replicate messages for each response if num_responses>1
            msg_list = messages if num_responses == 1 else [messages] * num_responses
            params_list = [
                SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty,
                    seed=j + attempt * (num_responses or 1),
                ) for j in range(num_responses)
            ]
            outputs = llm.chat(msg_list, sampling_params=params_list)
            parsed_results = []

            # outputs is assumed to be a sequence where outputs[i].outputs[0].text is the text
            for out in outputs:
                try:
                    raw_text = out.outputs[0].text
                except Exception as e:
                    # fallback: try to stringify the whole object
                    raw_text = str(out)
                try:
                    if is_reasoning_model:
                        parsed = get_reasoning_model_output(raw_text)
                    else:
                        parsed = extract_json_with_reasoning(raw_text)
                    parsed['_raw_text'] = raw_text
                    parsed_results.append(parsed)
                except Exception as e:
                    parsed_results.append({
                        "Parse_fail": True,
                        "error": str(e),
                        "_raw_text": raw_text,
                    })

            if num_responses == 1:
                return parsed_results[0], attempt
            else:
                return parsed_results, attempt
        except Exception as e:
            logging.exception("LLM call failed on attempt %d: %s", attempt, e)
            # try again until max_retries
            continue

    # If we exhausted retries, return a failure marker
    fail_msg = {"Parse_fail": True, "error": "Exceeded max_retries without successful LLM call"}
    return ([fail_msg] * num_responses) if num_responses > 1 else (fail_msg, max_retries - 1)
