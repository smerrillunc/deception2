#!/usr/bin/env python3
import argparse
import logging
import os
import time
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from vllm import LLM

from llm_agent import LLMAgent
from werewolf_environment import WerewolfEnvironment
from utils import append_jsonl, atomic_write_json, get_model_output, set_global_seed


def init_llm(model_name: str, max_tokens: int, seed: int = 0) -> LLM:
    return LLM(model=model_name, max_model_len=max_tokens, seed=seed, gpu_memory_utilization=0.9)


def make_agents(llm, model_name: str, n_players: int, is_reasoning_model: bool):
    reasoning_instruction = "COD" if is_reasoning_model else "COT"
    instruction_format = "default" if is_reasoning_model else "reasoning"

    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
    agents = []
    for i in range(n_players):
        ag = LLMAgent(
            name=names[i % len(names)],
            model=llm,
            model_name=model_name,
            reasoning_instruction=reasoning_instruction,
            instruction_format=instruction_format,
        )
        ag.discuss_format = "reasoning" if instruction_format == "reasoning" else "default"
        ag.vote_format = "reasoning" if instruction_format == "reasoning" else "default"
        agents.append(ag)
    return agents


def state_summary(env):
    actor = env.agents[env.active_player_idx]
    return {
        "phase": env.phase,
        "active_player": actor.name,
        "day": env.day,
        "history_len": len(env.history),
        "game_type": "werewolf",
    }


def deception_from_action(action, env):
    if env.phase != "DISCUSS":
        return None
    target = action.get("Target") if isinstance(action, dict) else None
    claim = str(action.get("Claim", "")).lower() if isinstance(action, dict) else ""
    if not target or claim not in {"werewolf", "villager"}:
        return None
    idx = env._player_index_by_name(target)
    truthful = env.is_truthful_claim(idx, claim)
    return not truthful


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mine deceptive Werewolf examples with an LLM.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--is_reasoning_model", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--num_players", type=int, default=5)
    parser.add_argument("--max_games", type=int, default=1000)
    parser.add_argument("--max_turns", type=int, default=100)
    parser.add_argument("--target_deceptive", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--save_all", action="store_true", default=True)
    parser.add_argument("--save_only_deceptive", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_global_seed(args.seed)

    meta = {
        "model_name": args.model_name,
        "is_reasoning_model": args.is_reasoning_model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
        "num_players": args.num_players,
        "max_games": args.max_games,
        "max_turns": args.max_turns,
        "target_deceptive": args.target_deceptive,
        "save_all": args.save_all,
        "save_only_deceptive": args.save_only_deceptive,
        "seed": args.seed,
        "strategy": "game_flow",
        "game_type": "werewolf",
        "timestamp": time.time(),
    }
    atomic_write_json(os.path.join(args.output_dir, "meta.json"), meta)

    llm = init_llm(args.model_name, args.max_tokens, seed=args.seed)
    output_path = os.path.join(args.output_dir, "deception_samples.jsonl")

    total_states = total_samples = total_deceptive = 0

    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        tokenizer = None

    for game_idx in range(args.max_games):
        agents = make_agents(llm, args.model_name, args.num_players, args.is_reasoning_model)
        env = WerewolfEnvironment(agents, seed=args.seed + game_idx)

        for turn_idx in range(args.max_turns):
            if env.game_over():
                break

            state = env.get_state()
            messages = state["messages"]
            action, seed_used = get_model_output(
                llm=llm,
                messages=messages,
                is_reasoning_model=args.is_reasoning_model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                num_responses=1,
                max_retries=args.max_retries,
            )

            deceptive = deception_from_action(action if isinstance(action, dict) else {}, env)
            prompt_text = None
            if tokenizer is not None:
                try:
                    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    prompt_text = None

            rec = {
                "state_id": total_states,
                "sample_idx": 0,
                "seed": seed_used,
                "deceptive": deceptive,
                "naturally_deceptive": deceptive,
                "action": action,
                "messages": messages,
                "prompt": prompt_text,
                "game_id": game_idx,
                "turn_idx": turn_idx,
                "truth_context": env.truth_context(),
                **state_summary(env),
            }

            save_all = args.save_all and not args.save_only_deceptive
            if save_all or deceptive is True:
                append_jsonl(rec, output_path)

            total_samples += 1
            if deceptive is True:
                total_deceptive += 1
            total_states += 1

            try:
                env.manual_step(action if isinstance(action, dict) else {"Action": "SPEAK", "Claim": "villager"})
            except Exception:
                logging.exception("Error applying discuss action; continuing.")

            if env.phase == "VOTE":
                vote_state = env.get_state()
                vote_action, _ = get_model_output(
                    llm=llm,
                    messages=vote_state["messages"],
                    is_reasoning_model=args.is_reasoning_model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    repetition_penalty=args.repetition_penalty,
                    num_responses=1,
                    max_retries=args.max_retries,
                )
                try:
                    env.manual_step(vote_action if isinstance(vote_action, dict) else {"Action": "VOTE", "Target": env.agents[env.active_player_idx].name})
                except Exception:
                    logging.exception("Error applying vote action; continuing.")

            if args.log_every and total_states % args.log_every == 0:
                logging.info("states=%d samples=%d deceptive=%d", total_states, total_samples, total_deceptive)

            if args.target_deceptive and total_deceptive >= args.target_deceptive:
                break

        if args.target_deceptive and total_deceptive >= args.target_deceptive:
            break

    logging.info("Done. states=%d samples=%d deceptive=%d output=%s", total_states, total_samples, total_deceptive, output_path)


if __name__ == "__main__":
    main()
