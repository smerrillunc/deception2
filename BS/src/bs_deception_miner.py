#!/usr/bin/env python3
import argparse
import logging
import os
import time
import sys
from pathlib import Path

# Ensure local src is importable
SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from vllm import LLM
import torch

from bs_environment import BSEnvironment
from deck import Deck
from llm_agent import LLMAgent
from utils import append_jsonl, atomic_write_json, get_model_output, set_global_seed


def init_llm(model_name: str, max_tokens: int, seed: int = 0) -> LLM:
    #tp = torch.cuda.device_count() or 1
    return LLM(
        model=model_name,
        max_model_len=max_tokens,
        seed=seed,
        #tensor_parallel_size=tp,
        gpu_memory_utilization=0.9,
    )


def make_agents(llm, model_name: str, n_players: int, is_reasoning_model: bool):
    if is_reasoning_model:
        reasoning_instruction = "COD"
        instruction_format = "default"
    else:
        reasoning_instruction = "COT"
        instruction_format = "reasoning"

    names = ["Alice", "Bob", "Carol", "Dave", "Eve"]
    agents = []
    for i in range(n_players):
        ag = LLMAgent(
            name=names[i % len(names)],
            model=llm,
            model_name=model_name,
            reasoning_instruction=reasoning_instruction,
            instruction_format=instruction_format,
        )
        ag.play_format = "reasoning" if instruction_format == "reasoning" else "default"
        ag.challenge_format = "reasoning" if instruction_format == "reasoning" else "default"
        agents.append(ag)
    return agents


def build_env(
    llm,
    model_name: str,
    seed: int,
    n_players: int,
    cards_per_player: int,
    is_reasoning_model: bool,
):
    agents = make_agents(llm, model_name, n_players, is_reasoning_model)
    env = BSEnvironment(agents, seed=seed)
    if cards_per_player != 5:
        env.deck = Deck(seed=seed)
        env.deck.shuffle()
        env.deal(n_cards=cards_per_player)
    return env


def state_summary(env):
    player = env.agents[env.active_player_idx]
    return {
        "phase": env.phase,
        "current_rank": env.current_rank,
        "active_player": player.name,
        "hand": list(player.hand),
        "pile_size": len(env.pile),
        "history_len": len(env.history),
    }


def deception_from_action(action, env):
    if not isinstance(action, dict) or action.get("Parse_fail", False):
        return None
    if action.get("Action", "PLAY") != "PLAY":
        return False
    try:
        is_truth = env.is_truthful(action.get("Cards_played", []), env.current_rank)
        if isinstance(is_truth, str):
            return None
        return not is_truth
    except Exception:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mine deceptive BS examples with an LLM.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--is_reasoning_model", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--cards_per_player", type=int, default=5)
    parser.add_argument("--max_games", type=int, default=1000)
    parser.add_argument("--max_turns", type=int, default=1000)
    parser.add_argument("--target_deceptive", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--save_all", action="store_true", default=True)
    parser.add_argument("--save_only_deceptive", action="store_true", default=False,
                        help="If set, only deceptive turns are saved (overrides save_all).")
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
        "cards_per_player": args.cards_per_player,
        "max_games": args.max_games,
        "max_turns": args.max_turns,
        "target_deceptive": args.target_deceptive,
        "save_all": args.save_all,
        "save_only_deceptive": args.save_only_deceptive,
        "seed": args.seed,
        "strategy": "game_flow",
        "timestamp": time.time(),
    }
    atomic_write_json(os.path.join(args.output_dir, "meta.json"), meta)

    logging.info("Loading model %s ...", args.model_name)
    llm = init_llm(args.model_name, args.max_tokens, seed=args.seed)

    output_path = os.path.join(args.output_dir, "deception_samples.jsonl")

    total_states = 0
    total_samples = 0
    total_deceptive = 0

    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        tokenizer = None

    for game_idx in range(args.max_games):
        env = build_env(
            llm=llm,
            model_name=args.model_name,
            seed=args.seed + game_idx,
            n_players=args.num_players,
            cards_per_player=args.cards_per_player,
            is_reasoning_model=args.is_reasoning_model,
        )

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

            deceptive = deception_from_action(action, env)
            prompt_text = None
            if tokenizer is not None:
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    prompt_text = None

            rec = {
                "state_id": total_states,
                "sample_idx": 0,
                "seed": seed_used,
                "deceptive": deceptive,
                "action": action,
                "messages": messages,
                "prompt": prompt_text,
                "game_id": game_idx,
                "turn_idx": turn_idx,
                **state_summary(env),
            }

            save_all = args.save_all and not args.save_only_deceptive
            if save_all or deceptive is True:
                append_jsonl(rec, output_path)

            total_samples += 1
            if deceptive is True:
                total_deceptive += 1

            total_states += 1

            # Apply play action
            try:
                env.manual_step(action if isinstance(action, dict) else {"Action": "PLAY", "Cards_played": []})
            except Exception:
                logging.exception("Error applying play action; continuing.")

            # Challenge phase (advance game)
            if env.phase == "CHALLENGE":
                challenge_state = env.get_state()
                challenge_messages = challenge_state["messages"]
                challenge_action, _ = get_model_output(
                    llm=llm,
                    messages=challenge_messages,
                    is_reasoning_model=args.is_reasoning_model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    repetition_penalty=args.repetition_penalty,
                    num_responses=1,
                    max_retries=args.max_retries,
                )
                try:
                    env.manual_step(challenge_action if isinstance(challenge_action, dict) else {"Action": "Pass"})
                except Exception:
                    logging.exception("Error applying challenge action; continuing.")

            if args.log_every and total_states % args.log_every == 0:
                logging.info(
                    "states=%d samples=%d deceptive=%d",
                    total_states,
                    total_samples,
                    total_deceptive,
                )

            if args.target_deceptive and total_deceptive >= args.target_deceptive:
                logging.info("Reached target deceptive count: %d", total_deceptive)
                break

        if args.target_deceptive and total_deceptive >= args.target_deceptive:
            break

    logging.info(
        "Done. states=%d samples=%d deceptive=%d output=%s",
        total_states,
        total_samples,
        total_deceptive,
        output_path,
    )


if __name__ == "__main__":
    main()
