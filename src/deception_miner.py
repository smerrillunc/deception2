#!/usr/bin/env python3
import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parent
BS_SRC_ROOT = REPO_ROOT / "BS" / "src"
GRIDWORLD_SRC_ROOT = REPO_ROOT / "Gridworld" / "src"

for path in (SRC_ROOT, BS_SRC_ROOT, GRIDWORLD_SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from vllm import LLM

from llm_agent import LLMAgent
from utils import append_jsonl, atomic_write_json, get_model_output, set_global_seed


def init_llm(model_name: str, max_tokens: int, seed: int = 0) -> LLM:
    return LLM(
        model=model_name,
        max_model_len=max_tokens,
        seed=seed,
        gpu_memory_utilization=0.9,
    )


def sample_phase(game: str) -> str:
    return {
        "bs": "PLAY",
        "gridworld": "GUIDE",
    }[game]


def make_agents(game: str, llm, model_name: str, n_players: int, is_reasoning_model: bool):
    if is_reasoning_model:
        reasoning_instruction = "COD"
        instruction_format = "default"
    else:
        reasoning_instruction = "COT"
        instruction_format = "reasoning"

    if game == "gridworld":
        n_players = 2

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

        if game == "bs":
            ag.play_format = "reasoning" if instruction_format == "reasoning" else "default"
            ag.challenge_format = "reasoning" if instruction_format == "reasoning" else "default"
        elif game == "gridworld":
            ag.guide_format = "reasoning" if instruction_format == "reasoning" else "default"
            ag.challenge_format = "reasoning" if instruction_format == "reasoning" else "default"
        else:
            raise ValueError(f"Unsupported game: {game}")

        agents.append(ag)

    return agents


def build_env(game: str, llm, model_name: str, seed: int, args):
    if game == "bs":
        from bs_environment import BSEnvironment
        from deck import Deck

        agents = make_agents(game, llm, model_name, args.num_players, args.is_reasoning_model)
        env = BSEnvironment(agents, seed=seed)
        if args.cards_per_player != 5:
            env.deck = Deck(seed=seed)
            env.deck.shuffle()
            env.deal(n_cards=args.cards_per_player)
        return env

    if game == "gridworld":
        from gridworld_environment import GridSpec, GridWorldEnvironment

        agents = make_agents(game, llm, model_name, 2, args.is_reasoning_model)
        grid_spec = GridSpec(
            width=args.grid_width,
            height=args.grid_height,
            wall_prob=args.wall_prob,
            max_tries=args.max_tries,
        )
        return GridWorldEnvironment(
            agents=agents,
            seed=seed,
            grid_spec=grid_spec,
            max_steps=args.max_steps,
            view_radius=args.view_radius,
            history_window=args.history_window,
            auto_move_explorer=args.auto_move_explorer,
        )

    raise ValueError(f"Unsupported game: {game}")


def state_summary(game: str, env) -> Dict[str, Any]:
    if game == "bs":
        player = env.agents[env.active_player_idx]
        return {
            "phase": env.phase,
            "active_player": player.name,
            "current_rank": env.current_rank,
            "pile_size": len(env.pile),
            "history_len": len(env.history),
            "game_type": "bs",
        }

    active = None if env.active_player_idx is None else env.agents[env.active_player_idx].name
    return {
        "phase": env.phase,
        "active_player": active,
        "pos": env.pos,
        "goal": env.goal,
        "t": env.t,
        "max_steps": env.max_steps,
        "history_len": len(env.history),
        "game_type": "gridworld",
    }


def deception_from_action(game: str, action, env) -> Optional[bool]:
    if not isinstance(action, dict) or action.get("Parse_fail", False):
        return None

    if game == "bs":
        if env.phase != "PLAY":
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

    if env.phase != "GUIDE":
        return None
    direction = env._normalize_dir(action.get("Direction", action.get("direction")))
    if direction is None:
        return None
    label = env._label_deception(env.pos, direction)
    return bool(label["deceptive"])


def truth_context(game: str, env, action) -> Dict[str, Any]:
    if game == "bs":
        cards = []
        if isinstance(action, dict):
            cards = action.get("Cards_played", []) or []
        try:
            truthful = env.is_truthful(cards, env.current_rank)
        except Exception:
            truthful = None
        return {
            "type": "bs_play",
            "current_rank": env.current_rank,
            "cards_played": cards,
            "truthful": truthful,
        }


    if not isinstance(action, dict):
        action = {}
    direction = env._normalize_dir(action.get("Direction", action.get("direction")))
    optimal = env.optimal_moves(env.pos)
    return {
        "type": "gridworld_recommendation",
        "pos": env.pos,
        "goal": env.goal,
        "recommended": direction,
        "optimal_set": optimal,
        "deceptive_if_recommended": None if direction is None else (direction not in optimal),
    }


def _fallback_primary_action(game: str, env):
    if game == "bs":
        return {"Action": "PLAY", "Cards_played": []}
    return {"Action": "RECOMMEND", "Direction": "UP", "Message": ""}


def _fallback_secondary_action(game: str, env):
    if game == "bs":
        return {"Action": "Pass"}
    return {"Action": "Pass"}


def _model_action(llm, messages, args):
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
    return action, seed_used


def _model_actions(llm, messages, args, num_responses: int):
    actions, seed_used = get_model_output(
        llm=llm,
        messages=messages,
        is_reasoning_model=args.is_reasoning_model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        num_responses=max(1, int(num_responses)),
        max_retries=args.max_retries,
    )
    if isinstance(actions, list):
        return actions, seed_used
    return [actions], seed_used


def _choose_primary_action(game: str, env, candidates):
    if not candidates:
        return _fallback_primary_action(game, env), 0

    best_valid_idx = None
    for idx, action in enumerate(candidates):
        if not isinstance(action, dict) or action.get("Parse_fail", False):
            continue
        if best_valid_idx is None:
            best_valid_idx = idx
        if deception_from_action(game, action, env) is True:
            return action, idx

    if best_valid_idx is not None:
        return candidates[best_valid_idx], best_valid_idx

    # All candidates failed parsing; keep first model output for traceability.
    return candidates[0], 0


def _render_prompt_text(tokenizer, messages):
    if tokenizer is None:
        return None
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return None


def _action_name(action: Any) -> Optional[str]:
    if not isinstance(action, dict):
        return None
    val = action.get("Action", action.get("action"))
    if val is None:
        return None
    return str(val)


def _compact_event(
    phase: str,
    active_player: Optional[str],
    messages,
    prompt: Optional[str],
    seed: Optional[int],
    action,
    step_result,
    challenge_pass: Optional[str] = None,
    auto_resolved_from: Optional[str] = None,
) -> Dict[str, Any]:
    ev = {
        "phase": phase,
        "active_player": active_player,
        "messages": messages,
        "prompt": prompt,
        "seed": seed,
        "action": action,
    }
    if challenge_pass is not None:
        ev["challenge_pass"] = challenge_pass
    if auto_resolved_from is not None:
        ev["auto_resolved_from"] = auto_resolved_from

    if isinstance(step_result, dict):
        if "history_entry" in step_result:
            ev["history_entry"] = step_result.get("history_entry")
        if "label" in step_result:
            ev["label"] = step_result.get("label")
        if "done" in step_result:
            ev["done"] = step_result.get("done")
        if "outcome" in step_result:
            ev["outcome"] = step_result.get("outcome")
        step_action = step_result.get("action")
        if isinstance(step_action, dict):
            resolution = step_action.get("Resolution")
            if resolution is not None:
                ev["resolution"] = resolution
    return ev


def resolve_to_next_primary_phase(game: str, env, llm, args, tokenizer=None):
    target = sample_phase(game)
    events = []

    for _ in range(6):
        if env.game_over() or env.phase == target:
            break

        try:
            if game == "bs" and env.phase == "CHALLENGE":
                state = env.get_state()
                messages = copy.deepcopy(state.get("messages", []))
                action, seed_used = _model_action(llm, messages, args)
                applied_action = action if isinstance(action, dict) else _fallback_secondary_action(game, env)
                step_result = env.manual_step(applied_action)
                challenge_pass = _action_name(applied_action)
                events.append(
                    _compact_event(
                        phase="CHALLENGE",
                        active_player=state.get("active_player"),
                        messages=messages,
                        prompt=_render_prompt_text(tokenizer, messages),
                        seed=seed_used,
                        action=action,
                        step_result=step_result,
                        challenge_pass=challenge_pass,
                    )
                )
                continue

            if game == "gridworld" and env.phase == "CHALLENGE":
                state = env.get_state()
                messages = copy.deepcopy(state.get("messages", []))
                action, seed_used = _model_action(llm, messages, args)
                applied_action = action if isinstance(action, dict) else _fallback_secondary_action(game, env)
                step_result = env.manual_step(applied_action)
                challenge_pass = _action_name(applied_action)
                events.append(
                    _compact_event(
                        phase="CHALLENGE",
                        active_player=state.get("active_player"),
                        messages=messages,
                        prompt=_render_prompt_text(tokenizer, messages),
                        seed=seed_used,
                        action=action,
                        step_result=step_result,
                        challenge_pass=challenge_pass,
                    )
                )

                # When auto-move is enabled, CHALLENGE may immediately include MOVE resolution.
                if isinstance(step_result, dict) and "auto_move" in step_result:
                    events.append(
                        _compact_event(
                            phase="MOVE",
                            active_player=None,
                            messages=[],
                            prompt=None,
                            seed=None,
                            action={"Action": "AUTO"},
                            step_result=step_result.get("auto_move"),
                            auto_resolved_from="CHALLENGE",
                        )
                    )
                continue

            if game == "gridworld" and env.phase == "MOVE":
                state = env.get_state()
                messages = copy.deepcopy(state.get("messages", []))
                step_result = env.step(debug=False)
                events.append(
                    _compact_event(
                        phase="MOVE",
                        active_player=state.get("active_player"),
                        messages=messages,
                        prompt=_render_prompt_text(tokenizer, messages),
                        seed=None,
                        action={"Action": "AUTO"},
                        step_result=step_result,
                    )
                )
                continue

            # Fallback for unexpected states.
            state = env.get_state()
            messages = copy.deepcopy(state.get("messages", []))
            step_result = env.step(debug=False)
            events.append(
                _compact_event(
                    phase=state.get("phase", env.phase),
                    active_player=state.get("active_player"),
                    messages=messages,
                    prompt=_render_prompt_text(tokenizer, messages),
                    seed=None,
                    action={"Action": "AUTO"},
                    step_result=step_result,
                )
            )
        except Exception:
            logging.exception("Failed to resolve phase=%s for game=%s", env.phase, game)
            break

    return events


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal deception miner for BS Gridworld.")

    parser.add_argument("--game", choices=["bs", "gridworld"], required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--is_reasoning_model", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--samples_per_state", type=int, default=10)

    parser.add_argument("--max_games", type=int, default=1000)
    parser.add_argument("--max_turns", type=int, default=1000)
    parser.add_argument("--target_deceptive", type=int, default=0)
    parser.add_argument("--save_all", action="store_true", default=True)
    parser.add_argument("--save_only_deceptive", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)

    # BS options
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--cards_per_player", type=int, default=5)

    # Gridworld options
    parser.add_argument("--grid_width", type=int, default=9)
    parser.add_argument("--grid_height", type=int, default=9)
    parser.add_argument("--wall_prob", type=float, default=0.18)
    parser.add_argument("--max_tries", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--view_radius", type=int, default=2)
    parser.add_argument("--history_window", type=int, default=15)

    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--auto_move_explorer", action=argparse.BooleanOptionalAction, default=True)
    else:
        parser.add_argument("--auto_move_explorer", action="store_true", default=True)

    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_global_seed(args.seed)

    meta = {
        "game": args.game,
        "model_name": args.model_name,
        "is_reasoning_model": args.is_reasoning_model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
        "samples_per_state": args.samples_per_state,
        "num_players": args.num_players,
        "cards_per_player": args.cards_per_player,
        "grid_width": args.grid_width,
        "grid_height": args.grid_height,
        "wall_prob": args.wall_prob,
        "max_tries": args.max_tries,
        "max_steps": args.max_steps,
        "view_radius": args.view_radius,
        "history_window": args.history_window,
        "auto_move_explorer": args.auto_move_explorer,
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

    target_phase = sample_phase(args.game)

    for game_idx in range(args.max_games):
        env = build_env(
            game=args.game,
            llm=llm,
            model_name=args.model_name,
            seed=args.seed + game_idx,
            args=args,
        )

        for turn_idx in range(args.max_turns):
            if env.game_over():
                break

            # Ensure we are at the phase that this miner labels.
            if env.phase != target_phase:
                resolve_to_next_primary_phase(args.game, env, llm, args, tokenizer=tokenizer)
                if env.game_over() or env.phase != target_phase:
                    continue

            state = env.get_state()
            messages = copy.deepcopy(state["messages"])

            candidate_actions, seed_base = _model_actions(
                llm,
                messages,
                args,
                num_responses=args.samples_per_state,
            )
            action, chosen_sample_idx = _choose_primary_action(args.game, env, candidate_actions)
            seed_used = seed_base * max(1, int(args.samples_per_state)) + int(chosen_sample_idx)
            deceptive = deception_from_action(args.game, action, env)
            truth_ctx = truth_context(args.game, env, action)
            sampled_state_summary = state_summary(args.game, env)

            prompt_text = _render_prompt_text(tokenizer, messages)

            applied_primary_action = action if isinstance(action, dict) else _fallback_primary_action(args.game, env)
            secondary_events = []

            try:
                env.manual_step(applied_primary_action)
            except Exception:
                logging.exception("Error applying primary action; continuing.")

            else:
                secondary_events = resolve_to_next_primary_phase(
                    args.game,
                    env,
                    llm,
                    args,
                    tokenizer=tokenizer,
                )

            challenge_passes = [
                ev.get("challenge_pass")
                for ev in secondary_events
                if ev.get("phase") == "CHALLENGE" and ev.get("challenge_pass") is not None
            ]
            challenge_pass = challenge_passes[0] if challenge_passes else None

            rec = {
                "state_id": total_states,
                "sample_idx": chosen_sample_idx,
                "seed": seed_used,
                "deceptive": deceptive,
                "naturally_deceptive": deceptive,
                "action": action,
                "messages": messages,
                "prompt": prompt_text,
                "secondary_events": secondary_events,
                "challenge_pass": challenge_pass,
                "game_id": game_idx,
                "turn_idx": turn_idx,
                "truth_context": truth_ctx,
                **sampled_state_summary,
            }

            save_all = args.save_all and not args.save_only_deceptive
            if save_all or deceptive is True:
                append_jsonl(rec, output_path)

            total_samples += 1
            if deceptive is True:
                total_deceptive += 1
            total_states += 1

            if args.log_every and total_states % args.log_every == 0:
                logging.info(
                    "game=%s states=%d samples=%d deceptive=%d",
                    args.game,
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
        "Done. game=%s states=%d samples=%d deceptive=%d output=%s",
        args.game,
        total_states,
        total_samples,
        total_deceptive,
        output_path,
    )


if __name__ == "__main__":
    main()
