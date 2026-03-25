#!/usr/bin/env python3
import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parent
BS_SRC_ROOT = REPO_ROOT / "BS" / "src"
GRIDWORLD_SRC_ROOT = REPO_ROOT / "Gridworld" / "src"
INTERVIEW_SRC_ROOT = REPO_ROOT / "Interview" / "src"

for path in (SRC_ROOT, BS_SRC_ROOT, GRIDWORLD_SRC_ROOT, INTERVIEW_SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from vllm import LLM

from deception_dataset import (
    LABEL_FILTER_CHOICES,
    keep_record_for_label_filter,
    normalize_label_filter,
)
from llm_agent import LLMAgent
from utils import (
    append_jsonl,
    atomic_write_json,
    get_model_output,
    prepare_messages_for_model,
    set_global_seed,
)


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
        "interview": "CANDIDATE_RESPONSE",
    }[game]


def make_agents(game: str, llm, model_name: str, n_players: int, is_reasoning_model: bool):
    if is_reasoning_model:
        reasoning_instruction = "COD"
        instruction_format = "default"
    else:
        reasoning_instruction = "COT"
        instruction_format = "reasoning"

    if game in {"gridworld", "interview"}:
        n_players = 2

    if game == "interview":
        names = ["Candidate", "HiringManager"]
    else:
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
        elif game == "interview":
            pass
        else:
            raise ValueError(f"Unsupported game: {game}")

        agents.append(ag)

    return agents


def _load_interview_scenarios(args):
    cached = getattr(args, "_interview_scenarios_cache", None)
    if cached is not None:
        return cached

    if not args.interview_conversations_path:
        raise ValueError("--interview_conversations_path is required when --game interview.")

    from interview_environment import load_interview_scenarios_from_path

    scenarios = load_interview_scenarios_from_path(args.interview_conversations_path)
    if not scenarios:
        raise ValueError(f"No interview conversation seeds found in {args.interview_conversations_path}")

    args._interview_scenarios_cache = scenarios
    return scenarios


def _iter_jsonl(path: str):
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                logging.warning("Skipping unreadable JSONL row at %s:%d", path, line_idx)
                continue
            if isinstance(row, dict):
                yield row


def _load_processed_interview_conversation_ids(output_path: str, processed_path: str) -> set[str]:
    seen: set[str] = set()
    for row in _iter_jsonl(processed_path):
        conversation_id = row.get("conversation_id")
        if conversation_id:
            seen.add(str(conversation_id))
    for row in _iter_jsonl(output_path):
        conversation_id = row.get("conversation_id")
        if conversation_id:
            seen.add(str(conversation_id))
    return seen


def _append_processed_interview_conversation(
    processed_path: str,
    env,
    game_idx: int,
    game_saved_count: int,
    game_sample_count: int,
    game_deceptive_count: int,
    game_truthful_count: int,
    game_unknown_count: int,
) -> None:
    append_jsonl(
        {
            "processed_at": time.time(),
            "game": "interview",
            "game_id": game_idx,
            "conversation_id": env.current_scenario.conversation_id,
            "scenario_name": env.current_scenario.name,
            "base_scenario_name": env.current_scenario.base_scenario_name,
            "saved_count": int(game_saved_count),
            "sample_count": int(game_sample_count),
            "deceptive_count": int(game_deceptive_count),
            "truthful_count": int(game_truthful_count),
            "unknown_count": int(game_unknown_count),
            "outcome": env.outcome,
        },
        processed_path,
    )


def build_env(game: str, llm, model_name: str, seed: int, args, game_idx: int = 0):
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

    if game == "interview":
        from interview_environment import InterviewSpec, JobInterviewEnvironment

        scenarios = _load_interview_scenarios(args)
        scenario = scenarios[int(game_idx)]
        agents = make_agents(game, llm, model_name, 2, args.is_reasoning_model)
        spec = InterviewSpec(
            auto_generate_dialogue=False,
            generated_dialogue_turns=0,
        )
        return JobInterviewEnvironment(
            agents=agents,
            seed=seed,
            scenario_name=scenario.name,
            private_profile_name=args.interview_private_profile_name,
            scenarios=scenarios,
            spec=spec,
            history_window=args.history_window,
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
    if game == "interview":
        offer_total_comp = env._offer_total_comp(env.current_scenario.offer)
        return {
            "phase": env.phase,
            "active_player": active,
            "conversation_id": env.current_scenario.conversation_id,
            "scenario_name": env.current_scenario.name,
            "base_scenario_name": env.current_scenario.base_scenario_name,
            "offer_total_comp": offer_total_comp,
            "candidate_target_total_comp": env._candidate_target_total_comp(),
            "history_len": len(env.history),
            "game_type": "interview",
        }

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
    if isinstance(action, dict) and action.get("Parse_fail", False):
        return None

    if game == "bs":
        if not isinstance(action, dict):
            return None
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

    if game == "interview":
        if env.phase != "CANDIDATE_RESPONSE":
            return None
        try:
            label = env.label_candidate_action(action)
        except Exception:
            return None
        return bool(label.get("deceptive"))

    if not isinstance(action, dict):
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

    if game == "interview":
        normalized = env.normalize_candidate_action(action)
        label = env.label_candidate_action(normalized)
        return {
            "type": "interview_final_response",
            "decision": normalized.get("Decision"),
            "offer_total_comp": env._offer_total_comp(env.current_scenario.offer),
            "candidate_target_total_comp": env._candidate_target_total_comp(),
            "true_has_other_offer": bool(env.current_private_state.has_other_offer),
            "true_other_offer_company": env.current_private_state.other_offer_company,
            "true_other_offer_base_salary": env.current_private_state.other_offer_base_salary,
            "label_preview": label,
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
    if game == "interview":
        return env.get_truthful_action()
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
        model_name=args.model_name,
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
        model_name=args.model_name,
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


def _sample_balanced_candidates(game: str, env, candidates):
    deceptive_candidates = []
    truthful_candidates = []
    unknown_candidates = []

    for idx, action in enumerate(candidates):
        label = deception_from_action(game, action, env)
        row = (idx, action, label)
        if label is True:
            deceptive_candidates.append(row)
        elif label is False:
            truthful_candidates.append(row)
        else:
            unknown_candidates.append(row)

    picked_deceptive = random.choice(deceptive_candidates) if deceptive_candidates else None
    picked_truthful = random.choice(truthful_candidates) if truthful_candidates else None

    return picked_deceptive, picked_truthful, unknown_candidates


def _targets_reached(
    total_deceptive: int,
    total_truthful: int,
    args,
    use_target_deceptive: bool,
    use_target_truthful: bool,
) -> bool:
    if not (use_target_deceptive or use_target_truthful):
        return False

    deceptive_done = (not use_target_deceptive) or (total_deceptive >= args.target_deceptive)
    truthful_done = (not use_target_truthful) or (total_truthful >= args.target_truthful)
    return deceptive_done and truthful_done


def _render_prompt_text(tokenizer, messages, model_name: Optional[str] = None):
    if tokenizer is None:
        return None
    try:
        prepared = prepare_messages_for_model(messages, model_name=model_name)
        return tokenizer.apply_chat_template(
            prepared,
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
                messages = prepare_messages_for_model(messages, model_name=args.model_name)
                action, seed_used = _model_action(llm, messages, args)
                applied_action = action if isinstance(action, dict) else _fallback_secondary_action(game, env)
                step_result = env.manual_step(applied_action)
                challenge_pass = _action_name(applied_action)
                events.append(
                    _compact_event(
                        phase="CHALLENGE",
                        active_player=state.get("active_player"),
                        messages=messages,
                        prompt=_render_prompt_text(tokenizer, messages, model_name=args.model_name),
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
                messages = prepare_messages_for_model(messages, model_name=args.model_name)
                action, seed_used = _model_action(llm, messages, args)
                applied_action = action if isinstance(action, dict) else _fallback_secondary_action(game, env)
                step_result = env.manual_step(applied_action)
                challenge_pass = _action_name(applied_action)
                events.append(
                    _compact_event(
                        phase="CHALLENGE",
                        active_player=state.get("active_player"),
                        messages=messages,
                        prompt=_render_prompt_text(tokenizer, messages, model_name=args.model_name),
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
                messages = prepare_messages_for_model(messages, model_name=args.model_name)
                step_result = env.step(debug=False)
                events.append(
                    _compact_event(
                        phase="MOVE",
                        active_player=state.get("active_player"),
                        messages=messages,
                        prompt=_render_prompt_text(tokenizer, messages, model_name=args.model_name),
                        seed=None,
                        action={"Action": "AUTO"},
                        step_result=step_result,
                    )
                )
                continue

            # Fallback for unexpected states.
            state = env.get_state()
            messages = copy.deepcopy(state.get("messages", []))
            messages = prepare_messages_for_model(messages, model_name=args.model_name)
            step_result = env.step(debug=False)
            events.append(
                _compact_event(
                    phase=state.get("phase", env.phase),
                    active_player=state.get("active_player"),
                    messages=messages,
                    prompt=_render_prompt_text(tokenizer, messages, model_name=args.model_name),
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
    parser = argparse.ArgumentParser(description="Universal deception miner for BS, Gridworld, and Interview.")

    parser.add_argument("--game", choices=["bs", "gridworld", "interview"], required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--is_reasoning_model", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--samples_per_state", type=int, default=1)

    parser.add_argument("--max_games", type=int, default=1000)
    parser.add_argument("--max_turns", type=int, default=1000)
    parser.add_argument("--target_deceptive", type=int, default=3000)
    parser.add_argument("--target_truthful", type=int, default=3000)
    parser.add_argument("--label_filter", type=str, choices=LABEL_FILTER_CHOICES, default="all")
    parser.add_argument("--only_deceptive", action="store_true", default=False)
    parser.add_argument("--only_truthful", action="store_true", default=False)
    parser.add_argument("--save_all", action="store_true", default=True)
    parser.add_argument("--save_only_deceptive", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

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

    # Interview options
    parser.add_argument("--interview_conversations_path", type=str, default=None)
    parser.add_argument("--interview_private_profile_name", type=str, default=None)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    else:
        parser.add_argument("--resume", dest="resume", action="store_true")
        parser.add_argument("--no-resume", dest="resume", action="store_false")
        parser.set_defaults(resume=True)

    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    legacy_only_deceptive = args.save_only_deceptive or (not args.save_all)
    label_filter = normalize_label_filter(
        args.label_filter,
        only_deceptive=(args.only_deceptive or legacy_only_deceptive),
        only_truthful=args.only_truthful,
    )
    use_target_deceptive = args.target_deceptive > 0 and label_filter != "truthful_only"
    use_target_truthful = args.target_truthful > 0 and label_filter != "deceptive_only"
    if int(args.num_shards) <= 0:
        raise ValueError("--num_shards must be positive.")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards.")

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
        "interview_conversations_path": args.interview_conversations_path,
        "interview_private_profile_name": args.interview_private_profile_name,
        "max_games": args.max_games,
        "max_turns": args.max_turns,
        "target_deceptive": args.target_deceptive,
        "target_truthful": args.target_truthful,
        "label_filter": label_filter,
        "use_target_deceptive": use_target_deceptive,
        "use_target_truthful": use_target_truthful,
        "save_all": args.save_all,
        "save_only_deceptive": args.save_only_deceptive,
        "resume": args.resume,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "seed": args.seed,
        "strategy": "game_flow",
        "timestamp": time.time(),
    }
    atomic_write_json(os.path.join(args.output_dir, "meta.json"), meta)

    logging.info("Loading model %s ...", args.model_name)
    llm = init_llm(args.model_name, args.max_tokens, seed=args.seed)

    output_path = os.path.join(args.output_dir, "deception_samples.jsonl")
    processed_path = os.path.join(args.output_dir, "processed_interview_conversations.jsonl")

    total_states = 0
    total_samples = 0
    total_deceptive = 0
    total_truthful = 0
    total_unknown = 0
    total_saved = 0

    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        tokenizer = None

    target_phase = sample_phase(args.game)
    max_game_slots = args.max_games
    interview_scenarios = None
    processed_interview_ids: set[str] = set()
    if args.game == "interview":
        interview_scenarios = _load_interview_scenarios(args)
        meta["interview_num_seed_conversations"] = len(interview_scenarios)
        meta["processed_interview_manifest"] = processed_path
        atomic_write_json(os.path.join(args.output_dir, "meta.json"), meta)
        max_game_slots = min(args.max_games, len(interview_scenarios))
        if args.resume:
            processed_interview_ids = _load_processed_interview_conversation_ids(
                output_path=output_path,
                processed_path=processed_path,
            )
            if processed_interview_ids:
                logging.info(
                    "Resume enabled for interview miner: skipping %d previously processed conversations.",
                    len(processed_interview_ids),
                )

    for game_idx in range(max_game_slots):
        if game_idx % int(args.num_shards) != int(args.shard_index):
            continue
        if _targets_reached(
            total_deceptive,
            total_truthful,
            args,
            use_target_deceptive,
            use_target_truthful,
        ):
            break

        if args.game == "interview" and interview_scenarios is not None:
            conversation_id = interview_scenarios[game_idx].conversation_id
            if conversation_id and str(conversation_id) in processed_interview_ids:
                continue

        env = build_env(
            game=args.game,
            llm=llm,
            model_name=args.model_name,
            seed=args.seed + game_idx,
            args=args,
            game_idx=game_idx,
        )
        game_sample_count = 0
        game_saved_count = 0
        game_deceptive_count = 0
        game_truthful_count = 0
        game_unknown_count = 0
        game_processed_state_count = 0

        for turn_idx in range(args.max_turns):
            if _targets_reached(
                total_deceptive,
                total_truthful,
                args,
                use_target_deceptive,
                use_target_truthful,
            ):
                break

            if env.game_over():
                break

            # Ensure we are at the phase that this miner labels.
            if env.phase != target_phase:
                resolve_to_next_primary_phase(args.game, env, llm, args, tokenizer=tokenizer)
                if env.game_over() or env.phase != target_phase:
                    continue

            state = env.get_state()
            messages = copy.deepcopy(state["messages"])
            messages = prepare_messages_for_model(messages, model_name=args.model_name)

            candidate_actions, seed_base = _model_actions(
                llm,
                messages,
                args,
                num_responses=args.samples_per_state,
            )
            sampled_state_summary = state_summary(args.game, env)
            prompt_text = _render_prompt_text(tokenizer, messages, model_name=args.model_name)

            picked_deceptive, picked_truthful, _ = _sample_balanced_candidates(
                args.game,
                env,
                candidate_actions,
            )

            transition_candidates = []
            if picked_deceptive is not None:
                transition_candidates.append(picked_deceptive)
            if picked_truthful is not None:
                transition_candidates.append(picked_truthful)

            record_candidates = []
            need_deceptive = (not use_target_deceptive) or (total_deceptive < args.target_deceptive)
            need_truthful = (not use_target_truthful) or (total_truthful < args.target_truthful)
            if picked_deceptive is not None and need_deceptive:
                sample_idx, action, deceptive = picked_deceptive
                record_candidates.append((sample_idx, action, deceptive, truth_context(args.game, env, action)))
            if picked_truthful is not None and need_truthful:
                sample_idx, action, deceptive = picked_truthful
                record_candidates.append((sample_idx, action, deceptive, truth_context(args.game, env, action)))

            if transition_candidates:
                chosen_sample_idx, transition_action, _transition_deceptive = random.choice(transition_candidates)
            else:
                transition_action, chosen_sample_idx = _choose_primary_action(args.game, env, candidate_actions)
                transition_deceptive = deception_from_action(args.game, transition_action, env)
                if not record_candidates:
                    record_candidates.append(
                        (
                            chosen_sample_idx,
                            transition_action,
                            transition_deceptive,
                            truth_context(args.game, env, transition_action),
                        )
                    )

            applied_primary_action = (
                transition_action
                if isinstance(transition_action, dict)
                else _fallback_primary_action(args.game, env)
            )
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

            record_entries = []
            for sample_idx, action, deceptive, truth_ctx in record_candidates:
                seed_used = seed_base * max(1, int(args.samples_per_state)) + int(sample_idx)
                used_for_transition = int(sample_idx) == int(chosen_sample_idx)
                rec = {
                    "state_id": total_states,
                    "sample_idx": sample_idx,
                    "seed": seed_used,
                    "deceptive": deceptive,
                    "naturally_deceptive": deceptive,
                    "used_for_transition": used_for_transition,
                    "action": action,
                    "messages": messages,
                    "prompt": prompt_text,
                    "secondary_events": secondary_events if used_for_transition else [],
                    "challenge_pass": challenge_pass if used_for_transition else None,
                    "game_id": game_idx,
                    "turn_idx": turn_idx,
                    "truth_context": truth_ctx,
                    **sampled_state_summary,
                }
                record_entries.append(rec)

            for rec in record_entries:
                if keep_record_for_label_filter(rec, label_filter):
                    append_jsonl(rec, output_path)
                    total_saved += 1
                    game_saved_count += 1

                total_samples += 1
                game_sample_count += 1
                deceptive = rec.get("deceptive")
                if deceptive is True:
                    total_deceptive += 1
                    game_deceptive_count += 1
                elif deceptive is False:
                    total_truthful += 1
                    game_truthful_count += 1
                else:
                    total_unknown += 1
                    game_unknown_count += 1
            total_states += 1
            game_processed_state_count += 1

            if args.log_every and total_states % args.log_every == 0:
                logging.info(
                    "game=%s states=%d samples=%d deceptive=%d truthful=%d unknown=%d saved=%d filter=%s",
                    args.game,
                    total_states,
                    total_samples,
                    total_deceptive,
                    total_truthful,
                    total_unknown,
                    total_saved,
                    label_filter,
                )

            if _targets_reached(
                total_deceptive,
                total_truthful,
                args,
                use_target_deceptive,
                use_target_truthful,
            ):
                logging.info(
                    "Reached target counts: deceptive=%d/%d truthful=%d/%d",
                    total_deceptive,
                    args.target_deceptive,
                    total_truthful,
                    args.target_truthful,
                )
                break

        if args.game == "interview" and game_processed_state_count > 0:
            _append_processed_interview_conversation(
                processed_path=processed_path,
                env=env,
                game_idx=game_idx,
                game_saved_count=game_saved_count,
                game_sample_count=game_sample_count,
                game_deceptive_count=game_deceptive_count,
                game_truthful_count=game_truthful_count,
                game_unknown_count=game_unknown_count,
            )
            if env.current_scenario.conversation_id:
                processed_interview_ids.add(str(env.current_scenario.conversation_id))

        if _targets_reached(
            total_deceptive,
            total_truthful,
            args,
            use_target_deceptive,
            use_target_truthful,
        ):
            break

    logging.info(
        "Done. game=%s states=%d samples=%d deceptive=%d truthful=%d unknown=%d saved=%d filter=%s output=%s",
        args.game,
        total_states,
        total_samples,
        total_deceptive,
        total_truthful,
        total_unknown,
        total_saved,
        label_filter,
        output_path,
    )


if __name__ == "__main__":
    main()
