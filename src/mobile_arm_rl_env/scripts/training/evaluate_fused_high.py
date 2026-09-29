#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministic evaluation for the learned joint-subgoal high policy."""

import argparse
import collections
import json
import os
import sys

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_high_actor_critic import FusedHighActorCritic  # noqa: E402
from training.fused_high_env_client import FusedHighEnvironmentClient  # noqa: E402
from training.fused_reach_scenarios import (  # noqa: E402
    FusedReachScenarioSampler,
    load_fused_reach_scenarios,
    scenario_category_counts,
    scenario_step_budget,
    split_fused_reach_scenarios,
)
from training.hrl4in_low_actor_critic import (  # noqa: E402
    RunningObservationNormalizer,
)


def main():
    args = _parse_arguments()
    model = None
    normalizer = None
    checkpoint = None
    if not args.teacher_policy:
        if not args.checkpoint:
            raise ValueError(
                "--checkpoint is required unless --teacher-policy is used"
            )
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        arguments = dict(checkpoint.get("arguments", {}))
        model = FusedHighActorCritic(
            hidden_sizes=arguments.get("hidden_sizes", [256, 256]),
            initial_log_std=arguments.get("initial_log_std", -3.5),
        )
        if int(checkpoint.get("observation_dim", -1)) != model.OBS_DIM:
            raise ValueError("high checkpoint observation dimension mismatch")
        if int(checkpoint.get("action_dim", -1)) != model.ACTION_DIM:
            raise ValueError("high checkpoint action dimension mismatch")
        if int(checkpoint.get("subgoal_type_count", -1)) != (
                model.SUBGOAL_TYPE_COUNT):
            raise ValueError("high checkpoint subgoal-type dimension mismatch")
        if int(checkpoint.get("route_side_count", -1)) != model.ROUTE_SIDE_COUNT:
            raise ValueError("high checkpoint route-side dimension mismatch")
        if int(checkpoint.get("option_count", -1)) != model.OPTION_COUNT:
            raise ValueError("high checkpoint unified-option dimension mismatch")
        if str(checkpoint.get("policy_type")) != model.POLICY_TYPE:
            raise ValueError("checkpoint is not a route-option high policy")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        normalizer = RunningObservationNormalizer(
            observation_dim=model.OBS_DIM,
            normalized_dim=model.NORMALIZED_DIM,
        )
        normalizer.load_state_dict(checkpoint["normalizer"])

    scenario_set = load_fused_reach_scenarios(args.scenario_set)
    scenarios = split_fused_reach_scenarios(
        scenario_set["scenarios"],
        split=args.scenario_split,
        validation_fraction=args.scenario_validation_fraction,
        test_fraction=args.scenario_test_fraction,
        seed=args.scenario_split_seed,
    )
    scenarios = _filter_scenario_ids(scenarios, args.scenario_ids)
    sampler = FusedReachScenarioSampler(
        scenarios,
        seed=args.scenario_split_seed,
        shuffle=False,
    )
    default_budget = (
        int(args.scenario_step_budget)
        if args.scenario_step_budget > 0
        else int(scenario_set["recommended_step_budget"])
    )
    episode_count = int(args.episodes)
    if episode_count <= 0:
        episode_count = len(scenarios)
    print(
        "high_evaluation_scenarios path={} split={} scenarios={} budget={} "
        "categories={} policy={}".format(
            args.scenario_set,
            args.scenario_split,
            len(scenarios),
            default_budget,
            scenario_category_counts(scenarios),
            "safe_waypoint_teacher" if args.teacher_policy else args.checkpoint,
        )
    )

    totals = collections.Counter()
    category_episodes = collections.Counter()
    category_successes = collections.Counter()
    rewards = []
    final_distances = []
    action_magnitudes = []
    route_side_counts = collections.Counter()
    episode_results = []
    client = FusedHighEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )
    try:
        for episode in range(1, episode_count + 1):
            scenario = sampler.next()
            budget = scenario_step_budget(scenario, default_budget)
            observation, reset_info = client.reset_with_info(
                scenario=scenario,
                max_steps=budget,
            )
            episode_reward = 0.0
            high_steps = 0
            low_steps = 0
            safety_steps = 0
            overlap_steps = 0
            final_info = {}
            while low_steps < budget:
                if args.teacher_policy:
                    action = client.teacher_action.copy()
                    subgoal_type = int(
                        client.teacher_info.get("subgoal_type", 0)
                    )
                    route_side = int(
                        client.teacher_info.get("route_side", 0)
                    )
                else:
                    normalized = normalizer.normalize(observation)
                    observation_tensor = torch.from_numpy(
                        normalized
                    ).unsqueeze(0)
                    with torch.no_grad():
                        action_tensor, type_tensor, route_tensor = (
                            model.deterministic_decision(observation_tensor)
                        )
                        action = action_tensor.squeeze(0).cpu().numpy()
                        subgoal_type = int(type_tensor.item())
                        route_side = int(route_tensor.item())
                route_side_counts[route_side] += 1
                action_magnitudes.append(float(np.mean(np.abs(action))))
                observation, reward, done, info = client.step(
                    action,
                    subgoal_type=subgoal_type,
                    route_side=route_side,
                )
                episode_reward += float(reward)
                high_steps += 1
                low_steps += int(info.get("low_steps", 0))
                safety_steps += int(info.get("safety_steps", 0))
                overlap_steps += int(info.get("overlap_steps", 0))
                final_info = info
                if done:
                    break
            success = bool(final_info.get("success", False))
            collision = bool(final_info.get("collision", False))
            timeout = bool(final_info.get("timeout", not success and not collision))
            category = str(scenario.get("category", "unknown"))
            final_distance = float(final_info.get("dist", float("nan")))
            category_episodes[category] += 1
            category_successes[category] += int(success)
            totals["episodes"] += 1
            totals["successes"] += int(success)
            totals["collisions"] += int(collision)
            totals["timeouts"] += int(timeout)
            totals["high_steps"] += high_steps
            totals["low_steps"] += low_steps
            totals["safety_steps"] += safety_steps
            totals["overlap_steps"] += overlap_steps
            rewards.append(episode_reward)
            final_distances.append(final_distance)
            result = {
                "episode": episode,
                "scenario_id": scenario.get("scenario_id"),
                "category": category,
                "success": success,
                "collision": collision,
                "timeout": timeout,
                "reward": episode_reward,
                "high_steps": high_steps,
                "low_steps": low_steps,
                "final_distance": final_distance,
                "safety_steps": safety_steps,
                "overlap_steps": overlap_steps,
                "detour_side": reset_info.get("detour_side"),
            }
            episode_results.append(result)
            print(
                "high_episode={} scenario_id={} category={} high_steps={} "
                "low_steps={} reward={:.4f} success={} collision={} timeout={} "
                "final_distance={:.4f} safety_steps={} overlap_steps={}".format(
                    episode,
                    scenario.get("scenario_id"),
                    category,
                    high_steps,
                    low_steps,
                    episode_reward,
                    success,
                    collision,
                    timeout,
                    final_distance,
                    safety_steps,
                    overlap_steps,
                )
            )
    finally:
        client.close()

    success_rate = float(totals["successes"]) / float(max(totals["episodes"], 1))
    safety_rate = float(totals["safety_steps"]) / float(max(totals["low_steps"], 1))
    overlap_rate = float(totals["overlap_steps"]) / float(max(totals["low_steps"], 1))
    metrics = {
        "policy": "safe_waypoint_teacher" if args.teacher_policy else args.checkpoint,
        "checkpoint_steps": None if checkpoint is None else int(
            checkpoint.get("total_steps", 0)
        ),
        "checkpoint_update": None if checkpoint is None else int(
            checkpoint.get("update_index", 0)
        ),
        "episodes": int(totals["episodes"]),
        "successes": int(totals["successes"]),
        "success_rate": success_rate,
        "collisions": int(totals["collisions"]),
        "timeouts": int(totals["timeouts"]),
        "high_steps": int(totals["high_steps"]),
        "low_steps": int(totals["low_steps"]),
        "mean_reward": float(np.mean(rewards)),
        "mean_final_distance": float(np.nanmean(final_distances)),
        "mean_action_abs": float(np.mean(action_magnitudes)),
        "route_side_counts": dict(route_side_counts),
        "safety_steps": int(totals["safety_steps"]),
        "safety_rate": safety_rate,
        "overlap_steps": int(totals["overlap_steps"]),
        "overlap_rate": overlap_rate,
        "category_episodes": dict(category_episodes),
        "category_successes": dict(category_successes),
        "episode_results": episode_results,
    }
    print(
        "high_evaluation episodes={} successes={} success_rate={:.3f} "
        "collisions={} timeouts={} mean_reward={:.4f} "
        "mean_final_distance={:.4f} high_steps={} low_steps={} "
        "safety_rate={:.4f} overlap_rate={:.4f} action_abs={:.5f} "
        "route_sides={}".format(
            metrics["episodes"],
            metrics["successes"],
            metrics["success_rate"],
            metrics["collisions"],
            metrics["timeouts"],
            metrics["mean_reward"],
            metrics["mean_final_distance"],
            metrics["high_steps"],
            metrics["low_steps"],
            metrics["safety_rate"],
            metrics["overlap_rate"],
            metrics["mean_action_abs"],
            metrics["route_side_counts"],
        )
    )
    for category in sorted(category_episodes):
        count = int(category_episodes[category])
        category_success = int(category_successes[category])
        print(
            "high_category category={} episodes={} successes={} success_rate={:.3f}".format(
                category,
                count,
                category_success,
                float(category_success) / float(max(count, 1)),
            )
        )
    if args.metrics_output:
        directory = os.path.dirname(os.path.abspath(args.metrics_output))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        with open(args.metrics_output, "w") as stream:
            json.dump(metrics, stream, indent=2, sort_keys=True)
        print("high_metrics_saved={}".format(args.metrics_output))
    gate_pass = bool(
        success_rate >= args.min_success_rate
        and totals["collisions"] <= args.max_collisions
        and totals["timeouts"] <= args.max_timeouts
        and safety_rate <= args.max_safety_rate
    )
    print("high_gate_pass={}".format(gate_pass))
    if not gate_pass:
        raise SystemExit(2)


def _filter_scenario_ids(scenarios, requested_ids):
    if not requested_ids:
        return scenarios
    requested = set(int(value) for value in requested_ids)
    selected = [
        scenario for scenario in scenarios
        if int(scenario["scenario_id"]) in requested
    ]
    missing = requested - set(int(item["scenario_id"]) for item in selected)
    if missing:
        raise ValueError("scenario ids are not in selected split: {}".format(sorted(missing)))
    return selected


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--teacher-policy", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5565)
    parser.add_argument("--socket-timeout", type=float, default=180.0)
    parser.add_argument(
        "--scenario-set",
        default=(
            "src/mobile_arm_rl_env/config/"
            "fused_box_subgoal_randomized_scenarios.json"
        ),
    )
    parser.add_argument(
        "--scenario-split",
        choices=["all", "train", "validation", "test"],
        default="validation",
    )
    parser.add_argument("--scenario-validation-fraction", type=float, default=0.10)
    parser.add_argument("--scenario-test-fraction", type=float, default=0.20)
    parser.add_argument("--scenario-split-seed", type=int, default=123)
    parser.add_argument("--scenario-ids", type=int, nargs="+")
    parser.add_argument("--scenario-step-budget", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--metrics-output")
    parser.add_argument("--min-success-rate", type=float, default=0.0)
    parser.add_argument("--max-collisions", type=int, default=1000000)
    parser.add_argument("--max-timeouts", type=int, default=1000000)
    parser.add_argument("--max-safety-rate", type=float, default=1.0)
    args = parser.parse_args()
    if args.teacher_policy and args.checkpoint:
        parser.error("--teacher-policy and --checkpoint are mutually exclusive")
    return args


if __name__ == "__main__":
    main()
