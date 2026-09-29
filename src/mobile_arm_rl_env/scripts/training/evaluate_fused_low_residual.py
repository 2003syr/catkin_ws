#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministic Gazebo evaluation for the fused residual policy."""

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

from training.fused_low_residual_env_client import (  # noqa: E402
    FusedLowResidualEnvironmentClient,
)
from training.fused_residual_actor_critic import (  # noqa: E402
    FusedResidualActorCritic,
)
from training.fused_reach_scenarios import (  # noqa: E402
    FusedReachScenarioSampler,
    filter_scenario_categories,
    load_fused_reach_scenarios,
    scenario_category_counts,
    scenario_step_budget as get_scenario_step_budget,
    split_fused_reach_scenarios,
)
from training.hrl4in_low_actor_critic import (  # noqa: E402
    RunningObservationNormalizer,
)
from hrl.fused_low_level import FusedLowLevelState  # noqa: E402


def main():
    args = _parse_arguments()
    model = None
    normalizer = None
    checkpoint_steps = None
    checkpoint_update = None
    if not args.zero_residual:
        if not args.checkpoint:
            raise ValueError(
                "--checkpoint is required unless --zero-residual is used"
            )
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        checkpoint_steps = int(checkpoint.get("total_steps", 0))
        checkpoint_update = int(checkpoint.get("update_index", 0))
        arguments = checkpoint.get("arguments", {})
        model = FusedResidualActorCritic(
            hidden_sizes=arguments.get("hidden_sizes", [128, 128]),
            initial_log_std=arguments.get("initial_log_std", -3.0),
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()
        normalizer = RunningObservationNormalizer(
            observation_dim=model.OBS_DIM,
            normalized_dim=52,
        )
        normalizer.load_state_dict(checkpoint["normalizer"])

    scenario_sampler = None
    scenario_step_budget = None
    scenarios = []
    if args.scenario_set:
        scenario_set = load_fused_reach_scenarios(args.scenario_set)
        eligible_scenarios = filter_scenario_categories(
            scenario_set["scenarios"],
            args.scenario_categories,
        )
        scenarios = split_fused_reach_scenarios(
            eligible_scenarios,
            split=args.scenario_split,
            validation_fraction=args.scenario_validation_fraction,
            test_fraction=args.scenario_test_fraction,
            seed=args.scenario_split_seed,
        )
        scenarios = _filter_scenario_ids(scenarios, args.scenario_ids)
        scenario_sampler = FusedReachScenarioSampler(
            scenarios,
            seed=args.scenario_split_seed,
            shuffle=False,
        )
        scenario_step_budget = (
            int(args.scenario_step_budget)
            if args.scenario_step_budget > 0
            else int(scenario_set["recommended_step_budget"])
        )
        print(
            "fused_evaluation_scenarios path={} split={} scenarios={} "
            "step_budget={} categories={}".format(
                args.scenario_set,
                args.scenario_split,
                len(scenarios),
                scenario_step_budget,
                scenario_category_counts(scenarios),
            )
        )
    episode_count = int(args.episodes)
    if episode_count <= 0:
        if not scenarios:
            raise ValueError(
                "--episodes must be positive without --scenario-set"
            )
        episode_count = len(scenarios)

    successes = 0
    collisions = 0
    timeouts = 0
    tf_failures = 0
    total_steps = 0
    safety_steps = 0
    maximum_projection = 0.0
    final_distances = []
    rewards = []
    residual_magnitudes = []
    safety_reasons = collections.Counter()
    blocked_joint_steps = collections.Counter()
    avoidance_steps = 0
    maximum_avoidance = 0.0
    safety_joint_reasons = collections.Counter()
    phase_steps = collections.Counter()
    subgoal_type_steps = collections.Counter()
    fallback_steps = 0
    category_totals = collections.Counter()
    category_successes = collections.Counter()
    episode_results = []

    client = FusedLowResidualEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )
    try:
        for episode in range(1, episode_count + 1):
            scenario = (
                scenario_sampler.next()
                if scenario_sampler is not None else None
            )
            scenario_id = (
                scenario.get("scenario_id") if scenario is not None else None
            )
            category = (
                str(scenario.get("category", "random"))
                if scenario is not None else "random"
            )
            category_totals[category] += 1
            episode_budget = get_scenario_step_budget(
                scenario,
                scenario_step_budget
                if scenario_step_budget is not None else args.max_steps,
            )
            # The reset command must receive the same per-scenario budget that
            # the local evaluation loop uses; otherwise the server can mark a
            # dynamic-budget task terminal at the legacy global horizon.
            if scenario is not None:
                observation, reset_info = client.reset_with_info(
                    scenario=scenario,
                    max_steps=episode_budget,
                )
            else:
                observation, reset_info = client.reset_with_info(
                    max_steps=episode_budget
                )
            episode_reward = 0.0
            final_info = {}
            episode_safety_steps = 0
            episode_safety_reasons = collections.Counter()
            episode_blocked_joint_steps = collections.Counter()
            episode_avoidance_steps = 0
            episode_maximum_avoidance = 0.0
            episode_safety_joint_reasons = collections.Counter()
            episode_phase_steps = collections.Counter()
            episode_subgoal_type_steps = collections.Counter()
            episode_fallback_steps = 0
            arm_positions = _arm_positions(observation)
            initial_arm_positions = _arm_positions_from_reset(
                reset_info, observation
            )
            minimum_arm_positions = arm_positions.copy()
            maximum_arm_positions = arm_positions.copy()
            episode_limit = episode_budget
            for step in range(1, episode_limit + 1):
                if args.zero_residual:
                    residual = np.zeros(5, dtype=np.float32)
                else:
                    normalized = normalizer.normalize(observation)
                    tensor = torch.from_numpy(normalized).unsqueeze(0)
                    with torch.no_grad():
                        residual = (
                            model.deterministic_action(tensor)
                            .squeeze(0)
                            .cpu()
                            .numpy()
                        )
                residual_magnitudes.append(
                    float(np.mean(np.abs(residual)))
                )
                observation, reward, done, info = client.step(residual)
                arm_positions = _arm_positions(observation)
                minimum_arm_positions = np.minimum(
                    minimum_arm_positions, arm_positions
                )
                maximum_arm_positions = np.maximum(
                    maximum_arm_positions, arm_positions
                )
                episode_reward += reward
                final_info = info
                total_steps += 1

                projection = np.asarray(
                    info.get("projection", np.zeros(8)),
                    dtype=np.float32,
                )
                projection_max = float(np.max(np.abs(projection)))
                maximum_projection = max(
                    maximum_projection,
                    projection_max,
                )
                if projection_max > args.projection_tolerance:
                    safety_steps += 1
                    episode_safety_steps += 1
                    for joint_name, joint_info in info.get(
                            "safety_info", {}).items():
                        reason = str(joint_info.get("reason", "safe"))
                        raw_command = float(joint_info.get("cmd_raw", 0.0))
                        safe_command = float(joint_info.get(
                            "cmd_safe", raw_command
                        ))
                        if (
                                reason != "safe"
                                and abs(raw_command - safe_command)
                                > args.projection_tolerance):
                            safety_reasons[reason] += 1
                            episode_safety_reasons[reason] += 1
                            joint_reason = "{}:{}".format(
                                joint_name, reason
                            )
                            safety_joint_reasons[joint_reason] += 1
                            episode_safety_joint_reasons[joint_reason] += 1
                phase = str(info.get("residual_phase", "UNKNOWN"))
                subgoal_type = str(info.get(
                    "subgoal_type_name", "UNKNOWN"
                ))
                phase_steps[phase] += 1
                episode_phase_steps[phase] += 1
                subgoal_type_steps[subgoal_type] += 1
                episode_subgoal_type_steps[subgoal_type] += 1
                if bool(info.get("arm_finish_fallback_used", False)):
                    fallback_steps += 1
                    episode_fallback_steps += 1
                joint_diagnostics = dict(
                    info.get("joint_limit_diagnostics", {})
                )
                if not joint_diagnostics:
                    joint_diagnostics = dict(
                        info.get("teacher_diagnostics", {})
                    )
                for joint_name in joint_diagnostics.get(
                        "blocked_joints",
                        joint_diagnostics.get("joint_limit_blocked", [])):
                    joint_name = str(joint_name)
                    blocked_joint_steps[joint_name] += 1
                    episode_blocked_joint_steps[joint_name] += 1
                avoidance_action = np.asarray(
                    joint_diagnostics.get(
                        "joint_limit_avoidance_action", np.zeros(6)
                    ),
                    dtype=np.float32,
                )
                avoidance_magnitude = float(
                    np.max(np.abs(avoidance_action))
                )
                if (
                        bool(joint_diagnostics.get(
                            "joint_limit_avoidance_applied", False
                        ))
                        or avoidance_magnitude > 1.0e-8):
                    avoidance_steps += 1
                    episode_avoidance_steps += 1
                maximum_avoidance = max(
                    maximum_avoidance, avoidance_magnitude
                )
                episode_maximum_avoidance = max(
                    episode_maximum_avoidance,
                    avoidance_magnitude,
                )
                if done:
                    break

            success = bool(final_info.get("success", False))
            collision = bool(final_info.get("collision", False))
            timeout = bool(final_info.get("timeout", False))
            tf_ok = bool(final_info.get("tf_ok", True))
            final_distance = float(
                final_info.get("ee_distance", float("nan"))
            )
            successes += int(success)
            category_successes[category] += int(success)
            collisions += int(collision)
            timeouts += int(timeout)
            tf_failures += int(not tf_ok)
            final_distances.append(final_distance)
            rewards.append(episode_reward)
            episode_results.append({
                "episode": int(episode),
                "scenario_id": scenario_id,
                "category": category,
                "step_budget": int(episode_budget),
                "kinematic_lower_bound_steps": (
                    None if scenario is None
                    else scenario.get("kinematic_lower_bound_steps")
                ),
                "budget_to_lower_bound_ratio": (
                    None
                    if scenario is None
                    or scenario.get("kinematic_lower_bound_steps") is None
                    else float(episode_budget)
                    / float(scenario["kinematic_lower_bound_steps"])
                ),
                "steps": int(step),
                "reward": float(episode_reward),
                "success": success,
                "collision": collision,
                "timeout": timeout,
                "tf_ok": tf_ok,
                "final_distance": final_distance,
                "safety_steps": int(episode_safety_steps),
                "safety_reasons": dict(episode_safety_reasons),
                "safety_joint_reasons": dict(
                    episode_safety_joint_reasons
                ),
                "phase_steps": dict(episode_phase_steps),
                "subgoal_type_steps": dict(episode_subgoal_type_steps),
                "arm_finish_fallback_steps": int(
                    episode_fallback_steps
                ),
                "blocked_joint_steps": dict(
                    episode_blocked_joint_steps
                ),
                "joint_limit_avoidance_steps": int(
                    episode_avoidance_steps
                ),
                "maximum_joint_limit_avoidance": float(
                    episode_maximum_avoidance
                ),
                "initial_arm_positions": _json_list(
                    initial_arm_positions
                ),
                "final_arm_positions": _json_list(arm_positions),
                "minimum_arm_positions": _json_list(
                    minimum_arm_positions
                ),
                "maximum_arm_positions": _json_list(
                    maximum_arm_positions
                ),
                "final_base_pose": _json_list(
                    _base_pose(observation)
                ),
                "target_arm_positions": _json_list(
                    reset_info.get("target_arm_positions", [])
                ),
                "final_base_goal_xy": _json_list(
                    reset_info.get("final_base_goal_xy", [])
                ),
            })
            print(
                "residual_episode={} scenario_id={} category={} steps={} "
                "reward={:.4f} success={} "
                "collision={} timeout={} tf_ok={} final_distance={:.4f} "
                "safety_steps={} avoidance_steps={} fallback_steps={} "
                "phases={} safety_joints={}".format(
                    episode,
                    scenario_id,
                    category,
                    step,
                    episode_reward,
                    success,
                    collision,
                    timeout,
                    tf_ok,
                    final_distance,
                    episode_safety_steps,
                    episode_avoidance_steps,
                    episode_fallback_steps,
                    dict(episode_phase_steps),
                    dict(episode_safety_joint_reasons),
                )
            )
    finally:
        client.close()

    success_rate = float(successes) / float(episode_count)
    safety_rate = float(safety_steps) / float(max(total_steps, 1))
    gate_pass = bool(
        success_rate >= args.min_success_rate
        and collisions <= args.max_collisions
        and timeouts <= args.max_timeouts
        and tf_failures == 0
        and safety_rate <= args.max_safety_rate
    )
    mean_reward = float(np.mean(rewards))
    mean_final_distance = float(np.mean(final_distances))
    mean_residual = float(np.mean(residual_magnitudes))
    print(
        "residual_evaluation episodes={} successes={} success_rate={:.3f} "
        "collisions={} timeouts={} tf_failures={} mean_reward={:.4f} "
        "mean_final_distance={:.4f} residual_abs={:.6f} "
        "safety_steps={} safety_rate={:.4f} max_projection={:.6f} "
        "safety_reasons={} avoidance_steps={} max_avoidance={:.6f} "
        "blocked_joints={} fallback_steps={} phases={} safety_joints={}".format(
            episode_count,
            successes,
            success_rate,
            collisions,
            timeouts,
            tf_failures,
            mean_reward,
            mean_final_distance,
            mean_residual,
            safety_steps,
            safety_rate,
            maximum_projection,
            dict(safety_reasons),
            avoidance_steps,
            maximum_avoidance,
            dict(blocked_joint_steps),
            fallback_steps,
            dict(phase_steps),
            dict(safety_joint_reasons),
        )
    )
    category_metrics = {}
    for category in sorted(category_totals):
        count = int(category_totals[category])
        category_success = int(category_successes[category])
        category_rate = (
            float(category_success) / float(max(count, 1))
        )
        category_metrics[category] = {
            "episodes": count,
            "successes": category_success,
            "success_rate": category_rate,
        }
        print(
            "residual_category category={} episodes={} successes={} "
            "success_rate={:.3f}".format(
                category,
                count,
                category_success,
                category_rate,
            )
        )
    metrics = {
        "format_version": 1,
        "checkpoint": args.checkpoint,
        "zero_residual": bool(args.zero_residual),
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_update": checkpoint_update,
        "scenario_set": args.scenario_set,
        "scenario_split": args.scenario_split,
        "scenario_step_budget": scenario_step_budget,
        "episodes": int(episode_count),
        "successes": int(successes),
        "success_rate": success_rate,
        "collisions": int(collisions),
        "timeouts": int(timeouts),
        "tf_failures": int(tf_failures),
        "mean_reward": mean_reward,
        "mean_final_distance": mean_final_distance,
        "mean_residual_abs": mean_residual,
        "total_steps": int(total_steps),
        "safety_steps": int(safety_steps),
        "safety_rate": safety_rate,
        "maximum_projection": float(maximum_projection),
        "safety_reasons": dict(safety_reasons),
        "safety_joint_reasons": dict(safety_joint_reasons),
        "blocked_joint_steps": dict(blocked_joint_steps),
        "joint_limit_avoidance_steps": int(avoidance_steps),
        "maximum_joint_limit_avoidance": float(maximum_avoidance),
        "arm_finish_fallback_steps": int(fallback_steps),
        "phase_steps": dict(phase_steps),
        "subgoal_type_steps": dict(subgoal_type_steps),
        "categories": category_metrics,
        "episode_results": episode_results,
        "gate_pass": gate_pass,
    }
    if args.metrics_output:
        _write_metrics(args.metrics_output, metrics)
        print("residual_metrics_saved={}".format(args.metrics_output))
    print("residual_gate_pass={}".format(gate_pass))
    if not gate_pass:
        raise SystemExit(1)


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--zero-residual", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5563)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--scenario-set")
    parser.add_argument(
        "--scenario-split",
        choices=["all", "train", "validation", "test"],
        default="test",
    )
    parser.add_argument(
        "--scenario-validation-fraction", type=float, default=0.10
    )
    parser.add_argument(
        "--scenario-test-fraction", type=float, default=0.20
    )
    parser.add_argument("--scenario-split-seed", type=int, default=123)
    parser.add_argument("--scenario-step-budget", type=int, default=0)
    parser.add_argument(
        "--scenario-categories",
        nargs="+",
        help="optional category allow-list applied before splitting",
    )
    parser.add_argument(
        "--scenario-ids",
        nargs="+",
        help="optional scenario-id allow-list applied after split selection",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--min-success-rate", type=float, default=0.90)
    parser.add_argument("--max-collisions", type=int, default=0)
    parser.add_argument("--max-timeouts", type=int, default=0)
    parser.add_argument("--max-safety-rate", type=float, default=0.05)
    parser.add_argument("--projection-tolerance", type=float, default=1e-6)
    parser.add_argument("--metrics-output")
    return parser.parse_args()


def _filter_scenario_ids(scenarios, scenario_ids):
    if not scenario_ids:
        return list(scenarios)
    requested = set(str(value) for value in scenario_ids)
    selected = [
        scenario for scenario in scenarios
        if str(scenario.get("scenario_id")) in requested
    ]
    found = set(str(item.get("scenario_id")) for item in selected)
    missing = sorted(requested - found)
    if missing:
        raise ValueError(
            "scenario ids are not present in the selected split: {}".format(
                missing
            )
        )
    return selected


def _arm_positions(observation):
    observation = np.asarray(observation, dtype=np.float32)
    sensor = observation[FusedLowLevelState.SENSOR_SLICE]
    joint_positions = sensor[FusedLowLevelState.JOINT_POSITION_SLICE]
    return np.asarray(joint_positions[4:10], dtype=np.float32).copy()


def _base_pose(observation):
    observation = np.asarray(observation, dtype=np.float32)
    sensor = observation[FusedLowLevelState.SENSOR_SLICE]
    joint_positions = sensor[FusedLowLevelState.JOINT_POSITION_SLICE]
    return np.asarray(joint_positions[0:3], dtype=np.float32).copy()


def _arm_positions_from_reset(reset_info, observation):
    reset_positions = reset_info.get("initial_arm_positions")
    if reset_positions is None:
        return _arm_positions(observation)
    reset_positions = np.asarray(reset_positions, dtype=np.float32)
    if reset_positions.shape != (6,):
        return _arm_positions(observation)
    return reset_positions


def _json_list(values):
    return [float(value) for value in np.asarray(values).reshape(-1)]


def _write_metrics(path, metrics):
    expanded = os.path.abspath(os.path.expanduser(path))
    directory = os.path.dirname(expanded)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(expanded, "w") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
