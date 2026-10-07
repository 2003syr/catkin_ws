#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministic evaluation for the minimal two-layer high-level policy."""

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

from training.fused_basic_high_actor_critic import (  # noqa: E402
    FusedBasicHighActorCritic,
)
from training.fused_basic_high_env_client import (  # noqa: E402
    FusedBasicHighEnvironmentClient,
)
from training.fused_reach_scenarios import (  # noqa: E402
    FusedReachScenarioSampler,
    load_fused_reach_scenarios,
    scenario_category_counts,
    split_fused_reach_scenarios,
)
from training.hrl4in_low_actor_critic import (  # noqa: E402
    RunningObservationNormalizer,
)


STAGE_NAMES = ("DIRECT", "DETOUR", "TERMINAL")
TERMINAL_STAGE = 2


def main():
    args = _parse_arguments()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    payload = load_fused_reach_scenarios(args.scenario_set)
    scenarios = split_fused_reach_scenarios(
        payload["scenarios"],
        split=args.scenario_split,
        validation_fraction=args.scenario_validation_fraction,
        test_fraction=args.scenario_test_fraction,
        seed=args.scenario_split_seed,
    )
    scenarios = _basic_scenarios(scenarios, args.fixed_detour_side)
    if args.episodes > 0:
        scenarios = scenarios[:args.episodes]
    sampler = FusedReachScenarioSampler(
        scenarios, seed=args.scenario_split_seed, shuffle=False
    )
    budget = (
        args.scenario_step_budget
        if args.scenario_step_budget > 0
        else int(payload["recommended_step_budget"])
    )
    print(
        "basic_high_evaluation scenarios={} categories={} policy={}".format(
            len(scenarios),
            scenario_category_counts(scenarios),
            "teacher" if args.teacher_policy else args.checkpoint,
        )
    )

    model = None
    checkpoint = None
    normalizer = RunningObservationNormalizer(
        observation_dim=FusedBasicHighActorCritic.OBS_DIM,
        normalized_dim=FusedBasicHighActorCritic.NORMALIZED_DIM,
    )
    if not args.teacher_policy:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        _validate_checkpoint(checkpoint)
        model = FusedBasicHighActorCritic(
            hidden_sizes=checkpoint.get("hidden_sizes", [256, 256])
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        normalizer.load_state_dict(checkpoint["normalizer"])
        model.eval()

    stage_contract = (
        {} if checkpoint is None
        else checkpoint.get("stage_feasibility_contract", {})
    )
    stage_mask_enabled = bool(
        False
        if args.teacher_policy
        else (
            stage_contract.get("enabled", False)
            if args.stage_feasibility_mask is None
            else args.stage_feasibility_mask
        )
    )
    direct_escape_safety_rate = float(
        stage_contract.get("direct_escape_safety_rate", 0.80)
        if args.direct_escape_safety_rate is None
        else args.direct_escape_safety_rate
    )
    direct_escape_maximum_progress = float(
        stage_contract.get("direct_escape_maximum_progress", 0.05)
        if args.direct_escape_maximum_progress is None
        else args.direct_escape_maximum_progress
    )
    print(
        "basic_high_stage_policy_contract feasibility_mask={} "
        "stage_policy_frozen={} direct_escape_safety_rate={:.3f} "
        "direct_escape_maximum_progress={:.3f}".format(
            stage_mask_enabled,
            bool(stage_contract.get("freeze_stage_policy", False)),
            direct_escape_safety_rate,
            direct_escape_maximum_progress,
        )
    )

    client = FusedBasicHighEnvironmentClient(
        host=args.host, port=args.port, timeout=args.socket_timeout
    )
    evaluation_blend = args.terminal_student_blend
    if evaluation_blend is None and checkpoint is not None:
        curriculum = checkpoint.get("terminal_blend_curriculum", {})
        if bool(curriculum.get("enabled", False)):
            evaluation_blend = curriculum.get("active_blend")
    client.set_terminal_student_blend(evaluation_blend)
    print(
        "basic_high_terminal_student_blend requested={} server_default={}".format(
            evaluation_blend,
            client.metadata.get("terminal_student_blend", "unavailable"),
        )
    )
    print(
        "basic_high_diagnostics_contract={}".format(
            client.metadata.get("diagnostics_contract", "unavailable")
        )
    )
    totals = collections.Counter()
    categories = {}
    final_distances = []
    stage_counts = collections.Counter()
    requested_stage_counts = collections.Counter()
    executed_stage_counts = collections.Counter()
    option_termination_counts = collections.Counter()
    option_termination_by_stage = collections.Counter()
    stage_option_counts = collections.Counter()
    stage_low_steps = collections.Counter()
    stage_progress_sums = collections.Counter()
    stage_safety_steps = collections.Counter()
    stage_overlap_steps = collections.Counter()
    stage_residual_abs_sums = collections.Counter()
    stage_safety_projection_sums = collections.Counter()
    stage_base_projection_sums = collections.Counter()
    stage_ee_projection_sums = collections.Counter()
    stage_reward_term_sums = collections.Counter()
    reward_term_sums = collections.Counter()
    safety_reasons = collections.Counter()
    terminal_projection_max = 0.0
    terminal_student_alignment_sum = 0.0
    terminal_student_alignment_count = 0
    terminal_alignment_sum = 0.0
    terminal_alignment_count = 0
    terminal_pose_stage_counts = collections.Counter()
    stage_mask_counts = collections.Counter()
    try:
        for episode_index in range(len(scenarios)):
            scenario = sampler.next()
            episode_budget = int(scenario.get("step_budget") or budget)
            observation, unused_info = client.reset_with_info(
                scenario=scenario, max_steps=episode_budget
            )
            done = False
            episode_high_steps = 0
            episode_low_steps = 0
            episode_terminal_steps = 0
            episode_invalid_terminal_steps = 0
            episode_terminal_projection = 0.0
            episode_terminal_projection_max = 0.0
            episode_terminal_student_alignment_sum = 0.0
            episode_terminal_student_alignment_count = 0
            episode_terminal_alignment_sum = 0.0
            episode_terminal_alignment_count = 0
            episode_terminal_pose_stage_counts = collections.Counter()
            episode_requested_stages = collections.Counter()
            episode_executed_stages = collections.Counter()
            episode_terminations = collections.Counter()
            episode_safety_steps = 0
            episode_overlap_steps = 0
            episode_terminal_forced = 0
            episode_terminal_fallbacks = 0
            episode_terminal_norm_clips = 0
            episode_reward = 0.0
            episode_reward_terms = collections.Counter()
            episode_residual_abs_sum = 0.0
            episode_safety_projection_sum = 0.0
            episode_safety_projection_max = 0.0
            episode_base_projection_sum = 0.0
            episode_ee_projection_sum = 0.0
            while not done:
                if model is None:
                    action = client.teacher_action.copy()
                    stage = int(client.teacher_info.get("subgoal_type", 0))
                else:
                    normalized = normalizer.normalize(observation)
                    tensor = torch.from_numpy(normalized).to(device).unsqueeze(0)
                    stage_mask = None
                    if stage_mask_enabled:
                        stage_mask = model.stage_feasibility_mask(
                            tensor,
                            direct_safety_rate_threshold=(
                                direct_escape_safety_rate
                            ),
                            direct_maximum_progress=(
                                direct_escape_maximum_progress
                            ),
                        )
                        unavailable = (
                            (~stage_mask).sum(dim=0).detach().cpu().tolist()
                        )
                        for stage_index, count in enumerate(unavailable):
                            if int(count) > 0:
                                stage_mask_counts[
                                    "{}_blocked".format(
                                        _stage_name(stage_index)
                                    )
                                ] += int(count)
                    with torch.no_grad():
                        action_tensor, stage_tensor = (
                            model.deterministic_decision(
                                tensor, stage_mask=stage_mask
                            )
                        )
                    action = action_tensor.squeeze(0).cpu().numpy()
                    stage = int(stage_tensor.item())
                observation, reward, done, info = client.step(
                    action, subgoal_type=stage
                )
                episode_reward += float(reward)
                stage_counts[stage] += 1
                requested_stage = int(info.get(
                    "requested_subgoal_type", stage
                ))
                executed_stage = int(info.get("subgoal_type", stage))
                requested_stage_name = _stage_name(requested_stage)
                executed_stage_name = _stage_name(executed_stage)
                termination = str(info.get(
                    "option_termination", "unknown"
                ))
                step_low_steps = int(info.get("low_steps", 0))
                option_progress = float(info.get("option_progress", 0.0))
                step_safety_steps = int(info.get("safety_steps", 0))
                step_overlap_steps = int(info.get("overlap_steps", 0))
                step_residual_abs = _mean_abs(
                    info.get("low_residual_mean_abs", [])
                )
                step_safety_projection = _mean_abs(
                    info.get("low_safety_projection_mean_abs", [])
                )
                step_safety_projection_max = float(info.get(
                    "low_safety_projection_max", 0.0
                ))
                step_base_projection = float(info.get(
                    "base_command_projection", 0.0
                ))
                step_ee_projection = float(info.get(
                    "ee_command_projection", 0.0
                ))
                requested_stage_counts[requested_stage_name] += 1
                executed_stage_counts[executed_stage_name] += 1
                option_termination_counts[termination] += 1
                option_termination_by_stage[
                    "{}:{}".format(executed_stage_name, termination)
                ] += 1
                stage_option_counts[executed_stage_name] += 1
                stage_low_steps[executed_stage_name] += step_low_steps
                stage_progress_sums[executed_stage_name] += option_progress
                stage_safety_steps[
                    executed_stage_name
                ] += step_safety_steps
                stage_overlap_steps[
                    executed_stage_name
                ] += step_overlap_steps
                stage_residual_abs_sums[
                    executed_stage_name
                ] += step_residual_abs * step_low_steps
                stage_safety_projection_sums[
                    executed_stage_name
                ] += step_safety_projection * step_low_steps
                stage_base_projection_sums[
                    executed_stage_name
                ] += step_base_projection
                stage_ee_projection_sums[
                    executed_stage_name
                ] += step_ee_projection
                episode_requested_stages[requested_stage_name] += 1
                episode_executed_stages[executed_stage_name] += 1
                episode_terminations[termination] += 1
                episode_safety_steps += step_safety_steps
                episode_overlap_steps += step_overlap_steps
                episode_residual_abs_sum += (
                    step_residual_abs * step_low_steps
                )
                episode_safety_projection_sum += (
                    step_safety_projection * step_low_steps
                )
                episode_safety_projection_max = max(
                    episode_safety_projection_max,
                    step_safety_projection_max,
                )
                episode_base_projection_sum += step_base_projection
                episode_ee_projection_sum += step_ee_projection
                for reason, count in info.get("safety_reasons", {}).items():
                    safety_reasons[str(reason)] += int(count)
                reward_terms = info.get("high_reward", {}).get("terms", {})
                for term_name, term_value in reward_terms.items():
                    term_value = float(term_value)
                    reward_term_sums[str(term_name)] += term_value
                    stage_reward_term_sums[
                        "{}:{}".format(
                            executed_stage_name, str(term_name)
                        )
                    ] += term_value
                    episode_reward_terms[str(term_name)] += term_value
                episode_high_steps += 1
                episode_low_steps += step_low_steps
                episode_terminal_steps += int(
                    executed_stage == TERMINAL_STAGE
                )
                episode_invalid_terminal_steps += int(
                    bool(info.get("invalid_terminal", False))
                )
                projection = float(info.get(
                    "terminal_ee_projection", 0.0
                ))
                episode_terminal_projection += projection
                episode_terminal_projection_max = max(
                    episode_terminal_projection_max, projection
                )
                terminal_projection_max = max(
                    terminal_projection_max, projection
                )
                episode_terminal_forced += int(bool(info.get(
                    "terminal_stage_forced", False
                )))
                episode_terminal_fallbacks += int(bool(info.get(
                    "terminal_ee_progress_fallback", False
                )))
                episode_terminal_norm_clips += int(bool(info.get(
                    "terminal_ee_norm_clipped", False
                )))
                if executed_stage == TERMINAL_STAGE:
                    pose_stage = str(info.get(
                        "terminal_pose_stage", "UNKNOWN"
                    ))
                    terminal_pose_stage_counts[pose_stage] += 1
                    episode_terminal_pose_stage_counts[pose_stage] += 1
                    student_alignment = float(info.get(
                        "terminal_ee_student_alignment_cosine", 0.0
                    ))
                    executed_alignment = float(info.get(
                        "terminal_ee_executed_alignment_cosine",
                        info.get("terminal_ee_alignment_cosine", 0.0),
                    ))
                    if np.isfinite(student_alignment):
                        terminal_student_alignment_sum += student_alignment
                        terminal_student_alignment_count += 1
                        episode_terminal_student_alignment_sum += (
                            student_alignment
                        )
                        episode_terminal_student_alignment_count += 1
                    if np.isfinite(executed_alignment):
                        terminal_alignment_sum += executed_alignment
                        terminal_alignment_count += 1
                        episode_terminal_alignment_sum += executed_alignment
                        episode_terminal_alignment_count += 1
            category = str(scenario.get("category", "unknown"))
            category_counts = categories.setdefault(
                category, collections.Counter()
            )
            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            timeout = bool(info.get("timeout", False))
            distance = float(info.get("dist", float("inf")))
            totals["episodes"] += 1
            totals["successes"] += int(success)
            totals["collisions"] += int(collision)
            totals["timeouts"] += int(timeout)
            totals["high_steps"] += episode_high_steps
            totals["low_steps"] += episode_low_steps
            totals["terminal_steps"] += episode_terminal_steps
            totals["invalid_terminal_steps"] += (
                episode_invalid_terminal_steps
            )
            totals["terminal_projection"] += (
                episode_terminal_projection
            )
            totals["safety_steps"] += episode_safety_steps
            totals["overlap_steps"] += episode_overlap_steps
            totals["residual_abs_sum"] += episode_residual_abs_sum
            totals["safety_projection_sum"] += (
                episode_safety_projection_sum
            )
            totals["safety_projection_max"] = max(
                float(totals["safety_projection_max"]),
                episode_safety_projection_max,
            )
            totals["base_projection_sum"] += episode_base_projection_sum
            totals["ee_projection_sum"] += episode_ee_projection_sum
            totals["reward"] += episode_reward
            totals["terminal_forced_steps"] += episode_terminal_forced
            totals["terminal_progress_fallbacks"] += (
                episode_terminal_fallbacks
            )
            totals["terminal_norm_clips"] += episode_terminal_norm_clips
            category_counts["episodes"] += 1
            category_counts["successes"] += int(success)
            final_distances.append(distance)
            print(
                "basic_high_episode={} scenario_id={} category={} "
                "success={} collision={} timeout={} high_steps={} "
                "low_steps={} final_distance={:.4f} path_complete={} "
                "base_position_error={:.4f} base_yaw_error={:.4f} "
                "terminal_pose_stage={} stable_count={} "
                "terminal_steps={} invalid_terminal_steps={} "
                "terminal_projection={:.5f}".format(
                    episode_index + 1,
                    scenario.get("scenario_id"),
                    category,
                    success,
                    collision,
                    timeout,
                    episode_high_steps,
                    episode_low_steps,
                    distance,
                    bool(info.get("path_complete", False)),
                    float(info.get(
                        "final_base_position_error",
                        info.get("base_position_error", float("nan")),
                    )),
                    float(info.get(
                        "final_base_yaw_error",
                        info.get("base_yaw_error", float("nan")),
                    )),
                    str(info.get("terminal_pose_stage", "UNKNOWN")),
                    int(info.get("terminal_pose_stable_count", 0)),
                    episode_terminal_steps,
                    episode_invalid_terminal_steps,
                    episode_terminal_projection,
                )
            )
            print(
                "basic_high_episode_diagnostics episode={} "
                "requested_stages={} executed_stages={} terminations={} "
                "mean_low_steps_per_high={:.2f} safety_steps={} "
                "safety_rate={:.4f} overlap_steps={} overlap_rate={:.4f} "
                "terminal_forced={} terminal_projection_mean={:.5f} "
                "terminal_projection_max={:.5f} terminal_fallbacks={} "
                "terminal_norm_clips={} residual_abs={:.6f} "
                "safety_projection_abs={:.6f} "
                "safety_projection_max={:.6f} "
                "base_command_projection_mean={:.6f} "
                "ee_command_projection_mean={:.6f} "
                "terminal_pose_stages={} "
                "terminal_student_alignment_mean={:.6f} "
                "terminal_executed_alignment_mean={:.6f} reward={:.4f} "
                "reward_terms={}".format(
                    episode_index + 1,
                    dict(episode_requested_stages),
                    dict(episode_executed_stages),
                    dict(episode_terminations),
                    float(episode_low_steps) / float(max(
                        episode_high_steps, 1
                    )),
                    episode_safety_steps,
                    float(episode_safety_steps) / float(max(
                        episode_low_steps, 1
                    )),
                    episode_overlap_steps,
                    float(episode_overlap_steps) / float(max(
                        episode_low_steps, 1
                    )),
                    episode_terminal_forced,
                    episode_terminal_projection / float(max(
                        episode_terminal_steps, 1
                    )),
                    episode_terminal_projection_max,
                    episode_terminal_fallbacks,
                    episode_terminal_norm_clips,
                    episode_residual_abs_sum / float(max(
                        episode_low_steps, 1
                    )),
                    episode_safety_projection_sum / float(max(
                        episode_low_steps, 1
                    )),
                    episode_safety_projection_max,
                    episode_base_projection_sum / float(max(
                        episode_high_steps, 1
                    )),
                    episode_ee_projection_sum / float(max(
                        episode_high_steps, 1
                    )),
                    dict(episode_terminal_pose_stage_counts),
                    episode_terminal_student_alignment_sum / float(max(
                        episode_terminal_student_alignment_count, 1
                    )),
                    episode_terminal_alignment_sum / float(max(
                        episode_terminal_alignment_count, 1
                    )),
                    episode_reward,
                    dict(episode_reward_terms),
                )
            )
    finally:
        client.close()

    metrics = {
        "stage_feasibility_mask": bool(stage_mask_enabled),
        "stage_mask_counts": dict(stage_mask_counts),
        "terminal_student_blend": (
            None if evaluation_blend is None else float(evaluation_blend)
        ),
        "episodes": int(totals["episodes"]),
        "successes": int(totals["successes"]),
        "success_rate": float(totals["successes"])
        / float(max(totals["episodes"], 1)),
        "collisions": int(totals["collisions"]),
        "timeouts": int(totals["timeouts"]),
        "mean_final_distance": float(np.mean(final_distances)),
        "mean_reward": float(totals["reward"])
        / float(max(totals["episodes"], 1)),
        "high_steps": int(totals["high_steps"]),
        "low_steps": int(totals["low_steps"]),
        "terminal_steps": int(totals["terminal_steps"]),
        "invalid_terminal_steps": int(
            totals["invalid_terminal_steps"]
        ),
        "terminal_projection": float(totals["terminal_projection"]),
        "terminal_projection_mean": float(totals["terminal_projection"])
        / float(max(totals["terminal_steps"], 1)),
        "terminal_projection_max": float(terminal_projection_max),
        "terminal_student_alignment_cosine_mean": float(
            terminal_student_alignment_sum
        ) / float(max(terminal_student_alignment_count, 1)),
        "terminal_executed_alignment_cosine_mean": float(
            terminal_alignment_sum
        ) / float(max(terminal_alignment_count, 1)),
        "terminal_alignment_cosine_mean": float(terminal_alignment_sum)
        / float(max(terminal_alignment_count, 1)),
        "terminal_pose_stage_counts": dict(terminal_pose_stage_counts),
        "terminal_forced_steps": int(totals["terminal_forced_steps"]),
        "terminal_progress_fallbacks": int(
            totals["terminal_progress_fallbacks"]
        ),
        "terminal_norm_clips": int(totals["terminal_norm_clips"]),
        "stage_counts": dict(stage_counts),
        "requested_stage_counts": dict(requested_stage_counts),
        "executed_stage_counts": dict(executed_stage_counts),
        "option_termination_counts": dict(option_termination_counts),
        "option_termination_by_stage": dict(option_termination_by_stage),
        "mean_low_steps_by_stage": _counter_means(
            stage_low_steps, stage_option_counts
        ),
        "mean_option_progress_by_stage": _counter_means(
            stage_progress_sums, stage_option_counts
        ),
        "safety_rate_by_stage": _counter_means(
            stage_safety_steps, stage_low_steps
        ),
        "overlap_rate_by_stage": _counter_means(
            stage_overlap_steps, stage_low_steps
        ),
        "mean_residual_abs_by_stage": _counter_means(
            stage_residual_abs_sums, stage_low_steps
        ),
        "mean_safety_projection_abs_by_stage": _counter_means(
            stage_safety_projection_sums, stage_low_steps
        ),
        "mean_base_command_projection_by_stage": _counter_means(
            stage_base_projection_sums, stage_option_counts
        ),
        "mean_ee_command_projection_by_stage": _counter_means(
            stage_ee_projection_sums, stage_option_counts
        ),
        "mean_reward_terms": dict(
            (name, float(value) / float(max(totals["high_steps"], 1)))
            for name, value in reward_term_sums.items()
        ),
        "mean_reward_terms_by_stage": _stage_term_means(
            stage_reward_term_sums, stage_option_counts
        ),
        "safety_steps": int(totals["safety_steps"]),
        "safety_rate": float(totals["safety_steps"])
        / float(max(totals["low_steps"], 1)),
        "safety_reasons": dict(safety_reasons),
        "mean_residual_abs": float(totals["residual_abs_sum"])
        / float(max(totals["low_steps"], 1)),
        "mean_safety_projection_abs": float(
            totals["safety_projection_sum"]
        ) / float(max(totals["low_steps"], 1)),
        "safety_projection_max": float(totals["safety_projection_max"]),
        "mean_base_command_projection": float(
            totals["base_projection_sum"]
        ) / float(max(totals["high_steps"], 1)),
        "mean_ee_command_projection": float(totals["ee_projection_sum"])
        / float(max(totals["high_steps"], 1)),
        "overlap_steps": int(totals["overlap_steps"]),
        "overlap_rate": float(totals["overlap_steps"])
        / float(max(totals["low_steps"], 1)),
        "categories": dict(
            (name, dict(value)) for name, value in categories.items()
        ),
    }
    print("basic_high_evaluation {}".format(metrics))
    if args.metrics_output:
        with open(args.metrics_output, "w") as stream:
            json.dump(metrics, stream, indent=2, sort_keys=True)


def _stage_name(stage):
    stage = int(stage)
    if 0 <= stage < len(STAGE_NAMES):
        return STAGE_NAMES[stage]
    return "UNKNOWN_{}".format(stage)


def _counter_means(sums, counts):
    return dict(
        (
            name,
            float(sums.get(name, 0.0)) / float(max(count, 1)),
        )
        for name, count in counts.items()
    )


def _stage_term_means(sums, stage_counts):
    means = {}
    for key, value in sums.items():
        stage_name, unused_separator, term_name = str(key).partition(":")
        means.setdefault(stage_name, {})[term_name] = (
            float(value) / float(max(stage_counts.get(stage_name, 0), 1))
        )
    return means


def _mean_abs(value):
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return 0.0
    return float(np.mean(np.abs(array)))


def _basic_scenarios(scenarios, fixed_detour_side):
    result = []
    for source in scenarios:
        scenario = dict(source)
        if not bool(scenario.get("no_obstacle", False)):
            scenario["detour_side"] = float(fixed_detour_side)
        result.append(scenario)
    return result


def _validate_checkpoint(checkpoint):
    expected = {
        "observation_dim": FusedBasicHighActorCritic.OBS_DIM,
        "action_dim": FusedBasicHighActorCritic.ACTION_DIM,
        "subgoal_type_count": FusedBasicHighActorCritic.STAGE_COUNT,
        "policy_type": FusedBasicHighActorCritic.POLICY_TYPE,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError("basic checkpoint {} mismatch".format(key))


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
    parser.add_argument("--scenario-step-budget", type=int, default=0)
    parser.add_argument("--fixed-detour-side", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument(
        "--terminal-student-blend",
        type=float,
        default=None,
        help=(
            "override TERMINAL student authority for this evaluation; if "
            "omitted, use the checkpoint curriculum stage or server default"
        ),
    )
    stage_mask_group = parser.add_mutually_exclusive_group()
    stage_mask_group.add_argument(
        "--stage-feasibility-mask",
        dest="stage_feasibility_mask",
        action="store_true",
        help="enable the checkpoint-compatible semantic stage mask",
    )
    stage_mask_group.add_argument(
        "--no-stage-feasibility-mask",
        dest="stage_feasibility_mask",
        action="store_false",
        help="disable the semantic stage mask stored in the checkpoint",
    )
    parser.set_defaults(stage_feasibility_mask=None)
    parser.add_argument(
        "--direct-escape-safety-rate", type=float, default=None
    )
    parser.add_argument(
        "--direct-escape-maximum-progress", type=float, default=None
    )
    parser.add_argument("--metrics-output")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.teacher_policy):
        parser.error("supply exactly one of --checkpoint or --teacher-policy")
    if (
            args.terminal_student_blend is not None
            and not 0.0 <= args.terminal_student_blend <= 1.0):
        parser.error("--terminal-student-blend must be in [0, 1]")
    if (
            args.direct_escape_safety_rate is not None
            and not 0.0 <= args.direct_escape_safety_rate <= 1.0):
        parser.error("--direct-escape-safety-rate must be in [0, 1]")
    if (
            args.direct_escape_maximum_progress is not None
            and args.direct_escape_maximum_progress < 0.0):
        parser.error(
            "--direct-escape-maximum-progress must be non-negative"
        )
    return args


if __name__ == "__main__":
    main()
