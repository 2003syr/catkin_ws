#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministically evaluate a pose-guided planar PPO checkpoint."""

import argparse
import os
import sys

import numpy as np

try:
    import torch
except ImportError as error:
    raise RuntimeError(
        "Direct planar evaluation requires Python 3 with PyTorch"
    ) from error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.hrl4in_low_actor_critic import (
    RunningObservationNormalizer,
)
from training.planar_direct_env_client import (
    PlanarDirectEnvironmentClient,
)
from training.planar_subgoal_actor_critic import (
    PlanarSubgoalActorCritic,
)


def main():
    args = _parse_arguments()
    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    policy_type = checkpoint.get("policy_type")
    bc_policy_types = (
        "planar_pose_guided_bc",
    )
    if policy_type not in (
            "planar_pose_guided_ppo",) + bc_policy_types:
        raise RuntimeError(
            "checkpoint is not a 62-D pose-guided planar policy"
        )
    checkpoint_observation_dim = checkpoint.get("observation_dim")
    if checkpoint_observation_dim is None:
        checkpoint_observation_dim = checkpoint.get(
            "normalizer", {}
        ).get("observation_dim")
    if (
            int(checkpoint_observation_dim or -1)
            != PlanarSubgoalActorCritic.OBS_DIM):
        raise RuntimeError(
            "checkpoint observation dimension is not 62"
        )
    if (
            policy_type == "planar_pose_guided_ppo"
            and checkpoint.get("teacher_type") != "none"):
        raise RuntimeError("PPO checkpoint unexpectedly uses a teacher")
    checkpoint_arguments = checkpoint.get("arguments", {})
    model = PlanarSubgoalActorCritic(
        hidden_sizes=checkpoint.get(
            "hidden_sizes",
            checkpoint_arguments.get("hidden_sizes", [128, 128]),
        ),
        initial_log_std=checkpoint_arguments.get(
            "initial_log_std", -1.0
        ),
    ).to(device)
    model.load_compatible_state_dict(
        checkpoint["model"],
        actor_only=(policy_type in bc_policy_types),
    )
    model.eval()
    normalizer = RunningObservationNormalizer(
        observation_dim=model.OBS_DIM,
        normalized_dim=57,
    )
    normalizer.load_state_dict(checkpoint["normalizer"])
    client = PlanarDirectEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )

    successes = 0
    collisions = 0
    timeouts = 0
    local_limits = 0
    shield_events = 0
    detour_locked_steps = 0
    detour_releases = 0
    detour_sign_switches = 0
    high_level_reason_counts = {}
    shield_reason_counts = {}
    shield_phase_counts = {}
    rewards = []
    final_distances = []
    minimum_distances = []
    distance_ratios = []
    steps_values = []
    scenario_results = {}
    try:
        for episode in range(args.episodes):
            observation, reset_info = client.reset(
                return_info=True,
                evaluation_index=episode,
            )
            scenario_label = str(
                reset_info.get("scenario_label", "unspecified")
            )
            start_distance = _safe_float(
                reset_info.get("initial_distance"),
                default=float(np.linalg.norm(
                    np.asarray(
                        reset_info["subgoal"],
                        dtype=np.float32,
                    )[0:2]
                )),
            )
            episode_reward = 0.0
            episode_shield_events = 0
            episode_detour_locked_steps = 0
            episode_detour_releases = 0
            episode_detour_sign_switches = 0
            episode_high_level_reasons = {}
            episode_shield_reasons = {}
            episode_shield_phases = {}
            previous_detour_sign = 0.0
            previous_high_level_update = -1
            minimum_distance = start_distance
            final_info = {}
            done = False
            for step in range(args.max_steps):
                normalized = normalizer.normalize(observation)
                tensor = torch.from_numpy(
                    normalized
                ).to(device).unsqueeze(0)
                with torch.no_grad():
                    action = model.deterministic_action(
                        tensor
                    ).squeeze(0).cpu().numpy()
                observation, reward, done, info = client.step(action)
                episode_reward += float(reward)
                minimum_distance = min(
                    minimum_distance,
                    float(info.get(
                        "distance",
                        minimum_distance,
                    )),
                )
                shield_events += int(bool(
                    info.get("shield_intervened", False)
                ))
                episode_shield_events += int(bool(
                    info.get("shield_intervened", False)
                ))
                locked = bool(info.get("detour_locked", False))
                episode_detour_locked_steps += int(locked)
                selection = info.get("high_level_selection", {})
                selection_reason = str(
                    selection.get("reason", "unknown")
                )
                if bool(info.get("shield_intervened", False)):
                    for shield_reason in info.get(
                            "shield_reasons",
                            []):
                        shield_reason = str(shield_reason)
                        episode_shield_reasons[shield_reason] = (
                            episode_shield_reasons.get(
                                shield_reason,
                                0,
                            ) + 1
                        )
                        shield_reason_counts[shield_reason] = (
                            shield_reason_counts.get(
                                shield_reason,
                                0,
                            ) + 1
                        )
                        phase_key = "{}:{}".format(
                            selection_reason,
                            shield_reason,
                        )
                        episode_shield_phases[phase_key] = (
                            episode_shield_phases.get(
                                phase_key,
                                0,
                            ) + 1
                        )
                        shield_phase_counts[phase_key] = (
                            shield_phase_counts.get(
                                phase_key,
                                0,
                            ) + 1
                        )
                update_index = int(selection.get("update_index", -1))
                if update_index != previous_high_level_update:
                    episode_high_level_reasons[selection_reason] = (
                        episode_high_level_reasons.get(
                            selection_reason,
                            0,
                        ) + 1
                    )
                    high_level_reason_counts[selection_reason] = (
                        high_level_reason_counts.get(
                            selection_reason,
                            0,
                        ) + 1
                    )
                    current_sign = float(
                        selection.get("detour_turn_sign", 0.0)
                    )
                    if bool(selection.get("detour_released", False)):
                        episode_detour_releases += 1
                    if (
                            previous_detour_sign != 0.0
                            and current_sign != 0.0
                            and current_sign != previous_detour_sign):
                        episode_detour_sign_switches += 1
                    if current_sign != 0.0:
                        previous_detour_sign = current_sign
                    previous_high_level_update = update_index
                final_info = info
                if done:
                    break
            success = bool(final_info.get("success", False))
            collision = bool(final_info.get("collision", False))
            timeout = bool(final_info.get("timeout", False))
            local_limit = bool(not done)
            final_distance = float(final_info.get(
                "distance",
                start_distance,
            ))
            successes += int(success)
            collisions += int(collision)
            timeouts += int(timeout)
            local_limits += int(local_limit)
            detour_locked_steps += episode_detour_locked_steps
            detour_releases += episode_detour_releases
            detour_sign_switches += episode_detour_sign_switches
            rewards.append(episode_reward)
            final_distances.append(final_distance)
            minimum_distances.append(minimum_distance)
            distance_ratios.append(
                final_distance / max(start_distance, 1.0e-6)
            )
            steps_values.append(step + 1)
            final_selection = final_info.get(
                "high_level_selection",
                {},
            )
            if not isinstance(final_selection, dict):
                final_selection = {}
            detour_current_progress = _safe_float(final_selection.get(
                "detour_current_progress",
                float("nan"),
            ))
            detour_obstacle_progress = _safe_float(final_selection.get(
                "detour_obstacle_progress",
                float("nan"),
            ))
            detour_required_progress = _safe_float(final_selection.get(
                "detour_required_progress",
                float("nan"),
            ))
            detour_escape_distance = _safe_float(final_selection.get(
                "detour_escape_distance",
                float("nan"),
            ))
            detour_escape_clearance = _safe_float(final_selection.get(
                "detour_escape_rotation_clearance",
                float("nan"),
            ))
            detour_escape_direct_clear = bool(final_selection.get(
                "detour_escape_direct_clear",
                False,
            ))
            final_raw_action = np.asarray(final_info.get(
                "raw_policy_action",
                [float("nan"), float("nan")],
            ), dtype=np.float64)
            final_safe_action = np.asarray(final_info.get(
                "safe_policy_action",
                [float("nan"), float("nan")],
            ), dtype=np.float64)
            final_local_subgoal = np.asarray(final_info.get(
                "local_subgoal_pose_body",
                [float("nan"), float("nan"), float("nan")],
            ), dtype=np.float64)
            scenario_values = scenario_results.setdefault(
                scenario_label,
                {
                    "episodes": 0,
                    "successes": 0,
                    "collisions": 0,
                    "distance_ratios": [],
                },
            )
            scenario_values["episodes"] += 1
            scenario_values["successes"] += int(success)
            scenario_values["collisions"] += int(collision)
            scenario_values["distance_ratios"].append(
                distance_ratios[-1]
            )
            print(
                "evaluation_episode={:02d} scenario={} "
                "distance={:.4f}->{:.4f} "
                "minimum={:.4f} ratio={:.3f} reward={:.4f} "
                "success={} collision={} collision_source={} "
                "collision_sector={} directional_clearance={:.4f} "
                "minimum_clearance={:.4f} timeout={} "
                "local_step_limit={} steps={} shield={} "
                "detour_locked_steps={} detour_releases={} "
                "detour_sign_switches={} "
                "detour_progress={:.3f} obstacle_progress={:.3f} "
                "release_progress={:.3f} "
                "escape_distance={:.3f} escape_clearance={:.3f} "
                "escape_direct_clear={} "
                "raw_action={} safe_action={} local_subgoal_pose={} "
                "heading_error={:.3f} "
                "shield_clearance=(front:{:.3f},rear:{:.3f},"
                "left:{:.3f},right:{:.3f},sweep:{:.3f}) "
                "shield_reasons={} shield_phases={} "
                "high_level_reasons={}".format(
                    episode + 1,
                    scenario_label,
                    start_distance,
                    final_distance,
                    minimum_distance,
                    distance_ratios[-1],
                    episode_reward,
                    success,
                    collision,
                    final_info.get("collision_source", "none"),
                    final_info.get(
                        "proximity_collision_sector",
                        "unknown",
                    ),
                    _safe_float(final_info.get(
                        "minimum_directional_clearance",
                        float("nan"),
                    )),
                    _safe_float(final_info.get(
                        "minimum_footprint_clearance",
                        float("nan"),
                    )),
                    timeout,
                    local_limit,
                    step + 1,
                    episode_shield_events,
                    episode_detour_locked_steps,
                    episode_detour_releases,
                    episode_detour_sign_switches,
                    detour_current_progress,
                    detour_obstacle_progress,
                    detour_required_progress,
                    detour_escape_distance,
                    detour_escape_clearance,
                    detour_escape_direct_clear,
                    np.round(final_raw_action, 4).tolist(),
                    np.round(final_safe_action, 4).tolist(),
                    np.round(final_local_subgoal, 4).tolist(),
                    _safe_float(final_info.get(
                        "path_heading_error",
                        float("nan"),
                    )),
                    _safe_float(final_info.get(
                        "shield_front_clearance",
                        float("nan"),
                    )),
                    _safe_float(final_info.get(
                        "shield_rear_clearance",
                        float("nan"),
                    )),
                    _safe_float(final_info.get(
                        "shield_left_clearance",
                        float("nan"),
                    )),
                    _safe_float(final_info.get(
                        "shield_right_clearance",
                        float("nan"),
                    )),
                    _safe_float(final_info.get(
                        "shield_sweep_clearance",
                        float("nan"),
                    )),
                    episode_shield_reasons,
                    episode_shield_phases,
                    episode_high_level_reasons,
                )
            )
    finally:
        client.close()

    success_rate = float(successes) / float(args.episodes)
    print(
        "direct_evaluation episodes={} successes={} "
        "success_rate={:.3f} collisions={} timeouts={} "
        "local_step_limits={} policy_type={} mean_reward={:.4f} "
        "mean_final_distance={:.4f} mean_minimum_distance={:.4f} "
        "mean_distance_ratio={:.3f} mean_steps={:.1f} "
        "shield_events={} detour_locked_steps={} "
        "detour_releases={} detour_sign_switches={} "
        "shield_reasons={} shield_phases={} "
        "high_level_reasons={}".format(
            args.episodes,
            successes,
            success_rate,
            collisions,
            timeouts,
            local_limits,
            policy_type,
            float(np.mean(rewards)),
            float(np.mean(final_distances)),
            float(np.mean(minimum_distances)),
            float(np.mean(distance_ratios)),
            float(np.mean(steps_values)),
            shield_events,
            detour_locked_steps,
            detour_releases,
            detour_sign_switches,
            shield_reason_counts,
            shield_phase_counts,
            high_level_reason_counts,
        )
    )
    for scenario_label, values in sorted(scenario_results.items()):
        print(
            "direct_scenario scenario={} episodes={} successes={} "
            "success_rate={:.3f} collisions={} "
            "mean_distance_ratio={:.3f}".format(
                scenario_label,
                values["episodes"],
                values["successes"],
                float(values["successes"])
                / float(max(values["episodes"], 1)),
                values["collisions"],
                float(np.mean(values["distance_ratios"])),
            )
        )
    passed = bool(
        success_rate >= args.min_success_rate
        and collisions <= args.max_collisions
        and local_limits == 0
    )
    print("direct_gate_pass={}".format(passed))
    if not passed:
        raise SystemExit(1)


def _safe_float(value, default=float("nan")):
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--min-success-rate", type=float, default=0.60)
    parser.add_argument("--max-collisions", type=int, default=0)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes and max-steps must be positive")
    if not 0.0 <= args.min_success_rate <= 1.0:
        raise ValueError("min-success-rate must be in [0, 1]")
    if args.max_collisions < 0:
        raise ValueError("max-collisions cannot be negative")
    return args


if __name__ == "__main__":
    main()
