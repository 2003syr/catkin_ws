#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the obstacle-aware planar teacher in the dedicated scene."""

from __future__ import print_function

import os
import sys

import numpy as np
import rospy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.planar_base_training_env import PlanarBaseTrainingEnv
from training.planar_obstacle_teacher import RuleBasedPlanarAvoidancePolicy


DEFAULT_TARGETS = (
    (1.65, -0.13),
    (1.65, 0.10),
    (1.65, -0.40),
    (0.00, 1.20),
    (-1.10, -0.50),
)


def count_safety_events(safety_info):
    return sum(
        item.get("reason", "safe") != "safe"
        for item in safety_info.values()
    )


def parse_targets(value):
    targets = np.asarray(value, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets must have shape (N, 2)")
    if targets.shape[0] == 0 or not np.all(np.isfinite(targets)):
        raise ValueError("targets must be non-empty and finite")
    return targets


def main():
    rospy.init_node("evaluate_rule_based_planar_avoidance")
    episodes = int(rospy.get_param("~episodes", 10))
    seed = int(rospy.get_param("~seed", 123))
    required_success_rate = float(
        rospy.get_param("~required_success_rate", 0.80)
    )
    tracking_error_threshold = float(
        rospy.get_param("~tracking_error_threshold", 0.35)
    )
    tracking_failure_patience = int(
        rospy.get_param("~tracking_failure_patience", 20)
    )
    lateral_velocity_threshold = float(
        rospy.get_param("~lateral_velocity_threshold", 0.015)
    )
    arm_hold_tolerances = np.asarray(
        rospy.get_param(
            "~arm_hold_tolerances",
            [0.05, 0.05, 0.01, 0.05, 0.01, 0.05],
        ),
        dtype=np.float64,
    )
    targets = parse_targets(
        rospy.get_param("~targets", [list(value) for value in DEFAULT_TARGETS])
    )
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 <= required_success_rate <= 1.0:
        raise ValueError("required_success_rate must be in [0, 1]")
    if tracking_error_threshold <= 0.0:
        raise ValueError("tracking_error_threshold must be positive")
    if tracking_failure_patience <= 0:
        raise ValueError("tracking_failure_patience must be positive")
    if lateral_velocity_threshold <= 0.0:
        raise ValueError("lateral_velocity_threshold must be positive")
    if (
            arm_hold_tolerances.shape != (6,)
            or np.any(arm_hold_tolerances <= 0.0)):
        raise ValueError(
            "arm_hold_tolerances must contain six positive values"
        )

    env = PlanarBaseTrainingEnv(init_ros_node=False, seed=seed)
    policy = RuleBasedPlanarAvoidancePolicy(
        scan_clip=rospy.get_param("~scan_clip", 10.0),
        influence_distance=rospy.get_param(
            "~avoidance_influence_distance", 1.00
        ),
        attraction_gain=rospy.get_param("~attraction_gain", 1.00),
        repulsion_gain=rospy.get_param("~repulsion_gain", 1.40),
        tangential_gain=rospy.get_param("~tangential_gain", 1.50),
        emergency_margin=rospy.get_param("~emergency_margin", 0.12),
        emergency_tangent_gain=rospy.get_param(
            "~emergency_tangent_gain", 1.50
        ),
        goal_slow_radius=rospy.get_param("~goal_slow_radius", 0.30),
        previous_action_weight=rospy.get_param(
            "~previous_action_weight", 0.80
        ),
        side_clearance_hysteresis=rospy.get_param(
            "~side_clearance_hysteresis", 0.10
        ),
        detour_clearance=rospy.get_param("~detour_clearance", 0.90),
        detour_forward_distance=rospy.get_param(
            "~detour_forward_distance", 0.75
        ),
        detour_tolerance=rospy.get_param("~detour_tolerance", 0.10),
        detour_gain=rospy.get_param("~detour_gain", 1.50),
        turn_clearance=rospy.get_param("~turn_clearance", 1.15),
        reverse_speed=rospy.get_param("~reverse_speed", 0.50),
        minimum_turn_steps=rospy.get_param("~minimum_turn_steps", 60),
        minimum_pass_steps=rospy.get_param("~minimum_pass_steps", 240),
        sector_safety_distances=rospy.get_param(
            "~sector_safety_distances",
            [0.40, 0.40, 0.40, 0.40, 0.40],
        ),
    )

    successes = 0
    collisions = 0
    timeouts = 0
    total_safety_events = 0
    total_avoidance_steps = 0
    total_emergency_steps = 0
    total_detour_steps = 0
    total_detour_starts = 0
    total_detour_completions = 0
    control_failures = 0
    nonholonomic_failures = 0
    arm_hold_failures = 0
    maximum_arm_errors = np.zeros(6, dtype=np.float64)
    maximum_lateral_velocity = 0.0
    final_distances = []
    successful_steps = []
    episode_minimum_scans = []

    try:
        for episode in range(episodes):
            target_xy = targets[episode % targets.shape[0]]
            target = np.asarray(
                [target_xy[0], target_xy[1], env.target_z],
                dtype=np.float64,
            )
            observation = env.reset(target_position=target)
            policy.reset()
            start_distance = float(np.linalg.norm(observation[0:2]))
            episode_safety_events = 0
            avoidance_steps = 0
            emergency_steps = 0
            detour_steps = 0
            detour_starts = 0
            detour_completions = 0
            minimum_scan = float("inf")
            consecutive_tracking_failures = 0
            controller_tracking_failed = False
            nonholonomic_failed = False
            consecutive_lateral_failures = 0
            arm_hold_failed = False
            episode_arm_error = np.zeros(6, dtype=np.float64)
            maximum_tracking_error = 0.0
            info = {
                "success": False,
                "collision": False,
                "timeout": False,
                "distance": start_distance,
                "collision_source": "none",
            }

            for step in range(env.task.max_steps):
                action = policy.predict(observation)
                policy_info = policy.diagnostics()
                avoidance_steps += int(policy_info["avoidance_active"])
                emergency_steps += int(policy_info["emergency_active"])
                detour_steps += int(policy_info["detour_active"])
                detour_starts += int(policy_info["detour_started"])
                detour_completions += int(
                    policy_info["detour_completed"]
                )
                minimum_scan = min(
                    minimum_scan,
                    float(policy_info["minimum_scan"]),
                )
                observation, _, done, info = env.step(action)
                arm_positions = np.asarray(
                    info["structured_observation"]["joint_pos"][4:10],
                    dtype=np.float64,
                )
                arm_error = np.abs(
                    arm_positions - env.reset_arm_positions
                )
                episode_arm_error = np.maximum(
                    episode_arm_error,
                    arm_error,
                )
                if np.any(arm_error > arm_hold_tolerances):
                    arm_hold_failed = True
                    env.base_env.publish_zero_cmd()
                    break
                tracking_error = float(
                    info["observed_planar_tracking_error"]
                )
                lateral_velocity = abs(float(
                    info["measured_lateral_velocity"]
                ))
                maximum_lateral_velocity = max(
                    maximum_lateral_velocity,
                    lateral_velocity,
                )
                if lateral_velocity > lateral_velocity_threshold:
                    consecutive_lateral_failures += 1
                else:
                    consecutive_lateral_failures = 0
                if (
                        consecutive_lateral_failures
                        >= tracking_failure_patience):
                    nonholonomic_failed = True
                    env.base_env.publish_zero_cmd()
                    break
                maximum_tracking_error = max(
                    maximum_tracking_error,
                    tracking_error,
                )
                if (
                        float(np.max(np.abs(action))) >= 0.25
                        and tracking_error > tracking_error_threshold
                        and not policy_info["avoidance_active"]
                        and not policy_info["emergency_active"]
                        and policy_info["minimum_clearance"] > 0.15):
                    consecutive_tracking_failures += 1
                else:
                    consecutive_tracking_failures = 0
                episode_safety_events += count_safety_events(
                    info["safety_info"]
                )
                if (
                        consecutive_tracking_failures
                        >= tracking_failure_patience):
                    controller_tracking_failed = True
                    env.base_env.publish_zero_cmd()
                    break
                if done:
                    break

            success = bool(info["success"])
            collision = bool(info["collision"])
            timeout = bool(info["timeout"])
            final_distance = float(info["distance"])
            successes += int(success)
            collisions += int(collision)
            timeouts += int(timeout)
            control_failures += int(controller_tracking_failed)
            nonholonomic_failures += int(nonholonomic_failed)
            arm_hold_failures += int(arm_hold_failed)
            maximum_arm_errors = np.maximum(
                maximum_arm_errors,
                episode_arm_error,
            )
            total_safety_events += episode_safety_events
            total_avoidance_steps += avoidance_steps
            total_emergency_steps += emergency_steps
            total_detour_steps += detour_steps
            total_detour_starts += detour_starts
            total_detour_completions += detour_completions
            final_distances.append(final_distance)
            episode_minimum_scans.append(minimum_scan)
            if success:
                successful_steps.append(step + 1)

            rospy.loginfo(
                "avoidance_episode=%d target=[%.3f, %.3f] "
                "dist=%.4f->%.4f success=%s collision=%s source=%s "
                "timeout=%s steps=%d min_scan=%.3f avoidance_steps=%d "
                "emergency_steps=%d detour_steps=%d detours=%d/%d "
                "max_tracking_error=%.3f "
                "control_failed=%s nonholonomic_failed=%s "
                "max_lateral_velocity=%.4f "
                "arm_hold_failed=%s arm_error=%s "
                "safety_events=%d",
                episode + 1,
                target[0],
                target[1],
                start_distance,
                final_distance,
                success,
                collision,
                info.get("collision_source", "none"),
                timeout,
                step + 1,
                minimum_scan,
                avoidance_steps,
                emergency_steps,
                detour_steps,
                detour_completions,
                detour_starts,
                maximum_tracking_error,
                controller_tracking_failed,
                nonholonomic_failed,
                maximum_lateral_velocity,
                arm_hold_failed,
                np.round(episode_arm_error, 4).tolist(),
                episode_safety_events,
            )
    finally:
        env.stop()

    success_rate = float(successes) / float(episodes)
    mean_final_distance = float(np.mean(final_distances))
    mean_success_steps = (
        float(np.mean(successful_steps)) if successful_steps else float("nan")
    )
    rospy.loginfo(
        "avoidance_summary success=%d/%d rate=%.3f collisions=%d "
        "timeouts=%d mean_final_distance=%.4f mean_success_steps=%.1f "
        "minimum_scan=%.3f avoidance_steps=%d emergency_steps=%d "
        "detour_steps=%d detours=%d/%d control_failures=%d "
        "nonholonomic_failures=%d max_lateral_velocity=%.4f "
        "arm_hold_failures=%d max_arm_error=%s safety_events=%d "
        "required_rate=%.3f",
        successes,
        episodes,
        success_rate,
        collisions,
        timeouts,
        mean_final_distance,
        mean_success_steps,
        float(np.min(episode_minimum_scans)),
        total_avoidance_steps,
        total_emergency_steps,
        total_detour_steps,
        total_detour_completions,
        total_detour_starts,
        control_failures,
        nonholonomic_failures,
        maximum_lateral_velocity,
        arm_hold_failures,
        np.round(maximum_arm_errors, 4).tolist(),
        total_safety_events,
        required_success_rate,
    )
    if (
            success_rate < required_success_rate
            or collisions > 0
            or control_failures > 0
            or nonholonomic_failures > 0
            or arm_hold_failures > 0):
        rospy.logerr("Planar obstacle-avoidance acceptance failed")
        return 2
    rospy.loginfo("Planar obstacle-avoidance acceptance passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
