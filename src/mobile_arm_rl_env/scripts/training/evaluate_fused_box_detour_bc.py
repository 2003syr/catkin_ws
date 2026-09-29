#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Closed-loop evaluation of a fused BC policy in the box-detour task.

This evaluator deliberately uses the same ``FusedBoxDetourTrainingEnv`` that
was used to collect the box teacher dataset.  The generic fused BC evaluator
is kept separate because it samples direct, obstacle-free targets.
"""

from __future__ import division, print_function

import collections
import os
import sys

import numpy as np
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.evaluate_fused_low_bc import (  # noqa: E402
    PolicyServiceClient,
    _active_safety_reasons,
    _validate_parameters,
)
from training.fused_box_detour_training_env import (  # noqa: E402
    FusedBoxDetourTrainingEnv,
)
from hrl.fused_low_level import FusedLowLevelState  # noqa: E402


def _observation_action_mask_phase(observation):
    """Decode the stage that governed the action about to be executed."""
    action_mask = np.asarray(
        observation[FusedLowLevelState.ACTION_MASK_SLICE],
        dtype=np.float32,
    )
    for phase in ("BASE_APPROACH", "COORDINATED", "ARM_FINISH"):
        expected = FusedLowLevelState.action_mask_for_phase(phase)
        if np.allclose(action_mask, expected, atol=1.0e-6):
            return phase
    return "UNKNOWN"


def main():
    rospy.init_node("evaluate_fused_box_detour_bc")
    episodes = int(rospy.get_param("~episodes", 10))
    minimum_success_rate = float(
        rospy.get_param("~minimum_success_rate", 0.80)
    )
    maximum_collisions = int(
        rospy.get_param("~maximum_collisions", 0)
    )
    minimum_overlap_rate = float(
        rospy.get_param("~minimum_overlap_rate", 0.10)
    )
    # The box teacher dataset reports an 8.4% intervention rate.  A 5%
    # threshold would reject a faithful clone before we have a box-specific
    # safety-tuning pass.
    maximum_safety_rate = float(
        rospy.get_param("~maximum_safety_rate", 0.20)
    )
    safety_delta_threshold = float(
        rospy.get_param("~safety_delta_threshold", 0.01)
    )
    log_interval = int(rospy.get_param("~log_interval", 25))
    detour_side = float(
        rospy.get_param("~evaluation_detour_side", 0.0)
    )
    max_steps = int(rospy.get_param("~max_steps", 1200))

    _validate_parameters(
        episodes,
        minimum_success_rate,
        maximum_collisions,
        minimum_overlap_rate,
        maximum_safety_rate,
        safety_delta_threshold,
        log_interval,
    )
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    policy = PolicyServiceClient(
        host=rospy.get_param("~policy_host", "127.0.0.1"),
        port=rospy.get_param("~policy_port", 5563),
        timeout=rospy.get_param("~policy_timeout", 5.0),
    )
    environment = FusedBoxDetourTrainingEnv(
        init_ros_node=False,
        seed=rospy.get_param("~seed", 100123),
        enable_teacher=True,
    )

    successes = 0
    collisions = 0
    timeouts = 0
    arm_contacts = 0
    total_steps = 0
    overlap_steps = 0
    safety_steps = 0
    final_distances = []
    final_path_remaining = []
    inference_times = []
    teacher_squared_error = np.zeros(8, dtype=np.float64)
    executed_teacher_squared_error = np.zeros(8, dtype=np.float64)
    absolute_filter_delta = np.zeros(8, dtype=np.float64)
    maximum_filter_delta = np.zeros(8, dtype=np.float64)
    action_gate_delta = np.zeros(8, dtype=np.float64)
    action_gate_steps = 0
    action_gate_components = 0
    safety_reasons = collections.Counter()
    action_gate_reasons = collections.Counter()
    action_mask_phases = collections.Counter()
    side_counts = collections.Counter()

    rospy.loginfo(
        "box_bc_checkpoint=%s policy_type=%s validation_weighted_mse=%s",
        str(policy.metadata.get("checkpoint", "unknown")),
        str(policy.metadata.get("policy_type", "unknown")),
        str(policy.metadata.get("validation_weighted_mse", "unknown")),
    )

    try:
        for episode_index in range(episodes):
            episode_side = detour_side
            if abs(detour_side) < 0.5:
                episode_side = 1.0 if episode_index % 2 == 0 else -1.0
            scenario = {
                "scenario_id": "box_bc_eval_{:03d}".format(
                    episode_index + 1
                ),
                "category": "box_detour",
                "detour_side": episode_side,
            }
            observation = environment.reset(
                scenario=scenario,
                max_steps=max_steps,
            )
            initial_distance = float(
                environment.last_reset_info["initial_ee_distance"]
            )
            episode_overlap = 0
            episode_safety = 0
            episode_reward = 0.0
            info = {}
            episode_steps = 0

            while not rospy.is_shutdown():
                # Count the mask in the observation used for this prediction.
                # The post-step info already describes the next waypoint and
                # otherwise loses short stages at waypoint transitions.
                prediction_mask_phase = _observation_action_mask_phase(
                    observation
                )
                action_mask_phases[prediction_mask_phase] += 1
                action, inference_ms = policy.predict(observation)
                teacher_action = environment.teacher_action(observation)
                observation, reward, done, info = environment.step(action)
                executed_action = np.asarray(
                    info.get("box_gated_action", action), dtype=np.float32
                )
                safe_action = np.asarray(
                    info["safe_fused_action"], dtype=np.float32
                )
                gate_delta = executed_action - action
                action_gate_delta += np.abs(gate_delta.astype(np.float64))
                gate_active = bool(
                    np.max(np.abs(gate_delta)) > safety_delta_threshold
                )
                action_gate_steps += int(gate_active)
                action_gate_components += int(np.sum(
                    np.abs(gate_delta) > safety_delta_threshold
                ))
                if gate_active:
                    for reason in info.get(
                            "box_action_gate_reasons", []):
                        action_gate_reasons[reason] += 1

                # Safety filtering is measured after the structural task gate;
                # otherwise phase constraints would be misreported as Gazebo
                # safety interventions.
                action_delta = safe_action - executed_action
                absolute_delta = np.abs(action_delta.astype(np.float64))
                absolute_filter_delta += absolute_delta
                maximum_filter_delta = np.maximum(
                    maximum_filter_delta, absolute_delta
                )
                safety_active = bool(
                    np.max(absolute_delta) > safety_delta_threshold
                )
                base_active = bool(
                    abs(float(safe_action[0])) > 0.05
                    or abs(float(safe_action[1])) > 0.05
                )
                arm_active = bool(
                    np.max(np.abs(safe_action[2:8])) > 0.02
                )
                overlap_active = bool(base_active and arm_active)
                total_steps += 1
                episode_steps += 1
                overlap_steps += int(overlap_active)
                safety_steps += int(safety_active)
                episode_overlap += int(overlap_active)
                episode_safety += int(safety_active)
                episode_reward += float(reward)
                inference_times.append(inference_ms)
                teacher_squared_error += (
                    action.astype(np.float64)
                    - np.asarray(teacher_action, dtype=np.float64)
                ) ** 2
                executed_teacher_squared_error += (
                    executed_action.astype(np.float64)
                    - np.asarray(teacher_action, dtype=np.float64)
                ) ** 2
                if safety_active:
                    reasons = _active_safety_reasons(
                        info.get("safety_info", {})
                    )
                    if not reasons:
                        reasons = ["ACTION_CLIPPED"]
                    for reason in reasons:
                        safety_reasons[reason] += 1

                if (
                        episode_steps == 1
                        or episode_steps % log_interval == 0
                        or done):
                    rospy.loginfo(
                        "box_bc_step episode=%d side=%+.0f step=%d "
                        "ee_distance=%.4f path_remaining=%.4f "
                        "path_index=%d goal=%s passed_box=%s "
                        "mask_phase=%s "
                        "action=%s gated=%s safe=%s "
                        "gate=%s gate_reasons=%s safety=%s overlap=%s",
                        episode_index + 1,
                        episode_side,
                        episode_steps,
                        float(info.get("dist", float("nan"))),
                        float(info.get("path_remaining", float("nan"))),
                        int(info.get("box_path_index", -1)),
                        np.round(info.get(
                            "base_goal_xy", [float("nan"), float("nan")]
                        ), 3).tolist(),
                        str(info.get("box_passed", False)),
                        prediction_mask_phase,
                        np.round(action, 3).tolist(),
                        np.round(executed_action, 3).tolist(),
                        np.round(safe_action, 3).tolist(),
                        str(gate_active),
                        str(info.get("box_action_gate_reasons", [])),
                        str(safety_active),
                        str(overlap_active),
                    )
                if done:
                    break

            success = bool(info.get("success", False))
            collision = bool(
                info.get("collision", False)
                or info.get("collision_names", [])
                or info.get("arm_contact", False)
            )
            timeout = bool(info.get("timeout", False))
            if not success and not collision and not timeout:
                timeout = True
            successes += int(success)
            collisions += int(collision)
            timeouts += int(timeout)
            arm_contacts += int(bool(info.get("arm_contact", False)))
            side_counts["upper" if episode_side > 0.0 else "lower"] += 1
            final_distance = float(info.get("dist", float("nan")))
            path_remaining = float(
                info.get("path_remaining", float("nan"))
            )
            final_distances.append(final_distance)
            final_path_remaining.append(path_remaining)
            rospy.loginfo(
                "box_bc_episode=%d side=%+.0f distance=%.4f->%.4f "
                "path_remaining=%.4f reward=%.4f success=%s "
                "collision=%s arm_contact=%s timeout=%s steps=%d "
                "overlap_steps=%d safety_steps=%d",
                episode_index + 1,
                episode_side,
                initial_distance,
                final_distance,
                path_remaining,
                episode_reward,
                str(success),
                str(collision),
                str(bool(info.get("arm_contact", False))),
                str(timeout),
                episode_steps,
                episode_overlap,
                episode_safety,
            )
    finally:
        try:
            environment.stop()
        finally:
            policy.close()

    success_rate = successes / float(episodes)
    overlap_rate = overlap_steps / float(total_steps) if total_steps else 0.0
    safety_rate = safety_steps / float(total_steps) if total_steps else 0.0
    teacher_mse = (
        teacher_squared_error / float(total_steps)
        if total_steps else np.full(8, float("nan"))
    )
    executed_teacher_mse = (
        executed_teacher_squared_error / float(total_steps)
        if total_steps else np.full(8, float("nan"))
    )
    action_gate_rate = (
        action_gate_steps / float(total_steps)
        if total_steps else 0.0
    )
    action_gate_component_rate = (
        action_gate_components / float(total_steps * 8)
        if total_steps else 0.0
    )
    gate_pass = bool(
        success_rate >= minimum_success_rate
        and collisions <= maximum_collisions
        and overlap_rate >= minimum_overlap_rate
        and safety_rate <= maximum_safety_rate
    )
    rospy.loginfo(
        "box_bc_evaluation episodes=%d successes=%d success_rate=%.3f "
        "collisions=%d arm_contacts=%d timeouts=%d "
        "mean_final_distance=%.4f mean_path_remaining=%.4f "
        "overlap_rate=%.3f safety_rate=%.3f action_gate_rate=%.3f "
        "action_gate_component_rate=%.3f action_gate_delta=%s "
        "teacher_action_mse=%s executed_teacher_action_mse=%s "
        "mean_inference_ms=%.3f sides=%s safety_reasons=%s "
        "action_gate_reasons=%s action_mask_phases=%s",
        episodes,
        successes,
        success_rate,
        collisions,
        arm_contacts,
        timeouts,
        float(np.nanmean(final_distances)),
        float(np.nanmean(final_path_remaining)),
        overlap_rate,
        safety_rate,
        action_gate_rate,
        action_gate_component_rate,
        np.round(
            action_gate_delta / float(total_steps)
            if total_steps else action_gate_delta,
            7,
        ).tolist(),
        np.round(teacher_mse, 7).tolist(),
        np.round(executed_teacher_mse, 7).tolist(),
        float(np.mean(inference_times)) if inference_times else float("nan"),
        dict(side_counts),
        dict(safety_reasons),
        dict(action_gate_reasons),
        dict(action_mask_phases),
    )
    if gate_pass:
        rospy.loginfo("box_bc_gate_pass=True")
        return
    rospy.logerr(
        "box_bc_gate_pass=False required_success_rate=%.3f "
        "maximum_collisions=%d minimum_overlap_rate=%.3f "
        "maximum_safety_rate=%.3f",
        minimum_success_rate,
        maximum_collisions,
        minimum_overlap_rate,
        maximum_safety_rate,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
