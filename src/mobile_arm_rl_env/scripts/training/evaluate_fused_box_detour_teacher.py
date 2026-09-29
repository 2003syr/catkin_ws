#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Closed-loop teacher evaluation for the single-box fused task."""

from __future__ import division, print_function

import collections
import os
import sys

import numpy as np
import rospy

# ``rosrun`` executes this file from ``scripts/training`` rather than from
# the package root.  Add the scripts directory explicitly so the same module
# imports work for both rosrun and the project virtual-environment command.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_low_env_client import FusedLowEnvironmentClient
from hrl.fused_low_level import FusedLowLevelState


def _action_mask_phase(observation):
    mask = np.asarray(
        observation[FusedLowLevelState.ACTION_MASK_SLICE],
        dtype=np.float32,
    )
    for phase in ("BASE_APPROACH", "COORDINATED", "ARM_FINISH"):
        expected = FusedLowLevelState.action_mask_for_phase(phase)
        if np.allclose(mask, expected, atol=1.0e-6):
            return phase
    return "UNKNOWN"


def main():
    rospy.init_node("evaluate_fused_box_detour_teacher")
    episodes = int(rospy.get_param("~episodes", 10))
    minimum_success_rate = float(
        rospy.get_param("~minimum_success_rate", 0.80)
    )
    maximum_collisions = int(
        rospy.get_param("~maximum_collisions", 0)
    )
    log_interval = int(rospy.get_param("~log_interval", 25))
    side = float(rospy.get_param("~evaluation_detour_side", 1.0))
    host = str(rospy.get_param("~host", "127.0.0.1"))
    port = int(rospy.get_param("~port", 5564))
    socket_timeout = float(rospy.get_param("~socket_timeout", 120.0))
    max_steps = int(rospy.get_param("~max_steps", 1200))
    if episodes <= 0 or log_interval <= 0:
        raise ValueError("episodes and log_interval must be positive")

    client = FusedLowEnvironmentClient(
        host=host,
        port=port,
        timeout=socket_timeout,
    )
    successes = 0
    collisions = 0
    arm_contact_episodes = 0
    timeouts = 0
    total_steps = 0
    overlap_steps = 0
    safety_steps = 0
    final_distances = []
    final_path_remaining = []
    final_base_position_errors = []
    final_base_yaw_errors = []
    side_counts = collections.Counter()
    side_successes = collections.Counter()
    side_collisions = collections.Counter()
    action_mask_phases = collections.Counter()

    try:
        for episode_index in range(episodes):
            episode_side = side
            if abs(side) < 0.5:
                episode_side = 1.0 if episode_index % 2 == 0 else -1.0
            observation = client.reset(
                scenario={"detour_side": episode_side},
                max_steps=max_steps,
            )
            reset_info = dict(client.last_reset_info)
            initial_distance = float(reset_info["initial_ee_distance"])
            info = {}
            episode_steps = 0
            while not rospy.is_shutdown():
                executed_phase = _action_mask_phase(observation)
                action_mask_phases[executed_phase] += 1
                action = client.last_teacher_action.copy()
                observation, reward, done, info = client.step(action)
                safe_action = np.asarray(
                    info.get("safe_fused_action", action),
                    dtype=np.float32,
                )
                safety_active = bool(
                    np.max(np.abs(safe_action - action)) > 0.01
                )
                base_active = bool(
                    abs(float(safe_action[0])) > 0.05
                    or abs(float(safe_action[1])) > 0.05
                )
                arm_active = bool(
                    np.max(np.abs(safe_action[2:8])) > 0.02
                )
                overlap_steps += int(base_active and arm_active)
                safety_steps += int(safety_active)
                total_steps += 1
                episode_steps += 1
                episode_step = episode_steps
                if (
                        episode_step == 1
                        or episode_step % log_interval == 0
                        or done):
                    rospy.loginfo(
                        "box_teacher_step episode=%d step=%d side=%+.0f "
                        "distance=%.4f path_remaining=%.4f "
                        "goal=%s action=%s collision=%s phase=%s "
                        "arm_distance=%.4f arm_blend=%.3f "
                        "joint_limit_blocked=%s arm_hold=%s path_index=%d "
                        "passed_box=%s base_pose_error=(%.4f,%.4f) "
                        "terminal_pose_stage=%s terminal_stable=%d "
                        "arm_release_ready=%s "
                        "terminal_translation_refinement=%s "
                        "terminal_yaw_alignment=%s arm_stall=%d",
                        episode_index + 1,
                        episode_step,
                        episode_side,
                        float(info.get("dist", float("nan"))),
                        float(info.get("path_remaining", float("nan"))),
                        np.round(info.get("base_goal_xy", [0.0, 0.0]), 3).tolist(),
                        np.round(action, 3).tolist(),
                        str(info.get("collision", False)),
                        str(info.get("teacher_diagnostics", {}).get(
                            "phase", "unknown"
                        )),
                        float(info.get("teacher_diagnostics", {}).get(
                            "arm_distance", float("nan")
                        )),
                        float(info.get("teacher_diagnostics", {}).get(
                            "arm_blend", float("nan")
                        )),
                        str(info.get("teacher_diagnostics", {}).get(
                            "joint_limit_blocked", []
                        )),
                        str(info.get("arm_obstacle_hold", False)),
                        int(info.get("box_path_index", -1)),
                        str(info.get("teacher_diagnostics", {}).get(
                            "box_passed", False
                        )),
                        float(info.get(
                            "final_base_position_error", float("nan")
                        )),
                        float(info.get(
                            "final_base_yaw_error", float("nan")
                        )),
                        str(info.get("teacher_diagnostics", {}).get(
                            "terminal_pose_stage", "unknown"
                        )),
                        int(info.get("teacher_diagnostics", {}).get(
                            "terminal_pose_stable_count", 0
                        )),
                        str(info.get("teacher_diagnostics", {}).get(
                            "arm_box_release_ready", False
                        )),
                        str(info.get("teacher_diagnostics", {}).get(
                            "terminal_translation_refinement", False
                        )),
                        str(info.get("teacher_diagnostics", {}).get(
                            "terminal_yaw_alignment", False
                        )),
                        int(info.get("teacher_diagnostics", {}).get(
                            "arm_finish_stall_cycles", 0
                        )),
                    )
                if done:
                    break

            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            arm_contact = bool(info.get("arm_contact", False))
            timeout = bool(info.get("timeout", False))
            successes += int(success)
            collisions += int(collision)
            arm_contact_episodes += int(arm_contact)
            timeouts += int(timeout)
            side_counts["upper" if episode_side > 0.0 else "lower"] += 1
            side_name = "upper" if episode_side > 0.0 else "lower"
            side_successes[side_name] += int(success)
            side_collisions[side_name] += int(collision)
            final_distances.append(float(info.get("dist", float("nan"))))
            final_path_remaining.append(
                float(info.get("path_remaining", float("nan")))
            )
            final_base_position_errors.append(float(
                info.get("final_base_position_error", float("nan"))
            ))
            final_base_yaw_errors.append(abs(float(
                info.get("final_base_yaw_error", float("nan"))
            )))
            rospy.loginfo(
                "box_teacher_episode=%d side=%+.0f distance=%.4f->%.4f "
                "path_remaining=%.4f success=%s collision=%s "
                "arm_contact=%s timeout=%s steps=%d phase=%s "
                "arm_distance=%.4f base_pose_error=(%.4f,%.4f) "
                "terminal_pose_stage=%s terminal_stable=%d "
                "arm_release_ready=%s "
                "terminal_translation_refinement=%s "
                "terminal_yaw_alignment=%s arm_stall=%d",
                episode_index + 1,
                episode_side,
                initial_distance,
                final_distances[-1],
                final_path_remaining[-1],
                str(success),
                str(collision),
                str(arm_contact),
                str(timeout),
                episode_steps,
                str(info.get("teacher_diagnostics", {}).get(
                    "phase", "unknown"
                )),
                float(info.get("teacher_diagnostics", {}).get(
                    "arm_distance", float("nan")
                )),
                final_base_position_errors[-1],
                final_base_yaw_errors[-1],
                str(info.get("teacher_diagnostics", {}).get(
                    "terminal_pose_stage", "unknown"
                )),
                int(info.get("teacher_diagnostics", {}).get(
                    "terminal_pose_stable_count", 0
                )),
                str(info.get("teacher_diagnostics", {}).get(
                    "arm_box_release_ready", False
                )),
                str(info.get("teacher_diagnostics", {}).get(
                    "terminal_translation_refinement", False
                )),
                str(info.get("teacher_diagnostics", {}).get(
                    "terminal_yaw_alignment", False
                )),
                int(info.get("teacher_diagnostics", {}).get(
                    "arm_finish_stall_cycles", 0
                )),
            )
    finally:
        client.close()

    success_rate = successes / float(episodes)
    overlap_rate = overlap_steps / float(total_steps) if total_steps else 0.0
    safety_rate = safety_steps / float(total_steps) if total_steps else 0.0
    gate_pass = bool(
        success_rate >= minimum_success_rate
        and collisions <= maximum_collisions
    )
    rospy.loginfo(
        "box_teacher_evaluation episodes=%d successes=%d success_rate=%.3f "
        "collisions=%d timeouts=%d mean_final_distance=%.4f "
        "arm_contact_episodes=%d mean_path_remaining=%.4f "
        "mean_final_base_position_error=%.4f "
        "mean_final_base_yaw_error=%.4f "
        "overlap_rate=%.3f safety_rate=%.3f "
        "sides=%s side_successes=%s side_collisions=%s "
        "action_mask_phases=%s",
        episodes,
        successes,
        success_rate,
        collisions,
        timeouts,
        float(np.nanmean(final_distances)),
        arm_contact_episodes,
        float(np.nanmean(final_path_remaining)),
        float(np.nanmean(final_base_position_errors)),
        float(np.nanmean(final_base_yaw_errors)),
        overlap_rate,
        safety_rate,
        dict(side_counts),
        dict(side_successes),
        dict(side_collisions),
        dict(action_mask_phases),
    )
    if gate_pass:
        rospy.loginfo("box_teacher_gate_pass=True")
        return
    rospy.logerr(
        "box_teacher_gate_pass=False required_success_rate=%.3f "
        "maximum_collisions=%d",
        minimum_success_rate,
        maximum_collisions,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
