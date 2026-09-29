#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the obstacle-free coordinated base-arm teacher in Gazebo."""

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

from training.fused_low_training_env import FusedLowLevelTrainingEnv


def main():
    rospy.init_node("evaluate_fused_low_teacher")
    episodes = int(rospy.get_param("~episodes", 10))
    minimum_success_rate = float(
        rospy.get_param("~minimum_success_rate", 0.80)
    )
    maximum_collisions = int(
        rospy.get_param("~maximum_collisions", 0)
    )
    minimum_overlap_rate = float(
        rospy.get_param("~minimum_overlap_rate", 0.05)
    )
    log_interval = int(rospy.get_param("~log_interval", 25))
    if episodes <= 0 or log_interval <= 0:
        raise ValueError("episodes and log_interval must be positive")

    environment = FusedLowLevelTrainingEnv(
        init_ros_node=False,
        seed=rospy.get_param("~seed", 123),
        enable_teacher=True,
    )
    successes = 0
    collisions = 0
    timeouts = 0
    total_steps = 0
    overlap_steps = 0
    phase_counts = collections.Counter()
    final_distances = []
    base_final_distances = []

    try:
        for episode_index in range(episodes):
            observation = environment.reset()
            reset_info = dict(environment.last_reset_info)
            initial_distance = float(reset_info["initial_ee_distance"])
            episode_overlap = 0
            episode_phases = collections.Counter()
            info = {}

            while not rospy.is_shutdown():
                action = environment.teacher_action(observation)
                diagnostics = environment.teacher_diagnostics()
                phase = str(diagnostics.get("phase", "UNKNOWN"))
                phase_counts[phase] += 1
                episode_phases[phase] += 1
                arm_active = float(
                    np.max(np.abs(action[2:8]))
                ) > 0.02
                base_active = (
                    abs(float(action[0])) > 0.05
                    or abs(float(action[1])) > 0.05
                )
                if arm_active and base_active:
                    overlap_steps += 1
                    episode_overlap += 1

                observation, reward, done, info = environment.step(action)
                total_steps += 1
                episode_step = int(environment.base_env.step_count)
                if (
                        episode_step == 1
                        or episode_step % log_interval == 0
                        or done):
                    fusion = info.get("fusion_state", {})
                    rospy.loginfo(
                        "fused_teacher_step episode=%d step=%d "
                        "phase=%s ee_distance=%.4f base_distance=%.4f "
                        "arm_blend=%.3f action=%s safety=%s",
                        episode_index + 1,
                        episode_step,
                        phase,
                        float(info.get("dist", float("nan"))),
                        float(
                            fusion.get("base_distance", float("nan"))
                        ),
                        float(diagnostics.get("arm_blend", 0.0)),
                        np.round(action, 3).tolist(),
                        str(info.get("safety_info", [])),
                    )
                if done:
                    break

            success = bool(info.get("success", False))
            collision = bool(
                environment.last_sensor_observation is not None
                and (
                    info.get("collision", False)
                    or info.get("collision_names", [])
                )
            )
            # MobileArmReachEnv stores contact state in the next structured
            # observation; query it once at the terminal state for the gate.
            sensor_state = environment.base_env.get_sensor_state()
            collision = bool(collision or sensor_state[3])
            timed_out = bool(not success and not collision)
            successes += int(success)
            collisions += int(collision)
            timeouts += int(timed_out)
            final_distance = float(info.get("dist", float("nan")))
            final_distances.append(final_distance)
            base_distance = float(
                info.get("fusion_state", {}).get(
                    "base_distance",
                    float("nan"),
                )
            )
            base_final_distances.append(base_distance)
            rospy.loginfo(
                "fused_teacher_episode=%d distance=%.4f->%.4f "
                "base_final=%.4f success=%s collision=%s timeout=%s "
                "steps=%d overlap_steps=%d phases=%s",
                episode_index + 1,
                initial_distance,
                final_distance,
                base_distance,
                str(success),
                str(collision),
                str(timed_out),
                int(environment.base_env.step_count),
                episode_overlap,
                dict(episode_phases),
            )
    finally:
        environment.stop()

    success_rate = successes / float(episodes)
    overlap_rate = (
        overlap_steps / float(total_steps)
        if total_steps > 0 else 0.0
    )
    gate_pass = bool(
        success_rate >= minimum_success_rate
        and collisions <= maximum_collisions
        and overlap_rate >= minimum_overlap_rate
        and phase_counts.get("COORDINATED", 0) > 0
    )
    rospy.loginfo(
        "fused_teacher_evaluation episodes=%d successes=%d "
        "success_rate=%.3f collisions=%d timeouts=%d "
        "mean_final_distance=%.4f mean_base_final_distance=%.4f "
        "overlap_steps=%d overlap_rate=%.3f phases=%s",
        episodes,
        successes,
        success_rate,
        collisions,
        timeouts,
        float(np.mean(final_distances)),
        float(np.mean(base_final_distances)),
        overlap_steps,
        overlap_rate,
        dict(phase_counts),
    )
    if gate_pass:
        rospy.loginfo("fused_teacher_gate_pass=True")
        return
    rospy.logerr(
        "fused_teacher_gate_pass=False required_success_rate=%.3f "
        "maximum_collisions=%d minimum_overlap_rate=%.3f",
        minimum_success_rate,
        maximum_collisions,
        minimum_overlap_rate,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
