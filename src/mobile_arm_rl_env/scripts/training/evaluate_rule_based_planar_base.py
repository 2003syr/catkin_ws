#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the proportional x/y teacher over randomized planar targets."""

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


def count_safety_events(safety_info):
    return sum(
        item.get("reason", "safe") != "safe"
        for item in safety_info.values()
    )


def main():
    rospy.init_node("evaluate_rule_based_planar_base")
    episodes = int(rospy.get_param("~episodes", 20))
    seed = int(rospy.get_param("~seed", 123))
    base_gain = float(rospy.get_param("~base_gain", 5.0))
    required_success_rate = float(
        rospy.get_param("~required_success_rate", 0.90)
    )
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if base_gain <= 0.0:
        raise ValueError("base_gain must be positive")
    if not 0.0 <= required_success_rate <= 1.0:
        raise ValueError("required_success_rate must be in [0, 1]")

    env = PlanarBaseTrainingEnv(init_ros_node=False, seed=seed)
    successes = 0
    collisions = 0
    total_safety_events = 0
    final_distances = []
    successful_steps = []
    maximum_reset_error = 0.0

    try:
        for episode in range(episodes):
            observation = env.reset()
            start_distance = float(np.linalg.norm(observation[0:2]))
            target = env.target_position.copy()
            reset_error = float(
                env.last_reset_info["reset_position_error"]
            )
            maximum_reset_error = max(maximum_reset_error, reset_error)
            episode_safety_events = 0
            info = {
                "success": False,
                "collision": False,
                "distance": start_distance,
            }

            for step in range(env.task.max_steps):
                action = np.clip(
                    base_gain * observation[0:2],
                    -1.0,
                    1.0,
                )
                observation, reward, done, info = env.step(action)
                episode_safety_events += count_safety_events(
                    info["safety_info"]
                )
                if done:
                    break

            success = bool(info["success"])
            collision = bool(info["collision"])
            final_distance = float(info["distance"])
            successes += int(success)
            collisions += int(collision)
            total_safety_events += episode_safety_events
            final_distances.append(final_distance)
            if success:
                successful_steps.append(step + 1)

            rospy.loginfo(
                "planar_episode=%d target=[%.3f, %.3f] "
                "reset_error=%.5f dist=%.4f->%.4f success=%s "
                "collision=%s timeout=%s steps=%d safety_events=%d",
                episode + 1,
                target[0],
                target[1],
                reset_error,
                start_distance,
                final_distance,
                success,
                collision,
                bool(info["timeout"]),
                step + 1,
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
        "planar_summary success=%d/%d rate=%.3f collisions=%d "
        "mean_final_distance=%.4f mean_success_steps=%.1f "
        "safety_events=%d max_reset_error=%.5f required_rate=%.3f",
        successes,
        episodes,
        success_rate,
        collisions,
        mean_final_distance,
        mean_success_steps,
        total_safety_events,
        maximum_reset_error,
        required_success_rate,
    )
    if success_rate < required_success_rate or collisions > 0:
        rospy.logerr("Planar rule-based acceptance failed")
        return 2
    rospy.loginfo("Planar rule-based acceptance passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
