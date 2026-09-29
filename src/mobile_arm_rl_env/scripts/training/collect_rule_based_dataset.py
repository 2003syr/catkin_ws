#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Collect successful ARM_REACH demonstrations from RuleBasedLowPolicy."""

import os
import sys

import numpy as np
import rospy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from hrl.high_level_command import HighLevelCommand, TaskMode
from hrl.rule_based_low_policy import RuleBasedLowPolicy
from training.low_level_reach_env import LowLevelReachTrainingEnv


DEFAULT_OUTPUT_PATH = (
    "/tmp/mobile_arm_rl_training/rule_based_low_level_dataset.npz"
)


def main():
    rospy.init_node("collect_rule_based_low_level_dataset")

    seed = int(rospy.get_param("~seed", 123))
    episode_count = int(rospy.get_param("~episodes", 2000))
    maximum_steps = int(rospy.get_param("~max_steps", 300))
    log_every = int(rospy.get_param("~log_every", 10))
    output_path = os.path.abspath(
        os.path.expanduser(
            rospy.get_param("~output_path", DEFAULT_OUTPUT_PATH)
        )
    )
    if episode_count <= 0:
        raise ValueError("episodes must be positive")
    if maximum_steps <= 0:
        raise ValueError("max_steps must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")

    env = LowLevelReachTrainingEnv(init_ros_node=False, seed=seed)
    rospy.on_shutdown(env.stop)
    teacher = RuleBasedLowPolicy(
        jacobian_provider=env.kinematics_provider,
        arm_joint_max_velocity=[
            env.base_env.action_max_vel[name]
            for name in env.base_env.arm_joints
        ],
        arm_gain=rospy.get_param("~arm_cartesian_gain", 1.0),
        max_cartesian_speed=rospy.get_param(
            "~max_cartesian_speed",
            0.05,
        ),
        dls_damping=rospy.get_param("~dls_damping", 0.03),
    )
    command = HighLevelCommand(
        TaskMode.ARM_REACH,
        np.zeros(6, dtype=np.float32),
        base_priority=0.0,
        arm_priority=1.0,
        reason="rule-based demonstration collection",
    )

    dataset = _empty_dataset()
    completed_episodes = 0
    successful_episodes = 0

    try:
        for episode_id in range(episode_count):
            if rospy.is_shutdown():
                break

            observation = env.reset()
            target_position = env.target_position
            episode_records = []
            episode_success = False

            for step_id in range(maximum_steps):
                if rospy.is_shutdown():
                    break

                full_teacher_action = teacher.predict(
                    observation,
                    command,
                )
                teacher_action = np.asarray(
                    full_teacher_action[4:10],
                    dtype=np.float32,
                )
                next_observation, reward, done, info = env.step(
                    teacher_action
                )
                executed_action, safety_interventions = (
                    _executed_normalized_action(env, info)
                )

                episode_records.append({
                    "observation": observation.copy(),
                    "teacher_action": teacher_action.copy(),
                    "executed_action": executed_action,
                    "next_observation": next_observation.copy(),
                    "reward": float(reward),
                    "done": bool(done),
                    "step_success": bool(info["success"]),
                    "step_id": step_id,
                    "target_position": target_position.copy(),
                    "safety_interventions": safety_interventions,
                })
                observation = next_observation

                if done:
                    episode_success = bool(info["success"])
                    break

            if rospy.is_shutdown() and not episode_records:
                break

            _append_episode(
                dataset,
                episode_records,
                episode_id,
                episode_success,
            )
            completed_episodes += 1
            successful_episodes += int(episode_success)

            if (
                    completed_episodes % log_every == 0
                    or completed_episodes == episode_count):
                rospy.loginfo(
                    "Teacher dataset episodes=%d/%d success=%d samples=%d",
                    completed_episodes,
                    episode_count,
                    successful_episodes,
                    len(dataset["observations"]),
                )
    finally:
        if dataset["observations"]:
            _save_dataset(
                output_path=output_path,
                dataset=dataset,
                seed=seed,
                completed_episodes=completed_episodes,
                successful_episodes=successful_episodes,
                reset_arm_positions=env.reset_arm_positions,
                joint_names=env.base_env.arm_joints,
            )
            rospy.loginfo("Teacher dataset saved: %s", output_path)
        else:
            rospy.logwarn("No teacher transitions were collected")
        env.stop()


def _empty_dataset():
    return {
        "observations": [],
        "teacher_actions": [],
        "executed_actions": [],
        "next_observations": [],
        "rewards": [],
        "dones": [],
        "step_success": [],
        "episode_success": [],
        "episode_ids": [],
        "step_ids": [],
        "target_positions": [],
        "safety_interventions": [],
    }


def _executed_normalized_action(env, info):
    executed = np.zeros(env.ACTION_DIM, dtype=np.float32)
    interventions = np.zeros(env.ACTION_DIM, dtype=np.int8)
    for index, joint_name in enumerate(env.base_env.arm_joints):
        maximum_velocity = env.base_env.action_max_vel[joint_name]
        executed[index] = (
            float(info["cmd_dict"][joint_name]) / maximum_velocity
        )
        interventions[index] = int(
            info["safety_info"][joint_name]["reason"] != "safe"
        )
    return np.clip(executed, -1.0, 1.0), interventions


def _append_episode(dataset, records, episode_id, episode_success):
    for record in records:
        dataset["observations"].append(record["observation"])
        dataset["teacher_actions"].append(record["teacher_action"])
        dataset["executed_actions"].append(record["executed_action"])
        dataset["next_observations"].append(record["next_observation"])
        dataset["rewards"].append(record["reward"])
        dataset["dones"].append(record["done"])
        dataset["step_success"].append(record["step_success"])
        dataset["episode_success"].append(episode_success)
        dataset["episode_ids"].append(episode_id)
        dataset["step_ids"].append(record["step_id"])
        dataset["target_positions"].append(record["target_position"])
        dataset["safety_interventions"].append(
            record["safety_interventions"]
        )


def _save_dataset(
        output_path,
        dataset,
        seed,
        completed_episodes,
        successful_episodes,
        reset_arm_positions,
        joint_names):
    output_directory = os.path.dirname(output_path)
    if output_directory and not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    arrays = {
        "observations": np.asarray(
            dataset["observations"],
            dtype=np.float32,
        ),
        "teacher_actions": np.asarray(
            dataset["teacher_actions"],
            dtype=np.float32,
        ),
        "executed_actions": np.asarray(
            dataset["executed_actions"],
            dtype=np.float32,
        ),
        "next_observations": np.asarray(
            dataset["next_observations"],
            dtype=np.float32,
        ),
        "rewards": np.asarray(dataset["rewards"], dtype=np.float32),
        "dones": np.asarray(dataset["dones"], dtype=np.bool_),
        "step_success": np.asarray(
            dataset["step_success"],
            dtype=np.bool_,
        ),
        "episode_success": np.asarray(
            dataset["episode_success"],
            dtype=np.bool_,
        ),
        "episode_ids": np.asarray(dataset["episode_ids"], dtype=np.int32),
        "step_ids": np.asarray(dataset["step_ids"], dtype=np.int32),
        "target_positions": np.asarray(
            dataset["target_positions"],
            dtype=np.float32,
        ),
        "safety_interventions": np.asarray(
            dataset["safety_interventions"],
            dtype=np.int8,
        ),
        "format_version": np.asarray([1], dtype=np.int32),
        "observation_dim": np.asarray([46], dtype=np.int32),
        "action_dim": np.asarray([6], dtype=np.int32),
        "seed": np.asarray([seed], dtype=np.uint32),
        "completed_episodes": np.asarray(
            [completed_episodes],
            dtype=np.int32,
        ),
        "successful_episodes": np.asarray(
            [successful_episodes],
            dtype=np.int32,
        ),
        "reset_arm_positions": np.asarray(
            reset_arm_positions,
            dtype=np.float32,
        ),
        "joint_names": np.asarray(list(joint_names), dtype="S32"),
    }

    temporary_path = output_path + ".tmp"
    with open(temporary_path, "wb") as output_file:
        np.savez_compressed(output_file, **arrays)
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temporary_path, output_path)


if __name__ == "__main__":
    main()

