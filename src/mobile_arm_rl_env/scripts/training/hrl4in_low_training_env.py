#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Low-level PPO environment with HRL4IN subgoal semantics."""

import numpy as np
import rospy

from hrl.high_level_command import HighLevelCommand, TaskMode
from hrl.hrl4in_low_level import HRL4INLowLevelState
from hrl.rule_based_low_policy import RuleBasedLowPolicy
from training.low_level_reach_env import LowLevelReachTrainingEnv


class RuleBasedArmSubgoalGenerator(object):
    """Fixed high level used while the low-level policy is trained."""

    OBS_DIM = 46
    SUBGOAL_DIM = 6

    def __init__(self, max_position_subgoal=0.08):
        self.max_position_subgoal = float(max_position_subgoal)
        if self.max_position_subgoal <= 0.0:
            raise ValueError("max_position_subgoal must be positive")

    def predict(self, observation):
        sensor = HRL4INLowLevelState.as_sensor(observation)
        target_position = np.asarray(sensor[0:3], dtype=np.float64)
        end_effector_position = np.asarray(
            sensor[6:9],
            dtype=np.float64,
        )
        displacement = target_position - end_effector_position
        displacement = self._limit_norm(
            displacement,
            self.max_position_subgoal,
        )
        subgoal = np.zeros(self.SUBGOAL_DIM, dtype=np.float32)
        subgoal[3:6] = displacement.astype(np.float32)
        return HighLevelCommand(
            TaskMode.ARM_REACH,
            subgoal,
            base_priority=0.0,
            arm_priority=1.0,
            reason="fixed rule high policy for low-level PPO training",
        )

    @staticmethod
    def _limit_norm(vector, maximum):
        vector = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm > maximum and norm > 1e-9:
            return vector * (maximum / norm)
        return vector


class HRL4INLowTrainingEnv(object):
    """Expose subgoal rollouts while keeping Gazebo episodes continuous.

    A trainer treats each HRL4IN subgoal as a terminal low-level rollout.
    Calling reset() after a subgoal timeout or achievement starts a new
    high-level subgoal from the current Gazebo state.  Gazebo itself is reset
    only when the task episode finishes or reset(force_episode=True) is used.
    """

    OBS_DIM = HRL4INLowLevelState.LOW_LEVEL_INPUT_DIM
    ACTION_DIM = 10
    ARM_ACTION_SLICE = slice(4, 10)

    def __init__(
            self,
            reach_env=None,
            time_scale=30,
            max_position_subgoal=0.08,
            subgoal_tolerance=None,
            intrinsic_reward_scale=30.0,
            subgoal_achieved_reward=1.0,
            collision_reward_weight=0.0,
            extrinsic_reward_weight=0.0,
            enable_teacher=True):
        self.reach_env = reach_env or LowLevelReachTrainingEnv(
            init_ros_node=False
        )
        self.time_scale = int(time_scale)
        if self.time_scale <= 0:
            raise ValueError("time_scale must be positive")

        self.high_policy = RuleBasedArmSubgoalGenerator(
            max_position_subgoal=max_position_subgoal
        )
        self.subgoal_state = HRL4INLowLevelState(
            subgoal_tolerance=subgoal_tolerance,
            intrinsic_reward_scale=intrinsic_reward_scale,
            subgoal_achieved_reward=subgoal_achieved_reward,
            collision_reward_weight=collision_reward_weight,
            extrinsic_reward_weight=extrinsic_reward_weight,
        )
        self.enable_teacher = bool(enable_teacher)
        self.teacher = None
        if self.enable_teacher:
            base_env = self.reach_env.base_env
            self.teacher = RuleBasedLowPolicy(
                enable_base_motion=False,
                jacobian_provider=self.reach_env.kinematics_provider,
                arm_joint_max_velocity=[
                    base_env.action_max_vel[name]
                    for name in base_env.arm_joints
                ],
                arm_gain=float(rospy.get_param(
                    "~teacher_arm_cartesian_gain",
                    1.0,
                )),
                max_cartesian_speed=float(rospy.get_param(
                    "~teacher_max_cartesian_speed",
                    0.05,
                )),
                dls_damping=float(rospy.get_param(
                    "~teacher_dls_damping",
                    0.03,
                )),
                subgoal_tolerance=subgoal_tolerance,
                intrinsic_reward_scale=intrinsic_reward_scale,
                subgoal_achieved_reward=subgoal_achieved_reward,
                collision_reward_weight=collision_reward_weight,
                extrinsic_reward_weight=extrinsic_reward_weight,
            )

        self.current_observation = None
        self.current_command = None
        self.episode_needs_reset = True
        self.subgoal_needs_reset = True
        self.subgoal_index = 0
        self.total_steps = 0

    def reset(self, force_episode=False):
        """Return a new 68-dimensional low-level observation."""
        if (
                self.current_observation is None
                or self.episode_needs_reset
                or bool(force_episode)):
            self.current_observation = self.reach_env.reset()
            self.episode_needs_reset = False

        self.current_command = self.high_policy.predict(
            self.current_observation
        )
        low_observation = self.subgoal_state.begin_subgoal(
            self.current_observation,
            self.current_command,
        )
        if self.teacher is not None:
            self.teacher.begin_subgoal(
                self.current_observation,
                self.current_command,
            )

        self.subgoal_needs_reset = False
        self.subgoal_index += 1
        return low_observation["vector"].copy()

    def step(self, action):
        """Execute the learned action and return the intrinsic PPO reward."""
        if self.current_observation is None or self.subgoal_needs_reset:
            raise RuntimeError("reset must be called before step")

        action = self._as_vector(
            action,
            self.ACTION_DIM,
            "action",
        )
        low_observation = self.subgoal_state.low_level_observation(
            self.current_observation
        )
        action_mask = np.asarray(
            low_observation["action_mask"],
            dtype=np.float32,
        )
        masked_action = np.clip(action, -1.0, 1.0) * action_mask

        teacher_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        if self.teacher is not None:
            teacher_action = self.teacher.predict(
                self.current_observation,
                self.current_command,
            )

        next_observation, extrinsic_reward, episode_done, base_info = (
            self.reach_env.step(masked_action[self.ARM_ACTION_SLICE])
        )
        base_info = dict(base_info)
        safety_event_count = self._safety_event_count(base_info)
        collision_reward = float(
            base_info.get(
                "collision_reward",
                -float(safety_event_count),
            )
        )
        subgoal_timed_out = bool(
            self.subgoal_state.subgoal_steps + 1 >= self.time_scale
        )
        diagnostics = self.subgoal_state.observe_transition(
            next_observation,
            extrinsic_reward=extrinsic_reward,
            collision_reward=collision_reward,
            episode_done=episode_done,
            subgoal_timed_out=subgoal_timed_out,
        )
        if self.teacher is not None:
            self.teacher.observe_transition(
                next_observation,
                extrinsic_reward=extrinsic_reward,
                collision_reward=collision_reward,
                episode_done=episode_done,
                subgoal_timed_out=subgoal_timed_out,
            )

        self.current_observation = next_observation
        self.total_steps += 1
        self.subgoal_needs_reset = bool(diagnostics["subgoal_done"])
        self.episode_needs_reset = bool(episode_done)

        next_low_observation = self.subgoal_state.low_level_observation(
            next_observation
        )
        info = dict(base_info)
        info.update({
            "episode_done": bool(episode_done),
            "subgoal_done": bool(diagnostics["subgoal_done"]),
            "subgoal_achieved": bool(diagnostics["subgoal_achieved"]),
            "subgoal_timed_out": bool(diagnostics["subgoal_timed_out"]),
            "subgoal_steps": int(diagnostics["subgoal_steps"]),
            "subgoal_index": int(self.subgoal_index),
            "total_steps": int(self.total_steps),
            "teacher_action": teacher_action.copy(),
            "teacher_available": bool(self.teacher is not None),
            "raw_action": action.copy(),
            "masked_action": masked_action.copy(),
            "action_mask": action_mask.copy(),
            "remaining_subgoal": diagnostics[
                "remaining_subgoal"
            ].copy(),
            "ideal_next_state": diagnostics[
                "ideal_next_state"
            ].copy(),
            "intrinsic_reward": float(
                diagnostics["intrinsic_reward"]
            ),
            "intrinsic_progress_reward": float(
                diagnostics["intrinsic_progress_reward"]
            ),
            "intrinsic_achievement_reward": float(
                diagnostics["intrinsic_achievement_reward"]
            ),
            "safety_event_count": int(safety_event_count),
            "extrinsic_reward": float(extrinsic_reward),
        })
        return (
            next_low_observation["vector"].copy(),
            float(diagnostics["intrinsic_reward"]),
            bool(diagnostics["subgoal_done"]),
            info,
        )

    def stop(self):
        self.reach_env.stop()

    close = stop

    @staticmethod
    def _safety_event_count(info):
        safety_info = info.get("safety_info", {})
        return sum(
            1
            for value in safety_info.values()
            if value.get("reason", "safe") != "safe"
        )

    @staticmethod
    def _as_vector(value, expected_size, name):
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (expected_size,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name,
                    expected_size,
                    vector.shape,
                )
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("{} contains non-finite values".format(name))
        return vector
