#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from hrl.high_level_command import HighLevelCommand, TaskMode


class RuleBasedHighPolicy(object):
    """Interpretable task selector used to validate the HRL interfaces."""

    OBS_DIM = 46

    def __init__(
            self,
            arm_enter_distance=0.80,
            arm_exit_distance=1.00,
            min_joint_margin=0.08,
            max_position_subgoal=0.10):
        if arm_enter_distance >= arm_exit_distance:
            raise ValueError("arm_enter_distance must be below arm_exit_distance")

        self.arm_enter_distance = float(arm_enter_distance)
        self.arm_exit_distance = float(arm_exit_distance)
        self.min_joint_margin = float(min_joint_margin)
        self.max_position_subgoal = float(max_position_subgoal)
        self.current_mode = TaskMode.BASE_APPROACH

    def reset(self):
        self.current_mode = TaskMode.BASE_APPROACH

    def predict(self, observation):
        obs = self._as_vector(observation)
        target_in_base = obs[0:3]
        ee_in_base = obs[6:9]
        target_error_in_base = target_in_base - ee_in_base
        ee_target_distance = float(obs[10])
        joint_positions = obs[11:21]
        base_position = joint_positions[0:2]
        base_yaw = float(joint_positions[3])
        base_target_local = self._rotate_xy(
            target_in_base[0:2] - base_position,
            -base_yaw,
        )
        joint_margin = obs[31:41]

        if np.min(joint_margin) < self.min_joint_margin:
            self.current_mode = TaskMode.RECOVERY
            return HighLevelCommand(
                TaskMode.RECOVERY,
                np.zeros(6, dtype=np.float32),
                base_priority=0.0,
                arm_priority=1.0,
                reason="joint margin below safety threshold",
            )

        # Hysteresis prevents rapid switching near the workspace boundary.
        if self.current_mode == TaskMode.BASE_APPROACH:
            if ee_target_distance <= self.arm_enter_distance:
                self.current_mode = TaskMode.ARM_REACH
        elif self.current_mode in (TaskMode.ARM_REACH, TaskMode.RECOVERY):
            if ee_target_distance >= self.arm_exit_distance:
                self.current_mode = TaskMode.BASE_APPROACH
            else:
                self.current_mode = TaskMode.ARM_REACH

        if self.current_mode == TaskMode.BASE_APPROACH:
            subgoal = np.zeros(6, dtype=np.float32)
            # HRL4IN meta actions are relative state changes.  Planar base
            # displacement is expressed in the current robot-local frame.
            subgoal[0:2] = self._limit_norm(
                base_target_local,
                self.max_position_subgoal,
            )
            return HighLevelCommand(
                TaskMode.BASE_APPROACH,
                subgoal,
                base_priority=1.0,
                arm_priority=0.0,
                reason="target outside arm operating region",
            )

        subgoal = np.zeros(6, dtype=np.float32)
        # End-effector displacement occupies the arm task-state dimensions.
        subgoal[3:6] = self._limit_norm(
            target_error_in_base,
            self.max_position_subgoal,
        )
        return HighLevelCommand(
            TaskMode.ARM_REACH,
            subgoal,
            base_priority=0.1,
            arm_priority=1.0,
            reason="target inside arm operating region",
        )

    @classmethod
    def _as_vector(cls, observation):
        if isinstance(observation, dict):
            observation = observation["obs_vec"]
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != (cls.OBS_DIM,):
            raise ValueError(
                "observation must have shape (46,), got {}".format(vector.shape)
            )
        return vector

    @staticmethod
    def _limit_norm(vector, maximum):
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > maximum and norm > 1e-9:
            return vector * (maximum / norm)
        return vector

    @staticmethod
    def _rotate_xy(vector, angle):
        vector = np.asarray(vector, dtype=np.float32)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        return np.asarray([
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ], dtype=np.float32)
