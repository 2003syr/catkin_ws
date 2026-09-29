#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HRL4IN-style low-level subgoal state and reward bookkeeping."""

import numpy as np


class HRL4INLowLevelState(object):
    """Translate a relative meta action into a low-level control problem.

    HRL4IN keeps an ideal next state fixed for the lifetime of a subgoal:

        ideal_next_state = state_at_meta_step + original_subgoal
        remaining_subgoal = ideal_next_state - current_state

    The low-level policy observes the current sensor vector together with the
    remaining subgoal, subgoal mask and action mask.  This class implements
    that mechanism for the mobile manipulator while leaving the 46-dimensional
    ROS observation unchanged.

    Task-state layout:

        [base_x, base_y, base_yaw, ee_x, ee_y, ee_z]

    Base displacement is supplied by the high level in the robot-local frame,
    converted to the fixed base-joint frame for the ideal state, and rotated
    back to the current robot-local frame for the low-level observation.
    """

    OBS_DIM = 46
    SUBGOAL_DIM = 6
    ACTION_DIM = 10
    LOW_LEVEL_INPUT_DIM = OBS_DIM + SUBGOAL_DIM + SUBGOAL_DIM + ACTION_DIM

    END_EFFECTOR_POSITION_SLICE = slice(6, 9)
    JOINT_POSITION_SLICE = slice(11, 21)
    BASE_STATE_SLICE = slice(0, 3)
    ARM_STATE_SLICE = slice(3, 6)

    DEFAULT_TOLERANCE = np.asarray(
        [0.02, 0.02, 0.05, 0.01, 0.01, 0.01],
        dtype=np.float64,
    )

    def __init__(
            self,
            subgoal_tolerance=None,
            intrinsic_reward_scale=30.0,
            subgoal_achieved_reward=1.0,
            collision_reward_weight=0.0,
            extrinsic_reward_weight=0.0):
        if subgoal_tolerance is None:
            tolerance = self.DEFAULT_TOLERANCE.copy()
        else:
            tolerance = np.asarray(
                subgoal_tolerance,
                dtype=np.float64,
            )
            if tolerance.ndim == 0:
                tolerance = np.full(
                    self.SUBGOAL_DIM,
                    float(tolerance),
                    dtype=np.float64,
                )
        if tolerance.shape != (self.SUBGOAL_DIM,):
            raise ValueError(
                "subgoal_tolerance must be scalar or shape (6,), got {}".format(
                    tolerance.shape
                )
            )
        if np.any(tolerance <= 0.0):
            raise ValueError("subgoal_tolerance values must be positive")

        self.subgoal_tolerance = tolerance
        self.intrinsic_reward_scale = float(intrinsic_reward_scale)
        self.subgoal_achieved_reward = float(subgoal_achieved_reward)
        self.collision_reward_weight = float(collision_reward_weight)
        self.extrinsic_reward_weight = float(extrinsic_reward_weight)
        if self.intrinsic_reward_scale < 0.0:
            raise ValueError("intrinsic_reward_scale must be non-negative")
        self.reset()

    def reset(self):
        self.command = None
        self.original_subgoal = np.zeros(
            self.SUBGOAL_DIM,
            dtype=np.float64,
        )
        self.subgoal_mask = np.zeros(
            self.SUBGOAL_DIM,
            dtype=np.float64,
        )
        self.action_mask = np.zeros(
            self.ACTION_DIM,
            dtype=np.float64,
        )
        self.ideal_next_state = None
        self.remaining_subgoal_global = np.zeros(
            self.SUBGOAL_DIM,
            dtype=np.float64,
        )
        self.remaining_subgoal = np.zeros(
            self.SUBGOAL_DIM,
            dtype=np.float64,
        )
        self.previous_potential = 0.0
        self.subgoal_steps = 0
        self.subgoal_achieved = False
        self.subgoal_timed_out = False
        self.subgoal_done = False
        self.cumulative_extrinsic_reward = 0.0
        self.last_diagnostics = {}

    def begin_subgoal(self, observation, command):
        """Start one asynchronous HRL4IN low-level rollout."""
        sensor = self.as_sensor(observation)
        current_state = self.extract_task_state(sensor)

        self.command = command
        self.subgoal_mask = self._mask(
            command.subgoal_mask,
            self.SUBGOAL_DIM,
            "subgoal_mask",
        )
        self.action_mask = self._mask(
            command.action_mask,
            self.ACTION_DIM,
            "action_mask",
        )
        self.original_subgoal = (
            np.asarray(command.subgoal, dtype=np.float64)
            * self.subgoal_mask
        )

        # The high policy expresses planar base displacement in the current
        # robot-local frame.  Store the corresponding fixed-frame state target.
        fixed_frame_subgoal = self.original_subgoal.copy()
        fixed_frame_subgoal[0:2] = self._rotate_xy(
            fixed_frame_subgoal[0:2],
            current_state[2],
        )
        self.ideal_next_state = current_state + fixed_frame_subgoal

        self.remaining_subgoal_global = (
            self.ideal_next_state - current_state
        ) * self.subgoal_mask
        self.remaining_subgoal = self._to_low_level_frame(
            self.remaining_subgoal_global,
            current_state,
        )
        self.previous_potential = self._potential(
            self.remaining_subgoal_global
        )
        self.subgoal_steps = 0
        self.subgoal_achieved = False
        self.subgoal_timed_out = False
        self.subgoal_done = False
        self.cumulative_extrinsic_reward = 0.0
        self.last_diagnostics = {
            "policy_type": "hrl4in_low_level",
            "task_state": current_state.copy(),
            "original_subgoal": self.original_subgoal.copy(),
            "ideal_next_state": self.ideal_next_state.copy(),
            "remaining_subgoal": self.remaining_subgoal.copy(),
            "remaining_subgoal_global": (
                self.remaining_subgoal_global.copy()
            ),
            "subgoal_mask": self.subgoal_mask.copy(),
            "action_mask": self.action_mask.copy(),
            "subgoal_tolerance": self.subgoal_tolerance.copy(),
            "pre_potential": float(self.previous_potential),
            "post_potential": float(self.previous_potential),
            "intrinsic_progress_reward": 0.0,
            "intrinsic_achievement_reward": 0.0,
            "intrinsic_collision_reward": 0.0,
            "intrinsic_extrinsic_reward": 0.0,
            "intrinsic_reward": 0.0,
            "subgoal_steps": 0,
            "subgoal_achieved": False,
            "subgoal_timed_out": False,
            "subgoal_done": False,
            "low_level_mask": 1.0,
        }
        return self.low_level_observation(sensor)

    def low_level_observation(self, observation):
        """Return the structured input consumed by an HRL4IN low policy."""
        sensor = self.as_sensor(observation)
        vector = np.concatenate((
            sensor.astype(np.float32),
            self.remaining_subgoal.astype(np.float32),
            self.subgoal_mask.astype(np.float32),
            self.action_mask.astype(np.float32),
        ))
        return {
            "sensor": sensor.copy(),
            "subgoal": self.remaining_subgoal.astype(np.float32).copy(),
            "subgoal_mask": self.subgoal_mask.astype(np.float32).copy(),
            "action_mask": self.action_mask.astype(np.float32).copy(),
            "vector": vector.astype(np.float32),
        }

    def observe_transition(
            self,
            next_observation,
            extrinsic_reward=0.0,
            collision_reward=0.0,
            episode_done=False,
            subgoal_timed_out=False):
        """Update the remaining goal and HRL4IN intrinsic reward."""
        if self.command is None or self.ideal_next_state is None:
            raise RuntimeError("begin_subgoal must be called before transition")

        next_sensor = self.as_sensor(next_observation)
        next_state = self.extract_task_state(next_sensor)
        remaining_global = (
            self.ideal_next_state - next_state
        ) * self.subgoal_mask
        remaining_local = self._to_low_level_frame(
            remaining_global,
            next_state,
        )

        pre_potential = float(self.previous_potential)
        post_potential = self._potential(remaining_global)
        active = self.subgoal_mask > 0.5
        component_error = np.abs(remaining_global)
        achieved = bool(
            np.any(active)
            and np.all(
                component_error[active]
                < self.subgoal_tolerance[active]
            )
        )

        self.subgoal_steps += 1
        self.subgoal_achieved = achieved
        self.subgoal_timed_out = bool(subgoal_timed_out)
        self.subgoal_done = bool(
            achieved or self.subgoal_timed_out or episode_done
        )
        self.cumulative_extrinsic_reward += float(extrinsic_reward)

        progress_reward = (
            pre_potential - post_potential
        ) * self.intrinsic_reward_scale
        achievement_reward = (
            self.subgoal_achieved_reward if achieved else 0.0
        )
        collision_term = (
            float(collision_reward) * self.collision_reward_weight
        )
        extrinsic_term = (
            float(extrinsic_reward) * self.extrinsic_reward_weight
        )
        intrinsic_reward = (
            progress_reward
            + achievement_reward
            + collision_term
            + extrinsic_term
        )

        self.remaining_subgoal_global = remaining_global
        self.remaining_subgoal = remaining_local
        self.previous_potential = post_potential
        self.last_diagnostics.update({
            "task_state": next_state.copy(),
            "remaining_subgoal": remaining_local.copy(),
            "remaining_subgoal_global": remaining_global.copy(),
            "component_error": component_error.copy(),
            "pre_potential": pre_potential,
            "post_potential": post_potential,
            "intrinsic_progress_reward": float(progress_reward),
            "intrinsic_achievement_reward": float(achievement_reward),
            "intrinsic_collision_reward": float(collision_term),
            "intrinsic_extrinsic_reward": float(extrinsic_term),
            "intrinsic_reward": float(intrinsic_reward),
            "subgoal_steps": int(self.subgoal_steps),
            "subgoal_achieved": self.subgoal_achieved,
            "subgoal_timed_out": self.subgoal_timed_out,
            "subgoal_done": self.subgoal_done,
            "low_level_mask": 0.0 if self.subgoal_done else 1.0,
            "cumulative_extrinsic_reward": float(
                self.cumulative_extrinsic_reward
            ),
        })
        return self.diagnostics()

    def diagnostics(self):
        result = {}
        for key, value in self.last_diagnostics.items():
            result[key] = value.copy() if hasattr(value, "copy") else value
        return result

    @classmethod
    def extract_task_state(cls, observation):
        sensor = cls.as_sensor(observation)
        joint_positions = sensor[cls.JOINT_POSITION_SLICE]
        return np.asarray([
            joint_positions[0],
            joint_positions[1],
            joint_positions[3],
            sensor[cls.END_EFFECTOR_POSITION_SLICE][0],
            sensor[cls.END_EFFECTOR_POSITION_SLICE][1],
            sensor[cls.END_EFFECTOR_POSITION_SLICE][2],
        ], dtype=np.float64)

    @classmethod
    def as_sensor(cls, observation):
        if isinstance(observation, dict):
            observation = observation["obs_vec"]
        sensor = np.asarray(observation, dtype=np.float32)
        if sensor.shape != (cls.OBS_DIM,):
            raise ValueError(
                "observation must have shape (46,), got {}".format(
                    sensor.shape
                )
            )
        return sensor

    def _to_low_level_frame(self, remaining_global, current_state):
        remaining = np.asarray(
            remaining_global,
            dtype=np.float64,
        ).copy()
        remaining[0:2] = self._rotate_xy(
            remaining[0:2],
            -current_state[2],
        )
        return remaining * self.subgoal_mask

    def _potential(self, remaining_global):
        return float(np.linalg.norm(
            np.asarray(remaining_global, dtype=np.float64)
            * self.subgoal_mask
        ))

    @staticmethod
    def _rotate_xy(vector, angle):
        vector = np.asarray(vector, dtype=np.float64)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        return np.asarray([
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ], dtype=np.float64)

    @staticmethod
    def _mask(value, expected_size, name):
        mask = np.asarray(value, dtype=np.float64)
        if mask.shape != (expected_size,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name,
                    expected_size,
                    mask.shape,
                )
            )
        if not np.all(np.isfinite(mask)):
            raise ValueError("{} contains non-finite values".format(name))
        return np.clip(mask, 0.0, 1.0)
