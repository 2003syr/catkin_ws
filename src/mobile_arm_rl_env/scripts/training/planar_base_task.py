#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pure NumPy state and reward logic for the first-stage planar base task."""

from __future__ import print_function

import numpy as np


class PlanarBaseTask(object):
    """Convert the 46-D ROS observation into an 11-D base-policy problem.

    Compact policy observation layout:

        0:2   remaining target displacement in the base/body frame [m]
        2:4   measured forward velocity/yaw rate, normalized
        4:9   five obstacle-distance sectors, normalized to [0, 1]
        9:11  previous normalized [linear velocity, yaw rate] action

    The ROS environment and its ten-dimensional execution contract remain
    unchanged.  Only the policy-facing adapter is compact.
    """

    OBS_DIM = 11
    ACTION_DIM = 2
    SCAN_DIM = 5

    def __init__(
            self,
            success_threshold=0.03,
            max_steps=300,
            progress_reward_scale=10.0,
            success_reward=2.0,
            collision_penalty=2.0,
            time_penalty=0.01,
            action_penalty_scale=0.01,
            smoothness_penalty_scale=0.02,
            collision_distance=0.20,
            scan_clip=10.0,
            target_clip=2.0,
            base_velocity_scale=(0.20, 0.50)):
        self.success_threshold = float(success_threshold)
        self.max_steps = int(max_steps)
        self.progress_reward_scale = float(progress_reward_scale)
        self.success_reward = float(success_reward)
        self.collision_penalty = float(collision_penalty)
        self.time_penalty = float(time_penalty)
        self.action_penalty_scale = float(action_penalty_scale)
        self.smoothness_penalty_scale = float(
            smoothness_penalty_scale
        )
        self.collision_distance = float(collision_distance)
        self.scan_clip = float(scan_clip)
        self.target_clip = float(target_clip)
        self.base_velocity_scale = self._vector(
            base_velocity_scale,
            self.ACTION_DIM,
            "base_velocity_scale",
        )
        self._validate_parameters()
        self.previous_action = np.zeros(
            self.ACTION_DIM,
            dtype=np.float32,
        )
        self.previous_distance = None
        self.step_count = 0
        self.last_diagnostics = {}

    def reset(self, observation):
        self.previous_action.fill(0.0)
        self.step_count = 0
        remaining = self._remaining_xy(observation)
        self.previous_distance = float(np.linalg.norm(remaining))
        self.last_diagnostics = {
            "distance": self.previous_distance,
            "progress": 0.0,
            "success": False,
            "collision": False,
            "timeout": False,
            "done": False,
            "step_count": 0,
        }
        return self.encode(observation)

    def encode(self, observation):
        remaining = np.clip(
            self._remaining_xy(observation),
            -self.target_clip,
            self.target_clip,
        )
        velocity = np.clip(
            self._base_velocity(observation) / self.base_velocity_scale,
            -1.0,
            1.0,
        )
        scan = np.clip(
            self._scan(observation),
            0.0,
            self.scan_clip,
        ) / self.scan_clip
        vector = np.concatenate((
            remaining.astype(np.float32),
            velocity.astype(np.float32),
            scan.astype(np.float32),
            self.previous_action.astype(np.float32),
        ))
        if vector.shape != (self.OBS_DIM,):
            raise RuntimeError(
                "planar observation must have shape (11,), got {}".format(
                    vector.shape
                )
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("planar observation contains non-finite values")
        return vector

    def transition(self, observation, action):
        if self.previous_distance is None:
            raise RuntimeError("reset() must be called before transition()")
        action = np.clip(
            self._vector(action, self.ACTION_DIM, "action"),
            -1.0,
            1.0,
        ).astype(np.float32)
        remaining = self._remaining_xy(observation)
        distance = float(np.linalg.norm(remaining))
        progress = float(self.previous_distance - distance)
        scan = self._scan(observation)
        contact_collision = bool(observation.get("collision", False))
        proximity_collision = bool(
            float(np.min(scan)) < self.collision_distance
        )
        collision = bool(contact_collision or proximity_collision)
        if contact_collision:
            collision_source = "contact"
        elif proximity_collision:
            collision_source = "laser_proximity"
        else:
            collision_source = "none"
        success = bool(
            not collision and distance < self.success_threshold
        )

        self.step_count += 1
        timeout = bool(self.step_count >= self.max_steps and not success)
        done = bool(success or collision or timeout)

        progress_term = self.progress_reward_scale * progress
        action_term = -self.action_penalty_scale * float(
            np.mean(np.square(action))
        )
        smoothness_term = -self.smoothness_penalty_scale * float(
            np.mean(np.square(action - self.previous_action))
        )
        time_term = -self.time_penalty
        success_term = self.success_reward if success else 0.0
        collision_term = -self.collision_penalty if collision else 0.0
        reward = float(
            progress_term
            + action_term
            + smoothness_term
            + time_term
            + success_term
            + collision_term
        )

        self.previous_distance = distance
        self.previous_action = action.copy()
        self.last_diagnostics = {
            "distance": distance,
            "progress": progress,
            "progress_reward": progress_term,
            "action_penalty": action_term,
            "smoothness_penalty": smoothness_term,
            "time_penalty": time_term,
            "success_reward": success_term,
            "collision_penalty": collision_term,
            "reward": reward,
            "success": success,
            "collision": collision,
            "contact_collision": contact_collision,
            "proximity_collision": proximity_collision,
            "collision_source": collision_source,
            "timeout": timeout,
            "done": done,
            "step_count": self.step_count,
        }
        return self.encode(observation), reward, done, self.diagnostics()

    def diagnostics(self):
        return dict(self.last_diagnostics)

    @staticmethod
    def _remaining_xy(observation):
        if not isinstance(observation, dict):
            raise TypeError("planar task requires the structured observation")
        return PlanarBaseTask._vector(
            observation["base_to_target_body_pos"],
            3,
            "base_to_target_body_pos",
        )[0:2]

    @staticmethod
    def _base_velocity(observation):
        if not isinstance(observation, dict):
            raise TypeError("planar task requires the structured observation")
        if "observed_planar_body_velocity" in observation:
            return PlanarBaseTask._vector(
                observation["observed_planar_body_velocity"],
                3,
                "observed_planar_body_velocity",
            )[[0, 2]]
        joint_velocity = PlanarBaseTask._vector(
            observation["joint_vel"],
            10,
            "joint_vel",
        )
        heading = float(observation.get(
            "planar_base_heading",
            observation.get("planar_base_yaw", 0.0),
        ))
        cosine = np.cos(heading)
        sine = np.sin(heading)
        forward_velocity = (
            cosine * joint_velocity[0]
            + sine * joint_velocity[1]
        )
        return np.asarray([
            forward_velocity,
            joint_velocity[2],
        ], dtype=np.float64)

    @staticmethod
    def _scan(observation):
        if not isinstance(observation, dict):
            raise TypeError("planar task requires the structured observation")
        return PlanarBaseTask._vector(
            observation["scan_info"],
            PlanarBaseTask.SCAN_DIM,
            "scan_info",
        )

    @staticmethod
    def _vector(value, expected_size, name):
        vector = np.asarray(value, dtype=np.float64)
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

    def _validate_parameters(self):
        if self.success_threshold <= 0.0:
            raise ValueError("success_threshold must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.scan_clip <= 0.0:
            raise ValueError("scan_clip must be positive")
        if self.target_clip <= 0.0:
            raise ValueError("target_clip must be positive")
        if np.any(self.base_velocity_scale <= 0.0):
            raise ValueError("base_velocity_scale values must be positive")
        if self.collision_distance < 0.0:
            raise ValueError("collision_distance must be non-negative")
