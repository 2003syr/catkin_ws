#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Deterministic pose/path tracker for a differential tracked base."""

from __future__ import print_function

import math

import numpy as np


class RegulatedPathTracker(object):
    """Track an SE(2) lattice path using only normalized [v, omega]."""

    def __init__(
            self,
            maximum_linear_speed,
            maximum_yaw_rate,
            position_tolerance=0.07,
            waypoint_tolerance=0.065,
            yaw_tolerance=0.16,
            turn_in_place_angle=0.30,
            slow_distance=0.35,
            heading_gain=1.8,
            path_yaw_gain=0.35,
            minimum_linear_fraction=0.18,
            linear_slew_limit=0.10,
            angular_slew_limit=0.16):
        self.maximum_linear_speed = float(maximum_linear_speed)
        self.maximum_yaw_rate = float(maximum_yaw_rate)
        self.position_tolerance = float(position_tolerance)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.yaw_tolerance = float(yaw_tolerance)
        self.turn_in_place_angle = float(turn_in_place_angle)
        self.slow_distance = float(slow_distance)
        self.heading_gain = float(heading_gain)
        self.path_yaw_gain = float(path_yaw_gain)
        self.minimum_linear_fraction = float(minimum_linear_fraction)
        self.slew_limits = np.asarray([
            linear_slew_limit,
            angular_slew_limit,
        ], dtype=np.float64)
        self._validate()
        self.path = None
        self.path_index = 0
        self.previous_action = np.zeros(2, dtype=np.float64)

    def reset(self):
        self.path = None
        self.path_index = 0
        self.previous_action.fill(0.0)

    def set_path(self, path, current_pose=None):
        path = np.asarray(path, dtype=np.float64)
        if (
                path.ndim != 2
                or path.shape[1] != 3
                or path.shape[0] == 0
                or not np.all(np.isfinite(path))):
            raise ValueError("path must have finite shape (N, 3)")
        self.path = path.copy()
        self.path_index = 0
        if current_pose is not None:
            current_pose = self._vector(
                current_pose, 3, "current_pose"
            )
            distances = np.linalg.norm(
                self.path[:, 0:2] - current_pose[0:2],
                axis=1,
            )
            self.path_index = int(np.argmin(distances))

    def synchronize_executed_action(self, action):
        """Use the command that the safety layer actually allowed.

        The tracker must not integrate a requested turn as history when the
        emergency shield replaced it with zero.  Keeping these states aligned
        also makes the slew limiter recover immediately after an intervention.
        """
        self.previous_action = np.clip(
            self._vector(action, 2, "executed_action"),
            -1.0,
            1.0,
        )

    def compute_action(self, current_pose, goal_xy):
        if self.path is None:
            raise RuntimeError("set_path() must be called first")
        pose = self._vector(current_pose, 3, "current_pose")
        goal_xy = self._vector(goal_xy, 2, "goal_xy")
        goal_distance = float(np.linalg.norm(goal_xy - pose[0:2]))
        if goal_distance <= self.position_tolerance:
            return self._slew(np.zeros(2)), {
                "goal_reached": True,
                "path_index": int(self.path_index),
                "goal_distance": goal_distance,
                "target_distance": 0.0,
                "heading_error": 0.0,
                "mode": "GOAL_REACHED",
            }

        self._advance_completed_waypoints(pose)
        target_index = min(self.path_index, len(self.path) - 1)
        target = self.path[target_index]
        displacement = target[0:2] - pose[0:2]
        target_distance = float(np.linalg.norm(displacement))
        path_yaw_error = self._angle_difference(target[2], pose[2])

        if (
                target_distance <= self.waypoint_tolerance
                and abs(path_yaw_error) > self.yaw_tolerance):
            desired = np.asarray([
                0.0,
                np.clip(
                    self.heading_gain
                    * path_yaw_error
                    / self.maximum_yaw_rate,
                    -1.0,
                    1.0,
                ),
            ])
            mode = "ROTATE_TO_PATH_POSE"
            heading_error = path_yaw_error
        else:
            if target_distance <= self.waypoint_tolerance:
                lookahead_index = min(
                    target_index + 1,
                    len(self.path) - 1,
                )
                target = self.path[lookahead_index]
                displacement = target[0:2] - pose[0:2]
                target_distance = float(np.linalg.norm(displacement))
            target_heading = math.atan2(
                displacement[1],
                displacement[0],
            )
            heading_error = self._angle_difference(
                target_heading,
                pose[2],
            )
            if abs(heading_error) >= self.turn_in_place_angle:
                linear = 0.0
                mode = "ROTATE_TO_SEGMENT"
            else:
                alignment = max(0.0, math.cos(heading_error))
                distance_scale = min(
                    1.0,
                    max(
                        self.minimum_linear_fraction,
                        goal_distance / self.slow_distance,
                    ),
                )
                linear = distance_scale * alignment
                mode = "TRACK_PATH"
            yaw_rate = (
                self.heading_gain * heading_error
                + self.path_yaw_gain * path_yaw_error
            )
            desired = np.asarray([
                linear,
                np.clip(
                    yaw_rate / self.maximum_yaw_rate,
                    -1.0,
                    1.0,
                ),
            ])

        action = self._slew(desired)
        return action.astype(np.float32), {
            "goal_reached": False,
            "path_index": int(self.path_index),
            "goal_distance": goal_distance,
            "target_distance": target_distance,
            "heading_error": float(heading_error),
            "mode": mode,
        }

    def _advance_completed_waypoints(self, pose):
        while self.path_index < len(self.path) - 1:
            waypoint = self.path[self.path_index]
            distance = float(np.linalg.norm(
                waypoint[0:2] - pose[0:2]
            ))
            yaw_error = abs(self._angle_difference(
                waypoint[2],
                pose[2],
            ))
            if (
                    distance > self.waypoint_tolerance
                    or yaw_error > self.yaw_tolerance):
                break
            self.path_index += 1

    def _slew(self, desired):
        desired = np.clip(
            self._vector(desired, 2, "desired_action"),
            -1.0,
            1.0,
        )
        delta = np.clip(
            desired - self.previous_action,
            -self.slew_limits,
            self.slew_limits,
        )
        self.previous_action = np.clip(
            self.previous_action + delta,
            -1.0,
            1.0,
        )
        return self.previous_action.copy()

    @staticmethod
    def _angle_difference(value, reference):
        return math.atan2(
            math.sin(float(value) - float(reference)),
            math.cos(float(value) - float(reference)),
        )

    def _validate(self):
        if self.maximum_linear_speed <= 0.0:
            raise ValueError("maximum_linear_speed must be positive")
        if self.maximum_yaw_rate <= 0.0:
            raise ValueError("maximum_yaw_rate must be positive")
        if self.position_tolerance <= 0.0:
            raise ValueError("position_tolerance must be positive")
        if self.waypoint_tolerance <= 0.0:
            raise ValueError("waypoint_tolerance must be positive")
        if not 0.0 < self.yaw_tolerance < math.pi:
            raise ValueError("yaw_tolerance must be in (0, pi)")
        if not 0.0 < self.turn_in_place_angle < math.pi:
            raise ValueError("turn_in_place_angle must be in (0, pi)")
        if self.slow_distance <= self.position_tolerance:
            raise ValueError("slow_distance must exceed position_tolerance")
        if self.heading_gain <= 0.0 or self.path_yaw_gain < 0.0:
            raise ValueError("tracker gains are invalid")
        if not 0.0 <= self.minimum_linear_fraction <= 1.0:
            raise ValueError("minimum_linear_fraction must be in [0, 1]")
        if np.any(self.slew_limits <= 0.0):
            raise ValueError("slew limits must be positive")

    @staticmethod
    def _vector(value, size, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                "{} must be a finite shape-{} vector".format(name, size)
            )
        return vector
