#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Observable-only teacher for pose-guided local-subgoal tracking."""

from __future__ import print_function

import math

import numpy as np


class RuleBasedPlanarTracker(object):
    """Map the same 62-D observation seen by the learner to [v, omega].

    The controller contains no obstacle planner, map, global path, or hidden
    state.  It is deliberately limited to teaching nonholonomic goal tracking;
    PPO remains responsible for learning when to depart from this behavior.
    """

    OBS_DIM = 62
    ACTION_DIM = 2

    def __init__(
            self,
            heading_gain=1.6,
            slow_distance=0.35,
            turn_in_place_angle=0.35,
            success_distance=0.08,
            success_yaw_error=0.12,
            linear_slew_limit=0.12,
            angular_slew_limit=0.20):
        self.heading_gain = float(heading_gain)
        self.slow_distance = float(slow_distance)
        self.turn_in_place_angle = float(turn_in_place_angle)
        self.success_distance = float(success_distance)
        self.success_yaw_error = float(success_yaw_error)
        self.slew_limits = np.asarray([
            linear_slew_limit,
            angular_slew_limit,
        ], dtype=np.float32)
        self._validate()

    def predict(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.OBS_DIM,):
            raise ValueError(
                "planar tracker observation must have shape (62,)"
            )
        if not np.all(np.isfinite(observation)):
            raise ValueError("planar tracker observation is non-finite")

        goal_x, goal_y = observation[0:2]
        yaw_error = float(math.atan2(
            float(observation[2]),
            float(observation[3]),
        ))
        previous_action = observation[55:57]
        distance = float(math.hypot(goal_x, goal_y))
        position_heading_error = float(math.atan2(goal_y, goal_x))
        orientation_weight = float(np.clip(
            1.0 - distance / self.slow_distance,
            0.0,
            1.0,
        ))
        heading_error = float(math.atan2(
            (1.0 - orientation_weight)
            * math.sin(position_heading_error)
            + orientation_weight * math.sin(yaw_error),
            (1.0 - orientation_weight)
            * math.cos(position_heading_error)
            + orientation_weight * math.cos(yaw_error),
        ))

        if distance <= self.success_distance:
            if abs(yaw_error) <= self.success_yaw_error:
                desired = np.zeros(self.ACTION_DIM, dtype=np.float32)
            else:
                desired = np.asarray([
                    0.0,
                    np.clip(
                        self.heading_gain * yaw_error,
                        -1.0,
                        1.0,
                    ),
                ], dtype=np.float32)
        else:
            angular = float(np.clip(
                self.heading_gain * heading_error,
                -1.0,
                1.0,
            ))
            if abs(heading_error) >= self.turn_in_place_angle:
                linear = 0.0
            else:
                alignment = max(0.0, math.cos(heading_error))
                linear = min(1.0, distance / self.slow_distance)
                linear *= alignment
            desired = np.asarray([linear, angular], dtype=np.float32)

        delta = np.clip(
            desired - previous_action,
            -self.slew_limits,
            self.slew_limits,
        )
        action = np.clip(
            previous_action + delta,
            -1.0,
            1.0,
        )
        return action.astype(np.float32)

    def _validate(self):
        if self.heading_gain <= 0.0:
            raise ValueError("heading_gain must be positive")
        if self.slow_distance <= self.success_distance:
            raise ValueError(
                "slow_distance must exceed success_distance"
            )
        if not 0.0 < self.turn_in_place_angle <= math.pi:
            raise ValueError(
                "turn_in_place_angle must be in (0, pi]"
            )
        if self.success_distance <= 0.0:
            raise ValueError("success_distance must be positive")
        if not 0.0 < self.success_yaw_error <= math.pi:
            raise ValueError("success_yaw_error must be in (0, pi]")
        if np.any(self.slew_limits <= 0.0):
            raise ValueError("slew limits must be positive")
