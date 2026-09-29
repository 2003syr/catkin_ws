#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert relative joint high-level commands into fixed world targets."""

from __future__ import division

import math

import numpy as np


class FixedJointSubgoal(object):
    """World-frame targets held constant for one high-level period."""

    def __init__(
            self,
            base_goal_world,
            base_yaw_world,
            ee_goal_world,
            original_command):
        self.base_goal_world = self._vector(
            base_goal_world, 2, "base_goal_world"
        )
        self.base_yaw_world = self.wrap_angle(base_yaw_world)
        self.ee_goal_world = self._vector(
            ee_goal_world, 3, "ee_goal_world"
        )
        self.original_command = original_command

    def remaining(self, base_pose_world, ee_position_world):
        """Return the live 6-D error expected by the fused low level."""
        base_pose = self._vector(
            base_pose_world, 3, "base_pose_world"
        )
        ee_position = self._vector(
            ee_position_world, 3, "ee_position_world"
        )
        base_error_world = self.base_goal_world - base_pose[0:2]
        base_error_body = self.rotate_xy(
            base_error_world,
            -base_pose[2],
        )
        yaw_error = self.wrap_angle(
            self.base_yaw_world - base_pose[2]
        )
        ee_error_world = self.ee_goal_world - ee_position
        return np.concatenate((
            base_error_body,
            np.asarray([yaw_error], dtype=np.float64),
            ee_error_world,
        )).astype(np.float32)

    @staticmethod
    def rotate_xy(vector, angle):
        vector = np.asarray(vector, dtype=np.float64)
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        return np.asarray([
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ], dtype=np.float64)

    @staticmethod
    def wrap_angle(value):
        return (float(value) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _vector(value, size, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name,
                    size,
                    vector.shape,
                )
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("{} contains non-finite values".format(name))
        return vector.copy()


class SubgoalConverter(object):
    """Freeze body-relative base/EE displacements in the world frame."""

    @classmethod
    def convert(cls, command, base_pose_world, ee_position_world):
        base_pose = FixedJointSubgoal._vector(
            base_pose_world, 3, "base_pose_world"
        )
        ee_position = FixedJointSubgoal._vector(
            ee_position_world, 3, "ee_position_world"
        )
        base_delta_world = FixedJointSubgoal.rotate_xy(
            command.base_goal[0:2],
            base_pose[2],
        )
        ee_delta_xy_world = FixedJointSubgoal.rotate_xy(
            command.ee_goal[0:2],
            base_pose[2],
        )
        base_goal_world = base_pose[0:2] + base_delta_world
        base_yaw_world = FixedJointSubgoal.wrap_angle(
            base_pose[2] + float(command.base_goal[2])
        )
        ee_goal_world = ee_position.copy()
        ee_goal_world[0:2] += ee_delta_xy_world
        ee_goal_world[2] += float(command.ee_goal[2])
        return FixedJointSubgoal(
            base_goal_world=base_goal_world,
            base_yaw_world=base_yaw_world,
            ee_goal_world=ee_goal_world,
            original_command=command,
        )
