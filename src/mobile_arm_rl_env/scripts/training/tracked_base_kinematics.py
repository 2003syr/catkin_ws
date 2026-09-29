#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pure kinematic transforms for a tracked/unicycle planar base."""

from __future__ import print_function

import math

import numpy as np


# The CAD chassis longitudinal +X axis is mounted at -90 degrees relative to
# z-link by the fixed sway joint.  Keep the assembly unchanged and make this
# offset explicit in every policy/body-frame transform.
TRACKED_FORWARD_YAW_OFFSET = -0.5 * math.pi


def wrapped_angle_difference(value, reference):
    """Return value-reference on the principal [-pi, pi] interval."""
    value = float(value)
    reference = float(reference)
    if not np.isfinite(value) or not np.isfinite(reference):
        raise ValueError("angles must be finite")
    return float(math.atan2(
        math.sin(value - reference),
        math.cos(value - reference),
    ))


def unicycle_to_world_action(
        action,
        yaw,
        action_scale=1.0,
        forward_yaw_offset=TRACKED_FORWARD_YAW_OFFSET):
    """Map normalized [linear, yaw-rate] to normalized [xdot, ydot, yawdot]."""
    action = _vector(action, 2, "action")
    yaw = float(yaw)
    action_scale = float(action_scale)
    if not np.isfinite(yaw):
        raise ValueError("yaw must be finite")
    if not 0.0 < action_scale <= 1.0:
        raise ValueError("action_scale must be in (0, 1]")
    heading = yaw + float(forward_yaw_offset)
    linear = float(np.clip(action[0], -1.0, 1.0)) * action_scale
    yaw_rate = float(np.clip(action[1], -1.0, 1.0)) * action_scale
    return np.asarray([
        linear * np.cos(heading),
        linear * np.sin(heading),
        yaw_rate,
    ], dtype=np.float32)


def world_velocity_to_body(
        world_velocity,
        yaw,
        forward_yaw_offset=TRACKED_FORWARD_YAW_OFFSET):
    """Return [forward, lateral, yaw-rate] from world-frame pose rates."""
    velocity = _vector(world_velocity, 3, "world_velocity")
    yaw = float(yaw)
    if not np.isfinite(yaw):
        raise ValueError("yaw must be finite")
    heading = yaw + float(forward_yaw_offset)
    cosine = np.cos(heading)
    sine = np.sin(heading)
    return np.asarray([
        cosine * velocity[0] + sine * velocity[1],
        -sine * velocity[0] + cosine * velocity[1],
        velocity[2],
    ], dtype=np.float64)


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
    return vector
