#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared contracts for coordinated tracked-base and arm control.

This module intentionally does not import ROS.  The policy-facing action is

    [v, omega, joint1, joint2, joint3, joint4, joint5, joint6]

where every component is normalized to [-1, 1].  ``v`` is a tracked-base
forward velocity, not a virtual lateral joint command.
"""

from __future__ import division

import math

import numpy as np


TRACKED_FORWARD_YAW_OFFSET = -0.5 * math.pi


class FusedLowLevelState(object):
    """Build the fixed 66-dimensional coordinated low-level observation."""

    SENSOR_DIM = 46
    SUBGOAL_DIM = 6
    ACTION_DIM = 8
    INPUT_DIM = SENSOR_DIM + SUBGOAL_DIM + SUBGOAL_DIM + ACTION_DIM

    TARGET_POSITION_SLICE = slice(0, 3)
    END_EFFECTOR_POSITION_SLICE = slice(6, 9)
    JOINT_POSITION_SLICE = slice(11, 21)

    SENSOR_SLICE = slice(0, 46)
    SUBGOAL_SLICE = slice(46, 52)
    SUBGOAL_MASK_SLICE = slice(52, 58)
    ACTION_MASK_SLICE = slice(58, 66)

    # Public action-support masks used by the fused box task.  Action order is
    # [tracked_v, tracked_omega, joint1..joint6].
    ACTION_MASK_PHASES = (
        "BASE_APPROACH",
        "COORDINATED",
        "ARM_FINISH",
    )

    def __init__(self, forward_yaw_offset=TRACKED_FORWARD_YAW_OFFSET):
        self.forward_yaw_offset = float(forward_yaw_offset)
        self.subgoal_mask = np.ones(self.SUBGOAL_DIM, dtype=np.float32)
        self.action_mask = np.ones(self.ACTION_DIM, dtype=np.float32)
        self.last_diagnostics = {}

    def build(self, observation, base_goal_xy, action_mask=None):
        sensor = self.as_sensor(observation)
        base_goal_xy = self._vector(base_goal_xy, 2, "base_goal_xy")
        if action_mask is None:
            action_mask = self.action_mask
        action_mask = self._mask(action_mask, self.ACTION_DIM)
        q = np.asarray(
            sensor[self.JOINT_POSITION_SLICE],
            dtype=np.float64,
        )
        base_xy = q[0:2]
        planar_yaw = float(q[2])
        heading = planar_yaw + self.forward_yaw_offset

        base_error_fixed = base_goal_xy - base_xy
        base_error_body = self._rotate_xy(base_error_fixed, -heading)
        heading_error = self.wrap_angle(
            math.atan2(base_error_fixed[1], base_error_fixed[0])
            - heading
        )
        if np.linalg.norm(base_error_fixed) < 1.0e-6:
            heading_error = 0.0

        target_position = np.asarray(
            sensor[self.TARGET_POSITION_SLICE],
            dtype=np.float64,
        )
        end_effector_position = np.asarray(
            sensor[self.END_EFFECTOR_POSITION_SLICE],
            dtype=np.float64,
        )
        arm_error_fixed = target_position - end_effector_position
        remaining_subgoal = np.concatenate((
            base_error_body,
            np.asarray([heading_error], dtype=np.float64),
            arm_error_fixed,
        )).astype(np.float32)

        vector = np.concatenate((
            sensor,
            remaining_subgoal,
            self.subgoal_mask,
            action_mask,
        )).astype(np.float32)
        if vector.shape != (self.INPUT_DIM,):
            raise RuntimeError(
                "fused observation has shape {}, expected ({},)".format(
                    vector.shape,
                    self.INPUT_DIM,
                )
            )

        self.last_diagnostics = {
            "base_goal_xy": base_goal_xy.copy(),
            "base_xy": base_xy.copy(),
            "planar_yaw": planar_yaw,
            "tracked_heading": heading,
            "base_error_fixed": base_error_fixed.copy(),
            "base_error_body": base_error_body.copy(),
            "base_distance": float(np.linalg.norm(base_error_fixed)),
            "heading_error": float(heading_error),
            "arm_error_fixed": arm_error_fixed.copy(),
            "arm_distance": float(np.linalg.norm(arm_error_fixed)),
            "remaining_subgoal": remaining_subgoal.copy(),
            "action_mask": action_mask.copy(),
        }
        return vector

    @classmethod
    def action_mask_for_phase(cls, phase):
        """Return the binary action support for a fused control phase."""
        phase = str(phase).upper()
        if phase == "BASE_APPROACH":
            return np.asarray(
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
        if phase == "COORDINATED":
            return np.ones(cls.ACTION_DIM, dtype=np.float32)
        if phase == "ARM_FINISH":
            return np.asarray(
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            )
        raise ValueError(
            "unsupported fused action-mask phase: {}".format(phase)
        )

    def diagnostics(self):
        return self._copy_dict(self.last_diagnostics)

    @classmethod
    def as_sensor(cls, observation):
        if isinstance(observation, dict):
            observation = observation["obs_vec"]
        sensor = np.asarray(observation, dtype=np.float32)
        if sensor.shape != (cls.SENSOR_DIM,):
            raise ValueError(
                "sensor observation must have shape (46,), got {}".format(
                    sensor.shape
                )
            )
        if not np.all(np.isfinite(sensor)):
            raise ValueError("sensor observation contains non-finite values")
        return sensor

    @staticmethod
    def wrap_angle(angle):
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _rotate_xy(vector, angle):
        vector = np.asarray(vector, dtype=np.float64)
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        return np.asarray([
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ], dtype=np.float64)

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
        return vector

    @staticmethod
    def _mask(value, size):
        mask = np.asarray(value, dtype=np.float32)
        if mask.shape != (size,):
            raise ValueError(
                "action mask must have shape ({},), got {}".format(
                    size,
                    mask.shape,
                )
            )
        if not np.all(np.isfinite(mask)):
            raise ValueError("action mask contains non-finite values")
        if np.any(mask < 0.0) or np.any(mask > 1.0):
            raise ValueError("action mask values must be in [0, 1]")
        return mask.copy()

    @staticmethod
    def _copy_dict(values):
        return dict(
            (
                key,
                value.copy() if hasattr(value, "copy") else value,
            )
            for key, value in values.items()
        )


class FusedActionAdapter(object):
    """Map normalized [v, omega, arm6] into the historical 10-D executor."""

    ACTION_DIM = 8
    FULL_ACTION_DIM = 10

    def __init__(
            self,
            linear_action_scale=0.25,
            yaw_action_scale=0.25,
            forward_yaw_offset=TRACKED_FORWARD_YAW_OFFSET):
        self.linear_action_scale = float(linear_action_scale)
        self.yaw_action_scale = float(yaw_action_scale)
        self.forward_yaw_offset = float(forward_yaw_offset)
        if not 0.0 < self.linear_action_scale <= 1.0:
            raise ValueError("linear_action_scale must be in (0, 1]")
        if not 0.0 < self.yaw_action_scale <= 1.0:
            raise ValueError("yaw_action_scale must be in (0, 1]")

    def to_full_action(self, action, observation):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.ACTION_DIM,):
            raise ValueError(
                "fused action must have shape (8,), got {}".format(
                    action.shape
                )
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("fused action contains non-finite values")
        action = np.clip(action, -1.0, 1.0)
        sensor = FusedLowLevelState.as_sensor(observation)
        q = sensor[FusedLowLevelState.JOINT_POSITION_SLICE]
        heading = float(q[2]) + self.forward_yaw_offset

        forward = float(action[0]) * self.linear_action_scale
        full_action = np.zeros(self.FULL_ACTION_DIM, dtype=np.float32)
        full_action[0] = forward * math.cos(heading)
        full_action[1] = forward * math.sin(heading)
        full_action[2] = float(action[1]) * self.yaw_action_scale
        full_action[3] = 0.0
        full_action[4:10] = action[2:8]
        return full_action

    def from_full_action(self, full_action, observation):
        """Recover normalized fused coordinates from a filtered 10-D action.

        The historical safety layer filters commands in the virtual-joint
        executor.  Dataset collection needs the action that was actually
        allowed through that layer, expressed in the policy-facing
        ``[v, omega, arm6]`` coordinates.
        """
        full_action = np.asarray(full_action, dtype=np.float32)
        if full_action.shape != (self.FULL_ACTION_DIM,):
            raise ValueError(
                "full action must have shape (10,), got {}".format(
                    full_action.shape
                )
            )
        if not np.all(np.isfinite(full_action)):
            raise ValueError("full action contains non-finite values")
        sensor = FusedLowLevelState.as_sensor(observation)
        q = sensor[FusedLowLevelState.JOINT_POSITION_SLICE]
        heading = float(q[2]) + self.forward_yaw_offset

        forward = (
            float(full_action[0]) * math.cos(heading)
            + float(full_action[1]) * math.sin(heading)
        )
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        action[0] = forward / self.linear_action_scale
        action[1] = float(full_action[2]) / self.yaw_action_scale
        action[2:8] = full_action[4:10]
        return np.clip(action, -1.0, 1.0)


class CoordinatedRuleTeacher(object):
    """Obstacle-free teacher that overlaps base tracking and DLS arm reach."""

    ACTION_DIM = 8
    ARM_ACTION_SLICE = slice(2, 8)
    JOINT_POSITION_SLICE = FusedLowLevelState.JOINT_POSITION_SLICE

    DEFAULT_ARM_JOINT_MAX_VELOCITY = np.asarray(
        [0.50, 0.50, 0.03, 0.50, 0.03, 0.50],
        dtype=np.float64,
    )
    DEFAULT_ARM_JOINT_LIMITS = np.asarray([
        [-1.57, 1.57],
        [-1.45, 0.00],
        [0.00, 0.15],
        [-1.57, 1.57],
        [0.00, 0.15],
        [-1.57, 1.57],
    ], dtype=np.float64)
    DEFAULT_ARM_SOFT_MARGINS = np.asarray(
        [0.30, 0.20, 0.05, 0.30, 0.05, 0.30],
        dtype=np.float64,
    )
    ARM_JOINT_NAMES = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )

    def __init__(
            self,
            jacobian_provider,
            base_gain=2.5,
            yaw_gain=1.5,
            arm_gain=1.0,
            arm_start_distance=0.30,
            base_stop_distance=0.04,
            subgoal_base_stop_distance=0.02,
            heading_slow_angle=0.70,
            max_cartesian_speed=0.05,
            dls_damping=0.03,
            arm_joint_max_velocity=None,
            arm_joint_limits=None,
            arm_soft_margins=None,
            joint_limit_avoidance_gain=0.0,
            joint_limit_avoidance_activation=1.5):
        if jacobian_provider is None:
            raise ValueError("jacobian_provider is required")
        self.jacobian_provider = jacobian_provider
        self.base_gain = float(base_gain)
        self.yaw_gain = float(yaw_gain)
        self.arm_gain = float(arm_gain)
        self.arm_start_distance = float(arm_start_distance)
        self.base_stop_distance = float(base_stop_distance)
        self.subgoal_base_stop_distance = float(
            subgoal_base_stop_distance
        )
        self.heading_slow_angle = float(heading_slow_angle)
        self.max_cartesian_speed = float(max_cartesian_speed)
        self.dls_damping = float(dls_damping)
        self.arm_joint_max_velocity = np.asarray(
            self.DEFAULT_ARM_JOINT_MAX_VELOCITY
            if arm_joint_max_velocity is None
            else arm_joint_max_velocity,
            dtype=np.float64,
        )
        self.arm_joint_limits = np.asarray(
            self.DEFAULT_ARM_JOINT_LIMITS
            if arm_joint_limits is None
            else arm_joint_limits,
            dtype=np.float64,
        )
        self.arm_soft_margins = np.asarray(
            self.DEFAULT_ARM_SOFT_MARGINS
            if arm_soft_margins is None
            else arm_soft_margins,
            dtype=np.float64,
        )
        self.joint_limit_avoidance_gain = float(
            joint_limit_avoidance_gain
        )
        self.joint_limit_avoidance_activation = float(
            joint_limit_avoidance_activation
        )
        if self.arm_joint_max_velocity.shape != (6,):
            raise ValueError("arm_joint_max_velocity must have shape (6,)")
        if np.any(self.arm_joint_max_velocity <= 0.0):
            raise ValueError("arm joint velocity limits must be positive")
        if self.arm_joint_limits.shape != (6, 2):
            raise ValueError("arm_joint_limits must have shape (6, 2)")
        if np.any(
                self.arm_joint_limits[:, 1]
                <= self.arm_joint_limits[:, 0]):
            raise ValueError("arm joint upper limits must exceed lower limits")
        if self.arm_soft_margins.shape != (6,):
            raise ValueError("arm_soft_margins must have shape (6,)")
        if np.any(self.arm_soft_margins <= 0.0):
            raise ValueError("arm soft margins must be positive")
        if np.any(
                2.0 * self.arm_soft_margins
                >= (
                    self.arm_joint_limits[:, 1]
                    - self.arm_joint_limits[:, 0]
                )):
            raise ValueError(
                "arm soft margins must leave a non-empty interior"
            )
        if self.base_gain <= 0.0 or self.yaw_gain <= 0.0:
            raise ValueError("base and yaw gains must be positive")
        if self.arm_gain <= 0.0 or self.max_cartesian_speed <= 0.0:
            raise ValueError("arm gains and speed must be positive")
        if self.arm_start_distance <= self.base_stop_distance:
            raise ValueError(
                "arm_start_distance must exceed base_stop_distance"
            )
        if not (
                0.0 < self.subgoal_base_stop_distance
                <= self.base_stop_distance):
            raise ValueError(
                "subgoal_base_stop_distance must be in (0, base_stop_distance]"
            )
        if self.heading_slow_angle <= 0.0:
            raise ValueError("heading_slow_angle must be positive")
        if self.dls_damping <= 0.0:
            raise ValueError("dls_damping must be positive")
        if self.joint_limit_avoidance_gain < 0.0:
            raise ValueError(
                "joint_limit_avoidance_gain must be non-negative"
            )
        if self.joint_limit_avoidance_activation < 1.0:
            raise ValueError(
                "joint_limit_avoidance_activation must be at least 1.0"
            )
        self.last_diagnostics = {}

    def predict(self, fused_observation):
        fused_observation = np.asarray(
            fused_observation,
            dtype=np.float32,
        )
        if fused_observation.shape != (FusedLowLevelState.INPUT_DIM,):
            raise ValueError(
                "fused observation must have shape (66,), got {}".format(
                    fused_observation.shape
                )
            )
        sensor = fused_observation[FusedLowLevelState.SENSOR_SLICE]
        subgoal = np.asarray(
            fused_observation[FusedLowLevelState.SUBGOAL_SLICE],
            dtype=np.float64,
        )
        base_error_body = subgoal[0:2]
        heading_error = float(subgoal[2])
        arm_error = subgoal[3:6]
        base_distance = float(np.linalg.norm(base_error_body))

        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        if base_distance > self.base_stop_distance:
            heading_scale = max(
                0.0,
                math.cos(min(abs(heading_error), 0.5 * math.pi)),
            )
            if abs(heading_error) > self.heading_slow_angle:
                heading_scale = 0.0
            action[0] = float(np.clip(
                self.base_gain * base_distance * heading_scale,
                0.0,
                1.0,
            ))
            action[1] = float(np.clip(
                self.yaw_gain * heading_error,
                -1.0,
                1.0,
            ))

        arm_blend = float(np.clip(
            (
                self.arm_start_distance - base_distance
            ) / (
                self.arm_start_distance - self.base_stop_distance
            ),
            0.0,
            1.0,
        ))
        (
            arm_action,
            condition,
            cartesian_velocity,
            joint_limit_diagnostics,
        ) = self._arm_action(
            sensor,
            arm_error,
            enable_joint_limit_avoidance=(
                base_distance <= self.base_stop_distance
            ),
        )
        action[self.ARM_ACTION_SLICE] = (
            arm_blend * arm_action
        ).astype(np.float32)
        action = np.clip(action, -1.0, 1.0)

        if base_distance <= self.base_stop_distance:
            phase = "ARM_FINISH"
        elif arm_blend > 0.0:
            phase = "COORDINATED"
        else:
            phase = "BASE_APPROACH"
        self.last_diagnostics = {
            "phase": phase,
            "base_distance": base_distance,
            "heading_error": heading_error,
            "arm_distance": float(np.linalg.norm(arm_error)),
            "arm_blend": arm_blend,
            "cartesian_velocity": cartesian_velocity.copy(),
            "jacobian_condition": condition,
            "joint_limit_blocked": list(
                joint_limit_diagnostics["blocked_joints"]
            ),
            "unconstrained_arm_action": (
                joint_limit_diagnostics[
                    "unconstrained_arm_action"
                ].copy()
            ),
            "joint_limit_avoidance_applied": bool(
                joint_limit_diagnostics.get(
                    "joint_limit_avoidance_applied", False
                )
            ),
            "joint_limit_avoidance_action": np.asarray(
                joint_limit_diagnostics.get(
                    "joint_limit_avoidance_action", np.zeros(6)
                ),
                dtype=np.float32,
            ).copy(),
            "action": action.copy(),
        }
        return action

    def subgoal_components(
            self,
            fused_observation,
            arm_enabled=True,
            terminal_yaw_tolerance=0.08):
        """Build a closed-loop action for one fixed joint subgoal.

        The original :meth:`predict` method is intentionally compatible with
        the historical fused teacher.  It uses the final-heading error to
        modulate forward speed and therefore assumes that the goal is always
        in front of the tracked base.  That assumption is not valid for HRL
        waypoints: a short waypoint can be behind the chassis after an
        overshoot.  The HRL teacher therefore uses the live body-frame
        bearing ``atan2(error_y, error_x)`` and a small hybrid controller:

        * rotate in place while the waypoint bearing is large;
        * drive forward only after the bearing is aligned;
        * rotate to the requested final yaw after the position is reached;
        * release the arm only in the stable terminal phase.

        This keeps the historical forward-only action contract while making
        the fixed subgoal reachable without relying on a negative velocity.
        """
        fused_observation = np.asarray(
            fused_observation,
            dtype=np.float32,
        )
        if fused_observation.shape != (FusedLowLevelState.INPUT_DIM,):
            raise ValueError(
                "fused observation must have shape (66,), got {}".format(
                    fused_observation.shape
                )
            )
        sensor = fused_observation[FusedLowLevelState.SENSOR_SLICE]
        subgoal = np.asarray(
            fused_observation[FusedLowLevelState.SUBGOAL_SLICE],
            dtype=np.float64,
        )
        base_error_body = subgoal[0:2]
        yaw_error = FusedLowLevelState.wrap_angle(float(subgoal[2]))
        arm_error = subgoal[3:6]
        base_distance = float(np.linalg.norm(base_error_body))
        subgoal_stop_distance = self.subgoal_base_stop_distance
        bearing_error = (
            0.0
            if base_distance <= 1.0e-9
            else math.atan2(
                float(base_error_body[1]),
                float(base_error_body[0]),
            )
        )
        terminal_yaw_tolerance = float(terminal_yaw_tolerance)
        if terminal_yaw_tolerance <= 0.0:
            raise ValueError("terminal_yaw_tolerance must be positive")

        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        phase = "SUBGOAL_HOLD"
        if base_distance > subgoal_stop_distance:
            if abs(bearing_error) > self.heading_slow_angle:
                phase = "SUBGOAL_ALIGN"
                action[0] = 0.0
            else:
                phase = "SUBGOAL_DRIVE"
                action[0] = float(np.clip(
                    self.base_gain * max(
                        base_distance - subgoal_stop_distance,
                        0.0,
                    ),
                    0.0,
                    1.0,
                ))
            action[1] = float(np.clip(
                self.yaw_gain * bearing_error,
                -1.0,
                1.0,
            ))
        elif abs(yaw_error) > terminal_yaw_tolerance:
            phase = "SUBGOAL_ROTATE"
            action[0] = 0.0
            action[1] = float(np.clip(
                self.yaw_gain * yaw_error,
                -1.0,
                1.0,
            ))
        elif bool(arm_enabled):
            phase = "ARM_FINISH"

        arm_action = np.zeros(6, dtype=np.float32)
        cartesian_velocity = np.zeros(3, dtype=np.float64)
        condition = float("inf")
        joint_limit_diagnostics = {
            "blocked_joints": [],
            "unconstrained_arm_action": np.zeros(6, dtype=np.float32),
            "joint_limit_avoidance_applied": False,
            "joint_limit_avoidance_action": np.zeros(6, dtype=np.float32),
        }
        if bool(arm_enabled) and phase == "ARM_FINISH":
            (
                arm_action,
                condition,
                cartesian_velocity,
                joint_limit_diagnostics,
            ) = self._arm_action(
                sensor,
                arm_error,
                enable_joint_limit_avoidance=True,
            )
            action[self.ARM_ACTION_SLICE] = arm_action

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.last_diagnostics = {
            "phase": phase,
            "base_distance": base_distance,
            "base_stop_distance": subgoal_stop_distance,
            "heading_error": yaw_error,
            "bearing_error": float(bearing_error),
            "arm_distance": float(np.linalg.norm(arm_error)),
            "arm_blend": 1.0 if phase == "ARM_FINISH" else 0.0,
            "cartesian_velocity": np.asarray(
                cartesian_velocity,
                dtype=np.float64,
            ).copy(),
            "jacobian_condition": condition,
            "joint_limit_blocked": list(
                joint_limit_diagnostics.get("blocked_joints", [])
            ),
            "unconstrained_arm_action": np.asarray(
                joint_limit_diagnostics.get(
                    "unconstrained_arm_action",
                    np.zeros(6),
                ),
                dtype=np.float32,
            ).copy(),
            "joint_limit_avoidance_applied": bool(
                joint_limit_diagnostics.get(
                    "joint_limit_avoidance_applied", False
                )
            ),
            "joint_limit_avoidance_action": np.asarray(
                joint_limit_diagnostics.get(
                    "joint_limit_avoidance_action", np.zeros(6)
                ),
                dtype=np.float32,
            ).copy(),
            "action": action.copy(),
            "arm_enabled": bool(arm_enabled),
        }
        return {
            "sensor": np.asarray(sensor, dtype=np.float32).copy(),
            "base_action": action[0:2].copy(),
            "nominal_action": action.copy(),
            "cartesian_velocity": np.asarray(
                cartesian_velocity,
                dtype=np.float64,
            ).copy(),
            "arm_action": np.asarray(arm_action, dtype=np.float32).copy(),
            "arm_blend": 1.0 if phase == "ARM_FINISH" else 0.0,
            "phase": phase,
            "base_distance": base_distance,
            "base_stop_distance": subgoal_stop_distance,
            "heading_error": yaw_error,
            "bearing_error": float(bearing_error),
            "jacobian_condition": condition,
            "joint_limit_diagnostics": joint_limit_diagnostics,
        }

    def subgoal_residual_components(
            self,
            fused_observation,
            residual,
            base_scale=0.10,
            cartesian_scale=0.01,
            arm_enabled=True):
        """Compose a bounded residual around the HRL subgoal tracker."""
        residual = np.asarray(residual, dtype=np.float64)
        if residual.shape != (5,):
            raise ValueError(
                "residual must have shape (5,), got {}".format(
                    residual.shape
                )
            )
        if not np.all(np.isfinite(residual)):
            raise ValueError("residual contains non-finite values")
        residual = np.clip(residual, -1.0, 1.0)
        components = self.subgoal_components(
            fused_observation,
            arm_enabled=arm_enabled,
        )
        phase = str(components["phase"])
        base_action = np.asarray(
            components["base_action"],
            dtype=np.float64,
        ) + float(base_scale) * residual[0:2]
        base_action[0] = np.clip(base_action[0], 0.0, 1.0)
        base_action[1] = np.clip(base_action[1], -1.0, 1.0)
        if phase in ("SUBGOAL_ALIGN", "SUBGOAL_ROTATE", "ARM_FINISH", "SUBGOAL_HOLD"):
            if phase != "SUBGOAL_DRIVE":
                base_action[0] = 0.0
        if phase == "ARM_FINISH":
            base_action[0:2] = 0.0

        cartesian_velocity = np.asarray(
            components["cartesian_velocity"],
            dtype=np.float64,
        ).copy()
        if phase == "ARM_FINISH" and bool(arm_enabled):
            cartesian_velocity += float(cartesian_scale) * residual[2:5]
            speed = float(np.linalg.norm(cartesian_velocity))
            if speed > self.max_cartesian_speed:
                cartesian_velocity *= (
                    self.max_cartesian_speed / speed
                )
            arm_action, condition, _, joint_limit_diagnostics = (
                self.solve_cartesian_velocity(
                    components["sensor"],
                    cartesian_velocity,
                    enable_joint_limit_avoidance=True,
                )
            )
            arm_blend = 1.0
        else:
            arm_action = np.zeros(6, dtype=np.float32)
            condition = components["jacobian_condition"]
            joint_limit_diagnostics = components[
                "joint_limit_diagnostics"
            ]
            arm_blend = 0.0

        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        action[0:2] = base_action.astype(np.float32)
        action[self.ARM_ACTION_SLICE] = (
            arm_blend * np.asarray(arm_action, dtype=np.float32)
        )
        self.last_diagnostics.update({
            "cartesian_velocity": cartesian_velocity.copy(),
            "jacobian_condition": condition,
            "joint_limit_blocked": list(
                joint_limit_diagnostics.get("blocked_joints", [])
            ),
            "unconstrained_arm_action": np.asarray(
                joint_limit_diagnostics.get(
                    "unconstrained_arm_action", np.zeros(6)
                ),
                dtype=np.float32,
            ).copy(),
            "joint_limit_avoidance_applied": bool(
                joint_limit_diagnostics.get(
                    "joint_limit_avoidance_applied", False
                )
            ),
            "joint_limit_avoidance_action": np.asarray(
                joint_limit_diagnostics.get(
                    "joint_limit_avoidance_action", np.zeros(6)
                ),
                dtype=np.float32,
            ).copy(),
            "action": np.clip(action, -1.0, 1.0).copy(),
        })
        return {
            # Terminal stall recovery consumes the exact sensor state used
            # to build this action.  Keep it in the public component contract
            # just as subgoal_components() and predict_components() do.
            "sensor": np.asarray(
                components["sensor"],
                dtype=np.float32,
            ).copy(),
            "action": np.clip(action, -1.0, 1.0),
            "nominal_action": components["nominal_action"].copy(),
            "residual": residual.astype(np.float32),
            "base_action": base_action.astype(np.float32),
            "cartesian_velocity": cartesian_velocity.copy(),
            "arm_action": np.asarray(arm_action, dtype=np.float32).copy(),
            "arm_blend": arm_blend,
            "phase": phase,
            "base_distance": components["base_distance"],
            "heading_error": components["heading_error"],
            "bearing_error": components["bearing_error"],
            "jacobian_condition": condition,
            "joint_limit_diagnostics": joint_limit_diagnostics,
        }

    def diagnostics(self):
        return FusedLowLevelState._copy_dict(self.last_diagnostics)

    def predict_components(self, fused_observation):
        """Expose the nominal commands for a bounded residual policy."""
        nominal_action = self.predict(fused_observation)
        diagnostics = self.diagnostics()
        base_distance = float(diagnostics["base_distance"])
        arm_blend = float(np.clip(
            (self.arm_start_distance - base_distance)
            / (self.arm_start_distance - self.base_stop_distance),
            0.0,
            1.0,
        ))
        sensor = np.asarray(
            fused_observation[FusedLowLevelState.SENSOR_SLICE],
            dtype=np.float32,
        )
        return {
            "sensor": sensor.copy(),
            "base_action": nominal_action[0:2].copy(),
            "nominal_action": nominal_action.copy(),
            "cartesian_velocity": np.asarray(
                diagnostics["cartesian_velocity"],
                dtype=np.float64,
            ).copy(),
            "arm_blend": arm_blend,
            "phase": diagnostics["phase"],
            "base_distance": base_distance,
            "heading_error": float(diagnostics["heading_error"]),
        }

    def residual_components(
            self,
            fused_observation,
            residual,
            base_scale=0.10,
            cartesian_scale=0.01):
        """Apply a bounded [dv, domega, dvx, dvy, dvz] residual."""
        residual = np.asarray(residual, dtype=np.float64)
        if residual.shape != (5,):
            raise ValueError(
                "residual must have shape (5,), got {}".format(
                    residual.shape
                )
            )
        if not np.all(np.isfinite(residual)):
            raise ValueError("residual contains non-finite values")
        residual = np.clip(residual, -1.0, 1.0)
        components = self.predict_components(fused_observation)

        base_action = np.asarray(
            components["base_action"],
            dtype=np.float64,
        ) + float(base_scale) * residual[0:2]
        base_action[0] = np.clip(base_action[0], 0.0, 1.0)
        base_action[1] = np.clip(base_action[1], -1.0, 1.0)

        cartesian_velocity = np.asarray(
            components["cartesian_velocity"],
            dtype=np.float64,
        ) + float(cartesian_scale) * residual[2:5]
        speed = float(np.linalg.norm(cartesian_velocity))
        if speed > self.max_cartesian_speed:
            cartesian_velocity *= self.max_cartesian_speed / speed
        arm_action, condition, _, joint_limit_diagnostics = (
            self.solve_cartesian_velocity(
                components["sensor"],
                cartesian_velocity,
            )
        )
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        action[0:2] = base_action.astype(np.float32)
        action[self.ARM_ACTION_SLICE] = (
            components["arm_blend"] * arm_action
        ).astype(np.float32)
        return {
            "action": np.clip(action, -1.0, 1.0),
            "nominal_action": components["nominal_action"].copy(),
            "residual": residual.astype(np.float32),
            "base_action": base_action.astype(np.float32),
            "cartesian_velocity": cartesian_velocity.copy(),
            "arm_action": arm_action.copy(),
            "arm_blend": components["arm_blend"],
            "phase": components["phase"],
            "jacobian_condition": condition,
            "joint_limit_diagnostics": joint_limit_diagnostics,
        }

    def predict_residual(self, fused_observation, residual):
        return self.residual_components(
            fused_observation,
            residual,
        )["action"]

    def _arm_action(
            self,
            sensor,
            arm_error,
            enable_joint_limit_avoidance=False):
        cartesian_velocity = self.arm_gain * np.asarray(
            arm_error,
            dtype=np.float64,
        )
        speed = float(np.linalg.norm(cartesian_velocity))
        if speed > self.max_cartesian_speed:
            cartesian_velocity *= self.max_cartesian_speed / speed

        return self.solve_cartesian_velocity(
            sensor,
            cartesian_velocity,
            enable_joint_limit_avoidance=enable_joint_limit_avoidance,
        )

    def solve_cartesian_velocity(
            self,
            sensor,
            cartesian_velocity,
            enable_joint_limit_avoidance=False):
        """Solve Cartesian tracking plus an optional null-space limit task.

        Cartesian tracking remains the primary damped-least-squares task.  In
        the terminal arm phase, a secondary normalized joint velocity is
        projected through ``I - J#J`` so joints can move away from a soft
        limit without intentionally changing the requested end-effector
        velocity.  The historical behavior is preserved when the configured
        avoidance gain is zero or the caller does not enable the secondary
        task.
        """
        cartesian_velocity = np.asarray(cartesian_velocity, dtype=np.float64)
        if cartesian_velocity.shape != (3,):
            raise ValueError(
                "cartesian_velocity must have shape (3,), got {}".format(
                    cartesian_velocity.shape
                )
            )
        if not np.all(np.isfinite(cartesian_velocity)):
            raise ValueError("cartesian_velocity contains non-finite values")

        q = np.asarray(
            sensor[self.JOINT_POSITION_SLICE],
            dtype=np.float64,
        )
        jacobian = np.asarray(
            self.jacobian_provider.position_jacobian(q),
            dtype=np.float64,
        )
        if jacobian.shape != (3, 6):
            raise ValueError(
                "position Jacobian must have shape (3, 6), got {}".format(
                    jacobian.shape
                )
            )
        scaled = jacobian * self.arm_joint_max_velocity[np.newaxis, :]
        regularized = (
            np.dot(scaled, scaled.T)
            + (self.dls_damping ** 2) * np.eye(3, dtype=np.float64)
        )
        normalized = np.dot(
            scaled.T,
            np.linalg.solve(regularized, cartesian_velocity),
        )
        unconstrained = np.clip(normalized, -1.0, 1.0)

        # Active-set constrained DLS.  A joint already inside its soft-limit
        # band is removed from the solve whenever the unconstrained solution
        # would drive it farther toward that limit.  The remaining Jacobian
        # columns compensate, so the safety layer is a final backstop rather
        # than the component that defines most teacher labels.
        arm_q = q[4:10]
        avoidance_raw = np.zeros(6, dtype=np.float64)
        if (
                bool(enable_joint_limit_avoidance)
                and self.joint_limit_avoidance_gain > 0.0):
            avoidance_raw = self._joint_limit_avoidance_action(arm_q)
        active = np.ones(6, dtype=np.bool_)
        blocked = np.zeros(6, dtype=np.bool_)
        avoidance_projected = np.zeros(6, dtype=np.float64)
        for _ in range(6):
            primary = self._solve_active_dls(
                scaled,
                cartesian_velocity,
                active,
            )
            avoidance_projected = self._project_secondary_task(
                scaled,
                avoidance_raw,
                active,
            )
            normalized = primary + avoidance_projected
            newly_blocked = self._soft_limit_violations(
                normalized,
                arm_q,
            )
            newly_blocked = np.logical_and(newly_blocked, active)
            if not np.any(newly_blocked):
                break
            active[newly_blocked] = False
            blocked[newly_blocked] = True
        normalized[blocked] = 0.0

        singular_values = np.linalg.svd(scaled, compute_uv=False)
        smallest = float(np.min(singular_values))
        condition = (
            float(np.max(singular_values) / smallest)
            if smallest > 1.0e-9 else float("inf")
        )
        diagnostics = {
            "blocked_joints": [
                self.ARM_JOINT_NAMES[index]
                for index in np.flatnonzero(blocked)
            ],
            "unconstrained_arm_action": unconstrained,
            "joint_limit_avoidance_applied": bool(
                np.max(np.abs(avoidance_projected)) > 1.0e-8
            ),
            "joint_limit_avoidance_raw": avoidance_raw.copy(),
            "joint_limit_avoidance_action": avoidance_projected.copy(),
        }
        return (
            np.clip(normalized, -1.0, 1.0),
            condition,
            cartesian_velocity,
            diagnostics,
        )

    def _solve_active_dls(self, scaled_jacobian, velocity, active):
        result = np.zeros(6, dtype=np.float64)
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            return result
        active_jacobian = scaled_jacobian[:, active_indices]
        regularized = (
            np.dot(active_jacobian, active_jacobian.T)
            + (self.dls_damping ** 2)
            * np.eye(3, dtype=np.float64)
        )
        result[active_indices] = np.dot(
            active_jacobian.T,
            np.linalg.solve(regularized, velocity),
        )
        return result

    def _project_secondary_task(
            self,
            scaled_jacobian,
            secondary_action,
            active):
        """Project one normalized joint task into the active DLS null space."""
        result = np.zeros(6, dtype=np.float64)
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            return result
        secondary_action = np.asarray(
            secondary_action, dtype=np.float64
        )
        if np.max(np.abs(secondary_action[active_indices])) <= 1.0e-12:
            return result
        active_jacobian = scaled_jacobian[:, active_indices]
        regularized = (
            np.dot(active_jacobian, active_jacobian.T)
            + (self.dls_damping ** 2)
            * np.eye(3, dtype=np.float64)
        )
        pseudo_inverse = np.dot(
            active_jacobian.T,
            np.linalg.solve(
                regularized,
                np.eye(3, dtype=np.float64),
            ),
        )
        projector = (
            np.eye(active_indices.size, dtype=np.float64)
            - np.dot(pseudo_inverse, active_jacobian)
        )
        result[active_indices] = np.dot(
            projector,
            secondary_action[active_indices],
        )
        return result

    def _joint_limit_avoidance_action(self, arm_positions):
        """Return a smooth normalized velocity away from nearby limits."""
        arm_positions = np.asarray(arm_positions, dtype=np.float64)
        joint_range = (
            self.arm_joint_limits[:, 1]
            - self.arm_joint_limits[:, 0]
        )
        activation_width = np.minimum(
            self.joint_limit_avoidance_activation * self.arm_soft_margins,
            0.45 * joint_range,
        )
        lower_edge = self.arm_joint_limits[:, 0] + activation_width
        upper_edge = self.arm_joint_limits[:, 1] - activation_width
        lower_ratio = np.clip(
            (lower_edge - arm_positions)
            / np.maximum(activation_width, 1.0e-9),
            0.0,
            1.0,
        )
        upper_ratio = np.clip(
            (arm_positions - upper_edge)
            / np.maximum(activation_width, 1.0e-9),
            0.0,
            1.0,
        )
        return np.clip(
            self.joint_limit_avoidance_gain
            * (lower_ratio ** 2 - upper_ratio ** 2),
            -1.0,
            1.0,
        )

    def _soft_limit_violations(self, normalized_action, arm_positions):
        lower_distance = (
            arm_positions - self.arm_joint_limits[:, 0]
        )
        upper_distance = (
            self.arm_joint_limits[:, 1] - arm_positions
        )
        toward_lower = np.logical_and(
            normalized_action < -1.0e-8,
            lower_distance <= self.arm_soft_margins,
        )
        toward_upper = np.logical_and(
            normalized_action > 1.0e-8,
            upper_distance <= self.arm_soft_margins,
        )
        return np.logical_or(toward_lower, toward_upper)
