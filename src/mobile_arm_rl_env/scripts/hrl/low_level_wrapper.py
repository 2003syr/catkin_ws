#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Frozen fused-low policy and joint-subgoal observation wrapper."""

from __future__ import division

import json
import socket

import numpy as np

from hrl.fused_low_level import FusedLowLevelState
from hrl.high_level_command import SubgoalType
from hrl.subgoal_converter import SubgoalConverter


class ZeroFusedResidualPolicy(object):
    """In-process zero residual used by the frozen rule low level.

    High-level training must not depend on a second policy socket or on a
    learned low-level checkpoint.  This tiny policy deliberately returns no
    correction, so :class:`JointSubgoalResidualLowLevelWrapper` executes the
    coordinated rule teacher exactly while the upper policy is being
    optimized.
    """

    OBS_DIM = FusedLowLevelState.INPUT_DIM
    ACTION_DIM = 5

    def predict(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.OBS_DIM,):
            raise ValueError(
                "zero residual observation must have shape ({},), got {}".format(
                    self.OBS_DIM,
                    observation.shape,
                )
            )
        if not np.all(np.isfinite(observation)):
            raise ValueError("zero residual observation contains non-finite values")
        return np.zeros(self.ACTION_DIM, dtype=np.float32)

    def close(self):
        pass


class FrozenFusedLowPolicy(object):
    """Inference-only client for the existing 66-D/8-D policy server."""

    OBS_DIM = 66
    ACTION_DIM = 8

    def __init__(self, host="127.0.0.1", port=5563, timeout=5.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.socket = socket.create_connection(
            (self.host, self.port),
            self.timeout,
        )
        self.socket.settimeout(self.timeout)
        self.reader = self.socket.makefile("rb")
        self.last_inference_ms = 0.0
        metadata = self._request({"command": "ping"})
        if int(metadata.get("observation_dim", -1)) != self.OBS_DIM:
            raise RuntimeError("frozen low policy observation_dim is not 66")
        if int(metadata.get("action_dim", -1)) != self.ACTION_DIM:
            raise RuntimeError("frozen low policy action_dim is not 8")
        self.metadata = metadata

    def predict(self, observation):
        observation = self._vector(
            observation,
            self.OBS_DIM,
            "low observation",
        )
        response = self._request({
            "command": "predict",
            "observation": observation.astype(float).tolist(),
        })
        action = self._vector(
            response.get("action", []),
            self.ACTION_DIM,
            "low action",
        )
        self.last_inference_ms = float(
            response.get("inference_ms", 0.0)
        )
        return np.clip(action, -1.0, 1.0)

    def close(self):
        if self.socket is None:
            return
        try:
            self._request({"command": "close"})
        except Exception:
            pass
        try:
            self.reader.close()
        finally:
            self.socket.close()
            self.socket = None

    def _request(self, request):
        if self.socket is None:
            raise RuntimeError("frozen low policy is closed")
        payload = json.dumps(request, separators=(",", ":")) + "\n"
        self.socket.sendall(payload.encode("utf-8"))
        line = self.reader.readline()
        if not line:
            raise RuntimeError("frozen low policy server closed connection")
        response = json.loads(line.decode("utf-8"))
        if not bool(response.get("ok", False)):
            raise RuntimeError(
                "frozen low policy error: {}".format(
                    response.get("error", "unknown error")
                )
            )
        return response

    @staticmethod
    def _vector(value, size, name):
        vector = np.asarray(value, dtype=np.float32)
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


class FrozenFusedResidualPolicy(object):
    """Inference-only client for a 66-D/5-D residual PPO server.

    The residual policy does not know how to execute a command by itself.  A
    :class:`JointSubgoalResidualLowLevelWrapper` combines its bounded residual
    with the coordinated rule teacher in the ROS process, where the Jacobian
    provider and the active safety contract are already available.
    """

    OBS_DIM = 66
    ACTION_DIM = 5

    def __init__(self, host="127.0.0.1", port=5563, timeout=5.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.socket = socket.create_connection(
            (self.host, self.port),
            self.timeout,
        )
        self.socket.settimeout(self.timeout)
        self.reader = self.socket.makefile("rb")
        self.last_inference_ms = 0.0
        metadata = self._request({"command": "ping"})
        if int(metadata.get("observation_dim", -1)) != self.OBS_DIM:
            raise RuntimeError(
                "residual policy observation_dim is not 66"
            )
        if int(metadata.get("action_dim", -1)) != self.ACTION_DIM:
            raise RuntimeError(
                "residual policy action_dim is not 5"
            )
        self.metadata = metadata

    def predict(self, observation):
        observation = self._vector(
            observation,
            self.OBS_DIM,
            "residual observation",
        )
        response = self._request({
            "command": "predict",
            "observation": observation.astype(float).tolist(),
        })
        residual = self._vector(
            response.get("action", []),
            self.ACTION_DIM,
            "residual action",
        )
        self.last_inference_ms = float(
            response.get("inference_ms", 0.0)
        )
        return np.clip(residual, -1.0, 1.0)

    def close(self):
        if self.socket is None:
            return
        try:
            self._request({"command": "close"})
        except Exception:
            pass
        try:
            self.reader.close()
        finally:
            self.socket.close()
            self.socket = None

    def _request(self, request):
        if self.socket is None:
            raise RuntimeError("residual policy is closed")
        payload = json.dumps(request, separators=(",", ":")) + "\n"
        self.socket.sendall(payload.encode("utf-8"))
        line = self.reader.readline()
        if not line:
            raise RuntimeError("residual policy server closed connection")
        response = json.loads(line.decode("utf-8"))
        if not bool(response.get("ok", False)):
            raise RuntimeError(
                "residual policy error: {}".format(
                    response.get("error", "unknown error")
                )
            )
        return response

    @staticmethod
    def _vector(value, size, name):
        vector = np.asarray(value, dtype=np.float32)
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


class JointSubgoalLowLevelWrapper(object):
    """Condition the frozen low level on a fixed joint world subgoal."""

    OBS_DIM = FusedLowLevelState.INPUT_DIM
    ACTION_DIM = FusedLowLevelState.ACTION_DIM
    SUBGOAL_CONTEXT_CONTRACT = "three_class_in_subgoal_mask_v1"

    def __init__(
            self,
            policy,
            converter=None,
            encode_subgoal_type=False):
        self.policy = policy
        self.converter = converter or SubgoalConverter()
        self.encode_subgoal_type = bool(encode_subgoal_type)
        self.fixed_subgoal = None
        self.last_observation = None
        self.last_action = None
        self.last_components = {}

    def begin_subgoal(self, command, sensor_observation):
        sensor = FusedLowLevelState.as_sensor(sensor_observation)
        self.fixed_subgoal = self.converter.convert(
            command,
            self.base_pose(sensor),
            self.ee_position(sensor),
        )
        self.last_components = {}
        return self.fixed_subgoal

    def build_observation(
            self,
            sensor_observation,
            ee_rotation_world=None):
        if self.fixed_subgoal is None:
            raise RuntimeError("begin_subgoal must be called first")
        sensor = FusedLowLevelState.as_sensor(sensor_observation).copy()
        base_pose = self.base_pose(sensor)
        ee_position = self.ee_position(sensor)
        remaining = self.fixed_subgoal.remaining(
            base_pose,
            ee_position,
        )

        # Replace every target-derived feature used by the original fused
        # policy.  The 46-D proprioceptive/scan layout stays unchanged.
        ee_error_world = (
            self.fixed_subgoal.ee_goal_world - ee_position
        )
        if ee_rotation_world is None:
            ee_error_local = ee_error_world
        else:
            rotation = np.asarray(ee_rotation_world, dtype=np.float64)
            if rotation.shape != (3, 3):
                raise ValueError(
                    "ee_rotation_world must have shape (3, 3)"
                )
            ee_error_local = np.dot(rotation.T, ee_error_world)
        sensor[0:3] = self.fixed_subgoal.ee_goal_world
        sensor[3:6] = ee_error_local
        planar_target_delta = (
            self.fixed_subgoal.ee_goal_world[0:2] - base_pose[0:2]
        )
        sensor[9] = float(np.linalg.norm(np.concatenate((
            planar_target_delta,
            self.fixed_subgoal.ee_goal_world[2:3],
        ))))
        sensor[10] = float(np.linalg.norm(ee_error_world))

        vector = np.concatenate((
            sensor,
            remaining,
            self._subgoal_context(),
            np.ones(8, dtype=np.float32),
        )).astype(np.float32)
        if vector.shape != (self.OBS_DIM,):
            raise RuntimeError(
                "joint-subgoal low observation has shape {}".format(
                    vector.shape
                )
            )
        self.last_observation = vector.copy()
        return vector

    def _subgoal_context(self):
        """Return the compatible six-slot typed-subgoal context.

        Untyped policies retain the historical all-ones subgoal mask.  A
        typed residual policy receives ``[DIRECT, DETOUR, TERMINAL,
        base_xy_active, base_yaw_active, ee_active]`` in the same six slots,
        preserving the established 66-D observation dimension.
        """
        if not self.encode_subgoal_type:
            return np.ones(6, dtype=np.float32)
        command = getattr(self.fixed_subgoal, "original_command", None)
        if command is None or not hasattr(command, "subgoal_type_one_hot"):
            raise RuntimeError("typed subgoal context requires a typed command")
        ee_active = float(
            np.linalg.norm(
                np.asarray(command.ee_goal, dtype=np.float64)
            ) > 1.0e-6
        )
        return np.concatenate((
            command.subgoal_type_one_hot(),
            np.asarray([1.0, 1.0, ee_active], dtype=np.float32),
        )).astype(np.float32)

    def predict(self, sensor_observation, ee_rotation_world=None):
        observation = self.build_observation(
            sensor_observation,
            ee_rotation_world=ee_rotation_world,
        )
        action = np.asarray(
            self.policy.predict(observation),
            dtype=np.float32,
        )
        if action.shape != (self.ACTION_DIM,):
            raise RuntimeError(
                "frozen low policy returned shape {}".format(action.shape)
            )
        self.last_action = np.clip(action, -1.0, 1.0)
        return self.last_action.copy()

    def remaining(self, sensor_observation):
        if self.fixed_subgoal is None:
            raise RuntimeError("begin_subgoal must be called first")
        sensor = FusedLowLevelState.as_sensor(sensor_observation)
        return self.fixed_subgoal.remaining(
            self.base_pose(sensor),
            self.ee_position(sensor),
        )

    def close(self):
        if hasattr(self.policy, "close"):
            self.policy.close()

    @staticmethod
    def base_pose(sensor):
        sensor = FusedLowLevelState.as_sensor(sensor)
        q = sensor[FusedLowLevelState.JOINT_POSITION_SLICE]
        tracked_heading = (
            float(q[2]) + FusedLowLevelState().forward_yaw_offset
        )
        return np.asarray([
            q[0],
            q[1],
            tracked_heading,
        ], dtype=np.float64)

    @staticmethod
    def ee_position(sensor):
        sensor = FusedLowLevelState.as_sensor(sensor)
        return np.asarray(
            sensor[FusedLowLevelState.END_EFFECTOR_POSITION_SLICE],
            dtype=np.float64,
        )


class JointSubgoalResidualLowLevelWrapper(JointSubgoalLowLevelWrapper):
    """Apply a residual PPO correction around the rule teacher.

    The observation construction is identical to the frozen 8-D policy path.
    Only the final action composition differs: the residual actor produces
    ``[dv, domega, dvx, dvy, dvz]`` and the existing teacher converts that
    correction into a bounded fused ``[v, omega, arm6]`` command.
    """

    RESIDUAL_DIM = 5

    def __init__(
            self,
            policy,
            teacher,
            residual_base_scale=0.10,
            residual_cartesian_scale=0.01,
            converter=None,
            encode_subgoal_type=False):
        super(JointSubgoalResidualLowLevelWrapper, self).__init__(
            policy,
            converter=converter,
            encode_subgoal_type=encode_subgoal_type,
        )
        if teacher is None or not hasattr(teacher, "residual_components"):
            raise ValueError(
                "a CoordinatedRuleTeacher is required for residual HRL"
            )
        self.teacher = teacher
        self.residual_base_scale = float(residual_base_scale)
        self.residual_cartesian_scale = float(residual_cartesian_scale)
        if self.residual_base_scale < 0.0:
            raise ValueError("residual_base_scale must be non-negative")
        if self.residual_cartesian_scale < 0.0:
            raise ValueError(
                "residual_cartesian_scale must be non-negative"
            )
        self.last_residual = np.zeros(
            self.RESIDUAL_DIM,
            dtype=np.float32,
        )

    def predict(self, sensor_observation, ee_rotation_world=None):
        observation = self.build_observation(
            sensor_observation,
            ee_rotation_world=ee_rotation_world,
        )
        residual = np.asarray(
            self.policy.predict(observation),
            dtype=np.float32,
        )
        if residual.shape != (self.RESIDUAL_DIM,):
            raise RuntimeError(
                "residual policy returned shape {}".format(
                    residual.shape
                )
            )
        residual = np.clip(residual, -1.0, 1.0)
        arm_enabled = self._arm_subgoal_enabled()
        if hasattr(self.teacher, "subgoal_residual_components"):
            components = self.teacher.subgoal_residual_components(
                observation,
                residual,
                base_scale=self.residual_base_scale,
                cartesian_scale=self.residual_cartesian_scale,
                arm_enabled=arm_enabled,
            )
        else:
            components = self.teacher.residual_components(
                observation,
                residual,
                base_scale=self.residual_base_scale,
                cartesian_scale=self.residual_cartesian_scale,
            )
        action = components["action"]
        self.last_residual = residual.copy()
        self.last_components = dict(components)
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        return self.last_action.copy()

    def _arm_subgoal_enabled(self):
        """Return whether the active high-level command releases the arm."""
        if self.fixed_subgoal is None:
            return False
        command = getattr(self.fixed_subgoal, "original_command", None)
        if command is None:
            return True
        # The option type is selected by the upper policy.  DIRECT and
        # DETOUR are chassis options and structurally hold the arm; only a
        # TERMINAL option can release the arm toward its EE subgoal.  This
        # makes the upper discrete decision causal instead of diagnostic.
        if int(getattr(
                command,
                "subgoal_type",
                SubgoalType.TERMINAL,
        )) != SubgoalType.TERMINAL:
            return False
        return bool(
            np.linalg.norm(
                np.asarray(command.ee_goal, dtype=np.float64)
            ) > 1.0e-6
        )
