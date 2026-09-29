#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python-2 adapter for a Python-3 HRL4IN PPO inference server."""

import json
import socket

import numpy as np

from hrl.high_level_command import TaskMode
from hrl.hrl4in_low_level import HRL4INLowLevelState


class PPOInferenceClient(object):
    """Small newline-delimited JSON client kept compatible with Python 2.7."""

    OBS_DIM = 68
    ACTION_DIM = 10

    def __init__(
            self,
            host="127.0.0.1",
            port=5556,
            timeout=2.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        if self.timeout <= 0.0:
            raise ValueError("timeout must be positive")

        self.socket = socket.create_connection(
            (self.host, self.port),
            timeout=self.timeout,
        )
        self.socket.settimeout(self.timeout)
        self.receive_buffer = b""
        metadata = self._request({"command": "ping"})
        if int(metadata["observation_dim"]) != self.OBS_DIM:
            raise RuntimeError(
                "policy server observation dim is {}, expected {}".format(
                    metadata["observation_dim"],
                    self.OBS_DIM,
                )
            )
        if int(metadata["action_dim"]) != self.ACTION_DIM:
            raise RuntimeError(
                "policy server action dim is {}, expected {}".format(
                    metadata["action_dim"],
                    self.ACTION_DIM,
                )
            )
        self.checkpoint = str(metadata.get("checkpoint", ""))
        self.total_steps = int(metadata.get("total_steps", 0))

    def predict(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.OBS_DIM,):
            raise ValueError(
                "low-level observation must have shape (68,), got {}".format(
                    observation.shape
                )
            )
        response = self._request({
            "command": "predict",
            "observation": observation.tolist(),
        })
        action = np.asarray(response["action"], dtype=np.float32)
        if action.shape != (self.ACTION_DIM,):
            raise RuntimeError(
                "policy server action must have shape (10,), got {}".format(
                    action.shape
                )
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError("policy server returned a non-finite action")
        return action, float(response.get("inference_ms", 0.0))

    def close(self):
        if self.socket is None:
            return
        try:
            self._request({"command": "close"})
        except (socket.error, RuntimeError, ValueError):
            pass
        finally:
            try:
                self.socket.close()
            finally:
                self.socket = None

    def _request(self, request):
        if self.socket is None:
            raise RuntimeError("policy server connection is closed")
        payload = (
            json.dumps(request, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self.socket.sendall(payload)
        response = self._receive_response()
        if not bool(response.get("ok", False)):
            raise RuntimeError(
                "policy server error: {}".format(
                    response.get("error", "unknown error")
                )
            )
        return response["result"]

    def _receive_response(self):
        while b"\n" not in self.receive_buffer:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise RuntimeError("policy server closed the connection")
            self.receive_buffer += chunk
        raw_response, self.receive_buffer = self.receive_buffer.split(
            b"\n",
            1,
        )
        return json.loads(raw_response.decode("utf-8"))


class HRL4INPPOLowPolicy(object):
    """Use the trained PPO actor for ARM_REACH inside the ROS hierarchy."""

    OBS_DIM = 46
    ACTION_DIM = 10
    JOINT_POSITION_SLICE = slice(11, 21)
    ARM_ACTION_SLICE = slice(4, 10)
    ARM_SUBGOAL_SLICE = slice(3, 6)

    def __init__(
            self,
            host="127.0.0.1",
            port=5556,
            timeout=2.0,
            base_gain=5.0,
            enable_base_motion=False,
            recovery_gain=0.4,
            subgoal_tolerance=None,
            intrinsic_reward_scale=30.0,
            subgoal_achieved_reward=1.0,
            collision_reward_weight=0.0,
            extrinsic_reward_weight=0.0,
            inference_client=None):
        self.base_gain = float(base_gain)
        self.enable_base_motion = bool(enable_base_motion)
        self.recovery_gain = float(recovery_gain)
        self.inference_client = (
            inference_client
            if inference_client is not None
            else PPOInferenceClient(
                host=host,
                port=port,
                timeout=timeout,
            )
        )
        self.subgoal_state = HRL4INLowLevelState(
            subgoal_tolerance=subgoal_tolerance,
            intrinsic_reward_scale=intrinsic_reward_scale,
            subgoal_achieved_reward=subgoal_achieved_reward,
            collision_reward_weight=collision_reward_weight,
            extrinsic_reward_weight=extrinsic_reward_weight,
        )
        self.reset()

    @property
    def checkpoint(self):
        return getattr(self.inference_client, "checkpoint", "")

    @property
    def model_total_steps(self):
        return int(getattr(self.inference_client, "total_steps", 0))

    def reset(self):
        self.current_command = None
        self.subgoal_state.reset()
        self.last_diagnostics = {}

    def begin_subgoal(self, observation, command):
        self.current_command = command
        low_observation = self.subgoal_state.begin_subgoal(
            observation,
            command,
        )
        self.last_diagnostics = self.subgoal_state.diagnostics()
        return low_observation

    def low_level_observation(self, observation):
        return self.subgoal_state.low_level_observation(observation)

    def predict(self, observation, command):
        sensor = HRL4INLowLevelState.as_sensor(observation)
        if command is not self.current_command:
            self.begin_subgoal(sensor, command)

        low_observation = self.low_level_observation(sensor)
        action_mask = np.asarray(
            low_observation["action_mask"],
            dtype=np.float32,
        )
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        inference_ms = 0.0

        if command.mode == TaskMode.BASE_APPROACH:
            self._apply_base_approach(
                action,
                np.asarray(
                    low_observation["subgoal"][0:3],
                    dtype=np.float64,
                ),
            )
        elif command.mode == TaskMode.ARM_REACH:
            action, inference_ms = self.inference_client.predict(
                low_observation["vector"]
            )
        elif command.mode == TaskMode.RECOVERY:
            action[self.ARM_ACTION_SLICE] = self._recovery_action(sensor)

        unmasked_action = np.asarray(action, dtype=np.float32).copy()
        action = np.clip(action, -1.0, 1.0) * action_mask
        self._update_action_diagnostics(
            sensor,
            low_observation,
            unmasked_action,
            action,
            inference_ms,
            command,
        )
        return action.astype(np.float32)

    def _update_action_diagnostics(
            self,
            observation,
            low_observation,
            unmasked_action,
            masked_action,
            inference_ms,
            command):
        tracker = self.subgoal_state.diagnostics()
        remaining_subgoal = np.asarray(
            low_observation["subgoal"],
            dtype=np.float64,
        )
        error_base = remaining_subgoal[self.ARM_SUBGOAL_SLICE]
        task_state = HRL4INLowLevelState.extract_task_state(observation)
        target_position = np.asarray(
            observation[0:3],
            dtype=np.float64,
        )
        target_error_base = target_position - task_state[3:6]
        self.last_diagnostics = tracker
        self.last_diagnostics.update({
            "policy_type": "hrl4in_ppo",
            "task_mode": command.mode_name,
            "base_subgoal_error": remaining_subgoal[0:3].copy(),
            "error_base": error_base.copy(),
            "error_norm": float(np.linalg.norm(error_base)),
            "subgoal_distance": float(np.linalg.norm(error_base)),
            "target_error_base": target_error_base.copy(),
            "target_error_norm": float(np.linalg.norm(target_error_base)),
            "unmasked_action": unmasked_action.copy(),
            "masked_action": masked_action.copy(),
            "normalized_arm_action": masked_action[
                self.ARM_ACTION_SLICE
            ].copy(),
            "normalized_base_action": masked_action[0:4].copy(),
            "inference_ms": float(inference_ms),
            "cartesian_velocity": np.full(3, np.nan),
            "jacobian_condition": float("nan"),
        })

    def _apply_base_approach(self, action, remaining_subgoal):
        if not self.enable_base_motion:
            return
        action[0] = self.base_gain * remaining_subgoal[0]
        action[1] = self.base_gain * remaining_subgoal[1]
        # z and sway stay fixed in the verified first-stage planar mode.
        action[2] = 0.0
        action[3] = 0.0

    def observe_transition(
            self,
            next_observation,
            extrinsic_reward=0.0,
            collision_reward=0.0,
            episode_done=False,
            subgoal_timed_out=False):
        tracker = self.subgoal_state.observe_transition(
            next_observation,
            extrinsic_reward=extrinsic_reward,
            collision_reward=collision_reward,
            episode_done=episode_done,
            subgoal_timed_out=subgoal_timed_out,
        )
        self.last_diagnostics.update(tracker)
        # HRL4INLowLevelState reports its bookkeeping implementation as the
        # policy type. Preserve the actual executor type for runtime logging.
        self.last_diagnostics["policy_type"] = "hrl4in_ppo"
        remaining_arm_error = np.asarray(
            tracker["remaining_subgoal"][self.ARM_SUBGOAL_SLICE],
            dtype=np.float64,
        )
        self.last_diagnostics.update({
            "post_error_base": remaining_arm_error.copy(),
            "post_error_norm": float(tracker["post_potential"]),
            "subgoal_distance": float(tracker["post_potential"]),
        })
        return self.diagnostics()

    def _recovery_action(self, observation):
        joint_positions = observation[self.JOINT_POSITION_SLICE]
        arm_positions = joint_positions[self.ARM_ACTION_SLICE]
        return -self.recovery_gain * arm_positions

    def diagnostics(self):
        result = {}
        for key, value in self.last_diagnostics.items():
            result[key] = value.copy() if hasattr(value, "copy") else value
        return result

    def close(self):
        if self.inference_client is not None:
            self.inference_client.close()
