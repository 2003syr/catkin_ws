#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Python-3 client for the full-action fused low-level environment.

Unlike ``fused_low_residual_env_client``, this client exposes the original
8-D policy action directly.  The environment server also returns the rule
teacher action so PPO can use it as a supervised guidance term.
"""

import json
import socket

import numpy as np


class FusedLowEnvironmentClient(object):
    OBS_DIM = 66
    ACTION_DIM = 8

    def __init__(self, host="127.0.0.1", port=5562, timeout=120.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.socket = socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        )
        self.socket.settimeout(self.timeout)
        self.receive_buffer = b""
        self.last_reset_info = {}
        self.last_teacher_action = np.zeros(
            self.ACTION_DIM,
            dtype=np.float32,
        )
        self.last_teacher_available = False
        dimensions = self._request({"command": "ping"})
        if int(dimensions.get("observation_dim", -1)) != self.OBS_DIM:
            raise RuntimeError("fused environment observation dimension mismatch")
        if int(dimensions.get("action_dim", -1)) != self.ACTION_DIM:
            raise RuntimeError("fused environment action dimension mismatch")

    def reset(self, scenario=None, max_steps=None):
        request = {"command": "reset"}
        if scenario is not None:
            request["scenario"] = dict(scenario)
        if max_steps is not None:
            request["max_steps"] = int(max_steps)
        response = self._request(request)
        self.last_reset_info = dict(response.get("reset_info", {}))
        self.last_teacher_available = (
            response.get("teacher_action") is not None
        )
        self.last_teacher_action = self._teacher_action(
            response.get("teacher_action")
        )
        return self._observation(response["observation"])

    def reset_with_info(self, scenario=None, max_steps=None):
        self.reset(scenario=scenario, max_steps=max_steps)
        return self.last_reset_info.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.ACTION_DIM,):
            raise ValueError(
                "action must have shape (8,), got {}".format(action.shape)
            )
        response = self._request({
            "command": "step",
            "action": np.clip(action, -1.0, 1.0).tolist(),
        })
        info = dict(response.get("info", {}))
        # The server computes the teacher after the state transition.  Keep
        # its diagnostics next to the returned transition so evaluators can
        # distinguish an intentional ARM_FINISH command from a zero action
        # caused by a blocked or stale arm solve.
        info["teacher_diagnostics"] = dict(
            response.get("teacher_diagnostics", {})
        )
        teacher_value = response.get("teacher_action")
        if teacher_value is None:
            teacher_value = info.get("teacher_action")
        teacher_action = self._teacher_action(teacher_value)
        self.last_teacher_action = teacher_action.copy()
        self.last_teacher_available = teacher_value is not None
        info["teacher_action"] = teacher_action
        info["teacher_available"] = self.last_teacher_available
        return (
            self._observation(response["observation"]),
            float(response["reward"]),
            bool(response["done"]),
            info,
        )

    def close(self):
        if self.socket is None:
            return
        try:
            self._request({"command": "close"})
        except (OSError, RuntimeError):
            pass
        try:
            self.socket.close()
        finally:
            self.socket = None

    def _request(self, request):
        if self.socket is None:
            raise RuntimeError("environment client is closed")
        self.socket.sendall(
            (json.dumps(request, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
        while b"\n" not in self.receive_buffer:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise RuntimeError("environment server closed connection")
            self.receive_buffer += chunk
        raw, self.receive_buffer = self.receive_buffer.split(b"\n", 1)
        response = json.loads(raw.decode("utf-8"))
        if not bool(response.get("ok", False)):
            raise RuntimeError(
                "environment server error: {}".format(
                    response.get("error", "unknown error")
                )
            )
        return response["result"]

    @classmethod
    def _teacher_action(cls, value):
        if value is None:
            return np.zeros(cls.ACTION_DIM, dtype=np.float32)
        teacher_action = np.asarray(value, dtype=np.float32)
        if teacher_action.shape != (cls.ACTION_DIM,):
            raise RuntimeError(
                "environment returned invalid teacher action shape {}".format(
                    teacher_action.shape
                )
            )
        if not np.all(np.isfinite(teacher_action)):
            raise RuntimeError(
                "environment returned non-finite teacher action"
            )
        return np.clip(teacher_action, -1.0, 1.0)

    @classmethod
    def _observation(cls, value):
        observation = np.asarray(value, dtype=np.float32)
        if observation.shape != (cls.OBS_DIM,):
            raise RuntimeError(
                "environment returned invalid observation shape {}".format(
                    observation.shape
                )
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("environment returned non-finite observation")
        return observation

    def __enter__(self):
        return self

    def __exit__(self, _error_type, _error, _traceback):
        self.close()
