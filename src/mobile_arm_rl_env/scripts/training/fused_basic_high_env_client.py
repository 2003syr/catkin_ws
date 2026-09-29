#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Python-3 client for the minimal two-layer high-level environment."""

import json
import socket

import numpy as np


class FusedBasicHighEnvironmentClient(object):
    OBS_DIM = 86
    ACTION_DIM = 6
    STAGE_COUNT = 3
    POLICY_TYPE = "fused_basic_two_layer_ppo_v2"

    def __init__(self, host="127.0.0.1", port=5565, timeout=180.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.socket = socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        )
        self.socket.settimeout(self.timeout)
        self.receive_buffer = b""
        self.last_reset_info = {}
        self.last_step_info = {}
        self.terminal_student_blend = None
        self.teacher_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        self.teacher_info = {}
        self.metadata = dict(self._request({"command": "ping"}))
        expected = {
            "observation_dim": self.OBS_DIM,
            "action_dim": self.ACTION_DIM,
            "subgoal_type_count": self.STAGE_COUNT,
            "policy_type": self.POLICY_TYPE,
        }
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise RuntimeError(
                    "basic high environment {} mismatch: {} != {}".format(
                        key, self.metadata.get(key), value
                    )
                )

    def set_terminal_student_blend(self, value):
        if value is None:
            self.terminal_student_blend = None
            return
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "terminal_student_blend must be finite and in [0, 1]"
            )
        self.terminal_student_blend = value

    def reset(
            self,
            scenario=None,
            max_steps=None,
            terminal_student_blend=None):
        observation, unused_info = self.reset_with_info(
            scenario=scenario,
            max_steps=max_steps,
            terminal_student_blend=terminal_student_blend,
        )
        return observation

    def reset_with_info(
            self,
            scenario=None,
            max_steps=None,
            terminal_student_blend=None):
        request = {"command": "reset"}
        if scenario is not None:
            request["scenario"] = dict(scenario)
        if max_steps is not None:
            request["max_steps"] = int(max_steps)
        blend = (
            self.terminal_student_blend
            if terminal_student_blend is None
            else terminal_student_blend
        )
        if blend is not None:
            blend = float(blend)
            if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
                raise ValueError(
                    "terminal_student_blend must be finite and in [0, 1]"
                )
            request["terminal_student_blend"] = blend
        response = self._request(request)
        self.last_reset_info = dict(response.get("reset_info", {}))
        self.last_step_info = dict(self.last_reset_info)
        self._set_teacher(response)
        return self._observation(response["observation"]), dict(
            self.last_reset_info
        )

    def step(self, high_action, subgoal_type):
        high_action = np.asarray(high_action, dtype=np.float32)
        if high_action.shape != (self.ACTION_DIM,):
            raise ValueError("high action must have shape (6,)")
        if not np.all(np.isfinite(high_action)):
            raise ValueError("high action contains non-finite values")
        subgoal_type = int(subgoal_type)
        if subgoal_type not in range(self.STAGE_COUNT):
            raise ValueError("subgoal type must be DIRECT, DETOUR or TERMINAL")
        response = self._request({
            "command": "step",
            "action": np.clip(high_action, -1.0, 1.0).tolist(),
            "subgoal_type": subgoal_type,
        })
        self._set_teacher(response)
        self.last_step_info = dict(response["info"])
        return (
            self._observation(response["observation"]),
            float(response["reward"]),
            bool(response["done"]),
            dict(self.last_step_info),
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
            raise RuntimeError("basic high environment client is closed")
        self.socket.sendall((
            json.dumps(request, separators=(",", ":")) + "\n"
        ).encode("utf-8"))
        while b"\n" not in self.receive_buffer:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise RuntimeError("high environment server closed connection")
            self.receive_buffer += chunk
        raw, self.receive_buffer = self.receive_buffer.split(b"\n", 1)
        response = json.loads(raw.decode("utf-8"))
        if not bool(response.get("ok", False)):
            raise RuntimeError(
                "high environment server error: {}".format(
                    response.get("error", "unknown error")
                )
            )
        return response["result"]

    def _set_teacher(self, response):
        action = np.asarray(
            response.get("teacher_action", np.zeros(self.ACTION_DIM)),
            dtype=np.float32,
        )
        if action.shape != (self.ACTION_DIM,):
            raise RuntimeError("environment returned invalid teacher action")
        self.teacher_action = np.clip(action, -1.0, 1.0)
        self.teacher_info = dict(response.get("teacher_info", {}))

    @classmethod
    def _observation(cls, value):
        observation = np.asarray(value, dtype=np.float32)
        if observation.shape != (cls.OBS_DIM,):
            raise RuntimeError("environment returned invalid observation")
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("environment returned non-finite observation")
        return observation

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_error, unused_traceback):
        self.close()
