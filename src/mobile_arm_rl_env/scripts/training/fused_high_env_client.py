#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Python-3 client for the joint-subgoal high-level environment."""

import json
import socket

import numpy as np


class FusedHighEnvironmentClient(object):
    OBS_DIM = 82
    ACTION_DIM = 6

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
        self.teacher_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        self.teacher_info = {}
        self.metadata = dict(self._request({"command": "ping"}))
        if int(self.metadata.get("observation_dim", -1)) != self.OBS_DIM:
            raise RuntimeError("high environment observation dimension mismatch")
        if int(self.metadata.get("action_dim", -1)) != self.ACTION_DIM:
            raise RuntimeError("high environment action dimension mismatch")
        if int(self.metadata.get("route_side_count", -1)) != 3:
            raise RuntimeError("high environment route-side contract mismatch")
        if int(self.metadata.get("option_count", -1)) != 4:
            raise RuntimeError("high environment unified-option mismatch")
        if str(self.metadata.get("policy_type", "")) != (
                "fused_joint_subgoal_unified_option_ppo_v2"):
            raise RuntimeError("high environment policy contract mismatch")

    def reset(self, scenario=None, max_steps=None):
        observation, _ = self.reset_with_info(
            scenario=scenario,
            max_steps=max_steps,
        )
        return observation

    def reset_with_info(self, scenario=None, max_steps=None):
        request = {"command": "reset"}
        if scenario is not None:
            request["scenario"] = dict(scenario)
        if max_steps is not None:
            request["max_steps"] = int(max_steps)
        response = self._request(request)
        self.last_reset_info = dict(response.get("reset_info", {}))
        self._set_teacher(response)
        return self._observation(response["observation"]), dict(
            self.last_reset_info
        )

    def step(self, high_action, subgoal_type=0, route_side=0):
        high_action = np.asarray(high_action, dtype=np.float32)
        if high_action.shape != (self.ACTION_DIM,):
            raise ValueError(
                "high action must have shape (6,), got {}".format(
                    high_action.shape
                )
            )
        if not np.all(np.isfinite(high_action)):
            raise ValueError("high action contains non-finite values")
        subgoal_type = int(subgoal_type)
        if subgoal_type not in (0, 1, 2):
            raise ValueError("subgoal_type must be DIRECT, DETOUR, or TERMINAL")
        route_side = int(route_side)
        if route_side not in (0, 1, 2):
            raise ValueError("route_side must be NONE, UPPER, or LOWER")
        if subgoal_type != 1 and route_side != 0:
            raise ValueError("route_side must be NONE outside DETOUR")
        response = self._request({
            "command": "step",
            "action": np.clip(high_action, -1.0, 1.0).tolist(),
            "subgoal_type": subgoal_type,
            "route_side": route_side,
        })
        self._set_teacher(response)
        return (
            self._observation(response["observation"]),
            float(response["reward"]),
            bool(response["done"]),
            dict(response["info"]),
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

    def _set_teacher(self, response):
        teacher_action = np.asarray(
            response.get("teacher_action", np.zeros(self.ACTION_DIM)),
            dtype=np.float32,
        )
        if teacher_action.shape != (self.ACTION_DIM,):
            raise RuntimeError("environment returned invalid teacher action")
        self.teacher_action = np.clip(teacher_action, -1.0, 1.0)
        self.teacher_info = dict(response.get("teacher_info", {}))

    def _request(self, request):
        if self.socket is None:
            raise RuntimeError("high environment client is closed")
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

    @classmethod
    def _observation(cls, value):
        observation = np.asarray(value, dtype=np.float32)
        if observation.shape != (cls.OBS_DIM,):
            raise RuntimeError("environment returned invalid high observation")
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("environment returned non-finite observation")
        return observation

    def __enter__(self):
        return self

    def __exit__(self, _error_type, _error, _traceback):
        self.close()
