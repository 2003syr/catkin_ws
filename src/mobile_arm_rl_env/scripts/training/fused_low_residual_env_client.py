#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Python-3 client for the fused low-level residual environment."""

import json
import socket

import numpy as np


class FusedLowResidualEnvironmentClient(object):
    OBS_DIM = 66
    ACTION_DIM = 5

    def __init__(self, host="127.0.0.1", port=5563, timeout=120.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.socket = socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        )
        self.socket.settimeout(self.timeout)
        self.receive_buffer = b""
        self.last_reset_info = {}
        dimensions = self._request({"command": "ping"})
        self.metadata = dict(dimensions)
        if int(dimensions["observation_dim"]) != self.OBS_DIM:
            raise RuntimeError("environment observation dimension mismatch")
        if int(dimensions["action_dim"]) != self.ACTION_DIM:
            raise RuntimeError("environment residual action dimension mismatch")

    def reset(self, scenario=None, max_steps=None):
        observation, _ = self.reset_with_info(
            scenario=scenario,
            max_steps=max_steps,
        )
        return observation

    def reset_with_info(self, scenario=None, max_steps=None):
        """Reset and return the sampled task metadata with the observation."""
        request = {"command": "reset"}
        if scenario is not None:
            request["scenario"] = dict(scenario)
        if max_steps is not None:
            request["max_steps"] = int(max_steps)
        response = self._request(request)
        self.last_reset_info = dict(response.get("reset_info", {}))
        return (
            self._observation(response["observation"]),
            dict(self.last_reset_info),
        )

    def step(self, residual):
        residual = np.asarray(residual, dtype=np.float32)
        if residual.shape != (self.ACTION_DIM,):
            raise ValueError(
                "residual must have shape (5,), got {}".format(
                    residual.shape
                )
            )
        response = self._request({
            "command": "step",
            "action": np.clip(residual, -1.0, 1.0).tolist(),
        })
        info = dict(response["info"])
        info["nominal_action"] = np.asarray(
            response.get("nominal_action", info.get("nominal_action", [])),
            dtype=np.float32,
        )
        info["teacher_diagnostics"] = dict(
            response.get(
                "teacher_diagnostics",
                info.get("teacher_diagnostics", {}),
            )
        )
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
    def _observation(cls, value):
        observation = np.asarray(value, dtype=np.float32)
        if observation.shape != (cls.OBS_DIM,):
            raise RuntimeError("environment returned invalid observation shape")
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("environment returned non-finite observation")
        return observation

    def __enter__(self):
        return self

    def __exit__(self, _error_type, _error, _traceback):
        self.close()
