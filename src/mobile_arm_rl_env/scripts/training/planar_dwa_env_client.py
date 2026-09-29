#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Python-3 client for the DWA-assisted planar PPO environment."""

import json
import socket

import numpy as np


class PlanarDWAEnvironmentClient(object):
    OBS_DIM = 11
    ACTION_DIM = 2

    def __init__(
            self,
            host="127.0.0.1",
            port=5557,
            timeout=120.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.socket = socket.create_connection(
            (self.host, self.port),
            timeout=self.timeout,
        )
        self.socket.settimeout(self.timeout)
        self.receive_buffer = b""
        dimensions = self._request({"command": "ping"})
        if int(dimensions["observation_dim"]) != self.OBS_DIM:
            raise RuntimeError("planar observation dimension mismatch")
        if int(dimensions["action_dim"]) != self.ACTION_DIM:
            raise RuntimeError("planar action dimension mismatch")

    def reset(self):
        response = self._request({"command": "reset"})
        return self._observation(response["observation"])

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.ACTION_DIM,):
            raise ValueError("planar action must have shape (2,)")
        response = self._request({
            "command": "step",
            "action": action.tolist(),
        })
        return (
            self._observation(response["observation"]),
            float(response["reward"]),
            bool(response["done"]),
            dict(response["info"]),
        )

    def teacher_step(self):
        """Advance Gazebo with the current DWA action during BC warm-up."""
        response = self._request({"command": "teacher_step"})
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
        finally:
            try:
                self.socket.close()
            finally:
                self.socket = None

    def _request(self, request):
        if self.socket is None:
            raise RuntimeError("planar environment client is closed")
        payload = (
            json.dumps(request, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self.socket.sendall(payload)
        response = self._receive_response()
        if not bool(response.get("ok", False)):
            raise RuntimeError(
                "planar environment server error: {}".format(
                    response.get("error", "unknown error")
                )
            )
        return response["result"]

    def _receive_response(self):
        while b"\n" not in self.receive_buffer:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise RuntimeError(
                    "planar environment server closed the connection"
                )
            self.receive_buffer += chunk
        raw_response, self.receive_buffer = self.receive_buffer.split(
            b"\n",
            1,
        )
        return json.loads(raw_response.decode("utf-8"))

    @classmethod
    def _observation(cls, value):
        observation = np.asarray(value, dtype=np.float32)
        if observation.shape != (cls.OBS_DIM,):
            raise RuntimeError("planar observation must have shape (11,)")
        if not np.all(np.isfinite(observation)):
            raise RuntimeError(
                "planar environment returned non-finite observation"
            )
        return observation

    def __enter__(self):
        return self

    def __exit__(self, _error_type, _error, _traceback):
        self.close()
