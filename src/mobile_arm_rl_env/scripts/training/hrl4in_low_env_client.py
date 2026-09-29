#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Python-3 client for the ROS/Gazebo low-level environment server."""

import json
import socket

import numpy as np


class HRL4INLowEnvironmentClient(object):
    OBS_DIM = 68
    ACTION_DIM = 10

    def __init__(
            self,
            host="127.0.0.1",
            port=5555,
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
            raise RuntimeError(
                "environment observation dim is {}, expected {}".format(
                    dimensions["observation_dim"],
                    self.OBS_DIM,
                )
            )
        if int(dimensions["action_dim"]) != self.ACTION_DIM:
            raise RuntimeError(
                "environment action dim is {}, expected {}".format(
                    dimensions["action_dim"],
                    self.ACTION_DIM,
                )
            )

    def reset(self, force_episode=False):
        response = self._request({
            "command": "reset",
            "force_episode": bool(force_episode),
        })
        return self._observation(response["observation"])

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.ACTION_DIM,):
            raise ValueError(
                "action must have shape (10,), got {}".format(action.shape)
            )
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
            raise RuntimeError("environment client is closed")
        payload = (
            json.dumps(request, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self.socket.sendall(payload)
        response = self._receive_response()
        if not bool(response.get("ok", False)):
            raise RuntimeError(
                "environment server error: {}".format(
                    response.get("error", "unknown error")
                )
            )
        return response["result"]

    def _receive_response(self):
        while b"\n" not in self.receive_buffer:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise RuntimeError("environment server closed the connection")
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
            raise RuntimeError(
                "observation must have shape (68,), got {}".format(
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
