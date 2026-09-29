#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Python-3 client for the pose-guided planar PPO environment."""

import json
import socket

import numpy as np


class PlanarDirectEnvironmentClient(object):
    OBS_DIM = 62
    ACTION_DIM = 2

    def __init__(
            self,
            host="127.0.0.1",
            port=5558,
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
        contract = self._request({"command": "ping"})
        if int(contract["observation_dim"]) != self.OBS_DIM:
            raise RuntimeError("direct observation dimension mismatch")
        if int(contract["action_dim"]) != self.ACTION_DIM:
            raise RuntimeError("direct action dimension mismatch")
        if str(contract.get("teacher_type")) != "none":
            raise RuntimeError("direct environment unexpectedly has a teacher")
        self.curriculum_stage = str(
            contract.get("curriculum_stage", "unspecified")
        )
        self.last_reset_info = {}

    def reset(
            self,
            return_info=False,
            training_step=None,
            evaluation_index=None):
        request = {"command": "reset"}
        if training_step is not None:
            request["training_step"] = int(training_step)
        if evaluation_index is not None:
            request["evaluation_index"] = int(evaluation_index)
        response = self._request(request)
        observation = self._observation(response["observation"])
        self.curriculum_stage = str(
            response.get("curriculum_stage", self.curriculum_stage)
        )
        self.last_reset_info = dict(response)
        self.last_reset_info["observation"] = observation.copy()
        if return_info:
            return observation, dict(self.last_reset_info)
        return observation

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
            raise RuntimeError("direct environment client is closed")
        self.socket.sendall((
            json.dumps(request, separators=(",", ":")) + "\n"
        ).encode("utf-8"))
        response = self._receive_response()
        if not bool(response.get("ok", False)):
            raise RuntimeError(
                "direct environment server error: {}".format(
                    response.get("error", "unknown error")
                )
            )
        return response["result"]

    def _receive_response(self):
        while b"\n" not in self.receive_buffer:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise RuntimeError(
                    "direct environment server closed the connection"
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
            raise RuntimeError(
                "direct observation must have shape ({},)".format(
                    cls.OBS_DIM
                )
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError(
                "direct environment returned non-finite observation"
            )
        return observation

    def __enter__(self):
        return self

    def __exit__(self, _error_type, _error, _traceback):
        self.close()
