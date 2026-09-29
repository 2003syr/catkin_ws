#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""JSON server for the 66-D fused low-level residual environment.

The client sends only five bounded residuals.  The environment always adds
those residuals to the coordinated rule teacher and keeps the existing safety
filter and tracked-base kinematics unchanged.
"""

from __future__ import print_function

import json
import math
import os
import socket
import sys
import traceback

import numpy as np
import rospy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_low_residual_training_env import (  # noqa: E402
    FusedLowResidualTrainingEnv,
)
from training.fused_box_subgoal_residual_training_env import (  # noqa: E402
    FusedBoxSubgoalResidualTrainingEnv,
)


class FusedLowResidualEnvironmentServer(object):

    def __init__(self):
        self.host = str(rospy.get_param("~host", "127.0.0.1"))
        self.port = int(rospy.get_param("~port", 5563))
        self.environment_type = str(
            rospy.get_param("~environment_type", "standard")
        ).strip().lower()
        environment_class = {
            "standard": FusedLowResidualTrainingEnv,
            "fused": FusedLowResidualTrainingEnv,
            "box_subgoal": FusedBoxSubgoalResidualTrainingEnv,
            "fused_box_subgoal": FusedBoxSubgoalResidualTrainingEnv,
        }.get(self.environment_type)
        if environment_class is None:
            raise ValueError(
                "unknown residual environment_type: {}".format(
                    self.environment_type
                )
            )
        self.environment = environment_class(
            init_ros_node=False,
            seed=rospy.get_param("~seed", 123),
            enable_teacher=True,
        )
        self.running = True
        self.server_socket = None

    def serve_forever(self):
        self.server_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        self.server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(1.0)
        rospy.loginfo(
            "Fused residual environment type=%s listening on %s:%d "
            "obs=%d residual_action=%d nominal_action=8",
            self.environment_type,
            self.host,
            self.port,
            self.environment.OBS_DIM,
            self.environment.ACTION_DIM,
        )
        while self.running and not rospy.is_shutdown():
            try:
                connection, _ = self.server_socket.accept()
            except socket.timeout:
                continue
            try:
                self._serve_connection(connection)
            finally:
                connection.close()

    def _serve_connection(self, connection):
        connection.settimeout(1.0)
        receive_buffer = b""
        while self.running and not rospy.is_shutdown():
            try:
                chunk = connection.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                return
            receive_buffer += chunk
            while b"\n" in receive_buffer:
                raw_request, receive_buffer = receive_buffer.split(
                    b"\n",
                    1,
                )
                if not raw_request.strip():
                    continue
                should_close = False
                try:
                    request = json.loads(raw_request.decode("utf-8"))
                    result, should_close = self._handle_request(request)
                    response = {"ok": True, "result": _json_safe(result)}
                except Exception as error:
                    rospy.logerr(
                        "Fused residual request failed: %s\n%s",
                        error,
                        traceback.format_exc(),
                    )
                    response = {"ok": False, "error": str(error)}
                payload = (
                    json.dumps(response, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                connection.sendall(payload)
                if should_close:
                    return

    def _handle_request(self, request):
        command = str(request.get("command", ""))
        if command == "ping":
            return {
                "observation_dim": self.environment.OBS_DIM,
                "action_dim": self.environment.ACTION_DIM,
                "nominal_action_dim": 8,
                "policy_type": "fused_low_residual_ppo",
                "environment_type": self.environment_type,
                "subgoal_context_contract": getattr(
                    self.environment,
                    "SUBGOAL_CONTEXT_CONTRACT",
                    "legacy_all_ones_subgoal_mask",
                ),
                "subgoal_types": ["DIRECT", "DETOUR", "TERMINAL"],
                "action_semantics": (
                    "delta_v,delta_omega,delta_vx_ee,"
                    "delta_vy_ee,delta_vz_ee"
                ),
            }, False
        if command == "reset":
            observation = self.environment.reset(
                scenario=request.get("scenario"),
                max_steps=request.get("max_steps"),
            )
            if hasattr(self.environment, "nominal_action"):
                nominal_action = self.environment.nominal_action()
            else:
                nominal_action = self.environment.teacher.predict(observation)
            zero_residual = np.zeros(5, dtype=np.float32)
            return {
                "observation": observation,
                "nominal_action": nominal_action,
                "teacher_action": zero_residual,
                "zero_residual_action": zero_residual,
                "reset_info": self.environment.last_reset_info,
            }, False
        if command == "step":
            observation, reward, done, info = self.environment.step(
                request["action"]
            )
            nominal_action = info.get("nominal_action")
            if nominal_action is None:
                if hasattr(self.environment, "nominal_action"):
                    nominal_action = self.environment.nominal_action()
                else:
                    nominal_action = self.environment.teacher.predict(
                        observation
                    )
            zero_residual = np.zeros(5, dtype=np.float32)
            return {
                "observation": observation,
                "reward": reward,
                "done": done,
                "info": info,
                "nominal_action": nominal_action,
                "teacher_action": zero_residual,
                "zero_residual_action": zero_residual,
                "teacher_diagnostics": self._teacher_diagnostics(),
            }, False
        if command == "close":
            self._publish_zero_commands()
            return {"closed": True}, True
        if command == "shutdown":
            self._publish_zero_commands()
            self.running = False
            return {"shutdown": True}, True
        raise ValueError("unknown environment command: {}".format(command))

    def _teacher_diagnostics(self):
        """Expose the richest available diagnostics without changing control."""
        if hasattr(self.environment, "teacher_diagnostics"):
            return self.environment.teacher_diagnostics()
        return self.environment.teacher.diagnostics()

    def _publish_zero_commands(self):
        """Do not leave an episode's last command active after disconnect."""
        reach_env = getattr(self.environment, "reach_env", None)
        if reach_env is not None:
            reach_env._publish_zero_velocity(repeat=3)

    def stop(self):
        self.running = False
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except socket.error:
                pass
            self.server_socket = None
        self.environment.stop()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return dict(
            (str(key), _json_safe(item))
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return (
            value
            if not math.isnan(value) and not math.isinf(value)
            else None
        )
    return value


def main():
    rospy.init_node("fused_low_residual_environment_server")
    server = FusedLowResidualEnvironmentServer()
    rospy.on_shutdown(server.stop)
    try:
        server.serve_forever()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
