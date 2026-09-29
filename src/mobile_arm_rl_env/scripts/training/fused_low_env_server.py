#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Local JSON server for the coordinated 66-D/8-D low-level environment."""

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

from training.fused_low_training_env import FusedLowLevelTrainingEnv
from training.fused_box_detour_training_env import (
    FusedBoxDetourTrainingEnv,
)


class FusedLowEnvironmentServer(object):

    def __init__(self):
        self.host = str(rospy.get_param("~host", "127.0.0.1"))
        self.port = int(rospy.get_param("~port", 5562))
        environment_type = str(
            rospy.get_param("~environment_type", "standard")
        ).strip().lower()
        environment_class = {
            "standard": FusedLowLevelTrainingEnv,
            "fused": FusedLowLevelTrainingEnv,
            "box_detour": FusedBoxDetourTrainingEnv,
            "fused_box_detour": FusedBoxDetourTrainingEnv,
        }.get(environment_type)
        if environment_class is None:
            raise ValueError(
                "unknown fused environment_type: {}".format(
                    environment_type
                )
            )
        self.environment_type = environment_type
        self.environment = environment_class(
            init_ros_node=False,
            seed=rospy.get_param("~seed", 123),
            enable_teacher=rospy.get_param("~enable_teacher", True),
        )
        self.running = True
        self.server_socket = None

    def _teacher_label(self, observation):
        """Build the executable teacher label used by BC-guided PPO.

        Fused BC is trained against ``safe_teacher_actions``.  Returning the
        raw rule-teacher command here would make PPO's supervised term pull
        the actor away from its BC initialization whenever the joint-limit
        safety layer intervenes.
        """
        raw_action = self.environment.teacher_action(observation)
        if raw_action is None:
            return {
                "teacher_action": None,
                "raw_teacher_action": None,
                "teacher_safety_filter_active": False,
                "teacher_safety_filter_delta": 0.0,
            }
        safe_action, unused_filter_info = (
            self.environment.filter_fused_action(raw_action)
        )
        raw_action = np.asarray(raw_action, dtype=np.float32)
        safe_action = np.asarray(safe_action, dtype=np.float32)
        maximum_delta = float(np.max(np.abs(safe_action - raw_action)))
        return {
            # Keep the established client contract, but make its meaning
            # agree with the target used by train_fused_low_bc.py.
            "teacher_action": safe_action,
            "raw_teacher_action": raw_action,
            "teacher_safety_filter_active": bool(maximum_delta > 1.0e-6),
            "teacher_safety_filter_delta": maximum_delta,
        }

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
            "Fused low environment type=%s listening on %s:%d "
            "obs=%d action=%d",
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
                        "Fused low request failed: %s\n%s",
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
                "environment_type": self.environment_type,
                "action_semantics": (
                    "tracked_v,tracked_omega,"
                    "joint1,joint2,joint3,joint4,joint5,joint6"
                ),
            }, False
        if command == "reset":
            observation = self.environment.reset(
                scenario=request.get("scenario"),
                max_steps=request.get("max_steps"),
            )
            result = {
                "observation": observation,
                "reset_info": self.environment.last_reset_info,
            }
            result.update(self._teacher_label(observation))
            return result, False
        if command == "step":
            observation, reward, done, info = self.environment.step(
                request["action"]
            )
            result = {
                "observation": observation,
                "reward": reward,
                "done": done,
                "info": info,
            }
            result.update(self._teacher_label(observation))
            result["teacher_diagnostics"] = (
                self.environment.teacher_diagnostics()
            )
            return result, False
        if command == "close":
            return {"closed": True}, True
        if command == "shutdown":
            self.running = False
            return {"shutdown": True}, True
        raise ValueError("unknown environment command: {}".format(command))

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
    rospy.init_node("fused_low_environment_server")
    server = FusedLowEnvironmentServer()
    rospy.on_shutdown(server.stop)
    try:
        server.serve_forever()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
