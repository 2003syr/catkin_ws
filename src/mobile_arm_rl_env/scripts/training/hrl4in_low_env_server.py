#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python-2 ROS/Gazebo environment server for a Python-3 PPO trainer."""

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

from training.hrl4in_low_training_env import HRL4INLowTrainingEnv


class HRL4INLowEnvironmentServer(object):
    """Serve reset/step requests through newline-delimited JSON on localhost."""

    def __init__(self):
        self.host = str(rospy.get_param("~host", "127.0.0.1"))
        self.port = int(rospy.get_param("~port", 5555))
        self.backlog = int(rospy.get_param("~backlog", 1))
        self.environment = HRL4INLowTrainingEnv(
            time_scale=rospy.get_param("~time_scale", 30),
            max_position_subgoal=rospy.get_param(
                "~max_position_subgoal",
                0.08,
            ),
            subgoal_tolerance=rospy.get_param(
                "~subgoal_tolerance",
                [0.02, 0.02, 0.05, 0.01, 0.01, 0.01],
            ),
            intrinsic_reward_scale=rospy.get_param(
                "~intrinsic_reward_scale",
                30.0,
            ),
            subgoal_achieved_reward=rospy.get_param(
                "~subgoal_achieved_reward",
                1.0,
            ),
            collision_reward_weight=rospy.get_param(
                "~collision_reward_weight",
                0.0,
            ),
            extrinsic_reward_weight=rospy.get_param(
                "~extrinsic_reward_weight",
                0.0,
            ),
            enable_teacher=rospy.get_param("~enable_teacher", True),
        )
        self.server_socket = None
        self.running = True

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
        self.server_socket.listen(self.backlog)
        self.server_socket.settimeout(1.0)
        rospy.loginfo(
            "HRL4IN low environment server listening on %s:%d",
            self.host,
            self.port,
        )

        while self.running and not rospy.is_shutdown():
            try:
                connection, address = self.server_socket.accept()
            except socket.timeout:
                continue
            except socket.error:
                if self.running:
                    raise
                break

            rospy.loginfo(
                "HRL4IN PPO trainer connected from %s:%d",
                address[0],
                address[1],
            )
            try:
                self._serve_connection(connection)
            finally:
                try:
                    connection.close()
                except socket.error:
                    pass
                rospy.loginfo("HRL4IN PPO trainer disconnected")

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
                    response, should_close = self._handle_request(request)
                    response = {
                        "ok": True,
                        "result": _json_safe(response),
                    }
                except Exception as error:
                    rospy.logerr(
                        "HRL4IN environment request failed: %s\n%s",
                        error,
                        traceback.format_exc(),
                    )
                    response = {
                        "ok": False,
                        "error": str(error),
                    }

                payload = (
                    json.dumps(response, separators=(",", ":"))
                    + "\n"
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
            }, False

        if command == "reset":
            observation = self.environment.reset(
                force_episode=bool(request.get("force_episode", False))
            )
            return {
                "observation": observation,
                "subgoal_index": self.environment.subgoal_index,
            }, False

        if command == "step":
            observation, reward, done, info = self.environment.step(
                request["action"]
            )
            return {
                "observation": observation,
                "reward": reward,
                "done": done,
                "info": info,
            }, False

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
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
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
    rospy.init_node("hrl4in_low_environment_server")
    server = HRL4INLowEnvironmentServer()
    rospy.on_shutdown(server.stop)
    try:
        server.serve_forever()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
