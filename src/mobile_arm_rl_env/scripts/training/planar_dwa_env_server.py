#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python-2 ROS server for DWA-teacher planar PPO training."""

from __future__ import print_function

import json
import math
import os
import socket
import sys
import traceback

import numpy as np
import rospy

from actionlib_msgs.msg import GoalStatus


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.dwa_planar_teacher import DWAPlanarTeacher
from training.planar_base_training_env import PlanarBaseTrainingEnv


class PlanarDWAEnvironmentServer(object):
    """Let the learner drive Gazebo while DWA supplies advisory actions."""

    def __init__(self):
        self.host = str(rospy.get_param("~host", "127.0.0.1"))
        self.port = int(rospy.get_param("~port", 5557))
        self.backlog = int(rospy.get_param("~backlog", 1))
        self.teacher_wait_timeout = float(
            rospy.get_param("~teacher_wait_timeout", 0.30)
        )
        if self.teacher_wait_timeout < 0.0:
            raise ValueError("teacher_wait_timeout must be non-negative")

        self.environment = PlanarBaseTrainingEnv(
            init_ros_node=False,
            seed=rospy.get_param("~seed", 123),
        )
        maximum_linear_speed = (
            self.environment.base_env.action_max_vel["x"]
            * self.environment.base_action_scale
        )
        maximum_yaw_rate = (
            self.environment.base_env.action_max_vel["z"]
            * self.environment.base_action_scale
        )
        self.teacher = DWAPlanarTeacher(
            cmd_vel_topic=rospy.get_param(
                "~teacher_cmd_vel_topic",
                "/dwa_teacher_cmd_vel",
            ),
            move_base_action=rospy.get_param(
                "~move_base_action",
                "/move_base",
            ),
            global_frame=rospy.get_param(
                "~navigation_global_frame",
                "base_link",
            ),
            maximum_linear_speed=maximum_linear_speed,
            maximum_yaw_rate=maximum_yaw_rate,
            command_timeout=rospy.get_param(
                "~teacher_command_timeout",
                0.50,
            ),
            server_timeout=rospy.get_param(
                "~move_base_server_timeout",
                30.0,
            ),
        )
        self.training_targets = self._parse_targets(
            rospy.get_param("~training_targets", [])
        )
        self.target_index = 0
        self.episode_index = 0
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
            "Planar DWA PPO environment listening on %s:%d",
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
                "Planar PPO trainer connected from %s:%d",
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
                rospy.loginfo("Planar PPO trainer disconnected")

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
                    response = {
                        "ok": True,
                        "result": _json_safe(result),
                    }
                except Exception as error:
                    rospy.logerr(
                        "Planar DWA environment request failed: %s\n%s",
                        error,
                        traceback.format_exc(),
                    )
                    response = {
                        "ok": False,
                        "error": str(error),
                    }
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
                "teacher_type": "move_base_dwa",
            }, False
        if command == "reset":
            return self._reset(), False
        if command == "step":
            return self._step(
                learner_action=request["action"],
                execute_teacher=False,
            ), False
        if command == "teacher_step":
            return self._step(
                learner_action=None,
                execute_teacher=True,
            ), False
        if command == "close":
            return {"closed": True}, True
        if command == "shutdown":
            self.running = False
            return {"shutdown": True}, True
        raise ValueError("unknown environment command: {}".format(command))

    def _reset(self):
        self.teacher.cancel_goal()
        target = self._next_target()
        observation = self.environment.reset(target_position=target)
        target_position = self.environment.target_position
        goal_index = self.teacher.start_goal(target_position[0:2])
        self.episode_index += 1
        rospy.loginfo(
            "Planar DWA teacher episode=%d goal=%d target=[%.3f, %.3f]",
            self.episode_index,
            goal_index,
            target_position[0],
            target_position[1],
        )
        return {
            "observation": observation,
            "episode_index": self.episode_index,
            "target": target_position,
            "teacher_goal_index": goal_index,
        }

    def _step(self, learner_action, execute_teacher):
        teacher_action, teacher_available, teacher_info = (
            self.teacher.action(
                wait_timeout=self.teacher_wait_timeout
            )
        )
        teacher_goal_restarted = False
        teacher_restart_goal_index = -1
        if (
                not teacher_available
                and bool(teacher_info["goal_terminal"])
                and int(teacher_info["goal_status"])
                != GoalStatus.SUCCEEDED):
            teacher_restart_goal_index = self._restart_teacher_goal(
                teacher_info["goal_status"]
            )
            teacher_goal_restarted = True
            teacher_action, teacher_available, teacher_info = (
                self.teacher.action(
                    wait_timeout=max(
                        self.teacher_wait_timeout,
                        1.0,
                    )
                )
            )

        # A zero teacher action is the correct label if move_base reached its
        # tolerance just before the training task observes the same state.
        if (
                not teacher_available
                and int(teacher_info["goal_status"])
                == GoalStatus.SUCCEEDED):
            teacher_action = np.zeros(
                self.environment.ACTION_DIM,
                dtype=np.float32,
            )
            teacher_available = True
        if execute_teacher:
            if not teacher_available:
                raise RuntimeError(
                    "DWA teacher action unavailable during warm-up "
                    "(goal_status={}, command_age={})".format(
                        teacher_info["goal_status"],
                        teacher_info["command_age"],
                    )
                )
            executed_action = teacher_action
            learner_action = np.zeros(
                self.environment.ACTION_DIM,
                dtype=np.float32,
            )
        else:
            learner_action = np.asarray(
                learner_action,
                dtype=np.float32,
            )
            executed_action = learner_action
        observation, reward, done, info = self.environment.step(
            executed_action
        )
        # A learner action can move the robot back outside the environment's
        # tolerance after move_base has already declared success.  Re-issue
        # the same goal so the next DAgger state still receives a DWA label.
        if (
                not done
                and bool(teacher_info["goal_terminal"])):
            teacher_restart_goal_index = self._restart_teacher_goal(
                teacher_info["goal_status"]
            )
            teacher_goal_restarted = True
        info = dict(info)
        info.update({
            "teacher_action": teacher_action.copy(),
            "teacher_available": bool(teacher_available),
            "teacher_type": "move_base_dwa",
            "teacher_executed": bool(execute_teacher),
            "learner_action": learner_action.copy(),
            "executed_training_action": np.asarray(
                executed_action,
                dtype=np.float32,
            ).copy(),
            "teacher_goal_status": int(teacher_info["goal_status"]),
            "teacher_command_age": float(
                teacher_info["command_age"]
            ),
            "teacher_lateral_velocity": float(
                teacher_info["lateral_velocity"]
            ),
            "teacher_goal_restarted": bool(
                teacher_goal_restarted
            ),
            "teacher_restart_goal_index": int(
                teacher_restart_goal_index
            ),
        })
        if done:
            self.teacher.cancel_goal()
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }

    def _restart_teacher_goal(self, previous_status):
        target_position = self.environment.target_position
        goal_index = self.teacher.start_goal(target_position[0:2])
        rospy.logwarn(
            "Planar DWA teacher restarted goal=%d after terminal "
            "status=%d while the training episode remained active",
            goal_index,
            int(previous_status),
        )
        return goal_index

    def _next_target(self):
        if self.training_targets is None:
            return None
        target_xy = self.training_targets[
            self.target_index % len(self.training_targets)
        ]
        self.target_index += 1
        return np.asarray([
            target_xy[0],
            target_xy[1],
            self.environment.target_z,
        ], dtype=np.float64)

    @staticmethod
    def _parse_targets(value):
        targets = np.asarray(value, dtype=np.float64)
        if targets.size == 0:
            return None
        if (
                targets.ndim != 2
                or targets.shape[1] != 2
                or not np.all(np.isfinite(targets))):
            raise ValueError("training_targets must have shape (N, 2)")
        return [row.copy() for row in targets]

    def stop(self):
        self.running = False
        self.teacher.cancel_goal()
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
    rospy.init_node("planar_dwa_environment_server")
    server = PlanarDWAEnvironmentServer()
    rospy.on_shutdown(server.stop)
    try:
        server.serve_forever()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
