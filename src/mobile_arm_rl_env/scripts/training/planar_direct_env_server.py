#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python-2 ROS server for pose-guided planar subgoal PPO."""

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

from training.planar_subgoal_training_env import (
    PlanarSubgoalTrainingEnv,
)


class PlanarDirectEnvironmentServer(object):
    """Expose local-subgoal and path-preview rollouts without DWA."""

    def __init__(self):
        self.host = str(rospy.get_param("~host", "127.0.0.1"))
        self.port = int(rospy.get_param("~port", 5558))
        self.backlog = int(rospy.get_param("~backlog", 1))
        self.curriculum_stage = str(
            rospy.get_param("~curriculum_stage", "unspecified")
        )
        self.environment = PlanarSubgoalTrainingEnv(
            init_ros_node=False,
            seed=rospy.get_param("~seed", 123),
        )
        self.training_subgoals = self._parse_subgoals(
            rospy.get_param("~training_subgoals", [])
        )
        self.training_scenario_labels = self._parse_scenario_labels(
            rospy.get_param("~training_scenario_labels", []),
            self.training_subgoals,
        )
        self.curriculum = self._parse_curriculum(
            rospy.get_param("~curriculum", []),
            self.training_subgoals,
        )
        self.curriculum_indices = dict(
            (stage["name"], 0) for stage in self.curriculum
        )
        self.subgoal_index = 0
        self.episode_index = 0
        self.current_scenario_label = "unspecified"
        self.current_curriculum_stage = self.curriculum_stage
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
            "Pose-guided planar PPO environment listening on %s:%d",
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
                "Direct PPO trainer connected from %s:%d",
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
                rospy.loginfo("Direct PPO trainer disconnected")

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
                        "Direct planar environment request failed: %s\n%s",
                        error,
                        traceback.format_exc(),
                    )
                    response = {
                        "ok": False,
                        "error": str(error),
                    }
                connection.sendall((
                    json.dumps(response, separators=(",", ":")) + "\n"
                ).encode("utf-8"))
                if should_close:
                    return

    def _handle_request(self, request):
        command = str(request.get("command", ""))
        if command == "ping":
            return {
                "observation_dim": self.environment.OBS_DIM,
                "action_dim": self.environment.ACTION_DIM,
                "environment_type": "pose_guided_subgoal_rl",
                "teacher_type": "none",
                "scan_bins": 36,
                "path_preview_points": 5,
                "high_level_type": "rule_based_local_pose_subgoal",
                "curriculum_stage": self.curriculum_stage,
            }, False
        if command == "reset":
            return self._reset(request), False
        if command == "step":
            return self._step(request["action"]), False
        if command == "close":
            return {"closed": True}, True
        if command == "shutdown":
            self.running = False
            return {"shutdown": True}, True
        raise ValueError("unknown environment command: {}".format(command))

    def _reset(self, request):
        (
            subgoal,
            scenario_label,
            curriculum_stage,
        ) = self._next_subgoal(
            training_step=request.get("training_step"),
            evaluation_index=request.get("evaluation_index"),
        )
        observation = self.environment.reset(target_position=subgoal)
        self.episode_index += 1
        self.current_scenario_label = scenario_label
        self.current_curriculum_stage = curriculum_stage
        position = self.environment.target_position
        rospy.loginfo(
            "Direct PPO episode=%d stage=%s scenario=%s "
            "subgoal=[%.3f, %.3f]",
            self.episode_index,
            self.current_curriculum_stage,
            self.current_scenario_label,
            position[0],
            position[1],
        )
        return {
            "observation": observation,
            "episode_index": self.episode_index,
            "subgoal": position,
            "scenario_label": self.current_scenario_label,
            "curriculum_stage": self.current_curriculum_stage,
        }

    def _step(self, learner_action):
        observation, reward, done, info = self.environment.step(
            learner_action
        )
        info = dict(info)
        info.update({
            "teacher_available": False,
            "teacher_type": "none",
            "subgoal_position": self.environment.target_position,
            "scenario_label": self.current_scenario_label,
            "curriculum_stage": self.current_curriculum_stage,
        })
        if bool(info.get("high_level_replanned", False)):
            selection = info.get("high_level_selection", {})
            local_subgoal_pose = info.get(
                "local_subgoal_pose_body", [0.0, 0.0, 0.0]
            )
            rospy.loginfo(
                "Pose-guided high level update=%s trigger=%s selection=%s "
                "local_subgoal_pose=[%.3f, %.3f, %.3f] turn_sign=%.0f",
                info.get("high_level_updates", 0),
                info.get("high_level_replan_reason", ""),
                selection.get("reason", "unknown"),
                float(local_subgoal_pose[0]),
                float(local_subgoal_pose[1]),
                float(local_subgoal_pose[2]),
                float(selection.get("preferred_turn_sign", 0.0)),
            )
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }

    def _next_subgoal(self, training_step=None, evaluation_index=None):
        if self.training_subgoals is None:
            return None, "random", self.curriculum_stage

        if evaluation_index is not None:
            subgoal_index = int(evaluation_index) % len(
                self.training_subgoals
            )
            stage_name = "fixed_evaluation"
        elif training_step is not None and self.curriculum:
            stage = self._curriculum_stage_for_step(int(training_step))
            stage_name = stage["name"]
            cursor = self.curriculum_indices[stage_name]
            subgoal_index = stage["subgoal_indices"][
                cursor % len(stage["subgoal_indices"])
            ]
            self.curriculum_indices[stage_name] = cursor + 1
        else:
            subgoal_index = (
                self.subgoal_index % len(self.training_subgoals)
            )
            self.subgoal_index += 1
            stage_name = self.curriculum_stage

        subgoal_xy = self.training_subgoals[subgoal_index]
        subgoal = np.asarray([
            subgoal_xy[0],
            subgoal_xy[1],
            self.environment.target_z,
        ], dtype=np.float64)
        return (
            subgoal,
            self.training_scenario_labels[subgoal_index],
            stage_name,
        )

    def _curriculum_stage_for_step(self, training_step):
        for stage in self.curriculum:
            until_step = int(stage["until_step"])
            if until_step < 0 or training_step < until_step:
                return stage
        return self.curriculum[-1]

    @staticmethod
    def _parse_subgoals(value):
        subgoals = np.asarray(value, dtype=np.float64)
        if subgoals.size == 0:
            return None
        if (
                subgoals.ndim != 2
                or subgoals.shape[1] != 2
                or not np.all(np.isfinite(subgoals))):
            raise ValueError("training_subgoals must have shape (N, 2)")
        return [row.copy() for row in subgoals]

    @staticmethod
    def _parse_scenario_labels(value, training_subgoals):
        if training_subgoals is None:
            return []
        labels = [str(item) for item in list(value)]
        if not labels:
            return ["unspecified"] * len(training_subgoals)
        if len(labels) != len(training_subgoals):
            raise ValueError(
                "training_scenario_labels must match training_subgoals"
            )
        return labels

    @staticmethod
    def _parse_curriculum(value, training_subgoals):
        if not value:
            return []
        if training_subgoals is None:
            raise ValueError("curriculum requires training_subgoals")
        stages = []
        seen_names = set()
        previous_until_step = 0
        for raw_stage in list(value):
            if not isinstance(raw_stage, dict):
                raise ValueError("each curriculum stage must be a map")
            name = str(raw_stage.get("name", "")).strip()
            until_step = int(raw_stage.get("until_step", -1))
            indices = [
                int(item)
                for item in raw_stage.get("subgoal_indices", [])
            ]
            if not name or name in seen_names:
                raise ValueError(
                    "curriculum stage names must be unique and non-empty"
                )
            if not indices:
                raise ValueError(
                    "curriculum stage {} has no subgoals".format(name)
                )
            if any(
                    index < 0 or index >= len(training_subgoals)
                    for index in indices):
                raise ValueError(
                    "curriculum stage {} has an invalid subgoal index".format(
                        name
                    )
                )
            if until_step >= 0 and until_step <= previous_until_step:
                raise ValueError(
                    "curriculum until_step values must increase"
                )
            if stages and stages[-1]["until_step"] < 0:
                raise ValueError("open-ended curriculum stage must be last")
            stages.append({
                "name": name,
                "until_step": until_step,
                "subgoal_indices": indices,
            })
            seen_names.add(name)
            if until_step >= 0:
                previous_until_step = until_step
        if stages[-1]["until_step"] >= 0:
            raise ValueError(
                "last curriculum stage must use until_step=-1"
            )
        return stages

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
    rospy.init_node("planar_direct_environment_server")
    server = PlanarDirectEnvironmentServer()
    rospy.on_shutdown(server.stop)
    try:
        server.serve_forever()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
