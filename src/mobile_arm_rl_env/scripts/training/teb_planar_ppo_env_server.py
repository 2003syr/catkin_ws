#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ROS/Python-2 environment server for PPO on live TEB path previews."""

from __future__ import print_function

import json
import math
import os
import socket
import sys
import threading
import time
import traceback

import actionlib
import numpy as np
import rospy

from actionlib_msgs.msg import GoalStatus
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.collect_teb_planar_dataset import (
    ACTION_DIM,
    OBSERVATION_DIM,
)
from training.evaluate_teb_planar_bc import TebPlanarBcEvaluator


class TebPlanarPpoEnvironmentServer(TebPlanarBcEvaluator):
    """Execute learner actions while TEB supplies only a replanned path."""

    def __init__(self):
        self.host = str(rospy.get_param("~host", "127.0.0.1"))
        self.port = int(rospy.get_param("~port", 5558))
        self.backlog = int(rospy.get_param("~backlog", 1))
        self.control_period = float(
            rospy.get_param("~control_period", 0.10)
        )
        self.max_steps = int(rospy.get_param("~max_steps", 1000))
        self.success_distance = float(
            rospy.get_param("~success_distance", 0.08)
        )
        self.sample_wait_timeout = float(
            rospy.get_param("~sample_wait_timeout", 15.0)
        )
        self.message_timeout = float(
            rospy.get_param("~message_timeout", 0.75)
        )
        self.scan_clip = float(rospy.get_param("~scan_clip", 10.0))
        self.lookahead_distance = float(
            rospy.get_param("~lookahead_distance", 0.50)
        )
        self.path_point_spacing = float(
            rospy.get_param("~path_point_spacing", 0.10)
        )
        self.path_remaining_scale = float(
            rospy.get_param("~path_remaining_scale", 1.0)
        )
        self.max_linear_speed = float(
            rospy.get_param("~max_linear_speed", 0.05)
        )
        self.max_yaw_rate = float(
            rospy.get_param("~max_yaw_rate", 0.125)
        )
        self.goal_frame = str(
            rospy.get_param("~goal_frame", "base_link")
        )
        self.local_plan_topic = str(rospy.get_param(
            "~local_plan_topic",
            "/move_base/TebLocalPlannerROS/local_plan",
        ))
        self.reference_cmd_topic = str(rospy.get_param(
            "~reference_cmd_topic",
            "/teb_reference_cmd_vel",
        ))
        self.learner_cmd_topic = str(rospy.get_param(
            "~learner_cmd_topic",
            "/learner_cmd_vel",
        ))
        self.emergency_translation_distance = float(
            rospy.get_param(
                "~emergency_translation_distance", 0.11
            )
        )
        self.emergency_rotation_distance = float(
            rospy.get_param("~emergency_rotation_distance", 0.08)
        )
        self.safety_sector_half_angle = float(
            rospy.get_param(
                "~safety_sector_half_angle",
                math.radians(35.0),
            )
        )
        self.progress_reward_scale = float(
            rospy.get_param("~progress_reward_scale", 4.0)
        )
        self.local_progress_reward_scale = float(
            rospy.get_param("~local_progress_reward_scale", 8.0)
        )
        self.success_reward = float(
            rospy.get_param("~success_reward", 5.0)
        )
        self.collision_penalty = float(
            rospy.get_param("~collision_penalty", 5.0)
        )
        self.timeout_penalty = float(
            rospy.get_param("~timeout_penalty", 2.0)
        )
        self.planner_failure_penalty = float(
            rospy.get_param("~planner_failure_penalty", 2.0)
        )
        self.time_penalty = float(
            rospy.get_param("~time_penalty", 0.005)
        )
        self.action_penalty_scale = float(
            rospy.get_param("~action_penalty_scale", 0.002)
        )
        self.smoothness_penalty_scale = float(
            rospy.get_param("~smoothness_penalty_scale", 0.01)
        )
        self.shield_penalty = float(
            rospy.get_param("~shield_penalty", 0.02)
        )
        self.curriculum_stage = str(rospy.get_param(
            "~curriculum_stage",
            "teb_path_guided_all_routes",
        ))
        self.goals = self._parse_goals(rospy.get_param(
            "~goals",
            [[1.82, -0.13, 0.5 * math.pi]],
        ))
        self.goal_labels = self._parse_goal_labels(
            rospy.get_param("~goal_labels", []),
            len(self.goals),
        )
        self.scenario_labels = self._parse_scenario_labels(
            rospy.get_param("~scenario_labels", []),
            len(self.goals),
        )
        self._validate_server_parameters()

        self._lock = threading.Lock()
        self._scan = None
        self._odometry = None
        self._path = None
        self._command = None
        self._receive_times = {}
        self._episode_collision = False
        self._episode_collision_names = set()

        rospy.Subscriber(
            "/scan", LaserScan, self._scan_callback, queue_size=5
        )
        rospy.Subscriber(
            "/odom", Odometry, self._odometry_callback, queue_size=20
        )
        rospy.Subscriber(
            self.local_plan_topic,
            Path,
            self._path_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            self.reference_cmd_topic,
            Twist,
            self._command_callback,
            queue_size=20,
        )
        rospy.Subscriber(
            "/base_contacts",
            ContactsState,
            self._contacts_callback,
            queue_size=20,
        )
        self.command_publisher = rospy.Publisher(
            self.learner_cmd_topic,
            Twist,
            queue_size=5,
        )
        self.action_client = actionlib.SimpleActionClient(
            "/move_base",
            MoveBaseAction,
        )

        self.server_socket = None
        self.running = True
        self.route_cursor = 0
        self.episode_index = 0
        self.episode_active = False
        self.current_goal_index = 0
        self.current_target_world = None
        self.current_scenario_label = "unspecified"
        self.previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_observation = None
        self.last_teacher_action = np.zeros(
            ACTION_DIM, dtype=np.float32
        )
        self.previous_distance = 0.0
        self.previous_local_distance = 0.0
        self.initial_distance = 0.0
        self.minimum_distance = 0.0
        self.step_count = 0

    def serve_forever(self):
        rospy.loginfo("Waiting for move_base action server")
        if not self.action_client.wait_for_server(
                rospy.Duration(30.0)):
            raise RuntimeError("move_base action server is unavailable")
        self._wait_for_base_sensors()
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
            "TEB path-guided PPO environment listening on %s:%d",
            self.host,
            self.port,
        )
        try:
            while self.running and not rospy.is_shutdown():
                try:
                    connection, address = self.server_socket.accept()
                except socket.timeout:
                    continue
                rospy.loginfo(
                    "TEB PPO trainer connected from %s:%d",
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
                    self._stop_base()
                    rospy.loginfo("TEB PPO trainer disconnected")
        finally:
            self.action_client.cancel_all_goals()
            self._stop_base()
            if self.server_socket is not None:
                self.server_socket.close()

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
                request_data, receive_buffer = receive_buffer.split(
                    b"\n", 1
                )
                if not request_data.strip():
                    continue
                close_connection = False
                try:
                    request = json.loads(
                        request_data.decode("utf-8")
                    )
                    result, close_connection = self._handle_request(
                        request
                    )
                    response = {
                        "ok": True,
                        "result": _json_safe(result),
                    }
                except Exception as error:
                    rospy.logerr(
                        "TEB PPO environment request failed: %s\n%s",
                        error,
                        traceback.format_exc(),
                    )
                    self._stop_base()
                    response = {
                        "ok": False,
                        "error": str(error),
                    }
                connection.sendall((
                    json.dumps(response, separators=(",", ":")) + "\n"
                ).encode("utf-8"))
                if close_connection:
                    return

    def _handle_request(self, request):
        command = str(request.get("command", ""))
        if command == "ping":
            return {
                "observation_dim": OBSERVATION_DIM,
                "action_dim": ACTION_DIM,
                "environment_type": "teb_path_guided_planar_rl",
                "teacher_type": "none",
                "scan_bins": 36,
                "path_preview_points": 5,
                "high_level_type": "teb_local_path",
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
        # SimpleActionClient logs an internal state error if cancel_goal() is
        # sent after move_base has already reported DONE.  Cancel only goals
        # that are still pending or active.
        self._cancel_active_goal()
        self._stop_base()
        evaluation_index = request.get("evaluation_index")
        if evaluation_index is None:
            goal_index = self.route_cursor % len(self.goals)
            self.route_cursor += 1
        else:
            goal_index = int(evaluation_index) % len(self.goals)
        goal_pose = self.goals[goal_index]

        self.current_goal_index = goal_index
        self.current_scenario_label = self.scenario_labels[goal_index]
        self.previous_action = np.zeros(
            ACTION_DIM, dtype=np.float32
        )
        sample = None
        reset_error = None
        for reset_attempt in range(1, 4):
            if reset_attempt > 1:
                self._cancel_active_goal()
                self._stop_base()
                time.sleep(0.25)
            with self._lock:
                self._episode_collision = False
                self._episode_collision_names = set()
                self._path = None
                self._command = None
                self._receive_times.pop("path", None)
                self._receive_times.pop("command", None)
                odometry = self._odometry
            self.current_target_world = self._target_world_position(
                goal_pose,
                odometry,
            )
            self.action_client.send_goal(
                self._move_base_goal(goal_pose)
            )
            try:
                sample = self._wait_for_sample(
                    self.previous_action,
                    allow_terminal=True,
                )
            except RuntimeError as error:
                reset_error = error
                rospy.logwarn(
                    "TEB PPO reset route=%s attempt=%d/3 failed: "
                    "%s action_state=%d",
                    self.goal_labels[goal_index],
                    reset_attempt,
                    str(error),
                    self.action_client.get_state(),
                )
                continue
            if sample is not None:
                break
            if self.last_observation is not None:
                rospy.logwarn(
                    "TEB PPO reset route=%s became terminal before a "
                    "fresh path; reusing the last observation for the "
                    "terminal transition (action_state=%d)",
                    self.goal_labels[goal_index],
                    self.action_client.get_state(),
                )
                sample = (
                    self.last_observation.copy(),
                    self.last_teacher_action.copy(),
                    None,
                )
                break
            reset_error = RuntimeError(
                "goal became terminal before the first observation"
            )
        if sample is None:
            raise RuntimeError(
                "TEB PPO reset failed after 3 attempts for route {}: "
                "{}".format(
                    self.goal_labels[goal_index],
                    reset_error,
                )
            )

        observation, teacher_action, unused_scan = sample
        self.last_observation = observation.copy()
        self.last_teacher_action = teacher_action.copy()
        self.initial_distance = self._current_target_distance(
            self.current_target_world
        )
        self.previous_distance = self.initial_distance
        self.minimum_distance = self.initial_distance
        self.previous_local_distance = float(
            np.linalg.norm(observation[0:2])
        )
        self.step_count = 0
        self.episode_index += 1
        self.episode_active = True
        rospy.loginfo(
            "TEB PPO episode=%d route=%s scenario=%s "
            "goal=[%.3f,%.3f,%.3f] distance=%.4f",
            self.episode_index,
            self.goal_labels[goal_index],
            self.current_scenario_label,
            goal_pose[0],
            goal_pose[1],
            goal_pose[2],
            self.initial_distance,
        )
        return {
            "observation": observation,
            "episode_index": self.episode_index,
            "subgoal": goal_pose[0:2],
            "initial_distance": self.initial_distance,
            "target_world": self.current_target_world,
            "route_label": self.goal_labels[goal_index],
            "scenario_label": self.current_scenario_label,
            "curriculum_stage": self.curriculum_stage,
        }

    def _step(self, learner_action):
        if not self.episode_active:
            raise RuntimeError("reset must be called before step")
        raw_action = np.clip(
            np.asarray(learner_action, dtype=np.float32),
            -1.0,
            1.0,
        )
        if raw_action.shape != (ACTION_DIM,):
            raise ValueError("learner action must have shape (2,)")
        with self._lock:
            scan = self._scan
        if scan is None:
            raise RuntimeError("LaserScan is unavailable")
        safe_action, shield_reasons, clearances = (
            self._apply_emergency_shield(raw_action, scan)
        )
        self._publish_normalized_action(safe_action)
        time.sleep(self.control_period)
        self.step_count += 1

        # move_base/TEB stops publishing its local plan and reference command
        # as soon as a goal becomes terminal.  Check that state before waiting
        # for another TEB sample; otherwise a valid success can be misreported
        # as a sensor timeout.
        sample = None
        if (
                self.step_count < self.max_steps
                and not self._terminal_signal_present()):
            sample = self._wait_for_sample(
                safe_action,
                allow_terminal=True,
            )
        if sample is None:
            if self.last_observation is None:
                raise RuntimeError(
                    "terminal transition has no previous observation"
                )
            observation = self.last_observation.copy()
            teacher_action = self.last_teacher_action.copy()
        else:
            observation, teacher_action, unused_scan = sample
            self.last_observation = observation.copy()
            self.last_teacher_action = teacher_action.copy()

        distance = self._current_target_distance(
            self.current_target_world
        )
        local_distance = float(np.linalg.norm(observation[0:2]))
        progress = float(self.previous_distance - distance)
        local_progress = float(
            self.previous_local_distance - local_distance
        )
        self.minimum_distance = min(self.minimum_distance, distance)
        status = self.action_client.get_state()
        with self._lock:
            contact_collision = bool(self._episode_collision)
            collision_names = sorted(self._episode_collision_names)
        success = bool(
            status == GoalStatus.SUCCEEDED
            and not contact_collision
        )
        planner_failed = bool(status in (
            GoalStatus.PREEMPTED,
            GoalStatus.ABORTED,
            GoalStatus.REJECTED,
            GoalStatus.RECALLED,
            GoalStatus.LOST,
        ))
        timeout = bool(
            self.step_count >= self.max_steps
            and not success
            and not contact_collision
        )
        done = bool(
            success or contact_collision or planner_failed or timeout
        )

        action_penalty = (
            self.action_penalty_scale
            * float(np.mean(np.square(safe_action)))
        )
        smoothness_penalty = (
            self.smoothness_penalty_scale
            * float(np.mean(np.square(
                safe_action - self.previous_action
            )))
        )
        reward = float(
            self.progress_reward_scale * progress
            + self.local_progress_reward_scale * local_progress
            - self.time_penalty
            - action_penalty
            - smoothness_penalty
            - (self.shield_penalty if shield_reasons else 0.0)
            + (self.success_reward if success else 0.0)
            - (
                self.collision_penalty
                if contact_collision else 0.0
            )
            - (self.timeout_penalty if timeout else 0.0)
            - (
                self.planner_failure_penalty
                if planner_failed else 0.0
            )
        )
        path_cross_track = (
            float(observation[52]) * self.lookahead_distance
        )
        path_heading_error = float(observation[53]) * math.pi
        local_yaw_error = math.atan2(
            float(observation[2]),
            float(observation[3]),
        )
        teacher_mse = float(np.mean(np.square(
            raw_action - teacher_action
        )))
        self.previous_distance = distance
        self.previous_local_distance = local_distance
        self.previous_action = safe_action.copy()
        if done:
            self.episode_active = False
            if not success:
                self.action_client.cancel_goal()
            self._stop_base()
            rospy.loginfo(
                "TEB PPO result episode=%d route=%s status=%d "
                "success=%s collision=%s timeout=%s planner_failed=%s "
                "steps=%d distance=%.4f minimum=%.4f reward=%.4f "
                "teacher_mse=%.5f contacts=%s",
                self.episode_index,
                self.goal_labels[self.current_goal_index],
                status,
                str(success),
                str(contact_collision),
                str(timeout),
                str(planner_failed),
                self.step_count,
                distance,
                self.minimum_distance,
                reward,
                teacher_mse,
                str(collision_names),
            )
        info = {
            "initial_distance": self.initial_distance,
            "distance": distance,
            "minimum_distance": self.minimum_distance,
            "distance_reduction": (
                self.initial_distance - distance
            ),
            "distance_ratio": (
                0.0
                if success and self.initial_distance <= 1.0e-3
                else (
                    1.0
                    if self.initial_distance <= 1.0e-3
                    else distance / self.initial_distance
                )
            ),
            "progress": progress,
            "local_progress": local_progress,
            "path_cross_track_error": path_cross_track,
            "path_heading_error": path_heading_error,
            "local_subgoal_distance": local_distance,
            "local_subgoal_yaw_error": local_yaw_error,
            "local_subgoal_pose_body": [
                float(observation[0]),
                float(observation[1]),
                local_yaw_error,
            ],
            "success": success,
            "collision": contact_collision,
            "collision_source": (
                "contact" if contact_collision else "none"
            ),
            "collision_contacts": collision_names,
            "timeout": timeout,
            "planner_failed": planner_failed,
            "done": done,
            "step_count": self.step_count,
            "raw_policy_action": raw_action,
            "safe_policy_action": safe_action,
            "teacher_action": teacher_action,
            "teacher_action_mse": teacher_mse,
            "teacher_available": False,
            "teacher_type": "none",
            "shield_intervened": bool(shield_reasons),
            "shield_reasons": shield_reasons,
            "shield_front_blocked": bool(
                "front_emergency_stop" in shield_reasons
            ),
            "shield_front_clearance": clearances["front"],
            "shield_rear_clearance": clearances["rear"],
            "shield_left_clearance": clearances["minimum"],
            "shield_right_clearance": clearances["minimum"],
            "shield_sweep_clearance": clearances["minimum"],
            "minimum_directional_clearance": (
                clearances["front"]
                if safe_action[0] >= 0.0
                else clearances["rear"]
            ),
            "minimum_footprint_clearance": clearances["minimum"],
            "proximity_collision_sector": "none",
            "high_level_replanned": False,
            "high_level_replan_reason": "teb_continuous_replan",
            "high_level_updates": self.step_count,
            "high_level_selection": {
                "reason": "teb_local_path",
                "route": self.goal_labels[self.current_goal_index],
            },
            "scenario_label": self.current_scenario_label,
            "curriculum_stage": self.curriculum_stage,
        }
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }

    def _wait_for_sample(
            self,
            previous_action,
            allow_terminal=False):
        deadline = time.time() + self.sample_wait_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            sample = self._evaluation_sample(previous_action)
            if sample is not None:
                return sample
            if allow_terminal and self._terminal_signal_present():
                return None
            self._publish_normalized_action(
                np.zeros(ACTION_DIM, dtype=np.float32)
            )
            time.sleep(0.05)
        raise RuntimeError(
            "timed out waiting for fresh scan, odometry, TEB path "
            "and reference command"
        )

    def _terminal_signal_present(self):
        status = self.action_client.get_state()
        with self._lock:
            contact_collision = bool(self._episode_collision)
        return bool(
            contact_collision
            or status in (
                GoalStatus.PREEMPTED,
                GoalStatus.SUCCEEDED,
                GoalStatus.ABORTED,
                GoalStatus.REJECTED,
                GoalStatus.RECALLED,
                GoalStatus.LOST,
            )
        )

    def _cancel_active_goal(self):
        status = self.action_client.get_state()
        if status in (
                GoalStatus.PENDING,
                GoalStatus.ACTIVE):
            self.action_client.cancel_goal()

    @staticmethod
    def _parse_scenario_labels(values, goal_count):
        labels = [str(value) for value in values]
        if not labels:
            labels = ["unspecified"] * goal_count
        if len(labels) != goal_count:
            raise ValueError(
                "scenario_labels must contain one label per goal"
            )
        return labels

    def _validate_server_parameters(self):
        positive = {
            "port": self.port,
            "backlog": self.backlog,
            "control_period": self.control_period,
            "max_steps": self.max_steps,
            "success_distance": self.success_distance,
            "sample_wait_timeout": self.sample_wait_timeout,
            "message_timeout": self.message_timeout,
            "scan_clip": self.scan_clip,
            "lookahead_distance": self.lookahead_distance,
            "path_point_spacing": self.path_point_spacing,
            "path_remaining_scale": self.path_remaining_scale,
            "max_linear_speed": self.max_linear_speed,
            "max_yaw_rate": self.max_yaw_rate,
            "emergency_translation_distance": (
                self.emergency_translation_distance
            ),
            "emergency_rotation_distance": (
                self.emergency_rotation_distance
            ),
        }
        invalid = [
            name for name, value in positive.items()
            if float(value) <= 0.0
        ]
        if invalid:
            raise ValueError(
                "TEB PPO parameters must be positive: {}".format(
                    invalid
                )
            )


def _json_safe(value):
    if isinstance(value, dict):
        return dict(
            (str(key), _json_safe(item))
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    rospy.init_node("teb_planar_ppo_environment_server")
    server = TebPlanarPpoEnvironmentServer()
    server.serve_forever()


if __name__ == "__main__":
    main()
