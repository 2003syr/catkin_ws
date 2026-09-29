#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Closed-loop ROS evaluation for the 62-D TEB path-guided BC policy."""

from __future__ import print_function

import json
import math
import os
import socket
import sys
import threading
import time

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
    TebPlanarDatasetCollector,
)


class PolicyServiceClient(object):
    def __init__(self, host, port, timeout):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self._socket = socket.create_connection(
            (self.host, self.port),
            self.timeout,
        )
        self._socket.settimeout(self.timeout)
        self._reader = self._socket.makefile("rb")
        metadata = self.request({"command": "ping"})
        if int(metadata.get("observation_dim", -1)) != OBSERVATION_DIM:
            raise RuntimeError(
                "policy server observation_dim is not {}".format(
                    OBSERVATION_DIM
                )
            )
        if int(metadata.get("action_dim", -1)) != ACTION_DIM:
            raise RuntimeError(
                "policy server action_dim is not {}".format(ACTION_DIM)
            )
        self.metadata = metadata

    def predict(self, observation):
        response = self.request({
            "command": "predict",
            "observation": np.asarray(
                observation, dtype=np.float32
            ).tolist(),
        })
        action = np.asarray(response["action"], dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise RuntimeError(
                "policy service returned action shape {}".format(
                    action.shape
                )
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError("policy service returned a non-finite action")
        return (
            np.clip(action, -1.0, 1.0),
            float(response.get("inference_ms", 0.0)),
        )

    def request(self, payload):
        encoded = json.dumps(payload, separators=(",", ":")) + "\n"
        self._socket.sendall(encoded.encode("utf-8"))
        line = self._reader.readline()
        if not line:
            raise RuntimeError("policy service closed the connection")
        response = json.loads(line.decode("utf-8"))
        if not bool(response.get("ok", False)):
            raise RuntimeError(
                "policy service error: {}".format(
                    response.get("error", "unknown error")
                )
            )
        return response

    def close(self):
        try:
            self.request({"command": "close"})
        except Exception:
            pass
        try:
            self._reader.close()
        finally:
            self._socket.close()


class TebPlanarBcEvaluator(TebPlanarDatasetCollector):
    """Use TEB's path only as observation; execute the BC [v, omega]."""

    def __init__(self):
        self.episodes = int(rospy.get_param("~episodes", 6))
        self.sample_rate = float(rospy.get_param("~sample_rate", 10.0))
        self.maximum_episode_seconds = float(
            rospy.get_param("~maximum_episode_seconds", 180.0)
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
        self.required_success_rate = float(
            rospy.get_param("~required_success_rate", 1.0)
        )
        self.maximum_collisions = int(
            rospy.get_param("~maximum_collisions", 0)
        )
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
        self.policy_host = str(
            rospy.get_param("~policy_host", "127.0.0.1")
        )
        self.policy_port = int(rospy.get_param("~policy_port", 5559))
        self.policy_timeout = float(
            rospy.get_param("~policy_timeout", 5.0)
        )
        self.goals = self._parse_goals(rospy.get_param(
            "~goals",
            [[1.82, -0.13, 0.5 * math.pi]],
        ))
        self.goal_labels = self._parse_goal_labels(
            rospy.get_param("~goal_labels", []),
            len(self.goals),
        )
        self._validate_evaluation_parameters()

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
        self.policy = PolicyServiceClient(
            self.policy_host,
            self.policy_port,
            self.policy_timeout,
        )

    def run(self):
        rospy.loginfo(
            "TEB BC policy connected checkpoint=%s policy_type=%s "
            "teacher_type=%s",
            str(self.policy.metadata.get("checkpoint", "")),
            str(self.policy.metadata.get("policy_type", "")),
            str(self.policy.metadata.get("teacher_type", "")),
        )
        rospy.loginfo("Waiting for move_base action server")
        if not self.action_client.wait_for_server(
                rospy.Duration(30.0)):
            raise RuntimeError("move_base action server is unavailable")
        self._wait_for_base_sensors()

        results = []
        try:
            for episode_index in range(self.episodes):
                if rospy.is_shutdown():
                    break
                goal_index = episode_index % len(self.goals)
                results.append(self._evaluate_episode(
                    episode_index,
                    goal_index,
                ))
        finally:
            self.action_client.cancel_all_goals()
            self._stop_base()
            self.policy.close()

        if len(results) != self.episodes:
            raise RuntimeError(
                "evaluation stopped after {}/{} episodes".format(
                    len(results),
                    self.episodes,
                )
            )
        successes = sum(int(item["success"]) for item in results)
        collisions = sum(int(item["collision"]) for item in results)
        timeouts = sum(int(item["timeout"]) for item in results)
        total_steps = sum(item["steps"] for item in results)
        total_shields = sum(item["shield_events"] for item in results)
        teacher_squared_error = sum(
            item["teacher_squared_error"] for item in results
        )
        teacher_samples = sum(
            item["teacher_samples"] for item in results
        )
        inference_sum = sum(item["inference_ms_sum"] for item in results)
        inference_samples = sum(
            item["inference_samples"] for item in results
        )
        success_rate = float(successes) / float(self.episodes)
        mean_teacher_mse = (
            teacher_squared_error / float(teacher_samples)
            if teacher_samples > 0
            else float("nan")
        )
        mean_inference_ms = (
            inference_sum / float(inference_samples)
            if inference_samples > 0
            else float("nan")
        )
        rospy.loginfo(
            "teb_bc_evaluation episodes=%d successes=%d "
            "success_rate=%.3f collisions=%d timeouts=%d steps=%d "
            "shield_events=%d shield_rate=%.4f teacher_mse=%.6f "
            "mean_inference_ms=%.3f",
            self.episodes,
            successes,
            success_rate,
            collisions,
            timeouts,
            total_steps,
            total_shields,
            (
                float(total_shields) / float(max(total_steps, 1))
            ),
            mean_teacher_mse,
            mean_inference_ms,
        )
        passed = bool(
            success_rate >= self.required_success_rate
            and collisions <= self.maximum_collisions
        )
        rospy.loginfo("teb_bc_gate_pass=%s", str(passed))
        if not passed:
            raise RuntimeError(
                "TEB BC gate failed: success_rate={:.3f} "
                "required={:.3f} collisions={} maximum={}".format(
                    success_rate,
                    self.required_success_rate,
                    collisions,
                    self.maximum_collisions,
                )
            )

    def _evaluate_episode(self, episode_index, goal_index):
        goal_pose = self.goals[goal_index]
        with self._lock:
            self._episode_collision = False
            self._episode_collision_names = set()
            self._path = None
            self._command = None
            self._receive_times.pop("path", None)
            self._receive_times.pop("command", None)
            starting_odometry = self._odometry
        target_world = self._target_world_position(
            goal_pose,
            starting_odometry,
        )
        start_distance = self._current_target_distance(target_world)
        minimum_distance = start_distance
        previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        steps = 0
        shield_events = 0
        shield_reasons = {}
        teacher_squared_error = 0.0
        teacher_samples = 0
        inference_ms_sum = 0.0
        inference_samples = 0
        timed_out = False
        policy_error = None
        started = time.time()

        self.action_client.send_goal(self._move_base_goal(goal_pose))
        rospy.loginfo(
            "TEB BC evaluation episode=%d route=%s "
            "goal=[%.3f, %.3f, %.3f] start_distance=%.4f",
            episode_index + 1,
            self.goal_labels[goal_index],
            goal_pose[0],
            goal_pose[1],
            goal_pose[2],
            start_distance,
        )
        rate = rospy.Rate(self.sample_rate)
        while not rospy.is_shutdown():
            state = self.action_client.get_state()
            if state not in (
                    GoalStatus.PENDING,
                    GoalStatus.ACTIVE,
                    GoalStatus.PREEMPTING,
                    GoalStatus.RECALLING):
                break
            if time.time() - started > self.maximum_episode_seconds:
                timed_out = True
                self.action_client.cancel_goal()
                break
            with self._lock:
                collision = bool(self._episode_collision)
            if collision:
                self.action_client.cancel_goal()
                break

            sample = self._evaluation_sample(previous_action)
            if sample is None:
                self._publish_normalized_action(
                    np.zeros(ACTION_DIM, dtype=np.float32)
                )
                rate.sleep()
                continue
            observation, teacher_action, scan_message = sample
            try:
                learner_action, inference_ms = self.policy.predict(
                    observation
                )
            except Exception as error:
                policy_error = "{}: {}".format(
                    type(error).__name__,
                    error,
                )
                self.action_client.cancel_goal()
                break
            safe_action, reasons, clearances = self._apply_emergency_shield(
                learner_action,
                scan_message,
            )
            self._publish_normalized_action(safe_action)
            previous_action = safe_action
            steps += 1
            inference_ms_sum += inference_ms
            inference_samples += 1
            teacher_squared_error += float(np.mean(
                np.square(learner_action - teacher_action)
            ))
            teacher_samples += 1
            if reasons:
                shield_events += 1
                for reason in reasons:
                    shield_reasons[reason] = (
                        shield_reasons.get(reason, 0) + 1
                    )
            distance = self._current_target_distance(target_world)
            minimum_distance = min(minimum_distance, distance)
            if steps == 1 or steps % 50 == 0:
                rospy.loginfo(
                    "teb_bc_step episode=%d route=%s step=%d "
                    "distance=%.4f minimum=%.4f learner=%s safe=%s "
                    "teacher=%s clearance=(front:%.3f,rear:%.3f,"
                    "minimum:%.3f) shield=%s",
                    episode_index + 1,
                    self.goal_labels[goal_index],
                    steps,
                    distance,
                    minimum_distance,
                    np.round(learner_action, 4).tolist(),
                    np.round(safe_action, 4).tolist(),
                    np.round(teacher_action, 4).tolist(),
                    clearances["front"],
                    clearances["rear"],
                    clearances["minimum"],
                    str(reasons),
                )
            rate.sleep()

        self._stop_base()
        state = self.action_client.get_state()
        with self._lock:
            collision = bool(self._episode_collision)
            collision_names = sorted(self._episode_collision_names)
        success = bool(
            state == GoalStatus.SUCCEEDED
            and not collision
            and not timed_out
            and policy_error is None
        )
        end_distance = self._current_target_distance(target_world)
        rospy.loginfo(
            "teb_bc_episode=%d route=%s distance=%.4f->%.4f "
            "minimum=%.4f status=%d success=%s collision=%s "
            "contacts=%s timeout=%s steps=%d shield_events=%d "
            "shield_reasons=%s teacher_mse=%.6f "
            "mean_inference_ms=%.3f policy_error=%s",
            episode_index + 1,
            self.goal_labels[goal_index],
            start_distance,
            end_distance,
            minimum_distance,
            state,
            str(success),
            str(collision),
            str(collision_names),
            str(timed_out),
            steps,
            shield_events,
            str(shield_reasons),
            (
                teacher_squared_error / float(teacher_samples)
                if teacher_samples > 0
                else float("nan")
            ),
            (
                inference_ms_sum / float(inference_samples)
                if inference_samples > 0
                else float("nan")
            ),
            str(policy_error),
        )
        if policy_error is not None:
            raise RuntimeError(
                "policy inference failed: {}".format(policy_error)
            )
        return {
            "success": success,
            "collision": collision,
            "timeout": timed_out,
            "steps": steps,
            "shield_events": shield_events,
            "teacher_squared_error": teacher_squared_error,
            "teacher_samples": teacher_samples,
            "inference_ms_sum": inference_ms_sum,
            "inference_samples": inference_samples,
        }

    def _evaluation_sample(self, previous_action):
        wall_time = time.time()
        with self._lock:
            scan = self._scan
            odometry = self._odometry
            path = self._path
            command = self._command
            receive_times = dict(self._receive_times)
        required = ("scan", "odometry", "path", "command")
        if (
                scan is None
                or odometry is None
                or path is None
                or command is None
                or any(
                    wall_time - receive_times.get(name, 0.0)
                    > self.message_timeout
                    for name in required
                )
                or not path.poses):
            return None
        if (
                path.header.frame_id
                and path.header.frame_id != self.goal_frame):
            rospy.logwarn_throttle(
                2.0,
                "Skipping TEB plan in frame '%s'; expected '%s'",
                path.header.frame_id,
                self.goal_frame,
            )
            return None
        observation = self._encode_observation(
            scan,
            odometry,
            path,
            previous_action,
        )
        teacher_action = np.asarray([
            np.clip(
                float(command.linear.x) / self.max_linear_speed,
                -1.0,
                1.0,
            ),
            np.clip(
                float(command.angular.z) / self.max_yaw_rate,
                -1.0,
                1.0,
            ),
        ], dtype=np.float32)
        return observation, teacher_action, scan

    def _apply_emergency_shield(self, action, scan_message):
        safe = np.asarray(action, dtype=np.float32).copy()
        ranges = np.asarray(scan_message.ranges, dtype=np.float64)
        angles = (
            float(scan_message.angle_min)
            + np.arange(ranges.size, dtype=np.float64)
            * float(scan_message.angle_increment)
        )
        valid = np.isfinite(ranges) & (
            ranges >= float(scan_message.range_min)
        )
        if np.isfinite(float(scan_message.range_max)):
            valid &= ranges <= float(scan_message.range_max)
        ranges = np.where(valid, ranges, self.scan_clip)
        ranges = np.clip(ranges, 0.0, self.scan_clip)
        wrapped = np.mod(angles + math.pi, 2.0 * math.pi) - math.pi
        front_mask = np.abs(wrapped) <= self.safety_sector_half_angle
        rear_mask = (
            np.abs(np.abs(wrapped) - math.pi)
            <= self.safety_sector_half_angle
        )
        front = self._masked_minimum(ranges, front_mask)
        rear = self._masked_minimum(ranges, rear_mask)
        minimum = float(np.min(ranges)) if ranges.size else self.scan_clip
        reasons = []
        if (
                safe[0] > 0.0
                and front < self.emergency_translation_distance):
            safe[0] = 0.0
            reasons.append("front_emergency_stop")
        elif (
                safe[0] < 0.0
                and rear < self.emergency_translation_distance):
            safe[0] = 0.0
            reasons.append("rear_emergency_stop")
        if (
                abs(float(safe[1])) > 0.0
                and minimum < self.emergency_rotation_distance):
            safe[1] = 0.0
            reasons.append("rotation_emergency_stop")
        return safe, reasons, {
            "front": front,
            "rear": rear,
            "minimum": minimum,
        }

    @staticmethod
    def _masked_minimum(values, mask):
        selected = values[mask]
        if selected.size == 0:
            return float("inf")
        return float(np.min(selected))

    def _publish_normalized_action(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )
        message = Twist()
        message.linear.x = float(action[0]) * self.max_linear_speed
        message.linear.y = 0.0
        message.angular.z = float(action[1]) * self.max_yaw_rate
        self.command_publisher.publish(message)

    def _stop_base(self):
        zero = np.zeros(ACTION_DIM, dtype=np.float32)
        for unused_index in range(3):
            self._publish_normalized_action(zero)
            rospy.sleep(0.05)

    def _target_world_position(self, goal_pose, odometry):
        if odometry is None:
            return np.asarray(goal_pose[:2], dtype=np.float64)
        robot_pose = self._odometry_pose(odometry)
        odometry_frame = str(odometry.header.frame_id).lstrip("/")
        body_frame = str(odometry.child_frame_id).lstrip("/")
        goal_frame = self.goal_frame.lstrip("/")

        # In this virtual planar model ``base_link`` is intentionally used
        # as the fixed odometry/planning frame and ``base_footprint`` is the
        # moving body frame.  Therefore a goal expressed in base_link is an
        # absolute planar goal, despite the conventional ROS naming.
        if goal_frame == odometry_frame:
            return np.asarray(goal_pose[:2], dtype=np.float64)
        if goal_frame == body_frame or goal_frame == "base_footprint":
            cosine = math.cos(float(robot_pose[2]))
            sine = math.sin(float(robot_pose[2]))
            return np.asarray([
                robot_pose[0]
                + cosine * goal_pose[0]
                - sine * goal_pose[1],
                robot_pose[1]
                + sine * goal_pose[0]
                + cosine * goal_pose[1],
            ], dtype=np.float64)
        return np.asarray(goal_pose[:2], dtype=np.float64)

    def _current_target_distance(self, target_world):
        with self._lock:
            odometry = self._odometry
        if odometry is None:
            return float("nan")
        position = self._odometry_pose(odometry)[:2]
        return float(np.linalg.norm(position - target_world))

    def _validate_evaluation_parameters(self):
        positive = {
            "episodes": self.episodes,
            "sample_rate": self.sample_rate,
            "maximum_episode_seconds": self.maximum_episode_seconds,
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
            "policy_timeout": self.policy_timeout,
        }
        invalid = [
            name for name, value in positive.items()
            if float(value) <= 0.0
        ]
        if invalid:
            raise ValueError(
                "evaluation parameters must be positive: {}".format(
                    invalid
                )
            )
        if not 0.0 <= self.required_success_rate <= 1.0:
            raise ValueError(
                "required_success_rate must be in [0, 1]"
            )
        if self.maximum_collisions < 0:
            raise ValueError("maximum_collisions cannot be negative")


def main():
    rospy.init_node("evaluate_teb_planar_bc")
    evaluator = None
    try:
        evaluator = TebPlanarBcEvaluator()
        evaluator.run()
    finally:
        if evaluator is not None:
            evaluator._stop_base()


if __name__ == "__main__":
    main()
