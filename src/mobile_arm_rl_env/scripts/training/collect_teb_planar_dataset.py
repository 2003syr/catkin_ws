#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Collect successful TEB local-plan demonstrations for planar BC."""

from __future__ import print_function

import math
import os
import threading
import time

import actionlib
import numpy as np
import rospy

from actionlib_msgs.msg import GoalStatus
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion


OBSERVATION_DIM = 62
ACTION_DIM = 2
SCAN_BINS = 36
PATH_POINTS = 5


class TebPlanarDatasetCollector(object):
    """Pair observable TEB path previews with TEB's final [v, omega]."""

    def __init__(self):
        self.output_path = os.path.abspath(os.path.expanduser(
            rospy.get_param(
                "~output_path",
                "/tmp/mobile_arm_rl_training/teb_planar_teacher.npz",
            )
        ))
        self.episodes = int(rospy.get_param("~episodes", 10))
        self.max_attempts = int(rospy.get_param("~max_attempts", 20))
        self.minimum_episode_samples = int(
            rospy.get_param("~minimum_episode_samples", 20)
        )
        self.minimum_successes_per_goal = int(
            rospy.get_param("~minimum_successes_per_goal", 1)
        )
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
        self.cmd_vel_topic = str(
            rospy.get_param("~cmd_vel_topic", "/teb_cmd_vel")
        )
        self.goals = self._parse_goals(rospy.get_param(
            "~goals",
            [
                [1.82, -0.13, 0.5 * math.pi],
                [0.00, 0.00, 0.0],
            ],
        ))
        self.goal_labels = self._parse_goal_labels(
            rospy.get_param("~goal_labels", []),
            len(self.goals),
        )
        self._validate_parameters()

        self._lock = threading.Lock()
        self._scan = None
        self._odometry = None
        self._path = None
        self._command = None
        self._receive_times = {}
        self._episode_collision = False
        self._episode_collision_names = set()

        self.accepted_observations = []
        self.accepted_actions = []
        self.accepted_episode_ids = []
        self.accepted_goal_indices = []
        self.accepted_goal_labels = []
        self.accepted_goal_poses = []
        self.successful_episodes = 0
        self.attempted_episodes = 0
        self.collisions = 0
        self.timeouts = 0
        self.failed_goals = 0
        self.goal_success_counts = np.zeros(
            len(self.goals), dtype=np.int32
        )

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
            self.cmd_vel_topic,
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
        self.action_client = actionlib.SimpleActionClient(
            "/move_base", MoveBaseAction
        )

    def run(self):
        rospy.loginfo("Waiting for move_base action server")
        if not self.action_client.wait_for_server(
                rospy.Duration(30.0)):
            raise RuntimeError("move_base action server is unavailable")
        self._wait_for_base_sensors()
        goal_cursor = 0
        try:
            while (
                    not rospy.is_shutdown()
                    and not self._collection_complete()
                    and self.attempted_episodes < self.max_attempts):
                goal_index = goal_cursor % len(self.goals)
                goal_cursor += 1
                self._collect_attempt(goal_index)
        finally:
            self.action_client.cancel_all_goals()
            self._save_dataset()

        if not self._collection_complete():
            raise RuntimeError(
                "collected only {}/{} successful TEB episodes "
                "after {} attempts; per-goal successes={} required={}".format(
                    self.successful_episodes,
                    self.episodes,
                    self.attempted_episodes,
                    self.goal_success_counts.tolist(),
                    self.minimum_successes_per_goal,
                )
            )
        rospy.loginfo(
            "TEB dataset complete: %s episodes=%d samples=%d",
            self.output_path,
            self.successful_episodes,
            len(self.accepted_observations),
        )

    def _collect_attempt(self, goal_index):
        goal_pose = self.goals[goal_index]
        self.attempted_episodes += 1
        with self._lock:
            self._episode_collision = False
            self._episode_collision_names = set()
            self._path = None
            self._command = None
            self._receive_times.pop("path", None)
            self._receive_times.pop("command", None)

        self.action_client.send_goal(self._move_base_goal(goal_pose))
        rospy.loginfo(
            "TEB collection attempt=%d goal_index=%d route=%s "
            "goal=[%.3f, %.3f, %.3f]",
            self.attempted_episodes,
            goal_index,
            self.goal_labels[goal_index],
            goal_pose[0],
            goal_pose[1],
            goal_pose[2],
        )

        episode_observations = []
        episode_actions = []
        previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        started = time.time()
        rate = rospy.Rate(self.sample_rate)
        timed_out = False
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
            sample = self._sample(previous_action)
            if sample is not None:
                observation, action = sample
                episode_observations.append(observation)
                episode_actions.append(action)
                previous_action = action
            rate.sleep()

        state = self.action_client.get_state()
        if timed_out:
            self.timeouts += 1
        with self._lock:
            collision = bool(self._episode_collision)
            collision_names = sorted(self._episode_collision_names)
        success = bool(
            state == GoalStatus.SUCCEEDED
            and not collision
            and len(episode_observations) >= self.minimum_episode_samples
        )
        if collision:
            self.collisions += 1
        if not timed_out and state != GoalStatus.SUCCEEDED:
            self.failed_goals += 1

        if success:
            self.successful_episodes += 1
            episode_id = self.successful_episodes
            self.accepted_observations.extend(episode_observations)
            self.accepted_actions.extend(episode_actions)
            self.accepted_episode_ids.extend(
                [episode_id] * len(episode_observations)
            )
            self.accepted_goal_indices.append(goal_index)
            self.accepted_goal_labels.append(
                self.goal_labels[goal_index]
            )
            self.accepted_goal_poses.append(goal_pose.copy())
            self.goal_success_counts[goal_index] += 1
            self._save_dataset()
        rospy.loginfo(
            "TEB collection result attempt=%d accepted=%d/%d "
            "status=%d success=%s samples=%d collision=%s "
            "contacts=%s timeouts=%d",
            self.attempted_episodes,
            self.successful_episodes,
            self.episodes,
            state,
            str(success),
            len(episode_observations),
            str(collision),
            str(collision_names),
            self.timeouts,
        )

    def _sample(self, previous_action):
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
                )):
            return None
        if not path.poses:
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
        action = np.asarray([
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
        return observation, action

    def _encode_observation(
            self,
            scan_message,
            odometry,
            path_message,
            previous_action):
        robot_pose = self._odometry_pose(odometry)
        path_world = np.asarray([
            [pose.pose.position.x, pose.pose.position.y]
            for pose in path_message.poses
        ], dtype=np.float64)
        path_body = self._world_points_to_body(path_world, robot_pose)
        path_body = self._remove_duplicate_points(path_body)
        cumulative = self._cumulative_lengths(path_body)
        total_length = float(cumulative[-1])

        lookahead_distance = min(
            self.lookahead_distance,
            total_length,
        )
        local_subgoal = self._interpolate_path(
            path_body,
            cumulative,
            lookahead_distance,
        )
        local_yaw_error = self._path_heading_at_distance(
            path_body,
            cumulative,
            lookahead_distance,
        )

        preview = np.zeros((PATH_POINTS, 2), dtype=np.float32)
        path_mask = np.zeros(PATH_POINTS, dtype=np.float32)
        for index in range(PATH_POINTS):
            distance = (index + 1) * self.path_point_spacing
            if distance <= total_length + 1.0e-6:
                preview[index] = self._interpolate_path(
                    path_body,
                    cumulative,
                    min(distance, total_length),
                )
                path_mask[index] = 1.0
        if not np.any(path_mask):
            preview[0] = path_body[-1]
            path_mask[0] = 1.0

        first_valid = int(np.flatnonzero(path_mask)[0])
        preview_heading = math.atan2(
            float(preview[first_valid, 1]),
            float(preview[first_valid, 0]),
        )
        cross_track = self._distance_to_polyline(
            np.zeros(2, dtype=np.float64),
            path_body,
        )
        path_metrics = np.asarray([
            np.clip(cross_track / self.lookahead_distance, 0.0, 1.0),
            np.clip(preview_heading / math.pi, -1.0, 1.0),
            np.clip(
                total_length / self.path_remaining_scale,
                0.0,
                1.0,
            ),
        ], dtype=np.float32)
        velocity = np.asarray([
            np.clip(
                float(odometry.twist.twist.linear.x)
                / self.max_linear_speed,
                -1.0,
                1.0,
            ),
            np.clip(
                float(odometry.twist.twist.angular.z)
                / self.max_yaw_rate,
                -1.0,
                1.0,
            ),
        ], dtype=np.float32)
        observation = np.concatenate((
            np.clip(
                local_subgoal,
                -self.lookahead_distance,
                self.lookahead_distance,
            ).astype(np.float32),
            np.asarray([
                math.sin(local_yaw_error),
                math.cos(local_yaw_error),
            ], dtype=np.float32),
            velocity,
            self._resample_scan(scan_message),
            np.clip(
                preview.reshape(-1),
                -self.lookahead_distance,
                self.lookahead_distance,
            ).astype(np.float32),
            path_metrics,
            np.asarray(previous_action, dtype=np.float32),
            path_mask,
        ))
        if observation.shape != (OBSERVATION_DIM,):
            raise RuntimeError(
                "TEB observation has shape {}, expected ({},)".format(
                    observation.shape,
                    OBSERVATION_DIM,
                )
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("TEB observation contains non-finite values")
        return observation.astype(np.float32)

    def _resample_scan(self, message):
        ranges = np.asarray(message.ranges, dtype=np.float64)
        if ranges.size == 0:
            raise RuntimeError("received an empty LaserScan")
        angles = (
            float(message.angle_min)
            + np.arange(ranges.size, dtype=np.float64)
            * float(message.angle_increment)
        )
        ranges[~np.isfinite(ranges)] = self.scan_clip
        ranges = np.clip(ranges, 0.0, self.scan_clip)
        wrapped = np.mod(angles + math.pi, 2.0 * math.pi) - math.pi
        order = np.argsort(wrapped)
        wrapped = wrapped[order]
        ranges = ranges[order]
        wrapped = np.concatenate((
            wrapped - 2.0 * math.pi,
            wrapped,
            wrapped + 2.0 * math.pi,
        ))
        ranges = np.tile(ranges, 3)
        centers = np.linspace(
            -math.pi,
            math.pi,
            SCAN_BINS,
            endpoint=False,
        )
        return (
            np.interp(centers, wrapped, ranges) / self.scan_clip
        ).astype(np.float32)

    def _wait_for_base_sensors(self):
        deadline = time.time() + 30.0
        while not rospy.is_shutdown() and time.time() < deadline:
            with self._lock:
                ready = self._scan is not None and self._odometry is not None
            if ready:
                return
            time.sleep(0.05)
        raise RuntimeError("timed out waiting for /scan and /odom")

    def _save_dataset(self):
        directory = os.path.dirname(self.output_path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        observations = np.asarray(
            self.accepted_observations,
            dtype=np.float32,
        ).reshape((-1, OBSERVATION_DIM))
        actions = np.asarray(
            self.accepted_actions,
            dtype=np.float32,
        ).reshape((-1, ACTION_DIM))
        np.savez_compressed(
            self.output_path,
            observations=observations,
            teacher_actions=actions,
            episode_ids=np.asarray(
                self.accepted_episode_ids, dtype=np.int32
            ),
            accepted_goal_indices=np.asarray(
                self.accepted_goal_indices, dtype=np.int32
            ),
            accepted_goal_labels=np.asarray(
                self.accepted_goal_labels, dtype="S64"
            ),
            accepted_goal_poses=np.asarray(
                self.accepted_goal_poses, dtype=np.float32
            ).reshape((-1, 3)),
            configured_goals=np.asarray(self.goals, dtype=np.float32),
            configured_goal_labels=np.asarray(
                self.goal_labels, dtype="S64"
            ),
            goal_success_counts=self.goal_success_counts.copy(),
            minimum_successes_per_goal=np.asarray(
                [self.minimum_successes_per_goal], dtype=np.int32
            ),
            observation_dim=np.asarray(
                [OBSERVATION_DIM], dtype=np.int32
            ),
            action_dim=np.asarray([ACTION_DIM], dtype=np.int32),
            successful_episodes=np.asarray(
                [self.successful_episodes], dtype=np.int32
            ),
            attempted_episodes=np.asarray(
                [self.attempted_episodes], dtype=np.int32
            ),
            collision_episodes=np.asarray(
                [self.collisions], dtype=np.int32
            ),
            timeout_episodes=np.asarray(
                [self.timeouts], dtype=np.int32
            ),
            failed_goal_episodes=np.asarray(
                [self.failed_goals], dtype=np.int32
            ),
            teacher_type=np.asarray(["teb_local_plan_cmd_vel"]),
            observation_semantics=np.asarray([
                "teb_local_subgoal_xy2,teb_subgoal_yaw_sin_cos2,"
                "body_vw2,scan36,teb_path_preview_xy10,"
                "path_metrics3,previous_teacher_action2,path_mask5"
            ]),
            action_semantics=np.asarray([
                "normalized_teb_linear_velocity_yaw_rate"
            ]),
            local_plan_topic=np.asarray([self.local_plan_topic]),
            command_topic=np.asarray([self.cmd_vel_topic]),
            goal_frame=np.asarray([self.goal_frame]),
            max_linear_speed=np.asarray(
                [self.max_linear_speed], dtype=np.float32
            ),
            max_yaw_rate=np.asarray(
                [self.max_yaw_rate], dtype=np.float32
            ),
            lookahead_distance=np.asarray(
                [self.lookahead_distance], dtype=np.float32
            ),
            path_point_spacing=np.asarray(
                [self.path_point_spacing], dtype=np.float32
            ),
        )

    def _scan_callback(self, message):
        with self._lock:
            self._scan = message
            self._receive_times["scan"] = time.time()

    def _odometry_callback(self, message):
        with self._lock:
            self._odometry = message
            self._receive_times["odometry"] = time.time()

    def _path_callback(self, message):
        with self._lock:
            self._path = message
            self._receive_times["path"] = time.time()

    def _command_callback(self, message):
        with self._lock:
            self._command = message
            self._receive_times["command"] = time.time()

    def _contacts_callback(self, message):
        contacts = set()
        for state in message.states:
            names = (
                str(state.collision1_name),
                str(state.collision2_name),
            )
            if any(self._is_ground(name) for name in names):
                continue
            contacts.update(name for name in names if name)
        if contacts:
            with self._lock:
                self._episode_collision = True
                self._episode_collision_names.update(contacts)

    def _move_base_goal(self, pose):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.goal_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = float(pose[0])
        goal.target_pose.pose.position.y = float(pose[1])
        goal.target_pose.pose.orientation.z = math.sin(
            0.5 * float(pose[2])
        )
        goal.target_pose.pose.orientation.w = math.cos(
            0.5 * float(pose[2])
        )
        return goal

    @staticmethod
    def _odometry_pose(message):
        quaternion = message.pose.pose.orientation
        yaw = euler_from_quaternion([
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        ])[2]
        return np.asarray([
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw,
        ], dtype=np.float64)

    @staticmethod
    def _world_points_to_body(points, robot_pose):
        delta = points - robot_pose[0:2]
        cosine = math.cos(float(robot_pose[2]))
        sine = math.sin(float(robot_pose[2]))
        rotation = np.asarray([
            [cosine, sine],
            [-sine, cosine],
        ], dtype=np.float64)
        return np.dot(delta, rotation.T)

    @staticmethod
    def _remove_duplicate_points(points):
        if points.shape[0] == 1:
            return np.vstack((
                np.zeros((1, 2), dtype=np.float64),
                points,
            ))
        keep = [0]
        for index in range(1, points.shape[0]):
            if np.linalg.norm(points[index] - points[keep[-1]]) > 1.0e-5:
                keep.append(index)
        filtered = points[keep]
        if filtered.shape[0] == 1:
            filtered = np.vstack((
                np.zeros((1, 2), dtype=np.float64),
                filtered,
            ))
        return filtered

    @staticmethod
    def _cumulative_lengths(points):
        segment_lengths = np.linalg.norm(
            np.diff(points, axis=0),
            axis=1,
        )
        cumulative = np.concatenate((
            np.zeros(1, dtype=np.float64),
            np.cumsum(segment_lengths),
        ))
        if cumulative[-1] <= 1.0e-6:
            cumulative[-1] = 1.0e-6
        return cumulative

    @staticmethod
    def _interpolate_path(points, cumulative, distance):
        distance = float(np.clip(distance, 0.0, cumulative[-1]))
        upper = int(np.searchsorted(cumulative, distance, side="right"))
        upper = min(max(upper, 1), points.shape[0] - 1)
        lower = upper - 1
        segment_length = cumulative[upper] - cumulative[lower]
        if segment_length <= 1.0e-8:
            return points[upper].copy()
        fraction = (distance - cumulative[lower]) / segment_length
        return (
            (1.0 - fraction) * points[lower]
            + fraction * points[upper]
        )

    @staticmethod
    def _path_heading_at_distance(points, cumulative, distance):
        distance = float(np.clip(distance, 0.0, cumulative[-1]))
        upper = int(np.searchsorted(cumulative, distance, side="right"))
        upper = min(max(upper, 1), points.shape[0] - 1)
        lower = upper - 1
        direction = points[upper] - points[lower]
        return math.atan2(float(direction[1]), float(direction[0]))

    @classmethod
    def _distance_to_polyline(cls, point, polyline):
        return min(
            cls._point_segment_distance(
                point,
                polyline[index],
                polyline[index + 1],
            )
            for index in range(polyline.shape[0] - 1)
        )

    @staticmethod
    def _point_segment_distance(point, start, end):
        segment = end - start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1.0e-12:
            return float(np.linalg.norm(point - start))
        fraction = float(np.clip(
            np.dot(point - start, segment) / denominator,
            0.0,
            1.0,
        ))
        projection = start + fraction * segment
        return float(np.linalg.norm(point - projection))

    @staticmethod
    def _is_ground(name):
        lowered = str(name).lower()
        return "ground_plane" in lowered or "::ground" in lowered

    @staticmethod
    def _parse_goals(values):
        goals = []
        for index, value in enumerate(values):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (3,) or not np.all(np.isfinite(array)):
                raise ValueError(
                    "goal {} must contain finite [x, y, yaw]".format(
                        index
                    )
                )
            goals.append(array)
        if not goals:
            raise ValueError("at least one collection goal is required")
        return goals

    @staticmethod
    def _parse_goal_labels(values, goal_count):
        labels = [str(value) for value in values]
        if not labels:
            labels = [
                "goal_{:02d}".format(index)
                for index in range(goal_count)
            ]
        if len(labels) != goal_count:
            raise ValueError(
                "goal_labels must contain one label per goal"
            )
        if len(set(labels)) != len(labels):
            raise ValueError("goal_labels must be unique")
        return labels

    def _validate_parameters(self):
        positive_values = {
            "episodes": self.episodes,
            "max_attempts": self.max_attempts,
            "minimum_episode_samples": self.minimum_episode_samples,
            "minimum_successes_per_goal": (
                self.minimum_successes_per_goal
            ),
            "sample_rate": self.sample_rate,
            "maximum_episode_seconds": self.maximum_episode_seconds,
            "message_timeout": self.message_timeout,
            "scan_clip": self.scan_clip,
            "lookahead_distance": self.lookahead_distance,
            "path_point_spacing": self.path_point_spacing,
            "path_remaining_scale": self.path_remaining_scale,
            "max_linear_speed": self.max_linear_speed,
            "max_yaw_rate": self.max_yaw_rate,
        }
        invalid = [
            name for name, value in positive_values.items()
            if float(value) <= 0.0
        ]
        if invalid:
            raise ValueError(
                "collector parameters must be positive: {}".format(
                    invalid
                )
            )
        if self.max_attempts < self.episodes:
            raise ValueError("max_attempts must be >= episodes")
        required_episodes = (
            self.minimum_successes_per_goal * len(self.goals)
        )
        if self.episodes < required_episodes:
            raise ValueError(
                "episodes must be at least {} to provide {} successes "
                "for each of {} goals".format(
                    required_episodes,
                    self.minimum_successes_per_goal,
                    len(self.goals),
                )
            )

    def _collection_complete(self):
        return bool(
            self.successful_episodes >= self.episodes
            and np.all(
                self.goal_success_counts
                >= self.minimum_successes_per_goal
            )
        )


def main():
    rospy.init_node("collect_teb_planar_dataset")
    TebPlanarDatasetCollector().run()


if __name__ == "__main__":
    main()
