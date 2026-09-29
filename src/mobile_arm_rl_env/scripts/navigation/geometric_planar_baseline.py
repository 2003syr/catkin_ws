#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run the geometry-planner + deterministic-tracker navigation baseline."""

from __future__ import print_function

import math
import os
import sys
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from navigation.local_occupancy_grid import LocalOccupancyGrid
from navigation.regulated_path_tracker import RegulatedPathTracker
from navigation.se2_lattice_planner import SE2LatticePlanner
from training.planar_subgoal_training_env import PlanarSubgoalTrainingEnv
from training.tracked_base_kinematics import TRACKED_FORWARD_YAW_OFFSET


class GeometricPlanarBaseline(object):
    """Evaluate a non-learning navigation stack in the training scene."""

    def __init__(self):
        self.environment = PlanarSubgoalTrainingEnv(
            init_ros_node=False,
            seed=rospy.get_param("~seed", 123),
        )
        self.targets = self._parse_targets(
            rospy.get_param("~targets", [[1.70, -0.13]])
        )
        self.scenario_labels = self._parse_labels(
            rospy.get_param("~scenario_labels", []),
            len(self.targets),
        )
        self.episodes = int(
            rospy.get_param("~episodes", len(self.targets))
        )
        self.maximum_steps = int(
            rospy.get_param("~baseline_max_steps", 1200)
        )
        self.replan_interval = int(
            rospy.get_param("~planner_replan_interval", 50)
        )
        self.maximum_planning_failures = int(
            rospy.get_param("~maximum_planning_failures", 20)
        )
        self.required_success_rate = float(
            rospy.get_param("~required_success_rate", 1.0)
        )
        self.maximum_collisions = int(
            rospy.get_param("~maximum_collisions", 0)
        )
        self.log_interval = int(
            rospy.get_param("~baseline_log_interval", 25)
        )
        self.frame_id = str(
            rospy.get_param("~planning_frame", "world")
        )
        self._validate()

        self.occupancy = LocalOccupancyGrid(
            resolution=rospy.get_param("~map_resolution", 0.05),
            width=rospy.get_param("~map_width", 5.0),
            height=rospy.get_param("~map_height", 5.0),
            max_ray_length=rospy.get_param(
                "~map_max_ray_length", 3.0
            ),
            occupied_increment=rospy.get_param(
                "~map_occupied_increment", 4
            ),
            free_decrement=rospy.get_param(
                "~map_free_decrement", 1
            ),
            occupied_threshold=rospy.get_param(
                "~map_occupied_threshold", 2
            ),
        )
        self.planner = SE2LatticePlanner(
            heading_bins=rospy.get_param(
                "~planner_heading_bins", 16
            ),
            translation_cells=rospy.get_param(
                "~planner_translation_cells", 2
            ),
            allow_reverse=rospy.get_param(
                "~planner_allow_reverse", True
            ),
            reverse_cost=rospy.get_param(
                "~planner_reverse_cost", 2.5
            ),
            rotation_cost_radius=rospy.get_param(
                "~planner_rotation_cost_radius", 0.30
            ),
            goal_tolerance=rospy.get_param(
                "~planner_goal_tolerance", 0.07
            ),
            footprint_half_length=rospy.get_param(
                "~planner_footprint_half_length", 0.56
            ),
            footprint_half_width=rospy.get_param(
                "~planner_footprint_half_width", 0.10
            ),
            safety_margin=rospy.get_param(
                "~planner_safety_margin", 0.08
            ),
            side_detour_offset=rospy.get_param(
                "~planner_side_detour_offset", 0.55
            ),
            side_bias=rospy.get_param(
                "~planner_side_bias", 2.0
            ),
            maximum_expansions=rospy.get_param(
                "~planner_maximum_expansions", 120000
            ),
        )
        self.tracker = RegulatedPathTracker(
            maximum_linear_speed=self.environment.maximum_linear_speed,
            maximum_yaw_rate=self.environment.maximum_yaw_rate,
            position_tolerance=rospy.get_param(
                "~tracker_position_tolerance", 0.07
            ),
            waypoint_tolerance=rospy.get_param(
                "~tracker_waypoint_tolerance", 0.065
            ),
            yaw_tolerance=rospy.get_param(
                "~tracker_yaw_tolerance", 0.16
            ),
            turn_in_place_angle=rospy.get_param(
                "~tracker_turn_in_place_angle", 0.30
            ),
            slow_distance=rospy.get_param(
                "~tracker_slow_distance", 0.35
            ),
            heading_gain=rospy.get_param(
                "~tracker_heading_gain", 1.8
            ),
            path_yaw_gain=rospy.get_param(
                "~tracker_path_yaw_gain", 0.35
            ),
            minimum_linear_fraction=rospy.get_param(
                "~tracker_minimum_linear_fraction", 0.18
            ),
            linear_slew_limit=rospy.get_param(
                "~tracker_linear_slew_limit", 0.10
            ),
            angular_slew_limit=rospy.get_param(
                "~tracker_angular_slew_limit", 0.16
            ),
        )

        self._scan_lock = threading.Lock()
        self._scan_message = None
        self._scan_sequence = 0
        self._scan_subscriber = rospy.Subscriber(
            rospy.get_param("~scan_topic", "/scan"),
            LaserScan,
            self._scan_callback,
            queue_size=1,
        )
        self._map_publisher = rospy.Publisher(
            "~occupancy_grid",
            OccupancyGrid,
            queue_size=1,
            latch=True,
        )
        self._path_publisher = rospy.Publisher(
            "~selected_path",
            Path,
            queue_size=1,
            latch=True,
        )
        self._candidate_publishers = [
            rospy.Publisher(
                "~candidate_path_{}".format(index),
                Path,
                queue_size=1,
                latch=True,
            )
            for index in range(3)
        ]

    def run(self):
        rospy.loginfo(
            "Geometric planar baseline started: episodes=%d "
            "targets=%d full_footprint=(%.3f, %.3f) shield=%s",
            self.episodes,
            len(self.targets),
            self.planner.footprint_half_length,
            self.planner.footprint_half_width,
            self.environment.shield_enabled,
        )
        results = []
        try:
            for episode in range(self.episodes):
                if rospy.is_shutdown():
                    break
                target_index = episode % len(self.targets)
                results.append(self._run_episode(
                    episode + 1,
                    self.targets[target_index],
                    self.scenario_labels[target_index],
                ))
        finally:
            self.environment.stop()

        if not results:
            raise RuntimeError("geometric baseline produced no episodes")
        successes = sum(int(item["success"]) for item in results)
        collisions = sum(int(item["collision"]) for item in results)
        timeouts = sum(int(item["timeout"]) for item in results)
        success_rate = float(successes) / float(len(results))
        mean_steps = float(np.mean([
            item["steps"] for item in results
        ]))
        mean_ratio = float(np.mean([
            item["distance_ratio"] for item in results
        ]))
        gate_pass = bool(
            success_rate >= self.required_success_rate
            and collisions <= self.maximum_collisions
        )
        rospy.loginfo(
            "geometric_baseline episodes=%d successes=%d "
            "success_rate=%.3f collisions=%d timeouts=%d "
            "mean_steps=%.1f mean_distance_ratio=%.3f gate_pass=%s",
            len(results),
            successes,
            success_rate,
            collisions,
            timeouts,
            mean_steps,
            mean_ratio,
            gate_pass,
        )
        if not gate_pass:
            raise RuntimeError(
                "geometric navigation acceptance gate failed"
            )

    def _run_episode(self, episode, target_xy, scenario):
        scan_sequence_before_reset = self._current_scan_sequence()
        target = np.asarray([
            target_xy[0],
            target_xy[1],
            self.environment.target_z,
        ], dtype=np.float64)
        self.environment.reset(target_position=target)
        pose = self._current_pose()
        self.occupancy.reset(pose[0:2])
        self.tracker.reset()
        self._wait_for_fresh_scan(scan_sequence_before_reset)

        start_distance = float(np.linalg.norm(target_xy - pose[0:2]))
        minimum_distance = start_distance
        shield_events = 0
        rotation_overrides = 0
        clearance_translations = 0
        replans = 0
        planning_failures = 0
        committed_side = 0
        selected_candidate = None
        success = False
        collision = False
        timeout = False
        final_info = {}
        executed_steps = 0

        for step in range(self.maximum_steps):
            if rospy.is_shutdown():
                break
            pose = self._current_pose()
            self._fuse_latest_scan(pose)
            should_replan = bool(
                selected_candidate is None
                or step % self.replan_interval == 0
                or self.tracker.path is None
                or self.tracker.path_index >= len(self.tracker.path) - 1
            )
            if should_replan:
                try:
                    candidates = self.planner.plan_candidates(
                        self.occupancy,
                        pose,
                        target_xy,
                    )
                    selected_candidate = self._select_candidate(
                        candidates,
                        committed_side,
                    )
                    if committed_side == 0:
                        committed_side = self._infer_path_side(
                            selected_candidate["path"],
                            pose[0:2],
                            target_xy,
                        )
                    self.tracker.set_path(
                        selected_candidate["path"],
                        current_pose=pose,
                    )
                    replans += 1
                    planning_failures = 0
                    self._publish_map()
                    self._publish_candidates(
                        candidates,
                        selected_candidate,
                    )
                    rospy.loginfo(
                        "geometric_plan episode=%d replan=%d "
                        "selected=%s side=%d candidates=%d "
                        "length=%.3f cost=%.3f expansions=%d",
                        episode,
                        replans,
                        selected_candidate["name"],
                        committed_side,
                        len(candidates),
                        selected_candidate["length"],
                        selected_candidate["cost"],
                        selected_candidate["expansions"],
                    )
                except RuntimeError as error:
                    planning_failures += 1
                    self.environment.base_env.publish_zero_cmd()
                    rospy.logwarn(
                        "geometric_plan episode=%d failure=%d/%d: %s",
                        episode,
                        planning_failures,
                        self.maximum_planning_failures,
                        error,
                    )
                    if planning_failures >= self.maximum_planning_failures:
                        timeout = True
                        break
                    rospy.sleep(0.1)
                    continue

            action, tracker_info = self.tracker.compute_action(
                pose,
                target_xy,
            )
            action, resolver_info = self._resolve_shield_deadlock(
                action,
                tracker_info,
            )
            rotation_overrides += int(
                resolver_info["rotation_overridden"]
            )
            clearance_translations += int(
                resolver_info["clearance_translation"]
            )
            unused_observation, reward, done, info = (
                self.environment.step(action)
            )
            self.tracker.synchronize_executed_action(
                info["safe_policy_action"]
            )
            executed_steps += 1
            final_info = info
            distance = float(info.get(
                "distance",
                np.linalg.norm(target_xy - self._current_pose()[0:2]),
            ))
            minimum_distance = min(minimum_distance, distance)
            shield_events += int(
                info.get("shield_intervened", False)
            )

            if step % self.log_interval == 0 or done:
                rospy.loginfo(
                    "geometric_step episode=%d scenario=%s step=%d "
                    "mode=%s path=%d/%d distance=%.3f min=%.3f "
                    "action=[%.3f, %.3f] safe=[%.3f, %.3f] "
                    "shield=%s reasons=%s resolver=%s",
                    episode,
                    scenario,
                    step + 1,
                    tracker_info["mode"],
                    tracker_info["path_index"],
                    len(self.tracker.path),
                    distance,
                    minimum_distance,
                    action[0],
                    action[1],
                    info["safe_policy_action"][0],
                    info["safe_policy_action"][1],
                    info.get("shield_intervened", False),
                    info.get("shield_reasons", []),
                    resolver_info["reason"],
                )
            if done:
                success = bool(info.get("success", False))
                collision = bool(info.get("collision", False))
                timeout = bool(info.get("timeout", False))
                break
        else:
            timeout = True

        self.environment.base_env.publish_zero_cmd()
        final_pose = self._current_pose()
        final_distance = float(np.linalg.norm(
            target_xy - final_pose[0:2]
        ))
        if final_distance <= self.tracker.position_tolerance and not collision:
            success = True
            timeout = False
        distance_ratio = (
            final_distance / start_distance
            if start_distance > 1.0e-6 else 0.0
        )
        result = {
            "episode": episode,
            "scenario": scenario,
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "steps": int(final_info.get("step_count", executed_steps)),
            "start_distance": start_distance,
            "final_distance": final_distance,
            "minimum_distance": minimum_distance,
            "distance_ratio": distance_ratio,
            "shield_events": shield_events,
            "rotation_overrides": rotation_overrides,
            "clearance_translations": clearance_translations,
            "replans": replans,
            "planning_failures": planning_failures,
        }
        rospy.loginfo(
            "geometric_episode=%02d scenario=%s distance=%.3f->%.3f "
            "minimum=%.3f ratio=%.3f success=%s collision=%s "
            "timeout=%s steps=%d replans=%d planning_failures=%d "
            "shield_events=%d rotation_overrides=%d "
            "clearance_translations=%d",
            episode,
            scenario,
            start_distance,
            final_distance,
            minimum_distance,
            distance_ratio,
            success,
            collision,
            timeout,
            result["steps"],
            replans,
            planning_failures,
            shield_events,
            rotation_overrides,
            clearance_translations,
        )
        return result

    def _resolve_shield_deadlock(self, action, tracker_info):
        """Choose a feasible turn before the emergency shield zeros it."""
        action = np.asarray(action, dtype=np.float32).copy()
        diagnostics = {
            "reason": "none",
            "rotation_overridden": False,
            "clearance_translation": False,
        }
        if (
                abs(float(action[0])) > 0.05
                or abs(float(action[1])) <= 0.05):
            return action, diagnostics

        safe_action, shield_info = self.environment._apply_shield(action)
        if abs(float(safe_action[1])) > 0.05:
            return action, diagnostics
        blocked_reasons = set(shield_info.get("shield_reasons", []))
        directional_stop = bool(
            "left_rotation_stop" in blocked_reasons
            or "right_rotation_stop" in blocked_reasons
        )
        if not directional_stop:
            return action, diagnostics

        opposite = action.copy()
        opposite[1] = -opposite[1]
        opposite_safe, opposite_info = (
            self.environment._apply_shield(opposite)
        )
        if abs(float(opposite_safe[1])) > 0.05:
            diagnostics.update({
                "reason": "rotate_via_clear_side",
                "rotation_overridden": True,
            })
            tracker_info["mode"] = "ROTATE_VIA_CLEAR_SIDE"
            return opposite, diagnostics

        # No in-place rotation direction is currently safe.  Compare a short
        # forward and reverse escape, letting the same shield reject the
        # blocked direction.  This translation is only used to create enough
        # sweep clearance for the next planning/tracking cycle.
        translation_candidates = []
        for linear in (-0.25, 0.25):
            candidate = np.asarray([linear, 0.0], dtype=np.float32)
            candidate_safe, candidate_info = (
                self.environment._apply_shield(candidate)
            )
            translation_candidates.append((
                abs(float(candidate_safe[0])),
                candidate_safe,
                candidate_info,
            ))
        translation_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )
        magnitude, translated_action, translated_info = (
            translation_candidates[0]
        )
        if magnitude > 0.05:
            diagnostics.update({
                "reason": "translate_for_rotation_clearance",
                "clearance_translation": True,
            })
            tracker_info["mode"] = "TRANSLATE_FOR_ROTATION_CLEARANCE"
            return translated_action.astype(np.float32), diagnostics

        diagnostics["reason"] = "rotation_fully_blocked"
        return safe_action.astype(np.float32), diagnostics

    def _select_candidate(self, candidates, committed_side):
        if committed_side != 0:
            matching = [
                candidate for candidate in candidates
                if candidate["side_preference"] == committed_side
            ]
            if matching:
                return min(matching, key=lambda item: item["cost"])
        return min(candidates, key=lambda item: (
            item["cost"],
            item["length"],
        ))

    @staticmethod
    def _infer_path_side(path, start_xy, goal_xy):
        path = np.asarray(path, dtype=np.float64)
        start_xy = np.asarray(start_xy, dtype=np.float64)
        goal_xy = np.asarray(goal_xy, dtype=np.float64)
        direction = goal_xy - start_xy
        length = float(np.linalg.norm(direction))
        if length <= 1.0e-6 or len(path) == 0:
            return 0
        direction /= length
        relative = path[:, 0:2] - start_xy
        lateral = (
            direction[0] * relative[:, 1]
            - direction[1] * relative[:, 0]
        )
        largest = float(lateral[np.argmax(np.abs(lateral))])
        if abs(largest) < 0.12:
            return 0
        return 1 if largest > 0.0 else -1

    def _current_pose(self):
        q, unused_dq = self.environment.base_env.get_ordered_joint_state()
        return np.asarray([
            float(q[0]),
            float(q[1]),
            self._wrap(float(q[2]) + TRACKED_FORWARD_YAW_OFFSET),
        ], dtype=np.float64)

    def _scan_callback(self, message):
        with self._scan_lock:
            self._scan_message = message
            self._scan_sequence += 1

    def _current_scan_sequence(self):
        with self._scan_lock:
            return int(self._scan_sequence)

    def _wait_for_fresh_scan(self, previous_sequence):
        timeout = float(rospy.get_param("~scan_wait_timeout", 10.0))
        deadline = time.time() + timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            if self._current_scan_sequence() > previous_sequence:
                return
            time.sleep(0.02)
        raise RuntimeError("timed out waiting for a fresh LaserScan")

    def _fuse_latest_scan(self, pose):
        with self._scan_lock:
            message = self._scan_message
        if message is None:
            raise RuntimeError("LaserScan is unavailable")
        self.occupancy.update_scan(
            pose,
            message.ranges,
            message.angle_min,
            message.angle_increment,
            message.range_min,
            message.range_max,
        )

    def _publish_map(self):
        message = OccupancyGrid()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.frame_id
        message.info.resolution = self.occupancy.resolution
        message.info.width = self.occupancy.width_cells
        message.info.height = self.occupancy.height_cells
        message.info.origin.position.x = self.occupancy.origin[0]
        message.info.origin.position.y = self.occupancy.origin[1]
        message.info.origin.orientation.w = 1.0
        message.data = (
            self.occupancy.occupancy_message_values()
            .reshape(-1)
            .astype(np.int8)
            .tolist()
        )
        self._map_publisher.publish(message)

    def _publish_candidates(self, candidates, selected):
        for index, publisher in enumerate(self._candidate_publishers):
            if index < len(candidates):
                publisher.publish(self._path_message(
                    candidates[index]["path"]
                ))
            else:
                publisher.publish(self._path_message(
                    np.empty((0, 3), dtype=np.float64)
                ))
        self._path_publisher.publish(
            self._path_message(selected["path"])
        )

    def _path_message(self, path):
        message = Path()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.frame_id
        for row in np.asarray(path, dtype=np.float64):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = row[0]
            pose.pose.position.y = row[1]
            pose.pose.orientation.z = math.sin(0.5 * row[2])
            pose.pose.orientation.w = math.cos(0.5 * row[2])
            message.poses.append(pose)
        return message

    def _validate(self):
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.maximum_steps <= 0:
            raise ValueError("baseline_max_steps must be positive")
        if self.replan_interval <= 0:
            raise ValueError("planner_replan_interval must be positive")
        if self.maximum_planning_failures <= 0:
            raise ValueError(
                "maximum_planning_failures must be positive"
            )
        if not 0.0 <= self.required_success_rate <= 1.0:
            raise ValueError("required_success_rate must be in [0, 1]")
        if self.maximum_collisions < 0:
            raise ValueError("maximum_collisions must be non-negative")
        if self.log_interval <= 0:
            raise ValueError("baseline_log_interval must be positive")

    @staticmethod
    def _parse_targets(value):
        targets = np.asarray(value, dtype=np.float64)
        if (
                targets.ndim != 2
                or targets.shape[0] == 0
                or targets.shape[1] != 2
                or not np.all(np.isfinite(targets))):
            raise ValueError("targets must have shape (N, 2)")
        return [row.copy() for row in targets]

    @staticmethod
    def _parse_labels(value, count):
        labels = [str(item) for item in list(value)]
        if not labels:
            return ["unspecified"] * count
        if len(labels) != count:
            raise ValueError("scenario_labels must match targets")
        return labels

    @staticmethod
    def _wrap(angle):
        return math.atan2(math.sin(angle), math.cos(angle))


def main():
    rospy.init_node("geometric_planar_baseline")
    baseline = GeometricPlanarBaseline()
    rospy.on_shutdown(baseline.environment.stop)
    baseline.run()


if __name__ == "__main__":
    main()
