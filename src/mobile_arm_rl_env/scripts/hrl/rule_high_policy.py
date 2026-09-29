#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rule high-level baseline for the six-dimensional joint subgoal API."""

from __future__ import division

import math

import numpy as np

from hrl.high_level_command import (
    JointHighLevelCommand,
    RouteSide,
    SubgoalType,
)
from hrl.rule_based_planar_subgoal import (
    RuleBasedPlanarSubgoalGenerator,
)


class RuleBasedHighPolicy(object):
    """Generate simultaneous collision-aware base and EE subgoals."""

    def __init__(
            self,
            base_far_step=0.30,
            base_medium_step=0.15,
            ee_far_step=0.05,
            ee_medium_step=0.10,
            ee_near_step=0.15,
            far_distance=0.80,
            near_distance=0.30,
            maximum_yaw_step=0.50 * math.pi,
            planar_generator=None):
        self.base_far_step = float(base_far_step)
        self.base_medium_step = float(base_medium_step)
        self.ee_far_step = float(ee_far_step)
        self.ee_medium_step = float(ee_medium_step)
        self.ee_near_step = float(ee_near_step)
        self.far_distance = float(far_distance)
        self.near_distance = float(near_distance)
        self.maximum_yaw_step = float(maximum_yaw_step)
        self.planar_generator = (
            planar_generator or RuleBasedPlanarSubgoalGenerator(
                minimum_radius=0.15,
                maximum_radius=self.base_far_step,
            )
        )
        self.preferred_turn_sign = 0.0
        self.detour_locked = False
        self.detour_turn_sign = 0.0
        self.detour_lock_updates = 0
        self.detour_progress_direction_world = None
        self.detour_required_progress = None

    def reset(self, preferred_turn_sign=0.0):
        self.preferred_turn_sign = float(np.sign(preferred_turn_sign))
        self.detour_locked = False
        self.detour_turn_sign = 0.0
        self.detour_lock_updates = 0
        self.detour_progress_direction_world = None
        self.detour_required_progress = None

    def predict(self, high_observation):
        if not isinstance(high_observation, dict):
            raise TypeError("rule high policy requires structured observation")
        base_error = self._vector(
            high_observation["final_base_error_body"],
            2,
            "final_base_error_body",
        )
        ee_error = self._vector(
            high_observation["final_ee_error_body"],
            3,
            "final_ee_error_body",
        )
        raw_scan = np.asarray(
            high_observation["scan_bins"],
            dtype=np.float64,
        )
        scan = np.maximum(self._clearance_scan(raw_scan), 0.0)
        base_distance = float(
            high_observation["final_base_distance"]
        )
        ee_distance = float(high_observation["final_ee_distance"])
        base_pose = self._optional_vector(
            high_observation.get("base_pose_world"),
            3,
        )

        if base_distance <= 0.02:
            base_goal_xy = np.zeros(2, dtype=np.float32)
            reason = "terminal_base_hold"
        else:
            desired = self._limit_norm(
                base_error,
                self._base_step(base_distance),
            )
            desired_angle = float(math.atan2(desired[1], desired[0]))
            if (
                    self.detour_locked
                    and self._detour_release_ready(
                        desired,
                        scan,
                        raw_scan,
                        base_pose,
                    )):
                self.detour_locked = False
                self.detour_turn_sign = 0.0
                self.detour_lock_updates = 0
            if (
                    not self.detour_locked
                    and self.planar_generator.corridor_heading_feasible(
                        raw_scan,
                        desired_angle,
                        float(np.linalg.norm(desired)))):
                self.preferred_turn_sign = 0.0
            base_goal_xy, planar_info = self.planar_generator.select(
                desired,
                scan,
                preferred_turn_sign=self.preferred_turn_sign,
                committed_turn_sign=(
                    self.detour_turn_sign if self.detour_locked else 0.0
                ),
                raw_scan_ranges=raw_scan,
            )
            selected_sign = float(planar_info.get("turn_sign", 0.0))
            if (
                    not self.detour_locked
                    and planar_info.get("reason") == "detour_free_sector"
                    and selected_sign != 0.0):
                self.detour_locked = True
                self.detour_turn_sign = selected_sign
                self.detour_lock_updates = 0
                self.preferred_turn_sign = selected_sign
                self._record_detour_obstacle(
                    desired,
                    raw_scan,
                    base_pose,
                )
            elif self.detour_locked:
                self.detour_lock_updates += 1
            reason = str(planar_info.get("reason", "planar_subgoal"))

        if float(np.linalg.norm(base_goal_xy)) > 1.0e-6:
            base_yaw = float(np.clip(
                math.atan2(base_goal_xy[1], base_goal_xy[0]),
                -self.maximum_yaw_step,
                self.maximum_yaw_step,
            ))
        else:
            if self.detour_locked and self.detour_turn_sign != 0.0:
                base_yaw = self.detour_turn_sign * self.maximum_yaw_step
                reason = "detour_turn_in_place"
            else:
                base_yaw = float(np.clip(
                    high_observation["final_heading_error"],
                    -self.maximum_yaw_step,
                    self.maximum_yaw_step,
                ))
        ee_goal = self._limit_norm(
            ee_error,
            self._ee_step(ee_distance),
        )
        if base_distance <= 0.02:
            subgoal_type = SubgoalType.TERMINAL
        elif self.detour_locked or "detour" in str(reason).lower():
            subgoal_type = SubgoalType.DETOUR
        else:
            subgoal_type = SubgoalType.DIRECT
        route_side = (
            RouteSide.from_sign(self.detour_turn_sign)
            if subgoal_type == SubgoalType.DETOUR
            else RouteSide.NONE
        )
        return JointHighLevelCommand(
            base_goal=np.asarray([
                base_goal_xy[0],
                base_goal_xy[1],
                base_yaw,
            ], dtype=np.float32),
            ee_goal=ee_goal,
            reason=reason,
            subgoal_type=subgoal_type,
            route_side=route_side,
        )

    def _clearance_scan(self, scan_ranges):
        """Convert center LiDAR ranges to rectangular-footprint clearance."""
        scan = self._vector(
            scan_ranges,
            self.planar_generator.scan_bin_count,
            "scan_bins",
        )
        centers = self.planar_generator.bin_centers
        cosine = np.abs(np.cos(centers))
        sine = np.abs(np.sin(centers))
        longitudinal = np.full(scan.shape, np.inf, dtype=np.float64)
        lateral = np.full(scan.shape, np.inf, dtype=np.float64)
        longitudinal[cosine > 1.0e-9] = (
            self.planar_generator.footprint_half_length
            / cosine[cosine > 1.0e-9]
        )
        lateral[sine > 1.0e-9] = (
            self.planar_generator.footprint_half_width
            / sine[sine > 1.0e-9]
        )
        return scan - np.minimum(longitudinal, lateral)

    def _record_detour_obstacle(self, desired, raw_scan, base_pose):
        """Freeze the first blocking LiDAR point for detour-lock release."""
        if base_pose is None:
            self.detour_progress_direction_world = None
            self.detour_required_progress = None
            return
        desired = np.asarray(desired, dtype=np.float64)
        distance = float(np.linalg.norm(desired))
        if distance <= 1.0e-9:
            return
        desired_angle = float(math.atan2(desired[1], desired[0]))
        centers = self.planar_generator.bin_centers
        relative_angle = np.arctan2(
            np.sin(centers - desired_angle),
            np.cos(centers - desired_angle),
        )
        raw_scan = np.asarray(raw_scan, dtype=np.float64)
        longitudinal = raw_scan * np.cos(relative_angle)
        lateral = np.abs(raw_scan * np.sin(relative_angle))
        blocking = (
            np.isfinite(raw_scan)
            & (raw_scan > 0.0)
            & (longitudinal > 0.0)
            & (longitudinal <= (
                distance
                + self.planar_generator.footprint_half_length
                + self.planar_generator.clearance_margin
            ))
            & (lateral <= (
                self.planar_generator.footprint_half_width
                + self.planar_generator.clearance_margin
            ))
        )
        candidates = np.flatnonzero(blocking)
        if candidates.size == 0:
            forward = (
                np.isfinite(raw_scan)
                & (raw_scan > 0.0)
                & (longitudinal > 0.0)
                & (np.abs(relative_angle) < 0.5 * math.pi)
            )
            candidates = np.flatnonzero(forward)
        if candidates.size == 0:
            self.detour_progress_direction_world = None
            self.detour_required_progress = None
            return
        index = int(candidates[np.argmin(longitudinal[candidates])])
        obstacle_body = raw_scan[index] * np.asarray([
            math.cos(float(centers[index])),
            math.sin(float(centers[index])),
        ], dtype=np.float64)
        obstacle_world = self._body_to_world(obstacle_body, base_pose)
        direction_world = self._rotate_xy(
            desired / distance,
            float(base_pose[2]),
        )
        self.detour_progress_direction_world = direction_world
        self.detour_required_progress = float(
            np.dot(obstacle_world, direction_world)
            + self.planar_generator.footprint_half_length
        )

    def _detour_release_ready(
            self,
            desired,
            clearance_scan,
            raw_scan,
            base_pose):
        """Release only after the whole chassis has passed the blocker."""
        if base_pose is None or self.detour_lock_updates < 2:
            return False
        if (
                self.detour_progress_direction_world is None
                or self.detour_required_progress is None):
            return False
        current_progress = float(np.dot(
            base_pose[0:2],
            self.detour_progress_direction_world,
        ))
        if current_progress < self.detour_required_progress:
            return False
        distance = float(np.linalg.norm(desired))
        if distance <= 1.0e-9:
            return True
        desired_angle = float(math.atan2(desired[1], desired[0]))
        direct_clear = self.planar_generator.corridor_heading_feasible(
            raw_scan,
            desired_angle,
            distance,
        )
        centers = self.planar_generator.bin_centers
        inside_forward = (
            (centers * self.detour_turn_sign < 0.0)
            & (np.abs(centers) < math.radians(110.0))
        )
        side_clear = bool(
            not np.any(inside_forward)
            or np.min(clearance_scan[inside_forward]) >= 0.25
        )
        return bool(direct_clear and side_clear)

    @staticmethod
    def _body_to_world(point_body, base_pose):
        return np.asarray(base_pose[0:2], dtype=np.float64) + (
            RuleBasedHighPolicy._rotate_xy(point_body, base_pose[2])
        )

    @staticmethod
    def _rotate_xy(vector, angle):
        vector = np.asarray(vector, dtype=np.float64)
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        return np.asarray([
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ], dtype=np.float64)

    @staticmethod
    def _optional_vector(value, size):
        if value is None:
            return None
        return RuleBasedHighPolicy._vector(
            value,
            size,
            "base_pose_world",
        )

    def _base_step(self, distance):
        return (
            self.base_far_step
            if distance > self.far_distance
            else self.base_medium_step
        )

    def _ee_step(self, distance):
        if distance > self.far_distance:
            return self.ee_far_step
        if distance > self.near_distance:
            return self.ee_medium_step
        return self.ee_near_step

    @staticmethod
    def _limit_norm(vector, maximum):
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > float(maximum) and norm > 1.0e-9:
            vector = vector * (float(maximum) / norm)
        return vector.astype(np.float32)

    @staticmethod
    def _vector(value, size, name):
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (size,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name,
                    size,
                    vector.shape,
                )
            )
        return vector


class SafeWaypointHighPolicy(object):
    """Follow the box environment's fixed, footprint-safe world waypoints.

    Unlike the scan policy, this controller never reselects a left/right
    sector from a changing LiDAR frame.  ``FusedBoxDetourTrainingEnv`` owns a
    ``BoxDetourPath`` whose waypoints already include the inflated chassis
    footprint and clearance margin.  The current waypoint is converted back
    to a short body-frame command for one high-level period.
    """

    def __init__(
            self,
            low_environment,
            base_step=0.30,
            ee_step=0.08,
            maximum_yaw_step=0.50 * math.pi,
            arm_release_index=6):
        if low_environment is None:
            raise ValueError("low_environment is required")
        self.low_environment = low_environment
        self.base_step = float(base_step)
        self.ee_step = float(ee_step)
        self.maximum_yaw_step = float(maximum_yaw_step)
        self.arm_release_index = int(arm_release_index)
        if self.base_step <= 0.0 or self.ee_step <= 0.0:
            raise ValueError("safe waypoint steps must be positive")
        if self.maximum_yaw_step <= 0.0:
            raise ValueError("maximum_yaw_step must be positive")

    def reset(self, preferred_turn_sign=0.0):
        del preferred_turn_sign

    def predict(self, high_observation):
        if not isinstance(high_observation, dict):
            raise TypeError("safe waypoint policy requires structured observation")
        path = getattr(self.low_environment, "_path", None)
        if path is None:
            raise RuntimeError("safe waypoint policy requires an active BoxDetourPath")
        base_pose = np.asarray(
            high_observation.get("base_pose_world"),
            dtype=np.float64,
        )
        if base_pose.shape != (3,):
            raise ValueError("base_pose_world must have shape (3,)")
        waypoint = np.asarray(path.current_goal_xy, dtype=np.float64)
        delta_world = waypoint - base_pose[0:2]
        delta_body = self._rotate_xy(delta_world, -base_pose[2])
        base_goal = self._limit_norm(delta_body, self.base_step)

        # At a waypoint let the path/environment advance on the next sensor
        # update instead of making the low layer chase a stale lateral target.
        if np.linalg.norm(delta_world) <= 0.02:
            base_goal[:] = 0.0

        if path.final_waypoint_active or path.index >= len(path.waypoints) - 1:
            base_yaw = float(np.clip(
                high_observation.get("final_heading_error", 0.0),
                -self.maximum_yaw_step,
                self.maximum_yaw_step,
            ))
        elif np.linalg.norm(delta_body) > 0.02:
            base_yaw = float(np.clip(
                math.atan2(delta_body[1], delta_body[0]),
                -self.maximum_yaw_step,
                self.maximum_yaw_step,
            ))
        else:
            base_yaw = 0.0

        # Keep the arm fixed while the chassis crosses the obstacle.  Once
        # the post-box waypoint is active, track the final EE error with a
        # bounded local subgoal.
        if int(path.index) < self.arm_release_index:
            ee_goal = np.zeros(3, dtype=np.float32)
        else:
            ee_goal = self._limit_norm(
                high_observation.get(
                    "final_ee_error_body",
                    np.zeros(3, dtype=np.float32),
                ),
                self.ee_step,
            )
        subgoal_type = self._subgoal_type(path)
        route_side = (
            RouteSide.from_sign(path.side)
            if subgoal_type == SubgoalType.DETOUR
            else RouteSide.NONE
        )
        return JointHighLevelCommand(
            base_goal=np.asarray([
                base_goal[0],
                base_goal[1],
                base_yaw,
            ], dtype=np.float32),
            ee_goal=ee_goal,
            reason="safe_waypoint_{}".format(int(path.index)),
            subgoal_type=subgoal_type,
            route_side=route_side,
        )

    @staticmethod
    def _subgoal_type(path):
        """Classify the active route option without inferring from geometry.

        The first obstacle-route segment is a straight approach, the two
        middle segments own the actual bypass, the post-obstacle recovery is
        direct motion again, and the last waypoint owns precise terminal
        base/arm convergence.  A no-obstacle route stays DIRECT until its
        final pose becomes active.
        """
        index = int(path.index)
        waypoint_count = len(path.waypoints)
        if bool(path.final_waypoint_active) or index >= waypoint_count - 1:
            return SubgoalType.TERMINAL
        if bool(getattr(path, "direct_path", False)):
            return SubgoalType.DIRECT
        # The six-slot box route is
        #   1: clear straight approach
        #   2: move to the selected obstacle side
        #   3: cross the obstacle longitudinally
        #   4: clear straight recovery toward the terminal corridor
        #   5: precise terminal base pose (then arm finish at index 6)
        # Express this relative to the route length so the classification
        # remains correct if another route keeps the same semantic layout.
        if 2 <= index < waypoint_count - 2:
            return SubgoalType.DETOUR
        return SubgoalType.DIRECT

    @staticmethod
    def _rotate_xy(vector, angle):
        vector = np.asarray(vector, dtype=np.float64)
        cosine = math.cos(float(angle))
        sine = math.sin(float(angle))
        return np.asarray([
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ], dtype=np.float64)

    @staticmethod
    def _limit_norm(vector, maximum):
        vector = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm > float(maximum) and norm > 1.0e-9:
            vector = vector * (float(maximum) / norm)
        return vector.astype(np.float32)
