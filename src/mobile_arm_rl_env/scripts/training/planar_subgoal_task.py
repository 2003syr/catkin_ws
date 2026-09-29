#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pose-and-path-guided local-navigation state and reward logic."""

from __future__ import print_function

import math

import numpy as np

from hrl.rule_based_planar_subgoal import (
    RuleBasedPlanarSubgoalGenerator,
)


class PlanarSubgoalTask(object):
    """Expose a 62-D pose-guided low-level control problem.

    A rule-based high level is updated at a slower rate and chooses a short,
    LiDAR-feasible subgoal.  The low-level PPO sees that subgoal and a preview
    of the resulting local path, but not the final task goal.

    Observation layout:

        0:2    local subgoal position in the body frame [m]
        2:4    sin/cos of local subgoal yaw error
        4:6    measured forward velocity and yaw rate, normalized
        6:42   36 fixed-angle LiDAR bins, normalized
        42:52  five body-frame path-preview points [m]
        52:55  cross-track error, heading error, path remaining
        55:57  previous executed normalized [v, omega] action
        57:62  path-point validity mask
    """

    SCAN_DIM = 36
    PATH_POINT_COUNT = 5
    PATH_POINT_DIM = 2
    CONTINUOUS_DIM = 57
    OBS_DIM = 62
    ACTION_DIM = 2

    def __init__(
            self,
            success_threshold=0.08,
            max_steps=600,
            progress_reward_scale=4.0,
            path_progress_reward_scale=8.0,
            waypoint_reward=0.25,
            success_reward=5.0,
            collision_penalty=5.0,
            timeout_penalty=2.0,
            time_penalty=0.005,
            action_penalty_scale=0.002,
            smoothness_penalty_scale=0.01,
            reverse_penalty_scale=0.01,
            idle_penalty=0.01,
            idle_action_threshold=0.05,
            cross_track_penalty_scale=0.01,
            heading_penalty_scale=0.002,
            clearance_penalty_scale=0.05,
            preferred_clearance=0.25,
            collision_distance=0.18,
            motion_clearance_half_angle=0.610865,
            scan_clip=10.0,
            target_clip=2.0,
            footprint_half_length=0.56,
            footprint_half_width=0.10,
            base_velocity_scale=(0.05, 0.125),
            high_level_interval=10,
            local_subgoal_tolerance=0.10,
            local_subgoal_yaw_tolerance=0.17,
            local_subgoal_min_radius=0.20,
            local_subgoal_max_radius=0.50,
            local_subgoal_clearance_margin=0.18,
            path_internal_points=10,
            path_replan_cross_track=0.35,
            detour_release_cycles=3,
            detour_min_lock_updates=4,
            detour_release_rear_angle=1.919862,
            detour_release_side_clearance=0.25,
            detour_nearby_clearance=0.80,
            detour_escape_distance=0.30,
            detour_escape_clearance=0.28,
            detour_escape_max_distance=0.55):
        self.success_threshold = float(success_threshold)
        self.max_steps = int(max_steps)
        self.progress_reward_scale = float(progress_reward_scale)
        self.path_progress_reward_scale = float(
            path_progress_reward_scale
        )
        self.waypoint_reward = float(waypoint_reward)
        self.success_reward = float(success_reward)
        self.collision_penalty = float(collision_penalty)
        self.timeout_penalty = float(timeout_penalty)
        self.time_penalty = float(time_penalty)
        self.action_penalty_scale = float(action_penalty_scale)
        self.smoothness_penalty_scale = float(
            smoothness_penalty_scale
        )
        self.reverse_penalty_scale = float(reverse_penalty_scale)
        self.idle_penalty = float(idle_penalty)
        self.idle_action_threshold = float(idle_action_threshold)
        self.cross_track_penalty_scale = float(
            cross_track_penalty_scale
        )
        self.heading_penalty_scale = float(heading_penalty_scale)
        self.clearance_penalty_scale = float(
            clearance_penalty_scale
        )
        self.preferred_clearance = float(preferred_clearance)
        self.collision_distance = float(collision_distance)
        self.motion_clearance_half_angle = float(
            motion_clearance_half_angle
        )
        self.scan_clip = float(scan_clip)
        self.target_clip = float(target_clip)
        self.footprint_half_length = float(footprint_half_length)
        self.footprint_half_width = float(footprint_half_width)
        self.base_velocity_scale = self._vector(
            base_velocity_scale,
            self.ACTION_DIM,
            "base_velocity_scale",
        )
        self.high_level_interval = int(high_level_interval)
        self.local_subgoal_tolerance = float(
            local_subgoal_tolerance
        )
        self.local_subgoal_yaw_tolerance = float(
            local_subgoal_yaw_tolerance
        )
        self.local_subgoal_max_radius = float(
            local_subgoal_max_radius
        )
        self.path_internal_points = int(path_internal_points)
        self.path_replan_cross_track = float(
            path_replan_cross_track
        )
        self.detour_release_cycles = int(detour_release_cycles)
        self.detour_min_lock_updates = int(detour_min_lock_updates)
        self.detour_release_rear_angle = float(
            detour_release_rear_angle
        )
        self.detour_release_side_clearance = float(
            detour_release_side_clearance
        )
        self.detour_nearby_clearance = float(
            detour_nearby_clearance
        )
        self.detour_escape_distance = float(detour_escape_distance)
        self.detour_escape_clearance = float(
            detour_escape_clearance
        )
        self.detour_escape_max_distance = float(
            detour_escape_max_distance
        )
        self.high_level = RuleBasedPlanarSubgoalGenerator(
            scan_bin_count=self.SCAN_DIM,
            minimum_radius=local_subgoal_min_radius,
            maximum_radius=local_subgoal_max_radius,
            clearance_margin=local_subgoal_clearance_margin,
            footprint_half_length=self.footprint_half_length,
            footprint_half_width=self.footprint_half_width,
        )
        self._validate_parameters()

        self.previous_action = np.zeros(
            self.ACTION_DIM,
            dtype=np.float32,
        )
        self.previous_distance = None
        self.previous_local_distance = None
        self.initial_distance = None
        self.minimum_distance = None
        self.local_subgoal_world = None
        self.local_subgoal_yaw_world = None
        self.path_start_world = None
        self.path_points_world = None
        self.preferred_turn_sign = 0.0
        self.detour_locked = False
        self.detour_turn_sign = 0.0
        self.detour_clear_cycles = 0
        self.detour_lock_updates = 0
        self.detour_obstacle_cleared = False
        self.episode_start_world = None
        self.episode_goal_world = None
        self.episode_progress_direction = None
        self.detour_obstacle_world = None
        self.detour_obstacle_progress = None
        self.detour_required_progress = None
        self.detour_escape_active = False
        self.detour_escape_start_world = None
        self.detour_escape_direction_world = None
        self.detour_escape_goal_world = None
        self.detour_escape_updates = 0
        self.detour_escape_clear_cycles = 0
        self.steps_since_high_level = 0
        self.high_level_updates = 0
        self.high_level_info = {}
        self.step_count = 0
        self.last_diagnostics = {}

    def reset(self, observation):
        self.previous_action.fill(0.0)
        self.local_subgoal_world = None
        self.local_subgoal_yaw_world = None
        self.step_count = 0
        self.steps_since_high_level = 0
        self.high_level_updates = 0
        self.preferred_turn_sign = 0.0
        self.detour_locked = False
        self.detour_turn_sign = 0.0
        self.detour_clear_cycles = 0
        self.detour_lock_updates = 0
        self.detour_obstacle_cleared = False
        self.detour_obstacle_world = None
        self.detour_obstacle_progress = None
        self.detour_required_progress = None
        self.detour_escape_active = False
        self.detour_escape_start_world = None
        self.detour_escape_direction_world = None
        self.detour_escape_goal_world = None
        self.detour_escape_updates = 0
        self.detour_escape_clear_cycles = 0
        displacement = self._final_goal_xy(observation)
        base_pose = self._base_pose(observation)
        self.episode_start_world = base_pose[0:2].copy()
        self.episode_goal_world = self._body_to_world(
            displacement,
            base_pose,
        )
        episode_delta = (
            self.episode_goal_world - self.episode_start_world
        )
        episode_length = float(np.linalg.norm(episode_delta))
        if episode_length <= 1.0e-6:
            self.episode_progress_direction = np.asarray(
                [math.cos(base_pose[2]), math.sin(base_pose[2])],
                dtype=np.float64,
            )
        else:
            self.episode_progress_direction = (
                episode_delta / episode_length
            )
        self.previous_distance = float(np.linalg.norm(displacement))
        self.initial_distance = self.previous_distance
        self.minimum_distance = self.previous_distance
        self._update_high_level(observation, reason="episode_reset")
        self.last_diagnostics = {
            "initial_distance": self.initial_distance,
            "distance": self.previous_distance,
            "minimum_distance": self.minimum_distance,
            "distance_reduction": 0.0,
            "distance_ratio": 1.0,
            "progress": 0.0,
            "local_progress": 0.0,
            "success": False,
            "collision": False,
            "timeout": False,
            "done": False,
            "step_count": 0,
            "high_level_updates": self.high_level_updates,
            "detour_locked": self.detour_locked,
            "detour_turn_sign": self.detour_turn_sign,
        }
        return self.encode(observation)

    def encode(self, observation):
        if (
                self.local_subgoal_world is None
                or self.local_subgoal_yaw_world is None):
            raise RuntimeError("reset() must be called before encode()")
        (
            local_subgoal,
            preview,
            path_mask,
            path_metrics,
        ) = self._path_features(observation)
        velocity = np.clip(
            self._base_velocity(observation) / self.base_velocity_scale,
            -1.0,
            1.0,
        )
        scan = np.clip(
            self._scan(observation),
            0.0,
            self.scan_clip,
        ) / self.scan_clip
        local_yaw_error = self._local_subgoal_yaw_error(observation)
        local_yaw_encoding = np.asarray([
            math.sin(local_yaw_error),
            math.cos(local_yaw_error),
        ], dtype=np.float32)
        continuous = np.concatenate((
            np.clip(
                local_subgoal,
                -self.local_subgoal_max_radius,
                self.local_subgoal_max_radius,
            ).astype(np.float32),
            local_yaw_encoding,
            velocity.astype(np.float32),
            scan.astype(np.float32),
            np.clip(
                preview.reshape(-1),
                -self.local_subgoal_max_radius,
                self.local_subgoal_max_radius,
            ).astype(np.float32),
            path_metrics.astype(np.float32),
            self.previous_action.astype(np.float32),
        ))
        encoded = np.concatenate((
            continuous,
            path_mask.astype(np.float32),
        ))
        if continuous.shape != (self.CONTINUOUS_DIM,):
            raise RuntimeError(
                "continuous pose-path state must have shape ({},), got {}".format(
                    self.CONTINUOUS_DIM,
                    continuous.shape,
                )
            )
        if encoded.shape != (self.OBS_DIM,):
            raise RuntimeError(
                "pose-guided observation must have shape ({},), got {}".format(
                    self.OBS_DIM,
                    encoded.shape,
                )
            )
        if not np.all(np.isfinite(encoded)):
            raise ValueError(
                "pose-guided observation contains non-finite values"
            )
        return encoded

    def transition(self, observation, action):
        if self.previous_distance is None:
            raise RuntimeError("reset() must be called before transition()")
        action = np.clip(
            self._vector(action, self.ACTION_DIM, "action"),
            -1.0,
            1.0,
        ).astype(np.float32)
        final_displacement = self._final_goal_xy(observation)
        distance = float(np.linalg.norm(final_displacement))
        progress = float(self.previous_distance - distance)
        (
            local_subgoal,
            unused_preview,
            unused_mask,
            path_metrics,
        ) = self._path_features(observation)
        local_distance = float(np.linalg.norm(local_subgoal))
        local_yaw_error = self._local_subgoal_yaw_error(observation)
        local_progress = float(
            self.previous_local_distance - local_distance
        )
        cross_track = float(path_metrics[0]) * (
            self.local_subgoal_max_radius
        )
        heading_error = float(path_metrics[1]) * math.pi

        self.minimum_distance = min(self.minimum_distance, distance)
        scan = self._scan(observation)
        clearance_scan = self.clearance_scan(scan)
        minimum_clearance = float(np.min(clearance_scan))
        (
            directional_clearance,
            proximity_sector,
        ) = self.motion_direction_clearance(
            clearance_scan,
            action,
        )
        contact_collision = bool(observation.get("collision", False))
        proximity_collision = bool(
            directional_clearance < self.collision_distance
        )
        collision = bool(contact_collision or proximity_collision)
        success = bool(
            not collision and distance < self.success_threshold
        )
        local_achieved = bool(
            not success
            and local_distance < self.local_subgoal_tolerance
            and abs(local_yaw_error) < self.local_subgoal_yaw_tolerance
        )

        self.step_count += 1
        self.steps_since_high_level += 1
        timeout = bool(
            self.step_count >= self.max_steps
            and not success
            and not collision
        )
        done = bool(success or collision or timeout)

        progress_term = self.progress_reward_scale * progress
        path_progress_term = (
            self.path_progress_reward_scale * local_progress
        )
        waypoint_term = self.waypoint_reward if local_achieved else 0.0
        cross_track_term = -self.cross_track_penalty_scale * min(
            1.0,
            abs(cross_track) / self.local_subgoal_max_radius,
        )
        heading_term = -self.heading_penalty_scale * min(
            1.0,
            abs(heading_error) / math.pi,
        )
        clearance_term = -self.clearance_penalty_scale * min(
            1.0,
            max(0.0, self.preferred_clearance - minimum_clearance)
            / self.preferred_clearance,
        )
        action_term = -self.action_penalty_scale * float(
            np.mean(np.square(action))
        )
        smoothness_term = -self.smoothness_penalty_scale * float(
            np.mean(np.square(action - self.previous_action))
        )
        reverse_term = -self.reverse_penalty_scale * float(
            max(0.0, -float(action[0]))
        )
        idle = bool(
            distance > self.success_threshold
            and float(np.max(np.abs(action))) < self.idle_action_threshold
        )
        idle_term = -self.idle_penalty if idle else 0.0
        success_term = self.success_reward if success else 0.0
        collision_term = -self.collision_penalty if collision else 0.0
        timeout_term = -self.timeout_penalty if timeout else 0.0
        reward = float(
            progress_term
            + path_progress_term
            + waypoint_term
            + cross_track_term
            + heading_term
            + clearance_term
            - self.time_penalty
            + action_term
            + smoothness_term
            + reverse_term
            + idle_term
            + success_term
            + collision_term
            + timeout_term
        )

        if contact_collision:
            collision_source = "contact"
        elif proximity_collision:
            collision_source = "laser_proximity"
        else:
            collision_source = "none"

        replanned = False
        replan_reason = ""
        if not done:
            if local_achieved:
                replan_reason = "local_subgoal_achieved"
            elif cross_track >= self.path_replan_cross_track:
                replan_reason = "path_deviation"
            elif self.steps_since_high_level >= self.high_level_interval:
                replan_reason = "high_level_interval"
            if replan_reason:
                self._update_high_level(
                    observation,
                    reason=replan_reason,
                )
                replanned = True

        self.previous_distance = distance
        if not replanned:
            self.previous_local_distance = local_distance
        self.previous_action = action.copy()
        reported_local_subgoal = (
            self._world_to_body(
                self.local_subgoal_world,
                self._base_pose(observation),
            )
            if replanned else local_subgoal
        )
        reported_local_yaw_error = (
            self._local_subgoal_yaw_error(observation)
            if replanned else local_yaw_error
        )
        distance_reduction = float(self.initial_distance - distance)
        distance_ratio = float(
            distance / max(self.initial_distance, 1.0e-6)
        )
        self.last_diagnostics = {
            "initial_distance": self.initial_distance,
            "distance": distance,
            "minimum_distance": self.minimum_distance,
            "distance_reduction": distance_reduction,
            "distance_ratio": distance_ratio,
            "progress": progress,
            "local_subgoal_distance": local_distance,
            "local_subgoal_yaw_error": local_yaw_error,
            "local_progress": local_progress,
            "path_cross_track_error": cross_track,
            "path_heading_error": heading_error,
            "path_remaining_fraction": float(path_metrics[2]),
            "progress_reward": progress_term,
            "path_progress_reward": path_progress_term,
            "waypoint_reward": waypoint_term,
            "cross_track_penalty": cross_track_term,
            "heading_penalty": heading_term,
            "clearance_penalty": clearance_term,
            "action_penalty": action_term,
            "smoothness_penalty": smoothness_term,
            "reverse_penalty": reverse_term,
            "idle_penalty": idle_term,
            "time_penalty": -self.time_penalty,
            "success_reward": success_term,
            "collision_penalty": collision_term,
            "timeout_penalty": timeout_term,
            "reward": reward,
            "success": success,
            "local_subgoal_achieved": local_achieved,
            "collision": collision,
            "contact_collision": contact_collision,
            "proximity_collision": proximity_collision,
            "collision_source": collision_source,
            "minimum_lidar_range": float(np.min(scan)),
            "minimum_footprint_clearance": minimum_clearance,
            "minimum_directional_clearance": directional_clearance,
            "proximity_collision_sector": proximity_sector,
            "timeout": timeout,
            "done": done,
            "step_count": self.step_count,
            "high_level_replanned": replanned,
            "high_level_replan_reason": replan_reason,
            "high_level_updates": self.high_level_updates,
            "high_level_selection": dict(self.high_level_info),
            "detour_locked": self.detour_locked,
            "detour_turn_sign": self.detour_turn_sign,
            "detour_clear_cycles": self.detour_clear_cycles,
            "detour_lock_updates": self.detour_lock_updates,
            "detour_obstacle_cleared": self.detour_obstacle_cleared,
            "local_subgoal_body": reported_local_subgoal.copy(),
            "local_subgoal_pose_body": np.asarray([
                reported_local_subgoal[0],
                reported_local_subgoal[1],
                reported_local_yaw_error,
            ], dtype=np.float64),
            "local_subgoal_world": self.local_subgoal_world.copy(),
            "local_subgoal_yaw_world": float(
                self.local_subgoal_yaw_world
            ),
        }
        return self.encode(observation), reward, done, self.diagnostics()

    def diagnostics(self):
        return dict(self.last_diagnostics)

    def clearance_scan(self, scan_ranges):
        """Convert center-mounted LiDAR ranges to rectangle-edge clearance."""
        scan = self._vector(
            scan_ranges,
            self.SCAN_DIM,
            "scan_ranges",
        )
        centers = (
            -np.pi
            + (
                np.arange(self.SCAN_DIM, dtype=np.float64) + 0.5
            ) * (2.0 * np.pi / float(self.SCAN_DIM))
        )
        cosine = np.abs(np.cos(centers))
        sine = np.abs(np.sin(centers))
        longitudinal = np.full(self.SCAN_DIM, np.inf)
        lateral = np.full(self.SCAN_DIM, np.inf)
        longitudinal[cosine > 1.0e-9] = (
            self.footprint_half_length / cosine[cosine > 1.0e-9]
        )
        lateral[sine > 1.0e-9] = (
            self.footprint_half_width / sine[sine > 1.0e-9]
        )
        footprint_radius = np.minimum(longitudinal, lateral)
        return scan - footprint_radius

    def motion_direction_clearance(self, clearance_scan, action):
        """Return clearance only in the currently swept motion region."""
        clearance_scan = self._vector(
            clearance_scan,
            self.SCAN_DIM,
            "clearance_scan",
        )
        action = self._vector(action, self.ACTION_DIM, "action")
        centers = self.high_level.bin_centers
        linear = float(action[0])
        angular = float(action[1])
        motion_epsilon = 1.0e-4

        if linear > motion_epsilon:
            mask = (
                np.abs(centers)
                <= self.motion_clearance_half_angle
            )
            return float(np.min(clearance_scan[mask])), "front"
        if linear < -motion_epsilon:
            rear_error = np.abs(np.arctan2(
                np.sin(centers - math.pi),
                np.cos(centers - math.pi),
            ))
            mask = rear_error <= self.motion_clearance_half_angle
            return float(np.min(clearance_scan[mask])), "rear"
        rotation_half_angle = 0.25 * math.pi
        if angular > motion_epsilon:
            mask = (
                np.abs(centers - 0.5 * math.pi)
                <= rotation_half_angle
            )
            return (
                float(np.min(clearance_scan[mask])),
                "rotation_left",
            )
        if angular < -motion_epsilon:
            mask = (
                np.abs(centers + 0.5 * math.pi)
                <= rotation_half_angle
            )
            return (
                float(np.min(clearance_scan[mask])),
                "rotation_right",
            )
        return float("inf"), "stationary"

    def _update_high_level(self, observation, reason):
        final_goal = self._final_goal_xy(observation)
        base_pose = self._base_pose(observation)
        raw_scan = self._scan(observation)
        scan = np.maximum(self.clearance_scan(raw_scan), 0.0)
        release = self._detour_release_status(
            final_goal,
            scan,
            raw_scan,
            base_pose,
        )
        detour_released = False
        if self.detour_locked:
            self.detour_lock_updates += 1
            if release["ready"]:
                # World-frame progress is stable across LiDAR bearings and
                # cannot flicker merely because the chassis rotates.  Once
                # the rear of the footprint has passed the recorded blocker
                # and the direct footprint corridor is clear, waiting for
                # several additional high-level periods makes the base
                # overshoot the final goal while still committed to detour.
                self.detour_clear_cycles = self.detour_release_cycles
            else:
                self.detour_clear_cycles = 0
            self.detour_obstacle_cleared = bool(
                release["obstacle_cleared"]
            )
            if self.detour_clear_cycles >= self.detour_release_cycles:
                released_turn_sign = self.detour_turn_sign
                self.detour_locked = False
                self.detour_turn_sign = 0.0
                # Keep the chosen bypass side for the rest of the episode.
                # If the same long obstacle briefly re-enters the inflated
                # corridor while turning back to the goal, reacquiring it on
                # the opposite side creates an endless orbit around symmetric
                # frontal targets.
                self.preferred_turn_sign = released_turn_sign
                self.detour_clear_cycles = 0
                detour_released = True
                self.detour_escape_active = True
                self.detour_escape_start_world = base_pose[0:2].copy()
                self.detour_escape_direction_world = np.asarray([
                    math.cos(float(base_pose[2])),
                    math.sin(float(base_pose[2])),
                ], dtype=np.float64)
                self.detour_escape_goal_world = (
                    self.detour_escape_start_world
                    + self.detour_escape_max_distance
                    * self.detour_escape_direction_world
                )
                self.detour_escape_updates = 0
                self.detour_escape_clear_cycles = 0

        escape = self._detour_escape_status(
            final_goal,
            scan,
            raw_scan,
            base_pose,
        )
        if self.detour_escape_active:
            if escape["ready"]:
                self.detour_escape_clear_cycles += 1
            else:
                self.detour_escape_clear_cycles = 0
            if (
                    self.detour_escape_clear_cycles
                    >= self.detour_release_cycles):
                self.detour_escape_active = False

        if self.detour_escape_active:
            self.detour_escape_updates += 1
            local_body = self._detour_escape_local_subgoal(
                base_pose,
                escape["distance_travelled"],
            )
            selection = {
                "reason": "post_detour_clearance_escape",
                "selected_angle": float(math.atan2(
                    local_body[1],
                    local_body[0],
                )),
                "selected_radius": float(np.linalg.norm(local_body)),
                "turn_sign": float(np.sign(local_body[1])),
            }
        else:
            local_body, selection = self.high_level.select(
                final_goal,
                scan,
                preferred_turn_sign=self.preferred_turn_sign,
                committed_turn_sign=(
                    self.detour_turn_sign if self.detour_locked else 0.0
                ),
                raw_scan_ranges=raw_scan,
            )
        local_body = np.asarray(local_body, dtype=np.float64)
        if float(np.linalg.norm(local_body)) < 1.0e-6:
            local_body = self._fallback_turn_subgoal(scan)
            selection = dict(selection)
            selection["reason"] = "turn_in_place_fallback"
            selection["selected_angle"] = float(
                math.atan2(local_body[1], local_body[0])
            )
            selection["selected_radius"] = float(
                np.linalg.norm(local_body)
            )
            selection["turn_sign"] = float(np.sign(local_body[1]))

        selected_turn_sign = float(
            selection.get("turn_sign", 0.0)
        )
        if (
                not self.detour_locked
                and selection.get("reason") == "detour_free_sector"):
            if selected_turn_sign != 0.0:
                blocker_ahead = self._record_detour_obstacle(
                    final_goal,
                    raw_scan,
                    base_pose,
                )
                if blocker_ahead:
                    self.detour_locked = True
                    self.detour_turn_sign = selected_turn_sign
                    self.detour_lock_updates = 0
                    self.detour_clear_cycles = 0
                    self.detour_obstacle_cleared = False
                    self.preferred_turn_sign = selected_turn_sign
                else:
                    # The obstacle is behind in episode progress, but can
                    # still block the diagonal return path after the base
                    # starts turning toward the goal.  Resume the original
                    # fixed world-frame escape direction until that complete
                    # footprint corridor is stably clear.
                    if (
                            self.detour_escape_start_world is None
                            or self.detour_escape_direction_world is None):
                        self.detour_escape_start_world = (
                            base_pose[0:2].copy()
                        )
                        self.detour_escape_direction_world = np.asarray([
                            math.cos(float(base_pose[2])),
                            math.sin(float(base_pose[2])),
                        ], dtype=np.float64)
                        self.detour_escape_updates = 0
                    self.detour_escape_active = True
                    self.detour_escape_clear_cycles = 0
                    escape = self._detour_escape_status(
                        final_goal,
                        scan,
                        raw_scan,
                        base_pose,
                    )
                    local_body = self._detour_escape_local_subgoal(
                        base_pose,
                        escape["distance_travelled"],
                    )
                    selection = dict(selection)
                    selection.update({
                        "reason": (
                            "post_detour_clearance_escape_reentry"
                        ),
                        "selected_angle": float(math.atan2(
                            local_body[1],
                            local_body[0],
                        )),
                        "selected_radius": float(
                            np.linalg.norm(local_body)
                        ),
                        "turn_sign": 0.0,
                        "rejected_blocker_progress": (
                            self._finite_or_nan(
                                self.detour_obstacle_progress
                            )
                        ),
                        "rejected_blocker_is_behind": True,
                    })
        elif self.detour_locked:
            self.preferred_turn_sign = self.detour_turn_sign
        # Do not clear preferred_turn_sign after a direct segment.  It is
        # episode-local memory of the side used for the current obstacle and
        # is reset explicitly by reset().

        self.local_subgoal_world = self._body_to_world(
            local_body,
            base_pose,
        )
        if (
                self.detour_escape_active
                and self.detour_escape_direction_world is not None):
            self.local_subgoal_yaw_world = float(math.atan2(
                self.detour_escape_direction_world[1],
                self.detour_escape_direction_world[0],
            ))
        else:
            local_yaw = float(math.atan2(local_body[1], local_body[0]))
            self.local_subgoal_yaw_world = self._wrap_angle(
                float(base_pose[2]) + local_yaw
            )
        self.path_start_world = base_pose[0:2].copy()
        self.path_points_world = self._build_path(
            local_body,
            base_pose,
        )
        self.previous_local_distance = float(
            np.linalg.norm(local_body)
        )
        self.steps_since_high_level = 0
        self.high_level_updates += 1
        self.high_level_info = dict(selection)
        self.high_level_info.update({
            "update_reason": str(reason),
            "preferred_turn_sign": self.preferred_turn_sign,
            "update_index": self.high_level_updates,
            "detour_locked": self.detour_locked,
            "detour_turn_sign": self.detour_turn_sign,
            "detour_lock_updates": self.detour_lock_updates,
            "detour_clear_cycles": self.detour_clear_cycles,
            "detour_release_ready": bool(release["ready"]),
            "detour_obstacle_cleared": bool(
                release["obstacle_cleared"]
            ),
            "detour_direct_clear": bool(release["direct_clear"]),
            "detour_side_clearance": float(
                release["side_clearance"]
            ),
            "detour_obstacle_angle": float(
                release["obstacle_angle"]
            ),
            "detour_passed_obstacle": bool(
                release["passed_obstacle"]
            ),
            "detour_current_progress": float(
                release["current_progress"]
            ),
            "detour_obstacle_progress": float(
                release["obstacle_progress"]
            ),
            "detour_required_progress": float(
                release["required_progress"]
            ),
            "detour_released": detour_released,
            "detour_escape_active": self.detour_escape_active,
            "detour_escape_updates": self.detour_escape_updates,
            "detour_escape_clear_cycles": (
                self.detour_escape_clear_cycles
            ),
            "detour_escape_distance": float(
                escape["distance_travelled"]
            ),
            "detour_escape_rotation_clearance": float(
                escape["rotation_clearance"]
            ),
            "detour_escape_ready": bool(escape["ready"]),
            "detour_escape_direct_clear": bool(
                escape["direct_clear"]
            ),
            "subgoal_yaw_world": float(
                self.local_subgoal_yaw_world
            ),
            "subgoal_yaw_body_error": float(
                self._wrap_angle(
                    self.local_subgoal_yaw_world - base_pose[2]
                )
            ),
        })

    def _detour_release_status(
            self,
            final_goal,
            scan,
            raw_scan,
            base_pose):
        """Require the complete footprint to pass before ending a detour."""
        current_progress = self._episode_progress(base_pose[0:2])
        obstacle_progress = self._finite_or_nan(
            self.detour_obstacle_progress
        )
        required_progress = self._finite_or_nan(
            self.detour_required_progress
        )
        if not self.detour_locked or self.detour_turn_sign == 0.0:
            return {
                "ready": False,
                "direct_clear": False,
                "obstacle_cleared": False,
                "side_clearance": 0.0,
                "obstacle_angle": 0.0,
                "passed_obstacle": False,
                "current_progress": current_progress,
                "obstacle_progress": obstacle_progress,
                "required_progress": required_progress,
            }

        centers = self.high_level.bin_centers
        goal_distance = float(np.linalg.norm(final_goal))
        goal_angle = float(math.atan2(final_goal[1], final_goal[0]))
        angular_error = np.abs(np.arctan2(
            np.sin(centers - goal_angle),
            np.cos(centers - goal_angle),
        ))
        direct_index = int(np.argmin(angular_error))
        required_direct_clearance = (
            min(self.local_subgoal_max_radius, goal_distance)
            + self.high_level.clearance_margin
        )
        direct_clear = bool(
            scan[direct_index] >= required_direct_clearance
            and self.high_level.corridor_heading_feasible(
                raw_scan,
                goal_angle,
                min(self.local_subgoal_max_radius, goal_distance),
            )
        )

        # For a left detour the obstacle remains on the right, and vice
        # versa.  Do not release merely because the direct ray is visible:
        # the nearest point of that obstacle must also have moved behind the
        # rear-angle boundary and the forward/side footprint must be clear.
        inside_mask = (
            centers * self.detour_turn_sign < 0.0
        )
        inside_indices = np.flatnonzero(inside_mask)
        if inside_indices.size == 0:
            obstacle_angle = 0.0
            obstacle_nearby = False
        else:
            closest_inside = int(
                inside_indices[np.argmin(scan[inside_indices])]
            )
            obstacle_angle = float(centers[closest_inside])
            obstacle_nearby = bool(
                scan[closest_inside] < self.detour_nearby_clearance
            )
        obstacle_behind = bool(
            abs(obstacle_angle) >= self.detour_release_rear_angle
        )
        obstacle_cleared = bool(
            not obstacle_nearby or obstacle_behind
        )

        forward_side_mask = (
            inside_mask
            & (np.abs(centers) < self.detour_release_rear_angle)
        )
        if np.any(forward_side_mask):
            side_clearance = float(np.min(scan[forward_side_mask]))
        else:
            side_clearance = float("inf")
        side_clear = bool(
            side_clearance >= self.detour_release_side_clearance
        )
        minimum_updates_met = bool(
            self.detour_lock_updates >= self.detour_min_lock_updates
        )
        passed_obstacle = bool(
            self.detour_required_progress is not None
            and current_progress >= self.detour_required_progress
        )
        return {
            "ready": bool(
                minimum_updates_met
                and direct_clear
                and passed_obstacle
                and side_clear
            ),
            "direct_clear": direct_clear,
            "obstacle_cleared": obstacle_cleared,
            "side_clearance": side_clearance,
            "obstacle_angle": obstacle_angle,
            "passed_obstacle": passed_obstacle,
            "current_progress": current_progress,
            "obstacle_progress": obstacle_progress,
            "required_progress": required_progress,
        }

    def _record_detour_obstacle(
            self,
            final_goal,
            raw_scan,
            base_pose):
        """Store the first blocking LiDAR point in the world frame.

        A scan obstacle can disappear while the chassis turns even though the
        base has not passed it.  Keeping the point in the world frame gives
        the detour lock a geometric release condition independent of the
        current LiDAR bearing.
        """
        centers = self.high_level.bin_centers
        goal_distance = float(np.linalg.norm(final_goal))
        goal_angle = float(math.atan2(final_goal[1], final_goal[0]))
        relative_angle = np.arctan2(
            np.sin(centers - goal_angle),
            np.cos(centers - goal_angle),
        )
        ranges = np.asarray(raw_scan, dtype=np.float64)
        longitudinal = ranges * np.cos(relative_angle)
        lateral = np.abs(ranges * np.sin(relative_angle))
        corridor_half_width = (
            self.footprint_half_width
            + self.high_level.clearance_margin
        )
        corridor_length = (
            min(self.local_subgoal_max_radius, goal_distance)
            + self.footprint_half_length
            + self.high_level.clearance_margin
        )
        blocking = (
            np.isfinite(ranges)
            & (ranges > 0.0)
            & (ranges < self.scan_clip)
            & (longitudinal > 0.0)
            & (longitudinal <= corridor_length)
            & (lateral <= corridor_half_width)
        )
        candidates = np.flatnonzero(blocking)
        if candidates.size == 0:
            # The high level can also enter detour mode because the nearest
            # angular bin is blocked just outside the inflated corridor.
            # Retain that nearest forward point instead of allowing an
            # ungrounded LiDAR-only release.
            forward = (
                np.isfinite(ranges)
                & (ranges > 0.0)
                & (ranges < self.scan_clip)
                & (longitudinal > 0.0)
                & (np.abs(relative_angle) < 0.5 * math.pi)
            )
            candidates = np.flatnonzero(forward)
        if candidates.size == 0:
            self.detour_obstacle_world = None
            self.detour_obstacle_progress = None
            self.detour_required_progress = None
            return False

        index = int(candidates[np.argmin(longitudinal[candidates])])
        obstacle_body = ranges[index] * np.asarray([
            math.cos(float(centers[index])),
            math.sin(float(centers[index])),
        ], dtype=np.float64)
        self.detour_obstacle_world = self._body_to_world(
            obstacle_body,
            base_pose,
        )
        self.detour_obstacle_progress = self._episode_progress(
            self.detour_obstacle_world
        )
        self.detour_required_progress = float(
            self.detour_obstacle_progress
            + self.footprint_half_length
        )
        current_progress = self._episode_progress(base_pose[0:2])
        return bool(
            self.detour_obstacle_progress
            > current_progress + 1.0e-3
        )

    def _direct_local_subgoal(self, final_goal):
        distance = float(np.linalg.norm(final_goal))
        if distance <= 1.0e-9:
            return np.zeros(2, dtype=np.float64)
        radius = min(self.local_subgoal_max_radius, distance)
        return np.asarray(final_goal, dtype=np.float64) * (
            radius / distance
        )

    def _detour_escape_local_subgoal(
            self,
            base_pose,
            distance_travelled):
        escape_goal_progress = max(
            self.detour_escape_max_distance,
            distance_travelled + self.local_subgoal_max_radius,
        )
        self.detour_escape_goal_world = (
            self.detour_escape_start_world
            + escape_goal_progress
            * self.detour_escape_direction_world
        )
        escape_goal_body = self._world_to_body(
            self.detour_escape_goal_world,
            base_pose,
        )
        escape_goal_distance = float(np.linalg.norm(escape_goal_body))
        if escape_goal_distance <= 1.0e-9:
            return np.zeros(2, dtype=np.float64)
        escape_radius = min(
            self.local_subgoal_max_radius,
            escape_goal_distance,
        )
        return escape_goal_body * (
            escape_radius / escape_goal_distance
        )

    def _detour_escape_status(
            self,
            final_goal,
            scan,
            raw_scan,
            base_pose):
        if (
                not self.detour_escape_active
                or self.detour_escape_start_world is None
                or self.detour_escape_direction_world is None):
            return {
                "ready": False,
                "distance_travelled": 0.0,
                "rotation_clearance": 0.0,
                "direct_clear": False,
            }

        distance_travelled = float(np.dot(
            base_pose[0:2] - self.detour_escape_start_world,
            self.detour_escape_direction_world,
        ))
        goal_angle = float(math.atan2(final_goal[1], final_goal[0]))
        centers = self.high_level.bin_centers
        side_half_angle = 0.25 * math.pi
        if goal_angle >= 0.0:
            rotation_mask = (
                np.abs(centers - 0.5 * math.pi)
                <= side_half_angle
            )
        else:
            rotation_mask = (
                np.abs(centers + 0.5 * math.pi)
                <= side_half_angle
            )
        rotation_clearance = float(np.min(scan[rotation_mask]))
        enough_distance = bool(
            distance_travelled >= self.detour_escape_distance
        )
        enough_clearance = bool(
            rotation_clearance >= self.detour_escape_clearance
        )
        goal_distance = float(np.linalg.norm(final_goal))
        goal_angle = float(math.atan2(final_goal[1], final_goal[0]))
        direct_radius = min(
            self.local_subgoal_max_radius,
            goal_distance,
        )
        angular_error = np.abs(np.arctan2(
            np.sin(centers - goal_angle),
            np.cos(centers - goal_angle),
        ))
        direct_index = int(np.argmin(angular_error))
        direct_clear = bool(
            scan[direct_index] >= (
                direct_radius + self.high_level.clearance_margin
            )
            and self.high_level.corridor_heading_feasible(
                raw_scan,
                goal_angle,
                direct_radius,
            )
        )
        return {
            "ready": bool(
                enough_distance
                and enough_clearance
                and direct_clear
            ),
            "distance_travelled": distance_travelled,
            "rotation_clearance": rotation_clearance,
            "direct_clear": direct_clear,
        }

    def _episode_progress(self, point_world):
        if (
                self.episode_start_world is None
                or self.episode_progress_direction is None):
            return 0.0
        return float(np.dot(
            np.asarray(point_world, dtype=np.float64)
            - self.episode_start_world,
            self.episode_progress_direction,
        ))

    @staticmethod
    def _finite_or_nan(value):
        return float("nan") if value is None else float(value)

    def _build_path(self, local_body, base_pose):
        # The selected LiDAR sector certifies the straight ray, not an
        # arbitrary spline.  Sampling that ray keeps every preview point
        # consistent with the high-level feasibility check and naturally
        # asks a nonholonomic learner to rotate before translating.
        values = []
        for index in range(1, self.path_internal_points + 1):
            t = float(index) / float(self.path_internal_points)
            point_body = t * local_body
            values.append(self._body_to_world(point_body, base_pose))
        return np.asarray(values, dtype=np.float64)

    def _path_features(self, observation):
        base_pose = self._base_pose(observation)
        local_body = self._world_to_body(
            self.local_subgoal_world,
            base_pose,
        )
        points_body = np.asarray([
            self._world_to_body(point, base_pose)
            for point in self.path_points_world
        ], dtype=np.float64)
        distances = np.linalg.norm(points_body, axis=1)
        nearest = int(np.argmin(distances))
        if (
                distances[nearest] < 0.08
                and nearest + 1 < self.path_points_world.shape[0]):
            nearest += 1

        preview = np.zeros(
            (self.PATH_POINT_COUNT, self.PATH_POINT_DIM),
            dtype=np.float64,
        )
        mask = np.zeros(self.PATH_POINT_COUNT, dtype=np.float64)
        available = min(
            self.PATH_POINT_COUNT,
            self.path_points_world.shape[0] - nearest,
        )
        if available > 0:
            preview[0:available] = points_body[
                nearest:nearest + available
            ]
            mask[0:available] = 1.0

        cross_track = self._cross_track_error(
            base_pose[0:2],
        )
        heading_error = 0.0
        if available > 0:
            heading_error = float(math.atan2(
                preview[0, 1],
                preview[0, 0],
            ))
        path_remaining = self._path_remaining(
            base_pose[0:2],
            nearest,
        )
        metrics = np.asarray([
            np.clip(
                cross_track / self.local_subgoal_max_radius,
                0.0,
                1.0,
            ),
            np.clip(heading_error / math.pi, -1.0, 1.0),
            np.clip(
                path_remaining
                / max(2.0 * self.local_subgoal_max_radius, 1.0e-6),
                0.0,
                1.0,
            ),
        ], dtype=np.float32)
        return local_body, preview, mask, metrics

    def _local_subgoal_yaw_error(self, observation):
        base_pose = self._base_pose(observation)
        return self._wrap_angle(
            float(self.local_subgoal_yaw_world) - float(base_pose[2])
        )

    def _cross_track_error(self, base_xy):
        polyline = np.vstack((
            self.path_start_world.reshape(1, 2),
            self.path_points_world,
        ))
        return min(
            self._point_segment_distance(
                base_xy,
                polyline[index],
                polyline[index + 1],
            )
            for index in range(polyline.shape[0] - 1)
        )

    def _path_remaining(self, base_xy, nearest):
        points = self.path_points_world
        remaining = float(np.linalg.norm(points[nearest] - base_xy))
        if nearest + 1 < points.shape[0]:
            remaining += float(np.sum(np.linalg.norm(
                np.diff(points[nearest:], axis=0),
                axis=1,
            )))
        return remaining

    def _fallback_turn_subgoal(self, scan):
        centers = self.high_level.bin_centers
        left = float(np.mean(scan[centers > 0.0]))
        right = float(np.mean(scan[centers < 0.0]))
        if self.preferred_turn_sign != 0.0:
            turn_sign = self.preferred_turn_sign
        else:
            turn_sign = 1.0 if left >= right else -1.0
        return np.asarray([
            0.0,
            turn_sign * self.high_level.minimum_radius,
        ], dtype=np.float64)

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
    def _wrap_angle(value):
        return float(math.atan2(math.sin(value), math.cos(value)))

    @staticmethod
    def _body_to_world(point_body, base_pose):
        cosine = math.cos(float(base_pose[2]))
        sine = math.sin(float(base_pose[2]))
        return np.asarray([
            base_pose[0] + cosine * point_body[0] - sine * point_body[1],
            base_pose[1] + sine * point_body[0] + cosine * point_body[1],
        ], dtype=np.float64)

    @staticmethod
    def _world_to_body(point_world, base_pose):
        delta = np.asarray(point_world, dtype=np.float64) - base_pose[0:2]
        cosine = math.cos(float(base_pose[2]))
        sine = math.sin(float(base_pose[2]))
        return np.asarray([
            cosine * delta[0] + sine * delta[1],
            -sine * delta[0] + cosine * delta[1],
        ], dtype=np.float64)

    @staticmethod
    def _base_pose(observation):
        joint_position = PlanarSubgoalTask._vector(
            observation["joint_pos"],
            10,
            "joint_pos",
        )
        # q[2] is the virtual z-joint angle, not the tracked chassis forward
        # heading.  The CAD assembly has a fixed -90 degree yaw offset and the
        # ROS observation already exposes the corrected heading.  Using q[2]
        # here rotated every stored local path by 90 degrees in world space.
        heading = float(observation.get(
            "planar_base_heading",
            joint_position[2],
        ))
        if not np.isfinite(heading):
            raise ValueError("planar_base_heading is non-finite")
        return np.asarray([
            joint_position[0],
            joint_position[1],
            heading,
        ], dtype=np.float64)

    @staticmethod
    def _final_goal_xy(observation):
        if not isinstance(observation, dict):
            raise TypeError("subgoal task requires structured observation")
        return PlanarSubgoalTask._vector(
            observation["base_to_target_body_pos"],
            3,
            "base_to_target_body_pos",
        )[0:2]

    @staticmethod
    def _base_velocity(observation):
        if "observed_planar_body_velocity" not in observation:
            raise ValueError(
                "observed_planar_body_velocity is required for direct RL"
            )
        return PlanarSubgoalTask._vector(
            observation["observed_planar_body_velocity"],
            3,
            "observed_planar_body_velocity",
        )[[0, 2]]

    @staticmethod
    def _scan(observation):
        return PlanarSubgoalTask._vector(
            observation["scan_bins"],
            PlanarSubgoalTask.SCAN_DIM,
            "scan_bins",
        )

    @staticmethod
    def _vector(value, expected_size, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (expected_size,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name,
                    expected_size,
                    vector.shape,
                )
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("{} contains non-finite values".format(name))
        return vector

    def _validate_parameters(self):
        if self.success_threshold <= 0.0:
            raise ValueError("success_threshold must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.scan_clip <= 0.0 or self.target_clip <= 0.0:
            raise ValueError("scan_clip and target_clip must be positive")
        if (
                self.footprint_half_length <= 0.0
                or self.footprint_half_width <= 0.0):
            raise ValueError("footprint half extents must be positive")
        if np.any(self.base_velocity_scale <= 0.0):
            raise ValueError("base_velocity_scale values must be positive")
        if self.collision_distance < 0.0:
            raise ValueError("collision_distance must be non-negative")
        if not 0.0 < self.motion_clearance_half_angle <= math.pi:
            raise ValueError(
                "motion_clearance_half_angle must be in (0, pi]"
            )
        if self.timeout_penalty < 0.0:
            raise ValueError("timeout_penalty must be non-negative")
        if self.idle_action_threshold < 0.0:
            raise ValueError("idle_action_threshold must be non-negative")
        if self.high_level_interval <= 0:
            raise ValueError("high_level_interval must be positive")
        if self.local_subgoal_tolerance <= 0.0:
            raise ValueError("local_subgoal_tolerance must be positive")
        if not 0.0 < self.local_subgoal_yaw_tolerance <= math.pi:
            raise ValueError(
                "local_subgoal_yaw_tolerance must be in (0, pi]"
            )
        if self.local_subgoal_max_radius <= 0.0:
            raise ValueError("local_subgoal_max_radius must be positive")
        if self.path_internal_points < self.PATH_POINT_COUNT:
            raise ValueError(
                "path_internal_points must cover all preview points"
            )
        if self.path_replan_cross_track <= 0.0:
            raise ValueError("path_replan_cross_track must be positive")
        if self.preferred_clearance <= 0.0:
            raise ValueError("preferred_clearance must be positive")
        if self.detour_release_cycles <= 0:
            raise ValueError("detour_release_cycles must be positive")
        if self.detour_min_lock_updates < 0:
            raise ValueError(
                "detour_min_lock_updates cannot be negative"
            )
        if not 0.5 * math.pi < self.detour_release_rear_angle < math.pi:
            raise ValueError(
                "detour_release_rear_angle must be in (pi/2, pi)"
            )
        if (
                self.detour_release_side_clearance <= 0.0
                or self.detour_nearby_clearance <= 0.0):
            raise ValueError(
                "detour release clearances must be positive"
            )
        if (
                self.detour_escape_distance <= 0.0
                or self.detour_escape_clearance <= 0.0):
            raise ValueError(
                "detour escape distance and clearance must be positive"
            )
        if self.detour_escape_max_distance < self.detour_escape_distance:
            raise ValueError(
                "detour_escape_max_distance must not be smaller than "
                "detour_escape_distance"
            )
