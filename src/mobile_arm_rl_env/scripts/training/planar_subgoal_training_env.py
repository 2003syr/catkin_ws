#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Gazebo adapter for pose-guided subgoal-conditioned base RL."""

from __future__ import print_function

import math

import numpy as np
import rospy

from training.planar_base_training_env import PlanarBaseTrainingEnv
from training.planar_subgoal_task import PlanarSubgoalTask


class PlanarSubgoalTrainingEnv(PlanarBaseTrainingEnv):
    """Train [v, omega] from a local path with an emergency-only shield."""

    OBS_DIM = PlanarSubgoalTask.OBS_DIM
    ACTION_DIM = PlanarSubgoalTask.ACTION_DIM

    def __init__(self, base_env=None, init_ros_node=True, seed=None):
        super(PlanarSubgoalTrainingEnv, self).__init__(
            base_env=base_env,
            init_ros_node=init_ros_node,
            seed=seed,
        )
        if int(self.base_env.scan_bin_count) != PlanarSubgoalTask.SCAN_DIM:
            raise RuntimeError(
                "direct RL requires scan_bin_count={}".format(
                    PlanarSubgoalTask.SCAN_DIM
                )
            )
        maximum_linear_speed = (
            self.base_env.action_max_vel["x"] * self.base_action_scale
        )
        maximum_yaw_rate = (
            self.base_env.action_max_vel["z"] * self.base_action_scale
        )
        footprint_half_length = float(
            rospy.get_param("~shield_footprint_half_length", 0.56)
        )
        footprint_half_width = float(
            rospy.get_param("~shield_footprint_half_width", 0.10)
        )
        shield_front_half_angle = float(
            rospy.get_param(
                "~shield_front_half_angle",
                math.radians(35.0),
            )
        )
        self.task = PlanarSubgoalTask(
            success_threshold=rospy.get_param(
                "~subgoal_success_threshold", 0.08
            ),
            max_steps=rospy.get_param("~max_steps", 600),
            progress_reward_scale=rospy.get_param(
                "~progress_reward_scale", 4.0
            ),
            path_progress_reward_scale=rospy.get_param(
                "~path_progress_reward_scale", 8.0
            ),
            waypoint_reward=rospy.get_param(
                "~waypoint_reward", 0.25
            ),
            success_reward=rospy.get_param("~success_reward", 5.0),
            collision_penalty=rospy.get_param(
                "~collision_penalty", 5.0
            ),
            timeout_penalty=rospy.get_param(
                "~timeout_penalty", 2.0
            ),
            time_penalty=rospy.get_param("~time_penalty", 0.005),
            action_penalty_scale=rospy.get_param(
                "~action_penalty_scale", 0.002
            ),
            smoothness_penalty_scale=rospy.get_param(
                "~smoothness_penalty_scale", 0.01
            ),
            reverse_penalty_scale=rospy.get_param(
                "~reverse_penalty_scale", 0.01
            ),
            idle_penalty=rospy.get_param("~idle_penalty", 0.01),
            idle_action_threshold=rospy.get_param(
                "~idle_action_threshold", 0.05
            ),
            cross_track_penalty_scale=rospy.get_param(
                "~cross_track_penalty_scale", 0.01
            ),
            heading_penalty_scale=rospy.get_param(
                "~heading_penalty_scale", 0.002
            ),
            clearance_penalty_scale=rospy.get_param(
                "~clearance_penalty_scale", 0.05
            ),
            preferred_clearance=rospy.get_param(
                "~preferred_clearance", 0.25
            ),
            collision_distance=rospy.get_param(
                "~collision_distance", 0.18
            ),
            motion_clearance_half_angle=shield_front_half_angle,
            scan_clip=rospy.get_param("~scan_clip", 10.0),
            target_clip=max(2.0, self.target_max_distance),
            footprint_half_length=footprint_half_length,
            footprint_half_width=footprint_half_width,
            base_velocity_scale=(
                maximum_linear_speed,
                maximum_yaw_rate,
            ),
            high_level_interval=rospy.get_param(
                "~high_level_interval", 10
            ),
            local_subgoal_tolerance=rospy.get_param(
                "~local_subgoal_tolerance", 0.10
            ),
            local_subgoal_yaw_tolerance=rospy.get_param(
                "~local_subgoal_yaw_tolerance", 0.17
            ),
            local_subgoal_min_radius=rospy.get_param(
                "~local_subgoal_min_radius", 0.20
            ),
            local_subgoal_max_radius=rospy.get_param(
                "~local_subgoal_max_radius", 0.50
            ),
            local_subgoal_clearance_margin=rospy.get_param(
                "~local_subgoal_clearance_margin", 0.18
            ),
            path_internal_points=rospy.get_param(
                "~path_internal_points", 10
            ),
            path_replan_cross_track=rospy.get_param(
                "~path_replan_cross_track", 0.35
            ),
            detour_release_cycles=rospy.get_param(
                "~detour_release_cycles", 3
            ),
            detour_min_lock_updates=rospy.get_param(
                "~detour_min_lock_updates", 4
            ),
            detour_release_rear_angle=rospy.get_param(
                "~detour_release_rear_angle", math.radians(110.0)
            ),
            detour_release_side_clearance=rospy.get_param(
                "~detour_release_side_clearance", 0.25
            ),
            detour_nearby_clearance=rospy.get_param(
                "~detour_nearby_clearance", 0.80
            ),
        )
        self.shield_enabled = bool(
            rospy.get_param("~shield_enabled", True)
        )
        self.maximum_linear_speed = float(maximum_linear_speed)
        self.maximum_yaw_rate = float(maximum_yaw_rate)
        self.shield_footprint_half_length = footprint_half_length
        self.shield_footprint_half_width = footprint_half_width
        self.shield_min_clearance = float(
            rospy.get_param(
                "~shield_min_clearance",
                rospy.get_param("~shield_stop_distance", 0.22),
            )
        )
        self.shield_reaction_time = float(
            rospy.get_param("~shield_reaction_time", 0.10)
        )
        self.shield_braking_deceleration = float(
            rospy.get_param(
                "~shield_braking_deceleration", 0.30
            )
        )
        self.shield_rotation_clearance = float(
            rospy.get_param("~shield_rotation_clearance", 0.18)
        )
        self.shield_front_half_angle = shield_front_half_angle
        self.shield_penalty_scale = float(
            rospy.get_param("~shield_penalty_scale", 0.01)
        )
        self.shield_streak_penalty_scale = float(
            rospy.get_param("~shield_streak_penalty_scale", 0.0)
        )
        self.shield_streak_penalty_cap = int(
            rospy.get_param("~shield_streak_penalty_cap", 50)
        )
        self.escape_turn_reward_scale = float(
            rospy.get_param("~escape_turn_reward_scale", 0.0)
        )
        self.escape_turn_clearance_scale = float(
            rospy.get_param("~escape_turn_clearance_scale", 0.50)
        )
        self.shield_streak = 0
        if self.shield_min_clearance <= 0.0:
            raise ValueError("shield_min_clearance must be positive")
        if self.shield_reaction_time < 0.0:
            raise ValueError("shield_reaction_time cannot be negative")
        if self.shield_braking_deceleration <= 0.0:
            raise ValueError(
                "shield_braking_deceleration must be positive"
            )
        if self.shield_rotation_clearance <= 0.0:
            raise ValueError(
                "shield_rotation_clearance must be positive"
            )
        if not 0.0 < self.shield_front_half_angle <= math.pi:
            raise ValueError(
                "shield_front_half_angle must be in (0, pi]"
            )
        if self.shield_penalty_scale < 0.0:
            raise ValueError("shield_penalty_scale must be non-negative")
        if self.shield_streak_penalty_scale < 0.0:
            raise ValueError(
                "shield_streak_penalty_scale must be non-negative"
            )
        if self.shield_streak_penalty_cap <= 0:
            raise ValueError("shield_streak_penalty_cap must be positive")
        if self.escape_turn_reward_scale < 0.0:
            raise ValueError(
                "escape_turn_reward_scale must be non-negative"
            )
        if self.escape_turn_clearance_scale <= 0.0:
            raise ValueError(
                "escape_turn_clearance_scale must be positive"
            )

    def reset(self, target_position=None):
        self.shield_streak = 0
        return super(PlanarSubgoalTrainingEnv, self).reset(
            target_position=target_position
        )

    def step(self, base_action):
        raw_action = np.clip(
            self._vector(base_action, self.ACTION_DIM, "base_action"),
            -1.0,
            1.0,
        ).astype(np.float32)
        safe_action, shield_info = self._apply_shield(raw_action)
        observation, reward, done, info = super(
            PlanarSubgoalTrainingEnv,
            self,
        ).step(safe_action)
        intervention_magnitude = float(np.mean(np.abs(
            raw_action - safe_action
        )))
        shield_penalty = (
            -self.shield_penalty_scale * intervention_magnitude
        )
        if bool(shield_info["shield_intervened"]):
            self.shield_streak += 1
        else:
            self.shield_streak = 0
        streak_fraction = min(
            1.0,
            float(self.shield_streak)
            / float(self.shield_streak_penalty_cap),
        )
        shield_streak_penalty = (
            -self.shield_streak_penalty_scale * streak_fraction
        )
        clearance_advantage = (
            float(shield_info["shield_left_clearance"])
            - float(shield_info["shield_right_clearance"])
        )
        preferred_turn_sign = float(np.sign(clearance_advantage))
        turn_alignment = (
            preferred_turn_sign * float(np.sign(safe_action[1]))
        )
        normalized_clearance_advantage = min(
            1.0,
            abs(clearance_advantage)
            / self.escape_turn_clearance_scale,
        )
        escape_turn_reward = 0.0
        if (
                bool(shield_info["shield_front_blocked"])
                and turn_alignment > 0.0):
            escape_turn_reward = (
                self.escape_turn_reward_scale
                * abs(float(safe_action[1]))
                * normalized_clearance_advantage
            )
        reward = float(
            reward
            + shield_penalty
            + shield_streak_penalty
            + escape_turn_reward
        )
        info = dict(info)
        info.update(shield_info)
        info.update({
            "raw_policy_action": raw_action.copy(),
            "safe_policy_action": safe_action.copy(),
            "shield_intervention_magnitude": intervention_magnitude,
            "shield_penalty": shield_penalty,
            "shield_streak": int(self.shield_streak),
            "shield_streak_penalty": shield_streak_penalty,
            "escape_turn_reward": escape_turn_reward,
            "escape_preferred_turn_sign": preferred_turn_sign,
            "reward": reward,
        })
        return observation, reward, done, info

    def _apply_shield(self, action):
        safe_action = np.asarray(action, dtype=np.float32).copy()
        scan = self.base_env.get_scan_bins()
        clearance_scan = self.task.clearance_scan(scan)
        centers = (
            -math.pi
            + (
                np.arange(scan.size, dtype=np.float64) + 0.5
            ) * (2.0 * math.pi / float(scan.size))
        )
        front_mask = np.abs(centers) <= self.shield_front_half_angle
        rear_mask = (
            np.abs(
                np.arctan2(
                    np.sin(centers - math.pi),
                    np.cos(centers - math.pi),
                )
            ) <= self.shield_front_half_angle
        )
        side_half_angle = 0.25 * math.pi
        left_mask = np.abs(centers - 0.5 * math.pi) <= side_half_angle
        right_mask = np.abs(centers + 0.5 * math.pi) <= side_half_angle
        front_clearance = float(np.min(clearance_scan[front_mask]))
        rear_clearance = float(np.min(clearance_scan[rear_mask]))
        left_clearance = float(np.min(clearance_scan[left_mask]))
        right_clearance = float(np.min(clearance_scan[right_mask]))
        sweep_clearance = float(np.min(clearance_scan))
        reasons = []
        required_stop_distance = self.shield_min_clearance
        allowed_linear_action = 1.0
        requested_linear = float(safe_action[0])
        requested_angular = float(safe_action[1])
        translation_clearance = (
            front_clearance
            if requested_linear >= 0.0
            else rear_clearance
        )
        terminal_translation_stop = bool(
            self.shield_enabled
            and abs(requested_linear) > 1.0e-6
            and translation_clearance < self.task.collision_distance
        )
        terminal_rotation_stop = bool(
            self.shield_enabled
            and abs(requested_angular) > 1.0e-6
            and abs(requested_linear) <= 1.0e-6
            and sweep_clearance < self.task.collision_distance
            and left_clearance < self.task.collision_distance
            and right_clearance < self.task.collision_distance
        )
        terminal_emergency_stop = bool(
            terminal_translation_stop or terminal_rotation_stop
        )
        if terminal_translation_stop:
            # Stop translation at the terminal boundary, but preserve an
            # in-place escape turn whenever one side still has clearance.
            # An obstacle behind a forward-moving base is deliberately not
            # considered here because the base is moving away from it.
            safe_action[0] = 0.0
            allowed_linear_action = 0.0
            reasons.append("terminal_translation_stop")
        if terminal_rotation_stop:
            safe_action[1] = 0.0
            reasons.append("terminal_rotation_stop")
        if (
                self.shield_enabled
                and not terminal_translation_stop
                and safe_action[0] != 0.0):
            moving_forward = bool(safe_action[0] > 0.0)
            clearance = (
                front_clearance if moving_forward else rear_clearance
            )
            desired_speed = (
                abs(float(safe_action[0])) * self.maximum_linear_speed
            )
            required_stop_distance = (
                self.shield_min_clearance
                + desired_speed * self.shield_reaction_time
                + desired_speed ** 2
                / (2.0 * self.shield_braking_deceleration)
            )
            available_distance = max(
                0.0,
                clearance - self.shield_min_clearance,
            )
            braking = self.shield_braking_deceleration
            reaction = self.shield_reaction_time
            allowed_speed = (
                -braking * reaction
                + math.sqrt(
                    (braking * reaction) ** 2
                    + 2.0 * braking * available_distance
                )
            )
            allowed_linear_action = float(np.clip(
                allowed_speed / self.maximum_linear_speed,
                0.0,
                1.0,
            ))
            if abs(float(safe_action[0])) > allowed_linear_action:
                safe_action[0] = (
                    allowed_linear_action
                    if moving_forward else -allowed_linear_action
                )
                reasons.append(
                    "front_braking_clip"
                    if moving_forward else "rear_braking_clip"
                )
        front_blocked = bool(
            front_clearance < self.shield_min_clearance
            or "front_braking_clip" in reasons
        )
        if self.shield_enabled and not terminal_rotation_stop:
            if (
                    safe_action[1] > 0.0
                    and left_clearance < self.shield_rotation_clearance):
                safe_action[1] = 0.0
                reasons.append("left_rotation_stop")
            elif (
                    safe_action[1] < 0.0
                    and right_clearance < self.shield_rotation_clearance):
                safe_action[1] = 0.0
                reasons.append("right_rotation_stop")
        rotation_preserved = bool(
            front_blocked
            and abs(float(action[1])) > 1.0e-6
            and abs(float(safe_action[1])) > 1.0e-6
        )
        return safe_action, {
            "shield_enabled": bool(self.shield_enabled),
            "shield_intervened": bool(reasons),
            "shield_reasons": reasons,
            "shield_front_clearance": front_clearance,
            "shield_rear_clearance": rear_clearance,
            "shield_left_clearance": left_clearance,
            "shield_right_clearance": right_clearance,
            "shield_sweep_clearance": sweep_clearance,
            "shield_front_blocked": front_blocked,
            "shield_terminal_emergency_stop": terminal_emergency_stop,
            "shield_terminal_translation_stop": (
                terminal_translation_stop
            ),
            "shield_terminal_rotation_stop": terminal_rotation_stop,
            "shield_rotation_preserved": rotation_preserved,
            "shield_minimum_raw_range": float(np.min(scan)),
            "shield_footprint_half_length": (
                self.shield_footprint_half_length
            ),
            "shield_footprint_half_width": (
                self.shield_footprint_half_width
            ),
            "shield_required_stop_distance": required_stop_distance,
            "shield_allowed_linear_action": allowed_linear_action,
        }
