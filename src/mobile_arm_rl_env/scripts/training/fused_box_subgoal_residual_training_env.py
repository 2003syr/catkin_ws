#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Subgoal-conditioned residual training around the verified rule tracker.

The regular fused residual environment trains against the final reach target.
This adapter instead freezes the same safe-waypoint command that the HRL
evaluator uses, builds the 66-D observation from that fixed joint subgoal, and
executes only a bounded five-dimensional residual around the rule teacher.
The residual therefore receives dense tracking feedback for the currently
active waypoint rather than a sparse end-of-episode signal.
"""

from __future__ import division

import math

import numpy as np
import rospy

from hrl.fused_low_level import FusedLowLevelState
from hrl.high_level_command import SubgoalType
from hrl.low_level_wrapper import JointSubgoalResidualLowLevelWrapper
from hrl.rule_high_policy import SafeWaypointHighPolicy
from hrl.subgoal_converter import FixedJointSubgoal
from training.fused_box_detour_training_env import (
    FusedBoxDetourTrainingEnv,
)


class _NoPolicy(object):
    """Placeholder policy; the PPO action is supplied directly by ``step``."""

    def predict(self, _observation):
        raise RuntimeError("the training environment owns the residual action")


class FusedBoxSubgoalResidualTrainingEnv(FusedBoxDetourTrainingEnv):
    """5-D residual PPO environment driven by the safe waypoint high policy."""

    ACTION_DIM = 5
    RESIDUAL_DIM = 5
    SUBGOAL_CONTEXT_CONTRACT = (
        JointSubgoalResidualLowLevelWrapper.SUBGOAL_CONTEXT_CONTRACT
    )

    def __init__(self, *args, **kwargs):
        if kwargs.get("enable_teacher", True) is False:
            raise ValueError(
                "FusedBoxSubgoalResidualTrainingEnv requires the rule teacher"
            )
        super(FusedBoxSubgoalResidualTrainingEnv, self).__init__(
            *args, **kwargs
        )
        if self.teacher is None:
            raise RuntimeError("subgoal residual environment needs a teacher")

        self.residual_base_scale = float(rospy.get_param(
            "~residual_base_scale", 0.05
        ))
        self.residual_cartesian_scale = float(rospy.get_param(
            "~residual_cartesian_scale", 0.005
        ))
        self.high_level_interval = int(rospy.get_param(
            "~subgoal_high_level_interval", 20
        ))
        self.subgoal_stable_cycles = int(rospy.get_param(
            "~subgoal_stable_cycles", 3
        ))
        self.subgoal_base_tolerance = float(rospy.get_param(
            "~subgoal_base_tolerance", 0.04
        ))
        self.subgoal_yaw_tolerance = float(rospy.get_param(
            "~subgoal_yaw_tolerance", 0.08
        ))
        self.subgoal_ee_tolerance = float(rospy.get_param(
            "~subgoal_ee_tolerance", 0.05
        ))
        self.training_detour_side = float(rospy.get_param(
            "~detour_side", 0.0
        ))
        if (
                self.residual_base_scale < 0.0
                or self.residual_cartesian_scale < 0.0
                or self.high_level_interval <= 0
                or self.subgoal_stable_cycles <= 0
                or self.subgoal_base_tolerance <= 0.0
                or self.subgoal_yaw_tolerance <= 0.0
                or self.subgoal_ee_tolerance <= 0.0):
            raise ValueError("invalid subgoal residual training parameters")

        self.subgoal_wrapper = JointSubgoalResidualLowLevelWrapper(
            _NoPolicy(),
            self.teacher,
            residual_base_scale=self.residual_base_scale,
            residual_cartesian_scale=self.residual_cartesian_scale,
            encode_subgoal_type=True,
        )
        self.high_policy = SafeWaypointHighPolicy(self)
        self.active_command = None
        self.active_path_index = -1
        self.subgoal_low_steps = 0
        self.subgoal_stable_count = 0
        self.pending_subgoal_refresh = False
        self.last_observation = None
        self.last_residual = np.zeros(self.RESIDUAL_DIM, dtype=np.float32)
        self.last_reward_terms = {}

    def reset(self, scenario=None, max_steps=None):
        if scenario is None:
            next_episode = int(self.episode_index + 1)
            configured_side = self.training_detour_side
            if abs(configured_side) >= 0.5:
                episode_side = (
                    1.0 if configured_side > 0.0 else -1.0
                )
            else:
                episode_side = 1.0 if next_episode % 2 else -1.0
            scenario = {
                "scenario_id": "subgoal_residual_box_{:05d}".format(
                    next_episode
                ),
                "category": "subgoal_tracking_box",
                "detour_side": episode_side,
            }
        else:
            scenario = dict(scenario)
            scenario.setdefault("category", "subgoal_tracking_box")

        super(FusedBoxSubgoalResidualTrainingEnv, self).reset(
            scenario=scenario,
            max_steps=max_steps,
        )
        self.active_command = None
        self.active_path_index = -1
        self.subgoal_low_steps = 0
        self.subgoal_stable_count = 0
        self.pending_subgoal_refresh = False
        self.last_residual.fill(0.0)
        self.last_reward_terms = {}
        self._refresh_subgoal("reset")
        self.last_observation = self._build_subgoal_observation()
        self.last_reset_info["subgoal_contract"] = (
            "safe_waypoint_rule_plus_bounded_residual"
        )
        self.last_reset_info["subgoal_high_level_interval"] = (
            self.high_level_interval
        )
        self.last_reset_info["subgoal_context_contract"] = (
            self.SUBGOAL_CONTEXT_CONTRACT
        )
        self.last_reset_info["subgoal_type"] = int(
            self.active_command.subgoal_type
        )
        self.last_reset_info["subgoal_type_name"] = (
            self.active_command.subgoal_type_name
        )
        return self.last_observation.copy()

    def step(self, residual):
        if self.last_observation is None or self.active_command is None:
            raise RuntimeError("reset must be called before step")

        if self.pending_subgoal_refresh:
            self._refresh_subgoal("pending")

        active_subgoal_type = int(self.active_command.subgoal_type)
        active_subgoal_type_name = self.active_command.subgoal_type_name

        residual = np.asarray(residual, dtype=np.float32)
        if residual.shape != (self.RESIDUAL_DIM,):
            raise ValueError(
                "residual must have shape (5,), got {}".format(
                    residual.shape
                )
            )
        residual = np.clip(residual, -1.0, 1.0)
        observation = self._build_subgoal_observation()
        previous_error = self.subgoal_wrapper.remaining(
            self.last_sensor_observation
        )
        arm_enabled = bool(
            np.linalg.norm(
                np.asarray(self.active_command.ee_goal, dtype=np.float64)
            ) > 1.0e-6
        )
        components = self.teacher.subgoal_residual_components(
            observation,
            residual,
            base_scale=self.residual_base_scale,
            cartesian_scale=self.residual_cartesian_scale,
            arm_enabled=arm_enabled,
        )
        candidate_action = np.asarray(
            components["action"], dtype=np.float32
        )
        arm_finish_fallback_used = False
        if str(components["phase"]) == "ARM_FINISH":
            arm_distance_before = float(np.linalg.norm(previous_error[3:6]))
            if (
                    arm_distance_before + 1.0e-4
                    < self._arm_finish_best_distance):
                self._arm_finish_best_distance = arm_distance_before
                self._arm_finish_stall_cycles = 0
            else:
                self._arm_finish_stall_cycles += 1
            if self._arm_finish_stall_cycles >= self._arm_finish_stall_limit:
                reference_action = self._arm_action_with_joint_reference(
                    components["sensor"],
                    {
                        "cartesian_velocity": components[
                            "cartesian_velocity"
                        ],
                    },
                )
                if reference_action is not None:
                    candidate_action[2:8] = reference_action
                    arm_finish_fallback_used = True
        else:
            self._arm_finish_stall_cycles = 0
            self._arm_finish_best_distance = float("inf")
        _, unused_reward, done, info = self.step_joint_subgoal(
            candidate_action
        )
        del unused_reward
        episode_success = bool(info.get("success", False))
        episode_timeout = bool(info.get("timeout", False))

        current_error = self.subgoal_wrapper.remaining(
            self.last_sensor_observation
        )
        previous_metrics = self._error_metrics(previous_error, arm_enabled)
        current_metrics = self._error_metrics(current_error, arm_enabled)
        stable = self._subgoal_is_stable(current_error, arm_enabled)
        if stable:
            self.subgoal_stable_count += 1
        else:
            self.subgoal_stable_count = 0
        self.subgoal_low_steps += 1

        path_index_changed = int(self._path.index) != int(
            self.active_path_index
        )
        stable_count_before_refresh = int(self.subgoal_stable_count)
        refresh_reason = ""
        if path_index_changed:
            refresh_reason = "path_advanced"
        elif self.subgoal_stable_count >= self.subgoal_stable_cycles:
            refresh_reason = "stable_tube"
        elif self.subgoal_low_steps >= self.high_level_interval:
            refresh_reason = "interval"
        subgoal_completed = bool(
            path_index_changed
            or self.subgoal_stable_count >= self.subgoal_stable_cycles
            or episode_success
        )
        subgoal_timed_out = bool(
            episode_timeout
            or (
                refresh_reason == "interval" and not subgoal_completed
            )
        )
        subgoal_option_done = bool(
            done or subgoal_completed or subgoal_timed_out
        )
        if refresh_reason and not done:
            # Refresh before returning the next observation.  Otherwise the
            # PPO actor would receive one step of stale subgoal features after
            # the path advances, which is exactly the tracking mismatch this
            # environment is intended to eliminate.
            self._refresh_subgoal(refresh_reason)

        safe_action = np.asarray(
            info.get("safe_fused_action", candidate_action),
            dtype=np.float64,
        )
        projection = safe_action - candidate_action
        residual_change = residual - self.last_residual
        collision = bool(info.get("collision", False))
        tf_ok = bool(info.get("tf_ok", True))
        timeout = episode_timeout
        success = episode_success
        reward_profile = self._reward_profile(active_subgoal_type)

        reward_terms = {
            # Dense tracking terms are the main learning signal.  The arm
            # term is disabled while the rule upper layer intentionally holds
            # the arm during the detour.
            "base_progress": reward_profile["base_progress"] * (
                previous_metrics["base"] - current_metrics["base"]
            ),
            "yaw_progress": reward_profile["yaw_progress"] * (
                previous_metrics["yaw"] - current_metrics["yaw"]
            ),
            "ee_progress": reward_profile["ee_progress"] * (
                previous_metrics["ee"] - current_metrics["ee"]
            ) if arm_enabled else 0.0,
            "base_error": -reward_profile["base_error"] * (
                current_metrics["base"]
            ),
            "yaw_error": -reward_profile["yaw_error"] * (
                current_metrics["yaw"]
            ),
            "ee_error": -reward_profile["ee_error"] * current_metrics["ee"]
            if arm_enabled else 0.0,
            "subgoal_in_tube": 0.05 if stable else 0.0,
            "subgoal_completed": (
                reward_profile["completion"] if subgoal_completed else 0.0
            ),
            "subgoal_timeout": (
                -reward_profile["option_timeout"]
                if subgoal_timed_out else 0.0
            ),
            "residual_effort": -0.10 * float(np.dot(residual, residual)),
            "residual_smoothness": -0.25 * float(
                np.dot(residual_change, residual_change)
            ),
            "safety_projection": -reward_profile["safety"] * float(
                np.dot(projection, projection)
            ),
            "time": -0.002,
            "success": 30.0 if success else 0.0,
            "collision": -reward_profile["collision"] if collision else 0.0,
            "tf_error": -20.0 if not tf_ok else 0.0,
            "timeout": -10.0 if timeout else 0.0,
        }
        reward = float(sum(reward_terms.values()))
        observation = self._build_subgoal_observation()
        info = dict(info)
        info.update({
            "residual": residual.copy(),
            "nominal_action": np.asarray(
                components["nominal_action"], dtype=np.float32
            ).copy(),
            "candidate_action": candidate_action.copy(),
            "projection": projection.astype(np.float32),
            "residual_phase": str(components["phase"]),
            "joint_limit_diagnostics": dict(
                components.get("joint_limit_diagnostics", {})
            ),
            "arm_finish_fallback_used": bool(
                arm_finish_fallback_used
            ),
            "arm_finish_stall_cycles": int(
                self._arm_finish_stall_cycles
            ),
            "subgoal_error": current_error.copy(),
            "subgoal_base_distance": float(current_metrics["base"]),
            "subgoal_yaw_error": float(current_metrics["yaw"]),
            "subgoal_ee_distance": float(current_metrics["ee"]),
            "subgoal_metric": float(self._tracking_metric(
                current_metrics, arm_enabled
            )),
            "subgoal_stable": bool(stable),
            "subgoal_stable_count": stable_count_before_refresh,
            "subgoal_completed": bool(subgoal_completed),
            "subgoal_timed_out": bool(subgoal_timed_out),
            "subgoal_option_done": bool(subgoal_option_done),
            "subgoal_refresh_reason": refresh_reason,
            "subgoal_type": active_subgoal_type,
            "subgoal_type_name": active_subgoal_type_name,
            "subgoal_type_one_hot": SubgoalType.one_hot(
                active_subgoal_type
            ),
            "next_subgoal_type": int(self.active_command.subgoal_type),
            "next_subgoal_type_name": self.active_command.subgoal_type_name,
            "subgoal_context_contract": self.SUBGOAL_CONTEXT_CONTRACT,
            "subgoal_path_index": int(self._path.index),
            "subgoal_active_path_index": int(self.active_path_index),
            "subgoal_arm_enabled": bool(arm_enabled),
            "ee_distance": float(np.linalg.norm(
                self.last_sensor_observation[0:3]
                - self.last_sensor_observation[6:9]
            )),
            "reward_terms": reward_terms,
            "success": success,
            "collision": collision,
            "tf_ok": tf_ok,
            "timeout": timeout,
        })
        self.last_observation = observation.copy()
        self.last_residual = (
            np.zeros(self.RESIDUAL_DIM, dtype=np.float32)
            if subgoal_option_done else residual.copy()
        )
        self.last_reward_terms = dict(reward_terms)
        return observation, reward, bool(done), info

    def nominal_action(self):
        """Return the current rule action for diagnostics/reset responses."""
        if self.active_command is None:
            return np.zeros(8, dtype=np.float32)
        observation = self._build_subgoal_observation()
        arm_enabled = bool(
            np.linalg.norm(
                np.asarray(self.active_command.ee_goal, dtype=np.float64)
            ) > 1.0e-6
        )
        return np.asarray(
            self.teacher.subgoal_components(
                observation,
                arm_enabled=arm_enabled,
            )["nominal_action"],
            dtype=np.float32,
        )

    def _refresh_subgoal(self, reason):
        del reason
        high_observation = self._build_high_observation()
        self.active_command = self.high_policy.predict(high_observation)
        self.subgoal_wrapper.begin_subgoal(
            self.active_command,
            self.last_sensor_observation,
        )
        self.active_path_index = int(self._path.index)
        self.subgoal_low_steps = 0
        self.subgoal_stable_count = 0
        self.pending_subgoal_refresh = False

    def _build_subgoal_observation(self):
        return self.subgoal_wrapper.build_observation(
            self.last_sensor_observation
        )

    def _build_high_observation(self):
        sensor = np.asarray(self.last_sensor_observation, dtype=np.float64)
        base_pose = self.subgoal_wrapper.base_pose(sensor)
        ee_position = self.subgoal_wrapper.ee_position(sensor)
        path = self._path
        final_heading_error = 0.0
        if path.final_waypoint_active and path.final_position_reached(
                base_pose[0:2]):
            final_heading_error = path.final_yaw_error(base_pose[2])
        ee_error_world = self.reach_env.target_position - ee_position
        ee_error_body = FixedJointSubgoal.rotate_xy(
            ee_error_world[0:2],
            -base_pose[2],
        )
        return {
            "base_pose_world": base_pose.astype(np.float32),
            "final_heading_error": float(final_heading_error),
            "final_ee_error_body": np.asarray([
                ee_error_body[0],
                ee_error_body[1],
                ee_error_world[2],
            ], dtype=np.float32),
        }

    def _subgoal_is_stable(self, error, arm_enabled):
        error = np.asarray(error, dtype=np.float64)
        return bool(
            np.linalg.norm(error[0:2]) <= self.subgoal_base_tolerance
            and abs(float(error[2])) <= self.subgoal_yaw_tolerance
            and (
                not arm_enabled
                or np.linalg.norm(error[3:6]) <= self.subgoal_ee_tolerance
            )
        )

    @staticmethod
    def _error_metrics(error, arm_enabled):
        error = np.asarray(error, dtype=np.float64)
        return {
            "base": float(np.linalg.norm(error[0:2])),
            "yaw": abs(float(error[2])),
            "ee": float(np.linalg.norm(error[3:6])) if arm_enabled else 0.0,
        }

    @staticmethod
    def _tracking_metric(metrics, arm_enabled):
        value = float(metrics["base"] + 0.25 * metrics["yaw"])
        if arm_enabled:
            value += float(metrics["ee"])
        return value

    @staticmethod
    def _reward_profile(subgoal_type):
        """Return explicit option-level shaping for the three intents."""
        subgoal_type = SubgoalType.validate(subgoal_type)
        profiles = {
            SubgoalType.DIRECT: {
                "base_progress": 6.0,
                "yaw_progress": 1.5,
                "ee_progress": 0.0,
                "base_error": 0.15,
                "yaw_error": 0.04,
                "ee_error": 0.0,
                "completion": 1.0,
                "option_timeout": 0.5,
                "safety": 0.75,
                "collision": 30.0,
            },
            SubgoalType.DETOUR: {
                "base_progress": 6.0,
                "yaw_progress": 3.0,
                "ee_progress": 0.0,
                "base_error": 0.20,
                "yaw_error": 0.08,
                "ee_error": 0.0,
                "completion": 1.5,
                "option_timeout": 1.0,
                "safety": 1.50,
                "collision": 40.0,
            },
            SubgoalType.TERMINAL: {
                "base_progress": 4.0,
                "yaw_progress": 3.0,
                "ee_progress": 10.0,
                "base_error": 0.25,
                "yaw_error": 0.10,
                "ee_error": 0.75,
                "completion": 3.0,
                "option_timeout": 1.5,
                "safety": 1.00,
                "collision": 35.0,
            },
        }
        return dict(profiles[subgoal_type])
