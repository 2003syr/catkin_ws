#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Event-driven high-level options over a frozen fused low controller."""

from __future__ import division

import math

import numpy as np

from hrl.fused_low_level import FusedLowLevelState
from hrl.high_level_command import (
    HighLevelOption,
    JointHighLevelCommand,
    RouteSide,
    SubgoalType,
)
from hrl.high_level_reward import HighLevelReward
from hrl.rule_based_planar_subgoal import (
    RuleBasedPlanarSubgoalGenerator,
    project_path_corridor_subgoal,
)


class HighLevelEnv(object):
    """Minimal two-level environment with a frozen low-level policy."""

    SCAN_DIM = 36
    OBS_DIM = 82
    # The basic learner must observe the navigation state used by its rule
    # teacher.  The original 78-D vector omitted the active path waypoint and
    # progress, even though ``SafeWaypointHighPolicy`` used both.  That made
    # the imitation problem partially observable: the same student vector
    # could have two different teacher labels.  Eight bounded path-context
    # values make the upper-level process Markov without adding a route action.
    BASIC_PATH_CONTEXT_DIM = 8
    BASIC_OBS_DIM = 86
    ACTION_DIM = 6

    def __init__(
            self,
            low_environment,
            low_wrapper,
            high_level_interval=20,
            reward_model=None,
            scan_clip=10.0,
            action_limits=None,
            base_subgoal_tolerance=0.04,
            yaw_subgoal_tolerance=0.08,
            ee_subgoal_tolerance=0.05,
            subgoal_stable_cycles=3,
            option_min_low_steps=5,
            option_stall_steps=0,
            option_minimum_progress=0.002,
            option_safety_window=0,
            option_safety_rate_threshold=1.0,
            enable_route_options=True,
            terminal_option_contract=False,
            terminal_ee_step=0.08,
            terminal_student_blend=0.25,
            terminal_rotate_minimum_action=0.20,
            terminal_rotate_recovery_action=0.80,
            terminal_rotate_progress_epsilon=0.001,
            terminal_rotate_recovery_steps=8,
            terminal_rotate_stall_steps=20,
            detour_feasibility_guard=False,
            detour_corridor_half_width=0.08,
            detour_minimum_progress=0.05,
            detour_clearance_margin=0.04):
        self.low_environment = low_environment
        self.low_wrapper = low_wrapper
        self.high_level_interval = int(high_level_interval)
        self.reward_model = reward_model or HighLevelReward()
        self.scan_clip = float(scan_clip)
        self.action_limits = JointHighLevelCommand._limits(action_limits)
        self.base_subgoal_tolerance = float(base_subgoal_tolerance)
        self.yaw_subgoal_tolerance = float(yaw_subgoal_tolerance)
        self.ee_subgoal_tolerance = float(ee_subgoal_tolerance)
        self.subgoal_stable_cycles = int(subgoal_stable_cycles)
        self.option_min_low_steps = int(option_min_low_steps)
        self.option_stall_steps = int(option_stall_steps)
        self.option_minimum_progress = float(option_minimum_progress)
        self.option_safety_window = int(option_safety_window)
        self.option_safety_rate_threshold = float(
            option_safety_rate_threshold
        )
        self.enable_route_options = bool(enable_route_options)
        self.terminal_option_contract = bool(terminal_option_contract)
        self.terminal_ee_step = float(terminal_ee_step)
        self.terminal_student_blend = float(terminal_student_blend)
        self.terminal_rotate_minimum_action = float(
            terminal_rotate_minimum_action
        )
        self.terminal_rotate_recovery_action = float(
            terminal_rotate_recovery_action
        )
        self.terminal_rotate_progress_epsilon = float(
            terminal_rotate_progress_epsilon
        )
        self.terminal_rotate_recovery_steps = int(
            terminal_rotate_recovery_steps
        )
        self.terminal_rotate_stall_steps = int(
            terminal_rotate_stall_steps
        )
        self.detour_feasibility_guard = bool(detour_feasibility_guard)
        self.detour_corridor_half_width = float(
            detour_corridor_half_width
        )
        self.detour_minimum_progress = float(detour_minimum_progress)
        self.detour_clearance_margin = float(detour_clearance_margin)
        # The basic hierarchy predicts only a three-way semantic stage and a
        # physical 6-D subgoal.  Route-option memory is intentionally absent.
        # Keep the existing 82-D contract as the default so legacy evaluators
        # remain usable while the minimal two-layer learner uses 78 dimensions.
        if not self.enable_route_options:
            self.OBS_DIM = self.BASIC_OBS_DIM
        if self.high_level_interval <= 0:
            raise ValueError("high_level_interval must be positive")
        if self.scan_clip <= 0.0:
            raise ValueError("scan_clip must be positive")
        if (
                self.base_subgoal_tolerance <= 0.0
                or self.yaw_subgoal_tolerance <= 0.0
                or self.ee_subgoal_tolerance <= 0.0
                or self.subgoal_stable_cycles <= 0):
            raise ValueError("invalid subgoal completion tolerances")
        if (
                self.option_min_low_steps < 0
                or self.option_stall_steps < 0
                or self.option_minimum_progress < 0.0
                or self.option_safety_window < 0
                or not 0.0 <= self.option_safety_rate_threshold <= 1.0):
            raise ValueError("invalid event-driven option parameters")
        if (
                self.terminal_ee_step <= 0.0
                or not 0.0 <= self.terminal_student_blend <= 1.0):
            raise ValueError("invalid terminal option parameters")
        if (
                not 0.0 <= self.terminal_rotate_minimum_action <= 1.0
                or not 0.0 <= self.terminal_rotate_recovery_action <= 1.0
                or self.terminal_rotate_recovery_action
                < self.terminal_rotate_minimum_action
                or self.terminal_rotate_progress_epsilon < 0.0
                or self.terminal_rotate_recovery_steps <= 0
                or self.terminal_rotate_stall_steps
                < self.terminal_rotate_recovery_steps):
            raise ValueError("invalid terminal rotation recovery parameters")
        if (
                self.detour_corridor_half_width < 0.0
                or self.detour_minimum_progress < 0.0
                or self.detour_clearance_margin < 0.0):
            raise ValueError("invalid detour feasibility parameters")
        self.previous_base_subgoal_error = np.zeros(3, dtype=np.float32)
        self.previous_ee_subgoal_error = np.zeros(3, dtype=np.float32)
        self.previous_high_action = np.zeros(6, dtype=np.float32)
        self.latched_route_side = RouteSide.NONE
        self.previous_option = HighLevelOption.DIRECT
        self.previous_subgoal_type = SubgoalType.DIRECT
        self.last_option_duration = 0.0
        self.last_option_progress = 0.0
        self.last_option_safety_rate = 0.0
        self.last_option_stalled = False
        self.terminal_option_latched = False
        self.terminal_fixed_ee_goal_world = None
        self._last_terminal_guard = {}
        self._last_detour_guard = {}
        self._terminal_rotation_state = self._new_terminal_rotation_state()
        self.high_step = 0
        self.episode_done = False
        self.last_info = {}

    def set_terminal_student_blend(self, value):
        """Set the TERMINAL student authority for the next episode.

        The environment server calls this only at reset boundaries.  Keeping
        the value mutable lets PPO run a staged curriculum without restarting
        Gazebo, while preserving a fixed blend throughout each episode.
        """
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "terminal_student_blend must be finite and in [0, 1]"
            )
        self.terminal_student_blend = value

    def reset(self, scenario=None, max_steps=None):
        self.low_environment.reset(
            scenario=scenario,
            max_steps=max_steps,
        )
        self.previous_base_subgoal_error.fill(0.0)
        self.previous_ee_subgoal_error.fill(0.0)
        self.previous_high_action.fill(0.0)
        self.latched_route_side = RouteSide.NONE
        self.previous_option = HighLevelOption.DIRECT
        self.previous_subgoal_type = SubgoalType.DIRECT
        self.last_option_duration = 0.0
        self.last_option_progress = 0.0
        self.last_option_safety_rate = 0.0
        self.last_option_stalled = False
        self.terminal_option_latched = False
        self.terminal_fixed_ee_goal_world = None
        self._last_terminal_guard = {}
        self._last_detour_guard = {}
        self._terminal_rotation_state = self._new_terminal_rotation_state()
        self.high_step = 0
        self.episode_done = False
        self.last_info = {}
        return self.high_observation()

    def step(self, high_action):
        if self.episode_done:
            raise RuntimeError("episode is done; call reset before step")
        command = (
            high_action
            if isinstance(high_action, JointHighLevelCommand)
            else JointHighLevelCommand.from_action(
                high_action,
                limits=self.action_limits,
            )
        )
        requested_subgoal_type = int(command.subgoal_type)
        requested_high_action = command.to_action(
            limits=self.action_limits
        )
        requested_high_command = command.subgoal.copy()
        if self.enable_route_options:
            self._apply_route_decision(command)
        sensor = self._sensor()
        self.guard_detour_command(command, sensor=sensor)
        detour_guard = dict(self._last_detour_guard)
        routed_high_command = command.subgoal.copy()
        terminal_base_hold = self.guard_high_command(
            command, sensor=sensor
        )
        terminal_guard = dict(self._last_terminal_guard)
        executed_high_command = command.subgoal.copy()
        route_command_projection = (
            routed_high_command - requested_high_command
        )
        guard_command_projection = (
            executed_high_command - routed_high_command
        )
        high_command_projection = (
            executed_high_command - requested_high_command
        )
        self.low_wrapper.begin_subgoal(command, sensor)
        start_error = self.low_wrapper.remaining(sensor)
        start_error_measure = self._subgoal_error_measure(
            start_error, command
        )
        start_goal_distance = self._final_ee_distance(sensor)
        start_path_remaining = self._path_remaining(sensor)
        start_final_base_distance, start_final_yaw_error = (
            self._final_base_errors(sensor)
        )

        done = False
        low_info = {}
        low_steps = 0
        low_actions = []
        safe_low_actions = []
        low_residuals = []
        safety_steps = 0
        overlap_steps = 0
        safety_reasons = {}
        subgoal_stable_count = 0
        subgoal_done = False
        option_termination = "horizon"
        option_stall_count = 0
        best_error_measure = float(start_error_measure)
        option_control_stage = self._terminal_control_stage(command)
        start_control_stage = str(option_control_stage)
        option_stage_transitions = 0
        option_safety_history = []
        rotate_snapshot = self._terminal_rotation_snapshot()
        for _ in range(self.high_level_interval):
            sensor = self._sensor()
            low_action = self.low_wrapper.predict(
                sensor,
                ee_rotation_world=self._ee_rotation_world(sensor),
            )
            low_action = self._guard_terminal_rotation_action(
                low_action,
                command,
                sensor,
            )
            low_actions.append(np.asarray(low_action).copy())
            if hasattr(self.low_wrapper, "last_residual"):
                low_residuals.append(
                    np.asarray(
                        self.low_wrapper.last_residual,
                        dtype=np.float32,
                    ).copy()
                )
            _, _, done, low_info = (
                self.low_environment.step_joint_subgoal(low_action)
            )
            safe_action = np.asarray(
                low_info.get("safe_fused_action", low_action),
                dtype=np.float32,
            )
            safe_low_actions.append(safe_action.copy())
            filter_delta = np.max(np.abs(
                safe_action - np.asarray(low_action, dtype=np.float32)
            ))
            safety_active = bool(filter_delta > 0.01)
            safety_steps += int(safety_active)
            option_safety_history.append(safety_active)
            if self.option_safety_window > 0:
                option_safety_history = option_safety_history[
                    -self.option_safety_window:
                ]
            for safety_reason in active_safety_reasons(
                    low_info.get("safety_info", {})):
                safety_reasons[safety_reason] = (
                    safety_reasons.get(safety_reason, 0) + 1
                )
            base_active = bool(
                abs(float(safe_action[0])) > 0.05
                or abs(float(safe_action[1])) > 0.05
            )
            arm_active = bool(
                np.max(np.abs(safe_action[2:8])) > 0.02
            )
            overlap_steps += int(base_active and arm_active)
            low_steps += 1
            after_sensor = self._sensor()
            rotate_stalled = self._update_terminal_rotation_progress(
                command,
                sensor,
                after_sensor,
                low_action,
                safe_action,
            )
            if done:
                option_termination = "environment_done"
                break
            if rotate_stalled:
                option_termination = "rotate_stalled"
                break
            current_error = self.low_wrapper.remaining(after_sensor)
            if self._subgoal_is_stable(current_error, command):
                subgoal_stable_count += 1
            else:
                subgoal_stable_count = 0
            if subgoal_stable_count >= self.subgoal_stable_cycles:
                subgoal_done = True
                option_termination = "subgoal_done"
                break
            current_error_measure = self._subgoal_error_measure(
                current_error, command
            )
            current_control_stage = self._terminal_control_stage(command)
            if current_control_stage != option_control_stage:
                # Each deterministic TERMINAL substate has a different error
                # coordinate.  Comparing ARM_REACH distance with the previous
                # BASE_ROTATE yaw error creates false stalls exactly when the
                # controller has made valid progress into a new phase.
                option_control_stage = current_control_stage
                option_stage_transitions += 1
                best_error_measure = float(current_error_measure)
                option_stall_count = 0
                if current_control_stage != "BASE_ROTATE":
                    self._clear_terminal_rotation_streak()
                continue
            if (
                    best_error_measure - current_error_measure
                    >= self.option_minimum_progress):
                best_error_measure = float(current_error_measure)
                option_stall_count = 0
            else:
                option_stall_count += 1
            if low_steps < self.option_min_low_steps:
                continue
            if (
                    self.option_stall_steps > 0
                    and option_stall_count >= self.option_stall_steps):
                option_termination = "stalled"
                break
            if (
                    self.option_safety_window > 0
                    and len(option_safety_history)
                    >= self.option_safety_window
                    and float(sum(option_safety_history))
                    / float(len(option_safety_history))
                    >= self.option_safety_rate_threshold):
                option_termination = "safety_risk"
                break

        end_sensor = self._sensor()
        end_error = self.low_wrapper.remaining(end_sensor)
        end_goal_distance = self._final_ee_distance(end_sensor)
        end_path_remaining = self._path_remaining(end_sensor)
        end_final_base_distance, end_final_yaw_error = (
            self._final_base_errors(end_sensor)
        )
        success = bool(low_info.get("success", False))
        collision = bool(low_info.get("collision", False))
        timeout = bool(low_info.get("timeout", False))
        reward, reward_info = self.reward_model.calculate(
            start_goal_distance=start_goal_distance,
            end_goal_distance=end_goal_distance,
            start_subgoal_error=start_error,
            end_subgoal_error=end_error,
            success=success,
            collision=collision,
            timeout=timeout,
            start_path_remaining=start_path_remaining,
            end_path_remaining=end_path_remaining,
            start_final_base_distance=start_final_base_distance,
            end_final_base_distance=end_final_base_distance,
            start_final_yaw_error=start_final_yaw_error,
            end_final_yaw_error=end_final_yaw_error,
            terminal_stage=(
                int(command.subgoal_type) == SubgoalType.TERMINAL
            ),
            low_steps=low_steps,
            option_stalled=(
                option_termination in (
                    "stalled", "safety_risk", "rotate_stalled"
                )
            ),
            invalid_terminal=bool(terminal_guard.get(
                "invalid_terminal", False
            )),
        )
        self.previous_base_subgoal_error = end_error[0:3].copy()
        self.previous_ee_subgoal_error = end_error[3:6].copy()
        self.previous_high_action = command.to_action(
            limits=self.action_limits
        )
        self.previous_subgoal_type = int(command.subgoal_type)
        if self.enable_route_options:
            self.previous_option = int(command.option)
        self.last_option_duration = float(low_steps) / float(
            max(self.high_level_interval, 1)
        )
        end_error_measure = self._subgoal_error_measure(end_error, command)
        if (
                int(command.subgoal_type) == SubgoalType.TERMINAL
                and start_control_stage != option_control_stage):
            stage_rank = {
                "BASE_TRANSLATE": 0,
                "BASE_ROTATE": 1,
                "ARM_REACH": 2,
                "ALIGNED": 3,
            }
            self.last_option_progress = float(np.clip(
                float(
                    stage_rank.get(option_control_stage, 0)
                    - stage_rank.get(start_control_stage, 0)
                ) / 3.0,
                -1.0,
                1.0,
            ))
        else:
            self.last_option_progress = float(np.clip(
                (start_error_measure - end_error_measure)
                / max(start_error_measure, 1.0e-6),
                -1.0,
                1.0,
            ))
        self.last_option_safety_rate = float(safety_steps) / float(
            max(low_steps, 1)
        )
        self.last_option_stalled = bool(
            option_termination in (
                "stalled", "safety_risk", "rotate_stalled"
            )
        )
        if (
                int(command.subgoal_type) == SubgoalType.TERMINAL
                and (
                    subgoal_done
                    or done
                    or option_termination in (
                        "stalled", "safety_risk", "rotate_stalled"
                    )
                )):
            # A terminal target is persistent across ordinary high-level
            # horizons, but a completed/stalled target is replanned toward
            # the same final EE pose on the next upper-level decision.
            self.terminal_fixed_ee_goal_world = None
        self.high_step += 1
        self.episode_done = bool(done)
        high_observation = self.high_observation()
        low_action_mean_abs = np.zeros(8, dtype=np.float32)
        safe_low_action_mean_abs = np.zeros(8, dtype=np.float32)
        low_safety_projection_mean_abs = np.zeros(8, dtype=np.float32)
        low_safety_projection_max_abs = np.zeros(8, dtype=np.float32)
        low_safety_projection_max = 0.0
        if low_actions:
            low_action_array = np.asarray(low_actions, dtype=np.float32)
            low_action_mean_abs = np.mean(
                np.abs(low_action_array), axis=0
            )
        if safe_low_actions:
            safe_low_action_array = np.asarray(
                safe_low_actions, dtype=np.float32
            )
            safe_low_action_mean_abs = np.mean(
                np.abs(safe_low_action_array), axis=0
            )
        if low_actions and safe_low_actions:
            safety_projection = np.abs(
                safe_low_action_array - low_action_array
            )
            low_safety_projection_mean_abs = np.mean(
                safety_projection, axis=0
            )
            low_safety_projection_max_abs = np.max(
                safety_projection, axis=0
            )
            low_safety_projection_max = float(np.max(safety_projection))
        info = dict(low_info)
        rotate_diagnostics = self._terminal_rotation_diagnostics(
            rotate_snapshot
        )
        info.update({
            "high_step": int(self.high_step),
            "low_steps": int(low_steps),
            "high_command": executed_high_command.copy(),
            "high_action": self.previous_high_action.copy(),
            "requested_high_action": requested_high_action.copy(),
            "requested_high_command": requested_high_command.copy(),
            "routed_high_command": routed_high_command.copy(),
            "route_command_projection": route_command_projection.copy(),
            "guard_command_projection": guard_command_projection.copy(),
            "high_command_projection": high_command_projection.copy(),
            "base_command_projection": float(np.linalg.norm(
                high_command_projection[0:3]
            )),
            "ee_command_projection": float(np.linalg.norm(
                high_command_projection[3:6]
            )),
            "requested_subgoal_type": int(requested_subgoal_type),
            "requested_subgoal_type_name": SubgoalType.name(
                requested_subgoal_type
            ),
            "subgoal_type": int(command.subgoal_type),
            "subgoal_type_name": command.subgoal_type_name,
            "subgoal_type_one_hot": command.subgoal_type_one_hot(),
            "previous_subgoal_type": int(self.previous_subgoal_type),
            "fixed_base_goal_world": (
                self.low_wrapper.fixed_subgoal.base_goal_world.copy()
            ),
            "fixed_base_yaw_world": float(
                self.low_wrapper.fixed_subgoal.base_yaw_world
            ),
            "fixed_ee_goal_world": (
                self.low_wrapper.fixed_subgoal.ee_goal_world.copy()
            ),
            "start_subgoal_error": start_error.copy(),
            "end_subgoal_error": end_error.copy(),
            "high_reward": dict(reward_info),
            "low_action_mean_abs": low_action_mean_abs,
            "safe_low_action_mean_abs": safe_low_action_mean_abs,
            "low_safety_projection_mean_abs": (
                low_safety_projection_mean_abs
            ),
            "low_safety_projection_max_abs": (
                low_safety_projection_max_abs
            ),
            "low_safety_projection_max": low_safety_projection_max,
            "low_residual_mean_abs": (
                np.mean(np.abs(np.asarray(low_residuals)), axis=0)
                if low_residuals else np.zeros(5, dtype=np.float32)
            ),
            "safety_steps": int(safety_steps),
            "safety_reasons": dict(safety_reasons),
            "overlap_steps": int(overlap_steps),
            "subgoal_done": bool(subgoal_done),
            "subgoal_stable_count": int(subgoal_stable_count),
            "option_termination": str(option_termination),
            "option_stall_count": int(option_stall_count),
            "terminal_control_stage": str(option_control_stage),
            "terminal_control_stage_transitions": int(
                option_stage_transitions
            ),
            "option_progress": float(self.last_option_progress),
            "option_duration_fraction": float(self.last_option_duration),
            "option_safety_rate": float(self.last_option_safety_rate),
            "subgoal_tolerances": {
                "base": float(self.base_subgoal_tolerance),
                "yaw": float(self.yaw_subgoal_tolerance),
                "ee": float(self.ee_subgoal_tolerance),
                "stable_cycles": int(self.subgoal_stable_cycles),
            },
            "terminal_base_hold": bool(terminal_base_hold),
            "terminal_option_contract": bool(
                self.terminal_option_contract
            ),
            "terminal_student_blend": float(
                self.terminal_student_blend
            ),
            "terminal_option_latched": bool(
                self.terminal_option_latched
            ),
            "terminal_stage_forced": bool(terminal_guard.get(
                "stage_forced", False
            )),
            "terminal_eligible": bool(terminal_guard.get(
                "terminal_eligible", False
            )),
            "invalid_terminal": bool(terminal_guard.get(
                "invalid_terminal", False
            )),
            "terminal_ee_projection": float(terminal_guard.get(
                "ee_projection", 0.0
            )),
            "terminal_ee_requested_norm": float(terminal_guard.get(
                "ee_requested_norm", 0.0
            )),
            "terminal_ee_executed_norm": float(terminal_guard.get(
                "ee_executed_norm", 0.0
            )),
            "terminal_ee_alignment_cosine": float(terminal_guard.get(
                "ee_alignment_cosine", 0.0
            )),
            "terminal_ee_student_alignment_cosine": float(
                terminal_guard.get(
                    "ee_student_alignment_cosine", 0.0
                )
            ),
            "terminal_ee_executed_alignment_cosine": float(
                terminal_guard.get(
                    "ee_executed_alignment_cosine",
                    terminal_guard.get("ee_alignment_cosine", 0.0),
                )
            ),
            "terminal_ee_norm_clipped": bool(terminal_guard.get(
                "ee_norm_clipped", False
            )),
            "terminal_ee_progress_fallback": bool(terminal_guard.get(
                "ee_progress_fallback", False
            )),
            "terminal_ee_persistent_target": bool(terminal_guard.get(
                "ee_persistent_target", False
            )),
            "detour_guard_applied": bool(detour_guard.get(
                "applied", False
            )),
            "detour_guard_reason": str(detour_guard.get(
                "reason", "inactive"
            )),
            "detour_guard_projection": float(detour_guard.get(
                "projection", 0.0
            )),
            "detour_guard_requested_along": float(detour_guard.get(
                "requested_along", 0.0
            )),
            "detour_guard_executed_along": float(detour_guard.get(
                "executed_along", 0.0
            )),
            "detour_guard_requested_lateral": float(detour_guard.get(
                "requested_lateral", 0.0
            )),
            "detour_guard_executed_lateral": float(detour_guard.get(
                "executed_lateral", 0.0
            )),
            "detour_guard_scan_min": float(detour_guard.get(
                "scan_min", self.scan_clip
            )),
            "detour_guard_corridor_feasible": bool(detour_guard.get(
                "corridor_feasible", True
            )),
            "terminal_rotate_steps": int(rotate_diagnostics["steps"]),
            "terminal_rotate_guarded_steps": int(
                rotate_diagnostics["guarded_steps"]
            ),
            "terminal_rotate_recovery_steps": int(
                rotate_diagnostics["recovery_steps"]
            ),
            "terminal_rotate_blocked_steps": int(
                rotate_diagnostics["blocked_steps"]
            ),
            "terminal_rotate_wrong_direction_steps": int(
                rotate_diagnostics["wrong_direction_steps"]
            ),
            "terminal_rotate_progress": float(
                rotate_diagnostics["progress"]
            ),
            "terminal_rotate_no_progress_streak": int(
                rotate_diagnostics["no_progress_streak"]
            ),
            "terminal_rotate_stall_events": int(
                rotate_diagnostics["stall_events"]
            ),
        })
        if self.enable_route_options:
            info.update({
                "option": int(command.option),
                "option_name": command.option_name,
                "option_one_hot": command.option_one_hot(),
                "route_side": int(command.route_side),
                "route_side_name": command.route_side_name,
                "route_side_one_hot": command.route_side_one_hot(),
                "latched_route_side": int(self.latched_route_side),
            })
        teacher_components = getattr(
            self.low_wrapper,
            "last_components",
            {},
        )
        if teacher_components:
            info.update({
                "low_teacher_phase": str(
                    teacher_components.get("phase", "unknown")
                ),
                "low_teacher_bearing_error": float(
                    teacher_components.get("bearing_error", 0.0)
                ),
                "low_teacher_base_distance": float(
                    teacher_components.get("base_distance", 0.0)
                ),
            })
        self.last_info = info
        return high_observation, float(reward), bool(done), info

    def _subgoal_is_stable(self, remaining, command):
        """Check the live fixed-subgoal error for early transition.

        A high-level action is an SMDP option.  It should finish as soon as
        its fixed target has been held inside the tracking tube, rather than
        blindly executing the remainder of a long high-level interval.  The
        arm part is ignored when the command intentionally holds the arm at
        zero during a detour waypoint.
        """
        remaining = np.asarray(remaining, dtype=np.float64)
        if remaining.shape != (6,):
            raise ValueError(
                "remaining subgoal must have shape (6,), got {}".format(
                    remaining.shape
                )
            )
        if float(np.linalg.norm(remaining[0:2])) > (
                self.base_subgoal_tolerance):
            return False
        if abs(float(remaining[2])) > self.yaw_subgoal_tolerance:
            return False
        arm_enabled = bool(
            int(getattr(
                command,
                "subgoal_type",
                SubgoalType.TERMINAL,
            )) == SubgoalType.TERMINAL
            and
            np.linalg.norm(
                np.asarray(command.ee_goal, dtype=np.float64)
            ) > 1.0e-6
        )
        return bool(
            not arm_enabled
            or float(np.linalg.norm(remaining[3:6]))
            <= self.ee_subgoal_tolerance
        )

    def high_observation(self):
        sensor = self._sensor()
        base_pose = self.low_wrapper.base_pose(sensor)
        ee_position = self.low_wrapper.ee_position(sensor)
        final_ee_goal = np.asarray(sensor[0:3], dtype=np.float64)
        ee_delta_world = final_ee_goal - base_pose_to_xyz(base_pose)
        target_relative_base = ee_delta_world.copy()
        target_relative_base[0:2] = rotate_xy(
            ee_delta_world[0:2],
            -base_pose[2],
        )
        target_relative_ee = np.asarray(sensor[3:6], dtype=np.float64)
        final_base_goal, final_base_heading = self._final_base_pose()
        final_base_error_body = rotate_xy(
            final_base_goal - base_pose[0:2],
            -base_pose[2],
        )
        heading_error = wrap_angle(final_base_heading - base_pose[2])
        arm_q = np.asarray(sensor[15:21], dtype=np.float64)
        arm_margin = np.asarray(sensor[35:41], dtype=np.float64)
        ee_distance = self._final_ee_distance(sensor)
        base_distance = float(np.linalg.norm(
            final_base_goal - base_pose[0:2]
        ))
        scan = self._scan()
        decision_memory = (
            np.concatenate((
                RouteSide.one_hot(self.latched_route_side),
                HighLevelOption.one_hot(self.previous_option),
            ))
            if self.enable_route_options
            else SubgoalType.one_hot(self.previous_subgoal_type)
        )
        basic_path_context = (
            self._basic_path_context(base_pose)
            if not self.enable_route_options
            else np.zeros(0, dtype=np.float32)
        )
        vector = np.concatenate((
            target_relative_base,
            target_relative_ee,
            final_base_error_body,
            np.asarray([heading_error], dtype=np.float64),
            arm_q,
            arm_margin,
            np.asarray([ee_distance, base_distance], dtype=np.float64),
            self.previous_base_subgoal_error,
            self.previous_ee_subgoal_error,
            self.previous_high_action,
            decision_memory,
            np.asarray([
                self.last_option_duration,
                self.last_option_progress,
                self.last_option_safety_rate,
                float(self.last_option_stalled),
            ], dtype=np.float64),
            basic_path_context,
            np.clip(scan / self.scan_clip, 0.0, 1.0),
        )).astype(np.float32)
        if vector.shape != (self.OBS_DIM,):
            raise RuntimeError(
                "high observation has shape {}, expected ({},)".format(
                    vector.shape,
                    self.OBS_DIM,
                )
            )
        result = {
            "vector": vector,
            "final_target_relative_base": target_relative_base.astype(
                np.float32
            ),
            "final_target_relative_ee": target_relative_ee.astype(
                np.float32
            ),
            "final_base_error_body": final_base_error_body.astype(
                np.float32
            ),
            "final_ee_error_body": np.concatenate((
                rotate_xy(
                    (final_ee_goal - ee_position)[0:2],
                    -base_pose[2],
                ),
                np.asarray([
                    final_ee_goal[2] - ee_position[2]
                ], dtype=np.float64),
            )).astype(np.float32),
            "final_heading_error": float(heading_error),
            "final_ee_distance": float(ee_distance),
            "final_base_distance": float(base_distance),
            "base_pose_world": base_pose.astype(np.float32),
            "final_base_goal_world": final_base_goal.astype(np.float32),
            "scan_bins": scan.astype(np.float32),
            "previous_base_subgoal_error": (
                self.previous_base_subgoal_error.copy()
            ),
            "previous_ee_subgoal_error": (
                self.previous_ee_subgoal_error.copy()
            ),
            "previous_high_action": self.previous_high_action.copy(),
            "previous_subgoal_type": int(self.previous_subgoal_type),
            "previous_subgoal_type_name": SubgoalType.name(
                self.previous_subgoal_type
            ),
            "previous_subgoal_type_one_hot": SubgoalType.one_hot(
                self.previous_subgoal_type
            ),
            "last_option_duration": float(self.last_option_duration),
            "last_option_progress": float(self.last_option_progress),
            "last_option_safety_rate": float(
                self.last_option_safety_rate
            ),
            "last_option_stalled": bool(self.last_option_stalled),
            "terminal_option_latched": bool(
                self.terminal_option_latched
            ),
            "basic_path_context": basic_path_context.copy(),
        }
        if self.enable_route_options:
            result.update({
                "latched_route_side": int(self.latched_route_side),
                "latched_route_side_one_hot": RouteSide.one_hot(
                    self.latched_route_side
                ),
                "previous_option": int(self.previous_option),
                "previous_option_name": HighLevelOption.name(
                    self.previous_option
                ),
                "previous_option_one_hot": HighLevelOption.one_hot(
                    self.previous_option
                ),
            })
        return result

    def guard_detour_command(self, command, sensor=None):
        """Project unsafe DETOUR goals into the active path corridor.

        The student retains its command whenever it already lies inside the
        corridor and the complete footprint-inflated LiDAR corridor is free.
        Otherwise only the executable command is projected; the requested
        action remains in the PPO rollout for diagnostics and learning.
        """
        diagnostics = {
            "applied": False,
            "reason": "inactive",
            "projection": 0.0,
            "requested_along": 0.0,
            "executed_along": 0.0,
            "requested_lateral": 0.0,
            "executed_lateral": 0.0,
            "scan_min": self.scan_clip,
            "corridor_feasible": True,
        }
        if (
                not self.detour_feasibility_guard
                or int(command.subgoal_type) != SubgoalType.DETOUR):
            self._last_detour_guard = diagnostics
            return False
        path = getattr(self.low_environment, "_path", None)
        if path is None or bool(getattr(path, "direct_path", False)):
            diagnostics["reason"] = (
                "no_path" if path is None else "direct_path"
            )
            self._last_detour_guard = diagnostics
            return False
        if sensor is None:
            sensor = self._sensor()

        base_pose = self.low_wrapper.base_pose(sensor)
        waypoint_world = np.asarray(
            path.current_goal_xy, dtype=np.float64
        )
        waypoint_body = rotate_xy(
            waypoint_world - base_pose[0:2], -base_pose[2]
        )
        requested = np.asarray(command.base_goal[0:2], dtype=np.float64)
        maximum_radius = float(np.max(self.action_limits[0:2]))
        path_clearance = float(getattr(
            path, "clearance", self.detour_corridor_half_width
        ))
        corridor_half_width = min(
            self.detour_corridor_half_width,
            max(path_clearance, 0.0),
        )
        projected, corridor_info = project_path_corridor_subgoal(
            requested,
            waypoint_body,
            maximum_radius=maximum_radius,
            corridor_half_width=corridor_half_width,
            minimum_progress=self.detour_minimum_progress,
        )
        diagnostics.update(corridor_info)
        diagnostics["reason"] = (
            "path_corridor" if corridor_info["applied"] else "unchanged"
        )

        scan = self._scan()
        diagnostics["scan_min"] = float(np.min(scan))
        generator = RuleBasedPlanarSubgoalGenerator(
            scan_bin_count=self.SCAN_DIM,
            minimum_radius=min(
                max(self.detour_minimum_progress, 0.02), maximum_radius
            ),
            maximum_radius=maximum_radius,
            clearance_margin=self.detour_clearance_margin,
            footprint_half_length=float(getattr(
                path, "base_half_length", 0.56
            )),
            footprint_half_width=float(getattr(
                path, "base_half_width", 0.10
            )),
        )
        projected_norm = float(np.linalg.norm(projected))
        corridor_feasible = True
        if projected_norm > 1.0e-9:
            corridor_feasible = generator.corridor_heading_feasible(
                scan,
                math.atan2(projected[1], projected[0]),
                projected_norm,
            )
        diagnostics["corridor_feasible"] = bool(corridor_feasible)
        if not corridor_feasible:
            path_index = int(getattr(path, "index", 0))
            committed_side = (
                float(getattr(path, "side", 0.0))
                if path_index <= 3 else 0.0
            )
            lidar_goal, lidar_info = generator.select(
                waypoint_body,
                scan,
                preferred_turn_sign=committed_side,
                committed_turn_sign=committed_side,
                raw_scan_ranges=scan,
            )
            projected = np.asarray(lidar_goal, dtype=np.float32)
            diagnostics["reason"] = str(lidar_info.get(
                "reason", "lidar_corridor_projection"
            ))

        projection = float(np.linalg.norm(projected - requested))
        diagnostics["projection"] = projection
        diagnostics["applied"] = bool(projection > 1.0e-7)
        if diagnostics["applied"]:
            base_goal = command.base_goal.copy()
            base_goal[0:2] = projected
            if float(np.linalg.norm(projected)) > 1.0e-9:
                base_goal[2] = float(np.clip(
                    math.atan2(projected[1], projected[0]),
                    -self.action_limits[2],
                    self.action_limits[2],
                ))
            self._set_command_goal(command, base_goal=base_goal)
            self._append_reason(command, "detour_feasibility_projection")
        self._last_detour_guard = diagnostics
        return bool(diagnostics["applied"])

    @staticmethod
    def _new_terminal_rotation_state():
        return {
            "steps": 0,
            "guarded_steps": 0,
            "recovery_steps": 0,
            "blocked_steps": 0,
            "wrong_direction_steps": 0,
            "progress": 0.0,
            "stall_events": 0,
            "no_progress_streak": 0,
            "recovery_active": False,
            "stall_latched": False,
            "step_active": False,
            "commanded_yaw": 0.0,
        }

    def _terminal_rotation_snapshot(self):
        state = self._terminal_rotation_state
        return dict(
            (name, state[name]) for name in (
                "steps",
                "guarded_steps",
                "recovery_steps",
                "blocked_steps",
                "wrong_direction_steps",
                "progress",
                "stall_events",
            )
        )

    def _terminal_rotation_diagnostics(self, snapshot):
        state = self._terminal_rotation_state
        result = {}
        for name in (
                "steps",
                "guarded_steps",
                "recovery_steps",
                "blocked_steps",
                "wrong_direction_steps",
                "stall_events"):
            result[name] = int(state[name] - snapshot[name])
        result["progress"] = float(
            state["progress"] - snapshot["progress"]
        )
        result["no_progress_streak"] = int(
            state["no_progress_streak"]
        )
        return result

    def _clear_terminal_rotation_streak(self):
        state = self._terminal_rotation_state
        state["no_progress_streak"] = 0
        state["recovery_active"] = False
        state["stall_latched"] = False
        state["step_active"] = False
        state["commanded_yaw"] = 0.0

    def _guard_terminal_rotation_action(self, action, command, sensor):
        """Make BASE_ROTATE pure, directed, and strong enough to observe."""
        result = np.asarray(action, dtype=np.float32).copy()
        if result.shape != (8,):
            raise ValueError(
                "low action must have shape (8,), got {}".format(
                    result.shape
                )
            )
        state = self._terminal_rotation_state
        if self._terminal_control_stage(command) != "BASE_ROTATE":
            self._clear_terminal_rotation_streak()
            return result

        unused_distance, yaw_error = self._final_base_errors(sensor)
        del unused_distance
        original = result.copy()
        result[0] = 0.0
        result[2:8] = 0.0
        if abs(yaw_error) <= self.yaw_subgoal_tolerance:
            result[1] = 0.0
        else:
            magnitude = max(
                abs(float(result[1])),
                self.terminal_rotate_minimum_action,
            )
            recovery_active = bool(
                state["no_progress_streak"]
                >= self.terminal_rotate_recovery_steps
            )
            if recovery_active:
                magnitude = max(
                    magnitude, self.terminal_rotate_recovery_action
                )
                state["recovery_steps"] += 1
            state["recovery_active"] = recovery_active
            result[1] = float(np.sign(yaw_error)) * min(magnitude, 1.0)
        if float(np.max(np.abs(result - original))) > 1.0e-7:
            state["guarded_steps"] += 1
        state["step_active"] = True
        state["commanded_yaw"] = float(result[1])
        return result

    def _update_terminal_rotation_progress(
            self,
            command,
            before_sensor,
            after_sensor,
            commanded_action,
            safe_action):
        state = self._terminal_rotation_state
        if not state.get("step_active", False):
            return False
        state["step_active"] = False
        unused_distance, before_error = self._final_base_errors(
            before_sensor
        )
        del unused_distance
        unused_distance, after_error = self._final_base_errors(after_sensor)
        del unused_distance
        progress = abs(float(before_error)) - abs(float(after_error))
        state["steps"] += 1
        state["progress"] += float(progress)
        commanded_yaw = float(np.asarray(commanded_action)[1])
        safe_yaw = float(np.asarray(safe_action)[1])
        if abs(commanded_yaw) > 0.10 and abs(safe_yaw) < 0.05:
            state["blocked_steps"] += 1
        if progress < -self.terminal_rotate_progress_epsilon:
            state["wrong_direction_steps"] += 1
        if progress >= self.terminal_rotate_progress_epsilon:
            state["no_progress_streak"] = 0
            state["recovery_active"] = False
            state["stall_latched"] = False
        else:
            state["no_progress_streak"] += 1

        still_rotating = bool(
            self._terminal_control_stage(command) == "BASE_ROTATE"
        )
        stalled = bool(
            still_rotating
            and state["no_progress_streak"]
            >= self.terminal_rotate_stall_steps
        )
        if stalled and not state["stall_latched"]:
            state["stall_events"] += 1
            state["stall_latched"] = True
        if not still_rotating:
            self._clear_terminal_rotation_streak()
        return stalled

    def guard_high_command(self, command, sensor=None, update_latch=True):
        """Apply the executable-set contract for the terminal option.

        Before the final path waypoint is active, a premature TERMINAL action
        cannot release the arm.  Once TERMINAL becomes admissible it is
        latched for the rest of the episode, the chassis target is projected
        onto the true final base pose, and the EE target is projected into a
        progress cone toward the true final target.  The resulting fixed EE
        target survives ordinary high-level horizons, preventing an 80-step
        replan from moving the target while the low layer is tracking it.

        ``update_latch=False`` is used while constructing a teacher label so
        querying the teacher cannot mutate the student's environment state.
        """
        if sensor is None:
            sensor = self._sensor()
        requested_terminal = bool(
            int(command.subgoal_type) == SubgoalType.TERMINAL
        )
        path = getattr(self.low_environment, "_path", None)
        final_waypoint_active = bool(
            path is not None
            and getattr(path, "final_waypoint_active", False)
        )
        path_complete = bool(
            path is not None and getattr(path, "complete", False)
        )
        terminal_eligible = bool(
            final_waypoint_active or path_complete
        )
        stage_forced = False
        invalid_terminal = False

        if self.terminal_option_contract:
            if self.terminal_option_latched:
                stage_forced = not requested_terminal
                command.subgoal_type = SubgoalType.TERMINAL
                command.route_side = RouteSide.NONE
                requested_terminal = True
                if stage_forced:
                    self._append_reason(command, "terminal_stage_latched")
            elif requested_terminal and terminal_eligible and update_latch:
                self.terminal_option_latched = True

            if requested_terminal and not terminal_eligible:
                invalid_terminal = True
                self._set_command_goal(
                    command, ee_goal=np.zeros(3, dtype=np.float32)
                )
                self._append_reason(command, "terminal_not_eligible")

        terminal_aligned = bool(
            path is not None
            and (
                path_complete
                or bool(getattr(
                    self.low_environment,
                    "_terminal_pose_aligned_latched",
                    False,
                ))
            )
        )
        if not requested_terminal or invalid_terminal:
            self._last_terminal_guard = {
                "terminal_eligible": terminal_eligible,
                "invalid_terminal": invalid_terminal,
                "stage_forced": stage_forced,
                "ee_projection": 0.0,
                "ee_student_alignment_cosine": 0.0,
                "ee_executed_alignment_cosine": 0.0,
            }
            return False

        ee_projection = 0.0
        ee_diagnostics = {
            "requested_norm": 0.0,
            "executed_norm": 0.0,
            "student_alignment_cosine": 0.0,
            "executed_alignment_cosine": 0.0,
            "alignment_cosine": 0.0,
            "norm_clipped": False,
            "progress_fallback": False,
            "persistent_target": False,
        }
        if self.terminal_option_contract:
            if terminal_aligned:
                base_goal = np.zeros(3, dtype=np.float32)
            else:
                base_distance, yaw_error = self._final_base_errors(sensor)
                del base_distance
                base_pose = self.low_wrapper.base_pose(sensor)
                final_base_goal, unused_heading = self._final_base_pose()
                base_goal = np.concatenate((
                    rotate_xy(
                        final_base_goal - base_pose[0:2], -base_pose[2]
                    ),
                    np.asarray([yaw_error], dtype=np.float64),
                ))
                base_goal = np.clip(
                    base_goal,
                    -self.action_limits[0:3],
                    self.action_limits[0:3],
                ).astype(np.float32)
            ee_goal, ee_projection, ee_diagnostics = self._terminal_ee_goal(
                command, sensor, persist=bool(update_latch)
            )
            self._set_command_goal(
                command, base_goal=base_goal, ee_goal=ee_goal
            )
            self._append_reason(command, "terminal_goal_projection")
        elif terminal_aligned:
            self._set_command_goal(
                command, base_goal=np.zeros(3, dtype=np.float32)
            )

        if terminal_aligned:
            self._append_reason(command, "terminal_base_hold")
        self._last_terminal_guard = {
            "terminal_eligible": terminal_eligible,
            "invalid_terminal": False,
            "stage_forced": stage_forced,
            "ee_projection": float(ee_projection),
            "ee_requested_norm": float(
                ee_diagnostics["requested_norm"]
            ),
            "ee_executed_norm": float(
                ee_diagnostics["executed_norm"]
            ),
            "ee_alignment_cosine": float(
                ee_diagnostics["alignment_cosine"]
            ),
            "ee_student_alignment_cosine": float(
                ee_diagnostics["student_alignment_cosine"]
            ),
            "ee_executed_alignment_cosine": float(
                ee_diagnostics["executed_alignment_cosine"]
            ),
            "ee_norm_clipped": bool(ee_diagnostics["norm_clipped"]),
            "ee_progress_fallback": bool(
                ee_diagnostics["progress_fallback"]
            ),
            "ee_persistent_target": bool(
                ee_diagnostics["persistent_target"]
            ),
        }
        return bool(terminal_aligned)

    def _terminal_ee_goal(self, command, sensor, persist):
        """Return a bounded body-frame EE displacement toward the final EE."""
        base_pose = self.low_wrapper.base_pose(sensor)
        ee_position = self.low_wrapper.ee_position(sensor)
        requested = np.asarray(command.ee_goal, dtype=np.float64)
        final_error = self._final_ee_error_body(sensor)
        if self.terminal_fixed_ee_goal_world is not None:
            world_error = (
                self.terminal_fixed_ee_goal_world - ee_position
            )
            relative = np.concatenate((
                rotate_xy(world_error[0:2], -base_pose[2]),
                np.asarray([world_error[2]], dtype=np.float64),
            ))
            diagnostics = self._terminal_ee_diagnostics(
                requested,
                relative,
                final_error,
                persistent_target=True,
            )
            return (
                relative.astype(np.float32),
                float(np.linalg.norm(relative - requested)),
                diagnostics,
            )

        distance = float(np.linalg.norm(final_error))
        if distance <= 1.0e-9:
            projected = np.zeros(3, dtype=np.float64)
            diagnostics = self._terminal_ee_diagnostics(
                requested, projected, final_error
            )
            return (
                projected.astype(np.float32),
                float(np.linalg.norm(projected - requested)),
                diagnostics,
            )
        maximum_step = min(self.terminal_ee_step, distance)
        nominal = final_error * (maximum_step / distance)
        projected = (
            (1.0 - self.terminal_student_blend) * nominal
            + self.terminal_student_blend * requested
        )
        norm = float(np.linalg.norm(projected))
        norm_clipped = bool(norm > maximum_step)
        if norm > maximum_step:
            projected *= maximum_step / norm
        progress_fallback = bool(
            float(np.dot(projected, final_error)) <= 0.0
        )
        if progress_fallback:
            projected = nominal
        if persist:
            delta_world_xy = rotate_xy(projected[0:2], base_pose[2])
            self.terminal_fixed_ee_goal_world = np.asarray([
                ee_position[0] + delta_world_xy[0],
                ee_position[1] + delta_world_xy[1],
                ee_position[2] + projected[2],
            ], dtype=np.float64)
        diagnostics = self._terminal_ee_diagnostics(
            requested,
            projected,
            final_error,
            norm_clipped=norm_clipped,
            progress_fallback=progress_fallback,
        )
        return (
            projected.astype(np.float32),
            float(np.linalg.norm(projected - requested)),
            diagnostics,
        )

    @staticmethod
    def _terminal_ee_diagnostics(
            requested,
            executed,
            final_error,
            norm_clipped=False,
            progress_fallback=False,
            persistent_target=False):
        """Describe the executed terminal target without changing control."""
        requested = np.asarray(requested, dtype=np.float64)
        executed = np.asarray(executed, dtype=np.float64)
        final_error = np.asarray(final_error, dtype=np.float64)
        requested_norm = float(np.linalg.norm(requested))
        executed_norm = float(np.linalg.norm(executed))
        final_error_norm = float(np.linalg.norm(final_error))
        requested_denominator = requested_norm * final_error_norm
        student_alignment_cosine = (
            float(np.dot(requested, final_error))
            / requested_denominator
            if requested_denominator > 1.0e-12 else 0.0
        )
        executed_denominator = executed_norm * final_error_norm
        executed_alignment_cosine = (
            float(np.dot(executed, final_error))
            / executed_denominator
            if executed_denominator > 1.0e-12 else 0.0
        )
        return {
            "requested_norm": requested_norm,
            "executed_norm": executed_norm,
            "student_alignment_cosine": float(np.clip(
                student_alignment_cosine, -1.0, 1.0
            )),
            "executed_alignment_cosine": float(np.clip(
                executed_alignment_cosine, -1.0, 1.0
            )),
            # Compatibility alias retained for existing analysis scripts.
            "alignment_cosine": float(np.clip(
                executed_alignment_cosine, -1.0, 1.0
            )),
            "norm_clipped": bool(norm_clipped),
            "progress_fallback": bool(progress_fallback),
            "persistent_target": bool(persistent_target),
        }

    def _set_command_goal(self, command, base_goal=None, ee_goal=None):
        if base_goal is not None:
            command.base_goal[:] = np.asarray(base_goal, dtype=np.float32)
        if ee_goal is not None:
            command.ee_goal[:] = np.asarray(ee_goal, dtype=np.float32)
        command.subgoal[:] = np.concatenate((
            command.base_goal,
            command.ee_goal,
        )).astype(np.float32)
        if command.normalized_action is not None:
            command.normalized_action[:] = command.to_action(
                limits=self.action_limits
            )

    @staticmethod
    def _append_reason(command, reason):
        parts = str(command.reason).split("+") if command.reason else []
        if reason not in parts:
            parts.append(str(reason))
        command.reason = "+".join(parts)

    def _basic_path_context(self, base_pose):
        """Return bounded navigation state used by the safe-waypoint teacher.

        Layout:
          waypoint dx/limit, waypoint dy/limit, bearing/pi,
          path index fraction, remaining distance/scan clip,
          final-waypoint active, path complete, direct-path flag.
        """
        path = getattr(self.low_environment, "_path", None)
        if path is None:
            return np.zeros(self.BASIC_PATH_CONTEXT_DIM, dtype=np.float32)
        base_pose = np.asarray(base_pose, dtype=np.float64)
        final_goal, unused_heading = self._final_base_pose()
        waypoint = np.asarray(
            getattr(path, "current_goal_xy", final_goal),
            dtype=np.float64,
        )
        if waypoint.shape != (2,):
            raise RuntimeError("active path waypoint must have shape (2,)")
        delta_body = rotate_xy(
            waypoint - base_pose[0:2], -base_pose[2]
        )
        scaled_delta = np.clip(
            delta_body / self.action_limits[0:2], -1.0, 1.0
        )
        bearing = (
            math.atan2(delta_body[1], delta_body[0])
            if float(np.linalg.norm(delta_body)) > 1.0e-9 else 0.0
        )
        waypoints = getattr(path, "waypoints", None)
        waypoint_count = int(len(waypoints)) if waypoints is not None else 1
        path_index = int(getattr(path, "index", 0))
        index_fraction = float(np.clip(
            float(path_index) / float(max(waypoint_count - 1, 1)),
            0.0,
            1.0,
        ))
        if hasattr(path, "remaining_distance"):
            remaining = float(path.remaining_distance(base_pose[0:2]))
        else:
            remaining = float(np.linalg.norm(waypoint - base_pose[0:2]))
        final_waypoint_active = bool(getattr(
            path,
            "final_waypoint_active",
            path_index >= waypoint_count - 1,
        ))
        return np.asarray([
            scaled_delta[0],
            scaled_delta[1],
            np.clip(bearing / math.pi, -1.0, 1.0),
            index_fraction,
            np.clip(remaining / self.scan_clip, 0.0, 1.0),
            float(final_waypoint_active),
            float(bool(getattr(path, "complete", False))),
            float(bool(getattr(path, "direct_path", False))),
        ], dtype=np.float32)

    def _subgoal_error_measure(self, remaining, command):
        """Tolerance-normalized, phase-causal option error.

        TERMINAL deliberately freezes the arm during base translation and
        rotation.  Its stall monitor must therefore observe only the state
        variable that the active deterministic substate can change.
        """
        remaining = np.asarray(remaining, dtype=np.float64)
        if int(command.subgoal_type) == SubgoalType.TERMINAL:
            stage = self._terminal_control_stage(command)
            if stage == "BASE_TRANSLATE":
                return float(np.linalg.norm(remaining[0:2])) / (
                    self.base_subgoal_tolerance
                )
            if stage == "BASE_ROTATE":
                # The fixed subgoal yaw is clipped to the per-option action
                # limit.  For large terminal yaw errors it can be reached
                # before the true final pose, after which the clipped error
                # grows in the opposite direction and falsely looks stalled.
                # Monitor the authoritative final-pose yaw instead.
                if self.low_environment.last_sensor_observation is None:
                    final_yaw_error = float(remaining[2])
                else:
                    unused_distance, final_yaw_error = (
                        self._final_base_errors(self._sensor())
                    )
                    del unused_distance
                return abs(float(final_yaw_error)) / (
                    self.yaw_subgoal_tolerance
                )
            if stage in ("ARM_REACH", "ALIGNED"):
                return float(np.linalg.norm(remaining[3:6])) / (
                    self.ee_subgoal_tolerance
                )
        return float(max(
            float(np.linalg.norm(remaining[0:2]))
            / self.base_subgoal_tolerance,
            abs(float(remaining[2])) / self.yaw_subgoal_tolerance,
        ))

    def _terminal_control_stage(self, command):
        """Return the canonical deterministic substate for one option."""
        if int(command.subgoal_type) != SubgoalType.TERMINAL:
            return "NONE"
        stage = str(getattr(
            self.low_environment,
            "_terminal_pose_stage",
            "BASE_TRANSLATE",
        ))
        return {
            "NONE": "BASE_TRANSLATE",
            "TRANSLATE": "BASE_TRANSLATE",
            "ROTATE": "BASE_ROTATE",
        }.get(stage, stage)

    def _apply_route_decision(self, command):
        """Latch one DETOUR route option and synchronize the route reward."""
        if int(command.subgoal_type) != SubgoalType.DETOUR:
            command.route_side = RouteSide.NONE
            return
        requested = int(command.route_side)
        if requested == RouteSide.NONE:
            path = getattr(self.low_environment, "_path", None)
            requested = RouteSide.from_sign(
                0.0 if path is None else getattr(path, "side", 0.0)
            )
        if requested == RouteSide.NONE:
            requested = RouteSide.UPPER
        if self.latched_route_side == RouteSide.NONE:
            self.latched_route_side = requested
            selector = getattr(
                self.low_environment,
                "select_detour_side",
                None,
            )
            if selector is not None:
                selector(RouteSide.to_sign(requested))
        command.route_side = int(self.latched_route_side)

    def close(self):
        try:
            self.low_wrapper.close()
        finally:
            if hasattr(self.low_environment, "close"):
                self.low_environment.close()
            elif hasattr(self.low_environment, "stop"):
                self.low_environment.stop()

    def _sensor(self):
        sensor = self.low_environment.last_sensor_observation
        if sensor is None:
            raise RuntimeError("low environment has no current observation")
        return FusedLowLevelState.as_sensor(sensor)

    def _scan(self):
        structured = getattr(
            self.low_environment,
            "last_structured_observation",
            None,
        )
        if not isinstance(structured, dict):
            return np.full(self.SCAN_DIM, self.scan_clip, dtype=np.float64)
        scan = np.asarray(
            structured.get("scan_bins", []),
            dtype=np.float64,
        )
        if scan.shape != (self.SCAN_DIM,):
            raise RuntimeError(
                "high-level scan has shape {}, expected ({},)".format(
                    scan.shape,
                    self.SCAN_DIM,
                )
            )
        return scan

    def _final_base_pose(self):
        path = getattr(self.low_environment, "_path", None)
        if path is None:
            reset_info = self.low_environment.last_reset_info
            goal = np.asarray(reset_info["base_goal_xy"], dtype=np.float64)
            return goal, self.low_wrapper.base_pose(self._sensor())[2]
        planar_yaw = (
            float(path.goal_yaw)
            if path.goal_yaw is not None
            else float(self._sensor()[13])
        )
        tracked_heading = (
            planar_yaw + FusedLowLevelState().forward_yaw_offset
        )
        return path.goal_xy.copy(), wrap_angle(tracked_heading)

    def _final_ee_distance(self, sensor):
        return float(np.linalg.norm(
            np.asarray(sensor[0:3], dtype=np.float64)
            - self.low_wrapper.ee_position(sensor)
        ))

    def _final_ee_error_body(self, sensor):
        base_pose = self.low_wrapper.base_pose(sensor)
        ee_position = self.low_wrapper.ee_position(sensor)
        error_world = (
            np.asarray(sensor[0:3], dtype=np.float64) - ee_position
        )
        return np.concatenate((
            rotate_xy(error_world[0:2], -base_pose[2]),
            np.asarray([error_world[2]], dtype=np.float64),
        ))

    def _final_base_errors(self, sensor):
        base_pose = self.low_wrapper.base_pose(sensor)
        final_goal, final_heading = self._final_base_pose()
        return (
            float(np.linalg.norm(final_goal - base_pose[0:2])),
            float(wrap_angle(final_heading - base_pose[2])),
        )

    def _path_remaining(self, sensor):
        path = getattr(self.low_environment, "_path", None)
        if path is None or not hasattr(path, "remaining_distance"):
            return None
        base_pose = self.low_wrapper.base_pose(sensor)
        return float(path.remaining_distance(base_pose[0:2]))

    def _ee_rotation_world(self, sensor):
        reach_environment = getattr(
            self.low_environment,
            "reach_env",
            None,
        )
        provider = getattr(
            reach_environment,
            "kinematics_provider",
            None,
        )
        if provider is None:
            return None
        if not hasattr(provider, "end_effector_rotation"):
            return None
        q = np.asarray(sensor[11:21], dtype=np.float64)
        return provider.end_effector_rotation(q)


def rotate_xy(vector, angle):
    vector = np.asarray(vector, dtype=np.float64)
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    return np.asarray([
        cosine * vector[0] - sine * vector[1],
        sine * vector[0] + cosine * vector[1],
    ], dtype=np.float64)


def wrap_angle(value):
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def base_pose_to_xyz(base_pose):
    return np.asarray([
        base_pose[0],
        base_pose[1],
        0.0,
    ], dtype=np.float64)


def active_safety_reasons(safety_info):
    """Return joints whose safety filter changed the requested command."""
    if not isinstance(safety_info, dict):
        return []
    reasons = []
    for joint_name, joint_info in safety_info.items():
        if not isinstance(joint_info, dict):
            continue
        raw_command = float(joint_info.get("cmd_raw", 0.0))
        safe_command = float(
            joint_info.get("cmd_safe", raw_command)
        )
        if abs(raw_command - safe_command) > 1.0e-9:
            reasons.append("{}:{}".format(
                joint_name,
                str(joint_info.get("reason", "safe")),
            ))
    return sorted(set(reasons))
