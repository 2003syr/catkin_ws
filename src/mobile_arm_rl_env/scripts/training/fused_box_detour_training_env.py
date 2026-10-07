#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fused 66-D/8-D training task with one static box in the base path.

This is deliberately a separate environment.  The established obstacle-free
``FusedLowLevelTrainingEnv`` and its checkpoints are unchanged.  A deterministic
geometry route updates the local base sub-goal while the existing coordinated
teacher and action adapter continue to provide the low-level command contract.
"""

from __future__ import division

import numpy as np

from gazebo_msgs.msg import ModelState

from hrl.fused_low_level import (
    FusedLowLevelState,
    TRACKED_FORWARD_YAW_OFFSET,
)
from training.fused_box_detour import (
    BoxDetourPath,
    BoxStudentStallTakeover,
    box_action_mask_phase,
    box_reward_progress,
    box_student_arm_hold_required,
    compose_box_target_configuration,
    resolve_box_detour_side,
    update_terminal_control_state,
    update_terminal_pose_latch,
)
from training.fused_low_training_env import FusedLowLevelTrainingEnv


class FusedBoxDetourTrainingEnv(FusedLowLevelTrainingEnv):
    """Reach an arm target while the tracked base follows a box detour."""

    def __init__(self, *args, **kwargs):
        super(FusedBoxDetourTrainingEnv, self).__init__(*args, **kwargs)
        self.default_reset_base_positions = np.asarray(
            self.reset_base_positions,
            dtype=np.float64,
        ).copy()
        self.box_center_xy = self._vector(
            self._param_or_default(
                "~box_center_xy", [0.85, -0.13]
            ),
            2,
            "box_center_xy",
        )
        self.box_size_xy = self._vector(
            self._param_or_default(
                "~box_size_xy", [0.20, 0.32]
            ),
            2,
            "box_size_xy",
        )
        self.obstacle_model_name = str(self._param_or_default(
            "~obstacle_model_name", "planar_avoidance_obstacle"
        ))
        self.obstacle_repositioning_enabled = bool(self._param_or_default(
            "~enable_obstacle_repositioning", False
        ))
        self.obstacle_geometry_center_xy = self._vector(
            self._param_or_default(
                "~obstacle_geometry_center_xy", [0.85, -0.13]
            ),
            2,
            "obstacle_geometry_center_xy",
        )
        self.obstacle_hidden_center_xy = self._vector(
            self._param_or_default(
                "~obstacle_hidden_center_xy", [-100.0, -100.0]
            ),
            2,
            "obstacle_hidden_center_xy",
        )
        self.final_base_goal_xy = self._vector(
            self._param_or_default(
                "~final_base_goal_xy", [1.60, -0.13]
            ),
            2,
            "final_base_goal_xy",
        )
        self.base_half_length = float(
            self._param_or_default("~base_half_length", 0.56)
        )
        self.base_half_width = float(
            self._param_or_default("~base_half_width", 0.10)
        )
        self.path_clearance = float(
            self._param_or_default("~path_clearance", 0.15)
        )
        self.waypoint_tolerance = float(
            self._param_or_default("~waypoint_tolerance", 0.10)
        )
        self.final_base_position_tolerance = float(
            self._param_or_default(
                "~final_base_position_tolerance", 0.03
            )
        )
        self.final_base_yaw_tolerance = float(
            self._param_or_default(
                "~final_base_yaw_tolerance", 0.08
            )
        )
        if (
                self.final_base_position_tolerance <= 0.0
                or self.final_base_yaw_tolerance <= 0.0):
            raise ValueError("final base pose tolerances must be positive")
        self.terminal_pose_enter_position_tolerance = float(
            self._param_or_default(
                "~terminal_pose_enter_position_tolerance",
                self.final_base_position_tolerance,
            )
        )
        self.terminal_pose_exit_position_tolerance = float(
            self._param_or_default(
                "~terminal_pose_exit_position_tolerance",
                max(
                    2.0 * self.terminal_pose_enter_position_tolerance,
                    0.05,
                ),
            )
        )
        self.terminal_pose_stable_cycles = int(
            self._param_or_default("~terminal_pose_stable_cycles", 3)
        )
        if (
                self.terminal_pose_enter_position_tolerance <= 0.0
                or self.terminal_pose_exit_position_tolerance
                < self.terminal_pose_enter_position_tolerance
                or self.terminal_pose_stable_cycles <= 0):
            raise ValueError("invalid terminal pose hysteresis parameters")
        self.default_detour_side = float(
            self._param_or_default("~detour_side", 1.0)
        )
        self.collision_penalty = float(
            self._param_or_default("~box_collision_penalty", 8.0)
        )
        self.timeout_penalty = float(
            self._param_or_default("~box_timeout_penalty", 1.0)
        )
        self.path_reward_scale = float(
            self._param_or_default("~box_path_reward_scale", 5.0)
        )
        self.ee_reward_scale = float(
            self._param_or_default("~box_ee_reward_scale", 2.0)
        )
        self.yaw_reward_scale = float(
            self._param_or_default("~box_yaw_reward_scale", 2.0)
        )
        self.time_penalty = float(
            self._param_or_default("~box_time_penalty", 0.001)
        )
        self.success_bonus = float(
            self._param_or_default("~box_success_bonus", 20.0)
        )
        if min(
                self.path_reward_scale,
                self.ee_reward_scale,
                self.yaw_reward_scale,
                self.time_penalty,
                self.success_bonus) < 0.0:
            raise ValueError("box reward parameters must be non-negative")
        # Keep the student's action support aligned with the coordinated
        # teacher during the box task.  These are structural task constraints
        # rather than a replacement policy: the network still chooses the
        # action inside the allowed phase/kinematic support.
        self.enable_student_action_guard = bool(
            self._param_or_default("~enable_student_action_guard", True)
        )
        self.student_heading_slow_angle = float(
            self._param_or_default("~teacher_heading_slow_angle", 0.70)
        )
        self.enable_student_stall_takeover = bool(
            self._param_or_default(
                "~enable_student_stall_takeover", False
            )
        )
        self._student_stall_takeover = BoxStudentStallTakeover(
            base_stall_cycles=int(self._param_or_default(
                "~student_base_stall_cycles", 20
            )),
            arm_stall_cycles=int(self._param_or_default(
                "~student_arm_finish_stall_cycles", 20
            )),
            minimum_progress=float(self._param_or_default(
                "~student_stall_minimum_progress", 1.0e-4
            )),
        )
        self._student_takeover_diagnostics = (
            self._student_stall_takeover.diagnostics(0, 6)
        )
        self._cached_teacher_action = None
        self._cached_teacher_step = -1
        self._path = None
        self._initial_path_remaining = 1.0
        self._previous_path_remaining = None
        self._previous_ee_distance = None
        self._last_student_action_gate = {
            "raw_action": np.zeros(8, dtype=np.float32),
            "gated_action": np.zeros(8, dtype=np.float32),
            "reasons": [],
            "action_mask_phase": "BASE_APPROACH",
            "heading_error": 0.0,
            "path_index": 0,
            "box_passed": False,
            "path_complete": False,
        }
        self._arm_hold_release_x = float("inf")
        self._arm_obstacle_hold = False
        self._final_base_yaw = 0.0
        self._terminal_pose_stage = "NONE"
        self._terminal_pose_stable_count = 0
        self._terminal_pose_aligned_latched = False
        self._terminal_arm_stable_count = 0
        self._terminal_arm_aligned_latched = False
        self.terminal_arm_stable_cycles = int(
            self._param_or_default("~terminal_arm_stable_cycles", 1)
        )
        if self.terminal_arm_stable_cycles <= 0:
            raise ValueError("terminal_arm_stable_cycles must be positive")
        self._coordinated_arm_blend = 0.0
        self._arm_box_release_ready = False
        self._arm_box_release_margin = float(
            self._param_or_default("~arm_box_release_margin", 0.02)
        )
        if self._arm_box_release_margin < 0.0:
            raise ValueError("arm_box_release_margin must be non-negative")
        self._arm_finish_stall_cycles = 0
        self._arm_finish_best_distance = float("inf")
        self._arm_finish_stall_limit = int(
            self._param_or_default("~arm_finish_stall_cycles", 20)
        )
        self._arm_finish_joint_reference_gain = float(
            self._param_or_default(
                "~arm_finish_joint_reference_gain", 0.35
            )
        )
        if (
                self._arm_finish_stall_limit <= 0
                or self._arm_finish_joint_reference_gain < 0.0):
            raise ValueError("invalid arm finish fallback parameters")
        self._arm_finish_fallback_used = False
        self._arm_finish_joint_error = np.zeros(6, dtype=np.float64)
        self._last_arm_obstacle_diagnostics = {
            "hold": False,
            "release_x": float("inf"),
            "path_index": 0,
            "passed_box": False,
        }

    def reset(self, scenario=None, max_steps=None):
        scenario = {} if scenario is None else dict(scenario)
        self.max_steps = (
            self.default_max_steps
            if max_steps is None else int(max_steps)
        )
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.base_env.max_steps = self.max_steps
        self._terminal_pose_stage = "NONE"
        self._terminal_pose_stable_count = 0
        self._terminal_pose_aligned_latched = False
        self._terminal_arm_stable_count = 0
        self._terminal_arm_aligned_latched = False
        self._coordinated_arm_blend = 0.0
        self._arm_box_release_ready = False
        self._arm_finish_stall_cycles = 0
        self._arm_finish_best_distance = float("inf")
        self._student_stall_takeover.reset()
        self._student_takeover_diagnostics = (
            self._student_stall_takeover.diagnostics(0, 6)
        )
        self._cached_teacher_action = None
        self._cached_teacher_step = -1

        self.reset_base_positions = self._vector(
            scenario.get(
                "initial_base_positions",
                self.default_reset_base_positions,
            ),
            3,
            "scenario.initial_base_positions",
        )
        no_obstacle = bool(scenario.get("no_obstacle", False))
        box_center = self._vector(
            scenario.get("box_center_xy", self.box_center_xy),
            2,
            "box_center_xy",
        )
        box_size = self._vector(
            scenario.get("box_size_xy", self.box_size_xy),
            2,
            "box_size_xy",
        )
        if not np.allclose(box_size, self.box_size_xy, atol=1.0e-9):
            raise ValueError(
                "scenario.box_size_xy must match the fixed Gazebo model; "
                "randomize position now and add a spawned model pool before "
                "randomizing physical size"
            )

        # Remove the previous episode's physical box before teleporting the
        # robot back to its next initial pose.  Repositioning the robot while
        # the old box is still nearby lets Gazebo's contact solver push the
        # freshly reset x joint away from its requested coordinate.  This is
        # especially visible when the next scenario is DIRECT/no-obstacle:
        # the box used to be hidden only *after* reset had already failed.
        self._clear_obstacle_for_robot_reset()

        # Reset the complete pose first, then sample a safe arm target from
        # the actual settled joint state.  In this model base_link is the
        # fixed world root and x/y are planar pose joints, so generated FK
        # targets must include the final base position.
        self._reset_and_wait()
        q, _ = self.base_env.get_ordered_joint_state()
        q = np.asarray(q, dtype=np.float64)

        target_arm = scenario.get("target_arm_positions")
        if target_arm is not None:
            target_arm = self._vector(
                target_arm, 6, "scenario.target_arm_positions"
            )
            chassis_clearance = self.reach_env._arm_chassis_clearance(
                target_arm
            )
            if chassis_clearance < self.reach_env.target_min_chassis_clearance:
                raise ValueError(
                    "box scenario target violates chassis clearance: {:.5f}".format(
                        chassis_clearance
                    )
                )
        else:
            # Joint-limit-safe samples can still place link4/link5 through the
            # chassis.  Match the established coordinated sampler and retry
            # until the independent chassis guard also accepts the pose.
            target_arm = None
            chassis_clearance = -float("inf")
            for _ in range(self.target_sample_attempts):
                candidate_arm = self.reach_env._sample_safe_arm_positions()
                candidate_clearance = self.reach_env._arm_chassis_clearance(
                    candidate_arm
                )
                if candidate_clearance >= self.reach_env.target_min_chassis_clearance:
                    target_arm = candidate_arm
                    chassis_clearance = candidate_clearance
                    break
            if target_arm is None:
                raise RuntimeError(
                    "Unable to sample a box target with chassis clearance"
                )

        side = resolve_box_detour_side(
            scenario.get("detour_side", self.default_detour_side),
            self.episode_index + 1,
        )
        final_goal = self._vector(
            scenario.get("final_base_goal_xy", self.final_base_goal_xy),
            2,
            "final_base_goal_xy",
        )
        requested_final_goal = final_goal.copy()
        final_base_yaw = float(
            scenario.get("final_base_yaw", q[2])
        )
        if not np.isfinite(final_base_yaw):
            raise ValueError("scenario.final_base_yaw must be finite")
        self._final_base_yaw = final_base_yaw
        target_position = scenario.get("target_position")
        self._place_obstacle(box_center, no_obstacle)
        self._path = BoxDetourPath(
            q[0:2],
            final_goal,
            box_center_xy=box_center,
            box_size_xy=box_size,
            base_half_length=float(
                scenario.get("base_half_length", self.base_half_length)
            ),
            base_half_width=float(
                scenario.get("base_half_width", self.base_half_width)
            ),
            clearance=float(
                scenario.get("path_clearance", self.path_clearance)
            ),
            side=side,
            waypoint_tolerance=float(
                scenario.get("waypoint_tolerance", self.waypoint_tolerance)
            ),
            goal_yaw=final_base_yaw,
            final_position_tolerance=float(scenario.get(
                "final_base_position_tolerance",
                self.final_base_position_tolerance,
            )),
            final_yaw_tolerance=float(scenario.get(
                "final_base_yaw_tolerance",
                self.final_base_yaw_tolerance,
            )),
            direct_path=no_obstacle,
        )
        # A requested goal inside the conservative post-box clearance would
        # force the base to reverse into the safety margin.  Promote the
        # terminal base goal to the first safe post-box pose; generate the arm
        # target at that same measured-pose contract.
        final_goal = self._path.goal_xy.copy()
        if target_position is None:
            target_q = compose_box_target_configuration(
                q,
                target_arm,
                final_goal,
                final_base_yaw=final_base_yaw,
            )
            target_position = (
                self.reach_env.kinematics_provider.end_effector_position(
                    target_q
                )
            )
        target_position = self._vector(
            target_position, 3, "scenario.target_position"
        )
        if not (
                self.reach_env.target_min_z
                <= target_position[2]
                <= self.reach_env.target_max_z):
            raise ValueError("box scenario target z is outside the configured range")
        # The teacher has no arm-obstacle observation.  The three-stage action
        # mask therefore holds the arm through waypoints 1..2, enables joint
        # base/arm motion while the remaining waypoint poses are reached, and
        # locks the base only after the terminal position and yaw both pass.
        # Keep the rear-clearance x coordinate for diagnostics only; the
        # latched path index owns the actual stage transitions.
        self._arm_hold_release_x = float(self._path.waypoints[3, 0])
        self._arm_obstacle_hold = False
        self._arm_finish_fallback_used = False
        self._arm_finish_joint_error = np.zeros(6, dtype=np.float64)
        self._last_arm_obstacle_diagnostics = {
            "hold": False,
            "release_x": self._arm_hold_release_x,
            "path_index": int(self._path.index),
            "passed_box": False,
        }
        self.base_goal_xy = self._path.current_goal_xy
        self.target_arm_positions = target_arm.copy()
        self.reach_env._set_target_position(target_position)
        self.reach_env._publish_target_immediately(repeat=3)

        structured_observation = self.base_env.reset()
        self.last_structured_observation = structured_observation
        self.last_sensor_observation = FusedLowLevelState.as_sensor(
            structured_observation
        )
        fused_observation = self._build_box_observation(
            structured_observation
        )
        self.episode_index += 1
        initial_ee_distance = float(np.linalg.norm(
            self.last_sensor_observation[0:3]
            - self.last_sensor_observation[6:9]
        ))
        self._initial_path_remaining = self._path.remaining_distance(q[0:2])
        self._previous_path_remaining = self._initial_path_remaining
        self._previous_ee_distance = initial_ee_distance
        displacement_fixed = final_goal - q[0:2]
        heading = float(q[2]) + TRACKED_FORWARD_YAW_OFFSET
        displacement_body = FusedLowLevelState._rotate_xy(
            displacement_fixed, -heading
        )
        self.last_reset_info = {
            "episode_index": self.episode_index,
            "base_goal_xy": final_goal.copy(),
            "requested_final_base_goal_xy": requested_final_goal,
            "base_displacement_body": displacement_body.copy(),
            "target_position": target_position.copy(),
            "target_arm_positions": target_arm.copy(),
            "final_base_yaw": final_base_yaw,
            "final_base_position_tolerance": (
                self._path.final_position_tolerance
            ),
            "final_base_yaw_tolerance": self._path.final_yaw_tolerance,
            "initial_arm_positions": q[4:10].copy(),
            "target_chassis_clearance": chassis_clearance,
            "scenario_id": scenario.get("scenario_id"),
            "scenario_category": str(scenario.get("category", "box_detour")),
            "no_obstacle": bool(no_obstacle),
            "detour_side": float(self._path.side),
            "box_path": self._path.metadata(),
            "reset_base_positions": self.reset_base_positions.copy(),
            "physical_obstacle_center_xy": (
                self.obstacle_hidden_center_xy.copy()
                if no_obstacle else box_center.copy()
            ),
            "obstacle_repositioning_enabled": bool(
                self.obstacle_repositioning_enabled
            ),
            "max_steps": self.max_steps,
            "initial_ee_distance": initial_ee_distance,
            "initial_path_length": self._initial_path_remaining,
            "action_mask_phase": self._action_mask_phase(),
            "action_mask": self._action_mask().copy(),
        }
        return fused_observation

    def select_detour_side(self, side):
        """Rebuild the active route for a high-level option decision.

        Scenario generation supplies a balanced teacher route, but the
        learned high policy owns the route option at execution time.  The
        selected side is applied before the corresponding high-level action
        is executed, so path-potential reward and waypoint advancement agree
        with the student's choice.  Rebuilding from the original start and
        advancing at the measured pose preserves progress already made along
        the common straight approach.
        """
        if self._path is None or self.last_sensor_observation is None:
            raise RuntimeError("reset must be called before selecting a route")
        if bool(self._path.direct_path):
            return 0.0
        selected = 1.0 if float(side) >= 0.0 else -1.0
        if selected == float(self._path.side):
            return selected

        old_path = self._path
        sensor = np.asarray(self.last_sensor_observation, dtype=np.float64)
        base_xy = sensor[11:13]
        base_yaw = float(sensor[13])
        new_path = BoxDetourPath(
            old_path.start_xy,
            old_path.requested_goal_xy,
            box_center_xy=old_path.box_center_xy,
            box_size_xy=old_path.box_size_xy,
            base_half_length=old_path.base_half_length,
            base_half_width=old_path.base_half_width,
            clearance=old_path.clearance,
            side=selected,
            waypoint_tolerance=old_path.waypoint_tolerance,
            goal_yaw=old_path.goal_yaw,
            final_position_tolerance=old_path.final_position_tolerance,
            final_yaw_tolerance=old_path.final_yaw_tolerance,
            direct_path=False,
        )
        new_path.advance(base_xy, planar_yaw=base_yaw)
        self._path = new_path
        self.base_goal_xy = new_path.current_goal_xy
        self._arm_hold_release_x = float(new_path.waypoints[3, 0])
        self._initial_path_remaining = new_path.remaining_distance(
            new_path.start_xy
        )
        self._previous_path_remaining = new_path.remaining_distance(base_xy)
        self._terminal_pose_stage = "NONE"
        self._terminal_pose_stable_count = 0
        self._terminal_pose_aligned_latched = False
        self._terminal_arm_stable_count = 0
        self._terminal_arm_aligned_latched = False
        if self.last_reset_info:
            self.last_reset_info["detour_side"] = float(selected)
            self.last_reset_info["box_path"] = new_path.metadata()
            self.last_reset_info["initial_path_length"] = float(
                self._initial_path_remaining
            )
        return selected

    def _place_obstacle(self, requested_center_xy, no_obstacle):
        """Move the real Gazebo box to the logical scenario position."""
        requested_center_xy = self._vector(
            requested_center_xy, 2, "requested_center_xy"
        )
        desired_center = (
            self.obstacle_hidden_center_xy
            if bool(no_obstacle) else requested_center_xy
        )
        position_changed = bool(
            no_obstacle
            or not np.allclose(
                requested_center_xy,
                self.box_center_xy,
                atol=1.0e-9,
            )
        )
        if not self.obstacle_repositioning_enabled:
            if position_changed:
                raise ValueError(
                    "scenario changes the physical obstacle but "
                    "~enable_obstacle_repositioning is false"
                )
            return
        self._move_obstacle_model(desired_center)

    def _clear_obstacle_for_robot_reset(self):
        """Move a dynamic obstacle away before resetting robot joints."""
        if not self.obstacle_repositioning_enabled:
            return
        self._move_obstacle_model(self.obstacle_hidden_center_xy)

    def _move_obstacle_model(self, desired_center):
        desired_center = self._vector(
            desired_center,
            2,
            "desired_obstacle_center_xy",
        )
        model_state = ModelState()
        model_state.model_name = self.obstacle_model_name
        model_state.pose.position.x = float(
            desired_center[0] - self.obstacle_geometry_center_xy[0]
        )
        model_state.pose.position.y = float(
            desired_center[1] - self.obstacle_geometry_center_xy[1]
        )
        model_state.pose.orientation.w = 1.0
        model_state.reference_frame = "world"
        response = self._set_model_state(model_state)
        if not response.success:
            raise RuntimeError(
                "Gazebo obstacle reposition failed: {}".format(
                    response.status_message
                )
            )

    def step(self, fused_action):
        if self._path is None or self.last_sensor_observation is None:
            raise RuntimeError("reset must be called before step")
        previous_path = float(self._previous_path_remaining)
        previous_ee = float(self._previous_ee_distance)
        previous_path_index = int(self._path.index)
        previous_phase = self._action_mask_phase()
        previous_yaw_error = self._path.final_yaw_error(
            float(self.last_sensor_observation[13])
        )

        raw_student_action = self._vector(
            fused_action,
            self.FUSED_ACTION_DIM,
            "fused_action",
        ).astype(np.float32)
        gated_action, gate_info = self._guard_student_action(
            raw_student_action
        )
        gated_action, gate_info = self._apply_student_stall_takeover(
            gated_action,
            gate_info,
        )

        # The parent executes the established eight-dimensional action and
        # safety filter.  Its reward/done are intentionally replaced below so
        # progress along a detour is rewarded even when EE distance temporarily
        # increases near the box.
        _, _, _, info = super(FusedBoxDetourTrainingEnv, self).step(
            gated_action
        )
        sensor = self.last_sensor_observation
        base_xy = np.asarray(sensor[11:13], dtype=np.float64)
        ee_distance = float(np.linalg.norm(sensor[0:3] - sensor[6:9]))
        contact_names = list(info.get("collision_names", []))
        arm_contact = any(
            (
                "link{}_collision".format(index) in name
                or "_arm_bumper" in name
                or "arm_contacts" in name
            )
            for name in contact_names
            for index in range(1, 7)
        )
        contact_collision = bool(
            info.get("collision", False)
            or contact_names
        )
        geometric_collision = self._path.geometric_collision(base_xy)
        collision = bool(contact_collision or geometric_collision)
        collision_source = (
            "arm_contact" if arm_contact
            else "base_contact" if contact_collision
            else "box_geometry" if geometric_collision
            else "none"
        )

        base_yaw = float(sensor[13])
        terminal_pose_aligned = self._update_terminal_pose_state(
            base_xy,
            base_yaw,
        )
        self._path.advance(
            base_xy,
            planar_yaw=base_yaw,
            terminal_pose_aligned=(
                terminal_pose_aligned
                if self._path.final_waypoint_active
                else None
            ),
        )
        self.base_goal_xy = self._path.current_goal_xy
        observation = self._build_box_observation(sensor)
        path_remaining = self._path.remaining_distance(base_xy)
        final_base_position_error = self._path.final_position_error(
            base_xy
        )
        final_base_yaw_error = self._path.final_yaw_error(base_yaw)
        current_phase = self._action_mask_phase()
        terminal_alignment_active = bool(
            previous_path_index >= len(self._path.waypoints) - 1
            or self._path.index >= len(self._path.waypoints) - 1
        )
        arm_finish_active = bool(
            previous_phase == "ARM_FINISH"
            or current_phase == "ARM_FINISH"
        )
        path_progress, ee_progress, yaw_progress = box_reward_progress(
            previous_path,
            path_remaining,
            self._initial_path_remaining,
            previous_ee,
            ee_distance,
            self.last_reset_info.get("initial_ee_distance", 1.0),
            self.success_threshold,
            previous_yaw_error,
            final_base_yaw_error,
            arm_finish_active=arm_finish_active,
            terminal_alignment_active=terminal_alignment_active,
        )
        tf_ok = bool(info.get("tf_ok", False))
        path_complete = bool(self._path.complete)
        success = bool(
            tf_ok
            and path_complete
            and ee_distance <= self.success_threshold
            and not collision
        )
        timeout = bool(self.base_env.step_count >= self.max_steps)
        reward = (
            self.path_reward_scale * float(path_progress)
            + self.ee_reward_scale * float(ee_progress)
            + self.yaw_reward_scale * float(yaw_progress)
            - self.time_penalty
        )
        if collision:
            reward -= self.collision_penalty
        if success:
            reward += self.success_bonus
        elif timeout or not tf_ok:
            reward -= self.timeout_penalty
        done = bool(success or collision or timeout or not tf_ok)

        self._student_takeover_diagnostics = (
            self._student_stall_takeover.update(
                self._path.index,
                len(self._path.waypoints),
                path_remaining,
                ee_distance,
            )
        )

        self._previous_path_remaining = path_remaining
        self._previous_ee_distance = ee_distance
        info = dict(info)
        info.update({
            "dist": ee_distance,
            "success": success,
            "collision": collision,
            "collision_source": collision_source,
            "collision_names": contact_names,
            "arm_contact": arm_contact,
            "geometric_collision": geometric_collision,
            "timeout": timeout,
            "path_complete": path_complete,
            "path_remaining": path_remaining,
            "final_base_position_error": final_base_position_error,
            "final_base_yaw_error": final_base_yaw_error,
            "final_base_pose_aligned": bool(path_complete),
            "terminal_pose_stage": self._terminal_pose_stage,
            "terminal_pose_stable_count": int(
                self._terminal_pose_stable_count
            ),
            "arm_box_release_ready": bool(self._arm_box_release_ready),
            "arm_box_release_margin": float(self._arm_box_release_margin),
            "path_progress": float(path_progress),
            "ee_progress": float(ee_progress),
            "yaw_progress": float(yaw_progress),
            "base_goal_xy": self.base_goal_xy.copy(),
            "box_path_index": int(self._path.index),
            "box_passed": bool(
                self._path.index >= len(self._path.waypoints) - 1
            ),
            "action_mask_phase": self._action_mask_phase(),
            "action_mask": self._action_mask().copy(),
            "box_path_waypoints": self._path.waypoints.copy(),
            "box_detour_side": float(self._path.side),
            "box_policy_action": raw_student_action.copy(),
            "box_gated_action": gated_action.copy(),
            "box_action_gate_reasons": list(gate_info["reasons"]),
            "box_action_gate": dict(gate_info),
            "student_stall_takeover": dict(
                self._student_takeover_diagnostics
            ),
            "box_reward_terms": {
                "path": self.path_reward_scale * float(path_progress),
                "ee": self.ee_reward_scale * float(ee_progress),
                "yaw": self.yaw_reward_scale * float(yaw_progress),
                "time": -self.time_penalty,
                "collision": -self.collision_penalty if collision else 0.0,
                "terminal": self.success_bonus if success else (
                    -self.timeout_penalty if timeout or not tf_ok else 0.0
                ),
            },
            "fusion_state": self.state_builder.diagnostics(),
            "arm_obstacle_hold": bool(self._arm_obstacle_hold),
            "arm_obstacle_diagnostics": dict(
                self._last_arm_obstacle_diagnostics
            ),
            "arm_finish_fallback_used": bool(
                self._arm_finish_fallback_used
            ),
            "arm_finish_joint_error": self._arm_finish_joint_error.copy(),
        })
        self._last_student_action_gate = dict(gate_info)
        return observation, float(reward), done, info

    def step_joint_subgoal(self, fused_action):
        """Execute one HRL low step with the shared box action contract.

        The high-level wrapper owns the fixed joint subgoal, but it must not
        bypass the route-aware action gate.  In particular, arm hold, heading
        rotation stop and final-base hold are safety/control invariants shared
        with the original box task.  The HRL path still supplies its own
        lightweight reward and termination bookkeeping below.
        """
        raw_action = self._vector(
            fused_action,
            self.FUSED_ACTION_DIM,
            "fused_action",
        ).astype(np.float32)
        gated_action, gate_info = self._guard_student_action(raw_action)
        gated_action, gate_info = self._apply_student_stall_takeover(
            gated_action,
            gate_info,
        )
        _, _, _, parent_info = FusedLowLevelTrainingEnv.step(
            self,
            gated_action,
        )
        self._last_student_action_gate = dict(gate_info)
        sensor = self.last_sensor_observation
        base_xy = np.asarray(sensor[11:13], dtype=np.float64)
        base_yaw = float(sensor[13])
        ee_distance = float(np.linalg.norm(sensor[0:3] - sensor[6:9]))
        contact_names = list(parent_info.get("collision_names", []))
        arm_contact = any(
            (
                "link{}_collision".format(index) in name
                or "_arm_bumper" in name
                or "arm_contacts" in name
            )
            for name in contact_names
            for index in range(1, 7)
        )
        contact_collision = bool(
            parent_info.get("collision", False) or contact_names
        )
        geometric_collision = self._path.geometric_collision(base_xy)
        collision = bool(contact_collision or geometric_collision)
        collision_source = (
            "arm_contact" if arm_contact
            else "base_contact" if contact_collision
            else "box_geometry" if geometric_collision
            else "none"
        )
        terminal_pose_aligned = self._update_joint_subgoal_terminal_pose(
            base_xy,
            base_yaw,
            ee_distance,
        )
        # Keep the shared BoxDetourPath authoritative for HRL safe-waypoint
        # control as well as for the legacy staged controller.  Without this
        # advancement, a high-level waypoint policy would chase waypoint 1
        # forever because step_joint_subgoal intentionally bypasses the old
        # route state machine.
        self._path.advance(
            base_xy,
            planar_yaw=base_yaw,
            terminal_pose_aligned=(
                terminal_pose_aligned
                if self._path.final_waypoint_active
                else None
            ),
        )
        self.base_goal_xy = self._path.current_goal_xy
        final_base_position_error = self._path.final_position_error(base_xy)
        final_base_yaw_error = self._path.final_yaw_error(base_yaw)
        tf_ok = bool(parent_info.get("tf_ok", False))
        success = bool(
            tf_ok
            and terminal_pose_aligned
            and ee_distance <= self.success_threshold
            and not collision
        )
        timeout = bool(self.base_env.step_count >= self.max_steps)
        done = bool(success or collision or timeout or not tf_ok)
        observation = self.state_builder.build(
            sensor,
            # HRL is tracking the active fixed subgoal/waypoint here.  Using
            # the final goal for the shared heading gate makes the gate stop
            # forward motion while the chassis is correctly aligned with an
            # intermediate detour waypoint.
            self._path.current_goal_xy,
            action_mask=np.ones(8, dtype=np.float32),
        )
        info = dict(parent_info)
        info.update({
            "dist": ee_distance,
            "success": success,
            "collision": collision,
            "collision_source": collision_source,
            "collision_names": contact_names,
            "arm_contact": arm_contact,
            "geometric_collision": geometric_collision,
            "timeout": timeout,
            "final_base_position_error": final_base_position_error,
            "final_base_yaw_error": final_base_yaw_error,
            "final_base_pose_aligned": bool(terminal_pose_aligned),
            "terminal_pose_stage": self._terminal_pose_stage,
            "terminal_pose_stable_count": int(
                self._terminal_pose_stable_count
            ),
            "terminal_arm_stable_count": int(
                self._terminal_arm_stable_count
            ),
            "terminal_arm_aligned": bool(
                self._terminal_arm_aligned_latched
            ),
            "action_mask_phase": "JOINT_SUBGOAL",
            "action_mask": np.ones(8, dtype=np.float32),
            "box_policy_action": raw_action.copy(),
            "box_gated_action": gated_action.copy(),
            "box_action_gate_reasons": list(
                gate_info.get("reasons", [])
            ),
            "box_action_gate": dict(gate_info),
        })
        return observation, 0.0, done, info

    def _update_joint_subgoal_terminal_pose(
            self, base_xy, base_yaw, ee_distance):
        """Advance BASE_TRANSLATE/BASE_ROTATE/ARM_REACH/ALIGNED."""
        position_error = self._path.final_position_error(base_xy)
        yaw_error = self._path.final_yaw_error(base_yaw)
        (
            self._terminal_pose_stage,
            self._terminal_pose_stable_count,
            self._terminal_arm_stable_count,
            self._terminal_pose_aligned_latched,
            self._terminal_arm_aligned_latched,
        ) = update_terminal_control_state(
            self._terminal_pose_stage,
            self._terminal_pose_stable_count,
            self._terminal_arm_stable_count,
            self._terminal_pose_aligned_latched,
            self._terminal_arm_aligned_latched,
            position_error,
            yaw_error,
            ee_distance,
            self.terminal_pose_enter_position_tolerance,
            self.terminal_pose_exit_position_tolerance,
            self.final_base_yaw_tolerance,
            self.terminal_pose_stable_cycles,
            self.success_threshold,
            self.terminal_arm_stable_cycles,
        )
        return bool(self._terminal_pose_aligned_latched)

    def _teacher_action_for_current_step(self):
        """Reuse the evaluator/collector label or compute it on demand."""
        step = int(self.base_env.step_count)
        if (
                self._cached_teacher_action is None
                or self._cached_teacher_step != step):
            observation = self._build_box_observation(
                self.last_sensor_observation
            )
            action = self.teacher_action(observation)
            if action is None:
                raise RuntimeError("teacher action is unavailable for takeover")
        return self._cached_teacher_action.copy()

    def _apply_student_stall_takeover(self, action, gate_info):
        """Replace only the stalled late-stage components with the teacher."""
        action = np.asarray(action, dtype=np.float32).copy()
        gate_info = dict(gate_info)
        gate_info["reasons"] = list(gate_info.get("reasons", []))
        if not self.enable_student_stall_takeover:
            return action, gate_info

        diagnostics = dict(self._student_takeover_diagnostics)
        base_active = bool(diagnostics.get("base_active", False))
        arm_active = bool(diagnostics.get("arm_active", False))
        if not base_active and not arm_active:
            return action, gate_info

        teacher_action = self._teacher_action_for_current_step()
        if base_active:
            action[0:2] = teacher_action[0:2]
            gate_info["reasons"].append("student_base_stall_takeover")
        if arm_active:
            action[2:8] = teacher_action[2:8]
            gate_info["reasons"].append("student_arm_finish_stall_takeover")
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        gate_info["gated_action"] = action.copy()
        gate_info["teacher_takeover"] = diagnostics
        return action, gate_info

    def _update_terminal_pose_state(self, base_xy, base_yaw):
        """Update the latched final-pose controller state.

        A small translation drift while rotating must not switch the teacher
        back to the translation controller and destroy the yaw alignment it
        just acquired.
        """
        if self._path is None or self._path.complete:
            self._terminal_pose_stage = (
                "ALIGNED" if self._path is not None and self._path.complete
                else "NONE"
            )
            self._terminal_pose_stable_count = 0
            return bool(self._terminal_pose_aligned_latched)
        if not self._path.final_waypoint_active:
            self._terminal_pose_stage = "NONE"
            self._terminal_pose_stable_count = 0
            self._terminal_pose_aligned_latched = False
            return False

        position_error = self._path.final_position_error(base_xy)
        yaw_error = abs(self._path.final_yaw_error(base_yaw))
        if self._terminal_pose_stage in ("NONE", "TRANSLATE"):
            self._terminal_pose_stage = "TRANSLATE"
            self._terminal_pose_stable_count = 0
            if (
                    position_error
                    <= self.terminal_pose_enter_position_tolerance):
                self._terminal_pose_stage = "ROTATE"
        elif self._terminal_pose_stage == "ROTATE":
            if position_error > self.terminal_pose_exit_position_tolerance:
                self._terminal_pose_stage = "TRANSLATE"
                self._terminal_pose_stable_count = 0
            elif yaw_error <= self.final_base_yaw_tolerance:
                self._terminal_pose_stable_count += 1
                if (
                        self._terminal_pose_stable_count
                        >= self.terminal_pose_stable_cycles):
                    self._terminal_pose_stage = "ALIGNED"
                    self._terminal_pose_aligned_latched = True
            else:
                self._terminal_pose_stable_count = 0
        return bool(self._terminal_pose_aligned_latched)

    def _update_coordinated_arm_action(self, action, sensor, path_index):
        """Release the arm only after the complete box exit is clear.

        The observation mask may expose coordinated base/arm support while
        crossing the box, but the teacher must not swing a link across the
        obstacle.  The arm is held until the terminal base pose is aligned
        and the measured base footprint is clear.
        """
        diagnostics = self.teacher_diagnostics()
        sensor = np.asarray(sensor, dtype=np.float64)
        base_xy = np.asarray(sensor[11:13], dtype=np.float64)
        final_index = len(self._path.waypoints) - 1
        arm_release_ready = bool(
            path_index >= final_index
            and self._terminal_pose_stage == "ALIGNED"
            and float(base_xy[0])
            >= float(self._arm_hold_release_x)
            - self._arm_box_release_margin
            and not self._path.geometric_collision(base_xy)
        )
        self._arm_box_release_ready = arm_release_ready
        if not arm_release_ready:
            self._coordinated_arm_blend = 0.0
            action[2:8] = 0.0
            return action

        waypoint_count = len(self._path.waypoints)
        route_fraction = float(path_index - 2) / max(
            float(waypoint_count - 2),
            1.0,
        )
        parent_blend = float(diagnostics.get("arm_blend", 0.0))
        requested_blend = max(
            parent_blend,
            float(np.clip(0.25 + 0.75 * route_fraction, 0.0, 1.0)),
        )
        self._coordinated_arm_blend = max(
            self._coordinated_arm_blend,
            requested_blend,
        )
        cartesian_velocity = diagnostics.get("cartesian_velocity")
        if cartesian_velocity is None:
            return action
        arm_action, _, _, _ = self.teacher.solve_cartesian_velocity(
            sensor,
            np.asarray(cartesian_velocity, dtype=np.float64),
        )
        action[2:8] = (
            self._coordinated_arm_blend * arm_action
        ).astype(np.float32)
        return action

    def _arm_action_with_joint_reference(self, sensor, diagnostics):
        """Add a bounded null-space reference during a stalled arm finish."""
        cartesian_velocity = diagnostics.get("cartesian_velocity")
        if cartesian_velocity is None:
            return None
        cartesian_action, _, _, _ = self.teacher.solve_cartesian_velocity(
            sensor,
            np.asarray(cartesian_velocity, dtype=np.float64),
        )
        sensor = np.asarray(sensor, dtype=np.float64)
        joint_positions = sensor[FusedLowLevelState.JOINT_POSITION_SLICE]
        arm_q = joint_positions[4:10]
        target_q = np.asarray(self.target_arm_positions, dtype=np.float64)
        jacobian = np.asarray(
            self.teacher.jacobian_provider.position_jacobian(
                joint_positions
            ),
            dtype=np.float64,
        )
        max_velocity = np.asarray(
            self.teacher.arm_joint_max_velocity,
            dtype=np.float64,
        )
        scaled_jacobian = jacobian * max_velocity[np.newaxis, :]
        damping = float(self.teacher.dls_damping)
        regularized = np.dot(scaled_jacobian, scaled_jacobian.T)
        regularized += (damping ** 2) * np.eye(3, dtype=np.float64)
        pseudo_inverse = np.dot(
            scaled_jacobian.T,
            np.linalg.inv(regularized),
        )
        null_projector = (
            np.eye(6, dtype=np.float64)
            - np.dot(pseudo_inverse, scaled_jacobian)
        )
        reference = np.clip(
            self._arm_finish_joint_reference_gain
            * (target_q - arm_q)
            / np.maximum(max_velocity, 1.0e-6),
            -1.0,
            1.0,
        )
        result = cartesian_action + np.dot(null_projector, reference)
        blocked = self.teacher._soft_limit_violations(result, arm_q)
        result[blocked] = 0.0
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def _build_box_observation(self, sensor_observation):
        """Build the observation and expose terminal-yaw alignment."""
        observation = self.state_builder.build(
            sensor_observation,
            self.base_goal_xy,
            action_mask=self._action_mask(),
        )
        if self._path is None:
            return observation
        sensor = FusedLowLevelState.as_sensor(sensor_observation)
        base_xy = np.asarray(sensor[11:13], dtype=np.float64)
        base_yaw = float(sensor[13])
        if self._path.complete:
            heading_error = 0.0
        elif (
                self._path.final_waypoint_active
                and self._path.final_position_reached(base_xy)):
            heading_error = self._path.final_yaw_error(base_yaw)
        else:
            return observation
        heading_index = FusedLowLevelState.SUBGOAL_SLICE.start + 2
        observation[heading_index] = float(heading_error)
        self.state_builder.last_diagnostics["heading_error"] = float(
            heading_error
        )
        self.state_builder.last_diagnostics["remaining_subgoal"] = (
            observation[FusedLowLevelState.SUBGOAL_SLICE].copy()
        )
        return observation

    def _action_mask_phase(self):
        """Map the active route waypoint to the three control stages."""
        if self._path is None:
            return "BASE_APPROACH"
        return box_action_mask_phase(
            self._path.index,
            len(self._path.waypoints),
        )

    def _action_mask(self):
        return FusedLowLevelState.action_mask_for_phase(
            self._action_mask_phase()
        )

    def _guard_student_action(self, fused_action):
        """Apply task constraints while preserving the student's choice.

        The teacher never reverses the tracked base or drives forward while
        the heading error is large.  For this box task the arm stays fixed
        until the terminal base pose is aligned.  At ARM_FINISH the base must
        stay put while the arm finishes the EE target.  Enforcing these
        invariants prevents small BC errors from turning into a different
        closed-loop task.
        """
        raw_action = np.asarray(fused_action, dtype=np.float32).copy()
        gated_action = raw_action.copy()
        reasons = []

        path_index = int(self._path.index)
        final_index = len(self._path.waypoints) - 1
        box_passed = bool(path_index >= final_index)
        path_complete = bool(self._path.complete)
        action_mask_phase = self._action_mask_phase()
        diagnostics = self.state_builder.diagnostics()
        heading_error = float(diagnostics.get("heading_error", 0.0))

        if not self.enable_student_action_guard:
            return gated_action, {
                "raw_action": raw_action.copy(),
                "gated_action": gated_action.copy(),
                "reasons": reasons,
                "action_mask_phase": action_mask_phase,
                "heading_error": heading_error,
                "path_index": path_index,
                "box_passed": box_passed,
                "path_complete": path_complete,
            }

        # The teacher's route is forward-only.  A negative policy velocity is
        # an out-of-support BC prediction and makes the chassis back into the
        # obstacle on the lower detour.
        if gated_action[0] < 0.0:
            gated_action[0] = 0.0
            reasons.append("forward_only_clip")

        # Rotate in place until the chassis is aligned with the active route
        # segment; this is the teacher's heading slow-angle rule.
        if (
                not path_complete
                and abs(heading_error) > self.student_heading_slow_angle
                and abs(float(gated_action[0])) > 1.0e-6):
            gated_action[0] = 0.0
            reasons.append("heading_rotation_stop")

        # Match the teacher's actual execution contract: keep the arm fixed
        # until the terminal base position and yaw have been aligned and
        # latched.  Tiny learned arm errors during the long COORDINATED stage
        # otherwise accumulate at joint2 and can enter the chassis guard's
        # hard envelope before ARM_FINISH begins.
        if (
                box_student_arm_hold_required(
                    path_index,
                    len(self._path.waypoints),
                )
                and np.max(np.abs(gated_action[2:8])) > 1.0e-6):
            gated_action[2:8] = 0.0
            reasons.append("arm_until_base_aligned_hold")

        # ARM_FINISH is entered only after the final position and yaw are both
        # reached.  From that point onward only the arm may reduce EE error.
        if (
                action_mask_phase == "ARM_FINISH"
                and np.max(np.abs(gated_action[0:2])) > 1.0e-6):
            gated_action[0:2] = 0.0
            reasons.append("final_base_hold")

        gated_action = np.clip(gated_action, -1.0, 1.0).astype(np.float32)
        return gated_action, {
            "raw_action": raw_action.copy(),
            "gated_action": gated_action.copy(),
            "reasons": reasons,
            "action_mask_phase": action_mask_phase,
            "heading_error": heading_error,
            "path_index": path_index,
            "box_passed": box_passed,
            "path_complete": path_complete,
        }

    def teacher_action(self, fused_observation):
        """Reach the complete terminal base pose before arm-only finish.

        The sampled arm target is defined at a specific world-frame base pose.
        Therefore the final waypoint remains COORDINATED until translation and
        planar yaw both satisfy their tolerances.  A box-specific rotation
        command handles the zero-distance case where the generic point tracker
        intentionally stops producing yaw.  If Cartesian progress stalls in
        ``ARM_FINISH``, a bounded null-space joint reference is added while
        preserving the Cartesian task as the primary objective.
        """
        action = super(FusedBoxDetourTrainingEnv, self).teacher_action(
            fused_observation
        )
        if action is None or self._path is None:
            return action
        action = np.asarray(action, dtype=np.float32).copy()
        sensor = np.asarray(
            fused_observation[FusedLowLevelState.SENSOR_SLICE],
            dtype=np.float64,
        )
        base_xy = np.asarray(sensor[11:13], dtype=np.float64)
        base_x = float(base_xy[0])
        base_yaw = float(sensor[13])
        release_x = float(self._arm_hold_release_x)
        # The active path index is the authoritative task-stage state.  Keep
        # teacher actions consistent with the same mask shown to the student:
        # arm held at indices 1..2, all actions enabled until the final pose is
        # reached, and base held only after path completion.
        path_index = int(self._path.index)
        action_mask_phase = self._action_mask_phase()
        passed_box = bool(path_index >= 5)
        hold = bool(action_mask_phase == "BASE_APPROACH")
        if hold:
            action[2:8] = 0.0
        terminal_pose_aligned = self._update_terminal_pose_state(
            base_xy,
            base_yaw,
        )
        if action_mask_phase == "COORDINATED":
            action = self._update_coordinated_arm_action(
                action,
                sensor,
                path_index,
            )
        final_position_error = self._path.final_position_error(base_xy)
        final_yaw_error = self._path.final_yaw_error(base_yaw)
        terminal_translation_refinement = bool(
            self._path.final_waypoint_active
            and self._terminal_pose_stage == "TRANSLATE"
            and final_position_error
            > self.terminal_pose_enter_position_tolerance
        )
        if terminal_translation_refinement:
            # CoordinatedRuleTeacher intentionally stops at 4 cm for generic
            # tasks.  The box route has a tighter terminal pose contract, so
            # continue its same forward/heading law through the final 2 cm.
            base_diagnostics = self.state_builder.diagnostics()
            base_distance = float(base_diagnostics.get(
                "base_distance", final_position_error
            ))
            heading_error = float(base_diagnostics.get(
                "heading_error", 0.0
            ))
            if base_distance > self._path.final_position_tolerance:
                heading_scale = max(
                    0.0,
                    np.cos(min(abs(heading_error), 0.5 * np.pi)),
                )
                if abs(heading_error) > self.student_heading_slow_angle:
                    heading_scale = 0.0
                action[0] = float(np.clip(
                    self.teacher.base_gain * base_distance * heading_scale,
                    0.0,
                    1.0,
                ))
                action[1] = float(np.clip(
                    self.teacher.yaw_gain * heading_error,
                    -1.0,
                    1.0,
                ))
        terminal_yaw_alignment = bool(
            self._path.final_waypoint_active
            and self._terminal_pose_stage == "ROTATE"
        )
        if terminal_yaw_alignment:
            # Keep the terminal rotation latched until the yaw error is stable
            # for several cycles.  The position hysteresis prevents a small
            # Gazebo drift from re-entering translation immediately.
            action[0] = 0.0
            action[1] = float(np.clip(
                self.teacher.yaw_gain * final_yaw_error,
                -1.0,
                1.0,
            ))
            action[2:8] = 0.0
        elif (
                self._path.final_waypoint_active
                and self._terminal_pose_stage == "ALIGNED"
        ):
            action[0:2] = 0.0

        teacher_diagnostics = self.teacher_diagnostics()
        phase = str(teacher_diagnostics.get("phase", "unknown"))
        arm_distance = float(
            teacher_diagnostics.get("arm_distance", float("inf"))
        )
        blocked_joints = list(
            teacher_diagnostics.get("joint_limit_blocked", [])
        )
        nominal_arm_norm = float(np.linalg.norm(action[2:8]))
        self._arm_finish_fallback_used = False
        self._arm_finish_joint_error = np.zeros(6, dtype=np.float64)
        if action_mask_phase == "ARM_FINISH":
            if arm_distance + 1.0e-4 < self._arm_finish_best_distance:
                self._arm_finish_best_distance = arm_distance
                self._arm_finish_stall_cycles = 0
            else:
                self._arm_finish_stall_cycles += 1
            if self._arm_finish_stall_cycles >= self._arm_finish_stall_limit:
                reference_action = self._arm_action_with_joint_reference(
                    sensor,
                    teacher_diagnostics,
                )
                if reference_action is not None:
                    action[2:8] = reference_action
                    self._arm_finish_fallback_used = True
                    joint_positions = sensor[
                        FusedLowLevelState.JOINT_POSITION_SLICE
                    ]
                    self._arm_finish_joint_error = (
                        np.asarray(self.target_arm_positions, dtype=np.float64)
                        - joint_positions[4:10]
                    )
        else:
            self._arm_finish_stall_cycles = 0
            self._arm_finish_best_distance = float("inf")
        self._arm_obstacle_hold = hold
        self._last_arm_obstacle_diagnostics = {
            "hold": hold,
            "release_x": release_x,
            "base_x": base_x,
            "path_index": path_index,
            "passed_box": passed_box,
            "action_mask_phase": action_mask_phase,
            "coordinated_active": bool(
                action_mask_phase == "COORDINATED"
            ),
            "phase": phase,
            "arm_distance": arm_distance,
            "nominal_arm_norm": nominal_arm_norm,
            "joint_limit_blocked": blocked_joints,
            "terminal_yaw_alignment": terminal_yaw_alignment,
            "terminal_translation_refinement": (
                terminal_translation_refinement
            ),
            "terminal_pose_stage": self._terminal_pose_stage,
            "terminal_pose_aligned": bool(terminal_pose_aligned),
            "terminal_pose_stable_count": int(
                self._terminal_pose_stable_count
            ),
            "final_base_position_error": final_position_error,
            "final_base_yaw_error": final_yaw_error,
            "arm_finish_fallback_used": bool(
                self._arm_finish_fallback_used
            ),
            "arm_finish_stall_cycles": int(
                self._arm_finish_stall_cycles
            ),
        }
        # Apply the same structural support used by the observation and the
        # student gate.  This also guarantees zero base action in ARM_FINISH.
        action *= self._action_mask()
        action = np.clip(action, -1.0, 1.0)
        self._cached_teacher_action = action.astype(np.float32).copy()
        self._cached_teacher_step = int(self.base_env.step_count)
        return action

    def teacher_diagnostics(self):
        """Expose parent phase diagnostics plus box-task finish state."""
        diagnostics = dict(
            super(FusedBoxDetourTrainingEnv, self).teacher_diagnostics()
        )
        nominal_phase = str(diagnostics.get("phase", "UNKNOWN"))
        action_mask_phase = self._action_mask_phase()
        diagnostics.update({
            "arm_obstacle_hold": bool(self._arm_obstacle_hold),
            # Dataset phase labels, observation masks, teacher support and
            # student support must all describe the same three-stage task.
            "teacher_nominal_phase": nominal_phase,
            "phase": action_mask_phase,
            "action_mask_phase": action_mask_phase,
            "action_mask": self._action_mask().copy(),
            "coordinated_arm_blend": float(
                self._coordinated_arm_blend
            ),
            "arm_box_release_ready": bool(self._arm_box_release_ready),
            "box_path_index": int(self._path.index) if self._path is not None else 0,
            "box_passed": bool(
                self._path is not None and self._path.index >= 5
            ),
            "arm_finish_fallback_used": bool(
                self._arm_finish_fallback_used
            ),
            "terminal_pose_stage": self._terminal_pose_stage,
            "terminal_pose_stable_count": int(
                self._terminal_pose_stable_count
            ),
            "terminal_pose_aligned": bool(
                self._terminal_pose_aligned_latched
            ),
            "terminal_yaw_alignment": bool(
                self._terminal_pose_stage == "ROTATE"
            ),
            "terminal_translation_refinement": bool(
                self._terminal_pose_stage == "TRANSLATE"
            ),
            "final_base_position_error": (
                self._path.final_position_error(
                    np.asarray(
                        self.last_sensor_observation[11:13],
                        dtype=np.float64,
                    )
                )
                if self._path is not None
                and self.last_sensor_observation is not None
                else float("nan")
            ),
            "final_base_yaw_error": (
                self._path.final_yaw_error(
                    float(self.last_sensor_observation[13])
                )
                if self._path is not None
                and self.last_sensor_observation is not None
                else float("nan")
            ),
            "arm_finish_joint_error": self._arm_finish_joint_error.copy(),
            "arm_finish_joint_error_norm": float(
                np.linalg.norm(self._arm_finish_joint_error)
            ),
            "arm_finish_stall_cycles": int(
                self._arm_finish_stall_cycles
            ),
        })
        return diagnostics

    @staticmethod
    def _param_or_default(name, default):
        # Importing rospy only happens through the parent environment.  Keep
        # this helper local so unit tests can construct geometry independently.
        import rospy
        return rospy.get_param(name, default)

    @staticmethod
    def _vector(value, size, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name, size, vector.shape
                )
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("{} contains non-finite values".format(name))
        return vector
