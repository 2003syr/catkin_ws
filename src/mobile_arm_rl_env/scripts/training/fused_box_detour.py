#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Geometry-only path utilities for the single-box fused task.

The low-level policy still receives the existing 66-D fused observation.  This
module only owns the deterministic local route used to update the high-level
base sub-goal.  Keeping the route geometry independent of ROS makes it easy to
unit-test the collision margin and progress bookkeeping.
"""

from __future__ import division

import math

import numpy as np


def compose_box_target_configuration(
        current_q,
        target_arm,
        final_base_goal_xy,
        final_base_yaw=None):
    """Build the world-frame FK configuration for the box task target.

    ``base_link`` is the fixed world root in this robot model.  The planar
    ``x`` and ``y`` joints therefore have to be set to the desired final base
    pose before evaluating arm forward kinematics.  Omitting them leaves the
    target at the episode start while the base drives around the box.
    """
    configuration = np.asarray(current_q, dtype=np.float64).copy()
    target_arm = np.asarray(target_arm, dtype=np.float64)
    final_base_goal_xy = np.asarray(final_base_goal_xy, dtype=np.float64)
    if configuration.shape != (10,):
        raise ValueError("current_q must have shape (10,)")
    if target_arm.shape != (6,):
        raise ValueError("target_arm must have shape (6,)")
    if final_base_goal_xy.shape != (2,):
        raise ValueError("final_base_goal_xy must have shape (2,)")
    if not (
            np.all(np.isfinite(configuration))
            and np.all(np.isfinite(target_arm))
            and np.all(np.isfinite(final_base_goal_xy))):
        raise ValueError("box target configuration contains non-finite values")
    configuration[0:2] = final_base_goal_xy
    if final_base_yaw is not None:
        final_base_yaw = float(final_base_yaw)
        if not np.isfinite(final_base_yaw):
            raise ValueError("final_base_yaw must be finite")
        configuration[2] = final_base_yaw
    configuration[4:10] = target_arm
    return configuration


def box_action_mask_phase(path_index, waypoint_count=6):
    """Return the action-support stage for the six-point box route.

    The active waypoint index has the following meaning:

    * 1..2: approach and move to the obstacle-side entry pose;
    * 3..last waypoint: coordinated motion through the obstacle and terminal
      base-pose alignment;
    * one past the last waypoint: hold the base while the arm finishes.
    """
    path_index = int(path_index)
    waypoint_count = int(waypoint_count)
    if waypoint_count < 4:
        raise ValueError("waypoint_count must be at least 4")
    if path_index < 0:
        raise ValueError("path_index must be non-negative")
    # The last waypoint remains a coordinated base/arm target.  ARM_FINISH
    # starts only after that waypoint has actually been reached, represented
    # by an index one past the waypoint array.
    if path_index >= waypoint_count:
        return "ARM_FINISH"
    if path_index <= 2:
        return "BASE_APPROACH"
    return "COORDINATED"


def box_student_arm_hold_required(path_index, waypoint_count=6):
    """Keep the student arm fixed until the final base pose is complete.

    The coordinated rule teacher already follows this contract.  Applying it
    to the student prevents small arm prediction errors from accumulating
    while the base spends hundreds of steps traversing and aligning.
    """
    return box_action_mask_phase(
        path_index,
        waypoint_count,
    ) != "ARM_FINISH"


def resolve_box_detour_side(configured_side, episode_index):
    """Resolve a fixed side or alternate sides when configured with zero."""
    configured_side = float(configured_side)
    episode_index = int(episode_index)
    if episode_index <= 0:
        raise ValueError("episode_index must be positive")
    if abs(configured_side) < 0.5:
        return 1.0 if episode_index % 2 else -1.0
    return 1.0 if configured_side > 0.0 else -1.0


def update_terminal_pose_latch(
        stage,
        stable_count,
        aligned_latched,
        position_error,
        yaw_error,
        enter_position_tolerance,
        exit_position_tolerance,
        yaw_tolerance,
        stable_cycles):
    """Pure terminal-pose hysteresis used by learned high-level control."""
    stage = str(stage)
    stable_count = int(stable_count)
    aligned_latched = bool(aligned_latched)
    if aligned_latched:
        return "ALIGNED", stable_count, True
    if stage in ("NONE", "TRANSLATE"):
        stage = "TRANSLATE"
        stable_count = 0
        if float(position_error) <= float(enter_position_tolerance):
            stage = "ROTATE"
    elif stage == "ROTATE":
        if float(position_error) > float(exit_position_tolerance):
            stage = "TRANSLATE"
            stable_count = 0
        elif abs(float(yaw_error)) <= float(yaw_tolerance):
            stable_count += 1
            if stable_count >= int(stable_cycles):
                stage = "ALIGNED"
                aligned_latched = True
        else:
            stable_count = 0
    return stage, stable_count, aligned_latched


def update_terminal_control_state(
        stage,
        pose_stable_count,
        arm_stable_count,
        pose_aligned_latched,
        arm_aligned_latched,
        position_error,
        yaw_error,
        ee_distance,
        enter_position_tolerance,
        exit_position_tolerance,
        yaw_tolerance,
        pose_stable_cycles,
        ee_tolerance,
        arm_stable_cycles=1):
    """Advance the deterministic TERMINAL controller state.

    The base and arm completion latches are deliberately separate.  Reaching
    the final base pose releases ``ARM_REACH`` but does not claim that the
    complete mobile-manipulation task is aligned.  Once the base latch is set
    it is never reopened by small Gazebo drift, so the arm can finish against
    a stationary chassis.
    """
    stage = str(stage)
    pose_stable_count = int(pose_stable_count)
    arm_stable_count = int(arm_stable_count)
    pose_aligned_latched = bool(pose_aligned_latched)
    arm_aligned_latched = bool(arm_aligned_latched)
    if int(pose_stable_cycles) <= 0 or int(arm_stable_cycles) <= 0:
        raise ValueError("terminal stable cycles must be positive")
    if float(ee_tolerance) <= 0.0:
        raise ValueError("terminal EE tolerance must be positive")

    if arm_aligned_latched:
        return (
            "ALIGNED",
            pose_stable_count,
            arm_stable_count,
            True,
            True,
        )

    if pose_aligned_latched:
        stage = "ARM_REACH"
        if float(ee_distance) <= float(ee_tolerance):
            arm_stable_count += 1
            if arm_stable_count >= int(arm_stable_cycles):
                stage = "ALIGNED"
                arm_aligned_latched = True
        else:
            arm_stable_count = 0
        return (
            stage,
            pose_stable_count,
            arm_stable_count,
            True,
            arm_aligned_latched,
        )

    legacy_stage = {
        "BASE_TRANSLATE": "TRANSLATE",
        "BASE_ROTATE": "ROTATE",
    }.get(stage, stage)
    (
        pose_stage,
        pose_stable_count,
        pose_aligned_latched,
    ) = update_terminal_pose_latch(
        legacy_stage,
        pose_stable_count,
        pose_aligned_latched,
        position_error,
        yaw_error,
        enter_position_tolerance,
        exit_position_tolerance,
        yaw_tolerance,
        pose_stable_cycles,
    )
    stage = {
        "TRANSLATE": "BASE_TRANSLATE",
        "ROTATE": "BASE_ROTATE",
        "ALIGNED": "ARM_REACH",
    }.get(pose_stage, pose_stage)
    if pose_aligned_latched:
        # Evaluate the arm on the next low-level sample.  This guarantees that
        # ARM_REACH is an observable state rather than an instantaneous alias
        # for ALIGNED when the EE happens to start inside its tolerance.
        arm_stable_count = 0
    return (
        stage,
        pose_stable_count,
        arm_stable_count,
        pose_aligned_latched,
        arm_aligned_latched,
    )


def box_reward_progress(
        previous_path_remaining,
        path_remaining,
        initial_path_remaining,
        previous_ee_distance,
        ee_distance,
        initial_ee_distance,
        success_threshold,
        previous_yaw_error,
        yaw_error,
        arm_finish_active=False,
        terminal_alignment_active=False):
    """Return potential differences used by the staged box reward.

    Early EE motion is normalized by the full initial reach distance.  Once
    ARM_FINISH is active, it is normalized by the terminal success threshold
    so the last few centimetres remain visible to PPO.  Yaw progress is only
    active at the terminal waypoint; otherwise route turns could be rewarded
    even when they do not reduce the final pose error.
    """
    initial_path_remaining = max(float(initial_path_remaining), 1.0e-6)
    initial_ee_distance = max(float(initial_ee_distance), 1.0e-6)
    success_threshold = max(float(success_threshold), 1.0e-6)
    path_progress = (
        float(previous_path_remaining) - float(path_remaining)
    ) / initial_path_remaining
    ee_normalizer = (
        success_threshold if arm_finish_active else initial_ee_distance
    )
    ee_progress = (
        float(previous_ee_distance) - float(ee_distance)
    ) / ee_normalizer
    yaw_progress = 0.0
    if terminal_alignment_active:
        yaw_progress = (
            abs(float(previous_yaw_error)) - abs(float(yaw_error))
        ) / math.pi
    return path_progress, ee_progress, yaw_progress


class BoxStudentStallTakeover(object):
    """Latch teacher takeover after late-stage student progress stalls.

    The monitor is deliberately ROS-independent.  It only decides *when* a
    late-stage takeover is needed; the environment still owns the teacher,
    action masks, safety filter, and actual command execution.
    """

    def __init__(
            self,
            base_stall_cycles=20,
            arm_stall_cycles=20,
            minimum_progress=1.0e-4):
        self.base_stall_cycles = int(base_stall_cycles)
        self.arm_stall_cycles = int(arm_stall_cycles)
        self.minimum_progress = float(minimum_progress)
        if self.base_stall_cycles <= 0 or self.arm_stall_cycles <= 0:
            raise ValueError("stall takeover cycles must be positive")
        if self.minimum_progress < 0.0:
            raise ValueError("minimum takeover progress must be non-negative")
        self.reset()

    def reset(self):
        self.base_best = float("inf")
        self.arm_best = float("inf")
        self.base_count = 0
        self.arm_count = 0
        self.base_latched = False
        self.arm_latched = False

    def update(
            self,
            path_index,
            waypoint_count,
            path_remaining,
            ee_distance):
        """Update progress and return the currently active takeovers."""
        path_index = int(path_index)
        waypoint_count = int(waypoint_count)
        path_remaining = float(path_remaining)
        ee_distance = float(ee_distance)
        phase = box_action_mask_phase(path_index, waypoint_count)
        late_base_stage = bool(
            phase == "COORDINATED"
            and path_index >= max(3, waypoint_count - 2)
        )

        if late_base_stage:
            if path_remaining + self.minimum_progress < self.base_best:
                self.base_best = path_remaining
                self.base_count = 0
            else:
                self.base_count += 1
            if self.base_count >= self.base_stall_cycles:
                self.base_latched = True
        elif phase != "ARM_FINISH":
            self.base_best = float("inf")
            self.base_count = 0

        if phase == "ARM_FINISH":
            if ee_distance + self.minimum_progress < self.arm_best:
                self.arm_best = ee_distance
                self.arm_count = 0
            else:
                self.arm_count += 1
            if self.arm_count >= self.arm_stall_cycles:
                self.arm_latched = True
        else:
            self.arm_best = float("inf")
            self.arm_count = 0

        return self.diagnostics(path_index, waypoint_count)

    def diagnostics(self, path_index, waypoint_count):
        phase = box_action_mask_phase(path_index, waypoint_count)
        return {
            "base_active": bool(
                self.base_latched and phase == "COORDINATED"
            ),
            "arm_active": bool(
                self.arm_latched and phase == "ARM_FINISH"
            ),
            "base_latched": bool(self.base_latched),
            "arm_latched": bool(self.arm_latched),
            "base_stall_count": int(self.base_count),
            "arm_stall_count": int(self.arm_count),
            "base_best_path_remaining": float(self.base_best),
            "arm_best_ee_distance": float(self.arm_best),
        }


class BoxDetourPath(object):
    """An axis-aligned multi-point route around a rectangular obstacle.

    The intermediate points deliberately separate longitudinal and lateral
    motion.  This prevents a low-level policy from cutting diagonally across
    an obstacle corner when the active sub-goal changes.
    """

    def __init__(
            self,
            start_xy,
            goal_xy,
            box_center_xy=(0.85, -0.13),
            box_size_xy=(0.20, 0.32),
            base_half_length=0.56,
            base_half_width=0.10,
            clearance=0.06,
            side=1.0,
            waypoint_tolerance=0.10,
            goal_yaw=None,
            final_position_tolerance=0.02,
            final_yaw_tolerance=0.05,
            direct_path=False):
        self.start_xy = self._vector(start_xy, 2, "start_xy")
        requested_goal_xy = self._vector(goal_xy, 2, "goal_xy")
        self.requested_goal_xy = requested_goal_xy.copy()
        self.box_center_xy = self._vector(
            box_center_xy, 2, "box_center_xy"
        )
        self.box_size_xy = self._vector(box_size_xy, 2, "box_size_xy")
        self.base_half_length = float(base_half_length)
        self.base_half_width = float(base_half_width)
        self.clearance = float(clearance)
        self.side = 1.0 if float(side) >= 0.0 else -1.0
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.goal_yaw = (
            None if goal_yaw is None else float(goal_yaw)
        )
        self.final_position_tolerance = float(
            final_position_tolerance
        )
        self.final_yaw_tolerance = float(final_yaw_tolerance)
        # Keep the six-slot route/action-mask contract while replacing the
        # obstacle bypass with a direct final-pose subgoal.
        self.direct_path = bool(direct_path)

        if np.any(self.box_size_xy <= 0.0):
            raise ValueError("box_size_xy must be positive")
        if self.base_half_length <= 0.0 or self.base_half_width <= 0.0:
            raise ValueError("base footprint half extents must be positive")
        if self.clearance < 0.0 or self.waypoint_tolerance <= 0.0:
            raise ValueError("clearance and waypoint_tolerance are invalid")
        if (
                self.final_position_tolerance <= 0.0
                or self.final_yaw_tolerance <= 0.0):
            raise ValueError("final pose tolerances must be positive")
        if self.goal_yaw is not None and not np.isfinite(self.goal_yaw):
            raise ValueError("goal_yaw must be finite")

        if self.direct_path:
            self.goal_xy = requested_goal_xy.copy()
            # Duplicate final points preserve the existing phase/mask
            # contract; advance() consumes them immediately once the direct
            # target pose is reached.
            self.waypoints = np.asarray([
                self.start_xy,
                self.goal_xy,
                self.goal_xy,
                self.goal_xy,
                self.goal_xy,
                self.goal_xy,
            ], dtype=np.float64)
        else:
            half = 0.5 * self.box_size_xy
            # The base center must clear the box in both axes.  The x offsets
            # use the longitudinal chassis half extent; the y offset uses its
            # width.
            before_x = (
                self.box_center_xy[0]
                - half[0]
                - self.base_half_length
                - self.clearance
            )
            after_x = (
                self.box_center_xy[0]
                + half[0]
                + self.base_half_length
                + self.clearance
            )
            # Do not require a reverse motion back into the conservative
            # clearance boundary.  The terminal target is the requested goal
            # when it is safely beyond the box, otherwise the first safe
            # post-box pose.
            self.goal_xy = np.asarray([
                max(float(requested_goal_xy[0]), float(after_x)),
                float(requested_goal_xy[1]),
            ], dtype=np.float64)
            detour_y = self.box_center_xy[1] + self.side * (
                half[1] + self.base_half_width + self.clearance
            )
            start_y = float(self.start_xy[1])
            goal_y = float(self.goal_xy[1])
            self.waypoints = np.asarray([
                self.start_xy,
                [before_x, start_y],
                [before_x, detour_y],
                [after_x, detour_y],
                [after_x, goal_y],
                self.goal_xy,
            ], dtype=np.float64)
        self.segment_lengths = np.linalg.norm(
            np.diff(self.waypoints, axis=0), axis=1
        )
        self.total_length = float(np.sum(self.segment_lengths))
        if self.total_length <= 0.0:
            raise ValueError("box detour route has zero length")
        self.index = 1

    @property
    def current_goal_xy(self):
        return self.waypoints[min(self.index, len(self.waypoints) - 1)].copy()

    @property
    def complete(self):
        return bool(self.index >= len(self.waypoints))

    def advance(
            self,
            position_xy,
            planar_yaw=None,
            terminal_pose_aligned=None):
        """Advance only after the active waypoint pose is reached.

        ``terminal_pose_aligned`` is an optional latched signal from the
        environment's terminal-pose controller.  The path class keeps the
        old instantaneous position/yaw behavior for callers that do not
        provide it, while the box environment can require a stable,
        hysteresis-controlled terminal pose before entering ``ARM_FINISH``.
        """
        position_xy = self._vector(position_xy, 2, "position_xy")
        while not self.complete:
            distance = float(np.linalg.norm(position_xy - self.current_goal_xy))
            final_waypoint = bool(
                self.index == len(self.waypoints) - 1
            )
            # A latched alignment signal already proves that the terminal
            # position and yaw were stable for the configured number of
            # cycles.  Complete the route from that authoritative state even
            # if Gazebo has drifted just outside the strict entry-position
            # tolerance by the time this method observes the next sample.
            if final_waypoint and terminal_pose_aligned is not None:
                if bool(terminal_pose_aligned):
                    self.index += 1
                    continue
                break
            position_tolerance = (
                self.final_position_tolerance
                if final_waypoint else self.waypoint_tolerance
            )
            if distance > position_tolerance:
                break
            if final_waypoint:
                if self.goal_yaw is not None:
                    if planar_yaw is None:
                        break
                    if abs(self.final_yaw_error(planar_yaw)) > self.final_yaw_tolerance:
                        break
            self.index += 1
        return self.current_goal_xy

    def final_position_error(self, position_xy):
        position_xy = self._vector(position_xy, 2, "position_xy")
        return float(np.linalg.norm(position_xy - self.goal_xy))

    def final_yaw_error(self, planar_yaw):
        if self.goal_yaw is None:
            return 0.0
        return self._wrap_angle(self.goal_yaw - float(planar_yaw))

    def final_position_reached(self, position_xy):
        return bool(
            self.final_position_error(position_xy)
            <= self.final_position_tolerance
        )

    @property
    def final_waypoint_active(self):
        return bool(
            not self.complete
            and self.index == len(self.waypoints) - 1
        )

    def remaining_distance(self, position_xy):
        """Approximate arc length from the current position to the goal."""
        position_xy = self._vector(position_xy, 2, "position_xy")
        if self.complete:
            return 0.0
        if self.final_waypoint_active:
            # The safe terminal pose can coincide with the preceding
            # post-box waypoint.  In that case the final segment has zero
            # arc length, so report the actual terminal position error.
            final_segment = self.segment_lengths[-1]
            if final_segment <= 1.0e-12:
                return self.final_position_error(position_xy)
        segment = min(max(self.index - 1, 0), len(self.segment_lengths) - 1)
        start = self.waypoints[segment]
        end = self.waypoints[segment + 1]
        vector = end - start
        denominator = float(np.dot(vector, vector))
        fraction = 0.0
        if denominator > 1.0e-12:
            fraction = float(np.clip(
                np.dot(position_xy - start, vector) / denominator,
                0.0,
                1.0,
            ))
        remaining = (1.0 - fraction) * self.segment_lengths[segment]
        if segment + 1 < len(self.segment_lengths):
            remaining += float(np.sum(self.segment_lengths[segment + 1:]))
        return float(max(0.0, remaining))

    def geometric_collision(self, position_xy):
        """Return collision against the footprint-inflated box."""
        if self.direct_path:
            return False
        position_xy = self._vector(position_xy, 2, "position_xy")
        half = 0.5 * self.box_size_xy
        inflated = np.asarray([
            half[0] + self.base_half_length,
            half[1] + self.base_half_width,
        ])
        delta = np.abs(position_xy - self.box_center_xy)
        return bool(np.all(delta <= inflated + self.clearance * 0.25))

    def metadata(self):
        return {
            "waypoints": self.waypoints.copy(),
            "requested_goal_xy": self.requested_goal_xy.copy(),
            "detour_side": float(self.side),
            "box_center_xy": self.box_center_xy.copy(),
            "box_size_xy": self.box_size_xy.copy(),
            "direct_path": bool(self.direct_path),
            "base_half_length": self.base_half_length,
            "base_half_width": self.base_half_width,
            "clearance": self.clearance,
            "total_length": self.total_length,
            "goal_yaw": self.goal_yaw,
            "final_position_tolerance": self.final_position_tolerance,
            "final_yaw_tolerance": self.final_yaw_tolerance,
        }

    @staticmethod
    def _wrap_angle(angle):
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

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
