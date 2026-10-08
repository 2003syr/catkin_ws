#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rule-based high-level generator for feasible planar local subgoals."""

from __future__ import print_function

import math

import numpy as np


def project_path_corridor_subgoal(
        requested_body_xy,
        waypoint_body_xy,
        maximum_radius,
        corridor_half_width,
        minimum_progress=0.0):
    """Project a local goal into a progress-making path corridor.

    The corridor is expressed in the robot body frame and points from the
    current base position to the active path waypoint.  Projection preserves
    as much of the student's command as possible while forbidding backwards
    progress, excessive lateral corner cutting, and an over-long local step.
    It is deliberately independent of ROS so the executable-set contract can
    be unit tested deterministically.
    """
    requested = RuleBasedPlanarSubgoalGenerator._vector(
        requested_body_xy, 2, "requested_body_xy"
    )
    waypoint = RuleBasedPlanarSubgoalGenerator._vector(
        waypoint_body_xy, 2, "waypoint_body_xy"
    )
    maximum_radius = float(maximum_radius)
    corridor_half_width = float(corridor_half_width)
    minimum_progress = float(minimum_progress)
    if maximum_radius <= 0.0:
        raise ValueError("maximum_radius must be positive")
    if corridor_half_width < 0.0:
        raise ValueError("corridor_half_width cannot be negative")
    if minimum_progress < 0.0:
        raise ValueError("minimum_progress cannot be negative")

    waypoint_distance = float(np.linalg.norm(waypoint))
    if waypoint_distance <= 1.0e-12:
        projected = np.zeros(2, dtype=np.float64)
        return projected.astype(np.float32), {
            "applied": bool(np.linalg.norm(requested) > 1.0e-9),
            "requested_along": 0.0,
            "executed_along": 0.0,
            "requested_lateral": 0.0,
            "executed_lateral": 0.0,
            "projection": float(np.linalg.norm(requested)),
        }

    tangent = waypoint / waypoint_distance
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    requested_along = float(np.dot(requested, tangent))
    requested_lateral = float(np.dot(requested, normal))
    maximum_along = min(waypoint_distance, maximum_radius)
    executed_along = float(np.clip(
        requested_along, 0.0, maximum_along
    ))
    if float(np.linalg.norm(requested)) > 1.0e-9:
        executed_along = max(
            executed_along,
            min(minimum_progress, maximum_along),
        )
    radial_lateral_limit = math.sqrt(max(
        0.0, maximum_radius ** 2 - executed_along ** 2
    ))
    lateral_limit = min(corridor_half_width, radial_lateral_limit)
    executed_lateral = float(np.clip(
        requested_lateral, -lateral_limit, lateral_limit
    ))
    projected = (
        executed_along * tangent + executed_lateral * normal
    )
    projection = float(np.linalg.norm(projected - requested))
    return projected.astype(np.float32), {
        "applied": bool(projection > 1.0e-7),
        "requested_along": requested_along,
        "executed_along": executed_along,
        "requested_lateral": requested_lateral,
        "executed_lateral": executed_lateral,
        "projection": projection,
    }


class RuleBasedPlanarSubgoalGenerator(object):
    """Select a short collision-free subgoal from LiDAR angular bins.

    This module is a high-level waypoint selector, not a velocity controller.
    The output remains a body-frame [x, y] subgoal for the learned low level.
    """

    def __init__(
            self,
            scan_bin_count=36,
            minimum_radius=0.20,
            maximum_radius=0.50,
            clearance_margin=0.18,
            footprint_half_length=0.56,
            footprint_half_width=0.10,
            turn_cost=0.15,
            clearance_gain=0.05,
            turn_hysteresis=0.20):
        self.scan_bin_count = int(scan_bin_count)
        self.minimum_radius = float(minimum_radius)
        self.maximum_radius = float(maximum_radius)
        self.clearance_margin = float(clearance_margin)
        self.footprint_half_length = float(
            footprint_half_length
        )
        self.footprint_half_width = float(footprint_half_width)
        self.turn_cost = float(turn_cost)
        self.clearance_gain = float(clearance_gain)
        self.turn_hysteresis = float(turn_hysteresis)
        self._validate()
        self.bin_centers = (
            -math.pi
            + (
                np.arange(self.scan_bin_count, dtype=np.float64) + 0.5
            ) * (2.0 * math.pi / float(self.scan_bin_count))
        )

    def select(
            self,
            final_goal_body_xy,
            scan_ranges,
            preferred_turn_sign=0.0,
            committed_turn_sign=0.0,
            raw_scan_ranges=None):
        goal = self._vector(final_goal_body_xy, 2, "final_goal_body_xy")
        scan = self._vector(
            scan_ranges,
            self.scan_bin_count,
            "scan_ranges",
        )
        if np.any(scan < 0.0):
            raise ValueError("scan_ranges cannot be negative")
        raw_scan = None
        if raw_scan_ranges is not None:
            raw_scan = self._vector(
                raw_scan_ranges,
                self.scan_bin_count,
                "raw_scan_ranges",
            )
            if np.any(raw_scan < 0.0):
                raise ValueError(
                    "raw_scan_ranges cannot be negative"
                )
        goal_distance = float(np.linalg.norm(goal))
        desired_angle = float(math.atan2(goal[1], goal[0]))
        local_direct_clear = True
        if raw_scan is not None:
            local_direct_clear = self.corridor_heading_feasible(
                raw_scan,
                desired_angle,
                goal_distance,
            )
        if (
                goal_distance <= self.minimum_radius
                and local_direct_clear):
            return goal.astype(np.float32), {
                "reason": "final_goal_is_local",
                "selected_angle": desired_angle,
                "selected_radius": goal_distance,
                "candidate_count": 1,
                "turn_sign": float(np.sign(goal[1])),
                "corridor_candidate_count": 1,
            }

        preferred_turn_sign = float(np.sign(preferred_turn_sign))
        committed_turn_sign = float(np.sign(committed_turn_sign))
        angular_error = np.arctan2(
            np.sin(self.bin_centers - desired_angle),
            np.cos(self.bin_centers - desired_angle),
        )
        planning_distance = (
            max(goal_distance, self.minimum_radius)
            if not local_direct_clear
            else goal_distance
        )
        available_radius = np.minimum(
            np.minimum(scan - self.clearance_margin, self.maximum_radius),
            planning_distance,
        )
        feasible = available_radius >= self.minimum_radius
        corridor_feasible = np.ones(
            self.scan_bin_count,
            dtype=np.bool_,
        )
        if raw_scan is not None:
            corridor_feasible = self._corridor_feasible(
                raw_scan,
                available_radius,
            )
            feasible = feasible & corridor_feasible
        feasible_indices = np.flatnonzero(feasible)
        if feasible_indices.size == 0:
            return np.zeros(2, dtype=np.float32), {
                "reason": "no_feasible_lidar_sector",
                "selected_angle": 0.0,
                "selected_radius": 0.0,
                "candidate_count": 0,
                "turn_sign": preferred_turn_sign,
                "corridor_candidate_count": 0,
            }

        desired_index = int(np.argmin(np.abs(angular_error)))
        direct_sector = bool(
            committed_turn_sign == 0.0 and feasible[desired_index]
        )
        committed_sector = False
        if committed_turn_sign != 0.0:
            # A tracked base must finish passing the obstacle before it turns
            # back towards the final goal.  During a committed detour, keep
            # all local goals on the selected side and prefer the forward
            # half-plane.  This prevents left/right and forward/reverse
            # oscillation when the direct ray becomes briefly visible beside
            # a long obstacle.
            same_side = (
                np.sign(self.bin_centers) == committed_turn_sign
            )
            forward = np.cos(self.bin_centers) >= 0.0
            committed_indices = np.flatnonzero(
                feasible & same_side & forward
            )
            if committed_indices.size == 0:
                committed_indices = np.flatnonzero(feasible & same_side)
            if committed_indices.size == 0:
                committed_indices = feasible_indices
            scores = (
                np.abs(angular_error)
                + self.turn_cost * np.abs(self.bin_centers)
                - self.clearance_gain
                * np.minimum(scan, self.maximum_radius * 2.0)
            )
            signs = np.sign(self.bin_centers)
            scores = scores + 4.0 * self.turn_hysteresis * (
                signs != committed_turn_sign
            ).astype(np.float64)
            selected_index = int(
                committed_indices[np.argmin(scores[committed_indices])]
            )
            selected_angle = float(self.bin_centers[selected_index])
            committed_sector = True
        elif direct_sector:
            selected_index = desired_index
            # Do not quantize a clear goal to the center of an even LiDAR
            # bin set; that introduced a persistent five-degree steering bias.
            selected_angle = desired_angle
        else:
            scores = (
                np.abs(angular_error)
                + self.turn_cost * np.abs(self.bin_centers)
                - self.clearance_gain
                * np.minimum(scan, self.maximum_radius * 2.0)
            )
            selection_indices = feasible_indices
            if preferred_turn_sign != 0.0:
                candidate_signs = np.sign(self.bin_centers)
                preferred_indices = np.flatnonzero(
                    feasible
                    & (candidate_signs == preferred_turn_sign)
                )
                if preferred_indices.size > 0:
                    # Episode-level detour memory is a hard side constraint,
                    # not a small score bonus.  A soft penalty allowed a
                    # symmetric frontal obstacle to be reacquired from the
                    # opposite side after release, creating an orbit.
                    selection_indices = preferred_indices
                else:
                    # Wait and turn towards the remembered side instead of
                    # silently selecting the only currently visible sector
                    # on the opposite side.  The task converts this zero
                    # translation into a same-side turn-in-place subgoal.
                    return np.zeros(2, dtype=np.float32), {
                        "reason": "no_feasible_preferred_detour",
                        "selected_angle": (
                            preferred_turn_sign * 0.5 * math.pi
                        ),
                        "selected_radius": 0.0,
                        "candidate_count": int(
                            feasible_indices.size
                        ),
                        "corridor_candidate_count": int(
                            np.count_nonzero(corridor_feasible)
                        ),
                        "turn_sign": preferred_turn_sign,
                        "committed_turn_sign": (
                            committed_turn_sign
                        ),
                    }
            selected_index = int(
                selection_indices[
                    np.argmin(scores[selection_indices])
                ]
            )
            selected_angle = float(self.bin_centers[selected_index])
        selected_radius = float(available_radius[selected_index])
        subgoal = np.asarray([
            selected_radius * math.cos(selected_angle),
            selected_radius * math.sin(selected_angle),
        ], dtype=np.float32)
        return subgoal, {
            "reason": (
                "committed_detour_sector"
                if committed_sector
                else (
                    "direct_free_sector"
                    if direct_sector
                    else "detour_free_sector"
                )
            ),
            "selected_angle": selected_angle,
            "selected_radius": selected_radius,
            "selected_scan": float(scan[selected_index]),
            "candidate_count": int(feasible_indices.size),
            "corridor_candidate_count": int(
                np.count_nonzero(corridor_feasible)
            ),
            "turn_sign": float(np.sign(selected_angle)),
            "committed_turn_sign": committed_turn_sign,
        }

    def _corridor_feasible(self, raw_scan, travel_radii):
        """Check the complete inflated chassis corridor for every heading."""
        feasible = np.ones(
            self.scan_bin_count,
            dtype=np.bool_,
        )
        for candidate_index, candidate_angle in enumerate(
                self.bin_centers):
            travel_radius = float(travel_radii[candidate_index])
            if travel_radius < self.minimum_radius:
                feasible[candidate_index] = False
                continue
            feasible[candidate_index] = (
                self.corridor_heading_feasible(
                    raw_scan,
                    candidate_angle,
                    travel_radius,
                )
            )
        return feasible

    def corridor_heading_feasible(
            self,
            raw_scan,
            heading,
            travel_radius):
        """Check one translated, footprint-inflated heading corridor."""
        raw_scan = self._vector(
            raw_scan,
            self.scan_bin_count,
            "raw_scan",
        )
        relative_angles = np.arctan2(
            np.sin(self.bin_centers - float(heading)),
            np.cos(self.bin_centers - float(heading)),
        )
        longitudinal = raw_scan * np.cos(relative_angles)
        lateral = np.abs(raw_scan * np.sin(relative_angles))
        maximum_longitudinal = (
            max(0.0, float(travel_radius))
            + self.footprint_half_length
            + self.clearance_margin
        )
        inflated_half_width = (
            self.footprint_half_width + self.clearance_margin
        )
        blocking = (
            (longitudinal > 0.0)
            & (longitudinal <= maximum_longitudinal)
            & (lateral <= inflated_half_width)
        )
        return not bool(np.any(blocking))

    def _validate(self):
        if self.scan_bin_count <= 0:
            raise ValueError("scan_bin_count must be positive")
        if self.minimum_radius <= 0.0:
            raise ValueError("minimum_radius must be positive")
        if self.maximum_radius < self.minimum_radius:
            raise ValueError(
                "maximum_radius must be at least minimum_radius"
            )
        if self.clearance_margin < 0.0:
            raise ValueError("clearance_margin cannot be negative")
        if (
                self.footprint_half_length <= 0.0
                or self.footprint_half_width <= 0.0):
            raise ValueError("footprint half extents must be positive")
        if (
                self.turn_cost < 0.0
                or self.clearance_gain < 0.0
                or self.turn_hysteresis < 0.0):
            raise ValueError("subgoal score weights cannot be negative")

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
