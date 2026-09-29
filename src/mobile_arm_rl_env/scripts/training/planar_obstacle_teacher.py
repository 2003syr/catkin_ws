#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Deterministic nonholonomic obstacle-avoidance teacher.

The policy action is ``[linear_velocity, yaw_rate]``.  It never outputs a
lateral body velocity, so every trajectory can be executed by a tracked or
differential-drive base.
"""

from __future__ import print_function

import math

import numpy as np

from hrl.laser_scan_sectors import SECTOR_NAMES


class RuleBasedPlanarAvoidancePolicy(object):
    """Goal seeking plus a stateful turn/pass obstacle manoeuvre."""

    OBS_DIM = 11
    ACTION_DIM = 2
    SCAN_SLICE = slice(4, 9)

    def __init__(
            self,
            scan_clip=10.0,
            influence_distance=1.00,
            attraction_gain=1.00,
            repulsion_gain=1.40,
            tangential_gain=1.50,
            emergency_margin=0.12,
            emergency_tangent_gain=1.50,
            goal_slow_radius=0.30,
            previous_action_weight=0.80,
            side_clearance_hysteresis=0.10,
            detour_clearance=0.90,
            detour_forward_distance=0.75,
            detour_tolerance=0.10,
            detour_gain=1.50,
            turn_clearance=1.15,
            reverse_speed=0.50,
            minimum_turn_steps=60,
            minimum_pass_steps=240,
            sector_safety_distances=(0.40, 0.40, 0.40, 0.40, 0.40)):
        self.scan_clip = float(scan_clip)
        self.influence_distance = float(influence_distance)
        self.attraction_gain = float(attraction_gain)
        self.repulsion_gain = float(repulsion_gain)
        self.tangential_gain = float(tangential_gain)
        self.emergency_margin = float(emergency_margin)
        self.emergency_tangent_gain = float(emergency_tangent_gain)
        self.goal_slow_radius = float(goal_slow_radius)
        self.previous_action_weight = float(previous_action_weight)
        self.side_clearance_hysteresis = float(
            side_clearance_hysteresis
        )
        self.detour_clearance = float(detour_clearance)
        self.detour_forward_distance = float(detour_forward_distance)
        self.detour_tolerance = float(detour_tolerance)
        self.detour_gain = float(detour_gain)
        self.turn_clearance = float(turn_clearance)
        self.reverse_speed = float(reverse_speed)
        self.minimum_turn_steps = int(minimum_turn_steps)
        self.minimum_pass_steps = int(minimum_pass_steps)
        self.sector_safety_distances = self._vector(
            sector_safety_distances,
            5,
            "sector_safety_distances",
        )
        self._validate_parameters()
        self.previous_action = np.zeros(2, dtype=np.float64)
        self.reset()

    def reset(self):
        self.previous_action.fill(0.0)
        self.avoidance_side = 0
        self.detour_active = False
        self.detour_phase = "none"
        self.side_obstacle_seen = False
        self.side_clear_steps = 0
        self.turn_steps = 0
        self.pass_steps = 0
        self.detour_rearm_required = False
        self.last_diagnostics = {}

    def predict(self, observation):
        observation = self._vector(
            observation,
            self.OBS_DIM,
            "observation",
        )
        goal = observation[0:2]
        goal_distance = float(np.linalg.norm(goal))
        goal_heading = math.atan2(float(goal[1]), float(goal[0]))
        scan = np.clip(
            observation[self.SCAN_SLICE] * self.scan_clip,
            0.0,
            self.scan_clip,
        )
        clearance = scan - self.sector_safety_distances

        front_distance = float(scan[0])
        raw_blocking_active = bool(
            front_distance < self.influence_distance
            and goal_distance > self.detour_tolerance
        )
        blocking_active = bool(
            raw_blocking_active and not self.detour_rearm_required
        )
        emergency_active = bool(
            front_distance
            <= self.sector_safety_distances[0] + self.emergency_margin
        )

        detour_started = False
        detour_completed = False
        detour_phase_completed = False
        if blocking_active and not self.detour_active:
            self.avoidance_side = self._select_avoidance_side(scan, goal)
            self.detour_active = True
            self.detour_phase = (
                "back_up"
                if front_distance < self.turn_clearance
                else "turn_away"
            )
            self.side_obstacle_seen = False
            self.side_clear_steps = 0
            self.turn_steps = 0
            self.pass_steps = 0
            detour_started = True

        if self.detour_active and self.detour_phase == "back_up":
            # A long tracked chassis and folded arm sweep a large radius while
            # rotating.  Reverse straight until that swept envelope is clear.
            if front_distance >= self.turn_clearance:
                self.detour_phase = "turn_away"
                detour_phase_completed = True

        if self.detour_active and self.detour_phase == "turn_away":
            self.turn_steps += 1
            side_distance = self._obstacle_side_distance(scan)
            if side_distance < self.influence_distance:
                self.side_obstacle_seen = True
            # Rotate in place until the obstacle has moved from the front
            # sector to the side opposite the selected avoidance direction.
            if (
                    front_distance >= self.influence_distance
                    and (
                        self.side_obstacle_seen
                        or self.turn_steps >= self.minimum_turn_steps
                    )):
                self.detour_phase = "pass_obstacle"
                # The time-based fallback means the obstacle has swept to the
                # side even if the five-sector reduction missed its narrow
                # angular interval.
                self.side_obstacle_seen = True
                self.pass_steps = 0
                detour_phase_completed = True

        if self.detour_active and self.detour_phase == "pass_obstacle":
            self.pass_steps += 1
            side_distance = self._obstacle_side_distance(scan)
            if side_distance < self.influence_distance:
                self.side_obstacle_seen = True
                self.side_clear_steps = 0
            elif (
                    self.side_obstacle_seen
                    and front_distance >= self.influence_distance):
                self.side_clear_steps += 1
            else:
                self.side_clear_steps = 0
            # Several consecutive clear scans prevent a single coarse-sector
            # transition from ending the pass while the long chassis is still
            # alongside the obstacle.
            if (
                    self.pass_steps >= self.minimum_pass_steps
                    and self.side_clear_steps >= 5):
                self.detour_active = False
                self.detour_phase = "none"
                self.avoidance_side = 0
                self.detour_rearm_required = True
                detour_completed = True

        if self.detour_active and self.detour_phase == "back_up":
            raw_action = np.asarray([
                -self.reverse_speed,
                0.0,
            ], dtype=np.float64)
        elif self.detour_active and self.detour_phase == "turn_away":
            raw_action = np.asarray([
                0.0,
                float(self.avoidance_side),
            ], dtype=np.float64)
        elif self.detour_active and self.detour_phase == "pass_obstacle":
            raw_action = self._pass_action(scan, goal_heading)
        else:
            raw_action = self._goal_action(goal_distance, goal_heading)

        if emergency_active and self.detour_phase != "back_up":
            if self.avoidance_side == 0:
                self.avoidance_side = self._select_avoidance_side(scan, goal)
            # A tracked base can rotate in place.  Stopping translation here
            # is safer and more realistic than the old lateral escape action.
            raw_action = np.asarray([
                0.0,
                float(self.avoidance_side),
            ], dtype=np.float64)

        # Global safety steering uses only the forward-side sectors. Rear
        # obstacles that the base has already passed must not create an orbit.
        left_distance = float(scan[1])
        right_distance = float(scan[4])
        side_denominator = max(
            self.influence_distance
            - max(
                self.sector_safety_distances[1],
                self.sector_safety_distances[4],
            ),
            1.0e-6,
        )
        left_proximity = float(np.clip(
            (self.influence_distance - left_distance) / side_denominator,
            0.0,
            1.0,
        ))
        right_proximity = float(np.clip(
            (self.influence_distance - right_distance) / side_denominator,
            0.0,
            1.0,
        ))
        side_avoidance_active = bool(
            max(left_proximity, right_proximity) > 0.0
        )
        if (
                side_avoidance_active
                and self.detour_phase not in ("back_up", "turn_away")):
            # Obstacles on the right require positive/left yaw and vice
            # versa.  This remains active after the nominal detour so the
            # goal controller cannot cut back across the obstacle corner.
            raw_action[1] += self.tangential_gain * (
                right_proximity - left_proximity
            )
            raw_action[0] *= max(
                0.15,
                1.0 - 0.75 * max(left_proximity, right_proximity),
            )

        raw_action = np.clip(raw_action, -1.0, 1.0)
        if emergency_active or (
                self.detour_active
                and self.detour_phase in ("back_up", "turn_away")):
            action = raw_action
        else:
            action = (
                self.previous_action_weight * self.previous_action
                + (1.0 - self.previous_action_weight) * raw_action
            )
            action = np.clip(action, -1.0, 1.0)
        self.previous_action = action.copy()

        avoidance_active = bool(
            blocking_active
            or self.detour_active
            or emergency_active
            or side_avoidance_active
        )
        self.last_diagnostics = {
            "goal_distance": goal_distance,
            "goal_heading": goal_heading,
            "scan": scan.copy(),
            "minimum_scan": float(np.min(scan)),
            "minimum_clearance": float(np.min(clearance)),
            "avoidance_active": avoidance_active,
            "blocking_active": blocking_active,
            "raw_blocking_active": raw_blocking_active,
            "blocking_sector": "front" if blocking_active else "none",
            "emergency_active": emergency_active,
            "emergency_sector": "front" if emergency_active else "none",
            "side_avoidance_active": side_avoidance_active,
            "left_proximity": left_proximity,
            "right_proximity": right_proximity,
            "avoidance_side": int(self.avoidance_side),
            "detour_active": bool(self.detour_active),
            "detour_started": bool(detour_started),
            "detour_completed": bool(detour_completed),
            "detour_phase_completed": bool(detour_phase_completed),
            "detour_phase": self.detour_phase,
            "detour_rearm_required": bool(
                self.detour_rearm_required
            ),
            "detour_waypoint": np.zeros(2, dtype=np.float64),
            "detour_waypoint_distance": 0.0,
            "side_obstacle_seen": bool(self.side_obstacle_seen),
            "side_clear_steps": int(self.side_clear_steps),
            "turn_steps": int(self.turn_steps),
            "pass_steps": int(self.pass_steps),
            "attraction": self._goal_action(
                goal_distance,
                goal_heading,
            ),
            "waypoint_attraction": np.zeros(2, dtype=np.float64),
            "repulsion": np.zeros(2, dtype=np.float64),
            "tangent": np.zeros(2, dtype=np.float64),
            "raw_action": raw_action.copy(),
            "action": action.copy(),
            "action_semantics": "linear_velocity_yaw_rate",
        }
        return action.astype(np.float32)

    def _goal_action(self, goal_distance, goal_heading):
        angular = float(np.clip(
            self.detour_gain * goal_heading,
            -1.0,
            1.0,
        ))
        alignment = max(0.0, math.cos(goal_heading))
        distance_scale = min(
            1.0,
            goal_distance / self.goal_slow_radius,
        )
        linear = self.attraction_gain * distance_scale * alignment
        if abs(goal_heading) > math.pi / 2.0:
            linear = 0.0
        return np.asarray([
            float(np.clip(linear, 0.0, 1.0)),
            angular,
        ], dtype=np.float64)

    def _pass_action(self, scan, goal_heading):
        side_distance = self._obstacle_side_distance(scan)
        if side_distance < self.influence_distance:
            normalized_error = np.clip(
                (self.detour_clearance - side_distance)
                / max(self.detour_clearance, 1.0e-6),
                0.0,
                1.0,
            )
            # While the obstacle is visible, steer only away from it.  A
            # coarse sector returning scan_clip must never pull the vehicle
            # back toward an obstacle that temporarily disappeared.
            wall_steering = (
                self.avoidance_side
                * self.tangential_gain
                * normalized_error
            )
        else:
            wall_steering = 0.0
        goal_steering = 0.05 * self.detour_gain * goal_heading
        angular = float(np.clip(
            wall_steering + goal_steering,
            -1.0,
            1.0,
        ))
        linear = min(0.70, max(0.25, self.detour_forward_distance))
        if float(scan[0]) < self.influence_distance:
            linear *= np.clip(
                (float(scan[0]) - self.sector_safety_distances[0])
                / max(
                    self.influence_distance
                    - self.sector_safety_distances[0],
                    1.0e-6,
                ),
                0.0,
                1.0,
            )
        return np.asarray([linear, angular], dtype=np.float64)

    def _obstacle_side_distance(self, scan):
        # Turning left places the obstacle on the right, and vice versa.
        indices = (3, 4) if self.avoidance_side > 0 else (1, 2)
        return float(min(scan[indices[0]], scan[indices[1]]))

    def diagnostics(self):
        result = {}
        for key, value in self.last_diagnostics.items():
            result[key] = value.copy() if isinstance(value, np.ndarray) else value
        return result

    def _select_avoidance_side(self, scan, goal):
        left_clearance = float(min(scan[1], scan[2]))
        right_clearance = float(min(scan[3], scan[4]))
        if left_clearance > right_clearance + self.side_clearance_hysteresis:
            return 1
        if right_clearance > left_clearance + self.side_clearance_hysteresis:
            return -1
        return 1 if float(goal[1]) >= 0.0 else -1

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
        if self.scan_clip <= 0.0:
            raise ValueError("scan_clip must be positive")
        if self.influence_distance <= 0.0:
            raise ValueError("influence_distance must be positive")
        if np.any(self.sector_safety_distances < 0.0):
            raise ValueError("sector safety distances must be non-negative")
        if np.any(
                self.sector_safety_distances >= self.influence_distance):
            raise ValueError(
                "sector safety distances must be below influence distance"
            )
        if self.attraction_gain <= 0.0:
            raise ValueError("attraction_gain must be positive")
        if self.repulsion_gain < 0.0 or self.tangential_gain < 0.0:
            raise ValueError("avoidance gains must be non-negative")
        if self.emergency_margin < 0.0:
            raise ValueError("emergency_margin must be non-negative")
        if self.emergency_tangent_gain < 0.0:
            raise ValueError("emergency_tangent_gain must be non-negative")
        if self.goal_slow_radius <= 0.0:
            raise ValueError("goal_slow_radius must be positive")
        if not 0.0 <= self.previous_action_weight < 1.0:
            raise ValueError("previous_action_weight must be in [0, 1)")
        if self.detour_clearance <= 0.0:
            raise ValueError("detour_clearance must be positive")
        if self.detour_forward_distance < 0.0:
            raise ValueError(
                "detour_forward_distance must be non-negative"
            )
        if self.detour_tolerance <= 0.0:
            raise ValueError("detour_tolerance must be positive")
        if self.detour_gain <= 0.0:
            raise ValueError("detour_gain must be positive")
        if self.turn_clearance <= self.influence_distance:
            raise ValueError(
                "turn_clearance must exceed influence_distance"
            )
        if not 0.0 < self.reverse_speed <= 1.0:
            raise ValueError("reverse_speed must be in (0, 1]")
        if self.minimum_turn_steps <= 0:
            raise ValueError("minimum_turn_steps must be positive")
        if self.minimum_pass_steps <= 0:
            raise ValueError("minimum_pass_steps must be positive")
