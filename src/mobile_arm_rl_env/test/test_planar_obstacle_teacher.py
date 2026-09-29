#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from training.planar_obstacle_teacher import RuleBasedPlanarAvoidancePolicy
from hrl.laser_scan_sectors import SECTOR_CENTERS


OBSTACLE_MIN = np.asarray([0.75, -0.29], dtype=np.float64)
OBSTACLE_MAX = np.asarray([0.95, 0.03], dtype=np.float64)
LASER_OFFSET = np.asarray([0.0, -0.13], dtype=np.float64)


def ray_box_distance(origin, direction, maximum=10.0):
    near = -float("inf")
    far = float("inf")
    for axis in range(2):
        if abs(direction[axis]) <= 1.0e-9:
            if not OBSTACLE_MIN[axis] <= origin[axis] <= OBSTACLE_MAX[axis]:
                return maximum
            continue
        first = (OBSTACLE_MIN[axis] - origin[axis]) / direction[axis]
        second = (OBSTACLE_MAX[axis] - origin[axis]) / direction[axis]
        near = max(near, min(first, second))
        far = min(far, max(first, second))
    if far < max(near, 0.0):
        return maximum
    return min(maximum, max(0.0, near))


def rotation(angle):
    return np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ], dtype=np.float64)


def simulated_scan(base_position, yaw):
    body_to_world = rotation(yaw)
    origin = (
        np.asarray(base_position, dtype=np.float64)
        + np.dot(body_to_world, LASER_OFFSET)
    )
    return np.asarray([
        ray_box_distance(
            origin,
            np.asarray([
                np.cos(yaw + angle),
                np.sin(yaw + angle),
            ]),
        )
        for angle in SECTOR_CENTERS
    ])


def compact_observation(goal=(1.0, 0.0), scan=(10.0,) * 5):
    observation = np.zeros(11, dtype=np.float32)
    observation[0:2] = goal
    observation[4:9] = np.asarray(scan, dtype=np.float32) / 10.0
    return observation


class RuleBasedPlanarAvoidancePolicyTest(unittest.TestCase):
    def test_clear_path_moves_toward_goal(self):
        policy = RuleBasedPlanarAvoidancePolicy(previous_action_weight=0.0)
        action = policy.predict(compact_observation())
        self.assertGreater(action[0], 0.95)
        self.assertAlmostEqual(float(action[1]), 0.0, places=6)
        self.assertFalse(policy.diagnostics()["avoidance_active"])

    def test_front_obstacle_backs_up_before_turning(self):
        policy = RuleBasedPlanarAvoidancePolicy(previous_action_weight=0.0)
        action = policy.predict(compact_observation(scan=(0.75, 10, 10, 10, 10)))
        diagnostics = policy.diagnostics()
        self.assertTrue(diagnostics["avoidance_active"])
        self.assertTrue(diagnostics["blocking_active"])
        self.assertEqual(diagnostics["blocking_sector"], "front")
        self.assertTrue(diagnostics["detour_active"])
        self.assertTrue(diagnostics["detour_started"])
        self.assertEqual(diagnostics["detour_phase"], "back_up")
        self.assertLess(action[0], 0.0)
        self.assertAlmostEqual(float(action[1]), 0.0, places=6)

    def test_detour_stays_active_after_blocker_leaves_scan(self):
        policy = RuleBasedPlanarAvoidancePolicy(previous_action_weight=0.0)
        policy.predict(compact_observation(
            goal=(1.65, 0.0),
            scan=(0.75, 10.0, 10.0, 10.0, 10.0),
        ))
        action = policy.predict(compact_observation(
            goal=(1.60, -0.1),
            scan=(10.0, 10.0, 10.0, 0.75, 0.75),
        ))
        diagnostics = policy.diagnostics()
        self.assertFalse(diagnostics["blocking_active"])
        self.assertTrue(diagnostics["detour_active"])
        self.assertEqual(diagnostics["detour_phase"], "pass_obstacle")
        self.assertGreater(action[1], 0.0)

    def test_detour_releases_after_obstacle_is_passed(self):
        policy = RuleBasedPlanarAvoidancePolicy(previous_action_weight=0.0)
        policy.predict(compact_observation(
            goal=(1.65, 0.0),
            scan=(0.75, 10.0, 10.0, 10.0, 10.0),
        ))
        policy.predict(compact_observation(
            goal=(1.4, -0.3),
            scan=(10.0, 10.0, 10.0, 0.7, 0.7),
        ))
        diagnostics = policy.diagnostics()
        self.assertTrue(diagnostics["detour_phase_completed"])
        self.assertEqual(diagnostics["detour_phase"], "pass_obstacle")
        self.assertTrue(diagnostics["detour_active"])
        action = None
        completed = False
        for unused_index in range(240):
            action = policy.predict(compact_observation(
                goal=(1.0, -0.2),
                scan=(10.0, 10.0, 10.0, 10.0, 10.0),
            ))
            completed = bool(
                completed or policy.diagnostics()["detour_completed"]
            )
        diagnostics = policy.diagnostics()
        self.assertTrue(completed)
        self.assertFalse(diagnostics["detour_active"])
        self.assertGreater(action[0], 0.0)

    def test_clearer_right_side_is_selected(self):
        policy = RuleBasedPlanarAvoidancePolicy(previous_action_weight=0.0)
        action = policy.predict(compact_observation(
            scan=(0.75, 0.70, 10.0, 10.0, 2.0),
        ))
        self.assertEqual(policy.diagnostics()["avoidance_side"], -1)
        self.assertLess(action[0], 0.0)
        self.assertAlmostEqual(float(action[1]), 0.0, places=6)

    def test_emergency_action_moves_away_from_front_obstacle(self):
        policy = RuleBasedPlanarAvoidancePolicy(previous_action_weight=0.0)
        action = policy.predict(compact_observation(scan=(0.20, 10, 10, 10, 10)))
        diagnostics = policy.diagnostics()
        self.assertTrue(diagnostics["emergency_active"])
        self.assertEqual(diagnostics["emergency_sector"], "front")
        self.assertLess(float(action[0]), 0.0)
        self.assertAlmostEqual(float(action[1]), 0.0, places=6)
        self.assertLessEqual(float(np.max(np.abs(action))), 1.000001)

    def test_obstacle_behind_goal_direction_does_not_create_orbit(self):
        policy = RuleBasedPlanarAvoidancePolicy(previous_action_weight=0.0)
        action = policy.predict(compact_observation(
            goal=(1.0, 0.0),
            scan=(10.0, 10.0, 0.55, 0.55, 10.0),
        ))
        diagnostics = policy.diagnostics()
        self.assertFalse(diagnostics["avoidance_active"])
        self.assertFalse(diagnostics["blocking_active"])
        self.assertGreater(action[0], 0.95)
        self.assertAlmostEqual(float(action[1]), 0.0, places=6)

    def test_output_is_finite_and_normalized(self):
        policy = RuleBasedPlanarAvoidancePolicy()
        action = policy.predict(compact_observation(
            goal=(-1.0, 0.5),
            scan=(0.5, 0.8, 0.6, 0.7, 0.9),
        ))
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertLessEqual(float(np.max(np.abs(action))), 1.000001)

    def test_kinematic_rollouts_pass_blocker_and_reach_targets(self):
        targets = (
            (1.65, -0.13),
            (1.65, 0.10),
            (1.65, -0.40),
            (0.00, 1.20),
            (-1.10, -0.50),
        )
        for target in targets:
            policy = RuleBasedPlanarAvoidancePolicy()
            position = np.zeros(2, dtype=np.float64)
            yaw = 0.0
            minimum_scan = 10.0
            for unused_step in range(1600):
                scan = simulated_scan(position, yaw)
                minimum_scan = min(minimum_scan, float(np.min(scan)))
                goal_world = np.asarray(target) - position
                goal_body = np.dot(rotation(-yaw), goal_world)
                observation = compact_observation(
                    goal=goal_body,
                    scan=scan,
                )
                action = policy.predict(observation)
                # Gazebo evaluation: v_max=0.20 m/s, w_max=0.50 rad/s,
                # action scale=0.25, one environment step per 0.10 seconds.
                linear_velocity = 0.05 * float(action[0])
                yaw_rate = 0.125 * float(action[1])
                position += (
                    0.1 * linear_velocity
                    * np.asarray([np.cos(yaw), np.sin(yaw)])
                )
                yaw = float(np.arctan2(
                    np.sin(yaw + 0.1 * yaw_rate),
                    np.cos(yaw + 0.1 * yaw_rate),
                ))
                if float(np.linalg.norm(np.asarray(target) - position)) < 0.04:
                    break
            self.assertLess(
                float(np.linalg.norm(np.asarray(target) - position)),
                0.04,
                "target={} final_position={} phase={} action={} scan={}".format(
                    target,
                    position.tolist(),
                    policy.diagnostics().get("detour_phase"),
                    action.tolist(),
                    scan.tolist(),
                ),
            )
            self.assertGreater(minimum_scan, 0.18)


if __name__ == "__main__":
    unittest.main()
