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

from training.planar_base_task import PlanarBaseTask
from training.tracked_base_kinematics import wrapped_angle_difference
from training.tracked_base_kinematics import (
    TRACKED_FORWARD_YAW_OFFSET,
    unicycle_to_world_action,
    world_velocity_to_body,
)


def observation(
        remaining=(1.0, -0.5),
        velocity=(0.1, -0.2),
        scan=(10.0, 5.0, 2.0, 1.0, 0.5),
        collision=False,
        yaw=0.0):
    joint_velocity = np.zeros(10, dtype=np.float32)
    heading = yaw + TRACKED_FORWARD_YAW_OFFSET
    joint_velocity[0] = velocity[0] * np.cos(heading)
    joint_velocity[1] = velocity[0] * np.sin(heading)
    joint_velocity[2] = velocity[1]
    return {
        "base_to_target_body_pos": np.asarray(
            [remaining[0], remaining[1], 0.4],
            dtype=np.float32,
        ),
        "planar_base_yaw": float(yaw),
        "planar_base_heading": float(heading),
        "joint_vel": joint_velocity,
        "scan_info": np.asarray(scan, dtype=np.float32),
        "collision": bool(collision),
    }


class PlanarBaseTaskTest(unittest.TestCase):
    def test_reset_yaw_error_wraps_across_complete_turn(self):
        expected = 0.5 * np.pi
        equivalent = expected - 2.0 * np.pi + 0.001
        self.assertAlmostEqual(
            wrapped_angle_difference(equivalent, expected),
            0.001,
            places=6,
        )

    def test_unicycle_mapping_has_zero_body_lateral_velocity(self):
        for yaw in (-2.0, -0.5, 0.0, 0.7, 2.5):
            world_action = unicycle_to_world_action(
                [0.8, -0.4],
                yaw,
                action_scale=0.25,
            )
            body_action = world_velocity_to_body(world_action, yaw)
            self.assertAlmostEqual(body_action[0], 0.2, places=6)
            self.assertAlmostEqual(body_action[1], 0.0, places=6)
            self.assertAlmostEqual(body_action[2], -0.1, places=6)

    def test_compact_state_prefers_position_derived_velocity(self):
        structured = observation(velocity=(9.0, 9.0))
        structured["observed_planar_body_velocity"] = np.asarray([
            0.05,
            0.0,
            -0.10,
        ], dtype=np.float32)
        encoded = PlanarBaseTask().reset(structured)
        np.testing.assert_allclose(encoded[2:4], [0.25, -0.20])

    def test_compact_observation_layout_is_eleven_dimensions(self):
        task = PlanarBaseTask()
        encoded = task.reset(observation())
        self.assertEqual(encoded.shape, (11,))
        np.testing.assert_allclose(encoded[0:2], [1.0, -0.5])
        np.testing.assert_allclose(encoded[2:4], [0.5, -0.4])
        np.testing.assert_allclose(
            encoded[4:9],
            [1.0, 0.5, 0.2, 0.1, 0.05],
        )
        np.testing.assert_array_equal(encoded[9:11], np.zeros(2))

    def test_progress_and_previous_action_are_recorded(self):
        task = PlanarBaseTask(
            action_penalty_scale=0.0,
            smoothness_penalty_scale=0.0,
            time_penalty=0.0,
        )
        task.reset(observation(remaining=(1.0, 0.0)))
        encoded, reward, done, info = task.transition(
            observation(remaining=(0.9, 0.0)),
            np.asarray([0.5, -0.25]),
        )
        self.assertGreater(reward, 0.0)
        self.assertFalse(done)
        self.assertAlmostEqual(info["progress"], 0.1, places=6)
        np.testing.assert_allclose(encoded[9:11], [0.5, -0.25])

    def test_success_terminates_once_inside_threshold(self):
        task = PlanarBaseTask(success_threshold=0.03)
        task.reset(observation(remaining=(0.10, 0.0)))
        _, reward, done, info = task.transition(
            observation(remaining=(0.02, 0.0)),
            np.zeros(2),
        )
        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertFalse(info["collision"])
        self.assertGreater(reward, 0.0)

    def test_collision_overrides_geometric_success(self):
        task = PlanarBaseTask(
            success_threshold=0.03,
            collision_distance=0.20,
        )
        task.reset(observation(remaining=(0.10, 0.0)))
        _, reward, done, info = task.transition(
            observation(
                remaining=(0.02, 0.0),
                scan=(10.0, 10.0, 0.10, 10.0, 10.0),
            ),
            np.zeros(2),
        )
        self.assertTrue(done)
        self.assertFalse(info["success"])
        self.assertTrue(info["collision"])
        self.assertLess(info["collision_penalty"], 0.0)

    def test_contact_sensor_collision_is_terminal(self):
        task = PlanarBaseTask(collision_distance=0.05)
        task.reset(observation())
        _, _, done, info = task.transition(
            observation(collision=True),
            np.zeros(2),
        )
        self.assertTrue(done)
        self.assertTrue(info["collision"])
        self.assertEqual(info["collision_source"], "contact")

    def test_timeout_terminates_at_configured_step(self):
        task = PlanarBaseTask(max_steps=2)
        task.reset(observation())
        _, _, first_done, _ = task.transition(
            observation(),
            np.zeros(2),
        )
        _, _, second_done, info = task.transition(
            observation(),
            np.zeros(2),
        )
        self.assertFalse(first_done)
        self.assertTrue(second_done)
        self.assertTrue(info["timeout"])


if __name__ == "__main__":
    unittest.main()
