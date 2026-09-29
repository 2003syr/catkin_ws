#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import math
import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "scripts",
))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from training.planar_subgoal_task import PlanarSubgoalTask


def observation(
        remaining=(0.8, -0.2),
        velocity=(0.01, -0.025),
        minimum_scan=10.0,
        collision=False,
        base_pose=(0.0, 0.0, 0.0)):
    scan = np.full(36, 10.0, dtype=np.float32)
    scan[18] = float(minimum_scan)
    return {
        "base_to_target_body_pos": np.asarray(
            [remaining[0], remaining[1], 0.4],
            dtype=np.float32,
        ),
        "observed_planar_body_velocity": np.asarray(
            [velocity[0], 0.0, velocity[1]],
            dtype=np.float32,
        ),
        "joint_pos": np.asarray(
            [
                base_pose[0], base_pose[1], base_pose[2],
                0.0, -0.3, 0.03, 0.0, 0.03, 0.0, 0.0,
            ],
            dtype=np.float32,
        ),
        "planar_base_heading": float(base_pose[2]),
        "scan_bins": scan,
        "collision": bool(collision),
    }


class PlanarSubgoalTaskTest(unittest.TestCase):
    def test_observation_contract_is_62_dimensions(self):
        encoded = PlanarSubgoalTask().reset(observation())
        self.assertEqual(encoded.shape, (62,))
        self.assertEqual(encoded[0:2].shape, (2,))
        self.assertEqual(encoded[2:4].shape, (2,))
        self.assertAlmostEqual(
            float(np.linalg.norm(encoded[2:4])),
            1.0,
            places=6,
        )
        self.assertEqual(encoded[4:6].shape, (2,))
        self.assertEqual(encoded[6:42].shape, (36,))
        self.assertEqual(encoded[42:52].shape, (10,))
        self.assertEqual(encoded[52:55].shape, (3,))
        np.testing.assert_array_equal(encoded[55:57], np.zeros(2))
        np.testing.assert_array_equal(encoded[57:62], np.ones(5))

    def test_high_level_uses_detour_when_front_is_blocked(self):
        value = observation(remaining=(1.0, 0.0))
        value["scan_bins"][17:19] = 0.10
        task = PlanarSubgoalTask()
        task.reset(value)
        self.assertEqual(
            task.high_level_info["reason"],
            "detour_free_sector",
        )
        local_body = task._world_to_body(
            task.local_subgoal_world,
            task._base_pose(value),
        )
        self.assertGreater(
            abs(local_body[1]),
            0.05,
        )
        self.assertTrue(task.detour_locked)
        self.assertNotEqual(task.detour_turn_sign, 0.0)

    def test_high_level_rejects_point_ray_that_chassis_cannot_fit(self):
        value = observation(remaining=(1.70, 0.20))
        value["scan_bins"][16] = 0.80
        task = PlanarSubgoalTask()
        task.reset(value)
        self.assertEqual(
            task.high_level_info["reason"],
            "detour_free_sector",
        )
        self.assertTrue(task.detour_locked)

    def test_detour_stays_committed_when_direct_ray_reappears(self):
        blocked = observation(remaining=(1.0, 0.0))
        blocked["scan_bins"][17:19] = 0.10
        task = PlanarSubgoalTask(
            detour_min_lock_updates=0,
            detour_release_cycles=3,
        )
        task.reset(blocked)
        original_sign = task.detour_turn_sign

        clear = observation(remaining=(0.8, 0.0))
        task._update_high_level(clear, reason="test_clear_1")

        self.assertTrue(task.detour_locked)
        self.assertEqual(task.detour_turn_sign, original_sign)
        self.assertEqual(
            task.high_level_info["reason"],
            "committed_detour_sector",
        )

    def test_released_detour_keeps_episode_side_preference(self):
        blocked = observation(remaining=(1.0, 0.0))
        blocked["scan_bins"][17:19] = 0.10
        task = PlanarSubgoalTask(
            detour_min_lock_updates=0,
            detour_release_cycles=2,
        )
        task.reset(blocked)
        original_sign = task.detour_turn_sign
        clear = observation(
            remaining=(0.1, 0.0),
            base_pose=(0.9, 0.0, 0.0),
        )
        task._update_high_level(clear, reason="release_1")
        task._update_high_level(clear, reason="release_2")
        self.assertFalse(task.detour_locked)
        self.assertTrue(task.detour_escape_active)
        self.assertEqual(task.preferred_turn_sign, original_sign)

    def test_clear_scan_does_not_release_before_world_obstacle_pass(self):
        blocked = observation(remaining=(1.0, 0.0))
        blocked["scan_bins"][17:19] = 0.10
        task = PlanarSubgoalTask(
            detour_min_lock_updates=0,
            detour_release_cycles=2,
        )
        task.reset(blocked)

        clear = observation(
            remaining=(0.8, 0.0),
            base_pose=(0.2, 0.0, 0.0),
        )
        task._update_high_level(clear, reason="early_clear_1")
        task._update_high_level(clear, reason="early_clear_2")

        self.assertTrue(task.detour_locked)
        self.assertFalse(
            task.high_level_info["detour_passed_obstacle"]
        )

    def test_passed_world_obstacle_does_not_reacquire_detour(self):
        blocked = observation(remaining=(1.0, 0.0))
        blocked["scan_bins"][17:19] = 0.10
        task = PlanarSubgoalTask(
            detour_min_lock_updates=0,
            detour_release_cycles=1,
        )
        task.reset(blocked)
        original_sign = task.detour_turn_sign

        clear = observation(
            remaining=(0.2, -0.3),
            base_pose=(0.8, 0.3, 0.0),
        )
        task._update_high_level(clear, reason="release")
        self.assertFalse(task.detour_locked)
        self.assertTrue(task.detour_escape_active)

        # Complete the bounded world-frame escape first.  Keeping the same
        # world target at (1.0, 0.0), the target is now behind the base while
        # its heading is still zero.
        escaped = observation(
            remaining=(-0.2, -0.3),
            base_pose=(1.2, 0.3, 0.0),
        )
        task._update_high_level(escaped, reason="escape_complete")
        self.assertFalse(task.detour_locked)
        self.assertFalse(task.detour_escape_active)

        # After a 180-degree chassis rotation the same fixed world target is
        # (0.2, 0.3) in the body frame.  The old obstacle can be visible in
        # front again, but its recorded world progress is behind the base, so
        # it must resume clearance escape without acquiring a new detour lock.
        reacquired = observation(
            remaining=(0.2, 0.3),
            base_pose=(1.2, 0.3, math.pi),
        )
        reacquired["scan_bins"][17:19] = 0.20
        task._update_high_level(reacquired, reason="reacquire")

        self.assertFalse(task.detour_locked)
        self.assertTrue(task.detour_escape_active)
        self.assertEqual(task.preferred_turn_sign, original_sign)
        self.assertEqual(
            task.high_level_info["reason"],
            "post_detour_clearance_escape_reentry",
        )

    def test_progress_is_dense_positive_reward(self):
        task = PlanarSubgoalTask(
            time_penalty=0.0,
            action_penalty_scale=0.0,
            smoothness_penalty_scale=0.0,
            reverse_penalty_scale=0.0,
            idle_penalty=0.0,
        )
        task.reset(observation(remaining=(0.8, 0.0)))
        unused_observation, reward, done, info = task.transition(
            observation(remaining=(0.7, 0.0)),
            [0.5, 0.0],
        )
        self.assertFalse(done)
        self.assertGreater(reward, 0.0)
        self.assertAlmostEqual(info["progress"], 0.1, places=6)

    def test_lidar_proximity_is_terminal(self):
        task = PlanarSubgoalTask(collision_distance=0.18)
        task.reset(observation())
        unused_observation, reward, done, info = task.transition(
            observation(minimum_scan=0.10),
            [0.5, 0.0],
        )
        self.assertTrue(done)
        self.assertTrue(info["collision"])
        self.assertFalse(info["success"])
        self.assertLess(reward, 0.0)

    def test_collision_distance_is_measured_from_footprint_edge(self):
        task = PlanarSubgoalTask(
            collision_distance=0.18,
            footprint_half_length=0.56,
            footprint_half_width=0.10,
        )
        task.reset(observation())
        unused_observation, unused_reward, done, info = task.transition(
            observation(minimum_scan=0.70),
            [0.5, 0.0],
        )
        self.assertTrue(done)
        self.assertTrue(info["proximity_collision"])
        self.assertLess(info["minimum_footprint_clearance"], 0.18)

    def test_rear_proximity_is_not_collision_while_moving_forward(self):
        task = PlanarSubgoalTask(collision_distance=0.18)
        task.reset(observation(remaining=(0.8, 0.0)))
        moving_away = observation(remaining=(0.7, 0.0))
        moving_away["scan_bins"][0] = 0.65
        unused_observation, unused_reward, done, info = task.transition(
            moving_away,
            [0.5, 0.0],
        )
        self.assertFalse(done)
        self.assertFalse(info["proximity_collision"])
        self.assertEqual(info["proximity_collision_sector"], "front")

    def test_rear_proximity_is_collision_while_reversing(self):
        task = PlanarSubgoalTask(collision_distance=0.18)
        task.reset(observation(remaining=(0.8, 0.0)))
        reversing = observation(remaining=(0.9, 0.0))
        reversing["scan_bins"][0] = 0.65
        unused_observation, unused_reward, done, info = task.transition(
            reversing,
            [-0.5, 0.0],
        )
        self.assertTrue(done)
        self.assertTrue(info["proximity_collision"])
        self.assertEqual(info["proximity_collision_sector"], "rear")

    def test_rotation_checks_only_the_side_being_entered(self):
        task = PlanarSubgoalTask(collision_distance=0.18)
        task.reset(observation(remaining=(0.8, 0.0)))
        close_right = observation(remaining=(0.8, 0.0))
        close_right["scan_bins"][9] = 0.25
        unused_observation, unused_reward, done, info = task.transition(
            close_right,
            [0.0, 0.5],
        )
        self.assertFalse(done)
        self.assertFalse(info["proximity_collision"])
        self.assertEqual(
            info["proximity_collision_sector"],
            "rotation_left",
        )

    def test_timeout_has_explicit_terminal_penalty(self):
        task = PlanarSubgoalTask(
            max_steps=1,
            timeout_penalty=2.0,
            progress_reward_scale=0.0,
            time_penalty=0.0,
            action_penalty_scale=0.0,
            smoothness_penalty_scale=0.0,
            reverse_penalty_scale=0.0,
            idle_penalty=0.0,
        )
        task.reset(observation(remaining=(0.8, 0.0)))
        unused_observation, reward, done, info = task.transition(
            observation(remaining=(0.7, 0.0)),
            [0.5, 0.0],
        )
        self.assertTrue(done)
        self.assertTrue(info["timeout"])
        self.assertAlmostEqual(info["timeout_penalty"], -2.0)
        self.assertAlmostEqual(reward, -2.0)


if __name__ == "__main__":
    unittest.main()
