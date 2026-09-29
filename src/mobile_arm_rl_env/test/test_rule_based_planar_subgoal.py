#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

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

from hrl.rule_based_planar_subgoal import (
    RuleBasedPlanarSubgoalGenerator,
)


class RuleBasedPlanarSubgoalGeneratorTest(unittest.TestCase):
    def test_clear_direction_tracks_final_goal(self):
        generator = RuleBasedPlanarSubgoalGenerator()
        subgoal, info = generator.select(
            [1.0, 0.0],
            np.full(36, 10.0),
        )
        self.assertGreater(subgoal[0], 0.0)
        self.assertLess(abs(subgoal[1]), 0.1)
        self.assertEqual(info["reason"], "direct_free_sector")

    def test_blocked_front_selects_detour(self):
        generator = RuleBasedPlanarSubgoalGenerator()
        scan = np.full(36, 10.0)
        scan[17:19] = 0.10
        subgoal, info = generator.select([1.0, 0.0], scan)
        self.assertGreater(np.linalg.norm(subgoal), 0.0)
        self.assertGreater(abs(subgoal[1]), 0.05)
        self.assertEqual(info["reason"], "detour_free_sector")

    def test_grazing_ray_is_blocked_by_inflated_chassis_corridor(self):
        generator = RuleBasedPlanarSubgoalGenerator(
            footprint_half_length=0.56,
            footprint_half_width=0.10,
            clearance_margin=0.18,
        )
        clearance_scan = np.full(36, 10.0)
        raw_scan = np.full(36, 10.0)
        # The goal ray is clear, but an obstacle one bin to its right lies
        # inside the swept width of the long tracked chassis.
        raw_scan[16] = 0.80
        subgoal, info = generator.select(
            [1.70, 0.20],
            clearance_scan,
            raw_scan_ranges=raw_scan,
        )
        self.assertGreater(np.linalg.norm(subgoal), 0.0)
        self.assertEqual(info["reason"], "detour_free_sector")
        self.assertGreater(abs(info["selected_angle"]), 0.20)

    def test_blocked_local_goal_does_not_bypass_corridor_check(self):
        generator = RuleBasedPlanarSubgoalGenerator()
        clearance_scan = np.full(36, 10.0)
        raw_scan = np.full(36, 10.0)
        raw_scan[17:19] = 0.65
        subgoal, info = generator.select(
            [0.13, 0.0],
            clearance_scan,
            raw_scan_ranges=raw_scan,
        )
        self.assertGreater(np.linalg.norm(subgoal), 0.13)
        self.assertEqual(info["reason"], "detour_free_sector")

    def test_reacquired_detour_obeys_preferred_side_as_hard_constraint(self):
        generator = RuleBasedPlanarSubgoalGenerator()
        scan = np.full(36, 10.0)
        scan[16:18] = 0.10
        subgoal, info = generator.select(
            [1.0, -0.2],
            scan,
            preferred_turn_sign=1.0,
        )
        self.assertEqual(info["reason"], "detour_free_sector")
        self.assertGreater(subgoal[1], 0.0)

    def test_committed_detour_does_not_switch_back_to_direct(self):
        generator = RuleBasedPlanarSubgoalGenerator()
        subgoal, info = generator.select(
            [1.0, 0.0],
            np.full(36, 10.0),
            committed_turn_sign=1.0,
        )
        self.assertGreater(subgoal[0], 0.0)
        self.assertGreater(subgoal[1], 0.0)
        self.assertEqual(info["reason"], "committed_detour_sector")
        self.assertEqual(info["committed_turn_sign"], 1.0)


if __name__ == "__main__":
    unittest.main()
