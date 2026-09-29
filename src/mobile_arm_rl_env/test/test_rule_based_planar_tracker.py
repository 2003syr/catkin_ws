#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import unittest
import math

import numpy as np


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "scripts",
))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from training.rule_based_planar_tracker import RuleBasedPlanarTracker


class RuleBasedPlanarTrackerTest(unittest.TestCase):
    def observation(self, goal, previous=(0.0, 0.0)):
        value = np.zeros(62, dtype=np.float32)
        value[0:2] = goal
        yaw = math.atan2(goal[1], goal[0])
        value[2:4] = [math.sin(yaw), math.cos(yaw)]
        value[6:42] = 1.0
        value[55:57] = previous
        value[57:62] = 1.0
        return value

    def test_forward_goal_produces_forward_only_action(self):
        action = RuleBasedPlanarTracker().predict(
            self.observation([0.5, 0.0])
        )
        self.assertGreater(action[0], 0.0)
        self.assertAlmostEqual(action[1], 0.0)

    def test_left_goal_turns_before_driving(self):
        action = RuleBasedPlanarTracker().predict(
            self.observation([0.0, 0.5])
        )
        self.assertAlmostEqual(action[0], 0.0)
        self.assertGreater(action[1], 0.0)

    def test_right_goal_has_negative_yaw_rate(self):
        action = RuleBasedPlanarTracker().predict(
            self.observation([0.0, -0.5])
        )
        self.assertLess(action[1], 0.0)

    def test_terminal_pose_rotates_to_explicit_yaw(self):
        value = self.observation([0.01, 0.0])
        value[2:4] = [1.0, 0.0]
        action = RuleBasedPlanarTracker().predict(value)
        self.assertAlmostEqual(action[0], 0.0)
        self.assertGreater(action[1], 0.0)


if __name__ == "__main__":
    unittest.main()
