#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import unittest

import numpy as np


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(PACKAGE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.dwa_planar_teacher import DWAPlanarTeacher


class DWAPlanarTeacherTest(unittest.TestCase):
    def test_command_is_normalized_to_training_action(self):
        action = DWAPlanarTeacher.normalize_command(
            linear_velocity=0.025,
            yaw_rate=-0.0625,
            maximum_linear_speed=0.05,
            maximum_yaw_rate=0.125,
        )
        np.testing.assert_allclose(action, [0.5, -0.5])

    def test_command_is_clipped_to_action_bounds(self):
        action = DWAPlanarTeacher.normalize_command(
            linear_velocity=0.20,
            yaw_rate=-1.0,
            maximum_linear_speed=0.05,
            maximum_yaw_rate=0.125,
        )
        np.testing.assert_allclose(action, [1.0, -1.0])

    def test_nonpositive_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            DWAPlanarTeacher.normalize_command(
                0.0,
                0.0,
                0.0,
                0.125,
            )


if __name__ == "__main__":
    unittest.main()

