#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pure-logic contracts for the deterministic geometric navigation stack."""

from __future__ import print_function

import os
import sys
import unittest

import numpy as np


PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PACKAGE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from navigation.local_occupancy_grid import LocalOccupancyGrid
from navigation.regulated_path_tracker import RegulatedPathTracker
from navigation.se2_lattice_planner import SE2LatticePlanner


class GeometricPlanarNavigationTest(unittest.TestCase):

    def test_planner_returns_full_pose_path(self):
        grid = LocalOccupancyGrid(
            resolution=0.05,
            width=4.0,
            height=4.0,
        )
        grid.reset([0.0, 0.0])
        planner = SE2LatticePlanner(
            footprint_half_length=0.20,
            footprint_half_width=0.10,
            safety_margin=0.03,
            maximum_expansions=30000,
        )
        candidates = planner.plan_candidates(
            grid,
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.8, 0.0]),
        )
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["path"].shape[1], 3)
        self.assertGreater(candidates[0]["length"], 0.0)

    def test_tracker_never_outputs_lateral_action(self):
        tracker = RegulatedPathTracker(
            maximum_linear_speed=0.05,
            maximum_yaw_rate=0.125,
        )
        tracker.set_path(np.asarray([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]))
        action, diagnostics = tracker.compute_action(
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0.5, 0.0]),
        )
        self.assertEqual(action.shape, (2,))
        self.assertGreaterEqual(action[0], 0.0)
        self.assertIn(
            diagnostics["mode"],
            ("TRACK_PATH", "ROTATE_TO_SEGMENT"),
        )

    def test_tracker_synchronizes_safety_filtered_action(self):
        tracker = RegulatedPathTracker(
            maximum_linear_speed=0.05,
            maximum_yaw_rate=0.125,
        )
        tracker.synchronize_executed_action([0.0, 0.0])
        np.testing.assert_allclose(
            tracker.previous_action,
            np.zeros(2),
        )


if __name__ == "__main__":
    unittest.main()
