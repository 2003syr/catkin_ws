#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import math
import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hrl.laser_scan_sectors import SECTOR_CENTERS, sectorize_scan


class LaserScanSectorsTest(unittest.TestCase):
    def test_five_known_obstacles_map_to_five_sectors(self):
        sample_count = 360
        angle_min = -math.pi
        increment = 2.0 * math.pi / sample_count
        ranges = np.full(sample_count, np.inf, dtype=np.float64)
        expected = [0.6, 1.2, 1.8, 2.4, 3.0]
        for center, distance in zip(SECTOR_CENTERS, expected):
            index = int(round((center - angle_min) / increment)) % sample_count
            ranges[index] = distance

        sectors = sectorize_scan(
            ranges,
            angle_min,
            increment,
            0.05,
            10.0,
        )
        np.testing.assert_allclose(sectors, expected, atol=1e-5)

    def test_no_return_uses_configured_maximum(self):
        sectors = sectorize_scan(
            np.full(180, np.inf),
            -math.pi,
            2.0 * math.pi / 180,
            0.05,
            30.0,
            output_max=10.0,
        )
        np.testing.assert_array_equal(sectors, np.full(5, 10.0))

    def test_invalid_short_ranges_are_ignored(self):
        ranges = np.full(360, np.inf)
        ranges[180] = 0.0
        sectors = sectorize_scan(
            ranges,
            -math.pi,
            2.0 * math.pi / 360,
            0.05,
            10.0,
        )
        self.assertEqual(float(sectors[0]), 10.0)


if __name__ == "__main__":
    unittest.main()
