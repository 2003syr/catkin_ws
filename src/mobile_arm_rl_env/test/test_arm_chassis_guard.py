#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import unittest

import numpy as np


SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scripts")
)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hrl.arm_chassis_guard import ArmChassisInterferenceGuard


class TranslatingPoseProvider(object):
    """Move a single protected link along x using joint1."""

    def link_transforms(self, joint_positions, link_names):
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = float(joint_positions[0])
        return dict((name, transform.copy()) for name in link_names)


def guard():
    return ArmChassisInterferenceGuard(
        pose_provider=TranslatingPoseProvider(),
        hard_clearance=0.02,
        soft_clearance=0.30,
        prediction_horizon=0.25,
        trajectory_samples=5,
        capsule_samples=3,
        chassis_center=(0.0, 0.0, 0.0),
        chassis_half_extents=(0.5, 0.5, 0.5),
        link_capsules={
            "link3": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.1),
        },
    )


class ArmChassisInterferenceGuardTest(unittest.TestCase):
    def test_safe_motion_is_unchanged(self):
        model = guard()
        q = np.asarray([1.0, 0, 0, 0, 0, 0], dtype=np.float64)
        velocity = np.asarray([0.2, 0, 0, 0, 0, 0], dtype=np.float64)
        filtered, info = model.filter_velocity(q, velocity)
        np.testing.assert_allclose(filtered, velocity)
        self.assertEqual(info["reason"], "safe")
        self.assertEqual(info["scale"], 1.0)

    def test_soft_zone_scales_motion_toward_chassis(self):
        model = guard()
        q = np.asarray([0.85, 0, 0, 0, 0, 0], dtype=np.float64)
        velocity = np.asarray([-0.4, 0, 0, 0, 0, 0], dtype=np.float64)
        filtered, info = model.filter_velocity(q, velocity)
        self.assertEqual(info["reason"], "chassis_interference_soft")
        self.assertGreater(info["scale"], 0.0)
        self.assertLess(info["scale"], 1.0)
        self.assertLess(abs(filtered[0]), abs(velocity[0]))

    def test_hard_envelope_clips_swept_command(self):
        model = guard()
        q = np.asarray([1.0, 0, 0, 0, 0, 0], dtype=np.float64)
        velocity = np.asarray([-2.0, 0, 0, 0, 0, 0], dtype=np.float64)
        filtered, info = model.filter_velocity(q, velocity)
        self.assertEqual(info["reason"], "chassis_interference_hard")
        self.assertGreaterEqual(info["scale"], 0.0)
        self.assertLess(info["scale"], 1.0)
        clearance, _ = model._trajectory_minimum_clearance(
            q,
            filtered,
            1.0,
        )
        self.assertGreaterEqual(clearance, model.hard_clearance - 1.0e-5)

    def test_interfering_pose_allows_only_recovery(self):
        model = guard()
        q = np.asarray([0.5, 0, 0, 0, 0, 0], dtype=np.float64)
        away = np.asarray([2.0, 0, 0, 0, 0, 0], dtype=np.float64)
        toward = -away

        recovered, recovery_info = model.filter_velocity(q, away)
        blocked, blocked_info = model.filter_velocity(q, toward)

        np.testing.assert_allclose(recovered, away)
        self.assertEqual(
            recovery_info["reason"],
            "chassis_interference_recovery",
        )
        np.testing.assert_allclose(blocked, np.zeros(6))
        self.assertEqual(
            blocked_info["reason"],
            "chassis_interference_hard",
        )


if __name__ == "__main__":
    unittest.main()
