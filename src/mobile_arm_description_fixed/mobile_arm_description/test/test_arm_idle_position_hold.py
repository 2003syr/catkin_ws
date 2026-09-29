#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import unittest


SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from arm_idle_position_hold import JointIdleHoldState


class JointIdleHoldStateTest(unittest.TestCase):
    def make_state(self):
        return JointIdleHoldState(
            command_timeout=0.30,
            position_gain=2.0,
            velocity_gain=0.20,
            maximum_velocity=0.05,
        )

    def test_captures_current_position_when_idle(self):
        state = self.make_state()
        command, entered = state.command(-0.30, 0.0, 1.0)
        self.assertTrue(entered)
        self.assertAlmostEqual(command, 0.0)
        self.assertAlmostEqual(state.target_position, -0.30)

    def test_corrects_position_error_and_velocity(self):
        state = self.make_state()
        state.command(-0.30, 0.0, 1.0)
        command, entered = state.command(-0.28, 0.001, 1.1)
        self.assertFalse(entered)
        self.assertLess(command, 0.0)
        self.assertAlmostEqual(command, -0.0402, places=6)

    def test_external_command_temporarily_disables_hold(self):
        state = self.make_state()
        state.command(-0.30, 0.0, 1.0)
        state.note_external_command(2.0)
        command, entered = state.command(-0.20, 0.02, 2.20)
        self.assertIsNone(command)
        self.assertFalse(entered)

        command, entered = state.command(-0.18, 0.0, 2.31)
        self.assertTrue(entered)
        self.assertAlmostEqual(command, 0.0)
        self.assertAlmostEqual(state.target_position, -0.18)

    def test_output_is_velocity_limited(self):
        state = self.make_state()
        state.command(0.0, 0.0, 1.0)
        command, _ = state.command(1.0, 0.0, 1.1)
        self.assertAlmostEqual(command, -0.05)

    def test_repeated_external_zeros_do_not_disable_existing_hold(self):
        state = self.make_state()
        state.command(-0.30, 0.0, 1.0)
        state.note_external_command(1.1, 0.0)
        command, entered = state.command(-0.28, 0.0, 1.1)
        self.assertFalse(entered)
        self.assertLess(command, 0.0)
        self.assertAlmostEqual(state.target_position, -0.30)

    def test_explicit_zero_releases_nonzero_external_command(self):
        state = self.make_state()
        state.command(-0.30, 0.0, 1.0)
        state.note_external_command(2.0, 0.1)
        command, _ = state.command(-0.20, 0.01, 2.1)
        self.assertIsNone(command)

        state.note_external_command(2.1, 0.0)
        command, entered = state.command(-0.20, 0.0, 2.1)
        self.assertTrue(entered)
        self.assertAlmostEqual(command, 0.0)
        self.assertAlmostEqual(state.target_position, -0.20)

    def test_configured_target_avoids_startup_capture_race(self):
        state = self.make_state()
        state.set_hold_target(-0.30)
        command, entered = state.command(0.0, 0.0, 0.1)
        self.assertFalse(entered)
        self.assertAlmostEqual(command, -0.05)
        self.assertAlmostEqual(state.target_position, -0.30)


if __name__ == "__main__":
    unittest.main()
