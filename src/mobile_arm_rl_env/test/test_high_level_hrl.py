#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import sys
import unittest

import numpy as np


SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "scripts",
))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from hrl.high_level_command import (
    HighLevelOption,
    JointHighLevelCommand,
    RouteSide,
    SubgoalType,
)
from hrl.high_level_env import HighLevelEnv
from hrl.high_level_reward import HighLevelReward
from hrl.low_level_wrapper import (
    JointSubgoalResidualLowLevelWrapper,
    JointSubgoalLowLevelWrapper,
    ZeroFusedResidualPolicy,
)
from hrl.rule_high_policy import RuleBasedHighPolicy, SafeWaypointHighPolicy
from hrl.rule_based_planar_subgoal import project_path_corridor_subgoal
from hrl.subgoal_converter import SubgoalConverter
from training.fused_box_detour import (
    update_terminal_control_state,
    update_terminal_pose_latch,
)


def sensor_observation():
    sensor = np.zeros(46, dtype=np.float32)
    sensor[0:3] = [1.0, 0.0, 0.5]
    sensor[3:6] = [1.0, 0.0, 0.0]
    sensor[6:9] = [0.0, 0.0, 0.5]
    sensor[9] = 1.0
    sensor[10] = 1.0
    sensor[11:14] = [0.0, 0.0, 0.5 * math.pi]
    sensor[31:41] = 1.0
    sensor[41:46] = 10.0
    return sensor


class SubgoalEchoPolicy(object):

    def __init__(self):
        self.observations = []

    def predict(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        self.observations.append(observation.copy())
        subgoal = observation[46:52]
        action = np.zeros(8, dtype=np.float32)
        action[0] = subgoal[0]
        action[1] = subgoal[2]
        action[2:5] = subgoal[3:6]
        return np.clip(action, -1.0, 1.0)


class FastSubgoalPolicy(SubgoalEchoPolicy):

    def predict(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        self.observations.append(observation.copy())
        subgoal = observation[46:52]
        action = np.zeros(8, dtype=np.float32)
        action[0] = np.sign(subgoal[0])
        action[1] = np.sign(subgoal[2])
        action[2:5] = np.sign(subgoal[3:6])
        return action


class ResidualTeacherStub(object):

    def residual_components(self, observation, residual, **unused_kwargs):
        del observation, residual, unused_kwargs
        return {"action": np.zeros(8, dtype=np.float32)}


class FakePath(object):

    def __init__(self):
        self.goal_xy = np.asarray([1.0, 0.0], dtype=np.float64)
        self.goal_yaw = 0.5 * math.pi
        self.side = 1.0
        self.waypoints = np.asarray([
            [0.5, 0.0],
            [1.0, 0.0],
        ], dtype=np.float64)
        self.index = 0
        self.complete = False
        self.direct_path = False
        self.final_waypoint_active = False

    @property
    def current_goal_xy(self):
        return self.waypoints[min(self.index, len(self.waypoints) - 1)]

    def remaining_distance(self, base_xy):
        return float(np.linalg.norm(
            self.current_goal_xy - np.asarray(base_xy, dtype=np.float64)
        ))


class FakeLowEnvironment(object):

    def __init__(self):
        self._path = FakePath()
        self.last_sensor_observation = None
        self.last_structured_observation = None
        self.last_reset_info = {}
        self.steps = 0
        self.selected_detour_sides = []

    def reset(self, scenario=None, max_steps=None):
        del scenario, max_steps
        self.steps = 0
        self.last_sensor_observation = sensor_observation()
        self.last_structured_observation = {
            "scan_bins": np.full(36, 10.0, dtype=np.float32),
        }
        self.last_reset_info = {
            "base_goal_xy": np.asarray([1.0, 0.0]),
        }
        return np.zeros(66, dtype=np.float32)

    def select_detour_side(self, side):
        self._path.side = float(side)
        self.selected_detour_sides.append(float(side))

    def step_joint_subgoal(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.steps += 1
        self.last_sensor_observation[11] += 0.01 * action[0]
        self.last_sensor_observation[13] += 0.01 * action[1]
        self.last_sensor_observation[6:9] += 0.01 * action[2:5]
        ee_error = (
            self.last_sensor_observation[0:3]
            - self.last_sensor_observation[6:9]
        )
        self.last_sensor_observation[3:6] = ee_error
        self.last_sensor_observation[10] = np.linalg.norm(ee_error)
        return np.zeros(66), 0.0, False, {
            "success": False,
            "collision": False,
            "timeout": False,
        }


class StuckRotateEnvironment(FakeLowEnvironment):

    def step_joint_subgoal(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.steps += 1
        self.last_sensor_observation[11] += 0.01 * action[0]
        self.last_sensor_observation[6:9] += 0.01 * action[2:5]
        return np.zeros(66), 0.0, False, {
            "success": False,
            "collision": False,
            "timeout": False,
            "safe_fused_action": action.copy(),
        }


class HighLevelHrlTest(unittest.TestCase):

    def test_path_corridor_projection_limits_corner_cutting(self):
        projected, diagnostics = project_path_corridor_subgoal(
            requested_body_xy=[-0.20, 0.30],
            waypoint_body_xy=[0.50, 0.0],
            maximum_radius=0.50,
            corridor_half_width=0.08,
            minimum_progress=0.05,
        )
        np.testing.assert_allclose(projected, [0.05, 0.08], atol=1.0e-6)
        self.assertTrue(diagnostics["applied"])
        self.assertLess(diagnostics["requested_along"], 0.0)
        self.assertAlmostEqual(diagnostics["executed_along"], 0.05)

    def test_detour_guard_projects_only_executable_path_corridor(self):
        low_environment = FakeLowEnvironment()
        wrapper = JointSubgoalLowLevelWrapper(SubgoalEchoPolicy())
        environment = HighLevelEnv(
            low_environment,
            wrapper,
            high_level_interval=1,
            enable_route_options=False,
            detour_feasibility_guard=True,
        )
        environment.reset()
        command = JointHighLevelCommand(
            base_goal=[-0.20, 0.30, 0.0],
            ee_goal=[0.0, 0.0, 0.0],
            subgoal_type=SubgoalType.DETOUR,
        )
        unused_observation, unused_reward, unused_done, info = (
            environment.step(command)
        )
        self.assertTrue(info["detour_guard_applied"])
        self.assertTrue(info["detour_guard_corridor_feasible"])
        self.assertGreater(info["detour_guard_projection"], 0.0)
        np.testing.assert_allclose(
            info["routed_high_command"][0:2],
            [0.05, 0.08],
            atol=1.0e-6,
        )

    def test_terminal_rotation_recovery_reports_no_progress(self):
        low_environment = StuckRotateEnvironment()
        wrapper = JointSubgoalLowLevelWrapper(SubgoalEchoPolicy())
        environment = HighLevelEnv(
            low_environment,
            wrapper,
            high_level_interval=10,
            enable_route_options=False,
            terminal_option_contract=True,
            option_stall_steps=0,
            terminal_rotate_recovery_steps=2,
            terminal_rotate_stall_steps=4,
        )
        environment.reset()
        low_environment._path.final_waypoint_active = True
        low_environment._path.goal_yaw += 0.50
        low_environment._terminal_pose_stage = "ROTATE"
        command = JointHighLevelCommand(
            base_goal=[0.0, 0.0, 0.0],
            ee_goal=[0.0, 0.0, 0.0],
            subgoal_type=SubgoalType.TERMINAL,
        )
        unused_observation, unused_reward, unused_done, info = (
            environment.step(command)
        )
        self.assertEqual(info["option_termination"], "rotate_stalled")
        self.assertEqual(info["low_steps"], 4)
        self.assertEqual(info["terminal_rotate_steps"], 4)
        self.assertEqual(info["terminal_rotate_recovery_steps"], 2)
        self.assertEqual(info["terminal_rotate_stall_events"], 1)
        self.assertEqual(info["terminal_rotate_no_progress_streak"], 4)
        self.assertAlmostEqual(info["terminal_rotate_progress"], 0.0)

    def test_joint_command_normalized_round_trip(self):
        action = np.asarray(
            [0.6, -0.4, 0.2, 0.5, -0.5, 1.0],
            dtype=np.float32,
        )
        command = JointHighLevelCommand.from_action(action)
        np.testing.assert_allclose(
            command.to_action(),
            action,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(command.action_mask, np.ones(8))

    def test_joint_command_carries_three_class_subgoal_type(self):
        command = JointHighLevelCommand(
            base_goal=[0.3, 0.0, 0.0],
            ee_goal=[0.0, 0.0, 0.0],
            subgoal_type=SubgoalType.DETOUR,
            route_side=RouteSide.UPPER,
        )
        self.assertEqual(command.subgoal_type_name, "DETOUR")
        self.assertEqual(command.route_side_name, "UPPER")
        np.testing.assert_array_equal(
            command.subgoal_type_one_hot(),
            [0.0, 1.0, 0.0],
        )
        np.testing.assert_array_equal(
            command.route_side_one_hot(),
            [0.0, 1.0, 0.0],
        )
        self.assertEqual(command.option, HighLevelOption.DETOUR_UPPER)
        self.assertEqual(command.option_name, "DETOUR_UPPER")

    def test_route_side_is_invalid_outside_detour(self):
        with self.assertRaises(ValueError):
            JointHighLevelCommand(
                base_goal=[0.0, 0.0, 0.0],
                ee_goal=[0.0, 0.0, 0.0],
                subgoal_type=SubgoalType.DIRECT,
                route_side=RouteSide.LOWER,
            )

    def test_typed_residual_context_preserves_66_dimensions(self):
        policy = SubgoalEchoPolicy()
        wrapper = JointSubgoalLowLevelWrapper(
            policy,
            encode_subgoal_type=True,
        )
        sensor = sensor_observation()
        wrapper.begin_subgoal(
            JointHighLevelCommand(
                [0.3, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                subgoal_type=SubgoalType.DETOUR,
            ),
            sensor,
        )
        observation = wrapper.build_observation(sensor)
        self.assertEqual(observation.shape, (66,))
        np.testing.assert_array_equal(
            observation[52:58],
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        )

    def test_safe_waypoint_assigns_route_intent(self):
        path = FakePath()
        path.waypoints = np.zeros((6, 2), dtype=np.float64)
        path.direct_path = False
        path.index = 1
        path.final_waypoint_active = False
        self.assertEqual(
            SafeWaypointHighPolicy._subgoal_type(path),
            SubgoalType.DIRECT,
        )
        path.index = 3
        self.assertEqual(
            SafeWaypointHighPolicy._subgoal_type(path),
            SubgoalType.DETOUR,
        )
        path.index = 4
        self.assertEqual(
            SafeWaypointHighPolicy._subgoal_type(path),
            SubgoalType.DIRECT,
        )
        path.index = 5
        path.final_waypoint_active = True
        self.assertEqual(
            SafeWaypointHighPolicy._subgoal_type(path),
            SubgoalType.TERMINAL,
        )

    def test_subgoal_is_fixed_in_world_for_full_high_period(self):
        command = JointHighLevelCommand(
            base_goal=[0.3, 0.0, 0.2],
            ee_goal=[0.1, 0.0, 0.05],
        )
        converter = SubgoalConverter()
        fixed = converter.convert(
            command,
            base_pose_world=[1.0, 2.0, 0.5 * math.pi],
            ee_position_world=[1.2, 2.0, 0.4],
        )
        np.testing.assert_allclose(
            fixed.base_goal_world,
            [1.0, 2.3],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            fixed.ee_goal_world,
            [1.2, 2.1, 0.45],
            atol=1.0e-6,
        )
        original_goal = fixed.base_goal_world.copy()
        remaining = fixed.remaining(
            base_pose_world=[1.0, 2.1, 0.5 * math.pi],
            ee_position_world=[1.2, 2.05, 0.42],
        )
        np.testing.assert_allclose(fixed.base_goal_world, original_goal)
        self.assertAlmostEqual(remaining[0], 0.2, places=5)

    def test_different_subgoals_change_low_observation_and_action(self):
        policy = SubgoalEchoPolicy()
        wrapper = JointSubgoalLowLevelWrapper(policy)
        sensor = sensor_observation()

        wrapper.begin_subgoal(
            JointHighLevelCommand([0.3, 0.0, 0.0], [0.05, 0.0, 0.0]),
            sensor,
        )
        first_action = wrapper.predict(sensor)
        first_observation = policy.observations[-1]
        wrapper.begin_subgoal(
            JointHighLevelCommand([0.0, 0.3, 0.4], [0.0, 0.05, 0.0]),
            sensor,
        )
        second_action = wrapper.predict(sensor)
        second_observation = policy.observations[-1]

        self.assertFalse(np.allclose(first_observation, second_observation))
        self.assertFalse(np.allclose(first_action, second_action))
        np.testing.assert_array_equal(
            first_observation[58:66],
            np.ones(8),
        )

    def test_one_high_step_executes_twenty_low_steps(self):
        low_environment = FakeLowEnvironment()
        wrapper = JointSubgoalLowLevelWrapper(SubgoalEchoPolicy())
        environment = HighLevelEnv(
            low_environment,
            wrapper,
            high_level_interval=20,
        )
        observation = environment.reset()
        self.assertEqual(observation["vector"].shape, (82,))
        np.testing.assert_array_equal(
            observation["vector"][35:38],
            [1.0, 0.0, 0.0],
        )
        np.testing.assert_array_equal(
            observation["vector"][38:42],
            [1.0, 0.0, 0.0, 0.0],
        )
        command = JointHighLevelCommand(
            base_goal=[0.3, 0.0, 0.0],
            ee_goal=[0.1, 0.0, 0.0],
        )
        _, reward, done, info = environment.step(command)
        self.assertEqual(low_environment.steps, 20)
        self.assertEqual(info["low_steps"], 20)
        self.assertFalse(done)
        self.assertTrue(np.isfinite(reward))
        np.testing.assert_allclose(
            info["fixed_base_goal_world"],
            [0.3, 0.0],
            atol=1.0e-6,
        )

    def test_basic_hierarchy_has_no_route_action_or_route_memory(self):
        low_environment = FakeLowEnvironment()
        wrapper = JointSubgoalLowLevelWrapper(SubgoalEchoPolicy())
        environment = HighLevelEnv(
            low_environment,
            wrapper,
            high_level_interval=1,
            enable_route_options=False,
        )
        observation = environment.reset()
        self.assertEqual(observation["vector"].shape, (86,))
        self.assertEqual(observation["basic_path_context"].shape, (8,))
        self.assertGreater(float(observation["basic_path_context"][0]), 0.0)
        np.testing.assert_array_equal(
            observation["vector"][35:38],
            [1.0, 0.0, 0.0],
        )
        command = JointHighLevelCommand(
            base_goal=[0.0, -0.1, 0.0],
            ee_goal=[0.0, 0.0, 0.0],
            subgoal_type=SubgoalType.DETOUR,
        )
        next_observation, unused_reward, unused_done, info = (
            environment.step(command)
        )
        self.assertNotIn("route_side", info)
        self.assertNotIn("option", info)
        self.assertEqual(low_environment.selected_detour_sides, [])
        np.testing.assert_array_equal(
            next_observation["vector"][35:38],
            [0.0, 1.0, 0.0],
        )

    def test_basic_terminal_latch_holds_base_but_keeps_arm_subgoal(self):
        low_environment = FakeLowEnvironment()
        wrapper = JointSubgoalLowLevelWrapper(SubgoalEchoPolicy())
        environment = HighLevelEnv(
            low_environment,
            wrapper,
            high_level_interval=1,
            enable_route_options=False,
        )
        environment.reset()
        low_environment._path.complete = True
        command = JointHighLevelCommand(
            base_goal=[0.3, -0.2, 0.4],
            ee_goal=[0.05, -0.04, 0.03],
            subgoal_type=SubgoalType.TERMINAL,
        )
        unused_observation, unused_reward, unused_done, info = (
            environment.step(command)
        )
        self.assertTrue(info["terminal_base_hold"])
        np.testing.assert_array_equal(info["high_action"][0:3], 0.0)
        np.testing.assert_allclose(
            info["high_action"][3:6],
            np.asarray([0.05, -0.04, 0.03])
            / JointHighLevelCommand.DEFAULT_LIMITS[3:6],
            atol=1.0e-6,
        )

    def test_terminal_contract_blocks_arm_before_final_waypoint(self):
        low_environment = FakeLowEnvironment()
        wrapper = JointSubgoalLowLevelWrapper(SubgoalEchoPolicy())
        environment = HighLevelEnv(
            low_environment,
            wrapper,
            high_level_interval=1,
            enable_route_options=False,
            terminal_option_contract=True,
        )
        environment.reset()
        command = JointHighLevelCommand(
            base_goal=[0.2, 0.0, 0.0],
            ee_goal=[0.1, 0.0, 0.0],
            subgoal_type=SubgoalType.TERMINAL,
        )
        unused_observation, unused_reward, unused_done, info = (
            environment.step(command)
        )
        self.assertTrue(info["invalid_terminal"])
        self.assertFalse(info["terminal_option_latched"])
        np.testing.assert_allclose(
            info["requested_high_command"][3:6], [0.1, 0.0, 0.0]
        )
        np.testing.assert_array_equal(info["high_command"][3:6], 0.0)
        self.assertAlmostEqual(info["ee_command_projection"], 0.1)

    def test_terminal_contract_latches_and_projects_toward_final_goal(self):
        low_environment = FakeLowEnvironment()
        wrapper = JointSubgoalLowLevelWrapper(SubgoalEchoPolicy())
        environment = HighLevelEnv(
            low_environment,
            wrapper,
            high_level_interval=1,
            enable_route_options=False,
            terminal_option_contract=True,
            terminal_ee_step=0.08,
            terminal_student_blend=0.20,
        )
        environment.reset()
        low_environment._path.final_waypoint_active = True
        terminal = JointHighLevelCommand(
            base_goal=[-0.2, 0.1, 0.3],
            ee_goal=[-0.1, 0.0, 0.0],
            subgoal_type=SubgoalType.TERMINAL,
        )
        unused_observation, unused_reward, unused_done, info = (
            environment.step(terminal)
        )
        self.assertTrue(info["terminal_option_latched"])
        self.assertFalse(info["invalid_terminal"])
        self.assertGreater(float(info["high_command"][0]), 0.0)
        self.assertGreater(float(info["high_command"][3]), 0.0)
        self.assertLessEqual(
            float(np.linalg.norm(info["high_command"][3:6])),
            0.08 + 1.0e-6,
        )
        self.assertAlmostEqual(
            info["terminal_ee_requested_norm"], 0.1, places=6
        )
        self.assertGreater(info["terminal_ee_executed_norm"], 0.0)
        self.assertLess(
            info["terminal_ee_student_alignment_cosine"], 0.0
        )
        self.assertGreater(
            info["terminal_ee_executed_alignment_cosine"], 0.0
        )
        self.assertGreater(info["terminal_ee_alignment_cosine"], 0.0)
        self.assertAlmostEqual(
            info["terminal_ee_alignment_cosine"],
            info["terminal_ee_executed_alignment_cosine"],
            places=7,
        )
        self.assertGreater(info["terminal_ee_projection"], 0.0)
        self.assertAlmostEqual(
            info["terminal_ee_projection"],
            float(np.linalg.norm(
                info["high_command"][3:6]
                - np.asarray([-0.1, 0.0, 0.0], dtype=np.float32)
            )),
            places=6,
        )
        np.testing.assert_allclose(
            info["requested_high_command"],
            [-0.2, 0.1, 0.3, -0.1, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            info["routed_high_command"],
            info["requested_high_command"],
        )
        np.testing.assert_allclose(
            info["high_command_projection"],
            info["high_command"] - info["requested_high_command"],
        )
        self.assertGreater(info["base_command_projection"], 0.0)
        self.assertAlmostEqual(
            info["ee_command_projection"],
            info["terminal_ee_projection"],
            places=6,
        )
        np.testing.assert_array_equal(
            info["low_safety_projection_mean_abs"], 0.0
        )
        np.testing.assert_array_equal(
            info["low_safety_projection_max_abs"], 0.0
        )
        self.assertEqual(info["low_safety_projection_max"], 0.0)
        self.assertFalse(info["terminal_ee_progress_fallback"])
        self.assertAlmostEqual(info["terminal_student_blend"], 0.20)

        environment.set_terminal_student_blend(0.60)
        self.assertAlmostEqual(environment.terminal_student_blend, 0.60)
        with self.assertRaises(ValueError):
            environment.set_terminal_student_blend(1.01)

        direct = JointHighLevelCommand(
            base_goal=[0.0, 0.0, 0.0],
            ee_goal=[0.0, 0.0, 0.0],
            subgoal_type=SubgoalType.DIRECT,
        )
        unused_observation, unused_reward, unused_done, info = (
            environment.step(direct)
        )
        self.assertTrue(info["terminal_stage_forced"])
        self.assertEqual(info["subgoal_type"], SubgoalType.TERMINAL)

    def test_detour_route_is_selected_once_and_latched(self):
        low_environment = FakeLowEnvironment()
        wrapper = JointSubgoalLowLevelWrapper(SubgoalEchoPolicy())
        environment = HighLevelEnv(
            low_environment,
            wrapper,
            high_level_interval=1,
        )
        environment.reset()
        upper = JointHighLevelCommand(
            base_goal=[0.0, 0.1, 0.0],
            ee_goal=[0.0, 0.0, 0.0],
            subgoal_type=SubgoalType.DETOUR,
            route_side=RouteSide.UPPER,
        )
        observation, unused_reward, unused_done, info = environment.step(
            upper
        )
        self.assertEqual(info["route_side"], RouteSide.UPPER)
        self.assertEqual(low_environment.selected_detour_sides, [1.0])
        np.testing.assert_array_equal(
            observation["vector"][35:38],
            [0.0, 1.0, 0.0],
        )

        lower = JointHighLevelCommand(
            base_goal=[0.0, -0.1, 0.0],
            ee_goal=[0.0, 0.0, 0.0],
            subgoal_type=SubgoalType.DETOUR,
            route_side=RouteSide.LOWER,
        )
        unused_observation, unused_reward, unused_done, info = (
            environment.step(lower)
        )
        self.assertEqual(info["route_side"], RouteSide.UPPER)
        self.assertEqual(low_environment.selected_detour_sides, [1.0])

    def test_high_step_finishes_after_stable_subgoal_tracking(self):
        low_environment = FakeLowEnvironment()
        wrapper = JointSubgoalLowLevelWrapper(FastSubgoalPolicy())
        environment = HighLevelEnv(
            low_environment,
            wrapper,
            high_level_interval=20,
            base_subgoal_tolerance=0.01,
            yaw_subgoal_tolerance=0.01,
            ee_subgoal_tolerance=0.01,
            subgoal_stable_cycles=3,
        )
        environment.reset()
        command = JointHighLevelCommand(
            # Keep the target halfway between the fake controller's 1 cm
            # increments.  A target exactly on the 1 cm tolerance boundary
            # makes this contract test depend on NumPy/Python float rounding.
            base_goal=[0.055, 0.0, 0.0],
            ee_goal=[0.0, 0.0, 0.0],
        )
        _, _, done, info = environment.step(command)
        self.assertFalse(done)
        self.assertTrue(info["subgoal_done"])
        self.assertEqual(info["subgoal_stable_count"], 3)
        self.assertEqual(info["low_steps"], 7)
        self.assertEqual(info["option_termination"], "subgoal_done")

    def test_event_driven_option_returns_control_when_stalled(self):
        low_environment = FakeLowEnvironment()
        wrapper = JointSubgoalLowLevelWrapper(SubgoalEchoPolicy())
        environment = HighLevelEnv(
            low_environment,
            wrapper,
            high_level_interval=20,
            option_min_low_steps=1,
            option_stall_steps=3,
            option_minimum_progress=10.0,
        )
        environment.reset()
        command = JointHighLevelCommand(
            base_goal=[0.3, 0.0, 0.0],
            ee_goal=[0.0, 0.0, 0.0],
        )
        observation, _, done, info = environment.step(command)
        self.assertFalse(done)
        self.assertEqual(info["option_termination"], "stalled")
        self.assertEqual(info["low_steps"], 3)
        self.assertEqual(observation["vector"].shape, (82,))
        self.assertEqual(float(observation["vector"][45]), 1.0)

    def test_minimal_reward_penalizes_no_subgoal_progress(self):
        model = HighLevelReward()
        reward, info = model.calculate(
            1.0,
            1.0,
            np.ones(6),
            np.ones(6),
        )
        self.assertEqual(reward, -5.0)
        self.assertTrue(info["subgoal_failed"])

    def test_high_reward_credits_path_progress_around_obstacle(self):
        model = HighLevelReward(
            progress_weight=2.0,
            path_progress_weight=10.0,
        )
        reward, info = model.calculate(
            1.0,
            1.1,
            np.ones(6),
            np.ones(6) * 0.9,
            start_path_remaining=2.0,
            end_path_remaining=1.8,
        )
        self.assertAlmostEqual(info["path_progress"], 0.2)
        self.assertGreater(reward, 0.0)

    def test_terminal_reward_credits_pose_progress_and_penalizes_delay(self):
        model = HighLevelReward(
            progress_weight=0.0,
            path_progress_weight=0.0,
            final_base_progress_weight=5.0,
            final_yaw_progress_weight=2.0,
            terminal_progress_weight=20.0,
            subgoal_failure_penalty=0.0,
            low_step_penalty=0.01,
            option_stall_penalty=3.0,
            invalid_terminal_penalty=4.0,
        )
        reward, info = model.calculate(
            0.5,
            0.4,
            np.ones(6),
            np.ones(6) * 0.9,
            start_final_base_distance=0.3,
            end_final_base_distance=0.2,
            start_final_yaw_error=0.2,
            end_final_yaw_error=0.1,
            terminal_stage=True,
            low_steps=10,
            option_stalled=True,
            invalid_terminal=True,
        )
        self.assertAlmostEqual(info["final_base_progress"], 0.1)
        self.assertAlmostEqual(info["final_yaw_progress"], 0.1)
        self.assertAlmostEqual(info["terms"]["low_step_cost"], -0.1)
        self.assertAlmostEqual(reward, -4.4)

    def test_frozen_rule_low_policy_returns_exact_zero_residual(self):
        policy = ZeroFusedResidualPolicy()
        residual = policy.predict(np.zeros(66, dtype=np.float32))
        np.testing.assert_array_equal(
            residual,
            np.zeros(5, dtype=np.float32),
        )

    def test_upper_option_type_owns_arm_release(self):
        wrapper = JointSubgoalResidualLowLevelWrapper(
            ZeroFusedResidualPolicy(),
            ResidualTeacherStub(),
        )
        sensor = sensor_observation()
        wrapper.begin_subgoal(
            JointHighLevelCommand(
                [0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                subgoal_type=SubgoalType.DETOUR,
            ),
            sensor,
        )
        self.assertFalse(wrapper._arm_subgoal_enabled())
        wrapper.begin_subgoal(
            JointHighLevelCommand(
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                subgoal_type=SubgoalType.TERMINAL,
            ),
            sensor,
        )
        self.assertTrue(wrapper._arm_subgoal_enabled())

    def test_hrl_terminal_latch_does_not_require_path_index(self):
        state = ("NONE", 0, False)
        states = []
        for _ in range(4):
            state = update_terminal_pose_latch(
                state[0],
                state[1],
                state[2],
                position_error=0.0,
                yaw_error=0.0,
                enter_position_tolerance=0.03,
                exit_position_tolerance=0.06,
                yaw_tolerance=0.08,
                stable_cycles=3,
            )
            states.append(state)
        self.assertEqual(states[0][0], "ROTATE")
        self.assertFalse(states[2][2])
        self.assertEqual(states[3][0], "ALIGNED")
        self.assertTrue(states[3][2])

    def test_terminal_control_state_separates_base_and_arm_completion(self):
        state = ("NONE", 0, 0, False, False)
        state = update_terminal_control_state(
            *(state + (
                0.0, 0.0, 0.20,
                0.03, 0.06, 0.08, 2, 0.05, 2,
            ))
        )
        self.assertEqual(state[0], "BASE_ROTATE")
        state = update_terminal_control_state(
            *(state + (
                0.0, 0.0, 0.20,
                0.03, 0.06, 0.08, 2, 0.05, 2,
            ))
        )
        state = update_terminal_control_state(
            *(state + (
                0.0, 0.0, 0.20,
                0.03, 0.06, 0.08, 2, 0.05, 2,
            ))
        )
        self.assertEqual(state[0], "ARM_REACH")
        self.assertTrue(state[3])
        self.assertFalse(state[4])
        state = update_terminal_control_state(
            *(state + (
                0.20, 1.0, 0.04,
                0.03, 0.06, 0.08, 2, 0.05, 2,
            ))
        )
        self.assertEqual(state[0], "ARM_REACH")
        state = update_terminal_control_state(
            *(state + (
                0.20, 1.0, 0.04,
                0.03, 0.06, 0.08, 2, 0.05, 2,
            ))
        )
        self.assertEqual(state[0], "ALIGNED")
        self.assertTrue(state[4])

    def test_terminal_stall_error_tracks_only_active_substate(self):
        environment = HighLevelEnv.__new__(HighLevelEnv)
        environment.base_subgoal_tolerance = 0.04
        environment.yaw_subgoal_tolerance = 0.08
        environment.ee_subgoal_tolerance = 0.05
        environment.low_environment = FakeLowEnvironment()
        command = JointHighLevelCommand(
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            subgoal_type=SubgoalType.TERMINAL,
        )
        remaining = np.asarray(
            [0.02, 0.0, 0.80, 0.50, 0.0, 0.0],
            dtype=np.float64,
        )
        environment.low_environment._terminal_pose_stage = (
            "BASE_TRANSLATE"
        )
        self.assertAlmostEqual(
            environment._subgoal_error_measure(remaining, command),
            0.5,
        )
        environment.low_environment._terminal_pose_stage = "BASE_ROTATE"
        self.assertAlmostEqual(
            environment._subgoal_error_measure(remaining, command),
            10.0,
        )
        environment.low_environment._terminal_pose_stage = "ARM_REACH"
        self.assertAlmostEqual(
            environment._subgoal_error_measure(remaining, command),
            10.0,
        )

    def test_rule_high_locks_detour_until_chassis_passes_obstacle(self):
        policy = RuleBasedHighPolicy()
        blocked_scan = np.full(36, 10.0, dtype=np.float32)
        blocked_scan[17:19] = 0.70
        first = self._rule_observation(
            base_pose=[0.0, 0.0, 0.0],
            base_error=[1.7, 0.0],
            scan=blocked_scan,
        )
        first_command = policy.predict(first)
        self.assertTrue(policy.detour_locked)
        self.assertNotEqual(policy.detour_turn_sign, 0.0)
        self.assertEqual(first_command.subgoal_type, SubgoalType.DETOUR)
        self.assertEqual(
            first_command.route_side,
            RouteSide.from_sign(policy.detour_turn_sign),
        )
        self.assertGreater(abs(first_command.base_goal[1]), 0.01)

        clear_but_not_passed = self._rule_observation(
            base_pose=[0.1, 0.1, 0.0],
            base_error=[1.6, -0.1],
            scan=np.full(36, 10.0, dtype=np.float32),
        )
        second_command = policy.predict(clear_but_not_passed)
        self.assertTrue(policy.detour_locked)
        self.assertEqual(second_command.reason, "committed_detour_sector")
        self.assertEqual(
            np.sign(second_command.base_goal[1]),
            np.sign(first_command.base_goal[1]),
        )

    def test_rule_high_uses_footprint_clearance_not_center_range(self):
        policy = RuleBasedHighPolicy()
        raw_scan = np.full(36, 10.0, dtype=np.float32)
        raw_scan[17:19] = 0.70
        clearance = policy._clearance_scan(raw_scan)
        self.assertLess(clearance[17], raw_scan[17] - 0.5)

    def test_rule_high_reset_seeds_requested_detour_side(self):
        policy = RuleBasedHighPolicy()
        blocked_scan = np.full(36, 10.0, dtype=np.float32)
        blocked_scan[17:19] = 0.70
        policy.reset(preferred_turn_sign=-1.0)
        command = policy.predict(self._rule_observation(
            base_pose=[0.0, 0.0, 0.0],
            base_error=[1.7, 0.0],
            scan=blocked_scan,
        ))
        self.assertLess(command.base_goal[1], 0.0)
        self.assertEqual(policy.detour_turn_sign, -1.0)

    @staticmethod
    def _rule_observation(base_pose, base_error, scan):
        base_pose = np.asarray(base_pose, dtype=np.float32)
        base_error = np.asarray(base_error, dtype=np.float32)
        return {
            "final_base_error_body": base_error,
            "final_ee_error_body": np.asarray(
                [1.0, 0.0, 0.0], dtype=np.float32
            ),
            "scan_bins": np.asarray(scan, dtype=np.float32),
            "final_base_distance": float(np.linalg.norm(base_error)),
            "final_ee_distance": 1.0,
            "final_heading_error": 0.0,
            "base_pose_world": base_pose,
        }


if __name__ == "__main__":
    unittest.main()
