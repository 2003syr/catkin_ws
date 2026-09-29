#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import json
import os
import sys
import tempfile
import unittest

import numpy as np


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "scripts",
))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hrl.fused_low_level import (  # noqa: E402
    CoordinatedRuleTeacher,
    FusedActionAdapter,
    FusedLowLevelState,
)
from training.fused_low_dataset import (  # noqa: E402
    DEFAULT_ACTION_LOSS_WEIGHTS,
    PHASE_TO_ID,
    episode_train_validation_split,
    phase_balanced_epoch_indices,
    summarize_samples,
    validate_action_mask_contract,
    validate_action_loss_weights,
)
from training.fused_box_detour import (  # noqa: E402
    BoxDetourPath,
    BoxStudentStallTakeover,
    box_action_mask_phase,
    box_reward_progress,
    box_student_arm_hold_required,
    compose_box_target_configuration,
    resolve_box_detour_side,
)
from training.fused_reach_scenarios import (  # noqa: E402
    FusedReachScenarioSampler,
    filter_scenario_categories,
    kinematic_lower_bound_steps,
    load_fused_reach_scenarios,
    scenario_step_budget,
    split_fused_reach_scenarios,
)
from training.select_fused_low_residual_checkpoint import (  # noqa: E402
    _selection_score,
)


class IdentityJacobianProvider(object):

    def position_jacobian(self, _joint_positions):
        return np.asarray([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)


class Joint2RedundantJacobianProvider(object):

    def position_jacobian(self, _joint_positions):
        # joint2 and joint4 have the same Cartesian effect, leaving a
        # one-dimensional null space that can move joint2 away from its upper
        # limit while joint4 compensates.
        return np.asarray([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)


def sensor_observation():
    sensor = np.zeros(46, dtype=np.float32)
    sensor[0:3] = [1.0, 0.0, 0.4]
    sensor[6:9] = [0.0, 0.0, 0.3]
    # q starts at index 11. z=pi/2 plus the -pi/2 model offset means the
    # tracked chassis is facing fixed-frame +x.
    sensor[11:21] = [
        0.0,
        0.0,
        0.5 * math.pi,
        1.2,
        0.0,
        -0.3,
        0.03,
        0.0,
        0.03,
        0.0,
    ]
    return sensor


class FusedLowLevelContractTest(unittest.TestCase):

    def test_box_target_fk_configuration_uses_final_base_position(self):
        current_q = np.asarray([
            0.0, 0.0, 0.4, 1.2,
            0.1, -0.2, 0.03, 0.4, 0.05, -0.6,
        ])
        target_arm = np.asarray([0.2, -0.3, 0.04, 0.5, 0.06, -0.7])
        target_q = compose_box_target_configuration(
            current_q,
            target_arm,
            [1.60, -0.13],
            final_base_yaw=0.75,
        )
        np.testing.assert_allclose(target_q[0:2], [1.60, -0.13])
        np.testing.assert_allclose(target_q[4:10], target_arm)
        self.assertEqual(target_q[2], 0.75)
        self.assertEqual(target_q[3], current_q[3])
        np.testing.assert_allclose(current_q[0:2], [0.0, 0.0])

    def test_box_detour_route_clears_inflated_obstacle(self):
        path = BoxDetourPath(
            [0.0, 0.0],
            [1.60, -0.13],
            box_center_xy=[0.85, -0.13],
            box_size_xy=[0.20, 0.32],
            base_half_length=0.56,
            base_half_width=0.10,
            clearance=0.06,
            side=1.0,
        )
        self.assertEqual(path.waypoints.shape, (6, 2))
        for point in path.waypoints[1:5]:
            self.assertFalse(path.geometric_collision(point))
        self.assertGreater(path.total_length, 1.60)

    def test_box_direct_path_keeps_goal_and_disables_geometry(self):
        path = BoxDetourPath(
            [0.0, 0.0],
            [1.60, -0.13],
            box_center_xy=[0.85, -0.13],
            box_size_xy=[0.20, 0.32],
            clearance=0.15,
            direct_path=True,
        )
        np.testing.assert_allclose(path.goal_xy, [1.60, -0.13])
        self.assertEqual(path.waypoints.shape, (6, 2))
        self.assertTrue(path.direct_path)
        self.assertFalse(path.geometric_collision([0.85, -0.13]))
        self.assertTrue(path.metadata()["direct_path"])
        path.advance(path.goal_xy, planar_yaw=0.0)
        self.assertTrue(path.complete)

    def test_box_detour_progress_reaches_final_waypoint(self):
        path = BoxDetourPath(
            [0.0, 0.0], [1.60, -0.13], side=-1.0
        )
        previous = path.remaining_distance([0.0, 0.0])
        for point in path.waypoints[1:]:
            path.advance(point)
            current = path.remaining_distance(point)
            self.assertLessEqual(current, previous + 1.0e-8)
            previous = current
        self.assertTrue(path.complete)
        self.assertLessEqual(
            path.remaining_distance(path.goal_xy),
            path.waypoint_tolerance,
        )

    def test_box_detour_does_not_complete_before_final_waypoint(self):
        path = BoxDetourPath(
            [0.0, 0.0],
            [1.60, -0.13],
            side=1.0,
            final_position_tolerance=0.02,
        )
        for point in path.waypoints[1:5]:
            path.advance(point)
        self.assertEqual(path.index, 5)
        self.assertTrue(path.final_waypoint_active)
        self.assertFalse(path.complete)
        self.assertGreater(path.remaining_distance(path.waypoints[4]), 0.0)
        path.advance(path.goal_xy)
        self.assertTrue(path.complete)

    def test_box_detour_promotes_goal_inside_clearance_to_safe_pose(self):
        path = BoxDetourPath(
            [0.0, 0.0],
            [1.60, -0.13],
            clearance=0.15,
        )
        np.testing.assert_allclose(path.goal_xy, [1.66, -0.13])
        np.testing.assert_allclose(
            path.metadata()["requested_goal_xy"], [1.60, -0.13]
        )

    def test_box_detour_requires_terminal_yaw_before_arm_finish(self):
        path = BoxDetourPath(
            [0.0, 0.0],
            [1.60, -0.13],
            side=-1.0,
            goal_yaw=0.40,
            final_position_tolerance=0.02,
            final_yaw_tolerance=0.05,
        )
        for point in path.waypoints[1:5]:
            path.advance(point, planar_yaw=0.0)
        path.advance(path.goal_xy, planar_yaw=0.0)
        self.assertFalse(path.complete)
        self.assertEqual(path.index, 5)
        path.advance(path.goal_xy, planar_yaw=0.37)
        self.assertTrue(path.complete)

    def test_box_detour_accepts_latched_terminal_pose_alignment(self):
        path = BoxDetourPath(
            [0.0, 0.0],
            [1.60, -0.13],
            side=1.0,
            goal_yaw=0.40,
            final_position_tolerance=0.03,
            final_yaw_tolerance=0.08,
        )
        for point in path.waypoints[1:5]:
            path.advance(point, planar_yaw=0.0)
        path.advance(
            path.goal_xy,
            planar_yaw=0.0,
            terminal_pose_aligned=False,
        )
        self.assertFalse(path.complete)
        drifted_position = path.goal_xy + np.asarray([0.04, 0.0])
        self.assertGreater(
            np.linalg.norm(drifted_position - path.goal_xy),
            path.final_position_tolerance,
        )
        path.advance(
            drifted_position,
            planar_yaw=0.0,
            terminal_pose_aligned=True,
        )
        self.assertTrue(path.complete)

    def test_box_student_stall_takeover_latches_late_base_stall(self):
        monitor = BoxStudentStallTakeover(
            base_stall_cycles=3,
            arm_stall_cycles=2,
        )
        for remaining in (0.40, 0.35, 0.30):
            state = monitor.update(4, 6, remaining, 0.20)
            self.assertFalse(state["base_active"])
        for _ in range(2):
            state = monitor.update(4, 6, 0.30, 0.20)
            self.assertFalse(state["base_active"])
        state = monitor.update(4, 6, 0.30, 0.20)
        self.assertTrue(state["base_active"])
        self.assertTrue(state["base_latched"])

    def test_box_student_stall_takeover_switches_to_arm_finish(self):
        monitor = BoxStudentStallTakeover(
            base_stall_cycles=1,
            arm_stall_cycles=2,
        )
        monitor.update(4, 6, 0.25, 0.12)
        base_state = monitor.update(4, 6, 0.25, 0.12)
        self.assertTrue(base_state["base_active"])
        first_arm = monitor.update(6, 6, 0.0, 0.10)
        self.assertFalse(first_arm["base_active"])
        self.assertFalse(first_arm["arm_active"])
        monitor.update(6, 6, 0.0, 0.10)
        stalled_arm = monitor.update(6, 6, 0.0, 0.10)
        self.assertTrue(stalled_arm["arm_active"])
        improved_arm = monitor.update(6, 6, 0.0, 0.08)
        self.assertTrue(improved_arm["arm_active"])

    def test_observation_is_66d_and_uses_z_as_yaw(self):
        builder = FusedLowLevelState()
        observation = builder.build(
            sensor_observation(),
            np.asarray([1.0, 0.0]),
        )
        self.assertEqual(observation.shape, (66,))
        np.testing.assert_allclose(
            observation[FusedLowLevelState.SUBGOAL_SLICE][0:3],
            [1.0, 0.0, 0.0],
            atol=1.0e-5,
        )
        np.testing.assert_allclose(observation[52:58], np.ones(6))
        np.testing.assert_allclose(observation[58:66], np.ones(8))

    def test_box_action_masks_encode_the_three_control_stages(self):
        base_mask = FusedLowLevelState.action_mask_for_phase(
            "BASE_APPROACH"
        )
        coordinated_mask = FusedLowLevelState.action_mask_for_phase(
            "COORDINATED"
        )
        arm_mask = FusedLowLevelState.action_mask_for_phase("ARM_FINISH")
        np.testing.assert_array_equal(
            base_mask,
            [1, 1, 0, 0, 0, 0, 0, 0],
        )
        np.testing.assert_array_equal(
            coordinated_mask,
            np.ones(8),
        )
        np.testing.assert_array_equal(
            arm_mask,
            [0, 0, 1, 1, 1, 1, 1, 1],
        )

    def test_box_action_mask_phase_uses_six_point_route(self):
        self.assertEqual(box_action_mask_phase(1, 6), "BASE_APPROACH")
        self.assertEqual(box_action_mask_phase(2, 6), "BASE_APPROACH")
        self.assertEqual(box_action_mask_phase(3, 6), "COORDINATED")
        self.assertEqual(box_action_mask_phase(4, 6), "COORDINATED")
        self.assertEqual(box_action_mask_phase(5, 6), "COORDINATED")
        self.assertEqual(box_action_mask_phase(6, 6), "ARM_FINISH")

    def test_student_arm_is_held_until_base_alignment_completes(self):
        self.assertTrue(box_student_arm_hold_required(1, 6))
        self.assertTrue(box_student_arm_hold_required(3, 6))
        self.assertTrue(box_student_arm_hold_required(5, 6))
        self.assertFalse(box_student_arm_hold_required(6, 6))

    def test_box_detour_side_zero_alternates_by_episode(self):
        self.assertEqual(resolve_box_detour_side(0.0, 1), 1.0)
        self.assertEqual(resolve_box_detour_side(0.0, 2), -1.0)
        self.assertEqual(resolve_box_detour_side(0.0, 3), 1.0)
        self.assertEqual(resolve_box_detour_side(1.0, 2), 1.0)
        self.assertEqual(resolve_box_detour_side(-1.0, 1), -1.0)
        with self.assertRaises(ValueError):
            resolve_box_detour_side(0.0, 0)

    def test_box_reward_keeps_last_centimetres_visible(self):
        path, ee, yaw = box_reward_progress(
            previous_path_remaining=1.0,
            path_remaining=0.9,
            initial_path_remaining=2.0,
            previous_ee_distance=0.07,
            ee_distance=0.05,
            initial_ee_distance=1.6,
            success_threshold=0.05,
            previous_yaw_error=0.6,
            yaw_error=0.4,
            arm_finish_active=True,
            terminal_alignment_active=True,
        )
        self.assertAlmostEqual(path, 0.05)
        self.assertAlmostEqual(ee, 0.4)
        self.assertAlmostEqual(yaw, 0.2 / math.pi)

    def test_box_reward_ignores_yaw_before_terminal_alignment(self):
        unused_path, ee, yaw = box_reward_progress(
            previous_path_remaining=1.0,
            path_remaining=1.0,
            initial_path_remaining=2.0,
            previous_ee_distance=1.0,
            ee_distance=0.9,
            initial_ee_distance=2.0,
            success_threshold=0.05,
            previous_yaw_error=1.0,
            yaw_error=0.0,
        )
        self.assertAlmostEqual(ee, 0.05)
        self.assertEqual(yaw, 0.0)

    def test_observation_accepts_a_dynamic_action_mask(self):
        builder = FusedLowLevelState()
        mask = FusedLowLevelState.action_mask_for_phase("ARM_FINISH")
        observation = builder.build(
            sensor_observation(),
            np.asarray([1.0, 0.0]),
            action_mask=mask,
        )
        np.testing.assert_array_equal(
            observation[FusedLowLevelState.ACTION_MASK_SLICE],
            mask,
        )

    def test_three_stage_dataset_contract_matches_masks_and_actions(self):
        observations = np.zeros((3, 66), dtype=np.float32)
        actions = np.zeros((3, 8), dtype=np.float32)
        phases = np.asarray([0, 1, 2], dtype=np.int8)
        observations[0, 58:66] = [1, 1, 0, 0, 0, 0, 0, 0]
        observations[1, 58:66] = np.ones(8)
        observations[2, 58:66] = [0, 0, 1, 1, 1, 1, 1, 1]
        contract = validate_action_mask_contract(
            observations,
            actions,
            phases,
        )
        self.assertEqual(contract["mode"], "three_stage")

    def test_three_stage_dataset_rejects_nonzero_inactive_action(self):
        observations = np.zeros((3, 66), dtype=np.float32)
        actions = np.zeros((3, 8), dtype=np.float32)
        phases = np.asarray([0, 1, 2], dtype=np.int8)
        observations[0, 58:66] = [1, 1, 0, 0, 0, 0, 0, 0]
        observations[1, 58:66] = np.ones(8)
        observations[2, 58:66] = [0, 0, 1, 1, 1, 1, 1, 1]
        actions[0, 2] = 0.1
        with self.assertRaises(ValueError):
            validate_action_mask_contract(observations, actions, phases)

    def test_phase_sampler_increases_rare_finish_coverage(self):
        phase_ids = np.asarray(
            [0] * 900 + [1] * 90 + [2] * 10,
            dtype=np.int8,
        )
        indices = np.arange(phase_ids.size, dtype=np.int64)
        sampled = phase_balanced_epoch_indices(
            indices,
            phase_ids,
            0.5,
            np.random.RandomState(123),
        )
        sampled_phases = phase_ids[sampled]
        finish_fraction = float(np.mean(sampled_phases == 2))
        approach_fraction = float(np.mean(sampled_phases == 0))
        self.assertGreater(finish_fraction, 0.04)
        self.assertLess(approach_fraction, 0.85)

    def test_action_adapter_enforces_nonholonomic_mapping(self):
        adapter = FusedActionAdapter(
            linear_action_scale=0.25,
            yaw_action_scale=0.25,
        )
        action = np.asarray(
            [1.0, 0.5, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            dtype=np.float32,
        )
        full = adapter.to_full_action(action, sensor_observation())
        np.testing.assert_allclose(
            full[0:4],
            [0.25, 0.0, 0.125, 0.0],
            atol=1.0e-7,
        )
        np.testing.assert_allclose(full[4:10], action[2:8])

    def test_action_adapter_recovers_filtered_fused_action(self):
        adapter = FusedActionAdapter(
            linear_action_scale=0.25,
            yaw_action_scale=0.25,
        )
        action = np.asarray(
            [0.7, -0.4, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            dtype=np.float32,
        )
        full = adapter.to_full_action(action, sensor_observation())
        recovered = adapter.from_full_action(full, sensor_observation())
        np.testing.assert_allclose(recovered, action, atol=1.0e-6)

    def test_teacher_has_a_coordinated_overlap_phase(self):
        builder = FusedLowLevelState()
        observation = builder.build(
            sensor_observation(),
            np.asarray([0.20, 0.0]),
        )
        teacher = CoordinatedRuleTeacher(
            IdentityJacobianProvider(),
            arm_start_distance=0.30,
            base_stop_distance=0.04,
        )
        action = teacher.predict(observation)
        self.assertGreater(action[0], 0.0)
        self.assertGreater(np.max(np.abs(action[2:8])), 0.0)
        self.assertEqual(teacher.diagnostics()["phase"], "COORDINATED")

    def test_teacher_arm_finish_stops_base_but_keeps_arm_command(self):
        builder = FusedLowLevelState()
        observation = builder.build(
            sensor_observation(),
            np.asarray([0.0, 0.0]),
        )
        teacher = CoordinatedRuleTeacher(
            IdentityJacobianProvider(),
            arm_start_distance=0.30,
            base_stop_distance=0.04,
        )
        action = teacher.predict(observation)
        self.assertEqual(teacher.diagnostics()["phase"], "ARM_FINISH")
        np.testing.assert_allclose(action[0:2], np.zeros(2), atol=1.0e-7)
        self.assertGreater(np.max(np.abs(action[2:8])), 0.0)

    def test_zero_residual_reproduces_teacher_action(self):
        builder = FusedLowLevelState()
        observation = builder.build(
            sensor_observation(),
            np.asarray([0.20, 0.0]),
        )
        teacher = CoordinatedRuleTeacher(
            IdentityJacobianProvider(),
            arm_start_distance=0.30,
            base_stop_distance=0.04,
        )
        nominal = teacher.predict(observation)
        residual_action = teacher.predict_residual(
            observation,
            np.zeros(5, dtype=np.float32),
        )
        np.testing.assert_allclose(
            residual_action,
            nominal,
            atol=1.0e-7,
            rtol=0.0,
        )

    def test_subgoal_teacher_rotates_before_driving_to_behind_goal(self):
        builder = FusedLowLevelState()
        observation = builder.build(
            sensor_observation(),
            np.asarray([0.0, 0.0]),
        )
        observation[FusedLowLevelState.SUBGOAL_SLICE] = np.asarray([
            -0.30, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        teacher = CoordinatedRuleTeacher(
            IdentityJacobianProvider(),
            arm_start_distance=0.30,
            base_stop_distance=0.04,
        )
        components = teacher.subgoal_components(
            observation,
            arm_enabled=False,
        )
        self.assertEqual(components["phase"], "SUBGOAL_ALIGN")
        self.assertAlmostEqual(float(components["base_action"][0]), 0.0)
        self.assertGreater(abs(float(components["base_action"][1])), 0.0)

    def test_subgoal_teacher_drives_only_after_bearing_is_aligned(self):
        builder = FusedLowLevelState()
        observation = builder.build(
            sensor_observation(),
            np.asarray([0.0, 0.0]),
        )
        observation[FusedLowLevelState.SUBGOAL_SLICE] = np.asarray([
            0.20, 0.02, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        teacher = CoordinatedRuleTeacher(IdentityJacobianProvider())
        components = teacher.subgoal_components(
            observation,
            arm_enabled=False,
        )
        self.assertEqual(components["phase"], "SUBGOAL_DRIVE")
        self.assertGreater(float(components["base_action"][0]), 0.0)

    def test_subgoal_teacher_does_not_stop_in_terminal_path_dead_zone(self):
        builder = FusedLowLevelState()
        observation = builder.build(
            sensor_observation(),
            np.asarray([0.0, 0.0]),
        )
        observation[FusedLowLevelState.SUBGOAL_SLICE] = np.asarray([
            0.03, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        teacher = CoordinatedRuleTeacher(IdentityJacobianProvider())
        components = teacher.subgoal_components(
            observation,
            arm_enabled=False,
        )
        self.assertEqual(components["phase"], "SUBGOAL_DRIVE")
        self.assertGreater(float(components["base_action"][0]), 0.0)
        self.assertAlmostEqual(
            float(components["base_stop_distance"]),
            0.02,
            places=6,
        )

    def test_subgoal_teacher_releases_arm_only_at_terminal_pose(self):
        builder = FusedLowLevelState()
        observation = builder.build(
            sensor_observation(),
            np.asarray([0.0, 0.0]),
        )
        observation[FusedLowLevelState.SUBGOAL_SLICE] = np.asarray([
            0.0, 0.0, 0.0, 0.05, 0.0, 0.0,
        ], dtype=np.float32)
        teacher = CoordinatedRuleTeacher(IdentityJacobianProvider())
        components = teacher.subgoal_components(
            observation,
            arm_enabled=True,
        )
        self.assertEqual(components["phase"], "ARM_FINISH")
        np.testing.assert_allclose(
            components["base_action"],
            np.zeros(2),
            atol=1.0e-7,
        )
        self.assertGreater(
            np.max(np.abs(components["arm_action"])),
            0.0,
        )

    def test_zero_residual_reproduces_subgoal_teacher_action(self):
        builder = FusedLowLevelState()
        observation = builder.build(
            sensor_observation(),
            np.asarray([0.0, 0.0]),
        )
        observation[FusedLowLevelState.SUBGOAL_SLICE] = np.asarray([
            -0.10, 0.02, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        teacher = CoordinatedRuleTeacher(IdentityJacobianProvider())
        nominal = teacher.subgoal_components(
            observation,
            arm_enabled=False,
        )["nominal_action"]
        residual_components = teacher.subgoal_residual_components(
            observation,
            np.zeros(5, dtype=np.float32),
            arm_enabled=False,
        )
        residual = residual_components["action"]
        np.testing.assert_allclose(
            residual,
            nominal,
            atol=1.0e-7,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            residual_components["sensor"],
            observation[FusedLowLevelState.SENSOR_SLICE],
            atol=0.0,
            rtol=0.0,
        )

    def test_residual_action_is_bounded_and_nonholonomic(self):
        builder = FusedLowLevelState()
        observation = builder.build(
            sensor_observation(),
            np.asarray([0.20, 0.0]),
        )
        teacher = CoordinatedRuleTeacher(
            IdentityJacobianProvider(),
            arm_start_distance=0.30,
            base_stop_distance=0.04,
        )
        action = teacher.predict_residual(
            observation,
            np.ones(5, dtype=np.float32),
        )
        self.assertEqual(action.shape, (8,))
        self.assertGreaterEqual(float(action[0]), 0.0)
        self.assertLessEqual(float(action[0]), 1.0)
        self.assertGreaterEqual(float(action[1]), -1.0)
        self.assertLessEqual(float(action[1]), 1.0)
        self.assertTrue(np.all(action[2:8] <= 1.0))
        self.assertTrue(np.all(action[2:8] >= -1.0))

    def test_teacher_defers_arm_while_base_goal_is_far(self):
        builder = FusedLowLevelState()
        observation = builder.build(
            sensor_observation(),
            np.asarray([0.60, 0.0]),
        )
        teacher = CoordinatedRuleTeacher(
            IdentityJacobianProvider(),
            arm_start_distance=0.30,
            base_stop_distance=0.04,
        )
        action = teacher.predict(observation)
        self.assertGreater(action[0], 0.0)
        np.testing.assert_allclose(action[2:8], np.zeros(6))
        self.assertEqual(teacher.diagnostics()["phase"], "BASE_APPROACH")

    def test_teacher_blocks_motion_deeper_into_arm_soft_limit(self):
        sensor = sensor_observation()
        # joint3 is already inside its lower soft-limit band.
        sensor[17] = 0.03
        builder = FusedLowLevelState()
        observation = builder.build(
            sensor,
            np.asarray([0.20, 0.0]),
        )
        # Identity Jacobian maps negative z error directly to joint3.
        observation[FusedLowLevelState.SUBGOAL_SLICE][5] = -0.10
        teacher = CoordinatedRuleTeacher(
            IdentityJacobianProvider(),
            arm_start_distance=0.30,
            base_stop_distance=0.04,
        )
        action = teacher.predict(observation)
        self.assertAlmostEqual(float(action[4]), 0.0)
        self.assertIn(
            "joint3",
            teacher.diagnostics()["joint_limit_blocked"],
        )

    def test_terminal_nullspace_task_moves_joint2_away_from_upper_limit(self):
        sensor = sensor_observation()
        sensor[16] = -0.01
        sensor[17] = 0.075
        sensor[19] = 0.075
        teacher = CoordinatedRuleTeacher(
            Joint2RedundantJacobianProvider(),
            joint_limit_avoidance_gain=0.40,
            joint_limit_avoidance_activation=1.5,
        )
        action, _, _, diagnostics = teacher.solve_cartesian_velocity(
            sensor,
            np.zeros(3, dtype=np.float64),
            enable_joint_limit_avoidance=True,
        )
        self.assertLess(float(action[1]), 0.0)
        self.assertGreater(float(action[3]), 0.0)
        self.assertTrue(diagnostics["joint_limit_avoidance_applied"])
        self.assertLess(
            float(diagnostics["joint_limit_avoidance_raw"][1]), 0.0
        )
        scaled_jacobian = (
            Joint2RedundantJacobianProvider().position_jacobian(None)
            * teacher.arm_joint_max_velocity[np.newaxis, :]
        )
        np.testing.assert_allclose(
            np.dot(scaled_jacobian, action),
            np.zeros(3),
            atol=1.0e-3,
        )

    def test_joint_limit_avoidance_is_inert_unless_enabled(self):
        sensor = sensor_observation()
        sensor[16] = -0.01
        teacher = CoordinatedRuleTeacher(
            Joint2RedundantJacobianProvider(),
            joint_limit_avoidance_gain=0.40,
        )
        action, _, _, diagnostics = teacher.solve_cartesian_velocity(
            sensor,
            np.zeros(3, dtype=np.float64),
        )
        np.testing.assert_allclose(action, np.zeros(6), atol=1.0e-9)
        self.assertFalse(diagnostics["joint_limit_avoidance_applied"])

    def test_fused_dataset_summary_reports_phase_and_filter_coverage(self):
        observations = np.zeros((3, 66), dtype=np.float32)
        actions = np.zeros((3, 8), dtype=np.float32)
        safe_actions = actions.copy()
        safe_actions[1, 4] = 0.25
        phases = np.asarray([
            PHASE_TO_ID["BASE_APPROACH"],
            PHASE_TO_ID["COORDINATED"],
            PHASE_TO_ID["COORDINATED"],
        ], dtype=np.int8)
        summary = summarize_samples(
            observations,
            actions,
            safe_actions,
            phases,
            np.asarray([False, True, False]),
        )
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(
            int(summary["phase_counts"][PHASE_TO_ID["COORDINATED"]]),
            2,
        )
        self.assertAlmostEqual(
            summary["safety_intervention_rate"],
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            summary["mean_absolute_filter_delta"][4],
            0.25 / 3.0,
        )

    def test_fused_bc_split_keeps_complete_episodes_separate(self):
        episode_ids = np.repeat(np.arange(10), 3)
        (
            train_indices,
            validation_indices,
            train_episodes,
            validation_episodes,
        ) = episode_train_validation_split(
            episode_ids,
            validation_fraction=0.20,
            seed=123,
        )
        self.assertEqual(validation_episodes.size, 2)
        self.assertEqual(train_episodes.size, 8)
        self.assertFalse(
            set(train_episodes.tolist())
            & set(validation_episodes.tolist())
        )
        self.assertEqual(train_indices.size, 24)
        self.assertEqual(validation_indices.size, 6)
        np.testing.assert_array_equal(
            np.unique(episode_ids[train_indices]),
            train_episodes,
        )
        np.testing.assert_array_equal(
            np.unique(episode_ids[validation_indices]),
            validation_episodes,
        )

    def test_fused_bc_action_weights_preserve_limit_joint_zero_targets(self):
        weights = validate_action_loss_weights(
            DEFAULT_ACTION_LOSS_WEIGHTS
        )
        self.assertEqual(weights.shape, (8,))
        self.assertEqual(float(weights[0]), 1.0)
        self.assertEqual(float(weights[1]), 1.0)
        self.assertEqual(float(weights[4]), float(weights[2]))
        self.assertEqual(float(weights[6]), float(weights[2]))
        self.assertLess(float(weights[7]), float(weights[2]))

    def test_fused_reach_scenarios_are_stratified_and_replayable(self):
        categories = (
            ["simple_success"] * 5
            + ["hard_success"] * 17
            + ["recoverable_failure"] * 8
        )
        records = []
        for index, category in enumerate(categories):
            records.append({
                "episode": index + 1,
                "category": category,
                "success_step": 80 + index,
                "initial_distance": 0.4,
                "final_distance": 0.05,
                "reset_info": {
                    "target_arm_positions": [0.0] * 6,
                    "base_displacement_body": [0.4, 0.0],
                    "target_position": [0.4, 0.0, 0.3],
                    "base_goal_xy": [0.4, 0.0],
                },
            })
        handle, path = tempfile.mkstemp(suffix=".json")
        os.close(handle)
        try:
            with open(path, "w") as stream:
                json.dump({
                    "schema_version": 1,
                    "recommended_step_budget": 120,
                    "records": records,
                }, stream)
            loaded = load_fused_reach_scenarios(path)
            self.assertEqual(loaded["recommended_step_budget"], 120)
            train = split_fused_reach_scenarios(
                loaded["scenarios"], "train", seed=123
            )
            validation = split_fused_reach_scenarios(
                loaded["scenarios"], "validation", seed=123
            )
            test = split_fused_reach_scenarios(
                loaded["scenarios"], "test", seed=123
            )
            self.assertEqual(len(train), 20)
            self.assertEqual(len(validation), 4)
            self.assertEqual(len(test), 6)
            identifiers = [
                set(item["scenario_id"] for item in split_items)
                for split_items in (train, validation, test)
            ]
            self.assertFalse(identifiers[0] & identifiers[1])
            self.assertFalse(identifiers[0] & identifiers[2])
            self.assertFalse(identifiers[1] & identifiers[2])
            self.assertEqual(len(set.union(*identifiers)), 30)

            first = FusedReachScenarioSampler(
                train, seed=7, shuffle=True
            )
            second = FusedReachScenarioSampler(
                train, seed=7, shuffle=True
            )
            first_order = [
                first.next()["scenario_id"] for _ in range(len(train))
            ]
            second_order = [
                second.next()["scenario_id"] for _ in range(len(train))
            ]
            self.assertEqual(first_order, second_order)
        finally:
            os.unlink(path)

    def test_schema_three_generates_replayable_single_box_curriculum(self):
        handle, path = tempfile.mkstemp(suffix=".json")
        os.close(handle)
        try:
            with open(path, "w") as stream:
                json.dump({
                    "schema_version": 3,
                    "recommended_step_budget": 3200,
                    "generator": {
                        "type": "single_box_subgoal_v1",
                        "count": 20,
                        "seed": 789,
                        "no_obstacle_fraction": 0.20,
                        "box_size_xy": [0.20, 0.32],
                    },
                }, stream)
            first = load_fused_reach_scenarios(path)
            second = load_fused_reach_scenarios(path)
            self.assertEqual(first["scenarios"], second["scenarios"])
            self.assertEqual(len(first["scenarios"]), 20)
            direct = [
                item for item in first["scenarios"]
                if item["category"] == "direct_clear"
            ]
            detour = [
                item for item in first["scenarios"]
                if item["category"] == "box_detour"
            ]
            self.assertEqual(len(direct), 4)
            self.assertEqual(len(detour), 16)
            self.assertTrue(all(item["no_obstacle"] for item in direct))
            self.assertEqual(
                set(item["detour_side"] for item in detour),
                {-1.0, 1.0},
            )
            self.assertTrue(all(
                item["box_size_xy"] == [0.2, 0.32]
                for item in first["scenarios"]
            ))
            for scenario in first["scenarios"]:
                self.assertEqual(len(scenario["initial_base_positions"]), 3)
                self.assertEqual(len(scenario["final_base_goal_xy"]), 2)
                self.assertTrue(np.isfinite(scenario["final_base_yaw"]))
                self.assertLessEqual(
                    abs(float(scenario["final_base_goal_xy"][0])),
                    1.75,
                )
            train = split_fused_reach_scenarios(
                first["scenarios"], "train", seed=123
            )
            validation = split_fused_reach_scenarios(
                first["scenarios"], "validation", seed=123
            )
            test = split_fused_reach_scenarios(
                first["scenarios"], "test", seed=123
            )
            self.assertEqual(len(train), 13)
            self.assertEqual(len(validation), 3)
            self.assertEqual(len(test), 4)
        finally:
            os.unlink(path)

    def test_recoverable_failure_sampler_hits_requested_fraction(self):
        scenarios = []
        categories = (
            ["hard_success"] * 12
            + ["recoverable_failure"] * 5
            + ["simple_success"] * 3
        )
        for index, category in enumerate(categories):
            scenarios.append({
                "scenario_id": index + 1,
                "category": category,
            })
        first = FusedReachScenarioSampler(
            scenarios,
            seed=123,
            shuffle=True,
            category_fractions={"recoverable_failure": 0.50},
        )
        second = FusedReachScenarioSampler(
            scenarios,
            seed=123,
            shuffle=True,
            category_fractions={"recoverable_failure": 0.50},
        )
        first_epoch = [first.next() for _ in range(len(scenarios))]
        second_epoch = [second.next() for _ in range(len(scenarios))]
        self.assertEqual(
            [item["scenario_id"] for item in first_epoch],
            [item["scenario_id"] for item in second_epoch],
        )
        counts = {}
        for scenario in first_epoch:
            category = scenario["category"]
            counts[category] = counts.get(category, 0) + 1
        self.assertEqual(counts["recoverable_failure"], 10)
        self.assertEqual(counts["hard_success"], 8)
        self.assertEqual(counts["simple_success"], 2)

    def test_scenario_category_filter_excludes_invalid_tasks(self):
        scenarios = [
            {"scenario_id": 1, "category": "hard_success"},
            {"scenario_id": 2, "category": "recoverable_failure"},
            {"scenario_id": 3, "category": "other_failure"},
            {"scenario_id": 4, "category": "invalid"},
        ]
        selected = filter_scenario_categories(
            scenarios,
            ["hard_success", "recoverable_failure"],
        )
        self.assertEqual(
            [item["scenario_id"] for item in selected],
            [1, 2],
        )

    def test_kinematic_budget_uses_maximum_simultaneous_component_time(self):
        steps = kinematic_lower_bound_steps(
            [0.50, 0.0],
            [0.0] * 6,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            control_dt=0.10,
            base_speed=0.05,
            yaw_speed=0.125,
            arm_joint_speeds=[0.50, 0.50, 0.03, 0.50, 0.03, 0.50],
        )
        self.assertEqual(steps, 100)

    def test_dynamic_scenario_budget_overrides_legacy_fallback(self):
        self.assertEqual(
            scenario_step_budget({"step_budget": 173}, 120),
            173,
        )
        self.assertEqual(
            scenario_step_budget({"scenario_id": 1}, 120),
            120,
        )

    def test_checkpoint_selection_preserves_teacher_successes_first(self):
        safe_common = {
            "collisions": 0,
            "tf_failures": 0,
            "safety_rate": 0.0,
            "mean_reward": 10.0,
            "mean_final_distance": 0.05,
            "mean_residual_abs": 0.02,
        }
        protected = dict(safe_common)
        protected.update({
            "success_rate": 0.80,
            "categories": {
                "hard_success": {"success_rate": 1.0},
                "simple_success": {"success_rate": 1.0},
                "recoverable_failure": {"success_rate": 0.25},
            },
        })
        regressed = dict(safe_common)
        regressed.update({
            "success_rate": 0.90,
            "categories": {
                "hard_success": {"success_rate": 0.90},
                "simple_success": {"success_rate": 1.0},
                "recoverable_failure": {"success_rate": 0.75},
            },
        })
        self.assertGreater(
            _selection_score(protected, 0.05),
            _selection_score(regressed, 0.05),
        )


if __name__ == "__main__":
    unittest.main()
