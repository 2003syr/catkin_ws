#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hrl.hierarchical_env import HierarchicalEnv
from hrl.high_level_command import HighLevelCommand, TaskMode
from hrl.hrl4in_ppo_low_policy import HRL4INPPOLowPolicy
from hrl.rule_based_high_policy import RuleBasedHighPolicy
from hrl.rule_based_low_policy import RuleBasedLowPolicy


def observation(
        distance=1.2,
        minimum_margin=1.0,
        ee_x=0.0,
        base_x=0.0,
        base_y=0.0,
        base_yaw=0.0):
    vector = np.zeros(46, dtype=np.float32)
    vector[0] = ee_x + distance
    vector[3] = distance
    vector[6] = ee_x
    vector[10] = abs(distance)
    vector[11] = base_x
    vector[12] = base_y
    vector[14] = base_yaw
    vector[31:41] = 1.0
    vector[31] = minimum_margin
    return {"obs_vec": vector}


class FakeJacobianProvider(object):
    def __init__(self):
        self.positions = []

    def position_jacobian(self, joint_positions):
        self.positions.append(np.asarray(joint_positions).copy())
        jacobian = np.zeros((3, 6), dtype=np.float64)
        jacobian[0, 0] = 1.0
        jacobian[1, 1] = 1.0
        jacobian[2, 3] = 1.0
        return jacobian


class FakeBaseEnv(object):
    def __init__(self):
        self.actions = []
        self.stopped = False

    def reset(self):
        return observation()

    def step(self, action):
        self.actions.append(action)
        return observation(), 0.0, False, {"dist": 1.2}

    def stop(self):
        self.stopped = True


class FakeSuccessfulBaseEnv(FakeBaseEnv):
    def __init__(self):
        super(FakeSuccessfulBaseEnv, self).__init__()
        self.zero_commands = 0

    def step(self, action):
        self.actions.append(action)
        return observation(0.01), 10.0, True, {
            "dist": 0.01,
            "success": True,
        }

    def publish_zero_cmd(self):
        self.zero_commands += 1


class FakePPOInferenceClient(object):
    def __init__(self):
        self.checkpoint = "/tmp/fake_ppo.pt"
        self.total_steps = 100000
        self.observations = []
        self.closed = False

    def predict(self, low_observation):
        self.observations.append(
            np.asarray(low_observation, dtype=np.float32).copy()
        )
        action = np.zeros(10, dtype=np.float32)
        action[4] = 0.25
        return action, 1.5

    def close(self):
        self.closed = True


class ImmediatelyAchievingLowPolicy(object):
    def __init__(self):
        self.achieved = False

    def reset(self):
        self.achieved = False

    def begin_subgoal(self, _observation, _command):
        self.achieved = False

    def predict(self, _observation, _command):
        return np.zeros(10, dtype=np.float32)

    def observe_transition(self, _next_observation):
        self.achieved = True

    def diagnostics(self):
        return {
            "subgoal_achieved": self.achieved,
            "post_error_norm": 0.0,
            "intrinsic_reward": 1.0,
        }


class HRLFrameworkTest(unittest.TestCase):
    def test_high_level_command_exposes_hrl4in_masks(self):
        command = HighLevelCommand(
            TaskMode.ARM_REACH,
            np.zeros(6, dtype=np.float32),
            base_priority=0.0,
            arm_priority=1.0,
        )
        np.testing.assert_array_equal(
            command.action_mask[0:4],
            np.zeros(4),
        )
        np.testing.assert_array_equal(
            command.action_mask[4:10],
            np.ones(6),
        )
        np.testing.assert_array_equal(
            command.subgoal_mask[0:3],
            np.zeros(3),
        )
        np.testing.assert_array_equal(
            command.subgoal_mask[3:6],
            np.ones(3),
        )

    def test_base_command_enables_only_verified_planar_actions(self):
        command = HighLevelCommand(
            TaskMode.BASE_APPROACH,
            np.zeros(6, dtype=np.float32),
            base_priority=1.0,
            arm_priority=0.0,
        )
        np.testing.assert_array_equal(
            command.action_mask,
            np.asarray(
                [1.0, 1.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            command.subgoal_mask,
            np.asarray(
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        )

    def test_high_policy_switches_tasks_and_recovers(self):
        policy = RuleBasedHighPolicy()
        self.assertEqual(policy.predict(observation(1.2)).mode, TaskMode.BASE_APPROACH)
        self.assertEqual(policy.predict(observation(0.7)).mode, TaskMode.ARM_REACH)
        self.assertEqual(
            policy.predict(observation(0.7, minimum_margin=0.01)).mode,
            TaskMode.RECOVERY,
        )

    def test_scheduler_updates_high_level_at_interval(self):
        base_env = FakeBaseEnv()
        env = HierarchicalEnv(
            base_env,
            RuleBasedHighPolicy(),
            RuleBasedLowPolicy(),
            high_interval=3,
        )
        env.reset()
        updates = [env.step()[3]["high_updated"] for _ in range(5)]
        self.assertEqual(updates, [True, False, False, True, False])
        self.assertTrue(all(action.shape == (10,) for action in base_env.actions))

    def test_scheduler_refreshes_high_level_after_subgoal_success(self):
        env = HierarchicalEnv(
            FakeBaseEnv(),
            RuleBasedHighPolicy(),
            ImmediatelyAchievingLowPolicy(),
            high_interval=10,
        )
        env.reset()
        first_info = env.step()[3]
        second_info = env.step()[3]
        self.assertTrue(first_info["subgoal_achieved"])
        self.assertTrue(second_info["high_updated"])

    def test_success_latch_blocks_policy_and_reward_updates(self):
        base_env = FakeSuccessfulBaseEnv()
        env = HierarchicalEnv(
            base_env,
            RuleBasedHighPolicy(),
            RuleBasedLowPolicy(),
            high_interval=10,
        )
        env.reset()
        _, reward, done, info = env.step()

        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertEqual(reward, 10.0)
        self.assertTrue(env.episode_done)
        self.assertTrue(env.success_latched)
        self.assertEqual(len(base_env.actions), 1)

        with self.assertRaises(RuntimeError):
            env.step()
        self.assertEqual(len(base_env.actions), 1)

        env.hold_zero()
        env.hold_zero()
        self.assertEqual(base_env.zero_commands, 2)

        env.reset()
        self.assertFalse(env.episode_done)
        self.assertFalse(env.success_latched)

    def test_internal_base_joints_are_disabled_by_default(self):
        high_policy = RuleBasedHighPolicy()
        command = high_policy.predict(observation(1.2))
        action = RuleBasedLowPolicy().predict(observation(1.2), command)
        np.testing.assert_array_equal(action[0:2], np.zeros(2))

    def test_fixed_base_keeps_observation_and_action_contract(self):
        obs = observation(0.4)
        self.assertEqual(obs["obs_vec"].shape, (46,))
        provider = FakeJacobianProvider()
        policy = RuleBasedLowPolicy(jacobian_provider=provider)
        command = RuleBasedHighPolicy().predict(obs)
        action = policy.predict(obs, command)
        self.assertEqual(action.shape, (10,))
        np.testing.assert_array_equal(action[0:4], np.zeros(4))

    def test_ppo_low_policy_preserves_hrl4in_contract(self):
        inference_client = FakePPOInferenceClient()
        policy = HRL4INPPOLowPolicy(
            inference_client=inference_client
        )
        obs = observation(0.04)
        command = RuleBasedHighPolicy().predict(obs)
        self.assertEqual(command.mode, TaskMode.ARM_REACH)

        action = policy.predict(obs, command)
        self.assertEqual(action.shape, (10,))
        np.testing.assert_array_equal(action[0:4], np.zeros(4))
        self.assertAlmostEqual(action[4], 0.25, places=5)
        self.assertEqual(inference_client.observations[0].shape, (68,))

        diagnostics = policy.diagnostics()
        self.assertEqual(diagnostics["policy_type"], "hrl4in_ppo")
        self.assertAlmostEqual(diagnostics["inference_ms"], 1.5)
        policy.close()
        self.assertTrue(inference_client.closed)

    def test_ppo_policy_executes_rule_based_planar_base_action(self):
        inference_client = FakePPOInferenceClient()
        policy = HRL4INPPOLowPolicy(
            enable_base_motion=True,
            base_gain=5.0,
            inference_client=inference_client,
        )
        obs = observation(1.2)
        command = RuleBasedHighPolicy().predict(obs)
        self.assertEqual(command.mode, TaskMode.BASE_APPROACH)

        action = policy.predict(obs, command)
        self.assertGreater(action[0], 0.0)
        self.assertEqual(action[1], 0.0)
        self.assertEqual(action[2], 0.0)
        self.assertEqual(action[3], 0.0)
        np.testing.assert_array_equal(action[4:10], np.zeros(6))
        self.assertEqual(len(inference_client.observations), 0)

    def test_high_policy_uses_remaining_planar_displacement(self):
        policy = RuleBasedHighPolicy(max_position_subgoal=0.10)
        obs = observation(1.2, base_x=0.5)
        command = policy.predict(obs)
        self.assertEqual(command.mode, TaskMode.BASE_APPROACH)
        self.assertGreater(command.subgoal[0], 0.0)
        self.assertLessEqual(
            float(np.linalg.norm(command.subgoal[0:2])),
            0.100001,
        )

    def test_arm_dls_recomputes_error_on_every_low_step(self):
        provider = FakeJacobianProvider()
        policy = RuleBasedLowPolicy(
            jacobian_provider=provider,
            max_cartesian_speed=0.05,
            dls_damping=0.03,
        )
        high_policy = RuleBasedHighPolicy()
        first_observation = observation(0.04)
        command = high_policy.predict(first_observation)
        self.assertEqual(command.mode, TaskMode.ARM_REACH)

        first_action = policy.predict(first_observation, command)
        low_observation = policy.low_level_observation(first_observation)
        # Keep the final target at x=0.04 and move the end effector to x=0.02.
        # A HRL4IN-style low layer must track the fixed absolute subgoal from
        # the first command instead of replacing it with a new final target.
        second_observation = observation(0.02, ee_x=0.02)
        policy.observe_transition(second_observation)
        diagnostics = policy.diagnostics()
        second_action = policy.predict(second_observation, command)

        self.assertGreater(first_action[4], 0.0)
        self.assertGreater(second_action[4], 0.0)
        self.assertLess(second_action[4], first_action[4])
        self.assertEqual(len(provider.positions), 2)
        np.testing.assert_array_equal(first_action[0:4], np.zeros(4))
        self.assertEqual(low_observation["vector"].shape, (68,))
        np.testing.assert_array_equal(
            low_observation["subgoal"][0:3],
            np.zeros(3),
        )

        self.assertAlmostEqual(
            diagnostics["absolute_position_subgoal"][0],
            0.04,
            places=5,
        )
        self.assertAlmostEqual(
            diagnostics["post_error_norm"],
            0.02,
            places=5,
        )
        self.assertGreater(diagnostics["intrinsic_reward"], 0.0)

    def test_scheduler_records_pre_and_post_low_step_error(self):
        provider = FakeJacobianProvider()
        base_env = FakeBaseEnv()
        env = HierarchicalEnv(
            base_env,
            RuleBasedHighPolicy(),
            RuleBasedLowPolicy(jacobian_provider=provider),
            high_interval=3,
        )
        env.reset()
        env.observation = observation(0.04)
        _, _, _, info = env.step()

        diagnostics = info["low_level_diagnostics"]
        self.assertAlmostEqual(diagnostics["error_norm"], 0.04, places=5)
        # FakeBaseEnv does not move the end effector, so the fixed 0.04 m
        # subgoal remains 0.04 m away even though its final target changes.
        self.assertAlmostEqual(diagnostics["post_error_norm"], 0.04, places=5)
        self.assertAlmostEqual(diagnostics["intrinsic_reward"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
