#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import types
import unittest

import numpy as np
import torch


SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "scripts",
))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_high_actor_critic import FusedHighActorCritic
from training.hrl4in_low_actor_critic import RunningObservationNormalizer
from training.train_fused_high_ppo import (
    _balanced_behavior_epoch_indices,
    _closed_loop_selection_key,
    _compact_behavior_episode,
    _dagger_risk_reasons,
    _dagger_teacher_probability,
    _encode_option_labels,
)


class FusedHighPpoTest(unittest.TestCase):

    def test_actor_critic_contract(self):
        model = FusedHighActorCritic(hidden_sizes=[32, 32])
        observations = torch.zeros((4, model.OBS_DIM))
        actions, subgoal_types, route_sides, log_probability, values = model.act(
            observations, deterministic=False
        )
        self.assertEqual(tuple(actions.shape), (4, 6))
        self.assertEqual(tuple(log_probability.shape), (4,))
        self.assertEqual(tuple(subgoal_types.shape), (4,))
        self.assertEqual(tuple(route_sides.shape), (4,))
        self.assertEqual(tuple(values.shape), (4,))
        self.assertTrue(bool(torch.all(actions <= 1.0)))
        self.assertTrue(bool(torch.all(actions >= -1.0)))
        direct_or_terminal = subgoal_types != 1
        self.assertTrue(bool(torch.all(
            route_sides[direct_or_terminal] == 0
        )))
        detour = subgoal_types == 1
        self.assertTrue(bool(torch.all(
            route_sides[detour] >= 1
        )))

    def test_actor_only_checkpoint_does_not_replace_critic(self):
        source = FusedHighActorCritic(hidden_sizes=[16])
        target = FusedHighActorCritic(hidden_sizes=[16])
        critic_before = target.critic.weight.detach().clone()
        target.load_actor_state_dict(source.state_dict())
        self.assertTrue(torch.equal(critic_before, target.critic.weight))
        observation = torch.randn((3, source.OBS_DIM))
        self.assertTrue(torch.allclose(
            source.deterministic_action(observation),
            target.deterministic_action(observation),
        ))

    def test_normalizer_preserves_scan_features(self):
        normalizer = RunningObservationNormalizer(
            observation_dim=82,
            normalized_dim=35,
        )
        observations = np.zeros((2, 82), dtype=np.float32)
        observations[:, 35:] = 0.75
        normalizer.update(observations)
        normalized = normalizer.normalize(observations)
        np.testing.assert_array_equal(
            normalized[:, 35:], observations[:, 35:]
        )

    def test_route_side_selects_a_distinct_continuous_expert(self):
        model = FusedHighActorCritic(hidden_sizes=[])
        with torch.no_grad():
            model.actor_mean.weight.zero_()
            model.actor_mean.bias.zero_()
            upper_offset = 1 * model.ACTION_DIM
            lower_offset = 2 * model.ACTION_DIM
            model.actor_mean.bias[upper_offset + 1] = 0.5
            model.actor_mean.bias[lower_offset + 1] = -0.5
        observations = torch.zeros((1, model.OBS_DIM))
        detour = torch.ones(1, dtype=torch.long)
        upper = torch.ones(1, dtype=torch.long)
        lower = torch.full((1,), 2, dtype=torch.long)
        upper_action = model.deterministic_action(
            observations, detour, upper
        )
        lower_action = model.deterministic_action(
            observations, detour, lower
        )
        self.assertGreater(float(upper_action[0, 1]), 0.0)
        self.assertLess(float(lower_action[0, 1]), 0.0)

    def test_latched_route_forces_matching_detour_expert(self):
        model = FusedHighActorCritic(hidden_sizes=[])
        with torch.no_grad():
            model.option_head.weight.zero_()
            model.option_head.bias[:] = torch.tensor([0.0, 4.0, 5.0, -1.0])
        observations = torch.zeros((2, model.OBS_DIM))
        observations[0, 35:38] = torch.tensor([0.0, 1.0, 0.0])
        observations[1, 35:38] = torch.tensor([0.0, 0.0, 1.0])
        detour = torch.ones(2, dtype=torch.long)
        route = model.deterministic_route_side(observations, detour)
        self.assertEqual(route.tolist(), [1, 2])

    def test_dagger_schedule_finishes_with_pure_student_rounds(self):
        probabilities = [
            _dagger_teacher_probability(index, 5, 2, 0.70, 0.20)
            for index in range(5)
        ]
        np.testing.assert_allclose(
            probabilities, [0.70, 0.45, 0.20, 0.0, 0.0]
        )

    def test_behavior_epoch_balances_each_observed_option(self):
        subgoal_types = np.asarray(
            [0, 1, 1, 1, 1, 1, 2], dtype=np.int64
        )
        route_sides = np.asarray(
            [0, 1, 1, 1, 2, 2, 0], dtype=np.int64
        )
        indices = _balanced_behavior_epoch_indices(
            subgoal_types,
            route_sides,
            np.random.RandomState(123),
        )
        counts = {}
        for index in indices:
            key = (subgoal_types[index], route_sides[index])
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(
            counts,
            {(0, 0): 3, (1, 1): 3, (1, 2): 3, (2, 0): 3},
        )

    def test_closed_loop_selection_prioritizes_success_count(self):
        safer_failure = {
            "successes": 8,
            "direct_successes": 2,
            "collisions": 0,
            "timeouts": 2,
            "mean_final_distance": 0.04,
            "safety_rate": 0.0,
        }
        less_safe_success = dict(safer_failure)
        less_safe_success.update({
            "successes": 9,
            "collisions": 1,
            "timeouts": 0,
            "mean_final_distance": 0.10,
            "safety_rate": 0.10,
        })
        self.assertGreater(
            _closed_loop_selection_key(less_safe_success),
            _closed_loop_selection_key(safer_failure),
        )

    def test_unlocked_alternate_detour_is_not_action_deviation_risk(self):
        observation = np.zeros(82, dtype=np.float32)
        observation[35] = 1.0
        args = types.SimpleNamespace(
            dagger_action_deviation_threshold=0.20,
            dagger_safety_rate_threshold=0.10,
            dagger_progress_stall_steps=3,
        )
        reasons = _dagger_risk_reasons(
            observation,
            np.ones(6, dtype=np.float32),
            1,
            1,
            -np.ones(6, dtype=np.float32),
            1,
            2,
            {},
            0,
            args,
        )
        self.assertEqual(reasons, [])

    def test_unified_option_labels_reject_invalid_pairs(self):
        options = _encode_option_labels(
            np.asarray([0, 1, 1, 2], dtype=np.int64),
            np.asarray([0, 1, 2, 0], dtype=np.int64),
        )
        np.testing.assert_array_equal(options, [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            _encode_option_labels(
                np.asarray([0], dtype=np.int64),
                np.asarray([2], dtype=np.int64),
            )

    def test_dagger_compaction_bounds_timeout_without_losing_risk(self):
        episode = {
            "scenario_id": 1,
            "source": "test",
            "observations": [np.asarray([index]) for index in range(20)],
            "actions": [np.zeros(6) for _ in range(20)],
            "types": [0 for _ in range(20)],
            "routes": [0 for _ in range(20)],
            "sample_weights": [1.0 for _ in range(20)],
            "interventions": [index == 9 for index in range(20)],
            "risk_flags": [index in (9, 18) for index in range(20)],
        }
        compact = _compact_behavior_episode(episode, 5)
        self.assertEqual(len(compact["observations"]), 5)
        self.assertTrue(any(compact["interventions"]))
        self.assertTrue(any(compact["risk_flags"]))


if __name__ == "__main__":
    unittest.main()
