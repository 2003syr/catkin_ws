#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "scripts",
))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from training.fused_low_actor_critic import (  # noqa: E402
    FusedLowActorCritic,
)
from training.train_fused_low_teacher_ppo import (  # noqa: E402
    _balanced_detour_sides,
    _detour_success_rates,
    _fixed_box_gate_action_mask,
    _masked_weighted_action_mse,
    _normalized_advantages,
    _ppo_mini_batch_indices,
    _ppo_update,
)
from training.train_fused_low_residual_ppo import (  # noqa: E402
    _balanced_rollout_ready as _typed_rollout_ready,
    _normalized_advantages as _typed_normalized_advantages,
    _ppo_mini_batch_indices as _typed_mini_batch_indices,
)
from training.ppo_rollout import PPORolloutBuffer  # noqa: E402


class FixedBoxActionGateTest(unittest.TestCase):

    def test_coordinated_phase_becomes_base_only(self):
        masks = torch.tensor([
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        actual = _fixed_box_gate_action_mask(masks)
        expected = torch.tensor([
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        self.assertTrue(torch.equal(actual, expected))

    def test_actor_uses_override_for_execution_and_probability(self):
        torch.manual_seed(7)
        model = FusedLowActorCritic(hidden_sizes=(16, 16))
        observations = torch.zeros((1, model.OBS_DIM))
        observations[:, 58:66] = 1.0
        fixed_mask = _fixed_box_gate_action_mask(
            observations[:, 58:66]
        )

        raw_action, executed_action, logp, unused_value = model.act(
            observations,
            deterministic=True,
            action_mask=fixed_mask,
        )
        self.assertTrue(torch.equal(
            executed_action[:, 2:8],
            torch.zeros_like(executed_action[:, 2:8]),
        ))

        changed_inactive = raw_action.clone()
        changed_inactive[:, 2:8] = 0.75
        original_logp = model.evaluate_actions(
            observations,
            raw_action,
            action_mask=fixed_mask,
        )[0]
        changed_logp = model.evaluate_actions(
            observations,
            changed_inactive,
            action_mask=fixed_mask,
        )[0]
        self.assertTrue(torch.allclose(original_logp, changed_logp))
        self.assertTrue(torch.isfinite(logp).all())


class ConservativePPOContractTest(unittest.TestCase):

    def test_balanced_detour_order_has_equal_sides_and_flips_start(self):
        first = _balanced_detour_sides(3, update_index=0)
        second = _balanced_detour_sides(3, update_index=1)
        self.assertEqual(first.count(1.0), 3)
        self.assertEqual(first.count(-1.0), 3)
        self.assertEqual(second.count(1.0), 3)
        self.assertEqual(second.count(-1.0), 3)
        self.assertEqual(first[0], 1.0)
        self.assertEqual(second[0], -1.0)

    def test_omega_weight_increases_omega_error_contribution(self):
        predicted = torch.tensor([[0.0, 1.0] + [0.0] * 6])
        target = torch.zeros_like(predicted)
        mask = torch.tensor([[1.0, 1.0] + [0.0] * 6])
        unweighted = _masked_weighted_action_mse(
            predicted,
            target,
            mask,
            torch.ones(8),
        )
        omega_weighted = _masked_weighted_action_mse(
            predicted,
            target,
            mask,
            torch.tensor([1.0, 4.0] + [1.0] * 6),
        )
        self.assertAlmostEqual(float(unweighted.item()), 0.5)
        self.assertAlmostEqual(float(omega_weighted.item()), 0.8)

    def test_balanced_batches_have_equal_upper_and_lower_steps(self):
        labels = [1.0] * 3 + [-1.0] * 7
        buffer = SimpleNamespace(size=len(labels))
        batches = list(_ppo_mini_batch_indices(
            buffer,
            batch_size=4,
            side_labels=labels,
        ))
        self.assertGreater(len(batches), 0)
        labels = np.asarray(labels)
        for indices in batches:
            selected = labels[indices]
            self.assertEqual(int(np.sum(selected > 0.0)), 2)
            self.assertEqual(int(np.sum(selected < 0.0)), 2)

    def test_advantages_are_normalized_separately_by_detour_side(self):
        advantages = torch.tensor([1.0, 3.0, 100.0, 104.0])
        normalized = _normalized_advantages(
            advantages,
            side_labels=[1.0, 1.0, -1.0, -1.0],
        )
        self.assertTrue(torch.allclose(
            normalized,
            torch.tensor([-1.0, 1.0, -1.0, 1.0]),
        ))

    def test_critic_warmup_does_not_change_actor(self):
        torch.manual_seed(11)
        np.random.seed(11)
        model = FusedLowActorCritic(hidden_sizes=(16, 16))
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
        buffer = PPORolloutBuffer(
            capacity=4,
            observation_dim=model.OBS_DIM,
            action_dim=model.ACTION_DIM,
        )
        for index in range(4):
            observation = np.zeros(model.OBS_DIM, dtype=np.float32)
            observation[58:60] = 1.0
            observation[index] = 0.1 * (index + 1)
            tensor = torch.from_numpy(observation).unsqueeze(0)
            with torch.no_grad():
                action, unused_executed, logp, value = model.act(tensor)
            buffer.add(
                observation=observation,
                action=action.squeeze(0).numpy(),
                teacher_action=np.zeros(model.ACTION_DIM, dtype=np.float32),
                teacher_valid=True,
                log_probability=float(logp.item()),
                value=float(value.item()),
                reward=1.0,
                done=(index == 3),
            )
        buffer.compute_returns_and_advantages(
            last_value=0.0,
            gamma=0.99,
            gae_lambda=0.95,
        )
        args = SimpleNamespace(
            ppo_epochs=1,
            batch_size=4,
            clip_ratio=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.0,
            maximum_gradient_norm=0.5,
            fixed_box_action_gate=True,
            teacher_action_weights=[1.0] * 8,
            reference_action_weights=[1.0] * 8,
            reference_coefficient=0.0,
        )
        actor_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name.startswith("actor_") or name == "log_std"
        }
        critic_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name.startswith("critic")
        }
        metrics = _ppo_update(
            model,
            optimizer,
            buffer,
            torch.device("cpu"),
            args,
            teacher_coefficient=1.0,
            reference_model=None,
            critic_only=True,
        )
        for name, before in actor_before.items():
            self.assertTrue(torch.equal(before, dict(
                model.named_parameters()
            )[name]))
        self.assertTrue(any(
            not torch.equal(before, dict(model.named_parameters())[name])
            for name, before in critic_before.items()
        ))
        self.assertTrue(metrics["actor_frozen"])

    def test_detour_success_is_reported_separately(self):
        rates = _detour_success_rates([
            {"side": 1.0, "success": True},
            {"side": 1.0, "success": False},
            {"side": -1.0, "success": True},
        ])
        self.assertEqual(rates["upper"], {
            "episodes": 2,
            "success": 0.5,
        })
        self.assertEqual(rates["lower"], {
            "episodes": 1,
            "success": 1.0,
        })


class TypedResidualPPOContractTest(unittest.TestCase):

    def test_rollout_waits_for_all_types_and_option_boundary(self):
        labels = [0] * 4 + [1] * 4 + [2] * 4
        buffer = SimpleNamespace(size=len(labels))
        self.assertFalse(_typed_rollout_ready(
            buffer,
            labels,
            target_steps=12,
            minimum_type_steps=4,
            option_done=False,
        ))
        self.assertTrue(_typed_rollout_ready(
            buffer,
            labels,
            target_steps=12,
            minimum_type_steps=4,
            option_done=True,
        ))
        self.assertFalse(_typed_rollout_ready(
            buffer,
            [0] * 4 + [1] * 7 + [2],
            target_steps=12,
            minimum_type_steps=4,
            option_done=True,
        ))

    def test_typed_batches_have_equal_samples(self):
        labels = [0] * 3 + [1] * 7 + [2] * 5
        buffer = SimpleNamespace(size=len(labels))
        batches = list(_typed_mini_batch_indices(
            buffer,
            batch_size=6,
            subgoal_type_labels=labels,
        ))
        self.assertTrue(batches)
        labels_array = np.asarray(labels)
        for batch in batches:
            selected = labels_array[batch]
            self.assertEqual(int(np.sum(selected == 0)), 2)
            self.assertEqual(int(np.sum(selected == 1)), 2)
            self.assertEqual(int(np.sum(selected == 2)), 2)

    def test_advantages_are_normalized_per_subgoal_type(self):
        advantages = torch.tensor([
            1.0, 3.0,
            10.0, 14.0,
            -5.0, 5.0,
        ])
        labels = [0, 0, 1, 1, 2, 2]
        normalized = _typed_normalized_advantages(
            advantages,
            labels,
        )
        labels_tensor = torch.tensor(labels)
        for subgoal_type in (0, 1, 2):
            selected = normalized[labels_tensor == subgoal_type]
            self.assertAlmostEqual(float(selected.mean()), 0.0, places=6)
            self.assertAlmostEqual(
                float(selected.std(unbiased=False)),
                1.0,
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
