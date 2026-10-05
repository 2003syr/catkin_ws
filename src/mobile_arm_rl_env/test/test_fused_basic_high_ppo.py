#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import unittest
import copy
from types import SimpleNamespace
import numpy as np

import torch


SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"
))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_basic_high_actor_critic import (
    FusedBasicHighActorCritic,
)
from training.hrl4in_low_actor_critic import RunningObservationNormalizer
from training.ppo_rollout import PPORolloutBuffer
from training.train_fused_basic_high_ppo import (
    _balanced_ppo_minibatch_indices,
    _balanced_stage_episode_indices,
    _collect_dagger_dataset,
    _dagger_intervention_required,
    _dagger_teacher_probability,
    _discount_reference_low_steps,
    _explained_variance,
    _initialize_ppo_from_checkpoint,
    _load_dagger_dataset,
    _low_step_gamma,
    _mean_abs,
    _normalize_ppo_advantages,
    _parameter_gradient_norm,
    _ppo_update,
    _recovery_checkpoint_path,
    _restore_random_state,
    _save_dagger_dataset,
    _stage_term_means,
    _new_terminal_blend_curriculum,
    _restore_terminal_blend_curriculum,
    _update_terminal_blend_curriculum,
)


class _OneScenarioSampler(object):

    def next(self):
        return {"scenario_id": 7, "step_budget": 3}


class _TwoStepDaggerClient(object):

    def __init__(self):
        self.teacher_action = np.full(6, 0.25, dtype=np.float32)
        self.teacher_info = {"subgoal_type": 0}
        self.executed = []
        self.steps = 0

    def reset_with_info(self, scenario=None, max_steps=None):
        self.steps = 0
        return np.zeros(
            FusedBasicHighActorCritic.OBS_DIM, dtype=np.float32
        ), {}

    def step(self, action, subgoal_type):
        self.executed.append((np.asarray(action).copy(), int(subgoal_type)))
        self.steps += 1
        done = self.steps >= 2
        return (
            np.full(
                FusedBasicHighActorCritic.OBS_DIM,
                self.steps,
                dtype=np.float32,
            ),
            0.0,
            done,
            {
                "success": done,
                "collision": False,
                "timeout": False,
                "low_steps": 1,
            },
        )


class _ConstantStudent(torch.nn.Module):

    def deterministic_decision(self, observations):
        count = int(observations.shape[0])
        return (
            torch.zeros((count, 6), dtype=torch.float32),
            torch.zeros((count,), dtype=torch.long),
        )


class FusedBasicHighPpoTest(unittest.TestCase):

    def test_terminal_blend_curriculum_advances_only_after_gate(self):
        args = SimpleNamespace(
            terminal_blend_annealing=True,
            terminal_blend_schedule=[0.20, 0.60, 1.00],
            terminal_blend_minimum_success_rate=0.80,
            terminal_blend_maximum_collisions=0,
            terminal_blend_required_gates=2,
        )
        curriculum = _new_terminal_blend_curriculum(args)
        failed = {"success_rate": 0.70, "collisions": 0}
        passed = {"success_rate": 0.80, "collisions": 0}

        event = _update_terminal_blend_curriculum(
            curriculum, failed, args
        )
        self.assertFalse(event["advanced"])
        self.assertEqual(curriculum["stage_index"], 0)

        event = _update_terminal_blend_curriculum(
            curriculum, passed, args
        )
        self.assertFalse(event["advanced"])
        self.assertEqual(event["gate_streak"], 1)

        event = _update_terminal_blend_curriculum(
            curriculum, passed, args
        )
        self.assertTrue(event["stage_completed"])
        self.assertTrue(event["advanced"])
        self.assertAlmostEqual(event["completed_blend"], 0.20)
        self.assertAlmostEqual(curriculum["active_blend"], 0.60)

    def test_terminal_blend_curriculum_restores_exact_stage(self):
        args = SimpleNamespace(
            terminal_blend_annealing=True,
            terminal_blend_schedule=[0.20, 0.40, 1.00],
        )
        saved = {
            "enabled": True,
            "schedule": [0.20, 0.40, 0.80, 1.00],
            "stage_index": 2,
            "active_blend": 0.80,
            "gate_streak": 1,
            "completed_stage_index": 1,
            "completed": False,
        }
        restored = _restore_terminal_blend_curriculum(saved, args)
        self.assertEqual(restored["stage_index"], 2)
        self.assertAlmostEqual(restored["active_blend"], 0.80)
        self.assertEqual(restored["gate_streak"], 1)
        self.assertEqual(restored["completed_stage_index"], 1)

    def test_actor_contract_has_stage_and_subgoal_only(self):
        model = FusedBasicHighActorCritic(hidden_sizes=[32, 32])
        observations = torch.zeros((5, model.OBS_DIM))
        actions, stages, log_probability, values = model.act(observations)
        self.assertEqual(model.OBS_DIM, 86)
        self.assertEqual(model.STAGE_COUNT, 3)
        self.assertFalse(hasattr(model, "route_head"))
        self.assertFalse(hasattr(model, "option_head"))
        self.assertEqual(tuple(actions.shape), (5, 6))
        self.assertEqual(tuple(stages.shape), (5,))
        self.assertEqual(tuple(log_probability.shape), (5,))
        self.assertEqual(tuple(values.shape), (5,))
        self.assertTrue(bool(torch.all(stages >= 0)))
        self.assertTrue(bool(torch.all(stages < 3)))

    def test_each_stage_has_a_conditional_continuous_head(self):
        model = FusedBasicHighActorCritic(hidden_sizes=[])
        with torch.no_grad():
            model.actor_mean.weight.zero_()
            model.actor_mean.bias.zero_()
            model.actor_mean.bias[1] = 0.5
            model.actor_mean.bias[model.ACTION_DIM + 1] = -0.5
        observations = torch.zeros((2, model.OBS_DIM))
        stages = torch.tensor([0, 1], dtype=torch.long)
        actions = model.deterministic_action(observations, stages)
        self.assertGreater(float(actions[0, 1]), 0.0)
        self.assertLess(float(actions[1, 1]), 0.0)

    def test_teacher_kl_is_finite_for_saturated_rule_actions(self):
        model = FusedBasicHighActorCritic(hidden_sizes=[16])
        observations = torch.zeros((3, model.OBS_DIM))
        teacher_actions = torch.tensor([
            [1.0, -1.0, 0.0, 0.1, -0.1, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.2, -0.2, 0.3, -0.3],
        ])
        teacher_stages = torch.tensor([0, 1, 2], dtype=torch.long)
        action_kl, stage_kl = model.teacher_kl(
            observations,
            teacher_actions,
            teacher_stages,
            teacher_standard_deviation=0.10,
        )
        self.assertEqual(tuple(action_kl.shape), (3,))
        self.assertEqual(tuple(stage_kl.shape), (3,))
        self.assertTrue(bool(torch.all(torch.isfinite(action_kl))))
        self.assertTrue(bool(torch.all(torch.isfinite(stage_kl))))
        self.assertTrue(bool(torch.all(action_kl >= 0.0)))
        self.assertTrue(bool(torch.all(stage_kl >= 0.0)))

    def test_reference_kl_is_zero_for_identical_frozen_policy(self):
        model = FusedBasicHighActorCritic(hidden_sizes=[16])
        reference = copy.deepcopy(model)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        observations = torch.randn((7, model.OBS_DIM))
        action_kl, stage_kl = model.reference_kl(
            observations, reference
        )
        self.assertTrue(torch.allclose(
            action_kl, torch.zeros_like(action_kl), atol=1e-7
        ))
        self.assertTrue(torch.allclose(
            stage_kl, torch.zeros_like(stage_kl), atol=1e-7
        ))

    def test_reference_kl_detects_student_policy_drift(self):
        model = FusedBasicHighActorCritic(hidden_sizes=[])
        reference = copy.deepcopy(model)
        with torch.no_grad():
            model.actor_mean.bias.add_(0.25)
            model.stage_head.bias[1] = 0.5
        observations = torch.zeros((4, model.OBS_DIM))
        action_kl, stage_kl = model.reference_kl(
            observations, reference
        )
        self.assertGreater(float(action_kl.mean()), 0.0)
        self.assertGreater(float(stage_kl.mean()), 0.0)

    def test_stage_balanced_advantages_have_per_stage_zero_mean(self):
        advantages = torch.tensor([
            1.0, 2.0, 10.0, 12.0, -5.0, -1.0,
        ])
        stages = torch.tensor([0, 0, 1, 1, 2, 2])
        normalized = _normalize_ppo_advantages(advantages, stages)
        for stage in range(3):
            selected = normalized[stages == stage]
            self.assertAlmostEqual(float(selected.mean()), 0.0, places=6)
            self.assertAlmostEqual(
                float(selected.std(unbiased=False)), 1.0, places=6
            )

    def test_ppo_diagnostic_helpers_report_value_fit_and_gradient_norm(self):
        targets = torch.tensor([1.0, 2.0, 3.0])
        perfect = targets.clone()
        constant = torch.zeros_like(targets)
        self.assertAlmostEqual(
            _explained_variance(perfect, targets), 1.0, places=6
        )
        self.assertLess(_explained_variance(constant, targets), 1.0)

        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3.0, 4.0])
        self.assertAlmostEqual(
            _parameter_gradient_norm([parameter]), 5.0, places=6
        )

    def test_rollout_smdp_uses_option_duration_for_td_and_gae(self):
        buffer = PPORolloutBuffer(
            capacity=2, observation_dim=1, action_dim=1
        )
        for value, reward, done, duration in (
                (1.0, 3.0, False, 2.0),
                (2.0, 4.0, True, 3.0)):
            buffer.add(
                observation=np.zeros(1, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                teacher_action=np.zeros(1, dtype=np.float32),
                teacher_valid=False,
                log_probability=0.0,
                value=value,
                reward=reward,
                done=done,
                duration=duration,
            )
        buffer.compute_returns_and_advantages(
            last_value=9.0,
            gamma=0.9,
            gae_lambda=0.95,
            duration_discount_reference=1.0,
        )

        expected_last_advantage = 4.0 - 2.0
        expected_first_delta = 3.0 + (0.9 ** 2.0) * 2.0 - 1.0
        expected_first_advantage = (
            expected_first_delta
            + (0.9 ** 2.0) * 0.95 * expected_last_advantage
        )
        self.assertAlmostEqual(
            float(buffer.transition_discounts[0]), 0.9 ** 2.0, places=6
        )
        self.assertAlmostEqual(
            float(buffer.transition_discounts[1]), 0.9 ** 3.0, places=6
        )
        self.assertAlmostEqual(
            float(buffer.advantages[0]), expected_first_advantage, places=6
        )
        self.assertAlmostEqual(float(buffer.advantages[1]), 2.0, places=6)
        self.assertAlmostEqual(
            float(buffer.returns[0]), expected_first_advantage + 1.0,
            places=6,
        )

    def test_rollout_high_step_mode_remains_backward_compatible(self):
        buffer = PPORolloutBuffer(
            capacity=1, observation_dim=1, action_dim=1
        )
        buffer.add(
            observation=np.zeros(1, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            teacher_action=np.zeros(1, dtype=np.float32),
            teacher_valid=False,
            log_probability=0.0,
            value=1.0,
            reward=3.0,
            done=False,
            duration=40.0,
        )
        buffer.compute_returns_and_advantages(
            last_value=2.0,
            gamma=0.9,
            gae_lambda=0.95,
        )
        self.assertAlmostEqual(
            float(buffer.transition_discounts[0]), 0.9, places=6
        )
        self.assertAlmostEqual(float(buffer.returns[0]), 4.8, places=6)

    def test_rollout_smdp_reference_preserves_nominal_gamma(self):
        buffer = PPORolloutBuffer(
            capacity=1, observation_dim=1, action_dim=1
        )
        buffer.add(
            observation=np.zeros(1, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            teacher_action=np.zeros(1, dtype=np.float32),
            teacher_valid=False,
            log_probability=0.0,
            value=0.0,
            reward=0.0,
            done=False,
            duration=80.0,
        )
        buffer.compute_returns_and_advantages(
            last_value=1.0,
            gamma=0.99,
            gae_lambda=0.95,
            duration_discount_reference=80.0,
        )
        self.assertAlmostEqual(
            float(buffer.transition_discounts[0]), 0.99, places=6
        )
        tensors = buffer.tensors(torch.device("cpu"))
        self.assertAlmostEqual(float(tensors["durations"][0]), 80.0)
        self.assertAlmostEqual(
            float(tensors["transition_discounts"][0]), 0.99, places=6
        )

    def test_smdp_reference_defaults_to_environment_interval(self):
        args = SimpleNamespace(
            discount_mode="smdp",
            smdp_discount_reference_low_steps=0.0,
        )
        reference = _discount_reference_low_steps(
            args, {"high_level_interval": 80}
        )
        self.assertEqual(reference, 80.0)
        self.assertAlmostEqual(
            _low_step_gamma(0.99, "smdp", reference) ** reference,
            0.99,
            places=6,
        )

        args.smdp_discount_reference_low_steps = 40.0
        self.assertEqual(
            _discount_reference_low_steps(
                args, {"high_level_interval": 80}
            ),
            40.0,
        )

    def test_actor_kl_stop_does_not_cancel_critic_epochs(self):
        torch.manual_seed(23)
        np.random.seed(23)
        model = FusedBasicHighActorCritic(hidden_sizes=[16])
        reference = copy.deepcopy(model)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
        buffer = PPORolloutBuffer(
            capacity=8,
            observation_dim=model.OBS_DIM,
            action_dim=model.ACTION_DIM,
        )
        observations = torch.randn((8, model.OBS_DIM))
        with torch.no_grad():
            actions, stages, log_probabilities, values = model.act(
                observations
            )
        teacher_action = np.full(
            model.ACTION_DIM, 0.75, dtype=np.float32
        )
        for index in range(8):
            buffer.add(
                observations[index].numpy(),
                actions[index].numpy(),
                teacher_action,
                True,
                float(log_probabilities[index]),
                float(values[index]),
                0.0,
                False,
            )
        buffer.advantages[:buffer.size] = np.linspace(
            -1.0, 1.0, buffer.size, dtype=np.float32
        )
        buffer.returns[:buffer.size] = (
            values.numpy()
            + np.linspace(-4.0, 4.0, buffer.size, dtype=np.float32)
        )
        critic_before = {
            name: value.clone()
            for name, value in model.state_dict().items()
            if name.startswith(("critic_backbone.", "critic."))
        }
        args = SimpleNamespace(
            stage_balanced_advantages=False,
            ppo_epochs=3,
            batch_size=4,
            clip_ratio=0.10,
            teacher_kl_std=0.10,
            teacher_terminal_weight=1.0,
            teacher_stage_coefficient=0.20,
            reference_stage_coefficient=0.20,
            entropy_coefficient=0.0,
            value_coefficient=0.25,
            maximum_gradient_norm=1.0,
            target_policy_kl=1.0e-12,
            policy_kl_stop_multiplier=1.5,
        )

        metrics = _ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            stages=stages.numpy(),
            teacher_stages=stages.numpy(),
            device=torch.device("cpu"),
            args=args,
            teacher_coefficient=1.0,
            reference_model=reference,
            reference_coefficient=0.0,
        )

        self.assertTrue(bool(metrics["kl_early_stop"]))
        self.assertEqual(int(metrics["epochs_completed"]), 1)
        self.assertEqual(int(metrics["full_epochs_completed"]), 0)
        self.assertEqual(int(metrics["minibatches_completed"]), 1)
        self.assertEqual(int(metrics["minibatches_expected"]), 6)
        self.assertEqual(int(metrics["critic_full_epochs_completed"]), 3)
        self.assertEqual(int(metrics["critic_minibatches_completed"]), 6)
        self.assertEqual(int(metrics["critic_minibatches_expected"]), 6)
        self.assertAlmostEqual(
            float(metrics["critic_update_fraction"]), 1.0
        )
        self.assertEqual(len(metrics["policy_kl_epoch_values"]), 0)
        self.assertEqual(len(metrics["policy_kl_minibatch_values"]), 1)
        self.assertEqual(int(metrics["policy_kl_check_count"]), 1)
        self.assertEqual(int(metrics["policy_kl_stop_epoch"]), 1)
        self.assertEqual(int(metrics["policy_kl_stop_minibatch"]), 1)
        self.assertEqual(
            int(metrics["policy_kl_stop_global_minibatch"]), 1
        )
        self.assertAlmostEqual(
            float(metrics["policy_kl_threshold"]), 1.5e-12
        )
        self.assertGreater(float(metrics["policy_kl"]), 0.0)
        self.assertAlmostEqual(
            float(metrics["policy_kl"]),
            float(metrics["policy_kl_minibatch_values"][-1]),
            places=6,
        )
        self.assertGreater(
            float(metrics["reference_kl"]),
            float(metrics["reference_kl_update_mean"]),
        )
        critic_changed = any(
            not torch.equal(model.state_dict()[name], value)
            for name, value in critic_before.items()
        )
        self.assertTrue(critic_changed)

    def test_ppo_minibatches_balance_small_rollout_remainder(self):
        np.random.seed(19)
        batches = list(_balanced_ppo_minibatch_indices(516, 128))
        sizes = [len(batch) for batch in batches]
        self.assertEqual(len(batches), 5)
        self.assertEqual(sum(sizes), 516)
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        self.assertEqual(
            sorted(np.concatenate(batches).tolist()),
            list(range(516)),
        )

    def test_rollout_diagnostic_helpers_preserve_stage_denominators(self):
        self.assertAlmostEqual(
            _mean_abs(np.asarray([-1.0, 0.0, 2.0], dtype=np.float32)),
            1.0,
            places=6,
        )
        means = _stage_term_means(
            {
                "DIRECT:goal_progress": 4.0,
                "TERMINAL:goal_progress": 3.0,
            },
            {"DIRECT": 2, "TERMINAL": 3},
        )
        self.assertAlmostEqual(means["DIRECT"]["goal_progress"], 2.0)
        self.assertAlmostEqual(means["TERMINAL"]["goal_progress"], 1.0)

    def test_ppo_initialization_loads_actor_but_keeps_fresh_critic(self):
        source = FusedBasicHighActorCritic(hidden_sizes=[16])
        target = FusedBasicHighActorCritic(hidden_sizes=[16])
        source_normalizer = RunningObservationNormalizer(
            observation_dim=source.OBS_DIM,
            normalized_dim=source.NORMALIZED_DIM,
        )
        target_normalizer = RunningObservationNormalizer(
            observation_dim=target.OBS_DIM,
            normalized_dim=target.NORMALIZED_DIM,
        )
        source_normalizer.update(np.ones((2, source.OBS_DIM)))
        with torch.no_grad():
            source.actor_mean.weight.fill_(0.25)
            source.critic.weight.fill_(9.0)
        fresh_critic = {
            name: value.clone()
            for name, value in target.state_dict().items()
            if name.startswith(("critic_backbone.", "critic."))
        }
        checkpoint = {
            "model": source.state_dict(),
            "normalizer": source_normalizer.state_dict(),
            "observation_dim": source.OBS_DIM,
            "action_dim": source.ACTION_DIM,
            "subgoal_type_count": source.STAGE_COUNT,
            "policy_type": source.POLICY_TYPE,
        }

        _initialize_ppo_from_checkpoint(
            checkpoint, target, target_normalizer
        )

        self.assertTrue(torch.equal(
            target.actor_mean.weight, source.actor_mean.weight
        ))
        for name, expected in fresh_critic.items():
            self.assertTrue(torch.equal(target.state_dict()[name], expected))
        self.assertTrue(np.allclose(
            target_normalizer.mean, source_normalizer.mean
        ))

    def test_recovery_checkpoint_path_is_separate_from_selected_output(self):
        self.assertEqual(
            _recovery_checkpoint_path("/tmp/basic_high.pt"),
            "/tmp/basic_high.recovery_latest.pt",
        )
        self.assertEqual(
            _recovery_checkpoint_path("/tmp/basic_high"),
            "/tmp/basic_high.recovery_latest.pt",
        )

    def test_random_state_restore_replays_all_training_rngs(self):
        import random

        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        expected = (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(1).item()),
        )
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)

        self.assertTrue(_restore_random_state(state))
        actual = (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(1).item()),
        )
        self.assertEqual(actual, expected)
        self.assertFalse(_restore_random_state(None))

    def test_dagger_takeover_catches_stage_or_action_disagreement(self):
        zeros = np.zeros(6, dtype=np.float32)
        intervention, stage_disagreement, deviation = (
            _dagger_intervention_required(
                zeros, 0, zeros, 1, 0.30
            )
        )
        self.assertTrue(intervention)
        self.assertTrue(stage_disagreement)
        self.assertEqual(deviation, 0.0)

        student = zeros.copy()
        student[2] = 0.31
        intervention, stage_disagreement, deviation = (
            _dagger_intervention_required(
                student, 1, zeros, 1, 0.30
            )
        )
        self.assertTrue(intervention)
        self.assertFalse(stage_disagreement)
        self.assertGreater(deviation, 0.30)

    def test_dagger_takeover_uses_stage_and_terminal_arm_thresholds(self):
        zeros = np.zeros(6, dtype=np.float32)
        thresholds = (0.20, 0.15, 0.08)
        student = zeros.copy()
        student[4] = 0.06

        direct = _dagger_intervention_required(
            student,
            0,
            zeros,
            0,
            thresholds,
            terminal_arm_deviation_threshold=0.05,
        )
        terminal = _dagger_intervention_required(
            student,
            2,
            zeros,
            2,
            thresholds,
            terminal_arm_deviation_threshold=0.05,
        )

        self.assertFalse(direct[0])
        self.assertTrue(terminal[0])
        self.assertFalse(terminal[1])
        self.assertAlmostEqual(terminal[2], 0.06, places=5)

    def test_dagger_base_deviation_can_remain_on_student_distribution(self):
        zeros = np.zeros(6, dtype=np.float32)
        student = zeros.copy()
        student[0] = 0.25
        diagnostic_only = _dagger_intervention_required(
            student,
            1,
            zeros,
            1,
            (0.20, 0.15, 0.08),
            terminal_arm_deviation_threshold=0.05,
            base_action_deviation_takeover=False,
        )
        guarded = _dagger_intervention_required(
            student,
            1,
            zeros,
            1,
            (0.20, 0.15, 0.08),
            terminal_arm_deviation_threshold=0.05,
            base_action_deviation_takeover=True,
        )
        self.assertFalse(diagnostic_only[0])
        self.assertTrue(guarded[0])

    def test_bc_sampling_does_not_let_one_long_episode_dominate(self):
        stages = np.asarray(
            [0] * 100 + [0] * 2 + [1] * 4 + [2] * 4,
            dtype=np.int64,
        )
        episode_ids = np.asarray(
            [10] * 100 + [11] * 2 + [20] * 4 + [30] * 4,
            dtype=np.int64,
        )
        indices = _balanced_stage_episode_indices(
            stages, episode_ids, np.random.RandomState(123)
        )
        sampled_stages = stages[indices]
        target_per_stage = max(
            int(np.sum(stages == stage)) for stage in range(3)
        )
        self.assertEqual(
            [int(np.sum(sampled_stages == stage)) for stage in range(3)],
            [target_per_stage] * 3,
        )
        direct_episode_ids = episode_ids[indices[sampled_stages == 0]]
        long_fraction = float(np.mean(direct_episode_ids == 10))
        self.assertGreater(long_fraction, 0.35)
        self.assertLess(long_fraction, 0.65)

    def test_dagger_teacher_probability_decays_across_rounds(self):
        probabilities = [
            _dagger_teacher_probability(index, 3, 0.80, 0.20)
            for index in (1, 2, 3)
        ]
        self.assertAlmostEqual(probabilities[0], 0.80)
        self.assertAlmostEqual(probabilities[1], 0.50)
        self.assertAlmostEqual(probabilities[2], 0.20)

    def test_dagger_labels_student_visited_states_with_teacher(self):
        client = _TwoStepDaggerClient()
        model = _ConstantStudent()
        normalizer = RunningObservationNormalizer(
            observation_dim=FusedBasicHighActorCritic.OBS_DIM,
            normalized_dim=FusedBasicHighActorCritic.NORMALIZED_DIM,
        )
        dataset = _collect_dagger_dataset(
            client=client,
            model=model,
            normalizer=normalizer,
            sampler=_OneScenarioSampler(),
            default_budget=3,
            episodes=1,
            device=torch.device("cpu"),
            teacher_probability=0.0,
            action_deviation_threshold=1.0,
            seed=123,
            dagger_round=1,
        )
        self.assertEqual(
            tuple(dataset["observations"].shape),
            (2, FusedBasicHighActorCritic.OBS_DIM),
        )
        self.assertTrue(np.allclose(dataset["actions"], 0.25))
        self.assertTrue(np.all(dataset["stages"] == 0))
        self.assertEqual(dataset["risk_intervention_count"], 0)
        self.assertEqual(dataset["teacher_execution_steps"], 0)
        self.assertEqual(tuple(dataset["final_distance"].shape), (2,))
        self.assertEqual(
            tuple(dataset["terminal_pose_stage"].shape), (2,)
        )
        self.assertTrue(all(
            np.allclose(action, 0.0)
            for action, unused_stage in client.executed
        ))
        self.assertEqual([stage for unused_action, stage in client.executed], [
            0, 0
        ])

    def test_dagger_dataset_is_saved_atomically_and_resumable(self):
        client = _TwoStepDaggerClient()
        model = _ConstantStudent()
        normalizer = RunningObservationNormalizer(
            observation_dim=FusedBasicHighActorCritic.OBS_DIM,
            normalized_dim=FusedBasicHighActorCritic.NORMALIZED_DIM,
        )
        dataset = _collect_dagger_dataset(
            client=client,
            model=model,
            normalizer=normalizer,
            sampler=_OneScenarioSampler(),
            default_budget=3,
            episodes=1,
            device=torch.device("cpu"),
            teacher_probability=0.0,
            action_deviation_threshold=1.0,
            seed=123,
            dagger_round=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dagger.latest.npz")
            _save_dagger_dataset(path, dataset, 1, "round_0001")
            restored, completed_round = _load_dagger_dataset(path)
            self.assertEqual(completed_round, 1)
            self.assertTrue(np.array_equal(
                restored["observations"], dataset["observations"]
            ))
            self.assertTrue(np.array_equal(
                restored["actions"], dataset["actions"]
            ))
            self.assertTrue(np.array_equal(
                restored["terminal_pose_stage"],
                dataset["terminal_pose_stage"],
            ))


if __name__ == "__main__":
    unittest.main()
