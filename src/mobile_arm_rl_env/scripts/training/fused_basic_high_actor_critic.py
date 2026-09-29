#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal high-level actor-critic for the two-layer PPO baseline."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class FusedBasicHighActorCritic(nn.Module):
    """Predict one semantic stage and one physical 6-D subgoal.

    There is deliberately no route output, option graph or learned
    termination head in this baseline.  The observation does include the
    active planner waypoint/progress used by the rule teacher, so imitation is
    Markov rather than depending on hidden teacher state.  The sign and
    magnitude of the continuous base subgoal express how the robot should
    move; the discrete output only selects ``DIRECT``, ``DETOUR`` or
    ``TERMINAL``.
    """

    OBS_DIM = 86
    ACTION_DIM = 6
    STAGE_COUNT = 3
    NORMALIZED_DIM = 35
    POLICY_TYPE = "fused_basic_two_layer_ppo_v2"

    def __init__(self, hidden_sizes=(256, 256), initial_log_std=-2.5):
        super(FusedBasicHighActorCritic, self).__init__()
        self.actor_backbone = self._make_backbone(hidden_sizes)
        self.critic_backbone = self._make_backbone(hidden_sizes)
        output_dim = int(hidden_sizes[-1]) if hidden_sizes else self.OBS_DIM
        self.actor_mean = nn.Linear(
            output_dim, self.STAGE_COUNT * self.ACTION_DIM
        )
        self.stage_head = nn.Linear(output_dim, self.STAGE_COUNT)
        self.critic = nn.Linear(output_dim, 1)
        self.log_std = nn.Parameter(torch.full(
            (self.ACTION_DIM,), float(initial_log_std)
        ))
        self._initialize()

    def act(self, observations, deterministic=False):
        all_means, stage_logits, values = self._outputs(observations)
        stage_distribution = Categorical(logits=stage_logits)
        stage = (
            torch.argmax(stage_logits, dim=-1)
            if deterministic else stage_distribution.sample()
        )
        mean = self._select_stage_means(all_means, stage)
        action_distribution = self._distribution(mean)
        raw_action = (
            mean if deterministic else action_distribution.rsample()
        )
        action = torch.tanh(raw_action)
        log_probability = self._log_probability(
            action_distribution, raw_action, action
        ) + stage_distribution.log_prob(stage)
        return action, stage, log_probability, values

    def evaluate_actions(self, observations, actions, stages):
        all_means, stage_logits, values = self._outputs(observations)
        stages = stages.long()
        mean = self._select_stage_means(all_means, stages)
        action_distribution = self._distribution(mean)
        stage_distribution = Categorical(logits=stage_logits)
        clipped = torch.clamp(actions, -1.0 + 1e-6, 1.0 - 1e-6)
        raw_action = 0.5 * (
            torch.log1p(clipped) - torch.log1p(-clipped)
        )
        log_probability = self._log_probability(
            action_distribution, raw_action, clipped
        ) + stage_distribution.log_prob(stages)
        entropy = (
            action_distribution.entropy().sum(dim=-1)
            + stage_distribution.entropy()
        )
        return (
            log_probability,
            entropy,
            values,
            torch.tanh(mean),
            stage_logits,
        )

    def deterministic_decision(self, observations):
        all_means, stage_logits, unused_values = self._outputs(observations)
        stages = torch.argmax(stage_logits, dim=-1)
        actions = torch.tanh(
            self._select_stage_means(all_means, stages)
        )
        return actions, stages

    def deterministic_action(self, observations, stages=None):
        all_means, stage_logits, unused_values = self._outputs(observations)
        if stages is None:
            stages = torch.argmax(stage_logits, dim=-1)
        return torch.tanh(
            self._select_stage_means(all_means, stages.long())
        )

    def get_value(self, observations):
        features = self.critic_backbone(observations)
        return self.critic(features).squeeze(-1)

    def teacher_kl(
            self,
            observations,
            teacher_actions,
            teacher_stages,
            teacher_standard_deviation=0.10):
        """KL from a narrow rule-teacher distribution to this policy.

        The rule teacher is deterministic, so its bounded action is treated
        as the mean of a narrow Gaussian before the common tanh transform.
        A true delta distribution would have infinite continuous KL.  The
        finite teacher standard deviation makes the imitation strength
        explicit and lets the loss supervise the student's exploration
        variance as well as its action mean.
        """
        teacher_standard_deviation = float(teacher_standard_deviation)
        if teacher_standard_deviation <= 0.0:
            raise ValueError("teacher_standard_deviation must be positive")
        all_means, stage_logits, unused_values = self._outputs(observations)
        teacher_stages = teacher_stages.long()
        student_means = self._select_stage_means(
            all_means, teacher_stages
        )
        clipped_actions = torch.clamp(
            teacher_actions, -1.0 + 1e-4, 1.0 - 1e-4
        )
        teacher_means = 0.5 * (
            torch.log1p(clipped_actions)
            - torch.log1p(-clipped_actions)
        )
        student_log_std = torch.clamp(self.log_std, -5.0, 0.0)
        student_variance = torch.exp(2.0 * student_log_std)
        teacher_variance = teacher_standard_deviation ** 2
        action_kl = (
            student_log_std
            - np.log(teacher_standard_deviation)
            + (
                teacher_variance
                + (teacher_means - student_means).pow(2)
            ) / (2.0 * student_variance)
            - 0.5
        ).mean(dim=-1)
        stage_kl = F.cross_entropy(
            stage_logits, teacher_stages, reduction="none"
        )
        return action_kl, stage_kl

    def reference_kl(self, observations, reference_model):
        """KL from a frozen BC reference policy to this policy.

        The continuous KL is calculated on the pre-tanh diagonal Gaussians.
        Since both policies use the same invertible tanh transform, this is
        also the KL between their corresponding squashed distributions.
        Action KL is averaged over the six dimensions so its scale remains
        comparable with the categorical stage KL.
        """
        with torch.no_grad():
            reference_means, reference_logits, unused_reference_values = (
                reference_model._outputs(observations)
            )
            reference_probabilities = torch.softmax(
                reference_logits, dim=-1
            )
            reference_log_probabilities = torch.log_softmax(
                reference_logits, dim=-1
            )
            reference_log_std = torch.clamp(
                reference_model.log_std, -5.0, 0.0
            )
            reference_variance = torch.exp(2.0 * reference_log_std)

        student_means, student_logits, unused_student_values = self._outputs(
            observations
        )
        student_log_probabilities = torch.log_softmax(
            student_logits, dim=-1
        )
        stage_kl = (
            reference_probabilities
            * (reference_log_probabilities - student_log_probabilities)
        ).sum(dim=-1)

        student_log_std = torch.clamp(self.log_std, -5.0, 0.0)
        student_variance = torch.exp(2.0 * student_log_std)
        per_action_kl = (
            student_log_std.view(1, 1, -1)
            - reference_log_std.view(1, 1, -1)
            + (
                reference_variance.view(1, 1, -1)
                + (reference_means - student_means).pow(2)
            ) / (2.0 * student_variance.view(1, 1, -1))
            - 0.5
        )
        per_stage_action_kl = per_action_kl.mean(dim=-1)
        action_kl = (
            reference_probabilities * per_stage_action_kl
        ).sum(dim=-1)
        return action_kl, stage_kl

    def _outputs(self, observations):
        actor_features = self.actor_backbone(observations)
        critic_features = self.critic_backbone(observations)
        return (
            self.actor_mean(actor_features).view(
                -1, self.STAGE_COUNT, self.ACTION_DIM
            ),
            self.stage_head(actor_features),
            self.critic(critic_features).squeeze(-1),
        )

    def _distribution(self, mean):
        standard_deviation = torch.exp(
            torch.clamp(self.log_std, -5.0, 0.0)
        )
        return Normal(mean, standard_deviation.expand_as(mean))

    @staticmethod
    def _select_stage_means(all_means, stages):
        batch = torch.arange(all_means.shape[0], device=all_means.device)
        return all_means[batch, stages.long()]

    @staticmethod
    def _log_probability(distribution, raw_action, action):
        return (
            distribution.log_prob(raw_action)
            - torch.log(1.0 - action.pow(2) + 1e-6)
        ).sum(dim=-1)

    @classmethod
    def _make_backbone(cls, hidden_sizes):
        layers = []
        input_dim = cls.OBS_DIM
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_dim, int(hidden_size)),
                nn.Tanh(),
            ])
            input_dim = int(hidden_size)
        return nn.Sequential(*layers)

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.stage_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
