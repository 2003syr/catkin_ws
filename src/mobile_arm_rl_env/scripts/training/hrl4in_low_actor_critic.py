#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PyTorch actor-critic for the 68-dimensional HRL4IN low-level input."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class RunningObservationNormalizer(object):
    """Normalize continuous state/subgoal values while preserving masks."""

    def __init__(
            self,
            observation_dim=68,
            normalized_dim=52,
            clip=10.0,
            epsilon=1e-4):
        self.observation_dim = int(observation_dim)
        self.normalized_dim = int(normalized_dim)
        self.clip = float(clip)
        self.count = float(epsilon)
        self.mean = np.zeros(self.normalized_dim, dtype=np.float64)
        self.variance = np.ones(self.normalized_dim, dtype=np.float64)

    def update(self, observations):
        observations = np.asarray(observations, dtype=np.float64)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        if observations.shape[1] != self.observation_dim:
            raise ValueError("invalid observation dimension")
        values = observations[:, :self.normalized_dim]
        batch_mean = values.mean(axis=0)
        batch_variance = values.var(axis=0)
        batch_count = values.shape[0]
        self._update_from_moments(
            batch_mean,
            batch_variance,
            batch_count,
        )

    def normalize(self, observations):
        observations = np.asarray(observations, dtype=np.float32)
        if observations.shape[-1] != self.observation_dim:
            raise ValueError("invalid observation dimension")
        normalized = observations.copy()
        standard_deviation = np.sqrt(
            self.variance + 1e-8
        ).astype(np.float32)
        normalized[..., :self.normalized_dim] = (
            normalized[..., :self.normalized_dim]
            - self.mean.astype(np.float32)
        ) / standard_deviation
        normalized[..., :self.normalized_dim] = np.clip(
            normalized[..., :self.normalized_dim],
            -self.clip,
            self.clip,
        )
        return normalized

    def state_dict(self):
        return {
            "observation_dim": self.observation_dim,
            "normalized_dim": self.normalized_dim,
            "clip": self.clip,
            "count": self.count,
            "mean": self.mean.copy(),
            "variance": self.variance.copy(),
        }

    def load_state_dict(self, state):
        if int(state["observation_dim"]) != self.observation_dim:
            raise ValueError("normalizer observation dimension mismatch")
        if int(state["normalized_dim"]) != self.normalized_dim:
            raise ValueError("normalizer normalized dimension mismatch")
        self.clip = float(state["clip"])
        self.count = float(state["count"])
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.variance = np.asarray(
            state["variance"],
            dtype=np.float64,
        ).copy()

    def _update_from_moments(
            self,
            batch_mean,
            batch_variance,
            batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        existing_sum = self.variance * self.count
        batch_sum = batch_variance * batch_count
        correction = (
            delta ** 2
            * self.count
            * batch_count
            / total_count
        )
        new_variance = (
            existing_sum + batch_sum + correction
        ) / total_count

        self.mean = new_mean
        self.variance = np.maximum(new_variance, 1e-8)
        self.count = total_count


class HRL4INLowActorCritic(nn.Module):
    OBS_DIM = 68
    ACTION_DIM = 10
    ACTION_MASK_SLICE = slice(58, 68)

    def __init__(
            self,
            hidden_sizes=(256, 256),
            initial_log_std=-0.5):
        super(HRL4INLowActorCritic, self).__init__()
        layers = []
        input_dim = self.OBS_DIM
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, int(hidden_size)))
            layers.append(nn.Tanh())
            input_dim = int(hidden_size)
        self.backbone = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(input_dim, self.ACTION_DIM)
        self.critic = nn.Linear(input_dim, 1)
        self.log_std = nn.Parameter(torch.full(
            (self.ACTION_DIM,),
            float(initial_log_std),
        ))
        self._initialize()

    def forward(self, observations):
        features = self.backbone(observations)
        mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)
        return mean, value

    def act(self, observations, deterministic=False):
        mean, value = self.forward(observations)
        distribution = self._distribution(mean)
        if deterministic:
            raw_action = mean
        else:
            raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        action_mask = self._action_mask(observations)
        masked_action = action * action_mask
        log_probability = self._squashed_log_probability(
            distribution,
            raw_action,
            action,
            action_mask,
        )
        return action, masked_action, log_probability, value

    def evaluate_actions(self, observations, actions):
        mean, value = self.forward(observations)
        distribution = self._distribution(mean)
        clipped_actions = torch.clamp(
            actions,
            -1.0 + 1e-6,
            1.0 - 1e-6,
        )
        raw_actions = 0.5 * (
            torch.log1p(clipped_actions)
            - torch.log1p(-clipped_actions)
        )
        action_mask = self._action_mask(observations)
        log_probability = self._squashed_log_probability(
            distribution,
            raw_actions,
            clipped_actions,
            action_mask,
        )
        entropy = (
            distribution.entropy() * action_mask
        ).sum(dim=-1)
        mean_action = torch.tanh(mean)
        return log_probability, entropy, value, mean_action

    def get_value(self, observations):
        _, value = self.forward(observations)
        return value

    def deterministic_action(self, observations):
        mean, _ = self.forward(observations)
        action = torch.tanh(mean)
        return action * self._action_mask(observations)

    def _distribution(self, mean):
        standard_deviation = torch.exp(
            torch.clamp(self.log_std, -5.0, 1.0)
        )
        return Normal(mean, standard_deviation.expand_as(mean))

    @classmethod
    def _action_mask(cls, observations):
        return torch.clamp(
            observations[..., cls.ACTION_MASK_SLICE],
            0.0,
            1.0,
        )

    @staticmethod
    def _squashed_log_probability(
            distribution,
            raw_action,
            action,
            action_mask):
        log_probability = distribution.log_prob(raw_action)
        log_probability = log_probability - torch.log(
            1.0 - action.pow(2) + 1e-6
        )
        return (log_probability * action_mask).sum(dim=-1)

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(
                    module.weight,
                    gain=np.sqrt(2.0),
                )
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
