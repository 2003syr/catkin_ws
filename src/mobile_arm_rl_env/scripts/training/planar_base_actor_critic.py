#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Actor-critic network for the compact tracked-base policy."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class PlanarBaseActorCritic(nn.Module):
    OBS_DIM = 11
    ACTION_DIM = 2

    def __init__(
            self,
            hidden_sizes=(128, 128),
            initial_log_std=-1.5):
        super(PlanarBaseActorCritic, self).__init__()
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
        return (
            self.actor_mean(features),
            self.critic(features).squeeze(-1),
        )

    def act(self, observations, deterministic=False):
        mean, value = self.forward(observations)
        distribution = self._distribution(mean)
        raw_action = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        log_probability = self._squashed_log_probability(
            distribution,
            raw_action,
            action,
        )
        return action, log_probability, value

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
        log_probability = self._squashed_log_probability(
            distribution,
            raw_actions,
            clipped_actions,
        )
        entropy = distribution.entropy().sum(dim=-1)
        return (
            log_probability,
            entropy,
            value,
            torch.tanh(mean),
        )

    def get_value(self, observations):
        unused_mean, value = self.forward(observations)
        return value

    def deterministic_action(self, observations):
        mean, unused_value = self.forward(observations)
        return torch.tanh(mean)

    def _distribution(self, mean):
        standard_deviation = torch.exp(
            torch.clamp(self.log_std, -5.0, 1.0)
        )
        return Normal(mean, standard_deviation.expand_as(mean))

    @staticmethod
    def _squashed_log_probability(
            distribution,
            raw_action,
            action):
        log_probability = distribution.log_prob(raw_action)
        log_probability = log_probability - torch.log(
            1.0 - action.pow(2) + 1e-6
        )
        return log_probability.sum(dim=-1)

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

