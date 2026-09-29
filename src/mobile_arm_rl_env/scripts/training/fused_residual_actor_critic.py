#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Actor-critic network for bounded task-space residual control."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class FusedResidualActorCritic(nn.Module):
    """66-D observation to [dv, domega, dvx, dvy, dvz].

    The actor mean is initialized to zero, so the policy initially delegates
    to the deterministic coordinated teacher exactly.
    """

    OBS_DIM = 66
    ACTION_DIM = 5
    POLICY_TYPE = "fused_low_residual_ppo"

    def __init__(self, hidden_sizes=(128, 128), initial_log_std=-3.0):
        super(FusedResidualActorCritic, self).__init__()
        self.actor_backbone = self._make_backbone(hidden_sizes)
        self.critic_backbone = self._make_backbone(hidden_sizes)
        actor_dim = int(hidden_sizes[-1]) if hidden_sizes else self.OBS_DIM
        critic_dim = actor_dim
        self.actor_mean = nn.Linear(actor_dim, self.ACTION_DIM)
        self.critic = nn.Linear(critic_dim, 1)
        self.log_std = nn.Parameter(torch.full(
            (self.ACTION_DIM,), float(initial_log_std)
        ))
        self._initialize()

    def forward(self, observations):
        actor_features = self.actor_backbone(observations)
        critic_features = self.critic_backbone(observations)
        return (
            self.actor_mean(actor_features),
            self.critic(critic_features).squeeze(-1),
        )

    def act(self, observations, deterministic=False):
        mean, value = self.forward(observations)
        distribution = self._distribution(mean)
        raw_action = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        log_probability = self._log_probability(
            distribution, raw_action, action
        )
        return action, action, log_probability, value

    def evaluate_actions(self, observations, actions):
        mean, value = self.forward(observations)
        distribution = self._distribution(mean)
        clipped = torch.clamp(actions, -1.0 + 1e-6, 1.0 - 1e-6)
        raw = 0.5 * (
            torch.log1p(clipped) - torch.log1p(-clipped)
        )
        log_probability = self._log_probability(
            distribution, raw, clipped
        )
        entropy = distribution.entropy().sum(dim=-1)
        return log_probability, entropy, value

    def get_value(self, observations):
        unused_mean, value = self.forward(observations)
        return value

    def deterministic_action(self, observations):
        mean = self.actor_mean(self.actor_backbone(observations))
        return torch.tanh(mean)

    def load_compatible_state_dict(self, state):
        target = self.state_dict()
        loaded = set()
        for name, value in state.items():
            if name in target and tuple(target[name].shape) == tuple(value.shape):
                target[name] = value
                loaded.add(name)
        required = set(target.keys())
        missing = sorted(required - loaded)
        if missing:
            raise RuntimeError(
                "checkpoint is missing network parameters: {}".format(
                    missing
                )
            )
        self.load_state_dict(target)

    def _distribution(self, mean):
        standard_deviation = torch.exp(
            torch.clamp(self.log_std, -5.0, 1.0)
        )
        return Normal(mean, standard_deviation.expand_as(mean))

    @staticmethod
    def _log_probability(distribution, raw_action, action):
        return (
            distribution.log_prob(raw_action)
            - torch.log(1.0 - action.pow(2) + 1e-6)
        ).sum(dim=-1)

    @staticmethod
    def _make_backbone(hidden_sizes):
        layers = []
        input_dim = FusedResidualActorCritic.OBS_DIM
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
        # Zero residual at initialization: exact nominal-teacher behavior.
        nn.init.constant_(self.actor_mean.weight, 0.0)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

