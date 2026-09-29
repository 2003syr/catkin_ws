#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Actor-critic network for pose-guided planar subgoal control."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class PlanarSubgoalActorCritic(nn.Module):
    OBS_DIM = 62
    ACTION_DIM = 2

    def __init__(
            self,
            hidden_sizes=(128, 128),
            initial_log_std=-1.0):
        super(PlanarSubgoalActorCritic, self).__init__()
        self.actor_backbone, output_dim = self._make_backbone(
            hidden_sizes
        )
        self.critic_backbone, unused_output_dim = self._make_backbone(
            hidden_sizes
        )
        if unused_output_dim != output_dim:
            raise RuntimeError("actor/critic backbone dimensions differ")
        self.actor_mean = nn.Linear(output_dim, self.ACTION_DIM)
        self.critic = nn.Linear(output_dim, 1)
        self.log_std = nn.Parameter(torch.full(
            (self.ACTION_DIM,),
            float(initial_log_std),
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
        return (
            log_probability,
            distribution.entropy().sum(dim=-1),
            value,
        )

    def get_value(self, observations):
        unused_mean, value = self.forward(observations)
        return value

    def deterministic_action(self, observations):
        features = self.actor_backbone(observations)
        mean = self.actor_mean(features)
        return torch.tanh(mean)

    def load_compatible_state_dict(self, state, actor_only=False):
        """Load current checkpoints or legacy shared-backbone checkpoints."""
        target = self.state_dict()
        loaded = set()
        for name, value in state.items():
            if name in target:
                if actor_only and not (
                        name.startswith("actor_backbone.")
                        or name.startswith("actor_mean.")):
                    continue
                target[name] = value
                loaded.add(name)
                continue
            if name.startswith("backbone."):
                suffix = name[len("backbone."):]
                actor_name = "actor_backbone." + suffix
                if actor_name in target:
                    target[actor_name] = value
                    loaded.add(actor_name)
                if not actor_only:
                    critic_name = "critic_backbone." + suffix
                    if critic_name in target:
                        target[critic_name] = value
                        loaded.add(critic_name)

        required = {
            name for name in target
            if (
                name.startswith("actor_backbone.")
                or name.startswith("actor_mean.")
            )
        }
        if not actor_only:
            required.update({
                name for name in target
                if (
                    name.startswith("critic_backbone.")
                    or name.startswith("critic.")
                    or name == "log_std"
                )
            })
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

    def _make_backbone(self, hidden_sizes):
        layers = []
        input_dim = self.OBS_DIM
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, int(hidden_size)))
            layers.append(nn.Tanh())
            input_dim = int(hidden_size)
        return nn.Sequential(*layers), input_dim
