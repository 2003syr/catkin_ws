#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Actor-critic for event-driven joint high-level options."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class FusedHighActorCritic(nn.Module):
    """Map the 82-D high state to one legal option and a 6-D subgoal.

    The discrete option is one of ``DIRECT``, ``DETOUR_UPPER``,
    ``DETOUR_LOWER`` and ``TERMINAL``. A single categorical distribution is
    deliberately used: independently predicting semantic type and route side
    made five invalid combinations representable and compounded small
    classification errors over long episodes.
    """

    OBS_DIM = 82
    ACTION_DIM = 6
    SUBGOAL_TYPE_COUNT = 3
    ROUTE_SIDE_COUNT = 3
    OPTION_COUNT = 4
    EXPERT_COUNT = OPTION_COUNT
    NORMALIZED_DIM = 35
    ROUTE_MEMORY_START = 35
    ROUTE_MEMORY_END = 38
    POLICY_TYPE = "fused_joint_subgoal_unified_option_ppo_v2"

    # HighLevelOption ordering without importing ROS-facing modules.
    OPTION_TYPES = (0, 1, 1, 2)
    OPTION_ROUTES = (0, 1, 2, 0)

    def __init__(self, hidden_sizes=(256, 256), initial_log_std=-3.0):
        super(FusedHighActorCritic, self).__init__()
        self.actor_backbone = self._make_backbone(hidden_sizes)
        self.critic_backbone = self._make_backbone(hidden_sizes)
        output_dim = int(hidden_sizes[-1]) if hidden_sizes else self.OBS_DIM
        self.actor_mean = nn.Linear(
            output_dim,
            self.OPTION_COUNT * self.ACTION_DIM,
        )
        self.option_head = nn.Linear(output_dim, self.OPTION_COUNT)
        self.critic = nn.Linear(output_dim, 1)
        self.log_std = nn.Parameter(torch.full(
            (self.ACTION_DIM,), float(initial_log_std)
        ))
        self._initialize()

    def forward(self, observations):
        actor_features = self.actor_backbone(observations)
        critic_features = self.critic_backbone(observations)
        all_means = self.actor_mean(actor_features).view(
            -1, self.OPTION_COUNT, self.ACTION_DIM
        )
        option_logits = self.option_head(actor_features)
        type_logits, route_logits = self._marginal_logits(option_logits)
        return (
            all_means,
            type_logits,
            route_logits,
            self.critic(critic_features).squeeze(-1),
        )

    def act(self, observations, deterministic=False):
        all_means, option_logits, value = self._outputs(observations)
        option_distribution = self._option_distribution(
            option_logits, observations
        )
        option = (
            torch.argmax(option_distribution.logits, dim=-1)
            if deterministic else option_distribution.sample()
        )
        subgoal_type, route_side = self._decode_options(option)
        mean = self._select_option_means(all_means, option)
        distribution = self._distribution(mean)
        raw_action = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        log_probability = self._log_probability(
            distribution, raw_action, action
        ) + option_distribution.log_prob(option)
        return action, subgoal_type, route_side, log_probability, value

    def evaluate_actions(
            self,
            observations,
            actions,
            subgoal_types,
            route_sides):
        all_means, option_logits, value = self._outputs(observations)
        options = self.encode_options(subgoal_types, route_sides)
        mean = self._select_option_means(all_means, options)
        distribution = self._distribution(mean)
        option_distribution = self._option_distribution(
            option_logits, observations
        )
        clipped = torch.clamp(actions, -1.0 + 1e-6, 1.0 - 1e-6)
        raw = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
        log_probability = self._log_probability(
            distribution, raw, clipped
        ) + option_distribution.log_prob(options)
        entropy = (
            distribution.entropy().sum(dim=-1)
            + option_distribution.entropy()
        )
        type_logits, route_logits = self._marginal_logits(option_logits)
        return (
            log_probability,
            entropy,
            value,
            torch.tanh(mean),
            type_logits,
            route_logits,
        )

    def get_value(self, observations):
        features = self.critic_backbone(observations)
        return self.critic(features).squeeze(-1)

    def deterministic_action(
            self,
            observations,
            subgoal_types=None,
            route_sides=None):
        all_means, option_logits, unused_value = self._outputs(observations)
        if subgoal_types is None and route_sides is None:
            options = torch.argmax(
                self._option_distribution(
                    option_logits, observations
                ).logits,
                dim=-1,
            )
        elif subgoal_types is not None and route_sides is not None:
            options = self.encode_options(subgoal_types, route_sides)
        else:
            raise ValueError(
                "subgoal_types and route_sides must be supplied together"
            )
        return torch.tanh(self._select_option_means(all_means, options))

    def deterministic_decision(self, observations):
        all_means, option_logits, unused_value = self._outputs(observations)
        options = torch.argmax(
            self._option_distribution(option_logits, observations).logits,
            dim=-1,
        )
        subgoal_types, route_sides = self._decode_options(options)
        actions = torch.tanh(
            self._select_option_means(all_means, options)
        )
        return actions, subgoal_types, route_sides

    def deterministic_subgoal_type(self, observations):
        unused_action, subgoal_type, unused_route = (
            self.deterministic_decision(observations)
        )
        return subgoal_type

    def deterministic_route_side(self, observations, subgoal_types=None):
        # ``subgoal_types`` remains accepted for API compatibility. Route is
        # always decoded from the same joint option as semantic type.
        unused_action, unused_type, route_side = (
            self.deterministic_decision(observations)
        )
        return route_side

    def option_logits(self, observations):
        return self.option_head(self.actor_backbone(observations))

    def subgoal_type_logits(self, observations):
        type_logits, unused_route = self._marginal_logits(
            self.option_logits(observations)
        )
        return type_logits

    def route_side_logits(self, observations):
        unused_type, route_logits = self._marginal_logits(
            self.option_logits(observations)
        )
        return route_logits

    def actor_state_dict(self):
        return dict(
            (name, value)
            for name, value in self.state_dict().items()
            if name.startswith("actor_backbone.")
            or name.startswith("actor_mean.")
            or name.startswith("option_head.")
        )

    def load_actor_state_dict(self, state):
        target = self.state_dict()
        loaded = []
        for name, value in state.items():
            if not (
                    name.startswith("actor_backbone.")
                    or name.startswith("actor_mean.")
                    or name.startswith("option_head.")):
                continue
            if name in target and tuple(target[name].shape) == tuple(value.shape):
                target[name] = value
                loaded.append(name)
        required = [
            name for name in target
            if name.startswith("actor_backbone.")
            or name.startswith("actor_mean.")
            or name.startswith("option_head.")
        ]
        missing = sorted(set(required) - set(loaded))
        if missing:
            raise RuntimeError(
                "checkpoint is missing high actor parameters: {}".format(
                    missing
                )
            )
        self.load_state_dict(target)

    def _outputs(self, observations):
        actor_features = self.actor_backbone(observations)
        critic_features = self.critic_backbone(observations)
        return (
            self.actor_mean(actor_features).view(
                -1, self.OPTION_COUNT, self.ACTION_DIM
            ),
            self.option_head(actor_features),
            self.critic(critic_features).squeeze(-1),
        )

    def _distribution(self, mean):
        standard_deviation = torch.exp(
            torch.clamp(self.log_std, -5.0, 0.0)
        )
        return Normal(mean, standard_deviation.expand_as(mean))

    @staticmethod
    def _select_option_means(all_means, options):
        batch = torch.arange(all_means.shape[0], device=all_means.device)
        return all_means[batch, options.long()]

    @classmethod
    def _select_means(cls, all_means, subgoal_types, route_sides):
        """Compatibility helper used by focused contract tests."""
        return cls._select_option_means(
            all_means,
            cls.encode_options(subgoal_types, route_sides),
        )

    @classmethod
    def encode_options(cls, subgoal_types, route_sides):
        subgoal_types = subgoal_types.long()
        route_sides = route_sides.long()
        valid = (
            ((subgoal_types == 0) & (route_sides == 0))
            | ((subgoal_types == 1) & (route_sides == 1))
            | ((subgoal_types == 1) & (route_sides == 2))
            | ((subgoal_types == 2) & (route_sides == 0))
        )
        if not bool(torch.all(valid).item()):
            raise ValueError("invalid high-level type/route pair")
        return torch.where(
            subgoal_types == 0,
            torch.zeros_like(subgoal_types),
            torch.where(
                subgoal_types == 2,
                torch.full_like(subgoal_types, 3),
                route_sides,
            ),
        )

    @classmethod
    def _decode_options(cls, options):
        option_types = torch.as_tensor(
            cls.OPTION_TYPES,
            dtype=torch.long,
            device=options.device,
        )
        option_routes = torch.as_tensor(
            cls.OPTION_ROUTES,
            dtype=torch.long,
            device=options.device,
        )
        return option_types[options.long()], option_routes[options.long()]

    @classmethod
    def _option_distribution(cls, option_logits, observations=None):
        valid = torch.ones_like(option_logits, dtype=torch.bool)
        if observations is not None:
            route_memory = observations[
                :, cls.ROUTE_MEMORY_START:cls.ROUTE_MEMORY_END
            ]
            if route_memory.shape[-1] != cls.ROUTE_SIDE_COUNT:
                raise ValueError("route memory has an invalid shape")
            remembered_side = torch.argmax(route_memory, dim=-1)
            remembered = torch.max(
                route_memory[:, 1:], dim=-1
            ).values > 0.5
            # Once a side is latched, only the opposite detour option is
            # illegal. DIRECT and TERMINAL remain available for option exit.
            valid[:, 2] &= ~(remembered & (remembered_side == 1))
            valid[:, 1] &= ~(remembered & (remembered_side == 2))
        masked_logits = torch.where(
            valid,
            option_logits,
            torch.full_like(option_logits, -1.0e9),
        )
        return Categorical(logits=masked_logits)

    @staticmethod
    def _marginal_logits(option_logits):
        type_logits = torch.stack((
            option_logits[:, 0],
            torch.logsumexp(option_logits[:, 1:3], dim=-1),
            option_logits[:, 3],
        ), dim=-1)
        route_logits = torch.stack((
            torch.logsumexp(option_logits[:, (0, 3)], dim=-1),
            option_logits[:, 1],
            option_logits[:, 2],
        ), dim=-1)
        return type_logits, route_logits

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
        nn.init.orthogonal_(self.option_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
