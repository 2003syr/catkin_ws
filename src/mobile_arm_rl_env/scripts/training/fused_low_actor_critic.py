#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Actor-critic network for coordinated tracked-base and arm control."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class FusedLowActorCritic(nn.Module):
    """66-D fused observation to normalized 8-D nonholonomic action."""

    OBS_DIM = 66
    ACTION_DIM = 8
    ACTION_MASK_SLICE = slice(58, 66)

    def __init__(
            self,
            hidden_sizes=(256, 256),
            initial_log_std=-2.5):
        super(FusedLowActorCritic, self).__init__()
        self.actor_backbone, actor_output_dim = self._make_backbone(
            hidden_sizes
        )
        self.critic_backbone, critic_output_dim = self._make_backbone(
            hidden_sizes
        )
        if actor_output_dim != critic_output_dim:
            raise RuntimeError("actor/critic backbone dimensions differ")
        self.actor_mean = nn.Linear(actor_output_dim, self.ACTION_DIM)
        self.critic = nn.Linear(critic_output_dim, 1)
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

    def act(self, observations, deterministic=False, action_mask=None):
        mean, value = self.forward(observations)
        distribution = self._distribution(mean)
        raw_action = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        action_mask = self._resolve_action_mask(
            observations,
            action_mask,
        )
        masked_action = action * action_mask
        log_probability = self._squashed_log_probability(
            distribution,
            raw_action,
            action,
            action_mask,
        )
        return action, masked_action, log_probability, value

    def evaluate_actions(self, observations, actions, action_mask=None):
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
        action_mask = self._resolve_action_mask(
            observations,
            action_mask,
        )
        log_probability = self._squashed_log_probability(
            distribution,
            raw_actions,
            clipped_actions,
            action_mask,
        )
        entropy = (
            distribution.entropy() * action_mask
        ).sum(dim=-1)
        return log_probability, entropy, value

    def get_value(self, observations):
        unused_mean, value = self.forward(observations)
        return value

    def deterministic_action(self, observations):
        features = self.actor_backbone(observations)
        mean = self.actor_mean(features)
        return torch.tanh(mean) * self._action_mask(observations)

    def load_compatible_state_dict(self, state, actor_only=False):
        target = self.state_dict()
        loaded = set()
        for name, value in state.items():
            if name not in target:
                continue
            if actor_only and not (
                    name.startswith("actor_backbone.")
                    or name.startswith("actor_mean.")):
                continue
            target[name] = value
            loaded.add(name)

        required = set(
            name for name in target
            if (
                name.startswith("actor_backbone.")
                or name.startswith("actor_mean.")
            )
        )
        if not actor_only:
            required.update(
                name for name in target
                if (
                    name.startswith("critic_backbone.")
                    or name.startswith("critic.")
                    or name == "log_std"
                )
            )
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

    @classmethod
    def _action_mask(cls, observations):
        return torch.clamp(
            observations[..., cls.ACTION_MASK_SLICE],
            0.0,
            1.0,
        )

    @classmethod
    def _resolve_action_mask(cls, observations, action_mask):
        if action_mask is None:
            return cls._action_mask(observations)
        if action_mask.shape != observations.shape[:-1] + (
                cls.ACTION_DIM,):
            raise ValueError(
                "action_mask shape {} does not match observations {}".format(
                    tuple(action_mask.shape),
                    tuple(observations.shape),
                )
            )
        return torch.clamp(action_mask, 0.0, 1.0)

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

    def _make_backbone(self, hidden_sizes):
        layers = []
        input_dim = self.OBS_DIM
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, int(hidden_size)))
            layers.append(nn.Tanh())
            input_dim = int(hidden_size)
        return nn.Sequential(*layers), input_dim


class HRL4INArmTransferredFusedPolicy(nn.Module):
    """Tracked-base branch plus a frozen, transferred HRL4IN arm branch.

    The fused observation is normalized by the 66-D dataset normalizer before
    it reaches this module.  The base branch consumes that representation
    directly.  The arm branch reconstructs the raw common state, creates the
    historical 68-D arm-only HRL4IN input, and applies the normalizer stored in
    the source checkpoint.  This preserves the old arm policy without mapping
    its holonomic x/y actions onto the tracked base.
    """

    OBS_DIM = 66
    ACTION_DIM = 8
    ACTION_MASK_SLICE = slice(58, 66)
    POLICY_TYPE = "fused_low_bc_hrl4in_arm"
    OLD_OBS_DIM = 68
    OLD_ACTION_DIM = 10
    OLD_ARM_ACTION_SLICE = slice(4, 10)

    def __init__(
            self,
            base_hidden_sizes,
            arm_hidden_sizes,
            fused_normalizer_state,
            arm_normalizer_state,
            arm_start_distance=0.30,
            base_stop_distance=0.04,
            arm_max_subgoal=0.08):
        super(HRL4INArmTransferredFusedPolicy, self).__init__()
        self.base_hidden_sizes = tuple(
            int(value) for value in base_hidden_sizes
        )
        self.arm_hidden_sizes = tuple(
            int(value) for value in arm_hidden_sizes
        )
        self.arm_start_distance = float(arm_start_distance)
        self.base_stop_distance = float(base_stop_distance)
        self.arm_max_subgoal = float(arm_max_subgoal)
        if self.arm_start_distance <= self.base_stop_distance:
            raise ValueError(
                "arm_start_distance must exceed base_stop_distance"
            )
        if self.arm_max_subgoal <= 0.0:
            raise ValueError("arm_max_subgoal must be positive")

        self.base_backbone, base_output_dim = self._make_backbone(
            self.OBS_DIM,
            self.base_hidden_sizes,
        )
        self.base_mean = nn.Linear(base_output_dim, 2)
        self.arm_backbone, arm_output_dim = self._make_backbone(
            self.OLD_OBS_DIM,
            self.arm_hidden_sizes,
        )
        self.arm_mean = nn.Linear(arm_output_dim, 6)
        self._initialize_base()
        self._register_normalizer(
            "fused",
            fused_normalizer_state,
            self.OBS_DIM,
        )
        self._register_normalizer(
            "arm",
            arm_normalizer_state,
            self.OLD_OBS_DIM,
        )
        self.arm_frozen = False

    @classmethod
    def from_hrl4in_checkpoint(
            cls,
            checkpoint,
            base_hidden_sizes,
            fused_normalizer_state,
            arm_start_distance=0.30,
            base_stop_distance=0.04,
            arm_max_subgoal=0.08):
        observation_dim = int(checkpoint.get("observation_dim", -1))
        action_dim = int(checkpoint.get("action_dim", -1))
        if observation_dim != cls.OLD_OBS_DIM:
            raise ValueError(
                "HRL4IN checkpoint observation_dim must be {}, got {}".format(
                    cls.OLD_OBS_DIM,
                    observation_dim,
                )
            )
        if action_dim != cls.OLD_ACTION_DIM:
            raise ValueError(
                "HRL4IN checkpoint action_dim must be {}, got {}".format(
                    cls.OLD_ACTION_DIM,
                    action_dim,
                )
            )
        arguments = checkpoint.get("arguments", {})
        arm_hidden_sizes = arguments.get("hidden_sizes", [256, 256])
        arm_normalizer_state = checkpoint.get("normalizer")
        if arm_normalizer_state is None:
            raise ValueError(
                "HRL4IN checkpoint does not contain a normalizer"
            )
        model = cls(
            base_hidden_sizes=base_hidden_sizes,
            arm_hidden_sizes=arm_hidden_sizes,
            fused_normalizer_state=fused_normalizer_state,
            arm_normalizer_state=arm_normalizer_state,
            arm_start_distance=arm_start_distance,
            base_stop_distance=base_stop_distance,
            arm_max_subgoal=arm_max_subgoal,
        )
        source_state = checkpoint.get("model")
        if source_state is None:
            raise ValueError("HRL4IN checkpoint does not contain model state")
        arm_state = model.arm_backbone.state_dict()
        for name in arm_state:
            source_name = "backbone." + name
            if source_name not in source_state:
                raise ValueError(
                    "HRL4IN checkpoint is missing {}".format(source_name)
                )
            if arm_state[name].shape != source_state[source_name].shape:
                raise ValueError(
                    "HRL4IN parameter shape mismatch for {}".format(
                        source_name
                    )
                )
            arm_state[name] = source_state[source_name]
        model.arm_backbone.load_state_dict(arm_state)
        source_weight = source_state.get("actor_mean.weight")
        source_bias = source_state.get("actor_mean.bias")
        if source_weight is None or source_bias is None:
            raise ValueError(
                "HRL4IN checkpoint is missing actor_mean parameters"
            )
        with torch.no_grad():
            model.arm_mean.weight.copy_(
                source_weight[cls.OLD_ARM_ACTION_SLICE]
            )
            model.arm_mean.bias.copy_(
                source_bias[cls.OLD_ARM_ACTION_SLICE]
            )
        model.freeze_arm()
        return model

    def deterministic_action(self, observations):
        base_features = self.base_backbone(observations)
        base_action = torch.tanh(self.base_mean(base_features))
        old_observations, base_distance = self._arm_observations(
            observations
        )
        if self.arm_frozen:
            with torch.no_grad():
                arm_features = self.arm_backbone(old_observations)
                arm_action = torch.tanh(self.arm_mean(arm_features))
        else:
            arm_features = self.arm_backbone(old_observations)
            arm_action = torch.tanh(self.arm_mean(arm_features))
        arm_blend = torch.clamp(
            (
                self.arm_start_distance - base_distance
            ) / (
                self.arm_start_distance - self.base_stop_distance
            ),
            0.0,
            1.0,
        ).unsqueeze(-1)
        action = torch.cat(
            (base_action, arm_action * arm_blend),
            dim=-1,
        )
        return action * self._action_mask(observations)

    def trainable_actor_parameters(self):
        parameters = self.trainable_base_parameters()
        if not self.arm_frozen:
            parameters += self.trainable_arm_parameters()
        return parameters

    def trainable_base_parameters(self):
        return (
            list(self.base_backbone.parameters())
            + list(self.base_mean.parameters())
        )

    def trainable_arm_parameters(self):
        return (
            list(self.arm_backbone.parameters())
            + list(self.arm_mean.parameters())
        )

    def freeze_arm(self):
        for parameter in self.arm_backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.arm_mean.parameters():
            parameter.requires_grad = False
        self.arm_backbone.eval()
        self.arm_mean.eval()
        self.arm_frozen = True

    def unfreeze_arm(self):
        for parameter in self.arm_backbone.parameters():
            parameter.requires_grad = True
        for parameter in self.arm_mean.parameters():
            parameter.requires_grad = True
        self.arm_frozen = False

    def train(self, mode=True):
        super(HRL4INArmTransferredFusedPolicy, self).train(mode)
        # The transferred policy must remain deterministic and frozen even
        # while the new base branch is trained.
        if self.arm_frozen:
            self.arm_backbone.eval()
            self.arm_mean.eval()
        return self

    @classmethod
    def _action_mask(cls, observations):
        return torch.clamp(
            observations[..., cls.ACTION_MASK_SLICE],
            0.0,
            1.0,
        )

    def _arm_observations(self, fused_observations):
        raw_common = fused_observations[..., :52] * (
            self.fused_std
        ) + self.fused_mean
        shape = tuple(fused_observations.shape[:-1]) + (
            self.OLD_OBS_DIM,
        )
        old_raw = torch.zeros(
            shape,
            dtype=fused_observations.dtype,
            device=fused_observations.device,
        )
        # Sensor semantics are identical.  Base subgoal entries are forced to
        # zero so the old arm branch sees the arm-only task it was trained on.
        old_raw[..., :46] = raw_common[..., :46]
        arm_subgoal = raw_common[..., 49:52]
        arm_subgoal_norm = torch.norm(
            arm_subgoal,
            dim=-1,
            keepdim=True,
        )
        arm_subgoal_scale = torch.clamp(
            self.arm_max_subgoal
            / torch.clamp(arm_subgoal_norm, min=1.0e-9),
            max=1.0,
        )
        old_raw[..., 49:52] = arm_subgoal * arm_subgoal_scale
        old_raw[..., 55:58] = torch.clamp(
            fused_observations[..., 55:58],
            0.0,
            1.0,
        )
        old_raw[..., 62:68] = torch.clamp(
            fused_observations[..., 60:66],
            0.0,
            1.0,
        )
        old_normalized = old_raw.clone()
        old_normalized[..., :52] = torch.clamp(
            (
                old_raw[..., :52] - self.arm_mean_normalizer
            ) / self.arm_std,
            -self.arm_clip,
            self.arm_clip,
        )
        base_distance = torch.norm(raw_common[..., 46:48], dim=-1)
        return old_normalized, base_distance

    def _register_normalizer(self, prefix, state, expected_dim):
        if int(state["observation_dim"]) != int(expected_dim):
            raise ValueError(
                "{} normalizer observation dimension mismatch".format(
                    prefix
                )
            )
        normalized_dim = int(state["normalized_dim"])
        if normalized_dim != 52:
            raise ValueError(
                "{} normalizer normalized_dim must be 52".format(prefix)
            )
        mean = torch.as_tensor(
            np.asarray(state["mean"], dtype=np.float32)
        )
        variance = torch.as_tensor(
            np.asarray(state["variance"], dtype=np.float32)
        )
        std = torch.sqrt(torch.clamp(variance, min=1.0e-8) + 1.0e-8)
        if prefix == "fused":
            self.register_buffer("fused_mean", mean)
            self.register_buffer("fused_std", std)
            self.register_buffer(
                "fused_clip",
                torch.tensor(float(state["clip"]), dtype=torch.float32),
            )
        else:
            # Avoid the name ``arm_mean`` because it is the output layer.
            self.register_buffer("arm_mean_normalizer", mean)
            self.register_buffer("arm_std", std)
            self.register_buffer(
                "arm_clip",
                torch.tensor(float(state["clip"]), dtype=torch.float32),
            )

    def _initialize_base(self):
        for module in self.base_backbone.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.base_mean.weight, gain=0.01)
        nn.init.constant_(self.base_mean.bias, 0.0)

    @staticmethod
    def _make_backbone(input_dim, hidden_sizes):
        layers = []
        output_dim = int(input_dim)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(output_dim, int(hidden_size)))
            layers.append(nn.Tanh())
            output_dim = int(hidden_size)
        return nn.Sequential(*layers), output_dim
