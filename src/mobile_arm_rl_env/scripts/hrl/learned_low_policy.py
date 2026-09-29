#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NumPy inference adapter for a behavior-cloned low-level arm policy."""

import os

import numpy as np

from hrl.high_level_command import TaskMode


class LearnedLowPolicy(object):
    """Load an exported MLP while preserving the existing HRL policy API."""

    OBS_DIM = 46
    ACTION_DIM = 10
    ARM_ACTION_DIM = 6
    ARM_ACTION_SLICE = slice(4, 10)
    ARM_POSITION_SLICE = slice(15, 21)

    def __init__(self, model_path, recovery_gain=0.4):
        self.model_path = os.path.abspath(os.path.expanduser(model_path))
        self.recovery_gain = float(recovery_gain)
        self.observation_mean = None
        self.observation_std = None
        self.weights = []
        self.biases = []
        self.last_diagnostics = {}
        self._load_model()

    def predict(self, observation, command):
        observation = self._as_observation(observation)
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        self.last_diagnostics = {}

        if command.mode == TaskMode.ARM_REACH:
            arm_action = self._forward(observation)
            action[self.ARM_ACTION_SLICE] = arm_action
            error_base = (
                observation[0:3].astype(np.float64)
                - observation[6:9].astype(np.float64)
            )
            self.last_diagnostics = {
                "policy_type": "learned_behavior_cloning",
                "error_base": error_base,
                "error_norm": float(np.linalg.norm(error_base)),
                # These fields preserve the existing runner's diagnostic
                # contract. A learned action has no explicit Cartesian
                # velocity or Jacobian solve, so NaN marks them as not
                # applicable instead of reporting fabricated values.
                "cartesian_velocity": np.full(3, np.nan),
                "jacobian_condition": float("nan"),
                "normalized_arm_action": arm_action.copy(),
            }

        elif command.mode == TaskMode.RECOVERY:
            action[self.ARM_ACTION_SLICE] = (
                -self.recovery_gain
                * observation[self.ARM_POSITION_SLICE]
            )

        return np.clip(action, -1.0, 1.0)

    def diagnostics(self):
        result = {}
        for key, value in self.last_diagnostics.items():
            result[key] = value.copy() if hasattr(value, "copy") else value
        return result

    def _forward(self, observation):
        hidden = (
            observation.astype(np.float32) - self.observation_mean
        ) / self.observation_std
        last_layer = len(self.weights) - 1
        for index, (weight, bias) in enumerate(
                zip(self.weights, self.biases)):
            hidden = np.dot(weight, hidden) + bias
            if index < last_layer:
                hidden = np.maximum(hidden, 0.0)
            else:
                hidden = np.tanh(hidden)
        return np.asarray(hidden, dtype=np.float32)

    def _load_model(self):
        if not os.path.isfile(self.model_path):
            raise IOError("learned low-policy model not found: {}".format(
                self.model_path
            ))

        archive = np.load(self.model_path)
        try:
            format_version = self._scalar(archive, "format_version")
            observation_dim = self._scalar(archive, "observation_dim")
            action_dim = self._scalar(archive, "action_dim")
            layer_count = self._scalar(archive, "layer_count")
            if format_version != 1:
                raise ValueError(
                    "unsupported learned-policy format version: {}".format(
                        format_version
                    )
                )
            if observation_dim != self.OBS_DIM:
                raise ValueError(
                    "model observation dim is {}, expected {}".format(
                        observation_dim,
                        self.OBS_DIM,
                    )
                )
            if action_dim != self.ARM_ACTION_DIM:
                raise ValueError(
                    "model action dim is {}, expected {}".format(
                        action_dim,
                        self.ARM_ACTION_DIM,
                    )
                )
            if layer_count < 1:
                raise ValueError("model must contain at least one layer")

            self.observation_mean = np.asarray(
                archive["observation_mean"],
                dtype=np.float32,
            ).copy()
            self.observation_std = np.asarray(
                archive["observation_std"],
                dtype=np.float32,
            ).copy()
            for index in range(layer_count):
                self.weights.append(np.asarray(
                    archive["weight_{}".format(index)],
                    dtype=np.float32,
                ).copy())
                self.biases.append(np.asarray(
                    archive["bias_{}".format(index)],
                    dtype=np.float32,
                ).copy())
        finally:
            archive.close()

        self._validate_model_shapes()

    def _validate_model_shapes(self):
        if self.observation_mean.shape != (self.OBS_DIM,):
            raise ValueError("observation_mean has an invalid shape")
        if self.observation_std.shape != (self.OBS_DIM,):
            raise ValueError("observation_std has an invalid shape")
        if np.any(self.observation_std <= 0.0):
            raise ValueError("observation_std must be positive")

        input_dim = self.OBS_DIM
        for index, (weight, bias) in enumerate(
                zip(self.weights, self.biases)):
            if weight.ndim != 2 or weight.shape[1] != input_dim:
                raise ValueError(
                    "weight_{} has invalid shape {}".format(
                        index,
                        weight.shape,
                    )
                )
            if bias.shape != (weight.shape[0],):
                raise ValueError(
                    "bias_{} has invalid shape {}".format(
                        index,
                        bias.shape,
                    )
                )
            input_dim = weight.shape[0]
        if input_dim != self.ARM_ACTION_DIM:
            raise ValueError(
                "final layer outputs {}, expected {}".format(
                    input_dim,
                    self.ARM_ACTION_DIM,
                )
            )

    @classmethod
    def _as_observation(cls, observation):
        if isinstance(observation, dict):
            observation = observation["obs_vec"]
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != (cls.OBS_DIM,):
            raise ValueError(
                "observation must have shape (46,), got {}".format(
                    vector.shape
                )
            )
        return vector

    @staticmethod
    def _scalar(archive, key):
        return int(np.asarray(archive[key]).reshape(-1)[0])
