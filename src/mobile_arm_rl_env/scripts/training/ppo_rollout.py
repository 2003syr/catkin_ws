#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rollout storage and GAE for single-environment low-level PPO."""

import numpy as np
import torch


class PPORolloutBuffer(object):
    def __init__(
            self,
            capacity,
            observation_dim=68,
            action_dim=10):
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        if self.capacity <= 0:
            raise ValueError("rollout capacity must be positive")

        self.observations = np.zeros(
            (self.capacity, self.observation_dim),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.capacity, self.action_dim),
            dtype=np.float32,
        )
        self.teacher_actions = np.zeros(
            (self.capacity, self.action_dim),
            dtype=np.float32,
        )
        self.teacher_valid = np.zeros(
            self.capacity,
            dtype=np.float32,
        )
        self.log_probabilities = np.zeros(
            self.capacity,
            dtype=np.float32,
        )
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.advantages = np.zeros(self.capacity, dtype=np.float32)
        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0

    @property
    def full(self):
        return self.size >= self.capacity

    def add(
            self,
            observation,
            action,
            teacher_action,
            teacher_valid,
            log_probability,
            value,
            reward,
            done):
        if self.full:
            raise RuntimeError("rollout buffer is full")
        index = self.size
        self.observations[index] = observation
        self.actions[index] = action
        self.teacher_actions[index] = teacher_action
        self.teacher_valid[index] = float(bool(teacher_valid))
        self.log_probabilities[index] = float(log_probability)
        self.values[index] = float(value)
        self.rewards[index] = float(reward)
        self.dones[index] = float(bool(done))
        self.size += 1

    def compute_returns_and_advantages(
            self,
            last_value,
            gamma=0.99,
            gae_lambda=0.95):
        if self.size == 0:
            raise RuntimeError("cannot compute returns for an empty rollout")
        last_advantage = 0.0
        next_value = float(last_value)
        for index in reversed(range(self.size)):
            non_terminal = 1.0 - self.dones[index]
            delta = (
                self.rewards[index]
                + float(gamma) * next_value * non_terminal
                - self.values[index]
            )
            last_advantage = (
                delta
                + float(gamma)
                * float(gae_lambda)
                * non_terminal
                * last_advantage
            )
            self.advantages[index] = last_advantage
            self.returns[index] = (
                self.advantages[index] + self.values[index]
            )
            next_value = self.values[index]

    def tensors(self, device):
        active = slice(0, self.size)
        return {
            "observations": torch.from_numpy(
                self.observations[active]
            ).to(device),
            "actions": torch.from_numpy(
                self.actions[active]
            ).to(device),
            "teacher_actions": torch.from_numpy(
                self.teacher_actions[active]
            ).to(device),
            "teacher_valid": torch.from_numpy(
                self.teacher_valid[active]
            ).to(device),
            "old_log_probabilities": torch.from_numpy(
                self.log_probabilities[active]
            ).to(device),
            "old_values": torch.from_numpy(
                self.values[active]
            ).to(device),
            "advantages": torch.from_numpy(
                self.advantages[active]
            ).to(device),
            "returns": torch.from_numpy(
                self.returns[active]
            ).to(device),
        }

    def mini_batch_indices(self, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = np.random.permutation(self.size)
        for start in range(0, self.size, batch_size):
            yield indices[start:start + batch_size]

    def clear(self):
        self.size = 0
