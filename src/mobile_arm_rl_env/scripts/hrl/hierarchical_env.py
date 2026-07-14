#!/usr/bin/env python
# -*- coding: utf-8 -*-


class HierarchicalEnv(object):
    """Run high- and low-level policies on two explicit time scales."""

    def __init__(self, base_env, high_policy, low_policy, high_interval=10):
        if int(high_interval) <= 0:
            raise ValueError("high_interval must be positive")
        self.base_env = base_env
        self.high_policy = high_policy
        self.low_policy = low_policy
        self.high_interval = int(high_interval)
        self.observation = None
        self.command = None
        self.low_step = 0
        self.high_step = 0

    def reset(self):
        self.observation = self.base_env.reset()
        self.command = None
        self.low_step = 0
        self.high_step = 0
        if hasattr(self.high_policy, "reset"):
            self.high_policy.reset()
        return self.observation

    def step(self):
        high_updated = self.command is None or (
            self.low_step % self.high_interval == 0
        )
        if high_updated:
            self.command = self.high_policy.predict(self.observation)
            self.high_step += 1

        action = self.low_policy.predict(self.observation, self.command)
        next_obs, reward, done, info = self.base_env.step(action)
        info = dict(info)
        info.update({
            "task_mode": self.command.mode_name,
            "task_reason": self.command.reason,
            "subgoal": self.command.subgoal.copy(),
            "base_priority": self.command.base_priority,
            "arm_priority": self.command.arm_priority,
            "high_updated": high_updated,
            "high_step": self.high_step,
            "low_step": self.low_step,
        })

        self.observation = next_obs
        self.low_step += 1
        return next_obs, reward, done, info

    def stop(self):
        if hasattr(self.base_env, "stop"):
            self.base_env.stop()
        else:
            self.base_env.publish_zero_cmd()

