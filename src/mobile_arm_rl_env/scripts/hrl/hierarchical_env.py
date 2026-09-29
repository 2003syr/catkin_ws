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
        self.steps_since_high_update = 0
        self.subgoal_achieved = False
        self.subgoal_done = False
        self.episode_done = False
        self.success_latched = False

    def reset(self):
        self.observation = self.base_env.reset()
        self.command = None
        self.low_step = 0
        self.high_step = 0
        self.steps_since_high_update = 0
        self.subgoal_achieved = False
        self.subgoal_done = False
        self.episode_done = False
        self.success_latched = False
        if hasattr(self.high_policy, "reset"):
            self.high_policy.reset()
        if hasattr(self.low_policy, "reset"):
            self.low_policy.reset()
        return self.observation

    def step(self):
        if self.episode_done:
            raise RuntimeError(
                "episode is done; call reset() before another policy step"
            )

        high_updated = (
            self.command is None
            or self.subgoal_done
        )
        if high_updated:
            self.command = self.high_policy.predict(self.observation)
            self.high_step += 1
            self.steps_since_high_update = 0
            self.subgoal_achieved = False
            self.subgoal_done = False
            if hasattr(self.low_policy, "begin_subgoal"):
                self.low_policy.begin_subgoal(
                    self.observation,
                    self.command,
                )

        action = self.low_policy.predict(self.observation, self.command)
        next_obs, reward, done, info = self.base_env.step(action)
        self.episode_done = bool(done)
        self.success_latched = bool(
            done and info.get("success", False)
        )
        if hasattr(self.low_policy, "observe_transition"):
            subgoal_timed_out = bool(
                self.steps_since_high_update + 1 >= self.high_interval
            )
            collision_reward = float(info.get("collision_reward", 0.0))
            try:
                self.low_policy.observe_transition(
                    next_obs,
                    extrinsic_reward=reward,
                    collision_reward=collision_reward,
                    episode_done=done,
                    subgoal_timed_out=subgoal_timed_out,
                )
            except TypeError:
                # Preserve the original minimal policy interface for simple
                # diagnostic policies that only accept next_observation.
                self.low_policy.observe_transition(next_obs)

        low_level_diagnostics = (
            self.low_policy.diagnostics()
            if hasattr(self.low_policy, "diagnostics") else {}
        )
        if (
                low_level_diagnostics
                and "error_base" in low_level_diagnostics
                and "post_error_norm" not in low_level_diagnostics):
            next_vector = next_obs["obs_vec"] if isinstance(
                next_obs, dict
            ) else next_obs
            post_error_base = next_vector[0:3] - next_vector[6:9]
            low_level_diagnostics["post_error_base"] = post_error_base.copy()
            low_level_diagnostics["post_error_norm"] = float(
                sum(value * value for value in post_error_base) ** 0.5
            )

        self.subgoal_achieved = bool(
            low_level_diagnostics.get("subgoal_achieved", False)
        )
        self.subgoal_done = bool(
            low_level_diagnostics.get(
                "subgoal_done",
                self.subgoal_achieved
                or self.steps_since_high_update + 1 >= self.high_interval
                or done,
            )
        )
        low_level_observation = (
            self.low_policy.low_level_observation(next_obs)
            if hasattr(self.low_policy, "low_level_observation")
            else None
        )
        info = dict(info)
        info.update({
            "task_mode": self.command.mode_name,
            "task_reason": self.command.reason,
            "subgoal": self.command.subgoal.copy(),
            "remaining_subgoal": low_level_diagnostics.get(
                "remaining_subgoal",
                self.command.subgoal,
            ).copy(),
            "ideal_next_state": low_level_diagnostics.get(
                "ideal_next_state",
                self.command.subgoal,
            ).copy(),
            "base_priority": self.command.base_priority,
            "arm_priority": self.command.arm_priority,
            "action_mask": self.command.action_mask.copy(),
            "subgoal_mask": self.command.subgoal_mask.copy(),
            "high_updated": high_updated,
            "high_step": self.high_step,
            "low_step": self.low_step,
            "subgoal_achieved": self.subgoal_achieved,
            "subgoal_done": self.subgoal_done,
            "subgoal_timed_out": bool(
                low_level_diagnostics.get("subgoal_timed_out", False)
            ),
            "low_level_mask": float(
                low_level_diagnostics.get(
                    "low_level_mask",
                    0.0 if self.subgoal_done else 1.0,
                )
            ),
            "subgoal_steps": int(
                low_level_diagnostics.get(
                    "subgoal_steps",
                    self.steps_since_high_update + 1,
                )
            ),
            "intrinsic_reward": float(
                low_level_diagnostics.get("intrinsic_reward", 0.0)
            ),
            "subgoal_distance": float(
                low_level_diagnostics.get("post_error_norm", 0.0)
            ),
            "low_level_diagnostics": low_level_diagnostics,
            "low_level_observation": low_level_observation,
        })

        self.observation = next_obs
        self.low_step += 1
        self.steps_since_high_update += 1
        return next_obs, reward, done, info

    def hold_zero(self):
        """Keep all controllers at zero without advancing policy or reward."""
        if hasattr(self.base_env, "publish_zero_cmd"):
            self.base_env.publish_zero_cmd()
        else:
            raise RuntimeError(
                "base environment does not expose publish_zero_cmd()"
            )

    def stop(self):
        # Stop the robot before closing an external inference connection.
        # A stalled socket must never delay the zero-velocity shutdown path.
        try:
            if hasattr(self.base_env, "stop"):
                self.base_env.stop()
            else:
                self.base_env.publish_zero_cmd()
        finally:
            if hasattr(self.low_policy, "close"):
                self.low_policy.close()
