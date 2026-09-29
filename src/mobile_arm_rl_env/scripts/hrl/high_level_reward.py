#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Minimal first-stage reward for joint-subgoal high-level control."""

from __future__ import division

import numpy as np


class HighLevelReward(object):

    def __init__(
            self,
            progress_weight=10.0,
            path_progress_weight=10.0,
            final_base_progress_weight=0.0,
            final_yaw_progress_weight=0.0,
            terminal_progress_weight=0.0,
            success_reward=100.0,
            collision_penalty=100.0,
            timeout_penalty=10.0,
            subgoal_failure_penalty=5.0,
            minimum_subgoal_progress=0.01,
            low_step_penalty=0.0,
            option_stall_penalty=0.0,
            invalid_terminal_penalty=0.0):
        self.progress_weight = float(progress_weight)
        self.path_progress_weight = float(path_progress_weight)
        self.final_base_progress_weight = float(
            final_base_progress_weight
        )
        self.final_yaw_progress_weight = float(
            final_yaw_progress_weight
        )
        self.terminal_progress_weight = float(
            terminal_progress_weight
        )
        self.success_reward = float(success_reward)
        self.collision_penalty = float(collision_penalty)
        self.timeout_penalty = float(timeout_penalty)
        self.subgoal_failure_penalty = float(
            subgoal_failure_penalty
        )
        self.minimum_subgoal_progress = float(
            minimum_subgoal_progress
        )
        self.low_step_penalty = float(low_step_penalty)
        self.option_stall_penalty = float(option_stall_penalty)
        self.invalid_terminal_penalty = float(
            invalid_terminal_penalty
        )
        if min(
                self.progress_weight,
                self.path_progress_weight,
                self.final_base_progress_weight,
                self.final_yaw_progress_weight,
                self.terminal_progress_weight,
                self.success_reward,
                self.collision_penalty,
                self.timeout_penalty,
                self.subgoal_failure_penalty,
                self.minimum_subgoal_progress,
                self.low_step_penalty,
                self.option_stall_penalty,
                self.invalid_terminal_penalty) < 0.0:
            raise ValueError("high-level reward values must be non-negative")

    def calculate(
            self,
            start_goal_distance,
            end_goal_distance,
            start_subgoal_error,
            end_subgoal_error,
            success=False,
            collision=False,
            timeout=False,
            start_path_remaining=None,
            end_path_remaining=None,
            start_final_base_distance=None,
            end_final_base_distance=None,
            start_final_yaw_error=None,
            end_final_yaw_error=None,
            terminal_stage=False,
            low_steps=0,
            option_stalled=False,
            invalid_terminal=False):
        start_error = self._vector(
            start_subgoal_error, "start_subgoal_error"
        )
        end_error = self._vector(
            end_subgoal_error, "end_subgoal_error"
        )
        goal_progress = (
            float(start_goal_distance) - float(end_goal_distance)
        )
        path_progress = 0.0
        if (
                start_path_remaining is not None
                and end_path_remaining is not None):
            path_progress = (
                float(start_path_remaining)
                - float(end_path_remaining)
            )
        final_base_progress = self._optional_progress(
            start_final_base_distance,
            end_final_base_distance,
        )
        final_yaw_progress = self._optional_progress(
            None if start_final_yaw_error is None else abs(float(
                start_final_yaw_error
            )),
            None if end_final_yaw_error is None else abs(float(
                end_final_yaw_error
            )),
        )
        base_progress = (
            float(np.linalg.norm(start_error[0:3]))
            - float(np.linalg.norm(end_error[0:3]))
        )
        ee_progress = (
            float(np.linalg.norm(start_error[3:6]))
            - float(np.linalg.norm(end_error[3:6]))
        )
        subgoal_failed = bool(
            not success
            and not collision
            and base_progress < self.minimum_subgoal_progress
            and ee_progress < self.minimum_subgoal_progress
        )
        terms = {
            "goal_progress": self.progress_weight * goal_progress,
            "path_progress": self.path_progress_weight * path_progress,
            "final_base_progress": (
                self.final_base_progress_weight * final_base_progress
            ),
            "final_yaw_progress": (
                self.final_yaw_progress_weight * final_yaw_progress
            ),
            "terminal_goal_progress": (
                self.terminal_progress_weight * goal_progress
                if terminal_stage else 0.0
            ),
            "success": self.success_reward if success else 0.0,
            "collision": -self.collision_penalty if collision else 0.0,
            "timeout": -self.timeout_penalty if timeout else 0.0,
            "subgoal_failure": (
                -self.subgoal_failure_penalty if subgoal_failed else 0.0
            ),
            "low_step_cost": -self.low_step_penalty * max(
                int(low_steps), 0
            ),
            "option_stall": (
                -self.option_stall_penalty if option_stalled else 0.0
            ),
            "invalid_terminal": (
                -self.invalid_terminal_penalty
                if invalid_terminal else 0.0
            ),
        }
        return float(sum(terms.values())), {
            "terms": terms,
            "goal_progress": float(goal_progress),
            "path_progress": float(path_progress),
            "final_base_progress": float(final_base_progress),
            "final_yaw_progress": float(final_yaw_progress),
            "base_subgoal_progress": float(base_progress),
            "ee_subgoal_progress": float(ee_progress),
            "subgoal_failed": subgoal_failed,
            "terminal_stage": bool(terminal_stage),
            "option_stalled": bool(option_stalled),
            "invalid_terminal": bool(invalid_terminal),
        }

    @staticmethod
    def _optional_progress(start_value, end_value):
        if start_value is None or end_value is None:
            return 0.0
        return float(start_value) - float(end_value)

    @staticmethod
    def _vector(value, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (6,):
            raise ValueError(
                "{} must have shape (6,), got {}".format(
                    name,
                    vector.shape,
                )
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("{} contains non-finite values".format(name))
        return vector
