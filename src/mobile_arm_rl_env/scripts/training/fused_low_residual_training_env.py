#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training adapter for bounded residual control around the rule teacher."""

import numpy as np
import rospy

from training.fused_low_training_env import FusedLowLevelTrainingEnv


class FusedLowResidualTrainingEnv(FusedLowLevelTrainingEnv):
    """66-D observation, 5-D residual action, existing safety execution."""

    ACTION_DIM = 5
    RESIDUAL_DIM = 5

    def __init__(self, *args, **kwargs):
        if kwargs.get("enable_teacher", True) is False:
            raise ValueError(
                "FusedLowResidualTrainingEnv requires the coordinated teacher"
            )
        super(FusedLowResidualTrainingEnv, self).__init__(*args, **kwargs)
        if self.teacher is None:
            raise RuntimeError("residual environment teacher is unavailable")
        self.residual_base_scale = float(
            rospy.get_param("~residual_base_scale", 0.10)
        )
        self.residual_cartesian_scale = float(
            rospy.get_param("~residual_cartesian_scale", 0.01)
        )
        self.last_observation = None
        self.last_residual = np.zeros(self.RESIDUAL_DIM, dtype=np.float32)
        self.last_reward_terms = {}

    def reset(self, scenario=None, max_steps=None):
        observation = super(FusedLowResidualTrainingEnv, self).reset(
            scenario=scenario,
            max_steps=max_steps,
        )
        self.last_observation = np.asarray(
            observation, dtype=np.float32
        ).copy()
        self.last_residual.fill(0.0)
        self.last_reward_terms = {}
        return self.last_observation.copy()

    def step(self, residual):
        if self.last_observation is None:
            raise RuntimeError("reset must be called before step")
        residual = np.asarray(residual, dtype=np.float32)
        if residual.shape != (self.RESIDUAL_DIM,):
            raise ValueError(
                "residual must have shape (5,), got {}".format(
                    residual.shape
                )
            )
        residual = np.clip(residual, -1.0, 1.0)
        previous = self.last_observation.copy()
        components = self.teacher.residual_components(
            previous,
            residual,
            base_scale=self.residual_base_scale,
            cartesian_scale=self.residual_cartesian_scale,
        )
        candidate_action = components["action"]
        observation, unused_reward, base_done, info = (
            super(FusedLowResidualTrainingEnv, self).step(candidate_action)
        )
        observation = np.asarray(observation, dtype=np.float32)
        info = dict(info)
        ee_previous = self._ee_distance(previous)
        ee_current = self._ee_distance(observation)
        base_previous = self._base_distance(previous)
        base_current = self._base_distance(observation)
        ee_progress = ee_previous - ee_current
        base_progress = base_previous - base_current
        safe_action = np.asarray(
            info.get("safe_fused_action", candidate_action),
            dtype=np.float64,
        )
        projection = safe_action - candidate_action
        residual_change = residual - self.last_residual

        collision = bool(info.get("collision", False))
        tf_ok = bool(info.get("tf_ok", True))
        success = bool(ee_current < self.success_threshold)
        timed_out = bool(
            not success
            and not collision
            and int(self.base_env.step_count) >= self.max_steps
        )
        reward_terms = {
            "ee_progress": 8.0 * ee_progress,
            "base_progress": (
                2.0 * base_progress
                if components["phase"] in ("BASE_APPROACH", "COORDINATED")
                else 0.0
            ),
            "time": -0.01,
            "residual_effort": -0.05 * float(np.dot(residual, residual)),
            "residual_smoothness": -0.20 * float(
                np.dot(residual_change, residual_change)
            ),
            "safety_projection": -0.50 * float(
                np.dot(projection, projection)
            ),
            "success": 10.0 if success else 0.0,
            "collision": -20.0 if collision else 0.0,
            "tf_error": -20.0 if not tf_ok else 0.0,
            "timeout": -3.0 if timed_out else 0.0,
        }
        reward = float(sum(reward_terms.values()))
        done = bool(base_done or success or collision or not tf_ok)
        info.update({
            "residual": residual.copy(),
            "nominal_action": components["nominal_action"].copy(),
            "candidate_action": candidate_action.copy(),
            "projection": projection.astype(np.float32),
            "residual_phase": components["phase"],
            "ee_distance": float(ee_current),
            "base_distance": float(base_current),
            "ee_progress": float(ee_progress),
            "base_progress": float(base_progress),
            "success": success,
            "collision": collision,
            "tf_ok": tf_ok,
            "timeout": timed_out,
            "reward_terms": reward_terms,
        })
        self.last_observation = observation.copy()
        self.last_residual = residual.copy()
        self.last_reward_terms = dict(reward_terms)
        return observation, reward, done, info

    @staticmethod
    def _ee_distance(observation):
        observation = np.asarray(observation, dtype=np.float64)
        return float(np.linalg.norm(observation[0:3] - observation[6:9]))

    @staticmethod
    def _base_distance(observation):
        observation = np.asarray(observation, dtype=np.float64)
        return float(np.linalg.norm(observation[46:48]))
