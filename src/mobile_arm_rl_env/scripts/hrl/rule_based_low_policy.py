#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from hrl.high_level_command import TaskMode


class RuleBasedLowPolicy(object):
    """Diagnostic low-level controller for checking hierarchical wiring.

    This policy is deliberately simple and is not an inverse-kinematics
    controller. It will be replaced by a goal-conditioned RL policy after the
    task switching, timing and safety pipeline have been verified in Gazebo.
    """

    OBS_DIM = 46
    ACTION_DIM = 10

    def __init__(self, base_gain=5.0, arm_gain=2.0, recovery_gain=0.4):
        self.base_gain = float(base_gain)
        self.arm_gain = float(arm_gain)
        self.recovery_gain = float(recovery_gain)

    def predict(self, observation, command):
        obs = self._as_vector(observation)
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)

        if command.mode == TaskMode.BASE_APPROACH:
            action[0] = self.base_gain * command.subgoal[0]
            action[1] = self.base_gain * command.subgoal[1]

        elif command.mode == TaskMode.ARM_REACH:
            error = command.subgoal[:3]
            # A conservative diagnostic mapping. The learned low-level policy
            # will replace this mapping; joint order remains fixed.
            action[4] = -self.arm_gain * error[1]
            action[5] = self.arm_gain * error[0]
            action[6] = self.arm_gain * error[2]
            action[7] = -0.5 * self.arm_gain * error[1]
            action[8] = -self.arm_gain * error[2]
            action[9] = 0.5 * self.arm_gain * error[0]

        elif command.mode == TaskMode.RECOVERY:
            joint_positions = obs[11:21]
            action[4:10] = -self.recovery_gain * joint_positions[4:10]

        return np.clip(action, -1.0, 1.0)

    @classmethod
    def _as_vector(cls, observation):
        if isinstance(observation, dict):
            observation = observation["obs_vec"]
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != (cls.OBS_DIM,):
            raise ValueError(
                "observation must have shape (46,), got {}".format(vector.shape)
            )
        return vector

