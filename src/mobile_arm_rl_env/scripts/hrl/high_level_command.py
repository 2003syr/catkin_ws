#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class TaskMode(object):
    """Stable task identifiers shared by rule-based and future RL policies."""

    BASE_APPROACH = 0
    ARM_REACH = 1
    RECOVERY = 2

    NAMES = {
        BASE_APPROACH: "BASE_APPROACH",
        ARM_REACH: "ARM_REACH",
        RECOVERY: "RECOVERY",
    }
    COUNT = 3

    @classmethod
    def validate(cls, mode):
        if mode not in cls.NAMES:
            raise ValueError("unknown task mode: {}".format(mode))


class HighLevelCommand(object):
    """A task decision plus an executable six-dimensional local goal."""

    SUBGOAL_DIM = 6

    def __init__(self, mode, subgoal, base_priority, arm_priority, reason=""):
        TaskMode.validate(mode)

        subgoal = np.asarray(subgoal, dtype=np.float32)
        if subgoal.shape != (self.SUBGOAL_DIM,):
            raise ValueError(
                "subgoal must have shape (6,), got {}".format(subgoal.shape)
            )

        self.mode = mode
        self.subgoal = subgoal
        self.base_priority = float(np.clip(base_priority, 0.0, 1.0))
        self.arm_priority = float(np.clip(arm_priority, 0.0, 1.0))
        self.reason = str(reason)

    @property
    def mode_name(self):
        return TaskMode.NAMES[self.mode]

    def mode_one_hot(self):
        encoded = np.zeros(TaskMode.COUNT, dtype=np.float32)
        encoded[self.mode] = 1.0
        return encoded

