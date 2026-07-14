#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hrl.hierarchical_env import HierarchicalEnv
from hrl.high_level_command import TaskMode
from hrl.rule_based_high_policy import RuleBasedHighPolicy
from hrl.rule_based_low_policy import RuleBasedLowPolicy


def observation(distance=1.2, minimum_margin=1.0):
    vector = np.zeros(46, dtype=np.float32)
    vector[3] = distance
    vector[10] = distance
    vector[31:41] = 1.0
    vector[31] = minimum_margin
    return {"obs_vec": vector}


class FakeBaseEnv(object):
    def __init__(self):
        self.actions = []
        self.stopped = False

    def reset(self):
        return observation()

    def step(self, action):
        self.actions.append(action)
        return observation(), 0.0, False, {"dist": 1.2}

    def stop(self):
        self.stopped = True


class HRLFrameworkTest(unittest.TestCase):
    def test_high_policy_switches_tasks_and_recovers(self):
        policy = RuleBasedHighPolicy()
        self.assertEqual(policy.predict(observation(1.2)).mode, TaskMode.BASE_APPROACH)
        self.assertEqual(policy.predict(observation(0.7)).mode, TaskMode.ARM_REACH)
        self.assertEqual(
            policy.predict(observation(0.7, minimum_margin=0.01)).mode,
            TaskMode.RECOVERY,
        )

    def test_scheduler_updates_high_level_at_interval(self):
        base_env = FakeBaseEnv()
        env = HierarchicalEnv(
            base_env,
            RuleBasedHighPolicy(),
            RuleBasedLowPolicy(),
            high_interval=3,
        )
        env.reset()
        updates = [env.step()[3]["high_updated"] for _ in range(5)]
        self.assertEqual(updates, [True, False, False, True, False])
        self.assertTrue(all(action.shape == (10,) for action in base_env.actions))


if __name__ == "__main__":
    unittest.main()

