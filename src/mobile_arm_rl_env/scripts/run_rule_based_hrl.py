#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

from hrl.hierarchical_env import HierarchicalEnv
from hrl.rule_based_high_policy import RuleBasedHighPolicy
from hrl.rule_based_low_policy import RuleBasedLowPolicy
from mobile_arm_env_check import MobileArmReachEnv


def main():
    rospy.init_node("rule_based_mobile_arm_hrl")

    high_interval = rospy.get_param("~high_interval", 10)
    base_env = MobileArmReachEnv(init_ros_node=False)
    env = HierarchicalEnv(
        base_env=base_env,
        high_policy=RuleBasedHighPolicy(
            arm_enter_distance=rospy.get_param("~arm_enter_distance", 0.80),
            arm_exit_distance=rospy.get_param("~arm_exit_distance", 1.00),
            min_joint_margin=rospy.get_param("~min_joint_margin", 0.08),
            max_position_subgoal=rospy.get_param(
                "~max_position_subgoal", 0.10
            ),
        ),
        low_policy=RuleBasedLowPolicy(),
        high_interval=high_interval,
    )

    observation = env.reset()
    rospy.loginfo(
        "Rule-based HRL started: obs_dim=%d action_dim=%d high_interval=%d",
        len(observation["obs_vec"]),
        base_env.action_dim,
        high_interval,
    )

    try:
        while not rospy.is_shutdown():
            observation, reward, done, info = env.step()
            if info["high_updated"]:
                rospy.loginfo(
                    "HRL decision high_step=%d mode=%s reason=%s subgoal=%s",
                    info["high_step"],
                    info["task_mode"],
                    info["task_reason"],
                    info["subgoal"],
                )
            rospy.loginfo_throttle(
                1.0,
                "HRL low_step=%d mode=%s dist=%.4f reward=%.4f",
                info["low_step"],
                info["task_mode"],
                info["dist"],
                reward,
            )
            if done:
                rospy.loginfo("HRL episode complete; resetting counters")
                observation = env.reset()
    finally:
        env.stop()


if __name__ == "__main__":
    main()

