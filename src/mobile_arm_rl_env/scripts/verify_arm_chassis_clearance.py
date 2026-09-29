#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Read-only verification of the predictive arm/chassis safety guard."""

from __future__ import print_function

import sys

import numpy as np
import rospy

from sensor_msgs.msg import JointState

from hrl.arm_chassis_guard import ARM_JOINTS, ArmChassisInterferenceGuard


RESET_POSE = np.asarray(
    [0.0, -0.30, 0.03, 0.0, 0.03, 0.0],
    dtype=np.float64,
)
NAVIGATION_POSE = np.asarray(
    [0.0, -0.30, 0.03, 0.0, 0.03, 0.0],
    dtype=np.float64,
)


def ordered_arm_positions(message):
    positions = dict(zip(message.name, message.position))
    missing = [name for name in ARM_JOINTS if name not in positions]
    if missing:
        raise RuntimeError(
            "joint_states is missing {}".format(missing)
        )
    return np.asarray(
        [positions[name] for name in ARM_JOINTS],
        dtype=np.float64,
    )


def describe_configuration(guard, label, positions):
    clearance, link_name = guard.minimum_clearance(positions)
    rospy.loginfo(
        "Arm chassis clearance label=%s clearance=%.4f closest=%s q=%s",
        label,
        clearance,
        link_name,
        np.round(positions, 4).tolist(),
    )
    return clearance


def main():
    rospy.init_node("verify_arm_chassis_clearance", anonymous=True)
    timeout = float(rospy.get_param("~timeout", 10.0))
    guard = ArmChassisInterferenceGuard(
        hard_clearance=rospy.get_param("~hard_clearance", 0.010),
        soft_clearance=rospy.get_param("~soft_clearance", 0.050),
        prediction_horizon=rospy.get_param(
            "~prediction_horizon",
            0.25,
        ),
        robot_description="/robot_description",
    )

    message = rospy.wait_for_message(
        "/joint_states",
        JointState,
        timeout=timeout,
    )
    current = ordered_arm_positions(message)
    current_clearance = describe_configuration(
        guard,
        "current",
        current,
    )
    describe_configuration(guard, "reach_reset", RESET_POSE)
    describe_configuration(guard, "navigation", NAVIGATION_POSE)
    zero_clearance = describe_configuration(
        guard,
        "known_unsafe_zero",
        np.zeros(6, dtype=np.float64),
    )

    constrained_directions = []
    test_speed = np.asarray(
        [0.5, 0.5, 0.03, 0.5, 0.03, 0.5],
        dtype=np.float64,
    )
    for index, joint_name in enumerate(ARM_JOINTS):
        for sign, suffix in ((1.0, "+"), (-1.0, "-")):
            velocity = np.zeros(6, dtype=np.float64)
            velocity[index] = sign * test_speed[index]
            _, info = guard.filter_velocity(current, velocity)
            if info["reason"] != "safe":
                constrained_directions.append(
                    "{}{}:{}:{:.3f}".format(
                        joint_name,
                        suffix,
                        info["reason"],
                        info["scale"],
                    )
                )

    rospy.loginfo(
        "Arm chassis guard hard=%.3f soft=%.3f constrained=%s",
        guard.hard_clearance,
        guard.soft_clearance,
        constrained_directions,
    )
    passed = bool(
        current_clearance >= guard.hard_clearance
        and zero_clearance < guard.hard_clearance
    )
    if not passed:
        rospy.logerr("Arm chassis clearance verification failed")
        return 2
    rospy.loginfo("Arm chassis clearance verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
