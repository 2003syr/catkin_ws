#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fail fast when an expected arm controller is not running."""

from __future__ import print_function

import sys
import time

import rospy
from controller_manager_msgs.srv import ListControllers


DEFAULT_CONTROLLERS = [
    "joint_state_controller",
    "joint1_velocity_controller",
    "joint2_velocity_controller",
    "joint3_velocity_controller",
    "joint4_velocity_controller",
    "joint5_velocity_controller",
    "joint6_velocity_controller",
]


def get_states(service):
    response = service()
    return dict((controller.name, controller.state) for controller in response.controller)


def main():
    rospy.init_node("verify_mobile_arm_controllers")
    expected = rospy.get_param("~expected_controllers", DEFAULT_CONTROLLERS)
    timeout = float(rospy.get_param("~timeout", 30.0))
    check_period = float(rospy.get_param("~check_period", 1.0))

    try:
        rospy.wait_for_service("/controller_manager/list_controllers", timeout=timeout)
    except rospy.ROSException as error:
        rospy.logfatal("Controller manager unavailable: %s", error)
        return 2

    service = rospy.ServiceProxy("/controller_manager/list_controllers", ListControllers)
    deadline = time.time() + timeout
    states = {}
    while not rospy.is_shutdown() and time.time() < deadline:
        try:
            states = get_states(service)
        except rospy.ServiceException as error:
            rospy.logwarn("Controller state query failed: %s", error)
            time.sleep(0.25)
            continue
        if all(states.get(name) == "running" for name in expected):
            break
        time.sleep(0.25)
    else:
        missing = [
            "{}={}".format(name, states.get(name, "missing"))
            for name in expected if states.get(name) != "running"
        ]
        rospy.logfatal("Controllers failed to reach running state: %s", missing)
        return 3

    for name in expected:
        rospy.loginfo("Controller verified: %s: running", name)
    rospy.loginfo("All %d required controllers are running", len(expected))

    while not rospy.is_shutdown():
        time.sleep(max(check_period, 0.1))
        try:
            states = get_states(service)
        except rospy.ServiceException as error:
            rospy.logerr("Controller state query failed after startup: %s", error)
            continue
        stopped = [name for name in expected if states.get(name) != "running"]
        if stopped:
            rospy.logfatal("Required controllers stopped: %s", stopped)
            return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
