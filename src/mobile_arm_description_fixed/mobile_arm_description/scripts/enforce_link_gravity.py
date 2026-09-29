#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Enforce Gazebo link gravity after URDF fixed-joint reduction.

Gazebo's URDF converter can overwrite ``turnGravityOff`` while reducing the
fixed virtual-base joints used by the planar model.  This node runs after the
model appears, preserves every inertial property and changes only the runtime
``gravity_mode`` flag for the requested links.
"""

from __future__ import print_function

import sys
import time

import rospy

from gazebo_msgs.srv import GetLinkProperties, SetLinkProperties


DEFAULT_LINKS = [
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
]


def scoped_name(model_name, link_name):
    if "::" in link_name:
        return link_name
    return "{}::{}".format(model_name, link_name)


def wait_for_link(get_properties, link_name, timeout):
    deadline = time.time() + timeout
    last_message = "link not available"
    while not rospy.is_shutdown() and time.time() < deadline:
        try:
            response = get_properties(link_name)
            if response.success:
                return response
            last_message = response.status_message
        except rospy.ServiceException as error:
            last_message = str(error)
        rospy.sleep(0.1)
    raise RuntimeError(
        "{} was not available within {:.1f}s: {}".format(
            link_name,
            timeout,
            last_message,
        )
    )


def disable_gravity(get_properties, set_properties, link_name, timeout):
    current = wait_for_link(get_properties, link_name, timeout)
    response = set_properties(
        link_name=link_name,
        com=current.com,
        gravity_mode=False,
        mass=current.mass,
        ixx=current.ixx,
        ixy=current.ixy,
        ixz=current.ixz,
        iyy=current.iyy,
        iyz=current.iyz,
        izz=current.izz,
    )
    if not response.success:
        raise RuntimeError(
            "failed to update {}: {}".format(
                link_name,
                response.status_message,
            )
        )

    verified = get_properties(link_name)
    if not verified.success or verified.gravity_mode:
        raise RuntimeError(
            "gravity verification failed for {}: {}".format(
                link_name,
                verified.status_message,
            )
        )


def main():
    rospy.init_node("enforce_mobile_arm_link_gravity")
    model_name = str(rospy.get_param("~model_name", "mobile_arm"))
    links = list(rospy.get_param("~links", DEFAULT_LINKS))
    timeout = float(rospy.get_param("~timeout", 30.0))
    if not links:
        rospy.logerr("No links were configured for gravity enforcement")
        return 2

    rospy.wait_for_service("/gazebo/get_link_properties", timeout=timeout)
    rospy.wait_for_service("/gazebo/set_link_properties", timeout=timeout)
    get_properties = rospy.ServiceProxy(
        "/gazebo/get_link_properties",
        GetLinkProperties,
    )
    set_properties = rospy.ServiceProxy(
        "/gazebo/set_link_properties",
        SetLinkProperties,
    )

    try:
        for link in links:
            link_name = scoped_name(model_name, str(link))
            disable_gravity(
                get_properties,
                set_properties,
                link_name,
                timeout,
            )
            rospy.loginfo(
                "Runtime gravity disabled and verified: %s",
                link_name,
            )
    except (RuntimeError, rospy.ROSException) as error:
        rospy.logerr("Runtime gravity enforcement failed: %s", str(error))
        return 2

    rospy.loginfo(
        "Runtime gravity enforcement complete for %d links",
        len(links),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
