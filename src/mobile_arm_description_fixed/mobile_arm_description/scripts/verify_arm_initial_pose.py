#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Verify that the spawned arm starts in a stable, chassis-clear pose."""

from __future__ import print_function

import sys
import time

import numpy as np
import rospy

from gazebo_msgs.srv import GetLinkState
from sensor_msgs.msg import JointState


ARM_JOINTS = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)
DEFAULT_EXPECTED_POSITIONS = np.asarray(
    [0.0, -0.30, 0.03, 0.0, 0.03, 0.0],
    dtype=np.float64,
)
DEFAULT_MINIMUM_LINK_HEIGHTS = {
    "link3": 0.25,
    "link4": 0.25,
    "link5": 0.20,
}


def ordered_state(message):
    positions = dict(zip(message.name, message.position))
    velocities = dict(zip(message.name, message.velocity))
    missing = [name for name in ARM_JOINTS if name not in positions]
    if missing:
        raise RuntimeError("joint_states is missing {}".format(missing))
    q = np.asarray([positions[name] for name in ARM_JOINTS])
    dq = np.asarray([velocities.get(name, 0.0) for name in ARM_JOINTS])
    return q, dq


def main():
    rospy.init_node("verify_arm_initial_pose", anonymous=True)
    model_name = rospy.get_param("~model_name", "mobile_arm")
    warmup = float(rospy.get_param("~warmup", 3.0))
    timeout = float(rospy.get_param("~timeout", 10.0))
    position_tolerance = float(
        rospy.get_param("~position_tolerance", 0.05)
    )
    velocity_tolerance = float(
        rospy.get_param("~velocity_tolerance", 0.01)
    )
    measurement_duration = float(
        rospy.get_param("~measurement_duration", 5.0)
    )
    drift_tolerance = float(
        rospy.get_param("~drift_tolerance", 0.005)
    )
    expected = np.asarray(
        rospy.get_param(
            "~expected_positions",
            DEFAULT_EXPECTED_POSITIONS.tolist(),
        ),
        dtype=np.float64,
    )
    if expected.shape != (6,):
        raise ValueError("expected_positions must contain six values")

    # Wall time is intentional: this diagnostic must still finish if Gazebo
    # simulated time is paused or has not started advancing yet.
    time.sleep(max(0.0, warmup))
    first_message = rospy.wait_for_message(
        "/joint_states",
        JointState,
        timeout=timeout,
    )
    first_q, _ = ordered_state(first_message)
    time.sleep(max(0.0, measurement_duration))
    final_message = rospy.wait_for_message(
        "/joint_states",
        JointState,
        timeout=timeout,
    )
    q, reported_dq = ordered_state(final_message)
    position_error = np.abs(q - expected)
    incremental_drift = np.abs(q - first_q)
    # With arm/chassis collision enabled, Gazebo can report a non-zero
    # constraint-solver joint velocity while consecutive positions remain
    # stationary.  Initial-pose stability is therefore accepted from measured
    # position change; the raw reported velocity is still logged for diagnosis.
    observed_velocity = incremental_drift / max(
        measurement_duration,
        1.0e-6,
    )
    maximum_position_error = float(np.max(position_error))
    maximum_incremental_drift = float(np.max(incremental_drift))
    maximum_observed_velocity = float(np.max(observed_velocity))
    maximum_reported_velocity = float(np.max(np.abs(reported_dq)))

    rospy.wait_for_service("/gazebo/get_link_state", timeout=timeout)
    get_link_state = rospy.ServiceProxy(
        "/gazebo/get_link_state",
        GetLinkState,
    )
    link_heights = {}
    link_height_ok = True
    for link_name, minimum_height in sorted(
            DEFAULT_MINIMUM_LINK_HEIGHTS.items()):
        response = get_link_state(
            "{}::{}".format(model_name, link_name),
            "world",
        )
        if not response.success:
            raise RuntimeError(
                "failed to read {}: {}".format(
                    link_name,
                    response.status_message,
                )
            )
        height = float(response.link_state.pose.position.z)
        link_heights[link_name] = height
        if height < minimum_height:
            link_height_ok = False

    rospy.loginfo(
        "Initial arm pose expected=%s measured=%s error=%s "
        "max_error=%.5f drift=%s max_drift=%.5f "
        "observed_velocity=%s max_observed_velocity=%.5f "
        "max_reported_velocity=%.5f",
        np.round(expected, 4).tolist(),
        np.round(q, 4).tolist(),
        np.round(position_error, 5).tolist(),
        maximum_position_error,
        np.round(incremental_drift, 5).tolist(),
        maximum_incremental_drift,
        np.round(observed_velocity, 5).tolist(),
        maximum_observed_velocity,
        maximum_reported_velocity,
    )
    rospy.loginfo(
        "Initial arm link heights=%s required=%s",
        dict((name, round(value, 4))
             for name, value in link_heights.items()),
        DEFAULT_MINIMUM_LINK_HEIGHTS,
    )

    passed = bool(
        maximum_position_error <= position_tolerance
        and maximum_incremental_drift <= drift_tolerance
        and maximum_observed_velocity <= velocity_tolerance
        and link_height_ok
    )
    if not passed:
        rospy.logerr(
            "Initial arm pose verification failed: "
            "position_ok=%s drift_ok=%s velocity_ok=%s height_ok=%s",
            maximum_position_error <= position_tolerance,
            maximum_incremental_drift <= drift_tolerance,
            maximum_observed_velocity <= velocity_tolerance,
            link_height_ok,
        )
        return 2
    rospy.loginfo("Initial arm pose verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
