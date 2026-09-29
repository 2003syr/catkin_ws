#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Publish standard planar odometry from the virtual x/y/yaw joint chain."""

from __future__ import print_function

import math
import threading

import numpy as np
import rospy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_from_euler


TRACKED_FORWARD_YAW_OFFSET = -0.5 * math.pi
PLANAR_JOINTS = ("x", "y", "z")


class PlanarBaseOdometry(object):
    def __init__(self):
        self.joint_state_topic = rospy.get_param(
            "~joint_state_topic", "/joint_states"
        )
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.odom_frame = rospy.get_param("~odom_frame", "base_link")
        self.base_frame = rospy.get_param(
            "~base_frame", "base_footprint"
        )
        self.velocity_filter_alpha = float(
            rospy.get_param("~velocity_filter_alpha", 0.35)
        )
        if not 0.0 < self.velocity_filter_alpha <= 1.0:
            raise ValueError("velocity_filter_alpha must be in (0, 1]")

        self._lock = threading.Lock()
        self._previous_position = None
        self._previous_stamp = None
        self._filtered_twist = np.zeros(3, dtype=np.float64)
        self.publisher = rospy.Publisher(
            self.odom_topic,
            Odometry,
            queue_size=20,
        )
        rospy.Subscriber(
            self.joint_state_topic,
            JointState,
            self._joint_state_callback,
            queue_size=100,
        )

    def _joint_state_callback(self, message):
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in PLANAR_JOINTS):
            return
        current = np.asarray(
            [positions[name] for name in PLANAR_JOINTS],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(current)):
            rospy.logerr_throttle(
                1.0,
                "Cannot publish odometry from non-finite joint positions",
            )
            return
        stamp = message.header.stamp
        if stamp == rospy.Time():
            stamp = rospy.Time.now()

        with self._lock:
            body_twist = self._estimate_twist(current, stamp)
            odometry = self._odometry_message(
                current,
                body_twist,
                stamp,
            )
        self.publisher.publish(odometry)

    def _estimate_twist(self, current, stamp):
        if self._previous_position is None:
            self._previous_position = current.copy()
            self._previous_stamp = stamp
            return self._filtered_twist.copy()
        dt = (stamp - self._previous_stamp).to_sec()
        if dt <= 1.0e-6 or dt > 1.0:
            self._previous_position = current.copy()
            self._previous_stamp = stamp
            self._filtered_twist[:] = 0.0
            return self._filtered_twist.copy()

        yaw_delta = math.atan2(
            math.sin(current[2] - self._previous_position[2]),
            math.cos(current[2] - self._previous_position[2]),
        )
        world_velocity = np.asarray([
            (current[0] - self._previous_position[0]) / dt,
            (current[1] - self._previous_position[1]) / dt,
            yaw_delta / dt,
        ], dtype=np.float64)
        midpoint_heading = (
            self._previous_position[2]
            + 0.5 * yaw_delta
            + TRACKED_FORWARD_YAW_OFFSET
        )
        cosine = math.cos(midpoint_heading)
        sine = math.sin(midpoint_heading)
        raw_body_twist = np.asarray([
            cosine * world_velocity[0] + sine * world_velocity[1],
            -sine * world_velocity[0] + cosine * world_velocity[1],
            world_velocity[2],
        ], dtype=np.float64)
        alpha = self.velocity_filter_alpha
        self._filtered_twist = (
            alpha * raw_body_twist
            + (1.0 - alpha) * self._filtered_twist
        )
        self._previous_position = current.copy()
        self._previous_stamp = stamp
        return self._filtered_twist.copy()

    def _odometry_message(self, position, body_twist, stamp):
        heading = position[2] + TRACKED_FORWARD_YAW_OFFSET
        quaternion = quaternion_from_euler(0.0, 0.0, heading)
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = self.odom_frame
        message.child_frame_id = self.base_frame
        message.pose.pose.position.x = float(position[0])
        message.pose.pose.position.y = float(position[1])
        message.pose.pose.orientation.x = quaternion[0]
        message.pose.pose.orientation.y = quaternion[1]
        message.pose.pose.orientation.z = quaternion[2]
        message.pose.pose.orientation.w = quaternion[3]
        message.twist.twist.linear.x = float(body_twist[0])
        message.twist.twist.linear.y = float(body_twist[1])
        message.twist.twist.angular.z = float(body_twist[2])
        message.pose.covariance = [
            1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0e-4, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0e6, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0e6, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0e6, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-3,
        ]
        message.twist.covariance = list(message.pose.covariance)
        return message


def main():
    rospy.init_node("planar_base_odom")
    PlanarBaseOdometry()
    rospy.loginfo(
        "Planar odometry publisher started: /joint_states -> /odom"
    )
    rospy.spin()


if __name__ == "__main__":
    main()
