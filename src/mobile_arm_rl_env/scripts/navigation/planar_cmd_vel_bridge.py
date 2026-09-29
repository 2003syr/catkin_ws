#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert ROS Navigation cmd_vel commands to the virtual planar joints."""

from __future__ import print_function

import math
import threading
import time

import numpy as np
import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


TRACKED_FORWARD_YAW_OFFSET = -0.5 * math.pi
PLANAR_JOINTS = ("x", "y", "z")


class PlanarCmdVelBridge(object):
    """Enforce differential-drive motion at the controller boundary."""

    def __init__(self):
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.joint_state_topic = rospy.get_param(
            "~joint_state_topic", "/joint_states"
        )
        self.command_timeout = float(
            rospy.get_param("~command_timeout", 0.30)
        )
        self.publish_rate = float(rospy.get_param("~publish_rate", 30.0))
        self.max_linear_speed = float(
            rospy.get_param("~max_linear_speed", 0.05)
        )
        self.max_yaw_rate = float(
            rospy.get_param("~max_yaw_rate", 0.125)
        )
        self.lateral_tolerance = float(
            rospy.get_param("~lateral_tolerance", 1.0e-4)
        )
        self.position_soft_margin = float(
            rospy.get_param("~position_soft_margin", 0.10)
        )
        if (
                self.command_timeout <= 0.0
                or self.publish_rate <= 0.0
                or self.max_linear_speed <= 0.0
                or self.max_yaw_rate <= 0.0
                or self.position_soft_margin <= 0.0):
            raise ValueError("bridge speed, timeout and rate parameters must be positive")

        self.position_limits = {
            "x": tuple(rospy.get_param("~x_limits", [-2.0, 2.0])),
            "y": tuple(rospy.get_param("~y_limits", [-2.0, 2.0])),
        }
        for name, bounds in self.position_limits.items():
            if len(bounds) != 2 or float(bounds[0]) >= float(bounds[1]):
                raise ValueError("{} limits are invalid".format(name))
            self.position_limits[name] = (
                float(bounds[0]),
                float(bounds[1]),
            )

        self._lock = threading.Lock()
        self._positions = {}
        self._linear = 0.0
        self._yaw_rate = 0.0
        self._last_command_wall_time = None
        self._timed_out = True

        self.publishers = dict(
            (
                name,
                rospy.Publisher(
                    "/{}_velocity_controller/command".format(name),
                    Float64,
                    queue_size=1,
                ),
            )
            for name in PLANAR_JOINTS
        )
        rospy.Subscriber(
            self.joint_state_topic,
            JointState,
            self._joint_state_callback,
            queue_size=50,
        )
        rospy.Subscriber(
            self.cmd_vel_topic,
            Twist,
            self._cmd_vel_callback,
            queue_size=10,
        )
        rospy.on_shutdown(self.stop)

    def _joint_state_callback(self, message):
        positions = dict(zip(message.name, message.position))
        with self._lock:
            for name in PLANAR_JOINTS:
                if name in positions:
                    self._positions[name] = float(positions[name])

    def _cmd_vel_callback(self, message):
        values = (
            float(message.linear.x),
            float(message.linear.y),
            float(message.angular.z),
        )
        if not np.all(np.isfinite(values)):
            rospy.logerr_throttle(
                1.0,
                "Rejected non-finite cmd_vel: %s",
                str(values),
            )
            return
        if abs(values[1]) > self.lateral_tolerance:
            rospy.logwarn_throttle(
                1.0,
                "Tracked base rejected lateral cmd_vel linear.y=%.5f",
                values[1],
            )
        with self._lock:
            self._linear = float(np.clip(
                values[0],
                -self.max_linear_speed,
                self.max_linear_speed,
            ))
            self._yaw_rate = float(np.clip(
                values[2],
                -self.max_yaw_rate,
                self.max_yaw_rate,
            ))
            self._last_command_wall_time = time.time()
            self._timed_out = False

    def run(self):
        rospy.loginfo(
            "Planar cmd_vel bridge started: topic=%s max_v=%.3f "
            "max_w=%.3f timeout=%.2f",
            self.cmd_vel_topic,
            self.max_linear_speed,
            self.max_yaw_rate,
            self.command_timeout,
        )
        period = 1.0 / self.publish_rate
        while not rospy.is_shutdown():
            commands = self._commands_at(time.time())
            self._publish(commands)
            time.sleep(period)

    def _commands_at(self, wall_time):
        with self._lock:
            positions = dict(self._positions)
            command_time = self._last_command_wall_time
            linear = self._linear
            yaw_rate = self._yaw_rate
            timed_out = self._timed_out

        if not all(name in positions for name in PLANAR_JOINTS):
            return dict((name, 0.0) for name in PLANAR_JOINTS)
        if (
                command_time is None
                or wall_time - command_time > self.command_timeout):
            if not timed_out:
                rospy.logwarn(
                    "Planar cmd_vel timeout: stopping x/y/z controllers"
                )
                with self._lock:
                    self._timed_out = True
            linear = 0.0
            yaw_rate = 0.0

        heading = positions["z"] + TRACKED_FORWARD_YAW_OFFSET
        commands = {
            "x": linear * math.cos(heading),
            "y": linear * math.sin(heading),
            "z": yaw_rate,
        }
        translation_scale = self._translation_limit_scale(
            positions,
            commands,
        )
        commands["x"] *= translation_scale
        commands["y"] *= translation_scale
        if translation_scale < 1.0:
            rospy.logwarn_throttle(
                1.0,
                "Planar position safety scaled translation to %.3f",
                translation_scale,
            )
        return commands

    def _translation_limit_scale(self, positions, commands):
        scale = 1.0
        for name in ("x", "y"):
            lower, upper = self.position_limits[name]
            position = positions[name]
            velocity = commands[name]
            if velocity > 0.0:
                distance = upper - position
            elif velocity < 0.0:
                distance = position - lower
            else:
                continue
            scale = min(
                scale,
                float(np.clip(
                    distance / self.position_soft_margin,
                    0.0,
                    1.0,
                )),
            )
        return scale

    def _publish(self, commands):
        for name in PLANAR_JOINTS:
            self.publishers[name].publish(Float64(float(commands[name])))

    def stop(self):
        zero = dict((name, 0.0) for name in PLANAR_JOINTS)
        for _ in range(3):
            self._publish(zero)
            time.sleep(0.03)


def main():
    rospy.init_node("planar_cmd_vel_bridge")
    PlanarCmdVelBridge().run()


if __name__ == "__main__":
    main()
