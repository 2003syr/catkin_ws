#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Verify the internal controllers through tracked-base [v, omega] commands."""

from __future__ import print_function

import math
import sys
import threading
import time

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


PLANAR_JOINTS = ("x", "y", "z")
LINEAR_COMMAND = 0.03
YAW_COMMAND = 0.10
LINEAR_VELOCITY_PEAK_LIMIT = 0.10
YAW_VELOCITY_PEAK_LIMIT = 0.30
LINEAR_TRACKING_ERROR_LIMIT = 0.02
YAW_TRACKING_ERROR_LIMIT = 0.075
LINEAR_EFFORT_PEAK_LIMIT = 18.0
YAW_EFFORT_PEAK_LIMIT = 9.0
# The CAD chassis longitudinal axis is mounted at -90 degrees relative to
# the internal z-link yaw coordinate.
TRACKED_FORWARD_YAW_OFFSET = -0.5 * math.pi


class JointMonitor(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.position = {}
        self.velocity = {}
        self.effort = {}
        self.received = threading.Event()

    def callback(self, message):
        with self.lock:
            self.position.update(dict(zip(
                message.name,
                message.position,
            )))
            self.velocity.update(dict(zip(
                message.name,
                message.velocity,
            )))
            self.effort.update(dict(zip(
                message.name,
                message.effort,
            )))
        self.received.set()

    def snapshot(self):
        with self.lock:
            missing = [
                name for name in PLANAR_JOINTS
                if (
                    name not in self.position
                    or name not in self.velocity
                    or name not in self.effort
                )
            ]
            if missing:
                raise RuntimeError(
                    "missing planar joint states: {}".format(missing)
                )
            return (
                dict(self.position),
                dict(self.velocity),
                dict(self.effort),
            )


def angle_difference(end, start):
    return math.atan2(
        math.sin(float(end) - float(start)),
        math.cos(float(end) - float(start)),
    )


def publish_commands(publishers, monitor, linear, yaw_rate):
    position, unused_velocity, unused_effort = monitor.snapshot()
    yaw = float(position["z"])
    heading = yaw + TRACKED_FORWARD_YAW_OFFSET
    commands = {
        "x": float(linear) * math.cos(heading),
        "y": float(linear) * math.sin(heading),
        "z": float(yaw_rate),
    }
    for name, publisher in publishers.items():
        publisher.publish(Float64(commands[name]))


def run_phase(
        publishers,
        monitor,
        linear,
        yaw_rate,
        duration=1.0):
    start_position, unused_velocity, unused_effort = monitor.snapshot()
    start_time = time.time()
    forward_samples = []
    lateral_samples = []
    yaw_samples = []
    effort_samples = []
    while (
            not rospy.is_shutdown()
            and time.time() - start_time < duration):
        publish_commands(
            publishers,
            monitor,
            linear,
            yaw_rate,
        )
        if time.time() - start_time > 0.25:
            position, velocity, effort = monitor.snapshot()
            yaw = float(position["z"])
            heading = yaw + TRACKED_FORWARD_YAW_OFFSET
            forward_velocity = (
                math.cos(heading) * float(velocity["x"])
                + math.sin(heading) * float(velocity["y"])
            )
            lateral_velocity = (
                -math.sin(heading) * float(velocity["x"])
                + math.cos(heading) * float(velocity["y"])
            )
            forward_samples.append(forward_velocity)
            lateral_samples.append(lateral_velocity)
            yaw_samples.append(float(velocity["z"]))
            effort_samples.append(
                max(abs(float(effort[name])) for name in PLANAR_JOINTS)
            )
        time.sleep(0.01)
    end_position, unused_velocity, unused_effort = monitor.snapshot()
    if not forward_samples:
        raise RuntimeError("no planar velocity samples")
    dx = float(end_position["x"] - start_position["x"])
    dy = float(end_position["y"] - start_position["y"])
    start_heading = (
        float(start_position["z"]) + TRACKED_FORWARD_YAW_OFFSET
    )
    forward_delta = (
        math.cos(start_heading) * dx + math.sin(start_heading) * dy
    )
    lateral_delta = (
        -math.sin(start_heading) * dx + math.cos(start_heading) * dy
    )
    yaw_delta = angle_difference(end_position["z"], start_position["z"])
    return {
        "forward_mean": sum(forward_samples) / len(forward_samples),
        "forward_peak": max(abs(value) for value in forward_samples),
        "forward_error_mean": (
            sum(abs(value - float(linear)) for value in forward_samples)
            / len(forward_samples)
        ),
        "lateral_max": max(abs(value) for value in lateral_samples),
        "yaw_mean": sum(yaw_samples) / len(yaw_samples),
        "yaw_peak": max(abs(value) for value in yaw_samples),
        "yaw_error_mean": (
            sum(abs(value - float(yaw_rate)) for value in yaw_samples)
            / len(yaw_samples)
        ),
        "effort_peak": max(effort_samples),
        "forward_delta": forward_delta,
        "lateral_delta": lateral_delta,
        "yaw_delta": yaw_delta,
    }


def stop_all(publishers, monitor, duration=0.5):
    start = time.time()
    maximum = 0.0
    while not rospy.is_shutdown() and time.time() - start < duration:
        publish_commands(publishers, monitor, 0.0, 0.0)
        if time.time() - start > 0.25:
            unused_position, velocity, unused_effort = monitor.snapshot()
            maximum = max(
                maximum,
                max(abs(float(velocity[name])) for name in PLANAR_JOINTS),
            )
        time.sleep(0.02)
    return maximum


def main():
    rospy.init_node("verify_planar_base_response")
    monitor = JointMonitor()
    rospy.Subscriber(
        "/joint_states",
        JointState,
        monitor.callback,
        queue_size=100,
    )
    publishers = dict(
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
    if not monitor.received.wait(10.0):
        rospy.logerr("No /joint_states received")
        return 2
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if all(
                publisher.get_num_connections() > 0
                for publisher in publishers.values()):
            break
        time.sleep(0.1)
    else:
        rospy.logerr("x/y/z controller subscribers are not connected")
        return 3

    try:
        linear_positive = run_phase(
            publishers, monitor, LINEAR_COMMAND, 0.0
        )
        stop_after_linear_positive = stop_all(publishers, monitor)
        linear_negative = run_phase(
            publishers, monitor, -LINEAR_COMMAND, 0.0
        )
        stop_after_linear_negative = stop_all(publishers, monitor)
        yaw_positive = run_phase(
            publishers, monitor, 0.0, YAW_COMMAND
        )
        stop_after_yaw_positive = stop_all(publishers, monitor)
        yaw_negative = run_phase(
            publishers, monitor, 0.0, -YAW_COMMAND
        )
        stop_after_yaw_negative = stop_all(publishers, monitor)
    finally:
        stop_all(publishers, monitor, duration=0.2)

    linear_ok = bool(
        linear_positive["forward_mean"] > 0.25 * LINEAR_COMMAND
        and linear_positive["forward_delta"] > 0.0
        and linear_negative["forward_mean"] < -0.25 * LINEAR_COMMAND
        and linear_negative["forward_delta"] < 0.0
        and linear_positive["lateral_max"] < 0.01
        and linear_negative["lateral_max"] < 0.01
        and abs(linear_positive["lateral_delta"]) < 0.01
        and abs(linear_negative["lateral_delta"]) < 0.01
        and linear_positive["forward_peak"] < LINEAR_VELOCITY_PEAK_LIMIT
        and linear_negative["forward_peak"] < LINEAR_VELOCITY_PEAK_LIMIT
        and (
            linear_positive["forward_error_mean"]
            < LINEAR_TRACKING_ERROR_LIMIT
        )
        and (
            linear_negative["forward_error_mean"]
            < LINEAR_TRACKING_ERROR_LIMIT
        )
        and linear_positive["yaw_peak"] < 0.05
        and linear_negative["yaw_peak"] < 0.05
        and linear_positive["effort_peak"] < LINEAR_EFFORT_PEAK_LIMIT
        and linear_negative["effort_peak"] < LINEAR_EFFORT_PEAK_LIMIT
    )
    yaw_ok = bool(
        yaw_positive["yaw_mean"] > 0.25 * YAW_COMMAND
        and yaw_positive["yaw_delta"] > 0.0
        and yaw_negative["yaw_mean"] < -0.25 * YAW_COMMAND
        and yaw_negative["yaw_delta"] < 0.0
        and abs(yaw_positive["forward_delta"]) < 0.01
        and abs(yaw_negative["forward_delta"]) < 0.01
        and yaw_positive["yaw_peak"] < YAW_VELOCITY_PEAK_LIMIT
        and yaw_negative["yaw_peak"] < YAW_VELOCITY_PEAK_LIMIT
        and yaw_positive["yaw_error_mean"] < YAW_TRACKING_ERROR_LIMIT
        and yaw_negative["yaw_error_mean"] < YAW_TRACKING_ERROR_LIMIT
        and yaw_positive["lateral_max"] < 0.01
        and yaw_negative["lateral_max"] < 0.01
        and yaw_positive["effort_peak"] < YAW_EFFORT_PEAK_LIMIT
        and yaw_negative["effort_peak"] < YAW_EFFORT_PEAK_LIMIT
    )
    zero_max = max(
        stop_after_linear_positive,
        stop_after_linear_negative,
        stop_after_yaw_positive,
        stop_after_yaw_negative,
    )
    zero_ok = bool(zero_max < 0.01)
    rospy.loginfo(
        "tracked linear +mean=%.5f +peak=%.5f +error=%.5f "
        "+effort=%.3f +delta=%.5f +lateral=%.5f "
        "-mean=%.5f -peak=%.5f -error=%.5f "
        "-effort=%.3f -delta=%.5f -lateral=%.5f pass=%s",
        linear_positive["forward_mean"],
        linear_positive["forward_peak"],
        linear_positive["forward_error_mean"],
        linear_positive["effort_peak"],
        linear_positive["forward_delta"],
        linear_positive["lateral_max"],
        linear_negative["forward_mean"],
        linear_negative["forward_peak"],
        linear_negative["forward_error_mean"],
        linear_negative["effort_peak"],
        linear_negative["forward_delta"],
        linear_negative["lateral_max"],
        linear_ok,
    )
    rospy.loginfo(
        "tracked yaw +mean=%.5f +peak=%.5f +error=%.5f "
        "+effort=%.3f +delta=%.5f "
        "-mean=%.5f -peak=%.5f -error=%.5f "
        "-effort=%.3f -delta=%.5f pass=%s "
        "zero_linear=(%.5f,%.5f) zero_yaw=(%.5f,%.5f) "
        "zero_max=%.5f",
        yaw_positive["yaw_mean"],
        yaw_positive["yaw_peak"],
        yaw_positive["yaw_error_mean"],
        yaw_positive["effort_peak"],
        yaw_positive["yaw_delta"],
        yaw_negative["yaw_mean"],
        yaw_negative["yaw_peak"],
        yaw_negative["yaw_error_mean"],
        yaw_negative["effort_peak"],
        yaw_negative["yaw_delta"],
        yaw_ok,
        stop_after_linear_positive,
        stop_after_linear_negative,
        stop_after_yaw_positive,
        stop_after_yaw_negative,
        zero_max,
    )
    if not (linear_ok and yaw_ok and zero_ok):
        rospy.logerr(
            "Tracked-base response verification failed: "
            "linear_ok=%s yaw_ok=%s zero_ok=%s",
            linear_ok,
            yaw_ok,
            zero_ok,
        )
        return 4
    rospy.loginfo("Tracked-base response verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
