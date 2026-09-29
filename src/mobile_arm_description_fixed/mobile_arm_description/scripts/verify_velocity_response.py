#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Exercise each arm velocity controller with small +/- commands and zero."""

from __future__ import print_function

import sys
import threading
import time

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
COMMANDS = {
    "joint1": 0.02,
    "joint2": 0.02,
    "joint3": 0.005,
    "joint4": 0.02,
    "joint5": 0.005,
    "joint6": 0.02,
}


class VelocityMonitor(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.velocity = {}
        self.received = threading.Event()

    def callback(self, message):
        with self.lock:
            self.velocity.update(dict(zip(message.name, message.velocity)))
        self.received.set()

    def get(self, name):
        with self.lock:
            return self.velocity.get(name)


def run_phase(publishers, monitor, active_joint, command, duration=0.6):
    start = time.time()
    samples = []
    while not rospy.is_shutdown() and time.time() - start < duration:
        for name, publisher in publishers.items():
            publisher.publish(Float64(command if name == active_joint else 0.0))
        elapsed = time.time() - start
        measured = monitor.get(active_joint)
        if elapsed > 0.2 and measured is not None:
            samples.append(float(measured))
        time.sleep(0.02)
    if not samples:
        raise RuntimeError("no velocity samples for {}".format(active_joint))
    return sum(samples) / len(samples), max(abs(value) for value in samples)


def main():
    rospy.init_node("verify_velocity_response")
    monitor = VelocityMonitor()
    rospy.Subscriber("/joint_states", JointState, monitor.callback, queue_size=100)
    publishers = dict(
        (name, rospy.Publisher(
            "/{}_velocity_controller/command".format(name),
            Float64,
            queue_size=1,
        ))
        for name in ARM_JOINTS
    )

    if not monitor.received.wait(10.0):
        rospy.logerr("No /joint_states received")
        return 2
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if all(publisher.get_num_connections() > 0 for publisher in publishers.values()):
            break
        time.sleep(0.1)
    else:
        rospy.logerr("Not all velocity controller subscribers are connected")
        return 3

    failures = []
    try:
        for name in ARM_JOINTS:
            magnitude = COMMANDS[name]
            positive_mean, _ = run_phase(
                publishers, monitor, name, magnitude
            )
            _, zero_after_positive = run_phase(
                publishers, monitor, name, 0.0
            )
            negative_mean, _ = run_phase(
                publishers, monitor, name, -magnitude
            )
            _, zero_after_negative = run_phase(
                publishers, monitor, name, 0.0
            )
            direction_ok = (
                positive_mean > 0.25 * magnitude
                and negative_mean < -0.25 * magnitude
            )
            zero_max = max(zero_after_positive, zero_after_negative)
            zero_ok = zero_max < 0.002
            passed = direction_ok and zero_ok
            rospy.loginfo(
                "%s +mean=%.6f -mean=%.6f zero_max=%.6f pass=%s",
                name, positive_mean, negative_mean, zero_max, passed,
            )
            if not passed:
                failures.append(name)
    finally:
        for _ in range(5):
            for publisher in publishers.values():
                publisher.publish(Float64(0.0))
            time.sleep(0.03)

    if failures:
        rospy.logerr("Velocity response verification failed: %s", failures)
        return 4
    rospy.loginfo("All arm velocity response checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
