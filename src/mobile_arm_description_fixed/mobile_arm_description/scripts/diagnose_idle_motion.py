#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measure idle joint motion without starting any RL/HRL node."""

from __future__ import print_function

import argparse
import csv
import datetime
import os
import sys
import threading
import time

import rospy
from controller_manager_msgs.srv import ListControllers
from gazebo_msgs.srv import GetLinkProperties
from sensor_msgs.msg import JointState


JOINT_ORDER = [
    "x", "y", "z", "sway",
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
]
ARM_JOINTS = set(["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])
PRISMATIC_JOINTS = set(["x", "y", "joint3", "joint5"])
LINKS = ["sway_link", "link1", "link2", "link3", "link4", "link5", "link6"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--label", default="idle")
    parser.add_argument(
        "--output-dir",
        default="/tmp/mobile_arm_idle_diagnostics",
    )
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


class JointRecorder(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._recording = False
        self._received = threading.Event()
        self.samples = dict((name, []) for name in JOINT_ORDER)

    def callback(self, message):
        positions = dict(zip(message.name, message.position))
        velocities = dict(zip(message.name, message.velocity))
        self._received.set()
        with self._lock:
            if not self._recording:
                return
            stamp = message.header.stamp.to_sec()
            if stamp <= 0.0:
                stamp = rospy.get_time()
            for name in JOINT_ORDER:
                if name not in positions:
                    continue
                self.samples[name].append((
                    stamp,
                    float(positions[name]),
                    float(velocities.get(name, 0.0)),
                ))

    def wait_for_first_message(self, timeout):
        return self._received.wait(timeout)

    def start(self):
        with self._lock:
            self.samples = dict((name, []) for name in JOINT_ORDER)
            self._recording = True

    def stop(self):
        with self._lock:
            self._recording = False


def controller_states():
    rospy.wait_for_service("/controller_manager/list_controllers", timeout=10.0)
    response = rospy.ServiceProxy(
        "/controller_manager/list_controllers",
        ListControllers,
    )()
    return dict((controller.name, controller.state) for controller in response.controller)


def gravity_modes():
    rospy.wait_for_service("/gazebo/get_link_properties", timeout=10.0)
    service = rospy.ServiceProxy("/gazebo/get_link_properties", GetLinkProperties)
    result = {}
    for link in LINKS:
        scoped_name = "mobile_arm::{}".format(link)
        try:
            response = service(scoped_name)
            if response.success:
                result[link] = bool(response.gravity_mode)
            elif link == "sway_link":
                result[link] = "NOT_PRESENT_OR_FIXED_JOINT_LUMPED"
            else:
                result[link] = "ERROR: {}".format(response.status_message)
        except rospy.ServiceException as error:
            result[link] = "ERROR: {}".format(error)
    return result


def summarize(samples):
    rows = []
    for name in JOINT_ORDER:
        joint_samples = samples[name]
        if not joint_samples:
            rows.append({
                "joint": name,
                "sample_count": 0,
                "initial_position": "",
                "final_position": "",
                "signed_drift": "",
                "absolute_drift": "",
                "cumulative_abs_position_change": "",
                "max_abs_velocity": "",
                "mean_abs_velocity": "",
                "unit": "m" if name in PRISMATIC_JOINTS else "rad",
                "idle_velocity_pass": True,
                "idle_drift_pass": True,
                "idle_pass": True,
            })
            continue

        positions = [sample[1] for sample in joint_samples]
        velocities = [sample[2] for sample in joint_samples]
        signed_drift = positions[-1] - positions[0]
        cumulative_change = sum(
            abs(positions[index] - positions[index - 1])
            for index in range(1, len(positions))
        )
        max_abs_velocity = max(abs(value) for value in velocities)
        mean_abs_velocity = sum(abs(value) for value in velocities) / len(velocities)
        if name in ARM_JOINTS:
            velocity_limit = 0.002
            drift_limit = 0.001 if name in PRISMATIC_JOINTS else 0.01
        elif name in set(["x", "y"]):
            velocity_limit = 0.0002
            drift_limit = 0.001
        else:
            velocity_limit = 0.002
            drift_limit = 0.01
        velocity_pass = max_abs_velocity < velocity_limit
        drift_pass = abs(signed_drift) < drift_limit
        rows.append({
            "joint": name,
            "sample_count": len(joint_samples),
            "initial_position": positions[0],
            "final_position": positions[-1],
            "signed_drift": signed_drift,
            "absolute_drift": abs(signed_drift),
            "cumulative_abs_position_change": cumulative_change,
            "max_abs_velocity": max_abs_velocity,
            "mean_abs_velocity": mean_abs_velocity,
            "unit": "m" if name in PRISMATIC_JOINTS else "rad",
            "idle_velocity_pass": velocity_pass,
            "idle_drift_pass": drift_pass,
            "idle_pass": velocity_pass and drift_pass,
        })
    return rows


def write_results(output_dir, label, samples, rows, controllers, gravity, args):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "{}_{}".format(timestamp, label)
    csv_path = os.path.join(output_dir, stem + "_summary.csv")
    timeseries_path = os.path.join(output_dir, stem + "_timeseries.csv")
    text_path = os.path.join(output_dir, stem + ".txt")
    fields = [
        "joint", "sample_count", "initial_position", "final_position",
        "signed_drift", "absolute_drift", "cumulative_abs_position_change",
        "max_abs_velocity", "mean_abs_velocity", "unit", "idle_velocity_pass",
        "idle_drift_pass", "idle_pass",
    ]
    with open(csv_path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with open(timeseries_path, "w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stamp", "joint", "position", "velocity"])
        for name in JOINT_ORDER:
            for stamp, position, velocity in samples[name]:
                writer.writerow([stamp, name, position, velocity])

    with open(text_path, "w") as stream:
        stream.write("label: {}\n".format(label))
        stream.write("warmup_seconds: {:.3f}\n".format(args.warmup))
        stream.write("record_seconds: {:.3f}\n\n".format(args.duration))
        stream.write("controllers:\n")
        for name in sorted(controllers):
            stream.write("  {}: {}\n".format(name, controllers[name]))
        stream.write("\ngravity_mode:\n")
        for name in LINKS:
            stream.write("  {}: {}\n".format(name, gravity.get(name, "missing")))
        stream.write("\njoint_metrics:\n")
        for row in rows:
            stream.write(
                "  {joint}: samples={sample_count} drift={signed_drift} "
                "max_abs_velocity={max_abs_velocity} "
                "mean_abs_velocity={mean_abs_velocity} pass={idle_pass}\n".format(
                    **row
                )
            )
    return csv_path, timeseries_path, text_path


def main():
    rospy.init_node("diagnose_idle_motion", anonymous=True)
    args = parse_args()
    if args.warmup < 0.0 or args.duration <= 0.0:
        raise ValueError("warmup must be >= 0 and duration must be > 0")

    recorder = JointRecorder()
    rospy.Subscriber("/joint_states", JointState, recorder.callback, queue_size=100)
    if not recorder.wait_for_first_message(10.0):
        raise RuntimeError("no /joint_states message received within 10 seconds")

    controllers = controller_states()
    gravity = gravity_modes()
    rospy.loginfo("Idle diagnostic warmup: %.1f seconds", args.warmup)
    time.sleep(args.warmup)
    recorder.start()
    rospy.loginfo("Idle diagnostic recording: %.1f seconds", args.duration)
    time.sleep(args.duration)
    recorder.stop()

    rows = summarize(recorder.samples)
    csv_path, timeseries_path, text_path = write_results(
        args.output_dir,
        args.label,
        recorder.samples,
        rows,
        controllers,
        gravity,
        args,
    )
    for row in rows:
        rospy.loginfo(
            "%s samples=%s drift=%s max|dq|=%s mean|dq|=%s pass=%s",
            row["joint"], row["sample_count"], row["signed_drift"],
            row["max_abs_velocity"], row["mean_abs_velocity"],
            row["idle_pass"],
        )
    rospy.loginfo("Idle diagnostic CSV: %s", csv_path)
    rospy.loginfo("Idle diagnostic time series: %s", timeseries_path)
    rospy.loginfo("Idle diagnostic summary: %s", text_path)


if __name__ == "__main__":
    main()
