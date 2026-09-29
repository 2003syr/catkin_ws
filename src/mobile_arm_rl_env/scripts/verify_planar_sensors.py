#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Verify the planar laser sectors and non-ground contact topic."""

from __future__ import print_function

import sys
import threading
import time

import numpy as np
import rospy

from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import LaserScan

from hrl.laser_scan_sectors import SECTOR_NAMES, sectorize_scan


class SensorMonitor(object):
    def __init__(self, ignored_contact_names):
        self.lock = threading.Lock()
        self.scan_event = threading.Event()
        self.contact_event = threading.Event()
        self.scan = None
        self.collision_names = []
        self.ignored_contact_names = tuple(ignored_contact_names)

    def scan_callback(self, message):
        with self.lock:
            self.scan = message
        self.scan_event.set()

    def contact_callback(self, message):
        collisions = []
        for state in message.states:
            names = "{} {}".format(
                state.collision1_name,
                state.collision2_name,
            )
            if any(
                    value and value in names
                    for value in self.ignored_contact_names):
                continue
            collisions.append(names)
        with self.lock:
            self.collision_names = collisions
        self.contact_event.set()

    def snapshot(self):
        with self.lock:
            return self.scan, list(self.collision_names)


def main():
    rospy.init_node("verify_planar_sensors")
    scan_topic = rospy.get_param("~scan_topic", "/scan")
    contact_topic = rospy.get_param("~contact_topic", "/base_contacts")
    timeout = float(rospy.get_param("~timeout", 10.0))
    expected_obstacle_max = float(
        rospy.get_param("~expected_obstacle_max", 2.0)
    )
    minimum_valid_clearance = float(
        rospy.get_param("~minimum_valid_clearance", 0.20)
    )
    output_max = float(rospy.get_param("~scan_output_max", 10.0))
    ignored = rospy.get_param(
        "~collision_ignore_names", ["ground_plane"]
    )
    monitor = SensorMonitor(ignored)
    scan_subscriber = rospy.Subscriber(
        scan_topic,
        LaserScan,
        monitor.scan_callback,
        queue_size=1,
    )
    contact_subscriber = rospy.Subscriber(
        contact_topic,
        ContactsState,
        monitor.contact_callback,
        queue_size=10,
    )
    try:
        if not monitor.scan_event.wait(timeout):
            rospy.logerr("No LaserScan received on %s", scan_topic)
            return 2
        if not monitor.contact_event.wait(timeout):
            rospy.logerr("No ContactsState received on %s", contact_topic)
            return 3

        scan, collision_names = monitor.snapshot()
        sectors = sectorize_scan(
            scan.ranges,
            scan.angle_min,
            scan.angle_increment,
            scan.range_min,
            scan.range_max,
            output_max=output_max,
        )
        angle_span = abs(float(scan.angle_increment)) * len(scan.ranges)
        coverage_ok = angle_span >= 1.9 * np.pi
        samples_ok = len(scan.ranges) >= 180
        obstacle_ok = float(np.min(sectors)) < expected_obstacle_max
        self_clearance_ok = bool(
            float(np.min(sectors)) >= minimum_valid_clearance
        )
        contact_ok = not collision_names
        rospy.loginfo(
            "Planar scan samples=%d span=%.3f sectors=%s values=%s",
            len(scan.ranges),
            angle_span,
            list(SECTOR_NAMES),
            [round(float(value), 3) for value in sectors],
        )
        rospy.loginfo(
            "Planar contacts non_ground=%s samples_ok=%s coverage_ok=%s "
            "obstacle_ok=%s self_clearance_ok=%s contact_ok=%s",
            collision_names,
            samples_ok,
            coverage_ok,
            obstacle_ok,
            self_clearance_ok,
            contact_ok,
        )
        if not (
                samples_ok
                and coverage_ok
                and obstacle_ok
                and self_clearance_ok
                and contact_ok):
            rospy.logerr("Planar sensor verification failed")
            return 4
        rospy.loginfo("Planar sensor verification passed")
        return 0
    finally:
        # Python 2 can report a daemon-thread exception if rospy subscribers
        # are still receiving data while the short-lived verifier exits.
        scan_subscriber.unregister()
        contact_subscriber.unregister()
        if not rospy.is_shutdown():
            rospy.signal_shutdown("planar sensor verification complete")
        time.sleep(0.10)


if __name__ == "__main__":
    sys.exit(main())
