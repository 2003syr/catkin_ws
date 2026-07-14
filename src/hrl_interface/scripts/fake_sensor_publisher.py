#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import math
import rospy
from sensor_msgs.msg import JointState, LaserScan
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler


class FakeSensorPublisher(object):
    def __init__(self):
        self.rate_hz = rospy.get_param("~rate", 20.0)
        self.t0 = rospy.Time.now().to_sec()

        self.joint_names = rospy.get_param("~joint_names", [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        ])

        self.pub_joint = rospy.Publisher("/joint_states", JointState, queue_size=10)
        self.pub_scan = rospy.Publisher("/scan", LaserScan, queue_size=10)
        self.pub_tag = rospy.Publisher("/tag_pose_camera", PoseStamped, queue_size=10)

        rospy.loginfo("FakeSensorPublisher started.")
        rospy.loginfo("Publish /joint_states, /scan, /tag_pose_camera")

    def publish_joint_states(self, t):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names

        # 模拟 6 个关节缓慢摆动，单位 rad
        msg.position = [
            0.30 * math.sin(0.30 * t),
            0.25 * math.sin(0.25 * t + 0.5),
            0.20 * math.sin(0.20 * t + 1.0),
            0.15 * math.sin(0.35 * t + 1.5),
            0.18 * math.sin(0.28 * t + 2.0),
            0.12 * math.sin(0.40 * t + 2.5),
        ]

        msg.velocity = [
            0.30 * 0.30 * math.cos(0.30 * t),
            0.25 * 0.25 * math.cos(0.25 * t + 0.5),
            0.20 * 0.20 * math.cos(0.20 * t + 1.0),
            0.15 * 0.35 * math.cos(0.35 * t + 1.5),
            0.18 * 0.28 * math.cos(0.28 * t + 2.0),
            0.12 * 0.40 * math.cos(0.40 * t + 2.5),
        ]

        msg.effort = [0.0] * 6
        self.pub_joint.publish(msg)

    def publish_scan(self, t):
        msg = LaserScan()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "laser_link"

        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = math.radians(1.0)

        n = int((msg.angle_max - msg.angle_min) / msg.angle_increment) + 1

        msg.time_increment = 0.0
        msg.scan_time = 1.0 / self.rate_hz
        msg.range_min = 0.05
        msg.range_max = 5.0

        ranges = []

        # 默认远距离
        for i in range(n):
            angle = msg.angle_min + i * msg.angle_increment

            # 模拟前方有一个周期性靠近/远离的障碍物
            front_obs = 1.0 + 0.4 * math.sin(0.5 * t)

            # 模拟右前方有固定障碍
            right_front_obs = 0.8

            d = 3.0

            # 前方 -15° ~ 15°
            if abs(angle) < math.radians(15):
                d = front_obs

            # 右前方 -60° ~ -30°
            if math.radians(-60) < angle < math.radians(-30):
                d = min(d, right_front_obs)

            ranges.append(d)

        msg.ranges = ranges
        self.pub_scan.publish(msg)

    def publish_tag_pose(self, t):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "camera_optical_frame"

        # 模拟 Tag 在相机坐标系下的位置
        # optical frame 常见定义：x 右，y 下，z 前
        msg.pose.position.x = 0.10 * math.sin(0.30 * t)
        msg.pose.position.y = 0.05 * math.sin(0.20 * t + 1.0)
        msg.pose.position.z = 0.80 + 0.20 * math.sin(0.15 * t)

        # 模拟 Tag 姿态有轻微变化
        roll = 0.05 * math.sin(0.25 * t)
        pitch = 0.10 * math.sin(0.18 * t + 0.5)
        yaw = 0.08 * math.sin(0.22 * t + 1.0)

        q = quaternion_from_euler(roll, pitch, yaw)

        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        self.pub_tag.publish(msg)

    def run(self):
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - self.t0

            self.publish_joint_states(t)
            self.publish_scan(t)
            self.publish_tag_pose(t)

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("fake_sensor_publisher")
    node = FakeSensorPublisher()
    node.run()
