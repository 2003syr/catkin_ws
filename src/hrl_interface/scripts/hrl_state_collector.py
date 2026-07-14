#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import csv
import math
import threading
import rospy

from sensor_msgs.msg import JointState, LaserScan
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray


def clamp(x, low, high):
    return max(low, min(high, x))


def safe_div(x, scale):
    if abs(scale) < 1e-9:
        return 0.0
    return x / scale


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_to_rpy(qx, qy, qz, qw):
    """
    四元数转 roll pitch yaw
    """
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class HRLStateCollector(object):
    def __init__(self):
        self.lock = threading.Lock()

        # ---------- 参数 ----------
        self.joint_names = rospy.get_param("~joint_names", [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
        ])

        self.joint_limits = rospy.get_param("~joint_limits", {})
        self.desired_tag_z = float(rospy.get_param("~desired_tag_z", 0.60))

        self.pos_error_scale = float(rospy.get_param("~pos_error_scale", 1.0))
        self.angle_error_scale = float(rospy.get_param("~angle_error_scale", math.pi))
        self.obstacle_safe_distance = float(rospy.get_param("~obstacle_safe_distance", 1.5))

        self.max_subgoal_xyz = float(rospy.get_param("~max_subgoal_xyz", 0.10))
        self.max_subgoal_rpy = float(rospy.get_param("~max_subgoal_rpy", 0.20))

        self.fake_k_pos = float(rospy.get_param("~fake_high_goal_k_pos", 0.25))
        self.fake_k_rpy = float(rospy.get_param("~fake_high_goal_k_rpy", 0.20))

        self.laser_sectors = rospy.get_param("~laser_sectors", {
            "front": [-15, 15],
            "front_left": [15, 60],
            "left": [60, 120],
            "back_left": [120, 165],
            "back": [165, -165],
            "back_right": [-165, -120],
            "right": [-120, -60],
            "front_right": [-60, -15],
        })

        self.sector_order = [
            "front",
            "front_left",
            "left",
            "back_left",
            "back",
            "back_right",
            "right",
            "front_right",
        ]

        self.log_file = rospy.get_param("~log_file", "logs/hrl_state_log.csv")
        self.print_rate = float(rospy.get_param("~print_rate", 2.0))

        # ---------- 状态缓存 ----------
        self.joint_pos = {}
        self.joint_vel = {}
        self.has_joint = False

        self.tag_pose = None
        self.has_tag = False
        self.last_tag_time = None

        self.obstacle_dist = [self.obstacle_safe_distance] * 8
        self.has_scan = False

        self.prev_high_goal = [0.0] * 6
        self.fake_high_goal = [0.0] * 6

        # ---------- ROS 通信 ----------
        self.sub_joint = rospy.Subscriber("/joint_states", JointState, self.cb_joint, queue_size=1)
        self.sub_scan = rospy.Subscriber("/scan", LaserScan, self.cb_scan, queue_size=1)
        self.sub_tag = rospy.Subscriber("/tag_pose_camera", PoseStamped, self.cb_tag, queue_size=1)

        self.pub_state = rospy.Publisher("/hrl/high_state", Float32MultiArray, queue_size=10)
        self.pub_goal = rospy.Publisher("/hrl/fake_high_goal", Float32MultiArray, queue_size=10)

        # ---------- 日志 ----------
        self.csv_file = None
        self.csv_writer = None
        self.prepare_logger()

        rospy.loginfo("HRLStateCollector started.")
        rospy.loginfo("High-level state topic: /hrl/high_state")
        rospy.loginfo("Fake high goal topic:   /hrl/fake_high_goal")

    def prepare_logger(self):
        pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = self.log_file

        if not os.path.isabs(log_path):
            log_path = os.path.join(pkg_path, log_path)

        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.csv_file = open(log_path, "w")
        self.csv_writer = csv.writer(self.csv_file)

        header = [
            "time",
            "has_tag",
            "ex", "ey", "ez",
            "eroll", "epitch", "eyaw",
        ]

        for name in self.joint_names:
            header.append("q_" + name)

        for name in self.joint_names:
            header.append("margin_" + name)

        for name in self.sector_order:
            header.append("obs_" + name)

        for i in range(6):
            header.append("prev_goal_" + str(i))

        for i in range(6):
            header.append("fake_goal_" + str(i))

        self.csv_writer.writerow(header)
        self.csv_file.flush()

        rospy.loginfo("Log file: %s", log_path)

    def cb_joint(self, msg):
        with self.lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.joint_pos[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.joint_vel[name] = msg.velocity[i]
            self.has_joint = True

    def cb_tag(self, msg):
        with self.lock:
            self.tag_pose = msg.pose
            self.has_tag = True
            self.last_tag_time = rospy.Time.now()

    def cb_scan(self, msg):
        sector_values = []

        for sector_name in self.sector_order:
            if sector_name not in self.laser_sectors:
                sector_values.append(self.obstacle_safe_distance)
                continue

            deg_min, deg_max = self.laser_sectors[sector_name]
            dist = self.get_sector_min_distance(msg, deg_min, deg_max)
            sector_values.append(dist)

        with self.lock:
            self.obstacle_dist = sector_values
            self.has_scan = True

    def get_sector_min_distance(self, scan, deg_min, deg_max):
        """
        支持普通区间，比如 [-15, 15]
        也支持跨越正负180度的区间，比如 [165, -165]
        """
        rad_min = math.radians(deg_min)
        rad_max = math.radians(deg_max)

        values = []

        for i, r in enumerate(scan.ranges):
            if math.isinf(r) or math.isnan(r):
                continue

            if r <= scan.range_min or r >= scan.range_max:
                continue

            angle = scan.angle_min + i * scan.angle_increment
            angle = normalize_angle(angle)

            inside = False

            if rad_min <= rad_max:
                if angle >= rad_min and angle <= rad_max:
                    inside = True
            else:
                # 跨越 -pi / pi
                if angle >= rad_min or angle <= rad_max:
                    inside = True

            if inside:
                values.append(r)

        if len(values) == 0:
            return self.obstacle_safe_distance

        return min(values)

    def get_joint_vector(self):
        q = []
        for name in self.joint_names:
            q.append(self.joint_pos.get(name, 0.0))
        return q

    def get_joint_margin(self, q):
        margins = []

        for i, name in enumerate(self.joint_names):
            qi = q[i]

            if name in self.joint_limits:
                qmin = float(self.joint_limits[name][0])
                qmax = float(self.joint_limits[name][1])
            else:
                qmin = -math.pi
                qmax = math.pi

            margin = min(qi - qmin, qmax - qi)
            total = qmax - qmin

            margin_norm = safe_div(margin, total)
            margin_norm = clamp(margin_norm, 0.0, 1.0)

            margins.append(margin_norm)

        return margins

    def get_camera_goal_error(self):
        """
        计算目标相对末端相机误差。

        这里假设 /tag_pose_camera 表示 Tag 在相机坐标系下的位姿。
        ROS optical frame 通常为：
        x 右，y 下，z 前。

        第一版定义：
        目标理想状态为：
        x = 0
        y = 0
        z = desired_tag_z
        姿态误差 = 当前 Tag 姿态的 rpy
        """
        if not self.has_tag or self.tag_pose is None:
            return [0.0] * 6, False

        p = self.tag_pose.position
        q = self.tag_pose.orientation

        roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)

        ex = p.x
        ey = p.y
        ez = p.z - self.desired_tag_z

        eroll = normalize_angle(roll)
        epitch = normalize_angle(pitch)
        eyaw = normalize_angle(yaw)

        return [ex, ey, ez, eroll, epitch, eyaw], True

    def normalize_state(self, e_camera, q, margin, obstacle, prev_goal):
        state = []

        # 1. 目标相对末端相机误差 6维
        state.append(clamp(safe_div(e_camera[0], self.pos_error_scale), -1.0, 1.0))
        state.append(clamp(safe_div(e_camera[1], self.pos_error_scale), -1.0, 1.0))
        state.append(clamp(safe_div(e_camera[2], self.pos_error_scale), -1.0, 1.0))

        state.append(clamp(safe_div(e_camera[3], self.angle_error_scale), -1.0, 1.0))
        state.append(clamp(safe_div(e_camera[4], self.angle_error_scale), -1.0, 1.0))
        state.append(clamp(safe_div(e_camera[5], self.angle_error_scale), -1.0, 1.0))

        # 2. 目标相对底盘误差：当前没有模型，先占位 3维
        # 后续有 TF 后替换为 [dx_base, dy_base, dyaw_base]
        state.extend([0.0, 0.0, 0.0])

        # 3. 关节角 6维，按关节范围归一化到 [-1,1]
        for i, name in enumerate(self.joint_names):
            qi = q[i]
            if name in self.joint_limits:
                qmin = float(self.joint_limits[name][0])
                qmax = float(self.joint_limits[name][1])
            else:
                qmin = -math.pi
                qmax = math.pi

            qnorm = 2.0 * safe_div((qi - qmin), (qmax - qmin)) - 1.0
            state.append(clamp(qnorm, -1.0, 1.0))

        # 4. 关节限位余量 6维，已经是 [0,1]
        state.extend(margin)

        # 5. 障碍物 8维，距离归一化到 [0,1]
        for d in obstacle:
            d_norm = clamp(safe_div(d, self.obstacle_safe_distance), 0.0, 1.0)
            state.append(d_norm)

        # 6. 上一次高层子目标 6维
        for i in range(3):
            state.append(clamp(safe_div(prev_goal[i], self.max_subgoal_xyz), -1.0, 1.0))

        for i in range(3, 6):
            state.append(clamp(safe_div(prev_goal[i], self.max_subgoal_rpy), -1.0, 1.0))

        return state

    def compute_fake_high_goal(self, e_camera):
        """
        假高层输出：
        用简单比例规则生成局部子目标。

        这里的 g_H 是相机坐标系下的局部位姿增量。
        第一版只用于接口验证，不直接代表最终控制策略。
        """
        gx = clamp(self.fake_k_pos * e_camera[0], -self.max_subgoal_xyz, self.max_subgoal_xyz)
        gy = clamp(self.fake_k_pos * e_camera[1], -self.max_subgoal_xyz, self.max_subgoal_xyz)
        gz = clamp(self.fake_k_pos * e_camera[2], -self.max_subgoal_xyz, self.max_subgoal_xyz)

        groll = clamp(self.fake_k_rpy * e_camera[3], -self.max_subgoal_rpy, self.max_subgoal_rpy)
        gpitch = clamp(self.fake_k_rpy * e_camera[4], -self.max_subgoal_rpy, self.max_subgoal_rpy)
        gyaw = clamp(self.fake_k_rpy * e_camera[5], -self.max_subgoal_rpy, self.max_subgoal_rpy)

        return [gx, gy, gz, groll, gpitch, gyaw]

    def publish_array(self, pub, data):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in data]
        pub.publish(msg)

    def write_log(self, e_camera, has_tag, q, margin, obstacle, prev_goal, fake_goal):
        if self.csv_writer is None:
            return

        row = [
            rospy.Time.now().to_sec(),
            int(has_tag),
            e_camera[0], e_camera[1], e_camera[2],
            e_camera[3], e_camera[4], e_camera[5],
        ]

        row.extend(q)
        row.extend(margin)
        row.extend(obstacle)
        row.extend(prev_goal)
        row.extend(fake_goal)

        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def run(self):
        rate_hz = rospy.get_param("~rate", 10.0)
        rate = rospy.Rate(rate_hz)

        last_print_time = rospy.Time.now()

        while not rospy.is_shutdown():
            with self.lock:
                q = self.get_joint_vector()
                margin = self.get_joint_margin(q)
                obstacle = list(self.obstacle_dist)
                prev_goal = list(self.prev_high_goal)
                e_camera, has_tag = self.get_camera_goal_error()

            fake_goal = self.compute_fake_high_goal(e_camera)
            high_state = self.normalize_state(e_camera, q, margin, obstacle, prev_goal)

            # 发布
            self.publish_array(self.pub_state, high_state)
            self.publish_array(self.pub_goal, fake_goal)

            # 保存上一高层输出
            with self.lock:
                self.prev_high_goal = list(fake_goal)
                self.fake_high_goal = list(fake_goal)

            # 记录日志
            self.write_log(e_camera, has_tag, q, margin, obstacle, prev_goal, fake_goal)

            # 打印
            now = rospy.Time.now()
            if (now - last_print_time).to_sec() >= 1.0 / max(self.print_rate, 0.1):
                last_print_time = now

                rospy.loginfo("has_tag=%s", str(has_tag))
                rospy.loginfo("e_camera_goal = [%.3f %.3f %.3f | %.3f %.3f %.3f]",
                              e_camera[0], e_camera[1], e_camera[2],
                              e_camera[3], e_camera[4], e_camera[5])
                rospy.loginfo("q = %s", ["%.3f" % x for x in q])
                rospy.loginfo("margin = %s", ["%.3f" % x for x in margin])
                rospy.loginfo("obstacle = %s", ["%.3f" % x for x in obstacle])
                rospy.loginfo("fake_goal = %s", ["%.3f" % x for x in fake_goal])
                rospy.loginfo("high_state_dim = %d", len(high_state))

            rate.sleep()

    def close(self):
        if self.csv_file is not None:
            self.csv_file.close()


if __name__ == "__main__":
    rospy.init_node("hrl_state_collector")

    node = HRLStateCollector()

    try:
        node.run()
    finally:
        node.close()
