#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import time

import rospy
import numpy as np
import tf

from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Float64

from hrl.arm_chassis_guard import ArmChassisInterferenceGuard
from hrl.laser_scan_sectors import SECTOR_NAMES, bin_scan, sectorize_scan
from training.tracked_base_kinematics import TRACKED_FORWARD_YAW_OFFSET


class MobileArmReachEnv(object):
    OBS_DIM = 46
    ACTION_DIM = 10

    def __init__(self, init_ros_node=True):
        if init_ros_node:
            rospy.init_node("mobile_arm_reach_env_check", anonymous=True)

        # =========================
        # 1. 坐标系参数
        # =========================
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.ee_frame = rospy.get_param("~ee_frame", "link6")
        self.target_frame = rospy.get_param("~target_frame", "target_frame")

        # Five sectors cover the full 360-degree plane.  Keeping this compact
        # representation preserves the 46-D ROS observation while allowing a
        # task-specific 11-D base-policy adapter.
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.contact_topic = rospy.get_param(
            "~contact_topic", "/base_contacts"
        )
        self.arm_contact_topic = str(
            rospy.get_param("~arm_contact_topic", "")
        ).strip()
        self.scan_output_max = float(
            rospy.get_param("~scan_output_max", 10.0)
        )
        self.scan_bin_count = int(
            rospy.get_param("~scan_bin_count", 36)
        )
        self.scan_stale_timeout = float(
            rospy.get_param("~scan_stale_timeout", 1.0)
        )
        self.scan_wait_timeout = float(
            rospy.get_param("~scan_wait_timeout", 10.0)
        )
        self.require_scan = bool(rospy.get_param("~require_scan", False))
        self.collision_ignore_names = tuple(
            str(value)
            for value in rospy.get_param(
                "~collision_ignore_names", ["ground_plane"]
            )
        )
        if self.scan_output_max <= 0.0:
            raise ValueError("scan_output_max must be positive")
        if self.scan_bin_count <= 0:
            raise ValueError("scan_bin_count must be positive")
        if self.scan_stale_timeout <= 0.0 or self.scan_wait_timeout <= 0.0:
            raise ValueError("scan timeouts must be positive")

        self._sensor_lock = threading.Lock()
        self._scan_condition = threading.Condition(self._sensor_lock)
        self._scan_received = threading.Event()
        self._contact_received = threading.Event()
        self._scan_info = np.full(5, self.scan_output_max, dtype=np.float64)
        self._scan_bins = np.full(
            self.scan_bin_count,
            self.scan_output_max,
            dtype=np.float64,
        )
        self._scan_wall_time = None
        self._collision_active = False
        self._collision_names = []
        self._contact_sources = {
            "base": False,
            "arm": False,
        }
        self._contact_source_names = {
            "base": [],
            "arm": [],
        }

        # TF 监听器
        self.listener = tf.TransformListener()

        # =========================
        # 2. 固定关节顺序
        # =========================
        # 后续 obs_vec 中 q、dq、q_margin 都严格按这个顺序排列
        # action 也按这个顺序解释
        self.expected_joints = [
            "x",
            "y",
            "z",
            "sway",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6"
        ]
        self.enable_planar_base = bool(
            rospy.get_param("~enable_planar_base", False)
        )
        # x/y/yaw are internal planar pose coordinates.  Stage-specific base
        # policies expose only [linear velocity, yaw rate] and perform the
        # nonholonomic mapping before these controllers are published.
        self.planar_base_joints = ["x", "y", "z"]
        self.arm_joints = self.expected_joints[4:10]
        if self.enable_planar_base:
            self.locked_virtual_joints = {"sway"}
            # Planar yaw is an unbounded configuration coordinate.  Treating
            # it like a finite revolute joint eventually blocked one turning
            # direction after several training episodes.
            self.continuous_joints = {"z"}
            self.controlled_joints = (
                self.planar_base_joints + self.arm_joints
            )
        else:
            self.locked_virtual_joints = {"x", "y", "z", "sway"}
            self.continuous_joints = set()
            self.controlled_joints = list(self.arm_joints)

        # =========================
        # 3. 机械臂位置限位
        # =========================
        # 注意：
        # 这里与 URDF 的物理限位保持一致。当前数值是为仿真和底盘
        # 净空选取的保守范围；拿到真实机械臂规格后应统一替换。
        self.joint_limits = {
            "x": (-2.0, 2.0),
            "y": (-2.0, 2.0),
            "z": (-6.28, 6.28),
            "sway": (-1.57, 1.57),

            "joint1": (-1.57, 1.57),
            "joint2": (-1.45, 0.0),
            "joint3": (0.00, 0.15),
            "joint4": (-1.57, 1.57),
            "joint5": (0.00, 0.15),
            "joint6": (-1.57, 1.57)
        }

        # =========================
        # 4. 动作空间设置
        # =========================
        # action 顺序与 expected_joints 保持一致
        self.action_dim = len(self.expected_joints)

        # 每个动作维度对应的最大速度
        # x, y, joint3, joint5 是 prismatic，单位 m/s
        # z, sway, joint1, joint2, joint4, joint6 是 revolute，单位 rad/s
        self.action_max_vel = {
            "x": 0.20,
            "y": 0.20,
            "z": 0.50,
            "sway": 0.50,
            "joint1": 0.50,
            "joint2": 0.50,
            "joint3": 0.03,
            "joint4": 0.50,
            "joint5": 0.03,
            "joint6": 0.50
        }

        # =========================
        # 5. 动作安全限制参数
        # =========================
        # hard_margin：
        #   进入硬限位区域后，禁止继续朝限位方向运动。
        #
        # soft_margin：
        #   进入软限位区域后，朝限位方向的速度逐渐减小。
        #
        # revolute 关节单位 rad
        # prismatic 关节单位 m
        self.limit_hard_margin = {
            "x": 0.05,
            "y": 0.05,
            "z": 0.05,
            "sway": 0.05,

            "joint1": 0.05,
            "joint2": 0.05,
            "joint3": 0.01,
            "joint4": 0.05,
            "joint5": 0.01,
            "joint6": 0.05
        }

        self.limit_soft_margin = {
            "x": 0.20,
            "y": 0.20,
            "z": 0.30,
            "sway": 0.20,

            "joint1": 0.30,
            "joint2": 0.20,
            "joint3": 0.05,
            "joint4": 0.30,
            "joint5": 0.05,
            "joint6": 0.30
        }

        # A joint can remain inside its actuator limits while a distal arm
        # link enters the chassis.  Keep that configuration-space constraint
        # in the existing post-policy safety layer.
        self.enable_chassis_interference_guard = bool(
            rospy.get_param(
                "~enable_chassis_interference_guard",
                True,
            )
        )
        self.arm_chassis_guard = None
        self.last_chassis_guard_info = {
            "reason": "disabled",
            "scale": 1.0,
            "current_clearance": float("inf"),
            "predicted_clearance": float("inf"),
            "closest_link": "none",
            "predicted_link": "none",
        }
        if self.enable_chassis_interference_guard:
            self.arm_chassis_guard = ArmChassisInterferenceGuard(
                hard_clearance=rospy.get_param(
                    "~chassis_hard_clearance",
                    0.010,
                ),
                soft_clearance=rospy.get_param(
                    "~chassis_soft_clearance",
                    0.050,
                ),
                prediction_horizon=rospy.get_param(
                    "~chassis_prediction_horizon",
                    0.25,
                ),
                robot_description="/robot_description",
            )

        # =========================
        # 6. Gazebo velocity controller 发布器
        # =========================
        # 发布到：
        # /x_velocity_controller/command
        # /joint1_velocity_controller/command
        # ...
        self.cmd_publishers = {}

        # The default mode publishes only arm commands. The isolated planar
        # launch additionally exposes x/y while z/sway remain locked.
        for joint_name in self.controlled_joints:
            topic_name = "/{}_velocity_controller/command".format(joint_name)

            self.cmd_publishers[joint_name] = rospy.Publisher(
                topic_name,
                Float64,
                queue_size=1
            )

        # 用字典保存 /joint_states 中收到的关节状态
        self.joint_pos_dict = {}
        self.joint_vel_dict = {}

        # 仅用于调试打印
        self.joint_names = []

        # =========================
        # 7. 奖励和终止条件参数
        # =========================
        self.prev_dist = None
        self.success_threshold = 0.05
        self.max_steps = 300
        self.step_count = 0
        self._stop_complete = False

        # 订阅 joint_states
        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber(
            self.scan_topic,
            LaserScan,
            self.scan_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.contact_topic,
            ContactsState,
            self.contact_callback,
            callback_args="base",
            queue_size=10,
        )
        if self.arm_contact_topic and (
                self.arm_contact_topic != self.contact_topic):
            rospy.Subscriber(
                self.arm_contact_topic,
                ContactsState,
                self.contact_callback,
                callback_args="arm",
                queue_size=10,
            )

        rospy.loginfo("Waiting for TF and controller publishers...")
        rospy.loginfo(
            "Planar base control=%s controlled_joints=%s",
            self.enable_planar_base,
            self.controlled_joints,
        )
        rospy.sleep(2.0)
        if self.require_scan and not self._scan_received.wait(
                self.scan_wait_timeout):
            raise RuntimeError(
                "No LaserScan received on {} within {:.1f}s".format(
                    self.scan_topic,
                    self.scan_wait_timeout,
                )
            )

    def joint_state_callback(self, msg):
        """
        读取 /joint_states，并保存为字典。
        不直接使用 msg 里的顺序，避免强化学习输入顺序混乱。
        """
        self.joint_names = list(msg.name)

        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_pos_dict[name] = msg.position[i]

            if i < len(msg.velocity):
                self.joint_vel_dict[name] = msg.velocity[i]
            else:
                self.joint_vel_dict[name] = 0.0

    def scan_callback(self, msg):
        """Store both legacy sectors and a policy-facing angular profile."""
        try:
            sectors = sectorize_scan(
                msg.ranges,
                msg.angle_min,
                msg.angle_increment,
                msg.range_min,
                msg.range_max,
                output_max=self.scan_output_max,
            )
            scan_bins = bin_scan(
                msg.ranges,
                msg.angle_min,
                msg.angle_increment,
                msg.range_min,
                msg.range_max,
                output_max=self.scan_output_max,
                bin_count=self.scan_bin_count,
            )
        except (TypeError, ValueError) as error:
            rospy.logwarn_throttle(
                2.0,
                "Invalid LaserScan on %s: %s",
                self.scan_topic,
                str(error),
            )
            return
        with self._scan_condition:
            self._scan_info = sectors.astype(np.float64)
            self._scan_bins = scan_bins.astype(np.float64)
            self._scan_wall_time = time.time()
            self._scan_received.set()
            self._scan_condition.notify_all()

    def contact_callback(self, msg, source="base"):
        """Track base and arm contacts while ignoring the ground plane."""
        collision_names = []
        for state in msg.states:
            names = "{} {}".format(
                state.collision1_name,
                state.collision2_name,
            )
            if any(
                    ignored and ignored in names
                    for ignored in self.collision_ignore_names):
                continue
            collision_names.append(names)
        with self._sensor_lock:
            source = str(source) if source in ("base", "arm") else "base"
            self._contact_sources[source] = bool(collision_names)
            self._contact_source_names[source] = collision_names
            self._collision_active = bool(
                self._contact_sources["base"]
                or self._contact_sources["arm"]
            )
            self._collision_names = (
                list(self._contact_source_names["base"])
                + list(self._contact_source_names["arm"])
            )
        self._contact_received.set()

    def get_sensor_state(self):
        with self._sensor_lock:
            scan_info = self._scan_info.copy()
            scan_wall_time = self._scan_wall_time
            collision_active = bool(self._collision_active)
            collision_names = list(self._collision_names)
        scan_age = (
            float("inf")
            if scan_wall_time is None
            else max(0.0, time.time() - scan_wall_time)
        )
        scan_ok = bool(
            self._scan_received.is_set()
            and scan_age <= self.scan_stale_timeout
        )
        return (
            scan_info,
            scan_ok,
            scan_age,
            collision_active,
            collision_names,
        )

    def wait_for_fresh_scan(self, timeout=None):
        """Wait for a new scan without ever accepting stale obstacle data.

        Gazebo can pause a sensor callback briefly while a model is reset or
        while the simulator is under load.  The old behavior raised as soon
        as the last scan exceeded ``scan_stale_timeout`` and consequently
        killed a multi-hour training run.  Here the environment blocks on the
        scan condition for at most ``scan_wait_timeout`` wall-clock seconds.
        A genuinely missing sensor still fails closed after that deadline.
        """
        timeout = (
            self.scan_wait_timeout if timeout is None else float(timeout)
        )
        if timeout <= 0.0:
            raise ValueError("fresh scan wait timeout must be positive")
        started = time.time()
        deadline = started + timeout
        with self._scan_condition:
            while not rospy.is_shutdown():
                now = time.time()
                scan_age = (
                    float("inf")
                    if self._scan_wall_time is None
                    else max(0.0, now - self._scan_wall_time)
                )
                if (
                        self._scan_received.is_set()
                        and scan_age <= self.scan_stale_timeout):
                    return True, scan_age, max(0.0, now - started)
                remaining = deadline - now
                if remaining <= 0.0:
                    return False, scan_age, max(0.0, now - started)
                self._scan_condition.wait(timeout=remaining)
        return False, float("inf"), max(0.0, time.time() - started)

    def get_scan_bins(self):
        """Return the latest fixed-angle scan profile."""
        with self._sensor_lock:
            return self._scan_bins.copy()

    def get_ordered_joint_state(self):
        """
        按 expected_joints 的固定顺序输出 q 和 dq。

        输出顺序永远是：
        x, y, z, sway, joint1, joint2, joint3, joint4, joint5, joint6
        """
        q = []
        dq = []

        for name in self.expected_joints:
            q.append(self.joint_pos_dict.get(name, 0.0))
            dq.append(self.joint_vel_dict.get(name, 0.0))

        return np.array(q), np.array(dq)

    def compute_joint_margin(self, q):
        """
        计算每个关节的归一化限位余量 q_margin。

        q_margin = 1：表示关节在范围中间，最安全
        q_margin = 0：表示关节到达上下限
        """
        margins = []

        for i, name in enumerate(self.expected_joints):
            if (
                    name in self.locked_virtual_joints
                    or name in self.continuous_joints):
                margins.append(1.0)
                continue
            q_i = q[i]

            q_min, q_max = self.joint_limits[name]
            q_range = q_max - q_min

            if q_range <= 1e-6:
                margin = 0.0
            else:
                lower_margin = q_i - q_min
                upper_margin = q_max - q_i

                margin = 2.0 * min(lower_margin, upper_margin) / q_range
                margin = np.clip(margin, 0.0, 1.0)

            margins.append(margin)

        return np.array(margins)

    def decode_action(self, action):
        """
        将强化学习输出的 action 映射为实际关节/底盘速度。

        输入：
        action: 10维，范围通常为 [-1, 1]

        输出：
        cmd_dict: 按关节名保存的实际速度指令
        """
        action = np.array(action, dtype=np.float32)

        if action.shape[0] != self.action_dim:
            rospy.logwarn(
                "Action dim mismatch: got %d, expected %d",
                action.shape[0],
                self.action_dim
            )

            if action.shape[0] < self.action_dim:
                action = np.pad(
                    action,
                    (0, self.action_dim - action.shape[0]),
                    "constant"
                )
            else:
                action = action[:self.action_dim]

        action = np.clip(action, -1.0, 1.0)

        cmd_dict = {}

        for i, name in enumerate(self.expected_joints):
            max_vel = self.action_max_vel[name]
            if name in self.locked_virtual_joints:
                cmd_dict[name] = 0.0
            else:
                cmd_dict[name] = float(action[i] * max_vel)

        return cmd_dict

    def apply_action_safety_filter(self, cmd_dict, q):
        """
        根据当前关节位置 q，对速度指令 cmd_dict 做安全限制。

        规则：
        1. 接近上限时，禁止继续正向运动
        2. 接近下限时，禁止继续负向运动
        3. 进入软限位区域时，逐渐降低朝限位方向的速度
        """

        safe_cmd_dict = {}
        safety_info = {}

        for i, joint_name in enumerate(self.expected_joints):
            cmd = cmd_dict.get(joint_name, 0.0)
            q_i = q[i]

            q_min, q_max = self.joint_limits[joint_name]
            hard_margin = self.limit_hard_margin[joint_name]
            soft_margin = self.limit_soft_margin[joint_name]

            if joint_name in self.continuous_joints:
                safe_cmd_dict[joint_name] = float(cmd)
                safety_info[joint_name] = {
                    "q": float(q_i),
                    "cmd_raw": float(cmd),
                    "cmd_safe": float(cmd),
                    "dist_to_lower": float("inf"),
                    "dist_to_upper": float("inf"),
                    "reason": "continuous_joint"
                }
                continue

            dist_to_lower = q_i - q_min
            dist_to_upper = q_max - q_i

            filtered_cmd = cmd
            reason = "safe"

            # =========================
            # 1. 硬限位保护
            # =========================
            # 已经接近上限，还想继续正向运动
            if dist_to_upper <= hard_margin and cmd > 0.0:
                filtered_cmd = 0.0
                reason = "upper_hard_limit"

            # 已经接近下限，还想继续负向运动
            elif dist_to_lower <= hard_margin and cmd < 0.0:
                filtered_cmd = 0.0
                reason = "lower_hard_limit"

            # =========================
            # 2. 软限位减速
            # =========================
            # 接近上限，且还在正向运动，则按距离缩小速度
            elif dist_to_upper <= soft_margin and cmd > 0.0:
                scale = dist_to_upper / soft_margin
                scale = np.clip(scale, 0.0, 1.0)

                filtered_cmd = cmd * scale
                reason = "upper_soft_limit"

            # 接近下限，且还在负向运动，则按距离缩小速度
            elif dist_to_lower <= soft_margin and cmd < 0.0:
                scale = dist_to_lower / soft_margin
                scale = np.clip(scale, 0.0, 1.0)

                filtered_cmd = cmd * scale
                reason = "lower_soft_limit"

            safe_cmd_dict[joint_name] = float(filtered_cmd)

            safety_info[joint_name] = {
                "q": float(q_i),
                "cmd_raw": float(cmd),
                "cmd_safe": float(filtered_cmd),
                "dist_to_lower": float(dist_to_lower),
                "dist_to_upper": float(dist_to_upper),
                "reason": reason
            }

        if self.arm_chassis_guard is not None:
            arm_positions = np.asarray(q[4:10], dtype=np.float64)
            arm_velocity = np.asarray([
                safe_cmd_dict[name] for name in self.arm_joints
            ], dtype=np.float64)
            try:
                filtered_velocity, guard_info = (
                    self.arm_chassis_guard.filter_velocity(
                        arm_positions,
                        arm_velocity,
                    )
                )
            except Exception as error:
                # Fail closed: a broken clearance estimate must never turn
                # into an unfiltered arm command.
                filtered_velocity = np.zeros(6, dtype=np.float64)
                guard_info = {
                    "reason": "chassis_guard_error",
                    "scale": 0.0,
                    "current_clearance": float("nan"),
                    "predicted_clearance": float("nan"),
                    "closest_link": "unknown",
                    "predicted_link": "unknown",
                    "error": str(error),
                }
                rospy.logerr_throttle(
                    2.0,
                    "Arm-chassis guard failed closed: %s",
                    str(error),
                )

            self.last_chassis_guard_info = dict(guard_info)
            guard_reason = guard_info["reason"]
            for index, joint_name in enumerate(self.arm_joints):
                previous_command = safe_cmd_dict[joint_name]
                safe_cmd_dict[joint_name] = float(
                    filtered_velocity[index]
                )
                joint_info = safety_info[joint_name]
                joint_info.update({
                    "chassis_clearance": float(
                        guard_info["current_clearance"]
                    ),
                    "predicted_chassis_clearance": float(
                        guard_info["predicted_clearance"]
                    ),
                    "chassis_closest_link": guard_info["closest_link"],
                    "chassis_predicted_link": guard_info["predicted_link"],
                    "chassis_velocity_scale": float(guard_info["scale"]),
                })
                command_changed = not np.isclose(
                    previous_command,
                    safe_cmd_dict[joint_name],
                    atol=1.0e-9,
                )
                if guard_reason != "safe" and (
                        command_changed
                        or abs(previous_command) > 1.0e-9):
                    if joint_info["reason"] == "safe":
                        joint_info["reason"] = guard_reason
                    else:
                        joint_info["reason"] += "+" + guard_reason

        return safe_cmd_dict, safety_info

    def publish_cmd_dict(self, cmd_dict, joint_names=None):
        """
        将 cmd_dict 发布到 Gazebo velocity controller。

        cmd_dict 示例：
        {
            "joint1": 0.1,
            "joint2": 0.0,
            ...
        }
        """
        if joint_names is None:
            joint_names = self.controlled_joints
        else:
            joint_names = tuple(joint_names)
            unknown_joints = set(joint_names) - set(self.controlled_joints)
            if unknown_joints:
                raise ValueError(
                    "Cannot publish uncontrolled joints: {}".format(
                        sorted(unknown_joints)
                    )
                )

        for joint_name in joint_names:
            if joint_name not in cmd_dict:
                continue

            if joint_name not in self.cmd_publishers:
                continue

            msg = Float64()
            msg.data = cmd_dict[joint_name]

            self.cmd_publishers[joint_name].publish(msg)

    def publish_zero_cmd(self):
        """
        发布零速度，用于 episode 结束或程序退出前停止机器人。
        """
        zero_cmd = {}

        for joint_name in self.expected_joints:
            zero_cmd[joint_name] = 0.0

        self.publish_cmd_dict(zero_cmd)

    def get_tf_error(self, from_frame, to_frame):
        """
        返回 to_frame 在 from_frame 坐标系下的位置和姿态。

        例如：
        get_tf_error("base_link", "target_frame")
        表示 target_frame 相对于 base_link 的位姿。
        """
        try:
            self.listener.waitForTransform(
                from_frame,
                to_frame,
                rospy.Time(0),
                rospy.Duration(0.5)
            )

            trans, rot = self.listener.lookupTransform(
                from_frame,
                to_frame,
                rospy.Time(0)
            )

            return np.array(trans), np.array(rot), True

        except Exception as e:
            rospy.logwarn(
                "TF lookup failed: %s -> %s, %s",
                from_frame,
                to_frame,
                str(e)
            )

            return np.zeros(3), np.array([0, 0, 0, 1]), False

    def get_observation(self):
        """
        构建强化学习输入 observation。

        obs_vec 维度为 46：

        0~2    : target_in_base_pos
        3~5    : target_in_ee_pos
        6~8    : ee_in_base_pos
        9      : base_to_target_dist
        10     : ee_to_target_dist
        11~20  : q
        21~30  : dq
        31~40  : q_margin
        41~45  : scan_info
        """

        target_in_base_pos, target_in_base_rot, ok1 = self.get_tf_error(
            self.base_frame,
            self.target_frame
        )

        target_in_ee_pos, target_in_ee_rot, ok2 = self.get_tf_error(
            self.ee_frame,
            self.target_frame
        )

        ee_in_base_pos, ee_in_base_rot, ok3 = self.get_tf_error(
            self.base_frame,
            self.ee_frame
        )

        q, dq = self.get_ordered_joint_state()
        q_margin = self.compute_joint_margin(q)
        base_to_target_vector = np.asarray(
            target_in_base_pos,
            dtype=np.float64,
        ).copy()
        if self.enable_planar_base:
            base_to_target_vector[0:2] -= q[0:2]
        yaw = float(q[2]) if self.enable_planar_base else 0.0
        heading = (
            yaw + TRACKED_FORWARD_YAW_OFFSET
            if self.enable_planar_base else 0.0
        )
        cosine = np.cos(heading)
        sine = np.sin(heading)
        base_to_target_body = base_to_target_vector.copy()
        base_to_target_body[0] = (
            cosine * base_to_target_vector[0]
            + sine * base_to_target_vector[1]
        )
        base_to_target_body[1] = (
            -sine * base_to_target_vector[0]
            + cosine * base_to_target_vector[1]
        )
        base_to_target_dist = np.linalg.norm(base_to_target_vector)
        ee_to_target_dist = np.linalg.norm(target_in_ee_pos)

        # Use the real 360-degree scan reduced to five fixed planar sectors.
        (
            scan_info,
            scan_ok,
            scan_age,
            collision_active,
            collision_names,
        ) = self.get_sensor_state()
        if self.require_scan and not scan_ok:
            stale_age = scan_age
            rospy.logwarn(
                "LaserScan on %s is stale (age=%.3fs); waiting up to %.1fs "
                "for a fresh frame",
                self.scan_topic,
                stale_age,
                self.scan_wait_timeout,
            )
            recovered, scan_age, waited = self.wait_for_fresh_scan(
                self.scan_wait_timeout
            )
            if not recovered:
                raise RuntimeError(
                    "LaserScan on {} did not recover within {:.1f}s "
                    "(last_age={:.3f}s)".format(
                        self.scan_topic,
                        self.scan_wait_timeout,
                        scan_age,
                    )
                )
            (
                scan_info,
                scan_ok,
                scan_age,
                collision_active,
                collision_names,
            ) = self.get_sensor_state()
            rospy.logwarn(
                "LaserScan on %s recovered after %.3fs "
                "(stale_age=%.3fs fresh_age=%.3fs)",
                self.scan_topic,
                waited,
                stale_age,
                scan_age,
            )
        scan_bins = self.get_scan_bins()

        obs_vec = np.concatenate([
            target_in_base_pos,
            target_in_ee_pos,
            ee_in_base_pos,
            np.array([base_to_target_dist]),
            np.array([ee_to_target_dist]),
            q,
            dq,
            q_margin,
            scan_info
        ])

        if len(obs_vec) != self.OBS_DIM:
            raise RuntimeError(
                "Observation dim changed: got {}, expected {}".format(
                    len(obs_vec), self.OBS_DIM
                )
            )

        obs = {
            "obs_vec": obs_vec,

            "target_in_base_pos": target_in_base_pos,
            "target_in_ee_pos": target_in_ee_pos,
            "ee_in_base_pos": ee_in_base_pos,

            "base_to_target_pos": target_in_base_pos,
            "base_to_target_relative_pos": base_to_target_vector,
            "base_to_target_body_pos": base_to_target_body,
            "planar_base_yaw": yaw,
            "planar_base_heading": heading,
            "ee_to_target_pos": target_in_ee_pos,

            "base_to_target_dist": base_to_target_dist,
            "ee_to_target_dist": ee_to_target_dist,

            "joint_pos": q,
            "joint_vel": dq,
            "q_margin": q_margin,
            "scan_info": scan_info,
            "scan_bins": scan_bins,
            "scan_bin_count": self.scan_bin_count,
            "scan_sector_names": SECTOR_NAMES,
            "scan_ok": scan_ok,
            "scan_age": scan_age,
            "collision": collision_active,
            "collision_names": collision_names,

            "tf_ok": ok1 and ok2 and ok3
        }

        return obs

    def compute_reward_done(self, obs):
        """
        根据末端 link6 到目标 target_frame 的距离计算 reward 和 done。
        """
        ee_error = obs["target_in_ee_pos"]
        dist = np.linalg.norm(ee_error)

        tf_ok = obs.get("tf_ok", False)

        if not tf_ok:
            reward = -100.0
            done = False
            success = False
            return reward, done, dist, success

        if self.prev_dist is None:
            progress_reward = 0.0
        else:
            progress_reward = self.prev_dist - dist

        self.prev_dist = dist

        time_penalty = -0.01
        distance_penalty = -dist

        success = dist < self.success_threshold
        success_reward = 10.0 if success else 0.0

        reward = (
            5.0 * progress_reward
            + distance_penalty
            + time_penalty
            + success_reward
        )

        self.step_count += 1
        done = success or self.step_count >= self.max_steps

        return reward, done, dist, success

    def reset(self):
        """
        重置一个 episode。
        当前还没有控制 Gazebo reset，只重置计数和历史距离，并发布零速度。
        """
        self.publish_zero_cmd()

        self.prev_dist = None
        self.step_count = 0

        obs = self.get_observation()
        return obs

    def step(self, action, publish_joints=None):
        """
        将 action 解码、安全过滤并发布到 Gazebo 控制器，然后读取状态、计算奖励。
        """

        # 1. action 映射为原始速度指令
        raw_cmd_dict = self.decode_action(action)

        # 2. 读取当前关节状态，用于动作安全限制
        q_now, dq_now = self.get_ordered_joint_state()

        # 3. 根据关节限位过滤动作
        safe_cmd_dict, safety_info = self.apply_action_safety_filter(
            raw_cmd_dict,
            q_now
        )

        # 4. 发布安全后的速度指令
        # Full HRL execution keeps the historical ten-dimensional publish
        # contract.  A stage-specific wrapper may explicitly publish only
        # the joints it owns; this prevents a base-only policy from racing an
        # independent arm position-hold controller with six zero commands.
        self.publish_cmd_dict(
            safe_cmd_dict,
            joint_names=publish_joints,
        )

        # 5. 等待 Gazebo 执行一小步
        rospy.sleep(0.1)

        # 6. 读取新状态并计算奖励
        obs = self.get_observation()
        reward, done, dist, success = self.compute_reward_done(obs)

        velocity_tracking = {}
        max_normalized_error = 0.0
        max_planar_normalized_error = 0.0
        for index, joint_name in enumerate(self.expected_joints):
            desired = float(safe_cmd_dict[joint_name])
            measured = float(obs["joint_vel"][index])
            error = desired - measured
            normalized_error = abs(error) / self.action_max_vel[joint_name]
            velocity_tracking[joint_name] = {
                "desired": desired,
                "measured": measured,
                "error": error,
                "normalized_error": normalized_error,
            }
            if joint_name in self.arm_joints:
                max_normalized_error = max(
                    max_normalized_error,
                    normalized_error,
                )
            if joint_name in self.planar_base_joints:
                max_planar_normalized_error = max(
                    max_planar_normalized_error,
                    normalized_error,
                )

        info = {
            "dist": dist,
            "success": success,
            "collision": bool(obs.get("collision", False)),
            "collision_names": list(obs.get("collision_names", [])),
            "tf_ok": bool(obs.get("tf_ok", False)),
            "cmd_dict": safe_cmd_dict,
            "raw_cmd_dict": raw_cmd_dict,
            "safety_info": safety_info,
            "velocity_tracking": velocity_tracking,
            "max_arm_velocity_tracking_error": max_normalized_error,
            "max_planar_velocity_tracking_error": (
                max_planar_normalized_error
            ),
            "arm_chassis_guard": dict(self.last_chassis_guard_info),
            "published_joints": list(
                self.controlled_joints
                if publish_joints is None else publish_joints
            ),
        }

        return obs, reward, done, info

    def stop(self):
        """Stop every controller and make shutdown visible in the log."""
        if self._stop_complete:
            return
        rospy.loginfo("HRL stop requested: publishing zero velocity to all controllers")
        # Repeat with wall-clock sleeps so shutdown still works when simulated
        # time has already stopped.
        for _ in range(3):
            self.publish_zero_cmd()
            time.sleep(0.05)
        self._stop_complete = True
        rospy.loginfo("HRL stop complete: zero velocity command published 3 times")


if __name__ == "__main__":
    env = MobileArmReachEnv()

    obs = env.reset()
    rate = rospy.Rate(10)

    rospy.loginfo("Fixed joint order: %s", env.expected_joints)
    rospy.loginfo("Action dim: %d", env.action_dim)
    rospy.loginfo("Now publishing SAFE action commands to Gazebo velocity controllers.")

    try:
        while not rospy.is_shutdown():
            # =====================================================
            # 默认零动作，机器人不会主动运动。
            #
            # 如果要测试安全限制，可以临时改成：
            #
            # action = np.array([
            #     0.0,   # x
            #     0.0,   # y
            #     0.0,   # z
            #     0.0,   # sway
            #     0.2,   # joint1 正向，如果 joint1 已经在上限，会被安全过滤成 0
            #     0.0,   # joint2
            #     0.0,   # joint3
            #     0.0,   # joint4
            #     0.0,   # joint5
            #     0.0    # joint6
            # ])
            #
            # 如果 joint1 已经在上限，想测试它能不能退回来：
            #
            # action = np.array([
            #     0.0,   # x
            #     0.0,   # y
            #     0.0,   # z
            #     0.0,   # sway
            #    -0.2,   # joint1 反向，远离上限，应该允许
            #     0.0,   # joint2
            #     0.0,   # joint3
            #     0.0,   # joint4
            #     0.0,   # joint5
            #     0.0    # joint6
            # ])
            # =====================================================
            action = np.zeros(env.action_dim)

            obs, reward, done, info = env.step(action)

            rospy.loginfo(
                "obs_dim=%d base_dist=%.4f ee_dist=%.4f reward=%.4f done=%s success=%s joints=%d tf_ok=%s",
                len(obs["obs_vec"]),
                obs["base_to_target_dist"],
                obs["ee_to_target_dist"],
                reward,
                str(done),
                str(info["success"]),
                len(obs["joint_pos"]),
                str(obs["tf_ok"])
            )

            rospy.loginfo("joint_order=%s", env.expected_joints)
            rospy.loginfo("q=%s", np.round(obs["joint_pos"], 3))
            rospy.loginfo("dq=%s", np.round(obs["joint_vel"], 3))
            rospy.loginfo("q_margin=%s", np.round(obs["q_margin"], 3))

            raw_cmd_list = [info["raw_cmd_dict"][name] for name in env.expected_joints]
            safe_cmd_list = [info["cmd_dict"][name] for name in env.expected_joints]

            rospy.loginfo("cmd_order=%s", env.expected_joints)
            rospy.loginfo("raw_cmd_list=%s", np.round(raw_cmd_list, 3))
            rospy.loginfo("safe_cmd_list=%s", np.round(safe_cmd_list, 3))

            # 只打印被安全过滤的关节，避免日志太长
            blocked_joints = []
            for joint_name in env.expected_joints:
                safety = info["safety_info"][joint_name]
                if safety["reason"] != "safe":
                    blocked_joints.append(
                        "{}:{}".format(joint_name, safety["reason"])
                    )

            if len(blocked_joints) > 0:
                rospy.logwarn("safety_filter=%s", blocked_joints)

            rospy.loginfo("obs_vec=%s", np.round(obs["obs_vec"], 3))

            if done:
                rospy.loginfo("Episode done, reset environment.")
                obs = env.reset()

            rate.sleep()

    except rospy.ROSInterruptException:
        pass

    finally:
        rospy.loginfo("Stopping all controllers with zero velocity.")
        env.publish_zero_cmd()
