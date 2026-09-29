#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Use move_base/DWA as an advisory teacher for a learned tracked-base policy."""

from __future__ import print_function

import threading
import time

import actionlib
import numpy as np
import rospy

from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_srvs.srv import Empty
from tf.transformations import quaternion_from_euler


TERMINAL_GOAL_STATES = (
    GoalStatus.PREEMPTED,
    GoalStatus.SUCCEEDED,
    GoalStatus.ABORTED,
    GoalStatus.REJECTED,
    GoalStatus.RECALLED,
    GoalStatus.LOST,
)


class DWAPlanarTeacher(object):
    """Return normalized [linear velocity, yaw rate] DWA demonstrations.

    The DWA command topic is deliberately not connected to the physical
    controller during training.  The learner drives Gazebo, while DWA
    replans from the learner's actual state and supplies an auxiliary target
    action for the PPO loss.
    """

    ACTION_DIM = 2

    def __init__(
            self,
            cmd_vel_topic="/dwa_teacher_cmd_vel",
            move_base_action="/move_base",
            global_frame="base_link",
            maximum_linear_speed=0.05,
            maximum_yaw_rate=0.125,
            command_timeout=0.50,
            server_timeout=30.0,
            clear_costmaps_service="/move_base/clear_costmaps"):
        self.cmd_vel_topic = str(cmd_vel_topic)
        self.global_frame = str(global_frame)
        self.maximum_linear_speed = float(maximum_linear_speed)
        self.maximum_yaw_rate = float(maximum_yaw_rate)
        self.command_timeout = float(command_timeout)
        self.server_timeout = float(server_timeout)
        if (
                self.maximum_linear_speed <= 0.0
                or self.maximum_yaw_rate <= 0.0
                or self.command_timeout <= 0.0
                or self.server_timeout <= 0.0):
            raise ValueError("DWA teacher scales and timeouts must be positive")

        self._lock = threading.Lock()
        self._latest_twist = None
        self._latest_wall_time = None
        self._goal_index = 0
        self._client = actionlib.SimpleActionClient(
            str(move_base_action),
            MoveBaseAction,
        )
        self._clear_costmaps = rospy.ServiceProxy(
            str(clear_costmaps_service),
            Empty,
        )
        rospy.Subscriber(
            self.cmd_vel_topic,
            Twist,
            self._twist_callback,
            queue_size=20,
        )

    def wait_until_ready(self):
        if not self._client.wait_for_server(
                rospy.Duration(self.server_timeout)):
            raise RuntimeError(
                "DWA teacher could not reach move_base within {:.1f}s".format(
                    self.server_timeout
                )
            )

    def start_goal(self, target_xy, target_yaw=0.0):
        target_xy = np.asarray(target_xy, dtype=np.float64)
        if target_xy.shape != (2,) or not np.all(np.isfinite(target_xy)):
            raise ValueError("DWA target_xy must be a finite 2-vector")
        target_yaw = float(target_yaw)
        if not np.isfinite(target_yaw):
            raise ValueError("DWA target yaw must be finite")

        self.wait_until_ready()
        self.cancel_goal()
        try:
            rospy.wait_for_service(
                self._clear_costmaps.resolved_name,
                timeout=min(self.server_timeout, 5.0),
            )
            self._clear_costmaps()
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logwarn(
                "DWA teacher could not clear costmaps: %s",
                str(error),
            )

        quaternion = quaternion_from_euler(0.0, 0.0, target_yaw)
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self.global_frame
        goal.target_pose.pose.position.x = float(target_xy[0])
        goal.target_pose.pose.position.y = float(target_xy[1])
        goal.target_pose.pose.orientation.x = float(quaternion[0])
        goal.target_pose.pose.orientation.y = float(quaternion[1])
        goal.target_pose.pose.orientation.z = float(quaternion[2])
        goal.target_pose.pose.orientation.w = float(quaternion[3])
        with self._lock:
            self._latest_twist = None
            self._latest_wall_time = None
            self._goal_index += 1
        self._client.send_goal(goal)
        return self._goal_index

    def action(self, wait_timeout=0.0):
        deadline = time.time() + max(float(wait_timeout), 0.0)
        while not rospy.is_shutdown():
            action, available, diagnostics = self._current_action()
            if available or time.time() >= deadline:
                return action, available, diagnostics
            if diagnostics["goal_terminal"]:
                return action, False, diagnostics
            time.sleep(0.01)
        return (
            np.zeros(self.ACTION_DIM, dtype=np.float32),
            False,
            self.diagnostics(),
        )

    def cancel_goal(self):
        self._client.cancel_all_goals()
        with self._lock:
            self._latest_twist = None
            self._latest_wall_time = None

    def diagnostics(self):
        unused_action, unused_available, diagnostics = (
            self._current_action()
        )
        return diagnostics

    def _twist_callback(self, message):
        values = np.asarray([
            message.linear.x,
            message.linear.y,
            message.angular.z,
        ], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            rospy.logerr_throttle(
                1.0,
                "DWA teacher rejected non-finite cmd_vel",
            )
            return
        with self._lock:
            self._latest_twist = values
            self._latest_wall_time = time.time()

    def _current_action(self):
        with self._lock:
            twist = (
                None
                if self._latest_twist is None
                else self._latest_twist.copy()
            )
            command_time = self._latest_wall_time
            goal_index = self._goal_index
        wall_age = (
            float("inf")
            if command_time is None
            else max(time.time() - command_time, 0.0)
        )
        status = int(self._client.get_state())
        terminal = bool(status in TERMINAL_GOAL_STATES)
        available = bool(
            twist is not None
            and wall_age <= self.command_timeout
            and not terminal
        )
        if available:
            action = self.normalize_command(
                twist[0],
                twist[2],
                self.maximum_linear_speed,
                self.maximum_yaw_rate,
            )
            lateral_velocity = float(twist[1])
        else:
            action = np.zeros(self.ACTION_DIM, dtype=np.float32)
            lateral_velocity = 0.0 if twist is None else float(twist[1])
        diagnostics = {
            "teacher_available": available,
            "goal_index": int(goal_index),
            "goal_status": status,
            "goal_terminal": terminal,
            "command_age": wall_age,
            "lateral_velocity": lateral_velocity,
            "action": action.copy(),
            "action_semantics": "normalized_linear_velocity_yaw_rate",
        }
        return action, available, diagnostics

    @staticmethod
    def normalize_command(
            linear_velocity,
            yaw_rate,
            maximum_linear_speed,
            maximum_yaw_rate):
        maximum_linear_speed = float(maximum_linear_speed)
        maximum_yaw_rate = float(maximum_yaw_rate)
        if maximum_linear_speed <= 0.0 or maximum_yaw_rate <= 0.0:
            raise ValueError("DWA command scales must be positive")
        values = np.asarray(
            [linear_velocity, yaw_rate],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("DWA command must be finite")
        return np.clip(
            values / np.asarray([
                maximum_linear_speed,
                maximum_yaw_rate,
            ], dtype=np.float64),
            -1.0,
            1.0,
        ).astype(np.float32)

