#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Hold each arm joint position only while its velocity input is idle.

The existing public command topics are preserved.  This node observes those
topics and publishes a small corrective velocity only after no external
publisher has sent a command for ``command_timeout`` seconds.  Messages
published by this node are identified by their ROS caller id and do not count
as external activity.
"""

from __future__ import print_function

import sys
import threading
import time

import numpy as np
import rospy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger, TriggerResponse


ARM_JOINTS = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)
PRISMATIC_JOINTS = frozenset(("joint3", "joint5"))


class JointIdleHoldState(object):
    """Pure per-joint idle/hold state machine used by the ROS node."""

    def __init__(
            self,
            command_timeout,
            position_gain,
            velocity_gain,
            maximum_velocity,
            position_deadband=1.0e-5,
            command_deadband=1.0e-8):
        self.command_timeout = float(command_timeout)
        self.position_gain = float(position_gain)
        self.velocity_gain = float(velocity_gain)
        self.maximum_velocity = float(maximum_velocity)
        self.position_deadband = float(position_deadband)
        self.command_deadband = float(command_deadband)
        self.last_external_time = float("-inf")
        self.external_active = False
        self.target_position = None
        self.holding = False
        self._validate()

    def note_external_command(self, now, value=None):
        """Track active motion commands without disabling hold on zeros.

        Base-only policies publish a ten-dimensional command whose six arm
        entries are zero.  Treating those zeros as activity disabled this
        controller for the entire base episode and allowed inertial arm drift.
        A non-zero command still takes precedence immediately; an explicit
        zero releases it and captures the stopped position on the next cycle.
        """
        if value is None or abs(float(value)) > self.command_deadband:
            self.last_external_time = float(now)
            self.external_active = True
            self.holding = False
            return
        if self.external_active:
            self.last_external_time = float("-inf")
            self.external_active = False
            self.holding = False

    def set_hold_target(self, position):
        position = float(position)
        if not np.isfinite(position):
            raise ValueError("hold target must be finite")
        self.target_position = position
        self.holding = True
        self.external_active = False
        self.last_external_time = float("-inf")

    def command(self, position, velocity, now):
        position = float(position)
        velocity = float(velocity)
        now = float(now)
        if now - self.last_external_time <= self.command_timeout:
            return None, False
        self.external_active = False

        entered_hold = False
        if not self.holding or self.target_position is None:
            self.target_position = position
            self.holding = True
            entered_hold = True

        error = self.target_position - position
        if abs(error) <= self.position_deadband:
            error = 0.0
        output = self.position_gain * error - self.velocity_gain * velocity
        output = float(np.clip(
            output,
            -self.maximum_velocity,
            self.maximum_velocity,
        ))
        return output, entered_hold

    def _validate(self):
        if self.command_timeout <= 0.0:
            raise ValueError("command_timeout must be positive")
        if self.position_gain < 0.0 or self.velocity_gain < 0.0:
            raise ValueError("hold gains must be non-negative")
        if self.maximum_velocity <= 0.0:
            raise ValueError("maximum_velocity must be positive")
        if self.position_deadband < 0.0:
            raise ValueError("position_deadband must be non-negative")
        if self.command_deadband < 0.0:
            raise ValueError("command_deadband must be non-negative")


class ArmIdlePositionHoldNode(object):
    def __init__(self):
        self.command_timeout = float(
            rospy.get_param("~command_timeout", 0.30)
        )
        self.control_rate = float(rospy.get_param("~control_rate", 50.0))
        self.position_gain = float(
            rospy.get_param("~position_gain", 2.0)
        )
        self.velocity_gain = float(
            rospy.get_param("~velocity_gain", 0.20)
        )
        self.max_revolute_velocity = float(
            rospy.get_param("~max_revolute_velocity", 0.05)
        )
        self.max_prismatic_velocity = float(
            rospy.get_param("~max_prismatic_velocity", 0.01)
        )
        configured_positions = rospy.get_param("~initial_positions", [])
        if configured_positions:
            if len(configured_positions) != len(ARM_JOINTS):
                raise ValueError(
                    "initial_positions must contain six values"
                )
            self.initial_positions = dict(zip(
                ARM_JOINTS,
                [float(value) for value in configured_positions],
            ))
        else:
            self.initial_positions = {}
        if self.control_rate <= 0.0:
            raise ValueError("control_rate must be positive")

        self.node_name = rospy.get_name()
        self.lock = threading.RLock()
        self.positions = {}
        self.velocities = {}
        self.state_received = threading.Event()
        self.hold_states = {}
        self.publishers = {}
        self.subscribers = []

        for joint_name in ARM_JOINTS:
            maximum_velocity = (
                self.max_prismatic_velocity
                if joint_name in PRISMATIC_JOINTS
                else self.max_revolute_velocity
            )
            self.hold_states[joint_name] = JointIdleHoldState(
                command_timeout=self.command_timeout,
                position_gain=self.position_gain,
                velocity_gain=self.velocity_gain,
                maximum_velocity=maximum_velocity,
            )
            if joint_name in self.initial_positions:
                self.hold_states[joint_name].set_hold_target(
                    self.initial_positions[joint_name]
                )
            topic = "/{}_velocity_controller/command".format(joint_name)
            self.publishers[joint_name] = rospy.Publisher(
                topic,
                Float64,
                queue_size=1,
            )
            self.subscribers.append(rospy.Subscriber(
                topic,
                Float64,
                self._command_callback,
                callback_args=joint_name,
                queue_size=20,
            ))

        rospy.Subscriber(
            "/joint_states",
            JointState,
            self._joint_state_callback,
            queue_size=20,
        )
        self.reset_targets_service = rospy.Service(
            "~reset_targets",
            Trigger,
            self._reset_targets_callback,
        )
        rospy.on_shutdown(self.stop)

    def _joint_state_callback(self, message):
        with self.lock:
            self.positions.update(dict(zip(message.name, message.position)))
            self.velocities.update(dict(zip(message.name, message.velocity)))
            if all(name in self.positions for name in ARM_JOINTS):
                self.state_received.set()

    def _command_callback(self, message, joint_name):
        connection = getattr(message, "_connection_header", {}) or {}
        caller_id = str(connection.get("callerid", ""))
        if caller_id == self.node_name:
            return
        with self.lock:
            self.hold_states[joint_name].note_external_command(
                time.time(),
                message.data,
            )

    def _reset_targets_callback(self, _request):
        """Restore configured hold targets after an external Gazebo reset.

        During a normal episode the state machine captures the position at
        which an active velocity command stops.  That is correct for idle
        holding, but the captured target belongs to the previous episode.
        A simulator reset must therefore explicitly replace it before physics
        resumes, otherwise the hold loop pulls the arm back toward the old
        terminal pose.
        """
        with self.lock:
            targets = {}
            for joint_name in ARM_JOINTS:
                if joint_name in self.initial_positions:
                    targets[joint_name] = self.initial_positions[joint_name]
                elif joint_name in self.positions:
                    targets[joint_name] = self.positions[joint_name]
                else:
                    return TriggerResponse(
                        success=False,
                        message=(
                            "joint state unavailable for {}".format(
                                joint_name
                            )
                        ),
                    )

            for joint_name, position in targets.items():
                self.hold_states[joint_name].set_hold_target(position)

        rospy.loginfo(
            "Arm idle hold targets reset: %s",
            [
                round(float(targets[name]), 6)
                for name in ARM_JOINTS
            ],
        )
        return TriggerResponse(
            success=True,
            message="restored {} arm hold targets".format(len(targets)),
        )

    def run(self):
        if not self.state_received.wait(30.0):
            raise RuntimeError(
                "arm idle hold did not receive all six joint states"
            )
        rospy.loginfo(
            "Arm idle position hold ready: timeout=%.2fs rate=%.1fHz "
            "configured_targets=%s",
            self.command_timeout,
            self.control_rate,
            [
                self.initial_positions.get(name)
                for name in ARM_JOINTS
            ] if self.initial_positions else "capture_current",
        )
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            now = time.time()
            commands = {}
            entered = []
            with self.lock:
                for joint_name in ARM_JOINTS:
                    if joint_name not in self.positions:
                        continue
                    command, entered_hold = self.hold_states[
                        joint_name
                    ].command(
                        self.positions[joint_name],
                        self.velocities.get(joint_name, 0.0),
                        now,
                    )
                    if command is not None:
                        commands[joint_name] = command
                    if entered_hold:
                        entered.append(joint_name)

            for joint_name, command in commands.items():
                self.publishers[joint_name].publish(Float64(command))
            if entered:
                rospy.loginfo(
                    "Arm idle hold captured joints=%s",
                    entered,
                )
            rate.sleep()

    def stop(self):
        # Leave every velocity controller with an explicit safe command.
        for _ in range(3):
            for publisher in self.publishers.values():
                publisher.publish(Float64(0.0))
            time.sleep(0.02)


def main():
    rospy.init_node("arm_idle_position_hold")
    try:
        ArmIdlePositionHoldNode().run()
    except (RuntimeError, ValueError, rospy.ROSException) as error:
        rospy.logerr("Arm idle position hold failed: %s", str(error))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
