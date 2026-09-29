#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Gazebo adapter for repeatable first-stage planar-base training."""

from __future__ import print_function

import math
import threading
import time

import numpy as np
import rospy
import tf

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelConfiguration, SetModelState
from std_srvs.srv import Empty

from mobile_arm_env_check import MobileArmReachEnv
from training.planar_base_task import PlanarBaseTask
from training.tracked_base_kinematics import (
    unicycle_to_world_action,
    wrapped_angle_difference,
    world_velocity_to_body,
)


class PlanarBaseTrainingEnv(object):
    """Expose an 11-D observation and nonholonomic [v, omega] action."""

    OBS_DIM = PlanarBaseTask.OBS_DIM
    ACTION_DIM = PlanarBaseTask.ACTION_DIM
    FULL_ACTION_DIM = MobileArmReachEnv.ACTION_DIM

    DEFAULT_RESET_ARM_POSITIONS = (
        0.0,
        -0.30,
        0.03,
        0.0,
        0.03,
        0.0,
    )
    DEFAULT_RESET_BASE_POSITIONS = (
        0.0,
        0.0,
        0.5 * math.pi,
    )

    def __init__(self, base_env=None, init_ros_node=True, seed=None):
        if init_ros_node:
            rospy.init_node("planar_base_training_env", anonymous=True)

        self.base_env = base_env or MobileArmReachEnv(init_ros_node=False)
        if not self.base_env.enable_planar_base:
            raise RuntimeError(
                "PlanarBaseTrainingEnv requires ~enable_planar_base:=true"
            )

        self.base_frame = self.base_env.base_frame
        self.target_frame = rospy.get_param(
            "~target_frame",
            "planar_training_target_frame",
        )
        self.base_env.target_frame = self.target_frame
        self.model_name = rospy.get_param("~model_name", "mobile_arm")
        self.robot_description_param = rospy.get_param(
            "~robot_description_param",
            "robot_description",
        )
        self.reset_arm_positions = self._vector(
            rospy.get_param(
                "~reset_arm_positions",
                list(self.DEFAULT_RESET_ARM_POSITIONS),
            ),
            6,
            "reset_arm_positions",
        )
        self.target_min_distance = float(
            rospy.get_param("~target_min_distance", 0.40)
        )
        self.target_max_distance = float(
            rospy.get_param("~target_max_distance", 1.20)
        )
        self.target_z = float(rospy.get_param("~target_z", 0.40))
        self.target_exclusion_circles = self._parse_exclusion_circles(
            rospy.get_param("~target_exclusion_circles", [])
        )
        self.target_sample_attempts = int(
            rospy.get_param("~target_sample_attempts", 200)
        )
        self.reset_position_tolerance = float(
            rospy.get_param("~reset_position_tolerance", 0.01)
        )
        self.reset_velocity_tolerance = float(
            rospy.get_param("~reset_velocity_tolerance", 0.02)
        )
        self.reset_timeout = float(rospy.get_param("~reset_timeout", 7.0))
        self.base_action_scale = float(
            rospy.get_param("~base_action_scale", 1.0)
        )
        self.service_timeout = float(
            rospy.get_param("~gazebo_service_timeout", 10.0)
        )
        self._validate_parameters()

        self.task = PlanarBaseTask(
            success_threshold=rospy.get_param(
                "~base_success_threshold", 0.03
            ),
            max_steps=rospy.get_param("~max_steps", 300),
            progress_reward_scale=rospy.get_param(
                "~progress_reward_scale", 10.0
            ),
            success_reward=rospy.get_param("~success_reward", 2.0),
            collision_penalty=rospy.get_param(
                "~collision_penalty", 2.0
            ),
            time_penalty=rospy.get_param("~time_penalty", 0.01),
            action_penalty_scale=rospy.get_param(
                "~action_penalty_scale", 0.01
            ),
            smoothness_penalty_scale=rospy.get_param(
                "~smoothness_penalty_scale", 0.02
            ),
            collision_distance=rospy.get_param(
                "~collision_distance", 0.20
            ),
            scan_clip=rospy.get_param("~scan_clip", 10.0),
            target_clip=max(2.0, self.target_max_distance),
            base_velocity_scale=[
                self.base_env.action_max_vel["x"] * self.base_action_scale,
                self.base_env.action_max_vel["z"] * self.base_action_scale,
            ],
        )

        self._rng = np.random.RandomState()
        self.seed(seed)
        self._target_lock = threading.Lock()
        self._target_position = np.zeros(3, dtype=np.float64)
        self._target_ready = False
        self._target_broadcaster = tf.TransformBroadcaster()
        self._target_timer = rospy.Timer(
            rospy.Duration(0.02),
            self._target_timer_callback,
        )

        self._pause_physics = rospy.ServiceProxy(
            "/gazebo/pause_physics",
            Empty,
        )
        self._unpause_physics = rospy.ServiceProxy(
            "/gazebo/unpause_physics",
            Empty,
        )
        self._set_model_configuration = rospy.ServiceProxy(
            "/gazebo/set_model_configuration",
            SetModelConfiguration,
        )
        self._set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state",
            SetModelState,
        )

        self.episode_index = 0
        self.last_reset_info = {}
        self._previous_planar_position = None
        self._previous_planar_stamp = None

    def seed(self, seed=None):
        if seed is None:
            seed = int(time.time() * 1000000.0) % (2 ** 32 - 1)
        seed = int(seed)
        self._rng.seed(seed)
        self.random_seed = seed
        return [seed]

    def reset(self, target_position=None):
        """Reset the robot and use a sampled or explicitly supplied target."""
        self._publish_zero_velocity(repeat=3)
        self._reset_gazebo_joints()
        self._publish_zero_velocity(repeat=3)
        reset_error, reset_max_velocity = self._wait_for_reset_state()

        if target_position is None:
            target_position = self._sample_target()
        else:
            target_position = self._validate_target_position(
                target_position
            )
        self._set_target_position(target_position)
        self._publish_target_immediately(repeat=3)

        structured_observation = self.base_env.reset()
        self._reset_planar_odometry(structured_observation)
        structured_observation["observed_planar_body_velocity"] = (
            np.zeros(3, dtype=np.float32)
        )
        observation = self.task.reset(structured_observation)
        self.episode_index += 1
        self.last_reset_info = {
            "episode_index": self.episode_index,
            "random_seed": self.random_seed,
            "target_position_in_base": target_position.copy(),
            "reset_base_positions": np.asarray(
                self.DEFAULT_RESET_BASE_POSITIONS,
                dtype=np.float64,
            ),
            "reset_arm_positions": self.reset_arm_positions.copy(),
            "reset_position_error": reset_error,
            "reset_max_velocity": reset_max_velocity,
        }
        return observation

    def step(self, base_action):
        """Execute normalized tracked-base [linear, yaw-rate] action."""
        base_action = np.clip(
            self._vector(base_action, self.ACTION_DIM, "base_action"),
            -1.0,
            1.0,
        ).astype(np.float32)
        full_action = np.zeros(self.FULL_ACTION_DIM, dtype=np.float32)
        executed_base_action = base_action * self.base_action_scale
        q, _ = self.base_env.get_ordered_joint_state()
        yaw = float(q[2])
        # Enforce the unicycle/no-slip constraint at the only point where the
        # policy action becomes a physical command.  There is no lateral body
        # action: world y velocity appears only after the chassis has yawed.
        full_action[0:3] = unicycle_to_world_action(
            base_action,
            yaw,
            action_scale=self.base_action_scale,
        )

        structured_observation, _, _, execution_info = self.base_env.step(
            full_action,
            publish_joints=self.base_env.planar_base_joints,
        )
        (
            observed_world_velocity,
            body_velocity,
            observation_dt,
        ) = self._observe_planar_velocity(structured_observation)
        structured_observation["observed_planar_body_velocity"] = (
            body_velocity.astype(np.float32)
        )
        observation, reward, done, task_info = self.task.transition(
            structured_observation,
            base_action,
        )
        desired_world_velocity = np.asarray([
            execution_info["cmd_dict"][name]
            for name in self.base_env.planar_base_joints
        ], dtype=np.float64)
        velocity_scales = np.asarray([
            self.base_env.action_max_vel[name]
            for name in self.base_env.planar_base_joints
        ], dtype=np.float64)
        observed_tracking_error = float(np.max(
            np.abs(desired_world_velocity - observed_world_velocity)
            / velocity_scales
        ))
        if done:
            self.base_env.publish_zero_cmd()

        info = dict(execution_info)
        info.update(task_info)
        info.update({
            "episode_index": self.episode_index,
            "base_action": base_action.copy(),
            "executed_base_action": executed_base_action.copy(),
            "world_planar_action": full_action[0:3].copy(),
            "base_yaw": yaw,
            "measured_forward_velocity": float(
                body_velocity[0]
            ),
            "measured_lateral_velocity": float(
                body_velocity[1]
            ),
            "measured_yaw_rate": float(body_velocity[2]),
            "commanded_lateral_velocity": 0.0,
            "observed_world_planar_velocity": (
                observed_world_velocity.copy()
            ),
            "observed_planar_dt": float(observation_dt),
            "observed_planar_tracking_error": observed_tracking_error,
            "full_action": full_action.copy(),
            "compact_observation": observation.copy(),
            "target_position_in_base": self.target_position.copy(),
            "structured_observation": structured_observation,
        })
        return observation, reward, done, info

    def _reset_planar_odometry(self, observation):
        self._previous_planar_position = np.asarray(
            observation["joint_pos"][0:3],
            dtype=np.float64,
        ).copy()
        self._previous_planar_stamp = float(rospy.get_time())

    def _observe_planar_velocity(self, observation):
        current_position = np.asarray(
            observation["joint_pos"][0:3],
            dtype=np.float64,
        )
        current_stamp = float(rospy.get_time())
        if self._previous_planar_position is None:
            self._previous_planar_position = current_position.copy()
            self._previous_planar_stamp = current_stamp
            return (
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                0.0,
            )
        dt = current_stamp - float(self._previous_planar_stamp)
        if dt <= 1.0e-6:
            dt = 0.1
        yaw_delta = np.arctan2(
            np.sin(
                current_position[2]
                - self._previous_planar_position[2]
            ),
            np.cos(
                current_position[2]
                - self._previous_planar_position[2]
            ),
        )
        world_velocity = np.asarray([
            (current_position[0] - self._previous_planar_position[0]) / dt,
            (current_position[1] - self._previous_planar_position[1]) / dt,
            yaw_delta / dt,
        ], dtype=np.float64)
        midpoint_yaw = (
            self._previous_planar_position[2] + 0.5 * yaw_delta
        )
        body_velocity = world_velocity_to_body(
            world_velocity,
            midpoint_yaw,
        )
        self._previous_planar_position = current_position.copy()
        self._previous_planar_stamp = current_stamp
        return world_velocity, body_velocity, dt

    @property
    def target_position(self):
        with self._target_lock:
            return self._target_position.copy()

    def stop(self):
        if self._target_timer is not None:
            self._target_timer.shutdown()
            self._target_timer = None
        self.base_env.stop()

    close = stop

    def _reset_gazebo_joints(self):
        service_names = (
            "/gazebo/pause_physics",
            "/gazebo/unpause_physics",
            "/gazebo/set_model_configuration",
            "/gazebo/set_model_state",
        )
        for service_name in service_names:
            rospy.wait_for_service(
                service_name,
                timeout=self.service_timeout,
            )

        joint_names = (
            list(self.base_env.planar_base_joints)
            + list(self.base_env.arm_joints)
        )
        joint_positions = np.concatenate((
            np.asarray(
                self.DEFAULT_RESET_BASE_POSITIONS,
                dtype=np.float64,
            ),
            self.reset_arm_positions,
        ))
        paused = False
        try:
            self._pause_physics()
            paused = True
            response = self._set_model_configuration(
                model_name=self.model_name,
                urdf_param_name=self.robot_description_param,
                joint_names=joint_names,
                joint_positions=joint_positions.tolist(),
            )
            if not response.success:
                raise RuntimeError(
                    "Gazebo planar reset failed: {}".format(
                        response.status_message
                    )
                )

            # SetModelConfiguration changes joint positions but does not clear
            # the model's linear/angular velocity.  A velocity-controlled base
            # can therefore leave the reset pose immediately, especially after
            # an obstacle contact.  Reset the model pose and twist while physics
            # is paused so every episode starts from the same stationary state.
            model_state = ModelState()
            model_state.model_name = self.model_name
            model_state.pose.orientation.w = 1.0
            model_state.reference_frame = "world"
            state_response = self._set_model_state(model_state)
            if not state_response.success:
                raise RuntimeError(
                    "Gazebo planar model-state reset failed: {}".format(
                        state_response.status_message
                    )
                )
        finally:
            if paused:
                self._unpause_physics()

    def _wait_for_reset_state(self):
        # This environment trains only the planar x/y/yaw stage.  The arm reset is
        # still requested above, but its CAD/contact-dependent settling is a
        # separate concern and must not prevent a base episode from starting.
        # The policy still has only the two physically valid controls [v, w].
        expected = np.asarray(
            self.DEFAULT_RESET_BASE_POSITIONS,
            dtype=np.float64,
        )
        indices = [0, 1, 2]
        deadline = time.time() + self.reset_timeout
        last_position_error = float("inf")
        last_max_velocity = float("inf")
        while not rospy.is_shutdown() and time.time() < deadline:
            q, dq = self.base_env.get_ordered_joint_state()
            controlled_q = np.asarray(q[indices], dtype=np.float64)
            controlled_dq = np.asarray(dq[indices], dtype=np.float64)
            position_error = controlled_q - expected
            # Gazebo may represent the same revolute-joint pose one complete
            # turn away (for example -3*pi/2 instead of +pi/2).  The planar
            # reset contract is periodic in yaw, so compare its principal
            # angular difference rather than the raw joint coordinate.
            position_error[2] = wrapped_angle_difference(
                controlled_q[2],
                expected[2],
            )
            last_position_error = float(
                np.max(np.abs(position_error))
            )
            last_max_velocity = float(
                np.max(np.abs(controlled_dq))
            )
            if (
                    last_position_error <= self.reset_position_tolerance
                    and last_max_velocity <= self.reset_velocity_tolerance):
                return last_position_error, last_max_velocity
            self.base_env.publish_zero_cmd()
            time.sleep(0.02)

        raise RuntimeError(
            "Gazebo planar reset did not settle within {:.1f}s "
            "(position_error={:.5f}, max_velocity={:.5f})".format(
                self.reset_timeout,
                last_position_error,
                last_max_velocity,
            )
        )

    def _sample_target(self):
        for _ in range(self.target_sample_attempts):
            radius = self._rng.uniform(
                self.target_min_distance,
                self.target_max_distance,
            )
            angle = self._rng.uniform(-math.pi, math.pi)
            target = np.asarray([
                radius * math.cos(angle),
                radius * math.sin(angle),
                self.target_z,
            ], dtype=np.float64)
            if self._target_is_free(target):
                return target
        raise RuntimeError(
            "Could not sample a free planar target in {} attempts".format(
                self.target_sample_attempts
            )
        )

    def _validate_target_position(self, target_position):
        target = self._vector(target_position, 3, "target_position")
        radius = float(np.linalg.norm(target[0:2]))
        if not self.target_min_distance <= radius <= self.target_max_distance:
            raise ValueError(
                "target radius {:.3f} is outside [{:.3f}, {:.3f}]".format(
                    radius,
                    self.target_min_distance,
                    self.target_max_distance,
                )
            )
        if not self._target_is_free(target):
            raise ValueError("target lies inside an exclusion circle")
        return target.copy()

    def _target_is_free(self, target_position):
        xy = np.asarray(target_position[0:2], dtype=np.float64)
        for center_x, center_y, radius in self.target_exclusion_circles:
            center = np.asarray([center_x, center_y], dtype=np.float64)
            if float(np.linalg.norm(xy - center)) < radius:
                return False
        return True

    def _set_target_position(self, target_position):
        target_position = self._vector(
            target_position,
            3,
            "target_position",
        )
        with self._target_lock:
            self._target_position = target_position.copy()
            self._target_ready = True

    def _target_timer_callback(self, _event):
        self._broadcast_target()

    def _publish_target_immediately(self, repeat):
        for _ in range(int(repeat)):
            self._broadcast_target()
            time.sleep(0.02)

    def _broadcast_target(self):
        with self._target_lock:
            if not self._target_ready:
                return
            target = self._target_position.copy()
        self._target_broadcaster.sendTransform(
            target.tolist(),
            (0.0, 0.0, 0.0, 1.0),
            rospy.Time.now(),
            self.target_frame,
            self.base_frame,
        )

    def _publish_zero_velocity(self, repeat):
        for _ in range(int(repeat)):
            self.base_env.publish_zero_cmd()
            time.sleep(0.03)

    def _validate_parameters(self):
        if self.target_min_distance <= 0.0:
            raise ValueError("target_min_distance must be positive")
        if self.target_max_distance <= self.target_min_distance:
            raise ValueError(
                "target_max_distance must exceed target_min_distance"
            )
        if self.target_max_distance >= 1.80:
            raise ValueError(
                "target_max_distance must leave margin inside x/y limits"
            )
        if self.reset_position_tolerance <= 0.0:
            raise ValueError("reset_position_tolerance must be positive")
        if self.reset_velocity_tolerance <= 0.0:
            raise ValueError("reset_velocity_tolerance must be positive")
        if self.reset_timeout <= 0.0 or self.service_timeout <= 0.0:
            raise ValueError("reset timeouts must be positive")
        if not 0.0 < self.base_action_scale <= 1.0:
            raise ValueError("base_action_scale must be in (0, 1]")
        if self.target_sample_attempts <= 0:
            raise ValueError("target_sample_attempts must be positive")

    @staticmethod
    def _parse_exclusion_circles(value):
        vector = np.asarray(value, dtype=np.float64)
        if vector.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if vector.ndim == 1:
            if vector.size % 3 != 0:
                raise ValueError(
                    "target_exclusion_circles must contain x, y, radius triples"
                )
            vector = vector.reshape((-1, 3))
        if vector.ndim != 2 or vector.shape[1] != 3:
            raise ValueError(
                "target_exclusion_circles must have shape (N, 3)"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("target_exclusion_circles contains non-finite values")
        if np.any(vector[:, 2] <= 0.0):
            raise ValueError("target exclusion radii must be positive")
        return vector.copy()

    @staticmethod
    def _vector(value, expected_size, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (expected_size,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name,
                    expected_size,
                    vector.shape,
                )
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("{} contains non-finite values".format(name))
        return vector
