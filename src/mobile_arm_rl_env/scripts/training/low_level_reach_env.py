#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import time

import numpy as np
import rospy
import tf

from gazebo_msgs.srv import SetModelConfiguration
from std_srvs.srv import Empty

from hrl.kdl_jacobian import KDLJacobianProvider
from mobile_arm_env_check import MobileArmReachEnv


class LowLevelReachTrainingEnv(object):
    """Episode-based ARM_REACH environment for a learned low-level policy.

    The public observation remains the existing 46-dimensional vector.  A
    learner controls only the six physical arm joints; this adapter inserts
    four leading zeros before the action enters MobileArmReachEnv, so the
    established 10-dimensional execution and safety pipeline is unchanged.

    Every reset stops the velocity controllers, restores a configured Gazebo
    joint pose, samples a nearby Cartesian goal and starts a new episode.
    Target TF is published relative to the fixed base because base motion is
    deliberately outside the scope of low-level ARM_REACH training.
    """

    OBS_DIM = 46
    ACTION_DIM = 6
    FULL_ACTION_DIM = 10
    ARM_ACTION_SLICE = slice(4, 10)

    DEFAULT_RESET_ARM_POSITIONS = (
        0.0,
        -0.30,
        0.03,
        0.0,
        0.03,
        0.0,
    )
    DEFAULT_TARGET_JOINT_SPAN = (
        0.35,
        0.35,
        0.04,
        0.35,
        0.04,
        0.35,
    )

    def __init__(
            self,
            base_env=None,
            kinematics_provider=None,
            init_ros_node=True,
            seed=None):
        if init_ros_node:
            rospy.init_node("low_level_reach_training_env", anonymous=True)

        self.base_env = base_env or MobileArmReachEnv(init_ros_node=False)
        self.base_frame = self.base_env.base_frame
        self.ee_frame = self.base_env.ee_frame
        self.target_frame = rospy.get_param(
            "~target_frame",
            "training_target_frame",
        )
        self.base_env.target_frame = self.target_frame

        self.model_name = rospy.get_param("~model_name", "mobile_arm")
        self.robot_description_param = rospy.get_param(
            "~robot_description_param",
            "robot_description",
        )
        self.reset_arm_positions = self._as_vector(
            rospy.get_param(
                "~reset_arm_positions",
                list(self.DEFAULT_RESET_ARM_POSITIONS),
            ),
            self.ACTION_DIM,
            "reset_arm_positions",
        )
        self.target_joint_span = self._as_vector(
            rospy.get_param(
                "~target_joint_span",
                list(self.DEFAULT_TARGET_JOINT_SPAN),
            ),
            self.ACTION_DIM,
            "target_joint_span",
        )
        description_param = "/" + self.robot_description_param.lstrip("/")
        self.kinematics_provider = (
            kinematics_provider
            if kinematics_provider is not None
            else KDLJacobianProvider(
                base_frame=self.base_frame,
                ee_frame=self.ee_frame,
                state_joint_names=self.base_env.expected_joints,
                controlled_joint_names=self.base_env.arm_joints,
                robot_description=description_param,
            )
        )

        self.target_min_distance = float(
            rospy.get_param("~target_min_distance", 0.08)
        )
        self.target_max_distance = float(
            rospy.get_param("~target_max_distance", 0.25)
        )
        self.target_min_z = float(rospy.get_param("~target_min_z", 0.10))
        self.target_max_z = float(rospy.get_param("~target_max_z", 1.20))
        self.target_min_chassis_clearance = float(
            rospy.get_param("~target_min_chassis_clearance", 0.015)
        )
        self.reset_position_tolerance = float(
            rospy.get_param("~reset_position_tolerance", 0.02)
        )
        self.reset_velocity_tolerance = float(
            rospy.get_param("~reset_velocity_tolerance", 0.02)
        )
        self.reset_timeout = float(rospy.get_param("~reset_timeout", 5.0))
        self.service_timeout = float(
            rospy.get_param("~gazebo_service_timeout", 10.0)
        )

        self._validate_parameters()
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

        self.episode_index = 0
        self.last_reset_info = {}

    def seed(self, seed=None):
        if seed is None:
            seed = int(time.time() * 1000000.0) % (2 ** 32 - 1)
        seed = int(seed)
        self._rng.seed(seed)
        self.random_seed = seed
        return [seed]

    def reset(self):
        """Restore Gazebo state, generate a goal and return a flat vector."""
        self._publish_zero_velocity(repeat=3)
        self._reset_gazebo_joints()
        self._publish_zero_velocity(repeat=3)
        self._wait_for_reset_state()

        ee_position = self._current_end_effector_position()
        (
            target_position,
            target_arm_positions,
            target_chassis_clearance,
        ) = self._sample_target(ee_position)
        self._set_target_position(target_position)
        self._publish_target_immediately(repeat=3)

        observation = self.base_env.reset()
        self.episode_index += 1
        self.last_reset_info = {
            "episode_index": self.episode_index,
            "random_seed": self.random_seed,
            "reset_arm_positions": self.reset_arm_positions.copy(),
            "target_position_in_base": target_position.copy(),
            "target_offset_from_ee": (
                target_position - ee_position
            ).copy(),
            "target_arm_positions": target_arm_positions.copy(),
            "target_chassis_clearance": target_chassis_clearance,
        }
        return self._flat_observation(observation)

    def step(self, arm_action):
        """Execute one normalized six-joint action through the safety layer."""
        arm_action = self._as_vector(
            arm_action,
            self.ACTION_DIM,
            "arm_action",
        )
        full_action = np.zeros(self.FULL_ACTION_DIM, dtype=np.float32)
        full_action[self.ARM_ACTION_SLICE] = np.clip(
            arm_action,
            -1.0,
            1.0,
        )

        observation, reward, done, info = self.base_env.step(full_action)
        info = dict(info)
        info.update({
            "episode_index": self.episode_index,
            "arm_action": arm_action.copy(),
            "full_action": full_action.copy(),
            "target_position_in_base": self.target_position.copy(),
        })
        return self._flat_observation(observation), reward, done, info

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
        for service_name in (
                "/gazebo/pause_physics",
                "/gazebo/unpause_physics",
                "/gazebo/set_model_configuration"):
            rospy.wait_for_service(service_name, timeout=self.service_timeout)

        paused = False
        try:
            self._pause_physics()
            paused = True
            response = self._set_model_configuration(
                model_name=self.model_name,
                urdf_param_name=self.robot_description_param,
                joint_names=list(self.base_env.arm_joints),
                joint_positions=self.reset_arm_positions.tolist(),
            )
            if not response.success:
                raise RuntimeError(
                    "Gazebo joint reset failed: {}".format(
                        response.status_message
                    )
                )
        finally:
            if paused:
                self._unpause_physics()

    def _wait_for_reset_state(self):
        deadline = time.time() + self.reset_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            q, dq = self.base_env.get_ordered_joint_state()
            arm_q = np.asarray(q[4:10], dtype=np.float64)
            arm_dq = np.asarray(dq[4:10], dtype=np.float64)
            position_error = float(
                np.max(np.abs(arm_q - self.reset_arm_positions))
            )
            maximum_velocity = float(np.max(np.abs(arm_dq)))
            if (
                    position_error <= self.reset_position_tolerance
                    and maximum_velocity <= self.reset_velocity_tolerance):
                return
            self.base_env.publish_zero_cmd()
            time.sleep(0.02)

        raise RuntimeError(
            "Gazebo reset did not settle within {:.1f}s".format(
                self.reset_timeout
            )
        )

    def _current_end_effector_position(self):
        position, _, ok = self.base_env.get_tf_error(
            self.base_frame,
            self.ee_frame,
        )
        if not ok:
            raise RuntimeError(
                "Cannot sample a target without {} -> {} TF".format(
                    self.base_frame,
                    self.ee_frame,
                )
            )
        return np.asarray(position, dtype=np.float64)

    def _sample_target(self, end_effector_position):
        for _ in range(100):
            target_arm_positions = self._sample_safe_arm_positions()
            target_chassis_clearance = self._arm_chassis_clearance(
                target_arm_positions
            )
            if (
                    target_chassis_clearance
                    < self.target_min_chassis_clearance):
                continue
            full_joint_positions = np.zeros(
                self.FULL_ACTION_DIM,
                dtype=np.float64,
            )
            full_joint_positions[self.ARM_ACTION_SLICE] = (
                target_arm_positions
            )
            candidate = self.kinematics_provider.end_effector_position(
                full_joint_positions
            )
            distance = float(
                np.linalg.norm(candidate - end_effector_position)
            )
            if (
                    self.target_min_distance <= distance
                    <= self.target_max_distance
                    and self.target_min_z <= candidate[2]
                    <= self.target_max_z):
                return (
                    candidate,
                    target_arm_positions,
                    target_chassis_clearance,
                )
        raise RuntimeError(
            "Unable to sample a reachable target with the configured bounds"
        )

    def _sample_safe_arm_positions(self):
        positions = np.empty(self.ACTION_DIM, dtype=np.float64)
        for index, joint_name in enumerate(self.base_env.arm_joints):
            lower, upper = self.base_env.joint_limits[joint_name]
            safety_margin = self.base_env.limit_soft_margin[joint_name]
            safe_lower = lower + safety_margin
            safe_upper = upper - safety_margin
            center = float(self.reset_arm_positions[index])
            span = float(self.target_joint_span[index])
            sample_lower = max(safe_lower, center - span)
            sample_upper = min(safe_upper, center + span)
            if sample_lower >= sample_upper:
                raise ValueError(
                    "No target sampling range remains for {}".format(
                        joint_name
                    )
                )
            positions[index] = self._rng.uniform(
                sample_lower,
                sample_upper,
            )
        return positions

    def _set_target_position(self, target_position):
        target_position = self._as_vector(
            target_position,
            3,
            "target_position",
        ).astype(np.float64)
        with self._target_lock:
            self._target_position = target_position
            self._target_ready = True

    def _arm_chassis_clearance(self, arm_positions):
        guard = getattr(self.base_env, "arm_chassis_guard", None)
        if guard is None:
            return float("inf")
        clearance, _ = guard.minimum_clearance(arm_positions)
        return float(clearance)

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
            position = self._target_position.copy()
        self._target_broadcaster.sendTransform(
            tuple(position.tolist()),
            (0.0, 0.0, 0.0, 1.0),
            rospy.Time.now(),
            self.target_frame,
            self.base_frame,
        )

    def _publish_zero_velocity(self, repeat):
        for _ in range(int(repeat)):
            self.base_env.publish_zero_cmd()
            time.sleep(0.02)

    def _validate_parameters(self):
        if self.target_min_distance <= 0.0:
            raise ValueError("target_min_distance must be positive")
        if self.target_max_distance <= self.target_min_distance:
            raise ValueError(
                "target_max_distance must exceed target_min_distance"
            )
        if self.target_max_z <= self.target_min_z:
            raise ValueError("target_max_z must exceed target_min_z")
        if self.target_min_chassis_clearance < 0.0:
            raise ValueError(
                "target_min_chassis_clearance must be non-negative"
            )
        if self.reset_position_tolerance <= 0.0:
            raise ValueError("reset_position_tolerance must be positive")
        if self.reset_velocity_tolerance <= 0.0:
            raise ValueError("reset_velocity_tolerance must be positive")
        if self.reset_timeout <= 0.0:
            raise ValueError("reset_timeout must be positive")
        if self.service_timeout <= 0.0:
            raise ValueError("gazebo_service_timeout must be positive")
        if np.any(self.target_joint_span <= 0.0):
            raise ValueError("target_joint_span values must be positive")

        for index, joint_name in enumerate(self.base_env.arm_joints):
            lower, upper = self.base_env.joint_limits[joint_name]
            position = self.reset_arm_positions[index]
            if not lower <= position <= upper:
                raise ValueError(
                    "reset position for {} is outside [{}, {}]".format(
                        joint_name,
                        lower,
                        upper,
                    )
                )

        guard = getattr(self.base_env, "arm_chassis_guard", None)
        if guard is not None:
            reset_clearance, reset_link = guard.minimum_clearance(
                self.reset_arm_positions
            )
            if reset_clearance < guard.hard_clearance:
                raise ValueError(
                    "reset arm pose interferes with chassis: "
                    "clearance={:.4f} link={}".format(
                        reset_clearance,
                        reset_link,
                    )
                )
            if (
                    self.target_min_chassis_clearance
                    < guard.hard_clearance):
                raise ValueError(
                    "target_min_chassis_clearance must not be below "
                    "the chassis hard clearance"
                )

    @classmethod
    def _flat_observation(cls, observation):
        vector = observation[
            "obs_vec"
        ] if isinstance(observation, dict) else observation
        return cls._as_vector(vector, cls.OBS_DIM, "observation")

    @staticmethod
    def _as_vector(value, expected_size, name):
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (expected_size,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name,
                    expected_size,
                    vector.shape,
                )
            )
        return vector
