#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Obstacle-free environment for coordinated tracked-base and arm control."""

from __future__ import division

import math
import time

import numpy as np
import rospy

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Trigger

from hrl.fused_low_level import (
    CoordinatedRuleTeacher,
    FusedActionAdapter,
    FusedLowLevelState,
    TRACKED_FORWARD_YAW_OFFSET,
)
from training.low_level_reach_env import LowLevelReachTrainingEnv


class FusedLowLevelTrainingEnv(object):
    """Expose a 66-D observation and normalized 8-D coordinated action."""

    OBS_DIM = FusedLowLevelState.INPUT_DIM
    ACTION_DIM = FusedLowLevelState.ACTION_DIM
    # Keep the executor action dimension available to residual subclasses
    # whose public policy action is intentionally smaller.
    FUSED_ACTION_DIM = FusedLowLevelState.ACTION_DIM
    SENSOR_DIM = FusedLowLevelState.SENSOR_DIM

    DEFAULT_RESET_BASE_POSITIONS = (0.0, 0.0, 0.5 * math.pi)

    def __init__(
            self,
            base_env=None,
            init_ros_node=True,
            seed=None,
            enable_teacher=True):
        if init_ros_node:
            rospy.init_node("fused_low_training_env", anonymous=True)

        self.reach_env = LowLevelReachTrainingEnv(
            base_env=base_env,
            init_ros_node=False,
            seed=seed,
        )
        self.base_env = self.reach_env.base_env
        if not self.base_env.enable_planar_base:
            raise RuntimeError(
                "FusedLowLevelTrainingEnv requires "
                "~enable_planar_base:=true"
            )

        self.state_builder = FusedLowLevelState()
        self.action_adapter = FusedActionAdapter(
            linear_action_scale=rospy.get_param(
                "~base_linear_action_scale",
                0.25,
            ),
            yaw_action_scale=rospy.get_param(
                "~base_yaw_action_scale",
                0.25,
            ),
        )
        self.reset_base_positions = self._vector(
            rospy.get_param(
                "~reset_base_positions",
                list(self.DEFAULT_RESET_BASE_POSITIONS),
            ),
            3,
            "reset_base_positions",
        )
        self.base_target_min_distance = float(
            rospy.get_param("~base_target_min_distance", 0.30)
        )
        self.base_target_max_distance = float(
            rospy.get_param("~base_target_max_distance", 0.55)
        )
        self.base_target_max_lateral = float(
            rospy.get_param("~base_target_max_lateral", 0.15)
        )
        self.target_sample_attempts = int(
            rospy.get_param("~target_sample_attempts", 100)
        )
        self.max_steps = int(rospy.get_param("~max_steps", 500))
        self.default_max_steps = self.max_steps
        self.success_threshold = float(
            rospy.get_param("~success_threshold", 0.05)
        )
        self.reset_settle_window = float(
            rospy.get_param("~reset_settle_window", 0.50)
        )
        self.reset_drift_tolerance = float(
            rospy.get_param("~reset_drift_tolerance", 0.002)
        )
        self.reset_attempts = int(
            rospy.get_param("~reset_attempts", 3)
        )
        # A completed rollout can leave Gazebo joints moving when the next
        # reset request arrives.  Stop the velocity controllers in real time
        # before pausing physics; otherwise set_model_configuration can be
        # immediately overwritten by a command that was still in flight.
        self.reset_pre_brake_time = float(
            rospy.get_param("~reset_pre_brake_time", 0.40)
        )
        # Reapply the joint configuration while physics is paused.  Gazebo's
        # model-state service clears the model twist, but is not a substitute
        # for setting the internal planar/arm joint coordinates.
        self.reset_configuration_repeats = int(
            rospy.get_param("~reset_configuration_repeats", 2)
        )
        self.idle_hold_reset_service = str(rospy.get_param(
            "~idle_hold_reset_service",
            "/arm_idle_position_hold/reset_targets",
        ))
        self.base_env.max_steps = self.max_steps
        self.base_env.success_threshold = self.success_threshold

        self._set_model_state = rospy.ServiceProxy(
            "/gazebo/set_model_state",
            SetModelState,
        )
        self._reset_idle_hold_targets = rospy.ServiceProxy(
            self.idle_hold_reset_service,
            Trigger,
        )
        self._validate_parameters()

        self.enable_teacher = bool(enable_teacher)
        self.teacher = None
        if self.enable_teacher:
            teacher_base_stop_distance = float(rospy.get_param(
                "~teacher_base_stop_distance",
                0.04,
            ))
            self.teacher = CoordinatedRuleTeacher(
                jacobian_provider=self.reach_env.kinematics_provider,
                base_gain=rospy.get_param("~teacher_base_gain", 2.5),
                yaw_gain=rospy.get_param("~teacher_yaw_gain", 1.5),
                arm_gain=rospy.get_param("~teacher_arm_gain", 1.0),
                arm_start_distance=rospy.get_param(
                    "~teacher_arm_start_distance",
                    0.30,
                ),
                base_stop_distance=teacher_base_stop_distance,
                subgoal_base_stop_distance=rospy.get_param(
                    "~teacher_subgoal_base_stop_distance",
                    min(0.02, teacher_base_stop_distance),
                ),
                heading_slow_angle=rospy.get_param(
                    "~teacher_heading_slow_angle",
                    0.70,
                ),
                max_cartesian_speed=rospy.get_param(
                    "~teacher_max_cartesian_speed",
                    0.05,
                ),
                dls_damping=rospy.get_param(
                    "~teacher_dls_damping",
                    0.03,
                ),
                joint_limit_avoidance_gain=rospy.get_param(
                    "~teacher_joint_limit_avoidance_gain",
                    0.0,
                ),
                joint_limit_avoidance_activation=rospy.get_param(
                    "~teacher_joint_limit_avoidance_activation",
                    1.5,
                ),
                arm_joint_limits=np.asarray([
                    self.base_env.joint_limits[name]
                    for name in self.base_env.arm_joints
                ], dtype=np.float64),
                arm_soft_margins=np.asarray([
                    self.base_env.limit_soft_margin[name]
                    for name in self.base_env.arm_joints
                ], dtype=np.float64),
            )

        self.base_goal_xy = np.zeros(2, dtype=np.float64)
        self.target_arm_positions = np.zeros(6, dtype=np.float64)
        self.episode_index = 0
        self.last_reset_info = {}
        self.last_sensor_observation = None
        self.last_structured_observation = None

    def reset(self, scenario=None, max_steps=None):
        self.max_steps = (
            self.default_max_steps
            if max_steps is None else int(max_steps)
        )
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.base_env.max_steps = self.max_steps
        self._reset_and_wait()

        q, _ = self.base_env.get_ordered_joint_state()
        q = np.asarray(q, dtype=np.float64)
        if scenario is None:
            (
                target_position,
                target_joint_positions,
                base_goal_xy,
                base_displacement_body,
                target_chassis_clearance,
            ) = self._sample_coordinated_target(q)
            scenario_id = None
            scenario_category = "random"
        else:
            (
                target_position,
                target_joint_positions,
                base_goal_xy,
                base_displacement_body,
                target_chassis_clearance,
            ) = self._target_from_scenario(q, scenario)
            scenario_id = scenario.get("scenario_id")
            scenario_category = str(
                scenario.get("category", "unknown")
            )
        self.target_arm_positions = target_joint_positions
        self.base_goal_xy = base_goal_xy
        self.reach_env._set_target_position(target_position)
        self.reach_env._publish_target_immediately(repeat=3)

        structured_observation = self.base_env.reset()
        self.last_structured_observation = structured_observation
        self.last_sensor_observation = FusedLowLevelState.as_sensor(
            structured_observation
        )
        fused_observation = self.state_builder.build(
            structured_observation,
            self.base_goal_xy,
        )
        self.episode_index += 1
        self.last_reset_info = {
            "episode_index": self.episode_index,
            "base_goal_xy": self.base_goal_xy.copy(),
            "base_displacement_body": base_displacement_body.copy(),
            "target_position": target_position.copy(),
            "target_arm_positions": target_joint_positions.copy(),
            "initial_arm_positions": q[4:10].copy(),
            "target_chassis_clearance": target_chassis_clearance,
            "scenario_id": scenario_id,
            "scenario_category": scenario_category,
            "max_steps": self.max_steps,
            "initial_ee_distance": float(
                np.linalg.norm(
                    self.last_sensor_observation[0:3]
                    - self.last_sensor_observation[6:9]
                )
            ),
        }
        return fused_observation

    def step(self, fused_action):
        if self.last_sensor_observation is None:
            raise RuntimeError("reset must be called before step")
        action_sensor_observation = self.last_sensor_observation.copy()
        fused_action = self._vector(
            fused_action,
            self.FUSED_ACTION_DIM,
            "fused_action",
        ).astype(np.float32)
        full_action = self.action_adapter.to_full_action(
            fused_action,
            self.last_sensor_observation,
        )
        structured_observation, reward, done, info = self.base_env.step(
            full_action,
            publish_joints=self.base_env.controlled_joints,
        )
        self.last_structured_observation = structured_observation
        safe_full_action = np.asarray([
            float(info["cmd_dict"][joint_name])
            / float(self.base_env.action_max_vel[joint_name])
            for joint_name in self.base_env.expected_joints
        ], dtype=np.float32)
        safe_fused_action = self.action_adapter.from_full_action(
            safe_full_action,
            action_sensor_observation,
        )
        self.last_sensor_observation = FusedLowLevelState.as_sensor(
            structured_observation
        )
        fused_observation = self.state_builder.build(
            structured_observation,
            self.base_goal_xy,
        )
        info = dict(info)
        info.update({
            "episode_index": self.episode_index,
            "fused_action": fused_action.copy(),
            "safe_fused_action": safe_fused_action.copy(),
            "full_action": full_action.copy(),
            "safe_full_action": safe_full_action.copy(),
            "base_goal_xy": self.base_goal_xy.copy(),
            "target_arm_positions": self.target_arm_positions.copy(),
            "fusion_state": self.state_builder.diagnostics(),
        })
        return fused_observation, reward, done, info

    def filter_fused_action(self, fused_action):
        """Apply the established safety layer without publishing a command."""
        if self.last_sensor_observation is None:
            raise RuntimeError("reset must be called before filtering")
        fused_action = self._vector(
            fused_action,
            self.FUSED_ACTION_DIM,
            "fused_action",
        ).astype(np.float32)
        sensor_observation = self.last_sensor_observation.copy()
        full_action = self.action_adapter.to_full_action(
            fused_action,
            sensor_observation,
        )
        raw_cmd_dict = self.base_env.decode_action(full_action)
        q, _ = self.base_env.get_ordered_joint_state()
        safe_cmd_dict, safety_info = (
            self.base_env.apply_action_safety_filter(
                raw_cmd_dict,
                q,
            )
        )
        safe_full_action = np.asarray([
            float(safe_cmd_dict[joint_name])
            / float(self.base_env.action_max_vel[joint_name])
            for joint_name in self.base_env.expected_joints
        ], dtype=np.float32)
        safe_fused_action = self.action_adapter.from_full_action(
            safe_full_action,
            sensor_observation,
        )
        return safe_fused_action, {
            "raw_fused_action": fused_action.copy(),
            "safe_fused_action": safe_fused_action.copy(),
            "raw_full_action": full_action.copy(),
            "safe_full_action": safe_full_action.copy(),
            "raw_cmd_dict": dict(raw_cmd_dict),
            "safe_cmd_dict": dict(safe_cmd_dict),
            "safety_info": dict(safety_info),
        }

    def teacher_action(self, fused_observation):
        if self.teacher is None:
            return None
        return self.teacher.predict(fused_observation)

    def teacher_diagnostics(self):
        if self.teacher is None:
            return {}
        return self.teacher.diagnostics()

    def stop(self):
        self.reach_env.stop()

    close = stop

    def _reset_and_wait(self):
        last_error = None
        for attempt in range(1, self.reset_attempts + 1):
            self._brake_before_reset()
            self._reset_full_pose()
            self._publish_planar_zero_velocity(repeat=3)
            try:
                self._wait_for_full_reset()
                if attempt > 1:
                    rospy.logwarn(
                        "Gazebo fused reset recovered on attempt %d/%d",
                        attempt,
                        self.reset_attempts,
                    )
                return
            except RuntimeError as error:
                last_error = error
                if attempt < self.reset_attempts:
                    rospy.logwarn(
                        "Gazebo fused reset attempt %d/%d failed: %s; "
                        "reapplying the complete reset",
                        attempt,
                        self.reset_attempts,
                        str(error),
                    )

        raise RuntimeError(
            "Gazebo fused reset failed after {} attempts: {}".format(
                self.reset_attempts,
                str(last_error),
            )
        )

    def _brake_before_reset(self):
        """Let active velocity controllers consume zero before pausing."""
        deadline = time.time() + self.reset_pre_brake_time
        published = False
        while not rospy.is_shutdown():
            self.base_env.publish_zero_cmd()
            published = True
            remaining = deadline - time.time()
            if remaining <= 0.0:
                break
            time.sleep(min(0.02, remaining))
        if not published:
            self.base_env.publish_zero_cmd()

    def _publish_planar_zero_velocity(self, repeat=1):
        """Stop the base without racing the independent arm hold node."""
        planar_joints = tuple(self.base_env.planar_base_joints)
        zero_commands = dict((name, 0.0) for name in planar_joints)
        for _ in range(int(repeat)):
            self.base_env.publish_cmd_dict(
                zero_commands,
                joint_names=planar_joints,
            )
            time.sleep(0.02)

    def _reset_full_pose(self):
        for service_name in (
                "/gazebo/pause_physics",
                "/gazebo/unpause_physics",
                "/gazebo/set_model_configuration",
                "/gazebo/set_model_state",
                self.idle_hold_reset_service):
            rospy.wait_for_service(
                service_name,
                timeout=self.reach_env.service_timeout,
            )

        joint_names = (
            list(self.base_env.planar_base_joints)
            + list(self.base_env.arm_joints)
        )
        joint_positions = np.concatenate((
            self.reset_base_positions,
            self.reach_env.reset_arm_positions,
        ))
        paused = False
        try:
            self.reach_env._pause_physics()
            paused = True
            # Clear the world/model twist first.  The final operation that
            # writes coordinates must be set_model_configuration, so the
            # model-state update cannot disturb planar or arm positions.
            model_state = ModelState()
            model_state.model_name = self.reach_env.model_name
            model_state.pose.orientation.w = 1.0
            model_state.reference_frame = "world"
            state_response = self._set_model_state(model_state)
            if not state_response.success:
                raise RuntimeError(
                    "Gazebo fused model-state reset failed: {}".format(
                        state_response.status_message
                    )
                )

            for reset_index in range(self.reset_configuration_repeats):
                response = self.reach_env._set_model_configuration(
                    model_name=self.reach_env.model_name,
                    urdf_param_name=self.reach_env.robot_description_param,
                    joint_names=joint_names,
                    joint_positions=joint_positions.tolist(),
                )
                if not response.success:
                    raise RuntimeError(
                        "Gazebo fused joint reset pass {}/{} failed: {}".format(
                            reset_index + 1,
                            self.reset_configuration_repeats,
                            response.status_message,
                        )
                    )

            hold_response = self._reset_idle_hold_targets()
            if not hold_response.success:
                raise RuntimeError(
                    "Arm idle-hold target reset failed: {}".format(
                        hold_response.message
                    )
                )
            # Stop only the planar controllers before physics resumes.  The
            # idle-hold node now owns all six arm command topics and must be
            # allowed to pull them to the freshly restored hold targets.
            self._publish_planar_zero_velocity(repeat=1)
        finally:
            if paused:
                self.reach_env._unpause_physics()

    def _wait_for_full_reset(self):
        expected = np.concatenate((
            self.reset_base_positions,
            self.reach_env.reset_arm_positions,
        ))
        indices = [0, 1, 2, 4, 5, 6, 7, 8, 9]
        deadline = time.time() + self.reach_env.reset_timeout
        last_position_error = float("inf")
        last_max_velocity = float("inf")
        last_max_drift = float("inf")
        last_actual_q = None
        last_actual_dq = None
        last_position_vector = None
        settle_reference = None
        settle_started = None
        while not rospy.is_shutdown() and time.time() < deadline:
            q, dq = self.base_env.get_ordered_joint_state()
            actual_q = np.asarray(q, dtype=np.float64)[indices]
            actual_dq = np.asarray(dq, dtype=np.float64)[indices]
            position_error = actual_q - expected
            position_error[2] = FusedLowLevelState.wrap_angle(
                position_error[2]
            )
            last_actual_q = actual_q.copy()
            last_actual_dq = actual_dq.copy()
            last_position_vector = position_error.copy()
            last_position_error = float(
                np.max(np.abs(position_error))
            )
            last_max_velocity = float(np.max(np.abs(actual_dq)))
            position_ok = bool(
                last_position_error
                <= self.reach_env.reset_position_tolerance
            )
            velocity_ok = bool(
                last_max_velocity
                <= self.reach_env.reset_velocity_tolerance
            )
            if position_ok and velocity_ok:
                return

            # Some Gazebo velocity controllers report a persistent joint
            # velocity spike even when the measured position is stationary.
            # Accept that state only after the actual joint coordinates remain
            # inside a tight drift band for a complete settle window.
            if position_ok:
                if settle_reference is None:
                    settle_reference = actual_q.copy()
                    settle_started = time.time()
                    last_max_drift = 0.0
                else:
                    drift = actual_q - settle_reference
                    drift[2] = FusedLowLevelState.wrap_angle(drift[2])
                    last_max_drift = float(np.max(np.abs(drift)))
                    if last_max_drift > self.reset_drift_tolerance:
                        settle_reference = actual_q.copy()
                        settle_started = time.time()
                        last_max_drift = 0.0
                    elif (
                            time.time() - settle_started
                            >= self.reset_settle_window):
                        rospy.logwarn(
                            "Gazebo fused reset accepted by measured drift "
                            "fallback: reported_velocity=%.5f drift=%.5f "
                            "window=%.2fs",
                            last_max_velocity,
                            last_max_drift,
                            self.reset_settle_window,
                        )
                        return
            else:
                settle_reference = None
                settle_started = None
                last_max_drift = float("inf")
            self._publish_planar_zero_velocity(repeat=1)
        joint_names = (
            list(self.base_env.planar_base_joints)
            + list(self.base_env.arm_joints)
        )
        worst_position = "unavailable"
        worst_velocity = "unavailable"
        if last_position_vector is not None:
            position_index = int(np.argmax(np.abs(last_position_vector)))
            velocity_index = int(np.argmax(np.abs(last_actual_dq)))
            worst_position = (
                "{} actual={:.5f} expected={:.5f} error={:.5f}".format(
                    joint_names[position_index],
                    last_actual_q[position_index],
                    expected[position_index],
                    last_position_vector[position_index],
                )
            )
            worst_velocity = "{} velocity={:.5f}".format(
                joint_names[velocity_index],
                last_actual_dq[velocity_index],
            )
        raise RuntimeError(
            "Gazebo fused reset did not settle within {:.1f}s "
            "(position_error={:.5f}, max_velocity={:.5f}, "
            "max_drift={:.5f}, worst_position=[{}], "
            "worst_velocity=[{}])".format(
                self.reach_env.reset_timeout,
                last_position_error,
                last_max_velocity,
                last_max_drift,
                worst_position,
                worst_velocity,
            )
        )

    def _sample_coordinated_target(self, current_q):
        heading = float(current_q[2]) + TRACKED_FORWARD_YAW_OFFSET
        for _ in range(self.target_sample_attempts):
            target_arm = self.reach_env._sample_safe_arm_positions()
            chassis_clearance = self.reach_env._arm_chassis_clearance(
                target_arm
            )
            if (
                    chassis_clearance
                    < self.reach_env.target_min_chassis_clearance):
                continue

            forward = self.reach_env._rng.uniform(
                self.base_target_min_distance,
                self.base_target_max_distance,
            )
            lateral = self.reach_env._rng.uniform(
                -self.base_target_max_lateral,
                self.base_target_max_lateral,
            )
            displacement_body = np.asarray(
                [forward, lateral],
                dtype=np.float64,
            )
            displacement_fixed = FusedLowLevelState._rotate_xy(
                displacement_body,
                heading,
            )

            target_q = np.asarray(current_q, dtype=np.float64).copy()
            target_q[0:2] += displacement_fixed
            target_q[4:10] = target_arm
            candidate = (
                self.reach_env.kinematics_provider.end_effector_position(
                    target_q
                )
            )
            if not (
                    self.reach_env.target_min_z <= candidate[2]
                    <= self.reach_env.target_max_z):
                continue
            return (
                candidate,
                target_arm.copy(),
                target_q[0:2].copy(),
                displacement_body,
                chassis_clearance,
            )
        raise RuntimeError(
            "Unable to sample a safe coordinated base-arm target"
        )

    def _target_from_scenario(self, current_q, scenario):
        target_arm = self._vector(
            scenario.get("target_arm_positions"),
            6,
            "scenario.target_arm_positions",
        )
        displacement_body = self._vector(
            scenario.get("base_displacement_body"),
            2,
            "scenario.base_displacement_body",
        )
        target_position = self._vector(
            scenario.get("target_position"),
            3,
            "scenario.target_position",
        )
        base_goal_xy = self._vector(
            scenario.get("base_goal_xy"),
            2,
            "scenario.base_goal_xy",
        )
        chassis_clearance = self.reach_env._arm_chassis_clearance(
            target_arm
        )
        if chassis_clearance < self.reach_env.target_min_chassis_clearance:
            raise ValueError(
                "scenario target violates chassis clearance: {:.5f}".format(
                    chassis_clearance
                )
            )
        if not (
                self.reach_env.target_min_z <= target_position[2]
                <= self.reach_env.target_max_z):
            raise ValueError(
                "scenario target z is outside the configured range"
            )
        return (
            target_position,
            target_arm.copy(),
            base_goal_xy.copy(),
            displacement_body.copy(),
            chassis_clearance,
        )

    def _validate_parameters(self):
        if self.base_target_min_distance <= 0.0:
            raise ValueError(
                "base_target_min_distance must be positive"
            )
        if (
                self.base_target_max_distance
                <= self.base_target_min_distance):
            raise ValueError(
                "base_target_max_distance must exceed minimum"
            )
        if self.base_target_max_lateral < 0.0:
            raise ValueError(
                "base_target_max_lateral must be non-negative"
            )
        if self.target_sample_attempts <= 0:
            raise ValueError("target_sample_attempts must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.success_threshold <= 0.0:
            raise ValueError("success_threshold must be positive")
        if self.reset_settle_window <= 0.0:
            raise ValueError("reset_settle_window must be positive")
        if self.reset_drift_tolerance <= 0.0:
            raise ValueError("reset_drift_tolerance must be positive")
        if self.reset_attempts <= 0:
            raise ValueError("reset_attempts must be positive")
        if self.reset_pre_brake_time < 0.0:
            raise ValueError("reset_pre_brake_time must be non-negative")
        if self.reset_configuration_repeats <= 0:
            raise ValueError(
                "reset_configuration_repeats must be positive"
            )
        if not self.idle_hold_reset_service:
            raise ValueError("idle_hold_reset_service must not be empty")

    @staticmethod
    def _vector(value, size, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,):
            raise ValueError(
                "{} must have shape ({},), got {}".format(
                    name,
                    size,
                    vector.shape,
                )
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("{} contains non-finite values".format(name))
        return vector
