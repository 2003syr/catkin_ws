#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from hrl.high_level_command import TaskMode
from hrl.hrl4in_low_level import HRL4INLowLevelState


class RuleBasedLowPolicy(object):
    """Rule executor behind an HRL4IN-compatible low-level interface.

    HRL4IN's low policy observes four inputs:

        sensor, remaining subgoal, subgoal mask, action mask

    This class keeps that interface and reward/termination mechanism, but uses
    a deterministic controller in place of PPO so the hierarchy can be wired
    into ROS before learned low-level control is introduced.  ARM_REACH uses a
    damped least-squares Cartesian controller; safety filtering still happens
    later in MobileArmReachEnv.
    """

    OBS_DIM = 46
    ACTION_DIM = 10

    TARGET_POSITION_SLICE = slice(0, 3)
    JOINT_POSITION_SLICE = slice(11, 21)
    BASE_ACTION_SLICE = slice(0, 4)
    ARM_ACTION_SLICE = slice(4, 10)
    BASE_SUBGOAL_SLICE = slice(0, 3)
    ARM_SUBGOAL_SLICE = slice(3, 6)

    DEFAULT_ARM_JOINT_MAX_VELOCITY = (
        0.50,
        0.50,
        0.03,
        0.50,
        0.03,
        0.50,
    )

    def __init__(
            self,
            base_gain=5.0,
            arm_gain=1.0,
            recovery_gain=0.4,
            enable_base_motion=False,
            jacobian_provider=None,
            arm_joint_max_velocity=None,
            max_cartesian_speed=0.05,
            dls_damping=0.03,
            subgoal_tolerance=None,
            intrinsic_reward_scale=30.0,
            subgoal_achieved_reward=1.0,
            collision_reward_weight=0.0,
            extrinsic_reward_weight=0.0):
        self.base_gain = float(base_gain)
        self.arm_gain = float(arm_gain)
        self.recovery_gain = float(recovery_gain)
        self.enable_base_motion = bool(enable_base_motion)
        self.jacobian_provider = jacobian_provider

        if arm_joint_max_velocity is None:
            arm_joint_max_velocity = self.DEFAULT_ARM_JOINT_MAX_VELOCITY
        self.arm_joint_max_velocity = np.asarray(
            arm_joint_max_velocity,
            dtype=np.float64,
        )
        if self.arm_joint_max_velocity.shape != (6,):
            raise ValueError("arm_joint_max_velocity must have shape (6,)")
        if np.any(self.arm_joint_max_velocity <= 0.0):
            raise ValueError("arm joint velocity limits must be positive")

        self.max_cartesian_speed = float(max_cartesian_speed)
        self.dls_damping = float(dls_damping)
        if self.max_cartesian_speed <= 0.0:
            raise ValueError("max_cartesian_speed must be positive")
        if self.dls_damping <= 0.0:
            raise ValueError("dls_damping must be positive")

        self.subgoal_state = HRL4INLowLevelState(
            subgoal_tolerance=subgoal_tolerance,
            intrinsic_reward_scale=intrinsic_reward_scale,
            subgoal_achieved_reward=subgoal_achieved_reward,
            collision_reward_weight=collision_reward_weight,
            extrinsic_reward_weight=extrinsic_reward_weight,
        )
        self.reset()

    def reset(self):
        self.current_command = None
        self.subgoal_state.reset()
        self.last_diagnostics = {}

    def begin_subgoal(self, observation, command):
        """Freeze the HRL4IN ideal next state at the meta-policy boundary."""
        self.current_command = command
        low_observation = self.subgoal_state.begin_subgoal(
            observation,
            command,
        )
        self.last_diagnostics = self.subgoal_state.diagnostics()
        return low_observation

    def low_level_observation(self, observation):
        """Expose the 68-dimensional structured input for a learned policy."""
        return self.subgoal_state.low_level_observation(observation)

    def predict(self, observation, command):
        sensor = self._as_vector(observation)
        if command is not self.current_command:
            self.begin_subgoal(sensor, command)

        low_observation = self.low_level_observation(sensor)
        remaining_subgoal = np.asarray(
            low_observation["subgoal"],
            dtype=np.float64,
        )
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)

        if command.mode == TaskMode.BASE_APPROACH:
            self._apply_base_approach(
                action,
                remaining_subgoal[self.BASE_SUBGOAL_SLICE],
            )
            self.last_diagnostics.update(
                self.subgoal_state.diagnostics()
            )

        elif command.mode == TaskMode.ARM_REACH:
            action[self.ARM_ACTION_SLICE] = self._arm_reach_action(
                sensor,
                remaining_subgoal[self.ARM_SUBGOAL_SLICE],
            )

        elif command.mode == TaskMode.RECOVERY:
            action[self.ARM_ACTION_SLICE] = self._recovery_action(sensor)
            self.last_diagnostics.update(
                self.subgoal_state.diagnostics()
            )

        # HRL4IN masks the sampled low-level action immediately before the
        # environment step.  The existing ROS safety layer remains downstream.
        unmasked_action = action.copy()
        action *= np.asarray(
            low_observation["action_mask"],
            dtype=np.float32,
        )
        self.last_diagnostics["unmasked_action"] = unmasked_action
        self.last_diagnostics["masked_action"] = action.copy()
        return np.clip(action, -1.0, 1.0)

    def _apply_base_approach(self, action, remaining_subgoal):
        """Populate base actions only when a real base interface is enabled."""
        if not self.enable_base_motion:
            return

        remaining_subgoal = np.asarray(
            remaining_subgoal,
            dtype=np.float64,
        )
        action[0] = self.base_gain * remaining_subgoal[0]
        action[1] = self.base_gain * remaining_subgoal[1]
        # z and sway remain fixed in the verified x/y planar-base stage.
        action[2] = 0.0
        action[3] = 0.0

    def _arm_reach_action(self, observation, error_base):
        if self.jacobian_provider is None:
            raise RuntimeError(
                "ARM_REACH requires a Jacobian provider; fixed joint "
                "heuristics are intentionally disabled"
            )

        error_base = np.asarray(error_base, dtype=np.float64)
        cartesian_velocity = self._limited_cartesian_velocity(error_base)
        joint_positions = np.asarray(
            observation[self.JOINT_POSITION_SLICE],
            dtype=np.float64,
        )
        jacobian = np.asarray(
            self.jacobian_provider.position_jacobian(joint_positions),
            dtype=np.float64,
        )
        if jacobian.shape != (3, 6):
            raise ValueError(
                "position Jacobian must have shape (3, 6), got {}".format(
                    jacobian.shape
                )
            )

        normalized_arm_action, condition = self._damped_least_squares(
            jacobian,
            cartesian_velocity,
        )
        target_position = np.asarray(
            observation[self.TARGET_POSITION_SLICE],
            dtype=np.float64,
        )
        end_effector_position = np.asarray(
            HRL4INLowLevelState.extract_task_state(observation)[3:6],
            dtype=np.float64,
        )
        target_error_base = target_position - end_effector_position
        tracker_diagnostics = self.subgoal_state.diagnostics()
        subgoal_distance = float(np.linalg.norm(error_base))
        self.last_diagnostics = tracker_diagnostics
        self.last_diagnostics.update({
            "error_base": error_base.copy(),
            "error_norm": subgoal_distance,
            "subgoal_distance": subgoal_distance,
            "absolute_position_subgoal": np.asarray(
                tracker_diagnostics["ideal_next_state"][3:6],
                dtype=np.float64,
            ).copy(),
            "target_error_base": target_error_base.copy(),
            "target_error_norm": float(np.linalg.norm(target_error_base)),
            "cartesian_velocity": cartesian_velocity.copy(),
            "jacobian_condition": condition,
            "normalized_arm_action": normalized_arm_action.copy(),
        })
        return normalized_arm_action.astype(np.float32)

    def _limited_cartesian_velocity(self, error_base):
        cartesian_velocity = self.arm_gain * error_base
        speed = float(np.linalg.norm(cartesian_velocity))
        if speed > self.max_cartesian_speed:
            cartesian_velocity = (
                cartesian_velocity * self.max_cartesian_speed / speed
            )
        return cartesian_velocity

    def _damped_least_squares(self, jacobian, cartesian_velocity):
        # Solve for normalized actions directly.  Column scaling converts each
        # normalized action into its physical joint-velocity contribution.
        scaled_jacobian = (
            jacobian * self.arm_joint_max_velocity[np.newaxis, :]
        )
        regularized = (
            np.dot(scaled_jacobian, scaled_jacobian.T)
            + (self.dls_damping ** 2) * np.eye(3, dtype=np.float64)
        )
        normalized_action = np.dot(
            scaled_jacobian.T,
            np.linalg.solve(regularized, cartesian_velocity),
        )

        singular_values = np.linalg.svd(
            scaled_jacobian,
            compute_uv=False,
        )
        smallest = float(np.min(singular_values))
        condition = (
            float(np.max(singular_values) / smallest)
            if smallest > 1e-9 else float("inf")
        )
        return normalized_action, condition

    def observe_transition(
            self,
            next_observation,
            extrinsic_reward=0.0,
            collision_reward=0.0,
            episode_done=False,
            subgoal_timed_out=False):
        """Update remaining subgoal, termination and intrinsic reward."""
        tracker_diagnostics = self.subgoal_state.observe_transition(
            next_observation,
            extrinsic_reward=extrinsic_reward,
            collision_reward=collision_reward,
            episode_done=episode_done,
            subgoal_timed_out=subgoal_timed_out,
        )
        self.last_diagnostics.update(tracker_diagnostics)

        remaining_arm_error = np.asarray(
            tracker_diagnostics["remaining_subgoal"][
                self.ARM_SUBGOAL_SLICE
            ],
            dtype=np.float64,
        )
        self.last_diagnostics.update({
            "post_error_base": remaining_arm_error.copy(),
            "post_error_norm": float(
                tracker_diagnostics["post_potential"]
            ),
            "subgoal_distance": float(
                tracker_diagnostics["post_potential"]
            ),
        })
        return self.diagnostics()

    def _recovery_action(self, observation):
        joint_positions = observation[self.JOINT_POSITION_SLICE]
        arm_positions = joint_positions[self.ARM_ACTION_SLICE]
        return -self.recovery_gain * arm_positions

    def diagnostics(self):
        result = {}
        for key, value in self.last_diagnostics.items():
            result[key] = value.copy() if hasattr(value, "copy") else value
        return result

    @classmethod
    def _as_vector(cls, observation):
        return HRL4INLowLevelState.as_sensor(observation)
