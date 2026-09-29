#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Collect successful coordinated base-arm teacher demonstrations."""

from __future__ import division, print_function

import collections
import os
import sys

import numpy as np
import rospy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_low_dataset import (
    ACTION_DIM,
    OBSERVATION_DIM,
    PHASE_NAMES,
    phase_id,
    summarize_samples,
    validate_action_mask_contract,
)
from training.fused_low_training_env import FusedLowLevelTrainingEnv


class FusedLowTeacherDatasetCollector(object):

    ACTION_NAMES = (
        "tracked_v",
        "tracked_omega",
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    )

    def __init__(self):
        self.output_path = os.path.abspath(os.path.expanduser(
            rospy.get_param(
                "~output_path",
                "/tmp/mobile_arm_rl_training/fused_low_teacher_100ep.npz",
            )
        ))
        self.episodes = int(rospy.get_param("~episodes", 100))
        self.max_attempts = int(rospy.get_param("~max_attempts", 125))
        self.minimum_episode_samples = int(
            rospy.get_param("~minimum_episode_samples", 20)
        )
        self.log_interval = int(rospy.get_param("~log_interval", 25))
        self.environment_type = str(
            rospy.get_param("~environment_type", "fused_low")
        )
        self.detour_side = float(
            rospy.get_param("~detour_side", 0.0)
        )
        self.structured_action_masks = self.environment_type in (
            "box_detour",
            "fused_box_detour",
        )
        self.max_steps = int(
            rospy.get_param("~max_steps", 500)
        )
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.max_attempts < self.episodes:
            raise ValueError("max_attempts must be >= episodes")
        if self.minimum_episode_samples <= 0 or self.log_interval <= 0:
            raise ValueError(
                "minimum_episode_samples and log_interval must be positive"
            )

        environment_kwargs = {
            "init_ros_node": False,
            "seed": rospy.get_param("~seed", 123),
            "enable_teacher": True,
        }
        if self.environment_type in (
                "box_detour",
                "fused_box_detour"):
            from training.fused_box_detour_training_env import (
                FusedBoxDetourTrainingEnv,
            )
            self.environment = FusedBoxDetourTrainingEnv(
                **environment_kwargs
            )
        elif self.environment_type == "fused_low":
            self.environment = FusedLowLevelTrainingEnv(
                **environment_kwargs
            )
        else:
            raise ValueError(
                "unsupported fused teacher environment_type: {}".format(
                    self.environment_type
                )
            )
        self.samples = collections.defaultdict(list)
        self.accepted_episode_metadata = collections.defaultdict(list)
        self.attempt_metadata = collections.defaultdict(list)
        self.successful_episodes = 0
        self.attempted_episodes = 0
        self.collision_episodes = 0
        self.timeout_episodes = 0

    def run(self):
        try:
            while (
                    not rospy.is_shutdown()
                    and self.successful_episodes < self.episodes
                    and self.attempted_episodes < self.max_attempts):
                self._collect_attempt()
        finally:
            self.environment.stop()
            self._save_dataset()

        if self.successful_episodes < self.episodes:
            raise RuntimeError(
                "collected only {}/{} successful fused teacher episodes "
                "after {} attempts".format(
                    self.successful_episodes,
                    self.episodes,
                    self.attempted_episodes,
                )
            )
        rospy.loginfo(
            "Fused teacher dataset complete: %s episodes=%d "
            "attempts=%d samples=%d",
            self.output_path,
            self.successful_episodes,
            self.attempted_episodes,
            len(self.samples["observations"]),
        )

    def _collect_attempt(self):
        self.attempted_episodes += 1
        scenario = None
        if self.environment_type in (
                "box_detour",
                "fused_box_detour"):
            side = self.detour_side
            if abs(side) < 0.5:
                side = 1.0 if self.attempted_episodes % 2 else -1.0
            scenario = {
                "scenario_id": "box_teacher_attempt_{}".format(
                    self.attempted_episodes
                ),
                "category": "box_detour",
                "detour_side": side,
            }
        observation = self.environment.reset(
            scenario=scenario,
            max_steps=self.max_steps,
        )
        reset_info = dict(self.environment.last_reset_info)
        episode = collections.defaultdict(list)
        episode_phase_counts = collections.Counter()
        episode_overlap = 0
        final_info = {}

        while not rospy.is_shutdown():
            teacher_action = self.environment.teacher_action(observation)
            diagnostics = self.environment.teacher_diagnostics()
            phase = str(diagnostics.get("phase", "UNKNOWN"))
            phase_index = phase_id(phase)
            teacher_limit_blocked = "|".join(
                str(name)
                for name in diagnostics.get(
                    "joint_limit_blocked",
                    [],
                )
            )
            base_active = bool(
                abs(float(teacher_action[0])) > 0.05
                or abs(float(teacher_action[1])) > 0.05
            )
            arm_active = bool(
                np.max(np.abs(teacher_action[2:8])) > 0.02
            )
            overlap_active = bool(base_active and arm_active)
            episode_overlap += int(overlap_active)
            episode_phase_counts[phase] += 1

            next_observation, reward, done, info = self.environment.step(
                teacher_action
            )
            safety_active, safety_reasons = self._safety_summary(info)
            episode["observations"].append(observation.copy())
            episode["teacher_actions"].append(teacher_action.copy())
            episode["safe_teacher_actions"].append(
                np.asarray(
                    info["safe_fused_action"],
                    dtype=np.float32,
                )
            )
            episode["phase_ids"].append(phase_index)
            episode["phase_labels"].append(phase)
            episode["box_path_indices"].append(int(
                diagnostics.get("box_path_index", -1)
            ))
            episode["detour_sides"].append(float(
                reset_info.get("detour_side", 0.0)
            ))
            episode["arm_blends"].append(
                float(diagnostics.get("arm_blend", 0.0))
            )
            episode["base_active"].append(base_active)
            episode["arm_active"].append(arm_active)
            episode["overlap_active"].append(overlap_active)
            episode["safety_filter_active"].append(safety_active)
            episode["safety_reasons"].append(safety_reasons)
            episode["teacher_limit_active"].append(
                bool(teacher_limit_blocked)
            )
            episode["teacher_limit_blocked"].append(
                teacher_limit_blocked
            )
            episode["rewards"].append(float(reward))
            episode["terminal"].append(bool(done))
            episode["ee_distances"].append(
                float(info.get("dist", float("nan")))
            )
            episode["base_distances"].append(float(
                info.get("fusion_state", {}).get(
                    "base_distance",
                    float("nan"),
                )
            ))
            observation = next_observation
            final_info = info

            step_count = int(self.environment.base_env.step_count)
            if (
                    step_count == 1
                    or step_count % self.log_interval == 0
                    or done):
                rospy.loginfo(
                    "fused_collection_step attempt=%d accepted=%d/%d "
                    "step=%d phase=%s ee_distance=%.4f "
                    "base_distance=%.4f overlap=%s safety=%s",
                    self.attempted_episodes,
                    self.successful_episodes,
                    self.episodes,
                    step_count,
                    phase,
                    episode["ee_distances"][-1],
                    episode["base_distances"][-1],
                    str(overlap_active),
                    safety_reasons or "NONE",
                )
            if done:
                break

        sensor_state = self.environment.base_env.get_sensor_state()
        collision = bool(
            final_info.get("collision", False)
            or final_info.get("collision_names", [])
            or sensor_state[3]
        )
        success = bool(final_info.get("success", False) and not collision)
        timeout = bool(not success and not collision)
        enough_samples = bool(
            len(episode["observations"]) >= self.minimum_episode_samples
        )
        accepted = bool(success and enough_samples)
        self.collision_episodes += int(collision)
        self.timeout_episodes += int(timeout)
        self._record_attempt(
            success,
            collision,
            timeout,
            len(episode["observations"]),
            reset_info,
            final_info,
            episode_overlap,
        )

        if accepted:
            self.successful_episodes += 1
            accepted_id = self.successful_episodes
            sample_count = len(episode["observations"])
            for name, values in episode.items():
                self.samples[name].extend(values)
            self.samples["episode_ids"].extend(
                [accepted_id] * sample_count
            )
            self.samples["attempt_ids"].extend(
                [self.attempted_episodes] * sample_count
            )
            self._record_accepted_episode(
                accepted_id,
                reset_info,
                final_info,
                sample_count,
                episode_overlap,
                episode_phase_counts,
            )
            self._save_dataset()

        rospy.loginfo(
            "fused_collection_result attempt=%d accepted=%d/%d "
            "success=%s collision=%s timeout=%s samples=%d "
            "overlap_steps=%d phases=%s",
            self.attempted_episodes,
            self.successful_episodes,
            self.episodes,
            str(success),
            str(collision),
            str(timeout),
            len(episode["observations"]),
            episode_overlap,
            dict(episode_phase_counts),
        )

    def _record_attempt(
            self,
            success,
            collision,
            timeout,
            steps,
            reset_info,
            final_info,
            overlap_steps):
        self.attempt_metadata["success"].append(success)
        self.attempt_metadata["collision"].append(collision)
        self.attempt_metadata["timeout"].append(timeout)
        self.attempt_metadata["steps"].append(steps)
        self.attempt_metadata["initial_ee_distance"].append(
            float(reset_info["initial_ee_distance"])
        )
        self.attempt_metadata["final_ee_distance"].append(
            float(final_info.get("dist", float("nan")))
        )
        self.attempt_metadata["final_base_distance"].append(float(
            final_info.get("fusion_state", {}).get(
                "base_distance",
                float("nan"),
            )
        ))
        self.attempt_metadata["overlap_steps"].append(overlap_steps)

    def _record_accepted_episode(
            self,
            episode_id,
            reset_info,
            final_info,
            steps,
            overlap_steps,
            phase_counts):
        metadata = self.accepted_episode_metadata
        metadata["episode_ids"].append(episode_id)
        metadata["steps"].append(steps)
        metadata["overlap_steps"].append(overlap_steps)
        metadata["initial_ee_distance"].append(
            float(reset_info["initial_ee_distance"])
        )
        metadata["final_ee_distance"].append(
            float(final_info.get("dist", float("nan")))
        )
        metadata["final_base_distance"].append(float(
            final_info.get("fusion_state", {}).get(
                "base_distance",
                float("nan"),
            )
        ))
        metadata["base_goal_xy"].append(reset_info["base_goal_xy"])
        metadata["base_displacement_body"].append(
            reset_info["base_displacement_body"]
        )
        metadata["target_position"].append(reset_info["target_position"])
        metadata["target_arm_positions"].append(
            reset_info["target_arm_positions"]
        )
        metadata["target_chassis_clearance"].append(
            float(reset_info["target_chassis_clearance"])
        )
        metadata["phase_counts"].append([
            int(phase_counts.get(name, 0))
            for name in PHASE_NAMES
        ])

    def _save_dataset(self):
        directory = os.path.dirname(self.output_path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        observations = self._matrix("observations", OBSERVATION_DIM)
        teacher_actions = self._matrix("teacher_actions", ACTION_DIM)
        safe_teacher_actions = self._matrix(
            "safe_teacher_actions",
            ACTION_DIM,
        )
        phase_ids = np.asarray(
            self.samples["phase_ids"],
            dtype=np.int8,
        )
        safety_active = np.asarray(
            self.samples["safety_filter_active"],
            dtype=np.bool_,
        )
        if observations.shape[0]:
            summary = summarize_samples(
                observations,
                teacher_actions,
                safe_teacher_actions,
                phase_ids,
                safety_active,
            )
            action_mask_contract = validate_action_mask_contract(
                observations,
                safe_teacher_actions,
                phase_ids,
            )
        else:
            summary = self._empty_summary()
            action_mask_contract = {
                "mode": (
                    "three_stage"
                    if self.structured_action_masks
                    else "legacy_all_enabled"
                )
            }

        np.savez_compressed(
            self.output_path,
            observations=observations,
            teacher_actions=teacher_actions,
            safe_teacher_actions=safe_teacher_actions,
            episode_ids=np.asarray(
                self.samples["episode_ids"], dtype=np.int32
            ),
            attempt_ids=np.asarray(
                self.samples["attempt_ids"], dtype=np.int32
            ),
            phase_ids=phase_ids,
            phase_labels=np.asarray(
                self.samples["phase_labels"], dtype="S16"
            ),
            box_path_indices=np.asarray(
                self.samples["box_path_indices"], dtype=np.int16
            ),
            detour_sides=np.asarray(
                self.samples["detour_sides"], dtype=np.float32
            ),
            arm_blends=np.asarray(
                self.samples["arm_blends"], dtype=np.float32
            ),
            base_active=np.asarray(
                self.samples["base_active"], dtype=np.bool_
            ),
            arm_active=np.asarray(
                self.samples["arm_active"], dtype=np.bool_
            ),
            overlap_active=np.asarray(
                self.samples["overlap_active"], dtype=np.bool_
            ),
            safety_filter_active=safety_active,
            safety_reasons=np.asarray(
                self.samples["safety_reasons"], dtype="S256"
            ),
            teacher_limit_active=np.asarray(
                self.samples["teacher_limit_active"], dtype=np.bool_
            ),
            teacher_limit_blocked=np.asarray(
                self.samples["teacher_limit_blocked"], dtype="S128"
            ),
            rewards=np.asarray(
                self.samples["rewards"], dtype=np.float32
            ),
            terminal=np.asarray(
                self.samples["terminal"], dtype=np.bool_
            ),
            ee_distances=np.asarray(
                self.samples["ee_distances"], dtype=np.float32
            ),
            base_distances=np.asarray(
                self.samples["base_distances"], dtype=np.float32
            ),
            observation_dim=np.asarray(
                [OBSERVATION_DIM], dtype=np.int32
            ),
            action_dim=np.asarray([ACTION_DIM], dtype=np.int32),
            observation_semantics=np.asarray([
                "sensor46,coordinated_remaining_subgoal6,"
                "subgoal_mask6,action_mask8"
            ]),
            action_mask_semantics=np.asarray([
                (
                    "BASE_APPROACH=base_only,COORDINATED=base+arm,"
                    "ARM_FINISH=arm_only"
                    if self.structured_action_masks
                    else "all_phases=base+arm"
                )
            ]),
            action_mask_contract_version=np.asarray([1], dtype=np.int32),
            action_mask_contract_mode=np.asarray([
                action_mask_contract["mode"]
            ], dtype="S32"),
            action_semantics=np.asarray([
                "normalized_tracked_v,normalized_tracked_omega,"
                "normalized_joint1,normalized_joint2,"
                "normalized_joint3,normalized_joint4,"
                "normalized_joint5,normalized_joint6"
            ]),
            action_names=np.asarray(self.ACTION_NAMES, dtype="S32"),
            teacher_type=np.asarray(
                ["coordinated_rule_teacher"], dtype="S64"
            ),
            environment_type=np.asarray(
                [self.environment_type], dtype="S32"
            ),
            configured_detour_side=np.asarray(
                [self.detour_side], dtype=np.float32
            ),
            policy_target=np.asarray(
                ["safe_teacher_actions"], dtype="S64"
            ),
            phase_names=np.asarray(PHASE_NAMES, dtype="S16"),
            phase_counts=summary["phase_counts"],
            phase_fractions=summary["phase_fractions"].astype(
                np.float32
            ),
            teacher_action_mean=summary[
                "teacher_action_mean"
            ].astype(np.float32),
            teacher_action_std=summary[
                "teacher_action_std"
            ].astype(np.float32),
            teacher_action_min=summary[
                "teacher_action_min"
            ].astype(np.float32),
            teacher_action_max=summary[
                "teacher_action_max"
            ].astype(np.float32),
            safe_action_mean=summary[
                "safe_action_mean"
            ].astype(np.float32),
            safe_action_std=summary[
                "safe_action_std"
            ].astype(np.float32),
            mean_absolute_filter_delta=summary[
                "mean_absolute_filter_delta"
            ].astype(np.float32),
            maximum_filter_delta=summary[
                "maximum_filter_delta"
            ].astype(np.float32),
            safety_intervention_rate=np.asarray([
                summary["safety_intervention_rate"]
            ], dtype=np.float32),
            successful_episodes=np.asarray(
                [self.successful_episodes], dtype=np.int32
            ),
            attempted_episodes=np.asarray(
                [self.attempted_episodes], dtype=np.int32
            ),
            collision_episodes=np.asarray(
                [self.collision_episodes], dtype=np.int32
            ),
            timeout_episodes=np.asarray(
                [self.timeout_episodes], dtype=np.int32
            ),
            accepted_episode_ids=self._accepted_vector(
                "episode_ids", np.int32
            ),
            accepted_episode_steps=self._accepted_vector(
                "steps", np.int32
            ),
            accepted_episode_overlap_steps=self._accepted_vector(
                "overlap_steps", np.int32
            ),
            accepted_initial_ee_distances=self._accepted_vector(
                "initial_ee_distance", np.float32
            ),
            accepted_final_ee_distances=self._accepted_vector(
                "final_ee_distance", np.float32
            ),
            accepted_final_base_distances=self._accepted_vector(
                "final_base_distance", np.float32
            ),
            accepted_base_goal_xy=self._accepted_matrix(
                "base_goal_xy", 2
            ),
            accepted_base_displacement_body=self._accepted_matrix(
                "base_displacement_body", 2
            ),
            accepted_target_positions=self._accepted_matrix(
                "target_position", 3
            ),
            accepted_target_arm_positions=self._accepted_matrix(
                "target_arm_positions", 6
            ),
            accepted_target_chassis_clearance=self._accepted_vector(
                "target_chassis_clearance", np.float32
            ),
            accepted_phase_counts=np.asarray(
                self.accepted_episode_metadata["phase_counts"],
                dtype=np.int32,
            ).reshape((-1, len(PHASE_NAMES))),
            attempt_success=self._attempt_vector("success", np.bool_),
            attempt_collision=self._attempt_vector(
                "collision", np.bool_
            ),
            attempt_timeout=self._attempt_vector("timeout", np.bool_),
            attempt_steps=self._attempt_vector("steps", np.int32),
            attempt_initial_ee_distances=self._attempt_vector(
                "initial_ee_distance", np.float32
            ),
            attempt_final_ee_distances=self._attempt_vector(
                "final_ee_distance", np.float32
            ),
            attempt_final_base_distances=self._attempt_vector(
                "final_base_distance", np.float32
            ),
            attempt_overlap_steps=self._attempt_vector(
                "overlap_steps", np.int32
            ),
        )
        if observations.shape[0]:
            rospy.loginfo(
                "Fused dataset checkpoint: %s accepted=%d/%d "
                "samples=%d phases=%s safety_rate=%.3f "
                "action_std=%s",
                self.output_path,
                self.successful_episodes,
                self.episodes,
                observations.shape[0],
                dict(
                    (name, int(summary["phase_counts"][index]))
                    for index, name in enumerate(PHASE_NAMES)
                ),
                summary["safety_intervention_rate"],
                np.round(summary["safe_action_std"], 3).tolist(),
            )

    def _matrix(self, name, width):
        return np.asarray(
            self.samples[name],
            dtype=np.float32,
        ).reshape((-1, width))

    def _accepted_vector(self, name, dtype):
        return np.asarray(
            self.accepted_episode_metadata[name],
            dtype=dtype,
        )

    def _accepted_matrix(self, name, width):
        return np.asarray(
            self.accepted_episode_metadata[name],
            dtype=np.float32,
        ).reshape((-1, width))

    def _attempt_vector(self, name, dtype):
        return np.asarray(self.attempt_metadata[name], dtype=dtype)

    @staticmethod
    def _safety_summary(info):
        reasons = []
        for joint_name, values in sorted(
                info.get("safety_info", {}).items()):
            reason = str(values.get("reason", "safe"))
            if reason not in ("safe", "continuous_joint"):
                reasons.append("{}:{}".format(joint_name, reason))
        return bool(reasons), "|".join(reasons)

    @staticmethod
    def _empty_summary():
        zeros = np.zeros(ACTION_DIM, dtype=np.float64)
        return {
            "phase_counts": np.zeros(
                len(PHASE_NAMES), dtype=np.int64
            ),
            "phase_fractions": np.zeros(
                len(PHASE_NAMES), dtype=np.float64
            ),
            "teacher_action_mean": zeros.copy(),
            "teacher_action_std": zeros.copy(),
            "teacher_action_min": zeros.copy(),
            "teacher_action_max": zeros.copy(),
            "safe_action_mean": zeros.copy(),
            "safe_action_std": zeros.copy(),
            "mean_absolute_filter_delta": zeros.copy(),
            "maximum_filter_delta": zeros.copy(),
            "safety_intervention_rate": 0.0,
        }


def main():
    rospy.init_node("collect_fused_low_teacher_dataset")
    FusedLowTeacherDatasetCollector().run()


if __name__ == "__main__":
    main()
