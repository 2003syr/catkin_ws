#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Collect teacher labels on states visited by the fused BC student.

The default environment is the original obstacle-free fused reach task.  Set
``~environment_type:=box_detour`` to collect DAgger states in the same static
box task used by the box teacher/BC chain.
"""

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

from training.evaluate_fused_low_bc import (
    PolicyServiceClient,
    _active_safety_reasons,
)
from training.fused_low_dataset import (
    ACTION_DIM,
    OBSERVATION_DIM,
    PHASE_NAMES,
    phase_id,
    validate_action_mask_contract,
)
from training.fused_low_training_env import FusedLowLevelTrainingEnv


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


class FusedLowDaggerCollector(object):

    def __init__(self):
        self.episodes = int(rospy.get_param("~episodes", 20))
        self.seed = int(rospy.get_param("~seed", 200123))
        self.teacher_beta = float(
            rospy.get_param("~teacher_beta", 0.20)
        )
        self.deviation_threshold = float(
            rospy.get_param("~deviation_threshold", 0.20)
        )
        self.safety_delta_threshold = float(
            rospy.get_param("~safety_delta_threshold", 0.01)
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
        self.output_path = os.path.abspath(os.path.expanduser(
            rospy.get_param(
                "~output_path",
                "/tmp/mobile_arm_rl_training/fused_low_dagger_20ep.npz",
            )
        ))
        self._validate_parameters()
        self.rng = np.random.RandomState(self.seed)
        self.policy = PolicyServiceClient(
            host=rospy.get_param("~policy_host", "127.0.0.1"),
            port=rospy.get_param("~policy_port", 5563),
            timeout=rospy.get_param("~policy_timeout", 5.0),
        )
        environment_kwargs = {
            "init_ros_node": False,
            "seed": self.seed,
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
                "unsupported DAgger environment_type: {}".format(
                    self.environment_type
                )
            )
        self.samples = collections.defaultdict(list)
        self.episode_metadata = collections.defaultdict(list)

    def collect(self):
        successes = 0
        collisions = 0
        timeouts = 0
        total_interventions = 0
        total_samples = 0
        try:
            for episode_index in range(self.episodes):
                scenario = None
                episode_side = 0.0
                if self.environment_type in (
                        "box_detour",
                        "fused_box_detour"):
                    episode_side = self.detour_side
                    if abs(episode_side) < 0.5:
                        episode_side = (
                            1.0 if episode_index % 2 == 0 else -1.0
                        )
                    scenario = {
                        "scenario_id": "box_dagger_{:03d}".format(
                            episode_index + 1
                        ),
                        "category": "box_detour_dagger",
                        "detour_side": episode_side,
                    }
                observation = self.environment.reset(
                    scenario=scenario,
                    max_steps=self.max_steps,
                )
                reset_info = dict(self.environment.last_reset_info)
                episode = collections.defaultdict(list)
                episode_interventions = 0
                info = {}

                while not rospy.is_shutdown():
                    student_action, inference_ms = self.policy.predict(
                        observation
                    )
                    teacher_action = np.asarray(
                        self.environment.teacher_action(observation),
                        dtype=np.float32,
                    )
                    diagnostics = self.environment.teacher_diagnostics()
                    (
                        safe_teacher_action,
                        teacher_safety_info,
                    ) = self.environment.filter_fused_action(
                        teacher_action
                    )
                    teacher_filter_delta = (
                        safe_teacher_action - teacher_action
                    )
                    teacher_safety_active = bool(
                        np.max(np.abs(teacher_filter_delta))
                        > self.safety_delta_threshold
                    )
                    teacher_safety_reasons = _active_safety_reasons(
                        teacher_safety_info.get("safety_info", {})
                    )
                    deviation = float(np.max(np.abs(
                        student_action - safe_teacher_action
                    )))
                    random_intervention = bool(
                        self.rng.uniform() < self.teacher_beta
                    )
                    deviation_intervention = bool(
                        deviation > self.deviation_threshold
                    )
                    teacher_intervention = bool(
                        random_intervention or deviation_intervention
                    )
                    intervention_reason = (
                        "deviation"
                        if deviation_intervention
                        else "beta"
                        if random_intervention
                        else "student"
                    )
                    executed_action = (
                        safe_teacher_action
                        if teacher_intervention
                        else student_action
                    )
                    (
                        next_observation,
                        reward,
                        done,
                        info,
                    ) = self.environment.step(executed_action)
                    safe_executed_action = np.asarray(
                        info["safe_fused_action"],
                        dtype=np.float32,
                    )
                    execution_delta = (
                        safe_executed_action - executed_action
                    )
                    execution_safety_active = bool(
                        np.max(np.abs(execution_delta))
                        > self.safety_delta_threshold
                    )
                    execution_safety_reasons = _active_safety_reasons(
                        info.get("safety_info", {})
                    )
                    phase = str(
                        diagnostics.get("phase", "UNKNOWN")
                    )
                    blocked = "|".join(
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

                    episode["observations"].append(observation.copy())
                    episode["teacher_actions"].append(
                        teacher_action.copy()
                    )
                    episode["safe_teacher_actions"].append(
                        safe_teacher_action.copy()
                    )
                    episode["student_actions"].append(
                        student_action.copy()
                    )
                    episode["executed_actions"].append(
                        executed_action.copy()
                    )
                    episode["safe_executed_actions"].append(
                        safe_executed_action.copy()
                    )
                    episode["phase_ids"].append(phase_id(phase))
                    episode["phase_labels"].append(phase)
                    episode["box_path_indices"].append(int(
                        diagnostics.get("box_path_index", -1)
                    ))
                    episode["detour_sides"].append(float(
                        reset_info.get("detour_side", 0.0)
                    ))
                    episode["arm_blends"].append(float(
                        diagnostics.get("arm_blend", 0.0)
                    ))
                    episode["base_active"].append(base_active)
                    episode["arm_active"].append(arm_active)
                    episode["overlap_active"].append(
                        bool(base_active and arm_active)
                    )
                    episode["safety_filter_active"].append(
                        teacher_safety_active
                    )
                    episode["safety_reasons"].append(
                        "|".join(teacher_safety_reasons)
                    )
                    episode["teacher_limit_active"].append(bool(blocked))
                    episode["teacher_limit_blocked"].append(blocked)
                    episode["teacher_interventions"].append(
                        teacher_intervention
                    )
                    episode["intervention_reasons"].append(
                        intervention_reason
                    )
                    episode["student_teacher_max_deviation"].append(
                        deviation
                    )
                    episode["execution_safety_filter_active"].append(
                        execution_safety_active
                    )
                    episode["execution_safety_reasons"].append(
                        "|".join(execution_safety_reasons)
                    )
                    episode["inference_ms"].append(inference_ms)
                    episode["rewards"].append(float(reward))
                    episode["terminal"].append(bool(done))
                    episode["ee_distances"].append(float(
                        info.get("dist", float("nan"))
                    ))
                    episode["base_distances"].append(float(
                        info.get("fusion_state", {}).get(
                            "base_distance",
                            float("nan"),
                        )
                    ))
                    episode_interventions += int(
                        teacher_intervention
                    )

                    step = int(self.environment.base_env.step_count)
                    if (
                            step == 1
                            or step % self.log_interval == 0
                            or done):
                        rospy.loginfo(
                            "fused_dagger_step environment=%s "
                            "episode=%d side=%+.0f step=%d "
                            "ee_distance=%.4f base_distance=%.4f "
                            "deviation=%.3f execute=%s "
                            "student=%s teacher=%s safety=%s",
                            self.environment_type,
                            episode_index + 1,
                            episode_side,
                            step,
                            float(info.get("dist", float("nan"))),
                            float(
                                info.get("fusion_state", {}).get(
                                    "base_distance",
                                    float("nan"),
                                )
                            ),
                            deviation,
                            intervention_reason,
                            np.round(student_action, 3).tolist(),
                            np.round(safe_teacher_action, 3).tolist(),
                            str(execution_safety_active),
                        )
                    observation = next_observation
                    if done:
                        break

                sensor_state = self.environment.base_env.get_sensor_state()
                success = bool(info.get("success", False))
                collision = bool(
                    info.get("collision", False)
                    or info.get("collision_names", [])
                    or sensor_state[3]
                )
                timeout = bool(not success and not collision)
                successes += int(success)
                collisions += int(collision)
                timeouts += int(timeout)
                sample_count = len(episode["observations"])
                total_samples += sample_count
                total_interventions += episode_interventions
                self._append_episode(
                    episode_index,
                    episode,
                    reset_info,
                    info,
                    success,
                    collision,
                    timeout,
                    episode_side,
                )
                self._save_dataset(
                    successes,
                    collisions,
                    timeouts,
                )
                rospy.loginfo(
                    "fused_dagger_episode environment=%s episode=%d "
                    "side=%+.0f samples=%d success=%s "
                    "collision=%s timeout=%s interventions=%d "
                    "intervention_rate=%.3f total_samples=%d",
                    self.environment_type,
                    episode_index + 1,
                    episode_side,
                    sample_count,
                    str(success),
                    str(collision),
                    str(timeout),
                    episode_interventions,
                    (
                        episode_interventions / float(sample_count)
                        if sample_count else 0.0
                    ),
                    total_samples,
                )
        finally:
            try:
                self.environment.stop()
            finally:
                self.policy.close()

        rospy.loginfo(
            "Fused DAgger dataset complete: %s episodes=%d "
            "successes=%d collisions=%d timeouts=%d samples=%d "
            "teacher_intervention_rate=%.3f",
            self.output_path,
            self.episodes,
            successes,
            collisions,
            timeouts,
            total_samples,
            (
                total_interventions / float(total_samples)
                if total_samples else 0.0
            ),
        )

    def _append_episode(
            self,
            episode_index,
            episode,
            reset_info,
            final_info,
            success,
            collision,
            timeout,
            episode_side):
        sample_count = len(episode["observations"])
        for name, values in episode.items():
            self.samples[name].extend(values)
        self.samples["episode_ids"].extend(
            [episode_index] * sample_count
        )
        self.episode_metadata["success"].append(success)
        self.episode_metadata["collision"].append(collision)
        self.episode_metadata["timeout"].append(timeout)
        self.episode_metadata["detour_side"].append(float(episode_side))
        self.episode_metadata["steps"].append(sample_count)
        self.episode_metadata["initial_ee_distance"].append(
            float(reset_info["initial_ee_distance"])
        )
        self.episode_metadata["final_ee_distance"].append(
            float(final_info.get("dist", float("nan")))
        )
        self.episode_metadata["final_base_distance"].append(float(
            final_info.get("fusion_state", {}).get(
                "base_distance",
                float("nan"),
            )
        ))

    def _save_dataset(self, successes, collisions, timeouts):
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
        if observations.shape[0]:
            action_mask_contract = validate_action_mask_contract(
                observations,
                safe_teacher_actions,
                phase_ids,
            )
        else:
            action_mask_contract = {
                "mode": (
                    "three_stage"
                    if self.structured_action_masks
                    else "legacy_all_enabled"
                )
            }
        success_mask = np.asarray(
            self.episode_metadata["success"],
            dtype=np.bool_,
        )
        final_ee = np.asarray(
            self.episode_metadata["final_ee_distance"],
            dtype=np.float32,
        )
        final_base = np.asarray(
            self.episode_metadata["final_base_distance"],
            dtype=np.float32,
        )
        np.savez_compressed(
            self.output_path,
            observations=observations,
            teacher_actions=teacher_actions,
            safe_teacher_actions=safe_teacher_actions,
            student_actions=self._matrix("student_actions", ACTION_DIM),
            executed_actions=self._matrix(
                "executed_actions",
                ACTION_DIM,
            ),
            safe_executed_actions=self._matrix(
                "safe_executed_actions",
                ACTION_DIM,
            ),
            episode_ids=np.asarray(
                self.samples["episode_ids"],
                dtype=np.int32,
            ),
            phase_ids=phase_ids,
            phase_labels=np.asarray(
                self.samples["phase_labels"],
                dtype="S16",
            ),
            box_path_indices=np.asarray(
                self.samples["box_path_indices"], dtype=np.int16
            ),
            detour_sides=np.asarray(
                self.samples["detour_sides"], dtype=np.float32
            ),
            arm_blends=np.asarray(
                self.samples["arm_blends"],
                dtype=np.float32,
            ),
            base_active=np.asarray(
                self.samples["base_active"],
                dtype=np.bool_,
            ),
            arm_active=np.asarray(
                self.samples["arm_active"],
                dtype=np.bool_,
            ),
            overlap_active=np.asarray(
                self.samples["overlap_active"],
                dtype=np.bool_,
            ),
            safety_filter_active=np.asarray(
                self.samples["safety_filter_active"],
                dtype=np.bool_,
            ),
            safety_reasons=np.asarray(
                self.samples["safety_reasons"],
                dtype="S256",
            ),
            teacher_limit_active=np.asarray(
                self.samples["teacher_limit_active"],
                dtype=np.bool_,
            ),
            teacher_limit_blocked=np.asarray(
                self.samples["teacher_limit_blocked"],
                dtype="S128",
            ),
            teacher_interventions=np.asarray(
                self.samples["teacher_interventions"],
                dtype=np.bool_,
            ),
            intervention_reasons=np.asarray(
                self.samples["intervention_reasons"],
                dtype="S32",
            ),
            student_teacher_max_deviation=np.asarray(
                self.samples["student_teacher_max_deviation"],
                dtype=np.float32,
            ),
            execution_safety_filter_active=np.asarray(
                self.samples["execution_safety_filter_active"],
                dtype=np.bool_,
            ),
            execution_safety_reasons=np.asarray(
                self.samples["execution_safety_reasons"],
                dtype="S256",
            ),
            inference_ms=np.asarray(
                self.samples["inference_ms"],
                dtype=np.float32,
            ),
            rewards=np.asarray(
                self.samples["rewards"],
                dtype=np.float32,
            ),
            terminal=np.asarray(
                self.samples["terminal"],
                dtype=np.bool_,
            ),
            ee_distances=np.asarray(
                self.samples["ee_distances"],
                dtype=np.float32,
            ),
            base_distances=np.asarray(
                self.samples["base_distances"],
                dtype=np.float32,
            ),
            observation_dim=np.asarray([OBSERVATION_DIM], dtype=np.int32),
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
            action_names=np.asarray(ACTION_NAMES, dtype="S32"),
            teacher_type=np.asarray(
                ["coordinated_rule_teacher"],
                dtype="S64",
            ),
            policy_target=np.asarray(
                ["safe_teacher_actions"],
                dtype="S64",
            ),
            dataset_type=np.asarray(["fused_low_dagger"], dtype="S64"),
            successful_episodes=np.asarray([successes], dtype=np.int32),
            attempted_episodes=np.asarray(
                [len(self.episode_metadata["success"])],
                dtype=np.int32,
            ),
            collision_episodes=np.asarray(
                [collisions],
                dtype=np.int32,
            ),
            timeout_episodes=np.asarray([timeouts], dtype=np.int32),
            accepted_final_ee_distances=final_ee[success_mask],
            accepted_final_base_distances=final_base[success_mask],
            episode_final_ee_distances=final_ee,
            episode_final_base_distances=final_base,
            episode_success=success_mask,
            episode_collision=np.asarray(
                self.episode_metadata["collision"],
                dtype=np.bool_,
            ),
            episode_timeout=np.asarray(
                self.episode_metadata["timeout"],
                dtype=np.bool_,
            ),
            teacher_beta=np.asarray(
                [self.teacher_beta],
                dtype=np.float32,
            ),
            deviation_threshold=np.asarray(
                [self.deviation_threshold],
                dtype=np.float32,
            ),
            environment_type=np.asarray(
                [self.environment_type],
                dtype="S32",
            ),
            configured_detour_side=np.asarray(
                [self.detour_side],
                dtype=np.float32,
            ),
            max_steps=np.asarray([self.max_steps], dtype=np.int32),
            episode_detour_side=np.asarray(
                self.episode_metadata["detour_side"],
                dtype=np.float32,
            ),
        )

    def _matrix(self, name, columns):
        values = np.asarray(self.samples[name], dtype=np.float32)
        return values.reshape((-1, columns))

    def _validate_parameters(self):
        if self.episodes <= 0 or self.log_interval <= 0:
            raise ValueError("episodes and log_interval must be positive")
        if not 0.0 <= self.teacher_beta <= 1.0:
            raise ValueError("teacher_beta must be in [0, 1]")
        if not 0.0 <= self.deviation_threshold <= 2.0:
            raise ValueError("deviation_threshold must be in [0, 2]")
        if self.safety_delta_threshold < 0.0:
            raise ValueError("safety_delta_threshold must be non-negative")
        if self.environment_type not in (
                "fused_low",
                "box_detour",
                "fused_box_detour"):
            raise ValueError(
                "unsupported DAgger environment_type: {}".format(
                    self.environment_type
                )
            )
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


def main():
    rospy.init_node("collect_fused_low_dagger_dataset")
    FusedLowDaggerCollector().collect()


if __name__ == "__main__":
    main()
