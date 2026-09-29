#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the minimal rule-high/frozen-low two-level HRL system."""

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

from hrl.high_level_command import JointHighLevelCommand
from hrl.high_level_env import HighLevelEnv
from hrl.low_level_wrapper import (
    FrozenFusedLowPolicy,
    FrozenFusedResidualPolicy,
    JointSubgoalLowLevelWrapper,
    JointSubgoalResidualLowLevelWrapper,
)
from hrl.rule_high_policy import RuleBasedHighPolicy
from hrl.rule_high_policy import SafeWaypointHighPolicy
from training.fused_box_detour_training_env import (
    FusedBoxDetourTrainingEnv,
)


def main():
    rospy.init_node("evaluate_joint_subgoal_hrl")
    episodes = int(rospy.get_param("~episodes", 2))
    max_steps = int(rospy.get_param("~max_steps", 1500))
    high_level_interval = int(
        rospy.get_param("~high_level_interval", 20)
    )
    base_subgoal_tolerance = float(
        rospy.get_param("~base_subgoal_tolerance", 0.04)
    )
    yaw_subgoal_tolerance = float(
        rospy.get_param("~yaw_subgoal_tolerance", 0.08)
    )
    ee_subgoal_tolerance = float(
        rospy.get_param("~ee_subgoal_tolerance", 0.05)
    )
    subgoal_stable_cycles = int(
        rospy.get_param("~subgoal_stable_cycles", 3)
    )
    minimum_success_rate = float(
        rospy.get_param("~minimum_success_rate", 0.0)
    )
    minimum_overlap_rate = float(
        rospy.get_param("~minimum_overlap_rate", 0.0)
    )
    maximum_collisions = int(
        rospy.get_param("~maximum_collisions", episodes)
    )
    detour_side = float(
        rospy.get_param("~evaluation_detour_side", 0.0)
    )
    no_obstacle = bool(rospy.get_param("~no_obstacle", False))
    low_policy_type = str(
        rospy.get_param("~low_policy_type", "fused")
    ).lower()
    high_policy_type = str(
        rospy.get_param("~high_policy_type", "scan_rule")
    ).lower()
    minimum_probe_action_delta = float(
        rospy.get_param("~minimum_probe_action_delta", 1.0e-4)
    )
    if episodes <= 0 or max_steps <= 0 or high_level_interval <= 0:
        raise ValueError("episode and step counts must be positive")
    if not 0.0 <= minimum_success_rate <= 1.0:
        raise ValueError("minimum_success_rate must be in [0, 1]")
    if not 0.0 <= minimum_overlap_rate <= 1.0:
        raise ValueError("minimum_overlap_rate must be in [0, 1]")
    if maximum_collisions < 0 or minimum_probe_action_delta < 0.0:
        raise ValueError("invalid evaluation threshold")

    use_residual_policy = low_policy_type == "residual"
    if low_policy_type not in ("fused", "residual"):
        raise ValueError(
            "low_policy_type must be fused or residual, got {}".format(
                low_policy_type
            )
        )
    if high_policy_type not in ("scan_rule", "safe_waypoint"):
        raise ValueError(
            "high_policy_type must be scan_rule or safe_waypoint, got {}".format(
                high_policy_type
            )
        )
    environment = FusedBoxDetourTrainingEnv(
        init_ros_node=False,
        seed=rospy.get_param("~seed", 100123),
        enable_teacher=use_residual_policy,
    )
    policy_host = rospy.get_param("~policy_host", "127.0.0.1")
    policy_port = rospy.get_param("~policy_port", 5563)
    policy_timeout = rospy.get_param("~policy_timeout", 5.0)
    if use_residual_policy:
        frozen_policy = FrozenFusedResidualPolicy(
            host=policy_host,
            port=policy_port,
            timeout=policy_timeout,
        )
        low_wrapper = JointSubgoalResidualLowLevelWrapper(
            frozen_policy,
            environment.teacher,
            residual_base_scale=rospy.get_param(
                "~residual_base_scale", 0.10
            ),
            residual_cartesian_scale=rospy.get_param(
                "~residual_cartesian_scale", 0.01
            ),
            encode_subgoal_type=bool(
                frozen_policy.metadata.get("subgoal_context_contract")
                == JointSubgoalLowLevelWrapper.SUBGOAL_CONTEXT_CONTRACT
            ),
        )
    else:
        frozen_policy = FrozenFusedLowPolicy(
            host=policy_host,
            port=policy_port,
            timeout=policy_timeout,
        )
        low_wrapper = JointSubgoalLowLevelWrapper(frozen_policy)
    high_environment = HighLevelEnv(
        low_environment=environment,
        low_wrapper=low_wrapper,
        high_level_interval=high_level_interval,
        scan_clip=rospy.get_param("~scan_output_max", 10.0),
        base_subgoal_tolerance=base_subgoal_tolerance,
        yaw_subgoal_tolerance=yaw_subgoal_tolerance,
        ee_subgoal_tolerance=ee_subgoal_tolerance,
        subgoal_stable_cycles=subgoal_stable_cycles,
    )
    if high_policy_type == "safe_waypoint":
        high_policy = SafeWaypointHighPolicy(environment)
    else:
        high_policy = RuleBasedHighPolicy()

    successes = 0
    collisions = 0
    timeouts = 0
    total_low_steps = 0
    total_overlap_steps = 0
    total_safety_steps = 0
    residual_magnitudes = []
    final_distances = []
    base_probe_deltas = []
    arm_probe_deltas = []
    safety_reasons = collections.Counter()
    action_gate_reasons = collections.Counter()
    low_teacher_phases = collections.Counter()
    subgoal_type_steps = collections.Counter()
    subgoal_type_options = collections.Counter()
    subgoal_type_completions = collections.Counter()

    rospy.loginfo(
        "joint_hrl_start low_policy_type=%s high_policy_type=%s "
        "policy=%s episodes=%d high_interval=%d no_obstacle=%s",
        low_policy_type,
        high_policy_type,
        str(frozen_policy.metadata.get("checkpoint", "unknown")),
        episodes,
        high_level_interval,
        str(no_obstacle),
    )
    try:
        for episode_index in range(episodes):
            episode_side = detour_side
            if abs(episode_side) < 0.5:
                episode_side = 1.0 if episode_index % 2 == 0 else -1.0
            scenario = {
                "scenario_id": "joint_hrl_eval_{:03d}".format(
                    episode_index + 1
                ),
                "category": "joint_subgoal_hrl",
                "detour_side": episode_side,
                "no_obstacle": no_obstacle,
            }
            high_observation = high_environment.reset(
                scenario=scenario,
                max_steps=max_steps,
            )
            # On a symmetric box either bypass is valid.  Seed the rule
            # policy with the requested evaluation side so alternating
            # episodes actually exercise both upper and lower detours.
            high_policy.reset(preferred_turn_sign=episode_side)
            probe = _verify_subgoal_response(
                low_wrapper,
                environment.last_sensor_observation,
            )
            base_probe_deltas.append(probe["base_action_delta"])
            arm_probe_deltas.append(probe["arm_action_delta"])
            if (
                    probe["base_action_delta"]
                    <= minimum_probe_action_delta):
                raise RuntimeError(
                    "low policy ignored distinct base subgoals: "
                    "base action delta={:.8f}".format(
                        probe["base_action_delta"]
                    )
                )
            if (
                    probe["arm_action_delta"]
                    <= minimum_probe_action_delta):
                raise RuntimeError(
                    "low policy ignored distinct EE subgoals: "
                    "arm action delta={:.8f}".format(
                        probe["arm_action_delta"]
                    )
                )

            episode_reward = 0.0
            episode_low_steps = 0
            episode_overlap_steps = 0
            episode_safety_steps = 0
            info = {}
            done = False
            while not rospy.is_shutdown() and not done:
                command = high_policy.predict(high_observation)
                high_observation, reward, done, info = (
                    high_environment.step(command)
                )
                episode_reward += float(reward)
                episode_low_steps += int(info["low_steps"])
                episode_overlap_steps += int(info["overlap_steps"])
                episode_safety_steps += int(info["safety_steps"])
                if use_residual_policy:
                    residual_magnitudes.append(float(np.mean(np.abs(
                        np.asarray(
                            info.get(
                                "low_residual_mean_abs",
                                np.zeros(5),
                            ),
                            dtype=np.float32,
                        )
                    ))))
                safety_reasons.update(info.get("safety_reasons", {}))
                action_gate_reasons.update(
                    info.get("box_action_gate_reasons", [])
                )
                phase = info.get("low_teacher_phase")
                if phase:
                    low_teacher_phases[str(phase)] += int(
                        info.get("low_steps", 0)
                    )
                subgoal_type_name = str(
                    info.get("subgoal_type_name", command.subgoal_type_name)
                )
                subgoal_type_steps[subgoal_type_name] += int(
                    info.get("low_steps", 0)
                )
                subgoal_type_options[subgoal_type_name] += 1
                subgoal_type_completions[subgoal_type_name] += int(
                    bool(info.get("subgoal_done", False))
                )
                rospy.loginfo(
                    "joint_hrl_step episode=%d high_step=%d low_steps=%d "
                    "command=%s base_goal_world=%s ee_goal_world=%s "
                    "base_pose=%s error=%s reward=%.4f overlap=%d "
                    "safety=%d safety_reasons=%s gate=%s phase=%s "
                    "subgoal_done=%s type=%s reason=%s",
                    episode_index + 1,
                    int(info["high_step"]),
                    int(info["low_steps"]),
                    np.round(command.subgoal, 3).tolist(),
                    np.round(info["fixed_base_goal_world"], 3).tolist(),
                    np.round(info["fixed_ee_goal_world"], 3).tolist(),
                    np.round(
                        high_observation["base_pose_world"], 3
                    ).tolist(),
                    np.round(info["end_subgoal_error"], 3).tolist(),
                    float(reward),
                    int(info["overlap_steps"]),
                    int(info["safety_steps"]),
                    str(info.get("safety_reasons", {})),
                    str(info.get("box_action_gate_reasons", [])),
                    str(info.get("low_teacher_phase", "unknown")),
                    str(info.get("subgoal_done", False)),
                    subgoal_type_name,
                    command.reason,
                )

            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            timeout = bool(info.get("timeout", False))
            successes += int(success)
            collisions += int(collision)
            timeouts += int(timeout)
            total_low_steps += episode_low_steps
            total_overlap_steps += episode_overlap_steps
            total_safety_steps += episode_safety_steps
            final_distance = float(info.get("dist", float("nan")))
            final_distances.append(final_distance)
            rospy.loginfo(
                "joint_hrl_episode=%d side=%+.0f reward=%.4f "
                "success=%s collision=%s timeout=%s high_steps=%d "
                "low_steps=%d final_distance=%.4f overlap_rate=%.3f "
                "safety_rate=%.3f base_probe_delta=%.6f "
                "arm_probe_delta=%.6f collision_source=%s "
                "collision_names=%s residual_abs=%.6f",
                episode_index + 1,
                episode_side,
                episode_reward,
                str(success),
                str(collision),
                str(timeout),
                int(info.get("high_step", 0)),
                episode_low_steps,
                final_distance,
                _rate(episode_overlap_steps, episode_low_steps),
                _rate(episode_safety_steps, episode_low_steps),
                probe["base_action_delta"],
                probe["arm_action_delta"],
                str(info.get("collision_source", "none")),
                str(info.get("collision_names", [])),
                float(
                    np.mean(np.abs(np.asarray(
                        info.get(
                            "low_residual_mean_abs",
                            np.zeros(5),
                        ),
                        dtype=np.float32,
                    )))
                ),
            )
    finally:
        high_environment.close()

    success_rate = _rate(successes, episodes)
    overlap_rate = _rate(total_overlap_steps, total_low_steps)
    rospy.loginfo(
        "joint_hrl_evaluation episodes=%d successes=%d success_rate=%.3f "
        "collisions=%d timeouts=%d mean_final_distance=%.4f "
        "overlap_rate=%.3f safety_rate=%.3f "
        "mean_base_probe_delta=%.6f mean_arm_probe_delta=%.6f "
        "residual_abs=%.6f safety_reasons=%s "
        "action_gate_reasons=%s low_teacher_phases=%s "
        "subgoal_type_steps=%s subgoal_type_completion=%s",
        episodes,
        successes,
        success_rate,
        collisions,
        timeouts,
        float(np.nanmean(final_distances)),
        overlap_rate,
        _rate(total_safety_steps, total_low_steps),
        float(np.mean(base_probe_deltas)),
        float(np.mean(arm_probe_deltas)),
        float(np.mean(residual_magnitudes))
        if residual_magnitudes else 0.0,
        str(dict(safety_reasons)),
        str(dict(action_gate_reasons)),
        str(dict(low_teacher_phases)),
        str(dict(subgoal_type_steps)),
        str(dict(
            (
                name,
                round(
                    float(subgoal_type_completions[name])
                    / float(max(subgoal_type_options[name], 1)),
                    3,
                ),
            )
            for name in sorted(subgoal_type_options)
        )),
    )
    if success_rate + 1.0e-12 < minimum_success_rate:
        raise RuntimeError("joint HRL success rate is below threshold")
    if overlap_rate + 1.0e-12 < minimum_overlap_rate:
        raise RuntimeError("joint HRL overlap rate is below threshold")
    if collisions > maximum_collisions:
        raise RuntimeError("joint HRL collision count exceeds threshold")


def _verify_subgoal_response(low_wrapper, sensor):
    base_first = JointHighLevelCommand(
        base_goal=[0.30, 0.10, 0.25],
        ee_goal=[0.03, 0.00, 0.00],
        reason="subgoal_probe_upper",
    )
    base_second = JointHighLevelCommand(
        base_goal=[0.30, -0.10, -0.25],
        ee_goal=[0.03, 0.00, 0.00],
        reason="subgoal_probe_lower",
    )
    arm_first = JointHighLevelCommand(
        # The HRL subgoal teacher releases the arm only after the base is
        # inside its terminal position tube.  Use a zero base displacement so
        # this probe reaches ARM_FINISH immediately instead of testing the
        # intentionally arm-held drive phase.
        base_goal=[0.00, 0.00, 0.00],
        ee_goal=[0.05, 0.00, 0.00],
        reason="subgoal_probe_arm_x",
    )
    arm_second = JointHighLevelCommand(
        base_goal=[0.00, 0.00, 0.00],
        ee_goal=[0.00, 0.05, 0.00],
        reason="subgoal_probe_arm_y",
    )
    low_wrapper.begin_subgoal(base_first, sensor)
    base_first_action = low_wrapper.predict(sensor)
    low_wrapper.begin_subgoal(base_second, sensor)
    base_second_action = low_wrapper.predict(sensor)
    low_wrapper.begin_subgoal(arm_first, sensor)
    arm_first_action = low_wrapper.predict(sensor)
    low_wrapper.begin_subgoal(arm_second, sensor)
    arm_second_action = low_wrapper.predict(sensor)
    return {
        "base_action_delta": float(np.max(np.abs(
            base_first_action[0:2] - base_second_action[0:2]
        ))),
        "arm_action_delta": float(np.max(np.abs(
            arm_first_action[2:8] - arm_second_action[2:8]
        ))),
    }


def _rate(numerator, denominator):
    return float(numerator) / max(float(denominator), 1.0)


if __name__ == "__main__":
    main()
