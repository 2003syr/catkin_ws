#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Closed-loop Gazebo evaluation for fused 66-D/8-D low policies."""

from __future__ import division, print_function

import collections
import json
import os
import socket
import sys

import numpy as np
import rospy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_low_training_env import FusedLowLevelTrainingEnv


class PolicyServiceClient(object):

    def __init__(self, host, port, timeout):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self._socket = socket.create_connection(
            (self.host, self.port),
            self.timeout,
        )
        self._socket.settimeout(self.timeout)
        self._reader = self._socket.makefile("rb")
        metadata = self.request({"command": "ping"})
        if int(metadata.get("observation_dim", -1)) != 66:
            raise RuntimeError("fused policy observation_dim is not 66")
        if int(metadata.get("action_dim", -1)) != 8:
            raise RuntimeError("fused policy action_dim is not 8")
        if str(metadata.get("policy_type", "")) not in (
                "fused_low_bc",
                "fused_low_teacher_ppo",
                "fused_low_bc_hrl4in_arm"):
            raise RuntimeError(
                "policy server is not serving a supported fused checkpoint"
            )
        self.metadata = metadata

    def predict(self, observation):
        response = self.request({
            "command": "predict",
            "observation": np.asarray(
                observation,
                dtype=np.float32,
            ).tolist(),
        })
        action = np.asarray(response["action"], dtype=np.float32)
        if action.shape != (8,):
            raise RuntimeError(
                "policy server returned action shape {}".format(
                    action.shape
                )
            )
        if not np.all(np.isfinite(action)):
            raise RuntimeError("policy server returned non-finite action")
        return (
            np.clip(action, -1.0, 1.0),
            float(response.get("inference_ms", 0.0)),
        )

    def request(self, request):
        payload = json.dumps(request, separators=(",", ":")) + "\n"
        self._socket.sendall(payload.encode("utf-8"))
        response_line = self._reader.readline()
        if not response_line:
            raise RuntimeError("policy server closed the connection")
        response = json.loads(response_line.decode("utf-8"))
        if not bool(response.get("ok", False)):
            raise RuntimeError(
                "policy server error: {}".format(
                    response.get("error", "unknown error")
                )
            )
        return response

    def close(self):
        try:
            self.request({"command": "close"})
        except Exception:
            pass
        try:
            self._reader.close()
        finally:
            self._socket.close()


def main():
    rospy.init_node("evaluate_fused_low_bc")
    episodes = int(rospy.get_param("~episodes", 20))
    minimum_success_rate = float(
        rospy.get_param("~minimum_success_rate", 0.90)
    )
    maximum_collisions = int(
        rospy.get_param("~maximum_collisions", 0)
    )
    minimum_overlap_rate = float(
        rospy.get_param("~minimum_overlap_rate", 0.50)
    )
    maximum_safety_rate = float(
        rospy.get_param("~maximum_safety_rate", 0.05)
    )
    safety_delta_threshold = float(
        rospy.get_param("~safety_delta_threshold", 0.01)
    )
    log_interval = int(rospy.get_param("~log_interval", 25))
    policy = PolicyServiceClient(
        host=rospy.get_param("~policy_host", "127.0.0.1"),
        port=rospy.get_param("~policy_port", 5563),
        timeout=rospy.get_param("~policy_timeout", 5.0),
    )
    environment = FusedLowLevelTrainingEnv(
        init_ros_node=False,
        seed=rospy.get_param("~seed", 100123),
        enable_teacher=True,
    )
    _validate_parameters(
        episodes,
        minimum_success_rate,
        maximum_collisions,
        minimum_overlap_rate,
        maximum_safety_rate,
        safety_delta_threshold,
        log_interval,
    )

    successes = 0
    collisions = 0
    timeouts = 0
    total_steps = 0
    overlap_steps = 0
    safety_steps = 0
    final_distances = []
    base_final_distances = []
    inference_times = []
    teacher_squared_error = np.zeros(8, dtype=np.float64)
    absolute_filter_delta = np.zeros(8, dtype=np.float64)
    maximum_filter_delta = np.zeros(8, dtype=np.float64)
    safety_reasons = collections.Counter()

    rospy.loginfo(
        "fused_bc_checkpoint=%s policy_type=%s "
        "validation_weighted_mse=%s arm_source_steps=%s "
        "arm_transferred_frozen=%s arm_finetuned=%s "
        "arm_max_subgoal=%s",
        str(policy.metadata.get("checkpoint", "unknown")),
        str(policy.metadata.get("policy_type", "unknown")),
        str(policy.metadata.get("validation_weighted_mse", "unknown")),
        str(policy.metadata.get("arm_source_total_steps", "none")),
        str(policy.metadata.get("arm_transferred_frozen", False)),
        str(policy.metadata.get("arm_finetuned", False)),
        str(policy.metadata.get("arm_max_subgoal", "none")),
    )
    try:
        for episode_index in range(episodes):
            observation = environment.reset()
            reset_info = dict(environment.last_reset_info)
            initial_distance = float(reset_info["initial_ee_distance"])
            episode_overlap = 0
            episode_safety = 0
            episode_reward = 0.0
            info = {}

            while not rospy.is_shutdown():
                action, inference_ms = policy.predict(observation)
                teacher_action = environment.teacher_action(observation)
                (
                    observation,
                    reward,
                    done,
                    info,
                ) = environment.step(action)
                safe_action = np.asarray(
                    info["safe_fused_action"],
                    dtype=np.float32,
                )
                action_delta = safe_action - action
                absolute_delta = np.abs(
                    action_delta.astype(np.float64)
                )
                absolute_filter_delta += absolute_delta
                maximum_filter_delta = np.maximum(
                    maximum_filter_delta,
                    absolute_delta,
                )
                safety_active = bool(
                    np.max(absolute_delta)
                    > safety_delta_threshold
                )
                base_active = bool(
                    abs(float(safe_action[0])) > 0.05
                    or abs(float(safe_action[1])) > 0.05
                )
                arm_active = bool(
                    np.max(np.abs(safe_action[2:8])) > 0.02
                )
                overlap_active = bool(base_active and arm_active)
                total_steps += 1
                overlap_steps += int(overlap_active)
                safety_steps += int(safety_active)
                episode_overlap += int(overlap_active)
                episode_safety += int(safety_active)
                episode_reward += float(reward)
                inference_times.append(inference_ms)
                teacher_squared_error += (
                    action.astype(np.float64)
                    - np.asarray(
                        teacher_action,
                        dtype=np.float64,
                    )
                ) ** 2
                if safety_active:
                    reasons = _active_safety_reasons(
                        info.get("safety_info", {})
                    )
                    if not reasons:
                        reasons = ["ACTION_CLIPPED"]
                    for reason in reasons:
                        safety_reasons[reason] += 1

                episode_step = int(environment.base_env.step_count)
                if (
                        episode_step == 1
                        or episode_step % log_interval == 0
                        or done):
                    fusion = info.get("fusion_state", {})
                    rospy.loginfo(
                        "fused_bc_step episode=%d step=%d "
                        "ee_distance=%.4f base_distance=%.4f "
                        "action=%s safe=%s overlap=%s safety=%s "
                        "filter_delta=%s inference_ms=%.3f",
                        episode_index + 1,
                        episode_step,
                        float(info.get("dist", float("nan"))),
                        float(
                            fusion.get("base_distance", float("nan"))
                        ),
                        np.round(action, 3).tolist(),
                        np.round(safe_action, 3).tolist(),
                        str(overlap_active),
                        str(safety_active),
                        np.round(action_delta, 4).tolist(),
                        inference_ms,
                    )
                if done:
                    break

            success = bool(info.get("success", False))
            sensor_state = environment.base_env.get_sensor_state()
            collision = bool(
                info.get("collision", False)
                or info.get("collision_names", [])
                or sensor_state[3]
            )
            timed_out = bool(not success and not collision)
            successes += int(success)
            collisions += int(collision)
            timeouts += int(timed_out)
            final_distance = float(
                info.get("dist", float("nan"))
            )
            base_final_distance = float(
                info.get("fusion_state", {}).get(
                    "base_distance",
                    float("nan"),
                )
            )
            final_distances.append(final_distance)
            base_final_distances.append(base_final_distance)
            rospy.loginfo(
                "fused_bc_episode=%d distance=%.4f->%.4f "
                "base_final=%.4f reward=%.4f success=%s collision=%s "
                "timeout=%s steps=%d overlap_steps=%d safety_steps=%d",
                episode_index + 1,
                initial_distance,
                final_distance,
                base_final_distance,
                episode_reward,
                str(success),
                str(collision),
                str(timed_out),
                int(environment.base_env.step_count),
                episode_overlap,
                episode_safety,
            )
    finally:
        try:
            environment.stop()
        finally:
            policy.close()

    success_rate = successes / float(episodes)
    overlap_rate = (
        overlap_steps / float(total_steps)
        if total_steps else 0.0
    )
    safety_rate = (
        safety_steps / float(total_steps)
        if total_steps else 0.0
    )
    teacher_action_mse = (
        teacher_squared_error / float(total_steps)
        if total_steps else np.full(8, float("nan"))
    )
    mean_absolute_filter_delta = (
        absolute_filter_delta / float(total_steps)
        if total_steps else np.full(8, float("nan"))
    )
    gate_pass = bool(
        success_rate >= minimum_success_rate
        and collisions <= maximum_collisions
        and overlap_rate >= minimum_overlap_rate
        and safety_rate <= maximum_safety_rate
    )
    rospy.loginfo(
        "fused_bc_evaluation episodes=%d successes=%d "
        "success_rate=%.3f collisions=%d timeouts=%d "
        "mean_final_distance=%.4f mean_base_final_distance=%.4f "
        "overlap_rate=%.3f safety_rate=%.3f safety_reasons=%s "
        "mean_filter_delta=%s max_filter_delta=%s "
        "teacher_action_mse=%s mean_inference_ms=%.3f",
        episodes,
        successes,
        success_rate,
        collisions,
        timeouts,
        float(np.mean(final_distances)),
        float(np.mean(base_final_distances)),
        overlap_rate,
        safety_rate,
        dict(safety_reasons),
        np.round(mean_absolute_filter_delta, 7).tolist(),
        np.round(maximum_filter_delta, 7).tolist(),
        np.round(teacher_action_mse, 7).tolist(),
        float(np.mean(inference_times)),
    )
    if gate_pass:
        rospy.loginfo("fused_bc_gate_pass=True")
        return
    rospy.logerr(
        "fused_bc_gate_pass=False required_success_rate=%.3f "
        "maximum_collisions=%d minimum_overlap_rate=%.3f "
        "maximum_safety_rate=%.3f",
        minimum_success_rate,
        maximum_collisions,
        minimum_overlap_rate,
        maximum_safety_rate,
    )
    raise SystemExit(1)


def _validate_parameters(
        episodes,
        minimum_success_rate,
        maximum_collisions,
        minimum_overlap_rate,
        maximum_safety_rate,
        safety_delta_threshold,
        log_interval):
    if episodes <= 0 or log_interval <= 0:
        raise ValueError("episodes and log_interval must be positive")
    for name, value in (
            ("minimum_success_rate", minimum_success_rate),
            ("minimum_overlap_rate", minimum_overlap_rate),
            ("maximum_safety_rate", maximum_safety_rate)):
        if not 0.0 <= value <= 1.0:
            raise ValueError("{} must be in [0, 1]".format(name))
    if maximum_collisions < 0:
        raise ValueError("maximum_collisions must be non-negative")
    if safety_delta_threshold < 0.0:
        raise ValueError("safety_delta_threshold must be non-negative")


def _active_safety_reasons(safety_info):
    """Return only joints whose safety filter actually changed a command."""
    if not isinstance(safety_info, dict):
        return []
    reasons = []
    for joint_name, joint_info in safety_info.items():
        if not isinstance(joint_info, dict):
            continue
        reason = str(joint_info.get("reason", "safe"))
        raw_command = float(joint_info.get("cmd_raw", 0.0))
        safe_command = float(
            joint_info.get("cmd_safe", raw_command)
        )
        if abs(raw_command - safe_command) > 1.0e-9:
            reasons.append("{}:{}".format(joint_name, reason))
    return sorted(set(reasons))


if __name__ == "__main__":
    main()
