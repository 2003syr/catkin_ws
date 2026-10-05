#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ROS/Gazebo JSON environment for learned high-level subgoals.

The server owns the complete simulator-facing hierarchy.  The low layer is
the frozen coordinated rule teacher plus a frozen residual PPO policy; the
client controls the semantic stage, route option, and normalized
six-dimensional high action.
"""

from __future__ import print_function

import json
import math
import os
import socket
import sys
import traceback

import numpy as np
import rospy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from hrl.high_level_env import HighLevelEnv  # noqa: E402
from hrl.high_level_command import (  # noqa: E402
    HighLevelOption,
    JointHighLevelCommand,
    RouteSide,
    SubgoalType,
)
from hrl.high_level_reward import HighLevelReward  # noqa: E402
from hrl.low_level_wrapper import (  # noqa: E402
    FrozenFusedResidualPolicy,
    JointSubgoalResidualLowLevelWrapper,
    ZeroFusedResidualPolicy,
)
from hrl.rule_high_policy import SafeWaypointHighPolicy  # noqa: E402
from training.fused_box_detour_training_env import (  # noqa: E402
    FusedBoxDetourTrainingEnv,
)


class FusedHighEnvironmentServer(object):

    def __init__(self):
        self.host = str(rospy.get_param("~host", "127.0.0.1"))
        self.port = int(rospy.get_param("~port", 5565))
        self.basic_hierarchy = bool(rospy.get_param(
            "~basic_hierarchy", False
        ))
        self.low_environment = FusedBoxDetourTrainingEnv(
            init_ros_node=False,
            seed=rospy.get_param("~seed", 789),
            enable_teacher=True,
        )
        self.low_policy_type = str(rospy.get_param(
            "~low_policy_type", "residual_ppo"
        )).strip().lower()
        if self.low_policy_type in ("residual_ppo", "ppo", "residual"):
            self.low_policy = FrozenFusedResidualPolicy(
                host=str(rospy.get_param(
                    "~low_policy_host", "127.0.0.1"
                )),
                port=int(rospy.get_param("~low_policy_port", 5573)),
                timeout=float(rospy.get_param(
                    "~low_policy_timeout", 10.0
                )),
            )
            low_contract = str(self.low_policy.metadata.get(
                "subgoal_context_contract",
                "legacy_all_ones_subgoal_mask",
            ))
            if low_contract != "three_class_in_subgoal_mask_v1":
                self.low_policy.close()
                raise ValueError(
                    "low PPO checkpoint uses incompatible subgoal context: "
                    "{}".format(low_contract)
                )
        elif self.low_policy_type in ("zero", "zero_residual", "rule"):
            self.low_policy = ZeroFusedResidualPolicy()
        else:
            raise ValueError(
                "unknown low_policy_type: {}".format(self.low_policy_type)
            )
        self.low_wrapper = JointSubgoalResidualLowLevelWrapper(
            self.low_policy,
            self.low_environment.teacher,
            residual_base_scale=float(rospy.get_param(
                "~residual_base_scale", 0.04
            )),
            residual_cartesian_scale=float(rospy.get_param(
                "~residual_cartesian_scale", 0.004
            )),
            encode_subgoal_type=True,
        )
        reward_model = HighLevelReward(
            progress_weight=float(rospy.get_param(
                "~high_ee_progress_weight", 2.0
            )),
            path_progress_weight=float(rospy.get_param(
                "~high_path_progress_weight", 10.0
            )),
            final_base_progress_weight=float(rospy.get_param(
                "~high_final_base_progress_weight", 0.0
            )),
            final_yaw_progress_weight=float(rospy.get_param(
                "~high_final_yaw_progress_weight", 0.0
            )),
            terminal_progress_weight=float(rospy.get_param(
                "~high_terminal_ee_progress_weight", 0.0
            )),
            success_reward=float(rospy.get_param(
                "~high_success_reward", 100.0
            )),
            collision_penalty=float(rospy.get_param(
                "~high_collision_penalty", 100.0
            )),
            timeout_penalty=float(rospy.get_param(
                "~high_timeout_penalty", 10.0
            )),
            subgoal_failure_penalty=float(rospy.get_param(
                "~high_subgoal_failure_penalty", 5.0
            )),
            minimum_subgoal_progress=float(rospy.get_param(
                "~high_minimum_subgoal_progress", 0.01
            )),
            low_step_penalty=float(rospy.get_param(
                "~high_low_step_penalty", 0.0
            )),
            option_stall_penalty=float(rospy.get_param(
                "~high_option_stall_penalty", 0.0
            )),
            invalid_terminal_penalty=float(rospy.get_param(
                "~high_invalid_terminal_penalty", 0.0
            )),
        )
        self.environment = HighLevelEnv(
            self.low_environment,
            self.low_wrapper,
            high_level_interval=int(rospy.get_param(
                "~high_level_interval", 20
            )),
            reward_model=reward_model,
            scan_clip=float(rospy.get_param("~scan_output_max", 10.0)),
            base_subgoal_tolerance=float(rospy.get_param(
                "~base_subgoal_tolerance", 0.04
            )),
            yaw_subgoal_tolerance=float(rospy.get_param(
                "~yaw_subgoal_tolerance", 0.08
            )),
            ee_subgoal_tolerance=float(rospy.get_param(
                "~ee_subgoal_tolerance", 0.05
            )),
            subgoal_stable_cycles=int(rospy.get_param(
                "~subgoal_stable_cycles", 3
            )),
            option_min_low_steps=int(rospy.get_param(
                "~option_min_low_steps", 10
            )),
            option_stall_steps=int(rospy.get_param(
                "~option_stall_steps", 20
            )),
            option_minimum_progress=float(rospy.get_param(
                "~option_minimum_progress", 0.002
            )),
            option_safety_window=int(rospy.get_param(
                "~option_safety_window", 10
            )),
            option_safety_rate_threshold=float(rospy.get_param(
                "~option_safety_rate_threshold", 0.8
            )),
            enable_route_options=not self.basic_hierarchy,
            terminal_option_contract=bool(rospy.get_param(
                "~terminal_option_contract", self.basic_hierarchy
            )),
            terminal_ee_step=float(rospy.get_param(
                "~terminal_ee_step", 0.08
            )),
            terminal_student_blend=float(rospy.get_param(
                "~terminal_student_blend", 0.25
            )),
        )
        self.teacher = SafeWaypointHighPolicy(
            self.low_environment,
            base_step=float(rospy.get_param(
                "~high_teacher_base_step", 0.30
            )),
            ee_step=float(rospy.get_param(
                "~high_teacher_ee_step", 0.08
            )),
        )
        self.running = True
        self.server_socket = None
        self.last_teacher_action = np.zeros(
            self.environment.ACTION_DIM, dtype=np.float32
        )
        self.last_teacher_info = {}

    def serve_forever(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(1.0)
        rospy.loginfo(
            "Fused high environment listening on %s:%d obs=%d action=%d "
            "low_policy=%s",
            self.host,
            self.port,
            self.environment.OBS_DIM,
            self.environment.ACTION_DIM,
            self.low_policy_type,
        )
        while self.running and not rospy.is_shutdown():
            try:
                connection, _ = self.server_socket.accept()
            except socket.timeout:
                continue
            try:
                self._serve_connection(connection)
            finally:
                connection.close()

    def _serve_connection(self, connection):
        connection.settimeout(1.0)
        receive_buffer = b""
        while self.running and not rospy.is_shutdown():
            try:
                chunk = connection.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                return
            receive_buffer += chunk
            while b"\n" in receive_buffer:
                raw_request, receive_buffer = receive_buffer.split(b"\n", 1)
                if not raw_request.strip():
                    continue
                should_close = False
                try:
                    request = json.loads(raw_request.decode("utf-8"))
                    result, should_close = self._handle_request(request)
                    response = {"ok": True, "result": _json_safe(result)}
                except Exception as error:
                    rospy.logerr(
                        "Fused high request failed: %s\n%s",
                        error,
                        traceback.format_exc(),
                    )
                    response = {"ok": False, "error": str(error)}
                connection.sendall((
                    json.dumps(response, separators=(",", ":")) + "\n"
                ).encode("utf-8"))
                if should_close:
                    return

    def _handle_request(self, request):
        command = str(request.get("command", ""))
        if command == "ping":
            metadata = {
                "observation_dim": self.environment.OBS_DIM,
                "action_dim": self.environment.ACTION_DIM,
                "subgoal_type_count": SubgoalType.COUNT,
                "subgoal_types": ["DIRECT", "DETOUR", "TERMINAL"],
                "action_semantics": (
                    "base_dx,base_dy,base_dyaw,ee_dx,ee_dy,ee_dz"
                ),
                "action_limits": self.environment.action_limits,
                "policy_type": (
                    "fused_basic_two_layer_ppo_v2"
                    if self.basic_hierarchy else
                    "fused_joint_subgoal_unified_option_ppo_v2"
                ),
                "low_policy_contract": self._low_policy_contract(),
                "low_policy_metadata": getattr(
                    self.low_policy, "metadata", {}
                ),
                "high_level_interval": self.environment.high_level_interval,
                "option_execution_contract": (
                    "event_driven_until_done_stall_risk_or_horizon_v1"
                ),
                "reward_contract": (
                    "stage_conditioned_terminal_potential_v2"
                    if self.environment.terminal_option_contract else
                    "path_and_ee_potential_v1"
                ),
                "teacher_type": "safe_waypoint",
            }
            if self.basic_hierarchy:
                metadata.update({
                    "state_contract": (
                        "physical_state_plus_active_path_context_v2"
                    ),
                    "basic_path_context": [
                        "waypoint_dx_scaled",
                        "waypoint_dy_scaled",
                        "waypoint_bearing_scaled",
                        "path_index_fraction",
                        "path_remaining_scaled",
                        "final_waypoint_active",
                        "path_complete",
                        "direct_path",
                    ],
                    "terminal_base_hold": True,
                    "terminal_option_contract": (
                        "eligible_latched_persistent_progress_cone_v1"
                    ),
                    "terminal_ee_step": float(
                        self.environment.terminal_ee_step
                    ),
                    "terminal_student_blend": float(
                        self.environment.terminal_student_blend
                    ),
                    "diagnostics_contract": (
                        "stage_option_action_projection_safety_"
                        "terminal_alignment_v3"
                    ),
                })
            if not self.basic_hierarchy:
                metadata.update({
                    "route_side_count": RouteSide.COUNT,
                    "route_sides": ["NONE", "UPPER", "LOWER"],
                    "option_count": HighLevelOption.COUNT,
                    "options": [
                        "DIRECT",
                        "DETOUR_UPPER",
                        "DETOUR_LOWER",
                        "TERMINAL",
                    ],
                })
            return metadata, False
        if command == "reset":
            if "terminal_student_blend" in request:
                self.environment.set_terminal_student_blend(
                    request["terminal_student_blend"]
                )
            observation = self.environment.reset(
                scenario=request.get("scenario"),
                max_steps=request.get("max_steps"),
            )
            self.teacher.reset(preferred_turn_sign=float(
                1.0 if self.basic_hierarchy else
                self.low_environment.last_reset_info.get("detour_side", 0.0)
            ))
            self._update_teacher(observation)
            reset_info = dict(self.low_environment.last_reset_info)
            reset_info["terminal_student_blend"] = float(
                self.environment.terminal_student_blend
            )
            return {
                "observation": observation["vector"],
                "teacher_action": self.last_teacher_action,
                "teacher_info": self.last_teacher_info,
                "reset_info": reset_info,
            }, False
        if command == "step":
            high_command = JointHighLevelCommand.from_action(
                request["action"],
                limits=self.environment.action_limits,
                subgoal_type=request.get(
                    "subgoal_type", SubgoalType.DIRECT
                ),
                route_side=(
                    RouteSide.NONE
                    if self.basic_hierarchy else request.get(
                        "route_side", RouteSide.NONE
                    )
                ),
            )
            observation, reward, done, info = self.environment.step(
                high_command
            )
            if done:
                self.last_teacher_action.fill(0.0)
                self.last_teacher_info = {"terminal": True}
            else:
                self._update_teacher(observation)
            return {
                "observation": observation["vector"],
                "reward": reward,
                "done": done,
                "info": info,
                "teacher_action": self.last_teacher_action,
                "teacher_info": self.last_teacher_info,
            }, False
        if command == "close":
            self._publish_zero_commands()
            return {"closed": True}, True
        if command == "shutdown":
            self._publish_zero_commands()
            self.running = False
            return {"shutdown": True}, True
        raise ValueError("unknown high environment command: {}".format(command))

    def _update_teacher(self, observation):
        teacher_command = self.teacher.predict(observation)
        self.environment.guard_high_command(
            teacher_command, update_latch=False
        )
        self.last_teacher_action = teacher_command.to_action(
            limits=self.environment.action_limits
        ).astype(np.float32)
        self.last_teacher_info = {
            "reason": str(teacher_command.reason),
            "option": int(teacher_command.option),
            "option_name": teacher_command.option_name,
            "subgoal_type": int(teacher_command.subgoal_type),
            "subgoal_type_name": teacher_command.subgoal_type_name,
            "route_side": int(teacher_command.route_side),
            "route_side_name": teacher_command.route_side_name,
        }

    def _publish_zero_commands(self):
        reach_env = getattr(self.low_environment, "reach_env", None)
        if reach_env is not None:
            reach_env._publish_zero_velocity(repeat=3)

    def _low_policy_contract(self):
        if self.low_policy_type in ("residual_ppo", "ppo", "residual"):
            return "coordinated_rule_plus_frozen_residual_ppo"
        return "coordinated_rule_zero_residual"

    def stop(self):
        self.running = False
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except socket.error:
                pass
            self.server_socket = None
        self.environment.close()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return dict((str(key), _json_safe(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if not math.isnan(value) and not math.isinf(value) else None
    return value


def main():
    rospy.init_node("fused_high_environment_server")
    server = FusedHighEnvironmentServer()
    rospy.on_shutdown(server.stop)
    try:
        server.serve_forever()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
