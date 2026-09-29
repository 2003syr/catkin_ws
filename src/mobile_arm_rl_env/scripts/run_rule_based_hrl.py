#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

from hrl.hierarchical_env import HierarchicalEnv
from hrl.hrl4in_ppo_low_policy import HRL4INPPOLowPolicy
from hrl.kdl_jacobian import KDLJacobianProvider
from hrl.rule_based_high_policy import RuleBasedHighPolicy
from hrl.rule_based_low_policy import RuleBasedLowPolicy
from mobile_arm_env_check import MobileArmReachEnv


def build_low_policy(base_env):
    policy_type = str(
        rospy.get_param("~low_policy_type", "rule_based")
    ).strip().lower()
    subgoal_tolerance = rospy.get_param(
        "~subgoal_tolerance",
        [0.02, 0.02, 0.05, 0.01, 0.01, 0.01],
    )
    reward_options = {
        "subgoal_tolerance": subgoal_tolerance,
        "intrinsic_reward_scale": rospy.get_param(
            "~intrinsic_reward_scale", 30.0
        ),
        "subgoal_achieved_reward": rospy.get_param(
            "~subgoal_achieved_reward", 1.0
        ),
        "collision_reward_weight": rospy.get_param(
            "~collision_reward_weight", 0.0
        ),
        "extrinsic_reward_weight": rospy.get_param(
            "~extrinsic_reward_weight", 0.0
        ),
    }

    if policy_type == "ppo":
        policy = HRL4INPPOLowPolicy(
            host=rospy.get_param("~ppo_host", "127.0.0.1"),
            port=rospy.get_param("~ppo_port", 5556),
            timeout=rospy.get_param("~ppo_timeout", 2.0),
            base_gain=rospy.get_param("~base_gain", 5.0),
            enable_base_motion=rospy.get_param(
                "~enable_base_motion", False
            ),
            recovery_gain=rospy.get_param("~recovery_gain", 0.4),
            **reward_options
        )
        rospy.loginfo(
            "Loaded HRL4IN PPO low policy through %s:%d "
            "checkpoint=%s total_steps=%d",
            rospy.get_param("~ppo_host", "127.0.0.1"),
            int(rospy.get_param("~ppo_port", 5556)),
            policy.checkpoint,
            policy.model_total_steps,
        )
        return policy_type, policy

    if policy_type != "rule_based":
        raise ValueError(
            "low_policy_type must be 'rule_based' or 'ppo', got '{}'".format(
                policy_type
            )
        )

    jacobian_provider = KDLJacobianProvider(
        base_frame=base_env.base_frame,
        ee_frame=base_env.ee_frame,
        state_joint_names=base_env.expected_joints,
        controlled_joint_names=base_env.arm_joints,
    )
    rospy.loginfo(
        "KDL chain joints=%s controlled_arm_joints=%s",
        jacobian_provider.chain_joint_names,
        jacobian_provider.controlled_joint_names,
    )
    policy = RuleBasedLowPolicy(
        enable_base_motion=rospy.get_param("~enable_base_motion", False),
        jacobian_provider=jacobian_provider,
        arm_joint_max_velocity=[
            base_env.action_max_vel[name] for name in base_env.arm_joints
        ],
        arm_gain=rospy.get_param("~arm_cartesian_gain", 1.0),
        max_cartesian_speed=rospy.get_param(
            "~max_cartesian_speed", 0.05
        ),
        dls_damping=rospy.get_param("~dls_damping", 0.03),
        **reward_options
    )
    return policy_type, policy


def main():
    rospy.init_node("rule_based_mobile_arm_hrl")

    high_interval = rospy.get_param("~high_interval", 10)
    base_env = MobileArmReachEnv(init_ros_node=False)
    low_policy_type, low_policy = build_low_policy(base_env)
    env = HierarchicalEnv(
        base_env=base_env,
        high_policy=RuleBasedHighPolicy(
            arm_enter_distance=rospy.get_param("~arm_enter_distance", 0.80),
            arm_exit_distance=rospy.get_param("~arm_exit_distance", 1.00),
            min_joint_margin=rospy.get_param("~min_joint_margin", 0.08),
            max_position_subgoal=rospy.get_param(
                "~max_position_subgoal", 0.10
            ),
        ),
        low_policy=low_policy,
        high_interval=high_interval,
    )

    observation = env.reset()
    last_high_distance = None
    holding_target = False
    held_distance = None
    hold_rate = rospy.Rate(10)
    rospy.loginfo(
        "Rule-based high policy HRL started: low_policy=%s "
        "obs_dim=%d action_dim=%d high_interval=%d",
        low_policy_type,
        len(observation["obs_vec"]),
        base_env.action_dim,
        high_interval,
    )
    if not env.low_policy.enable_base_motion:
        rospy.logwarn(
            "Base motion is disabled: URDF x/y are internal joints, not a "
            "mobile-base interface. Enable it only after validating the "
            "planar-chain directions and dynamics."
        )

    # Register before entering the control loop. rospy invokes shutdown hooks
    # before publishers are unregistered, so Ctrl+C can still deliver the
    # final zero command to Gazebo. The finally block remains as the normal
    # success/exception path and stop() is idempotent.
    rospy.on_shutdown(env.stop)

    try:
        while not rospy.is_shutdown():
            if holding_target:
                # A successful online episode is latched.  Keep publishing
                # zero velocity, but never call env.step() again: doing so
                # would create new subgoals and award the success bonus on
                # every control cycle.
                env.hold_zero()
                rospy.loginfo_throttle(
                    5.0,
                    "HRL SUCCESS_HOLD dist=%.4f: zero velocity active; "
                    "policy and reward updates paused",
                    held_distance,
                )
                hold_rate.sleep()
                continue

            observation, reward, done, info = env.step()
            if info["high_updated"]:
                current_distance = float(info["dist"])
                if last_high_distance is None:
                    distance_delta = 0.0
                    trend = "INITIAL"
                else:
                    # Positive delta means the end effector moved closer.
                    distance_delta = last_high_distance - current_distance
                    if distance_delta > 0.005:
                        trend = "APPROACHING"
                    elif distance_delta < -0.005:
                        trend = "MOVING_AWAY"
                    else:
                        trend = "STALLED"
                last_high_distance = current_distance

                joint_order = base_env.expected_joints
                raw_cmd = [
                    info["raw_cmd_dict"][name] for name in joint_order
                ]
                safe_cmd = [info["cmd_dict"][name] for name in joint_order]
                joint_position = observation["joint_pos"]
                safety_events = [
                    "{}:{}".format(name, info["safety_info"][name]["reason"])
                    for name in joint_order
                    if info["safety_info"][name]["reason"] != "safe"
                ]

                rospy.loginfo(
                    "HRL decision high_step=%d mode=%s reason=%s "
                    "dist=%.4f delta=%.4f trend=%s subgoal=%s",
                    info["high_step"],
                    info["task_mode"],
                    info["task_reason"],
                    current_distance,
                    distance_delta,
                    trend,
                    info["subgoal"],
                )
                rospy.loginfo(
                    "HRL execution order=%s raw_cmd=%s safe_cmd=%s q=%s",
                    joint_order,
                    [round(value, 4) for value in raw_cmd],
                    [round(value, 4) for value in safe_cmd],
                    [round(float(value), 4) for value in joint_position],
                )
                if safety_events:
                    rospy.logwarn("HRL safety_filter=%s", safety_events)
                else:
                    rospy.loginfo("HRL safety_filter=NONE")

                low_diagnostics = info.get("low_level_diagnostics", {})
                if "error_base" in low_diagnostics:
                    if info["task_mode"] == "BASE_APPROACH":
                        rospy.loginfo(
                            "HRL planar_base_control "
                            "remaining=%s normalized_action=%s",
                            [
                                round(float(value), 4)
                                for value in low_diagnostics[
                                    "base_subgoal_error"
                                ]
                            ],
                            [
                                round(float(value), 4)
                                for value in low_diagnostics[
                                    "normalized_base_action"
                                ]
                            ],
                        )
                    elif (
                            low_diagnostics.get("policy_type")
                            == "hrl4in_ppo"):
                        rospy.loginfo(
                            "HRL ppo_control subgoal_error=%s "
                            "subgoal_dist=%.4f->%.4f target_dist=%.4f "
                            "intrinsic=%.4f progress=%.4f "
                            "achieved_bonus=%.4f arm_action=%s "
                            "inference_ms=%.3f",
                            [
                                round(float(value), 4)
                                for value in low_diagnostics["error_base"]
                            ],
                            low_diagnostics["error_norm"],
                            low_diagnostics["post_error_norm"],
                            low_diagnostics["target_error_norm"],
                            low_diagnostics["intrinsic_reward"],
                            low_diagnostics[
                                "intrinsic_progress_reward"
                            ],
                            low_diagnostics[
                                "intrinsic_achievement_reward"
                            ],
                            [
                                round(float(value), 4)
                                for value in low_diagnostics[
                                    "normalized_arm_action"
                                ]
                            ],
                            low_diagnostics["inference_ms"],
                        )
                    else:
                        rospy.loginfo(
                            "HRL arm_control subgoal_error=%s "
                            "subgoal_dist=%.4f->%.4f target_dist=%.4f "
                            "intrinsic=%.4f progress=%.4f "
                            "achieved_bonus=%.4f "
                            "cartesian_velocity=%s "
                            "jacobian_condition=%.2f",
                            [
                                round(float(value), 4)
                                for value in low_diagnostics["error_base"]
                            ],
                            low_diagnostics["error_norm"],
                            low_diagnostics["post_error_norm"],
                            low_diagnostics["target_error_norm"],
                            low_diagnostics["intrinsic_reward"],
                            low_diagnostics[
                                "intrinsic_progress_reward"
                            ],
                            low_diagnostics[
                                "intrinsic_achievement_reward"
                            ],
                            [
                                round(float(value), 4)
                                for value in low_diagnostics[
                                    "cartesian_velocity"
                                ]
                            ],
                            low_diagnostics["jacobian_condition"],
                        )

                tracking = info["velocity_tracking"]
                desired_velocity = [
                    tracking[name]["desired"] for name in joint_order
                ]
                measured_velocity = [
                    tracking[name]["measured"] for name in joint_order
                ]
                tracking_error = [
                    tracking[name]["error"] for name in joint_order
                ]
                tracking_message = (
                    "HRL velocity_tracking desired=%s measured=%s error=%s "
                    "max_arm_normalized_error=%.3f"
                )
                tracking_args = (
                    [round(value, 4) for value in desired_velocity],
                    [round(value, 4) for value in measured_velocity],
                    [round(value, 4) for value in tracking_error],
                    info["max_arm_velocity_tracking_error"],
                )
                if info["max_arm_velocity_tracking_error"] > 0.35:
                    rospy.logwarn(tracking_message, *tracking_args)
                else:
                    rospy.loginfo(tracking_message, *tracking_args)
            rospy.loginfo_throttle(
                1.0,
                "HRL low_step=%d mode=%s target_dist=%.4f "
                "subgoal_dist=%.4f intrinsic=%.4f "
                "achieved=%s timed_out=%s subgoal_done=%s reward=%.4f",
                info["low_step"],
                info["task_mode"],
                info["dist"],
                info["subgoal_distance"],
                info["intrinsic_reward"],
                str(info["subgoal_achieved"]),
                str(info["subgoal_timed_out"]),
                str(info["subgoal_done"]),
                reward,
            )
            if done:
                if info["success"]:
                    held_distance = float(info["dist"])
                    holding_target = True
                    # Cancel the last non-zero command immediately.  The
                    # loop-top hold path republishes zero at 10 Hz afterward.
                    env.hold_zero()
                    rospy.loginfo(
                        "HRL target reached: dist=%.4f; entering "
                        "SUCCESS_HOLD with zero velocity. Policy, subgoal "
                        "and reward updates are now paused.",
                        held_distance,
                    )
                    continue
                rospy.loginfo(
                    "HRL time limit reached; resetting counters only"
                )
                observation = env.reset()
                last_high_distance = None
                holding_target = False
                held_distance = None
    except rospy.ROSInterruptException:
        # Ctrl+C and roslaunch shutdown can interrupt rospy.sleep() inside
        # env.step(). This is a normal stop path, not an execution failure.
        pass
    finally:
        rospy.loginfo("HRL node exiting; zero-velocity shutdown sequence begins")
        env.stop()


if __name__ == "__main__":
    main()
