#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect successful observable-only planar pose demonstrations."""

import argparse
import os
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.planar_direct_env_client import (
    PlanarDirectEnvironmentClient,
)
from training.rule_based_planar_tracker import (
    RuleBasedPlanarTracker,
)


def main():
    args = _parse_arguments()
    teacher = RuleBasedPlanarTracker(
        heading_gain=args.heading_gain,
        slow_distance=args.slow_distance,
        turn_in_place_angle=args.turn_in_place_angle,
        success_distance=args.success_distance,
        success_yaw_error=args.success_yaw_error,
        linear_slew_limit=args.linear_slew_limit,
        angular_slew_limit=args.angular_slew_limit,
    )
    client = PlanarDirectEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )
    accepted_observations = []
    accepted_actions = []
    accepted_episode_ids = []
    successful_episodes = 0
    attempted_episodes = 0
    collisions = 0
    timeouts = 0
    attempted_steps = 0
    start_time = time.time()
    try:
        while (
                successful_episodes < args.episodes
                and attempted_episodes < args.max_attempts):
            attempted_episodes += 1
            observation = client.reset()
            episode_observations = []
            episode_actions = []
            final_info = {}
            done = False
            for _ in range(args.max_steps):
                action = teacher.predict(observation)
                episode_observations.append(observation.copy())
                episode_actions.append(action.copy())
                observation, unused_reward, done, info = client.step(
                    action
                )
                attempted_steps += 1
                final_info = info
                if done:
                    break

            success = bool(final_info.get("success", False))
            collisions += int(bool(final_info.get("collision", False)))
            timeouts += int(bool(final_info.get("timeout", False)))
            if success:
                successful_episodes += 1
                accepted_observations.extend(episode_observations)
                accepted_actions.extend(episode_actions)
                accepted_episode_ids.extend(
                    [successful_episodes] * len(episode_observations)
                )
            elapsed = max(time.time() - start_time, 1.0e-6)
            print(
                "collection attempted={} accepted={}/{} success={} "
                "episode_steps={} samples={} collisions={} timeouts={} "
                "steps_per_second={:.2f}".format(
                    attempted_episodes,
                    successful_episodes,
                    args.episodes,
                    success,
                    len(episode_observations),
                    len(accepted_observations),
                    collisions,
                    timeouts,
                    attempted_steps / elapsed,
                )
            )
    finally:
        client.close()

    if successful_episodes < args.episodes:
        raise RuntimeError(
            "only collected {}/{} successful teacher episodes".format(
                successful_episodes,
                args.episodes,
            )
        )
    observations = np.asarray(
        accepted_observations,
        dtype=np.float32,
    )
    actions = np.asarray(accepted_actions, dtype=np.float32)
    episode_ids = np.asarray(accepted_episode_ids, dtype=np.int32)
    output = os.path.abspath(os.path.expanduser(args.output))
    directory = os.path.dirname(output)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    np.savez_compressed(
        output,
        observations=observations,
        teacher_actions=actions,
        episode_ids=episode_ids,
        observation_dim=np.asarray([62], dtype=np.int32),
        action_dim=np.asarray([2], dtype=np.int32),
        successful_episodes=np.asarray(
            [successful_episodes], dtype=np.int32
        ),
        attempted_episodes=np.asarray(
            [attempted_episodes], dtype=np.int32
        ),
        teacher_type=np.asarray(
            ["observable_rule_based_planar_tracker"]
        ),
        observation_semantics=np.asarray([
            "local_subgoal_xy2,subgoal_yaw_sin_cos2,body_vw2,"
            "scan36,path_preview_xy10,path_metrics3,"
            "previous_action2,path_mask5"
        ]),
        action_semantics=np.asarray([
            "normalized_linear_velocity_yaw_rate"
        ]),
    )
    print(
        "tracking dataset saved={} samples={} episodes={} "
        "attempts={}".format(
            output,
            observations.shape[0],
            successful_episodes,
            attempted_episodes,
        )
    )


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="/tmp/mobile_arm_rl_training/planar_tracking_teacher.npz",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=75)
    parser.add_argument("--max-steps", type=int, default=650)
    parser.add_argument("--heading-gain", type=float, default=1.6)
    parser.add_argument("--slow-distance", type=float, default=0.35)
    parser.add_argument(
        "--turn-in-place-angle",
        type=float,
        default=0.35,
    )
    parser.add_argument("--success-distance", type=float, default=0.08)
    parser.add_argument(
        "--success-yaw-error",
        type=float,
        default=0.12,
    )
    parser.add_argument("--linear-slew-limit", type=float, default=0.12)
    parser.add_argument("--angular-slew-limit", type=float, default=0.20)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_attempts < args.episodes:
        raise ValueError("invalid episode/attempt counts")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    return args


if __name__ == "__main__":
    main()
