#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run a trained low-level PPO actor without using Jacobian actions."""

import argparse
import os
import sys

import numpy as np

try:
    import torch
except ImportError as error:
    raise RuntimeError(
        "HRL4IN low-level evaluation requires Python 3 with PyTorch"
    ) from error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.hrl4in_low_actor_critic import (
    HRL4INLowActorCritic,
    RunningObservationNormalizer,
)
from training.hrl4in_low_env_client import (
    HRL4INLowEnvironmentClient,
)


def main():
    args = _parse_arguments()
    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_arguments = checkpoint.get("arguments", {})
    hidden_sizes = saved_arguments.get(
        "hidden_sizes",
        [256, 256],
    )
    initial_log_std = saved_arguments.get(
        "initial_log_std",
        -0.5,
    )

    model = HRL4INLowActorCritic(
        hidden_sizes=hidden_sizes,
        initial_log_std=initial_log_std,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalizer = RunningObservationNormalizer()
    normalizer.load_state_dict(checkpoint["normalizer"])

    client = HRL4INLowEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )
    subgoal_successes = 0
    subgoal_timeouts = 0
    completed_task_episodes = 0
    task_successes = 0
    task_step_limits = 0
    total_reward = 0.0
    current_task_reward = 0.0
    subgoal_rewards = []
    completed_task_rewards = []
    subgoal_steps = []
    successful_subgoal_steps = []
    final_subgoal_distances = []
    total_safety_events = 0
    try:
        observation = client.reset(force_episode=True)
        for subgoal_index in range(args.subgoals):
            subgoal_reward = 0.0
            subgoal_safety_events = 0
            while True:
                normalized = normalizer.normalize(observation)
                observation_tensor = torch.from_numpy(
                    normalized
                ).to(device).unsqueeze(0)
                with torch.no_grad():
                    action = model.deterministic_action(
                        observation_tensor
                    ).squeeze(0).cpu().numpy()
                observation, reward, done, info = client.step(action)
                subgoal_reward += float(reward)
                total_reward += float(reward)
                current_task_reward += float(reward)
                step_safety_events = int(
                    info.get("safety_event_count", 0)
                )
                subgoal_safety_events += step_safety_events
                total_safety_events += step_safety_events
                if done:
                    achieved = bool(
                        info.get("subgoal_achieved", False)
                    )
                    timed_out = bool(
                        info.get("subgoal_timed_out", False)
                    )
                    episode_done = bool(
                        info.get("episode_done", False)
                    )
                    task_success = bool(
                        info.get("success", False)
                    )
                    steps = int(info.get("subgoal_steps", 0))
                    remaining_subgoal = np.asarray(
                        info.get(
                            "remaining_subgoal",
                            np.zeros(6, dtype=np.float32),
                        ),
                        dtype=np.float32,
                    )
                    if remaining_subgoal.shape == (6,):
                        final_distance = float(np.linalg.norm(
                            remaining_subgoal[3:6]
                        ))
                    else:
                        final_distance = float("nan")

                    subgoal_successes += int(achieved)
                    subgoal_timeouts += int(timed_out)
                    subgoal_rewards.append(subgoal_reward)
                    subgoal_steps.append(steps)
                    final_subgoal_distances.append(final_distance)
                    if achieved:
                        successful_subgoal_steps.append(steps)

                    termination_reasons = []
                    if achieved:
                        termination_reasons.append("subgoal_achieved")
                    if timed_out:
                        termination_reasons.append("subgoal_timeout")
                    if task_success:
                        termination_reasons.append("task_success")
                    if episode_done and not task_success:
                        termination_reasons.append("episode_step_limit")
                    if not termination_reasons:
                        termination_reasons.append("other")

                    if episode_done:
                        completed_task_episodes += 1
                        task_successes += int(task_success)
                        task_step_limits += int(not task_success)
                        completed_task_rewards.append(
                            current_task_reward
                        )
                        current_task_reward = 0.0

                    print(
                        "subgoal={} reward={:.4f} achieved={} "
                        "timed_out={} task_success={} episode_done={} "
                        "reason={} steps={} final_distance={:.4f} "
                        "safety_events={}".format(
                            subgoal_index + 1,
                            subgoal_reward,
                            achieved,
                            timed_out,
                            task_success,
                            episode_done,
                            "+".join(termination_reasons),
                            steps,
                            final_distance,
                            subgoal_safety_events,
                        )
                    )
                    observation = client.reset()
                    break
    finally:
        client.close()

    print(
        "subgoal_success_rate={}/{} ({:.3f}) "
        "subgoal_timeouts={}/{} ({:.3f})".format(
            subgoal_successes,
            args.subgoals,
            _rate(subgoal_successes, args.subgoals),
            subgoal_timeouts,
            args.subgoals,
            _rate(subgoal_timeouts, args.subgoals),
        )
    )
    print(
        "task_success_rate={}/{} ({:.3f}) "
        "episode_step_limits={}".format(
            task_successes,
            completed_task_episodes,
            _rate(task_successes, completed_task_episodes),
            task_step_limits,
        )
    )
    print(
        "mean_subgoal_reward={:.4f} mean_task_reward={:.4f} "
        "mean_subgoal_steps={:.2f} "
        "mean_success_steps={:.2f}".format(
            _mean(subgoal_rewards),
            _mean(completed_task_rewards),
            _mean(subgoal_steps),
            _mean(successful_subgoal_steps),
        )
    )
    print(
        "mean_final_subgoal_distance={:.4f} "
        "total_safety_events={} "
        "incomplete_task_reward={:.4f}".format(
            _mean(final_subgoal_distances),
            total_safety_events,
            current_task_reward,
        )
    )


def _rate(numerator, denominator):
    return float(numerator) / float(max(int(denominator), 1))


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--subgoals", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    main()
