#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministically evaluate a post-DAgger tracked-base checkpoint."""

import argparse
import os
import sys

import numpy as np

try:
    import torch
except ImportError as error:
    raise RuntimeError(
        "Planar base evaluation requires Python 3 with PyTorch"
    ) from error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.hrl4in_low_actor_critic import (
    RunningObservationNormalizer,
)
from training.planar_base_actor_critic import PlanarBaseActorCritic
from training.planar_dwa_env_client import PlanarDWAEnvironmentClient


def main():
    args = _parse_arguments()
    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if not bool(checkpoint.get("dagger_completed", False)):
        raise RuntimeError(
            "checkpoint has not completed DAgger; refusing PPO gate "
            "evaluation"
        )
    checkpoint_arguments = checkpoint.get("arguments", {})
    hidden_sizes = checkpoint_arguments.get(
        "hidden_sizes",
        [128, 128],
    )
    model = PlanarBaseActorCritic(
        hidden_sizes=hidden_sizes,
        initial_log_std=checkpoint_arguments.get(
            "initial_log_std",
            -1.5,
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalizer = RunningObservationNormalizer(
        observation_dim=model.OBS_DIM,
        normalized_dim=4,
    )
    normalizer.load_state_dict(checkpoint["normalizer"])

    client = PlanarDWAEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )
    successes = 0
    collisions = 0
    timeouts = 0
    local_step_limits = 0
    episode_rewards = []
    final_distances = []
    episode_steps = []
    teacher_mse_values = []
    action_magnitude_values = []

    try:
        for episode_index in range(args.episodes):
            raw_observation = client.reset()
            start_distance = float(np.linalg.norm(
                raw_observation[0:2]
            ))
            episode_reward = 0.0
            final_info = {}
            done = False
            for step_index in range(args.max_steps):
                normalized_observation = normalizer.normalize(
                    raw_observation
                )
                observation_tensor = torch.from_numpy(
                    normalized_observation
                ).to(device).unsqueeze(0)
                with torch.no_grad():
                    action = model.deterministic_action(
                        observation_tensor
                    ).squeeze(0).cpu().numpy()
                (
                    raw_observation,
                    reward,
                    done,
                    info,
                ) = client.step(action)
                episode_reward += float(reward)
                action_magnitude_values.append(float(
                    np.mean(np.abs(action))
                ))
                if bool(info.get("teacher_available", False)):
                    teacher_action = np.asarray(
                        info["teacher_action"],
                        dtype=np.float32,
                    )
                    teacher_mse_values.append(float(np.mean(
                        np.square(action - teacher_action)
                    )))
                final_info = info
                if done:
                    break

            success = bool(final_info.get("success", False))
            collision = bool(final_info.get("collision", False))
            timeout = bool(final_info.get("timeout", False))
            local_step_limit = bool(not done)
            successes += int(success)
            collisions += int(collision)
            timeouts += int(timeout)
            local_step_limits += int(local_step_limit)
            final_distance = float(final_info.get(
                "distance",
                np.linalg.norm(raw_observation[0:2]),
            ))
            steps = step_index + 1
            episode_rewards.append(episode_reward)
            final_distances.append(final_distance)
            episode_steps.append(steps)
            print(
                "evaluation_episode={:02d} distance={:.4f}->{:.4f} "
                "reward={:.4f} success={} collision={} timeout={} "
                "local_step_limit={} steps={}".format(
                    episode_index + 1,
                    start_distance,
                    final_distance,
                    episode_reward,
                    success,
                    collision,
                    timeout,
                    local_step_limit,
                    steps,
                )
            )
    finally:
        client.close()

    success_rate = float(successes) / float(args.episodes)
    print(
        "deterministic_evaluation episodes={} successes={} "
        "success_rate={:.3f} collisions={} timeouts={} "
        "local_step_limits={} mean_reward={:.4f} "
        "mean_final_distance={:.4f} mean_steps={:.1f} "
        "learner_teacher_mse={:.5f} action_abs={:.4f}".format(
            args.episodes,
            successes,
            success_rate,
            collisions,
            timeouts,
            local_step_limits,
            _mean_or_nan(episode_rewards),
            _mean_or_nan(final_distances),
            _mean_or_nan(episode_steps),
            _mean_or_nan(teacher_mse_values),
            _mean_or_nan(action_magnitude_values),
        )
    )
    passed = bool(
        success_rate >= args.min_success_rate
        and collisions <= args.max_collisions
        and local_step_limits == 0
    )
    print("deterministic_gate_pass={}".format(passed))
    if not passed:
        raise SystemExit(1)


def _mean_or_nan(values):
    if not values:
        return float("nan")
    return float(np.mean(values))


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--min-success-rate", type=float, default=0.60)
    parser.add_argument("--max-collisions", type=int, default=0)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes and max-steps must be positive")
    if not 0.0 <= args.min_success_rate <= 1.0:
        raise ValueError("min-success-rate must be in [0, 1]")
    if args.max_collisions < 0:
        raise ValueError("max-collisions cannot be negative")
    return args


if __name__ == "__main__":
    main()
