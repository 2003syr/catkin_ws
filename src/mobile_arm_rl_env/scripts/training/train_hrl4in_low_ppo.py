#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train the HRL4IN low-level actor with PPO through the ROS TCP server."""

import argparse
import os
import random
import sys
import time

import numpy as np

try:
    import torch
except ImportError as error:
    raise RuntimeError(
        "HRL4IN low-level PPO requires Python 3 with PyTorch"
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
from training.ppo_rollout import PPORolloutBuffer


def main():
    args = _parse_arguments()
    _set_random_seed(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    model = HRL4INLowActorCritic(
        hidden_sizes=args.hidden_sizes,
        initial_log_std=args.initial_log_std,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )
    normalizer = RunningObservationNormalizer()
    total_steps = 0
    update_index = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        normalizer.load_state_dict(checkpoint["normalizer"])
        total_steps = int(checkpoint.get("total_steps", 0))
        update_index = int(checkpoint.get("update_index", 0))
        print(
            "resumed checkpoint={} total_steps={} update={}".format(
                args.resume,
                total_steps,
                update_index,
            )
        )

    buffer = PPORolloutBuffer(
        capacity=args.rollout_steps,
        observation_dim=model.OBS_DIM,
        action_dim=model.ACTION_DIM,
    )
    client = HRL4INLowEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )
    raw_observation = None
    last_done = True
    subgoal_count = 0
    achieved_count = 0
    timeout_count = 0
    reward_window = []
    start_time = time.time()

    try:
        raw_observation = client.reset(force_episode=True)
        normalizer.update(raw_observation)

        while total_steps < args.total_steps:
            normalized_observation = normalizer.normalize(
                raw_observation
            )
            observation_tensor = torch.from_numpy(
                normalized_observation
            ).to(device).unsqueeze(0)

            with torch.no_grad():
                (
                    action_tensor,
                    masked_action_tensor,
                    log_probability_tensor,
                    value_tensor,
                ) = model.act(observation_tensor)

            action = action_tensor.squeeze(0).cpu().numpy()
            masked_action = (
                masked_action_tensor.squeeze(0).cpu().numpy()
            )
            log_probability = float(
                log_probability_tensor.item()
            )
            value = float(value_tensor.item())

            (
                next_raw_observation,
                intrinsic_reward,
                subgoal_done,
                info,
            ) = client.step(masked_action)
            teacher_action = np.asarray(
                info.get(
                    "teacher_action",
                    np.zeros(model.ACTION_DIM),
                ),
                dtype=np.float32,
            )
            teacher_valid = bool(
                info.get("teacher_available", False)
            )

            buffer.add(
                observation=normalized_observation,
                action=action,
                teacher_action=teacher_action,
                teacher_valid=teacher_valid,
                log_probability=log_probability,
                value=value,
                reward=intrinsic_reward,
                done=subgoal_done,
            )
            reward_window.append(float(intrinsic_reward))
            total_steps += 1
            last_done = bool(subgoal_done)

            if subgoal_done:
                subgoal_count += 1
                achieved_count += int(
                    bool(info.get("subgoal_achieved", False))
                )
                timeout_count += int(
                    bool(info.get("subgoal_timed_out", False))
                )
                raw_observation = client.reset()
            else:
                raw_observation = next_raw_observation
            normalizer.update(raw_observation)

            if buffer.full:
                last_value = _bootstrap_value(
                    model,
                    normalizer,
                    raw_observation,
                    last_done,
                    device,
                )
                buffer.compute_returns_and_advantages(
                    last_value=last_value,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                )
                teacher_coefficient = _teacher_coefficient(
                    args.teacher_coefficient,
                    args.teacher_decay_steps,
                    total_steps,
                )
                metrics = _ppo_update(
                    model=model,
                    optimizer=optimizer,
                    buffer=buffer,
                    device=device,
                    clip_ratio=args.clip_ratio,
                    value_coefficient=args.value_coefficient,
                    entropy_coefficient=args.entropy_coefficient,
                    teacher_coefficient=teacher_coefficient,
                    maximum_gradient_norm=args.maximum_gradient_norm,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=args.batch_size,
                )
                buffer.clear()
                update_index += 1

                elapsed = max(time.time() - start_time, 1e-6)
                mean_reward = (
                    float(np.mean(reward_window[-args.reward_window:]))
                    if reward_window else 0.0
                )
                success_rate = (
                    float(achieved_count) / float(subgoal_count)
                    if subgoal_count else 0.0
                )
                print(
                    "update={} steps={} steps_per_second={:.2f} "
                    "reward={:.4f} success_rate={:.3f} timeouts={} "
                    "policy_loss={:.5f} value_loss={:.5f} "
                    "entropy={:.5f} teacher_loss={:.5f} "
                    "teacher_coef={:.5f}".format(
                        update_index,
                        total_steps,
                        total_steps / elapsed,
                        mean_reward,
                        success_rate,
                        timeout_count,
                        metrics["policy_loss"],
                        metrics["value_loss"],
                        metrics["entropy"],
                        metrics["teacher_loss"],
                        teacher_coefficient,
                    )
                )

                if (
                        args.checkpoint_interval > 0
                        and update_index % args.checkpoint_interval == 0):
                    _save_checkpoint(
                        args.output,
                        model,
                        optimizer,
                        normalizer,
                        total_steps,
                        update_index,
                        args,
                    )

        if buffer.size > 0:
            last_value = _bootstrap_value(
                model,
                normalizer,
                raw_observation,
                last_done,
                device,
            )
            buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            _ppo_update(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                device=device,
                clip_ratio=args.clip_ratio,
                value_coefficient=args.value_coefficient,
                entropy_coefficient=args.entropy_coefficient,
                teacher_coefficient=_teacher_coefficient(
                    args.teacher_coefficient,
                    args.teacher_decay_steps,
                    total_steps,
                ),
                maximum_gradient_norm=args.maximum_gradient_norm,
                ppo_epochs=args.ppo_epochs,
                batch_size=args.batch_size,
            )
            update_index += 1

        _save_checkpoint(
            args.output,
            model,
            optimizer,
            normalizer,
            total_steps,
            update_index,
            args,
        )
        print("model saved: {}".format(args.output))
    finally:
        client.close()


def _ppo_update(
        model,
        optimizer,
        buffer,
        device,
        clip_ratio,
        value_coefficient,
        entropy_coefficient,
        teacher_coefficient,
        maximum_gradient_norm,
        ppo_epochs,
        batch_size):
    data = buffer.tensors(device)
    advantages = data["advantages"]
    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)
    data["advantages"] = advantages

    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "teacher_loss": 0.0,
        "updates": 0,
    }
    for _ in range(int(ppo_epochs)):
        for indices_numpy in buffer.mini_batch_indices(batch_size):
            indices = torch.from_numpy(
                indices_numpy.astype(np.int64)
            ).to(device)
            observations = data["observations"].index_select(0, indices)
            actions = data["actions"].index_select(0, indices)
            old_log_probabilities = data[
                "old_log_probabilities"
            ].index_select(0, indices)
            old_values = data["old_values"].index_select(0, indices)
            returns = data["returns"].index_select(0, indices)
            batch_advantages = data["advantages"].index_select(
                0,
                indices,
            )
            teacher_actions = data["teacher_actions"].index_select(
                0,
                indices,
            )
            teacher_valid = data["teacher_valid"].index_select(
                0,
                indices,
            )

            (
                log_probabilities,
                entropy,
                values,
                mean_actions,
            ) = model.evaluate_actions(observations, actions)
            probability_ratio = torch.exp(
                log_probabilities - old_log_probabilities
            )
            unclipped_objective = (
                probability_ratio * batch_advantages
            )
            clipped_objective = (
                torch.clamp(
                    probability_ratio,
                    1.0 - clip_ratio,
                    1.0 + clip_ratio,
                )
                * batch_advantages
            )
            policy_loss = -torch.min(
                unclipped_objective,
                clipped_objective,
            ).mean()

            clipped_values = old_values + torch.clamp(
                values - old_values,
                -clip_ratio,
                clip_ratio,
            )
            value_loss = 0.5 * torch.max(
                (values - returns).pow(2),
                (clipped_values - returns).pow(2),
            ).mean()
            entropy_mean = entropy.mean()

            action_mask = observations[:, 58:68]
            active_dimensions = torch.clamp(
                action_mask.sum(dim=1),
                min=1.0,
            )
            per_sample_teacher_loss = (
                (mean_actions - teacher_actions).pow(2)
                * action_mask
            ).sum(dim=1) / active_dimensions
            valid_teacher_count = torch.clamp(
                teacher_valid.sum(),
                min=1.0,
            )
            teacher_loss = (
                per_sample_teacher_loss * teacher_valid
            ).sum() / valid_teacher_count
            teacher_loss = teacher_loss * (
                teacher_valid.sum() > 0
            ).float()

            total_loss = (
                policy_loss
                + value_coefficient * value_loss
                - entropy_coefficient * entropy_mean
                + teacher_coefficient * teacher_loss
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                maximum_gradient_norm,
            )
            optimizer.step()

            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["entropy"] += float(entropy_mean.item())
            totals["teacher_loss"] += float(teacher_loss.item())
            totals["updates"] += 1

    divisor = float(max(totals["updates"], 1))
    return {
        "policy_loss": totals["policy_loss"] / divisor,
        "value_loss": totals["value_loss"] / divisor,
        "entropy": totals["entropy"] / divisor,
        "teacher_loss": totals["teacher_loss"] / divisor,
    }


def _bootstrap_value(
        model,
        normalizer,
        raw_observation,
        done,
        device):
    if done:
        return 0.0
    normalized = normalizer.normalize(raw_observation)
    observation_tensor = torch.from_numpy(
        normalized
    ).to(device).unsqueeze(0)
    with torch.no_grad():
        return float(model.get_value(observation_tensor).item())


def _teacher_coefficient(initial, decay_steps, total_steps):
    initial = float(initial)
    decay_steps = int(decay_steps)
    if initial <= 0.0:
        return 0.0
    if decay_steps <= 0:
        return initial
    fraction = min(float(total_steps) / float(decay_steps), 1.0)
    return initial * (1.0 - fraction)


def _save_checkpoint(
        output_path,
        model,
        optimizer,
        normalizer,
        total_steps,
        update_index,
        args):
    output_path = os.path.abspath(os.path.expanduser(output_path))
    output_directory = os.path.dirname(output_path)
    if output_directory and not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    torch.save({
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "total_steps": int(total_steps),
        "update_index": int(update_index),
        "arguments": vars(args),
        "observation_dim": model.OBS_DIM,
        "action_dim": model.ACTION_DIM,
    }, output_path)


def _set_random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument(
        "--output",
        default="/tmp/mobile_arm_rl_training/hrl4in_low_ppo.pt",
    )
    parser.add_argument("--resume")
    parser.add_argument("--total-steps", type=int, default=500000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--teacher-coefficient", type=float, default=0.1)
    parser.add_argument("--teacher-decay-steps", type=int, default=200000)
    parser.add_argument(
        "--maximum-gradient-norm",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 256],
    )
    parser.add_argument("--initial-log-std", type=float, default=-0.5)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--reward-window", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
