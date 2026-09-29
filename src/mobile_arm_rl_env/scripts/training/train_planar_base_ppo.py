#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train the tracked-base policy with PPO plus a decaying DWA teacher loss."""

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
        "Planar base PPO requires Python 3 with PyTorch"
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
from training.ppo_rollout import PPORolloutBuffer


def main():
    args = _parse_arguments()
    _validate_arguments(args)
    _set_random_seed(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    model = PlanarBaseActorCritic(
        hidden_sizes=args.hidden_sizes,
        initial_log_std=args.initial_log_std,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )
    # Target displacement and measured velocity need running statistics.
    # Laser sectors and previous actions are already bounded.
    normalizer = RunningObservationNormalizer(
        observation_dim=model.OBS_DIM,
        normalized_dim=4,
    )
    total_steps = 0
    update_index = 0
    teacher_warmup_completed = bool(
        args.teacher_warmup_steps <= 0
    )
    dagger_completed = bool(args.dagger_steps <= 0)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        normalizer.load_state_dict(checkpoint["normalizer"])
        total_steps = int(checkpoint.get("total_steps", 0))
        update_index = int(checkpoint.get("update_index", 0))
        teacher_warmup_completed = bool(
            checkpoint.get("teacher_warmup_completed", False)
            or args.teacher_warmup_steps <= 0
        )
        dagger_completed = bool(
            checkpoint.get("dagger_completed", False)
            or args.dagger_steps <= 0
        )
        if args.resume_log_std is not None:
            _set_policy_log_std(
                model,
                optimizer,
                args.resume_log_std,
            )
        print(
            "resumed checkpoint={} total_steps={} update={} "
            "teacher_warmup_completed={} dagger_completed={} "
            "exploration_std={:.5f}".format(
                args.resume,
                total_steps,
                update_index,
                teacher_warmup_completed,
                dagger_completed,
                _policy_exploration_std(model),
            )
        )

    buffer = PPORolloutBuffer(
        capacity=args.rollout_steps,
        observation_dim=model.OBS_DIM,
        action_dim=model.ACTION_DIM,
    )
    client = PlanarDWAEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )
    raw_observation = None
    last_done = True
    episode_count = 0
    success_count = 0
    collision_count = 0
    timeout_count = 0
    teacher_valid_steps = 0
    reward_window = []
    distance_window = []
    action_magnitude_window = []
    teacher_mse_window = []
    demonstration_observations = np.empty(
        (0, model.OBS_DIM),
        dtype=np.float32,
    )
    demonstration_actions = np.empty(
        (0, model.ACTION_DIM),
        dtype=np.float32,
    )
    start_time = time.time()

    try:
        raw_observation = client.reset()
        normalizer.update(raw_observation)
        if not teacher_warmup_completed:
            (
                raw_observation,
                demonstration_observations,
                demonstration_actions,
            ) = _teacher_behavior_cloning_warmup(
                client=client,
                model=model,
                optimizer=optimizer,
                normalizer=normalizer,
                raw_observation=raw_observation,
                device=device,
                demonstration_steps=args.teacher_warmup_steps,
                epochs=args.teacher_warmup_epochs,
                batch_size=args.teacher_warmup_batch_size,
                maximum_gradient_norm=args.maximum_gradient_norm,
                log_interval=args.teacher_warmup_log_interval,
                maximum_episodes=args.teacher_warmup_max_episodes,
            )
            teacher_warmup_completed = True
            last_done = True
            _set_policy_log_std(
                model,
                optimizer,
                args.post_warmup_log_std,
            )
            bc_output = (
                args.bc_output
                if args.bc_output
                else _stage_checkpoint_path(args.output, "bc")
            )
            _save_checkpoint(
                bc_output,
                model,
                optimizer,
                normalizer,
                total_steps,
                update_index,
                teacher_warmup_completed,
                dagger_completed,
                args,
            )
            print(
                "pure behavior-cloning checkpoint saved: {}".format(
                    bc_output
                )
            )

        if not dagger_completed:
            (
                raw_observation,
                demonstration_observations,
                demonstration_actions,
            ) = _dagger_training_stage(
                client=client,
                model=model,
                optimizer=optimizer,
                normalizer=normalizer,
                raw_observation=raw_observation,
                device=device,
                initial_observations=demonstration_observations,
                initial_teacher_actions=demonstration_actions,
                total_steps=args.dagger_steps,
                rounds=args.dagger_rounds,
                epochs=args.dagger_epochs,
                batch_size=args.dagger_batch_size,
                beta_start=args.dagger_beta_start,
                beta_end=args.dagger_beta_end,
                recent_fraction=args.dagger_recent_fraction,
                maximum_gradient_norm=args.maximum_gradient_norm,
            )
            dagger_completed = True
            last_done = True
            _set_policy_log_std(
                model,
                optimizer,
                args.post_warmup_log_std,
            )
            warmup_output = (
                args.warmup_output
                if args.warmup_output
                else _warmup_checkpoint_path(args.output)
            )
            _save_checkpoint(
                warmup_output,
                model,
                optimizer,
                normalizer,
                total_steps,
                update_index,
                teacher_warmup_completed,
                dagger_completed,
                args,
            )
            print(
                "DAgger checkpoint saved: {} "
                "exploration_std={:.5f}".format(
                    warmup_output,
                    _policy_exploration_std(model),
                )
            )
        if args.dagger_only:
            print(
                "DAgger-only run complete; PPO was not started. "
                "Evaluate the deterministic checkpoint before PPO."
            )
            return
        run_start_steps = total_steps
        start_time = time.time()

        while total_steps < args.total_steps:
            normalized_observation = normalizer.normalize(
                raw_observation
            )
            observation_tensor = torch.from_numpy(
                normalized_observation
            ).to(device).unsqueeze(0)
            with torch.no_grad():
                action_tensor, log_probability_tensor, value_tensor = (
                    model.act(observation_tensor)
                )
            action = action_tensor.squeeze(0).cpu().numpy()
            (
                next_raw_observation,
                reward,
                done,
                info,
            ) = client.step(action)
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
                log_probability=float(log_probability_tensor.item()),
                value=float(value_tensor.item()),
                reward=float(reward),
                done=done,
            )
            reward_window.append(float(reward))
            distance = float(info.get(
                "distance",
                np.linalg.norm(next_raw_observation[0:2]),
            ))
            if np.isfinite(distance):
                distance_window.append(distance)
            action_magnitude_window.append(float(
                np.mean(np.abs(action))
            ))
            if teacher_valid:
                teacher_mse_window.append(float(np.mean(np.square(
                    action - teacher_action
                ))))
            teacher_valid_steps += int(teacher_valid)
            total_steps += 1
            last_done = bool(done)

            if done:
                episode_count += 1
                success_count += int(bool(info.get("success", False)))
                collision_count += int(bool(info.get("collision", False)))
                timeout_count += int(bool(info.get("timeout", False)))
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
                mean_reward = float(np.mean(
                    reward_window[-args.reward_window:]
                ))
                mean_distance = _window_mean(
                    distance_window,
                    args.reward_window,
                )
                mean_action_magnitude = _window_mean(
                    action_magnitude_window,
                    args.reward_window,
                )
                mean_teacher_mse = _window_mean(
                    teacher_mse_window,
                    args.reward_window,
                )
                success_rate = (
                    float(success_count) / float(episode_count)
                    if episode_count else 0.0
                )
                teacher_coverage = (
                    float(teacher_valid_steps)
                    / float(max(total_steps - run_start_steps, 1))
                )
                print(
                    "update={} steps={} steps_per_second={:.2f} "
                    "reward={:.4f} episodes={} success_rate={:.3f} "
                    "collisions={} timeouts={} teacher_coverage={:.3f} "
                    "distance={:.4f} action_abs={:.4f} "
                    "learner_teacher_mse={:.5f} "
                    "exploration_std={:.5f} "
                    "policy_loss={:.5f} value_loss={:.5f} "
                    "entropy={:.5f} teacher_loss={:.5f} "
                    "teacher_coef={:.5f}".format(
                        update_index,
                        total_steps,
                        (total_steps - run_start_steps) / elapsed,
                        mean_reward,
                        episode_count,
                        success_rate,
                        collision_count,
                        timeout_count,
                        teacher_coverage,
                        mean_distance,
                        mean_action_magnitude,
                        mean_teacher_mse,
                        _policy_exploration_std(model),
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
                        teacher_warmup_completed,
                        dagger_completed,
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
            teacher_warmup_completed,
            dagger_completed,
            args,
        )
        print("model saved: {}".format(args.output))
    finally:
        client.close()


def _teacher_behavior_cloning_warmup(
        client,
        model,
        optimizer,
        normalizer,
        raw_observation,
        device,
        demonstration_steps,
        epochs,
        batch_size,
        maximum_gradient_norm,
        log_interval,
        maximum_episodes):
    """Collect DWA-controlled transitions, then initialize the actor by BC."""
    demonstration_steps = int(demonstration_steps)
    epochs = int(epochs)
    batch_size = int(batch_size)
    log_interval = int(log_interval)
    observations = []
    teacher_actions = []
    episode_observations = []
    episode_teacher_actions = []
    episode_count = 0
    success_count = 0
    collision_count = 0
    timeout_count = 0
    episode_rewards = []
    current_episode_reward = 0.0
    final_distances = []
    collection_start = time.time()
    attempted_steps = 0
    next_log_attempt = (
        log_interval if log_interval > 0 else demonstration_steps
    )

    print(
        "teacher warm-up collecting {} DWA steps before PPO".format(
            demonstration_steps
        )
    )
    while len(observations) < demonstration_steps:
        current_observation = np.asarray(
            raw_observation,
            dtype=np.float32,
        ).copy()
        (
            next_raw_observation,
            reward,
            done,
            info,
        ) = client.teacher_step()
        attempted_steps += 1
        if not bool(info.get("teacher_available", False)):
            raise RuntimeError(
                "DWA teacher did not provide a valid warm-up action"
            )
        teacher_action = np.asarray(
            info["teacher_action"],
            dtype=np.float32,
        )
        if teacher_action.shape != (model.ACTION_DIM,):
            raise RuntimeError(
                "DWA teacher action must have shape ({},)".format(
                    model.ACTION_DIM
                )
            )
        episode_observations.append(current_observation)
        episode_teacher_actions.append(teacher_action.copy())
        current_episode_reward += float(reward)
        normalizer.update(next_raw_observation)
        raw_observation = next_raw_observation

        if done:
            episode_count += 1
            success_count += int(bool(info.get("success", False)))
            collision_count += int(bool(info.get("collision", False)))
            timeout_count += int(bool(info.get("timeout", False)))
            episode_rewards.append(current_episode_reward)
            final_distances.append(float(info.get(
                "distance",
                np.linalg.norm(raw_observation[0:2]),
            )))
            if bool(info.get("success", False)):
                remaining = demonstration_steps - len(observations)
                observations.extend(episode_observations[:remaining])
                teacher_actions.extend(
                    episode_teacher_actions[:remaining]
                )
            episode_observations = []
            episode_teacher_actions = []
            current_episode_reward = 0.0
            if (
                    episode_count >= maximum_episodes
                    and len(observations) < demonstration_steps):
                raise RuntimeError(
                    "DWA warm-up collected only {}/{} successful "
                    "steps after {} episodes (successes={}, "
                    "collisions={}, timeouts={})".format(
                        len(observations),
                        demonstration_steps,
                        episode_count,
                        success_count,
                        collision_count,
                        timeout_count,
                    )
                )
            raw_observation = client.reset()
            normalizer.update(raw_observation)

        collected = len(observations)
        if (
                collected == demonstration_steps
                or attempted_steps >= next_log_attempt):
            elapsed = max(time.time() - collection_start, 1e-6)
            success_rate = (
                float(success_count) / float(episode_count)
                if episode_count else 0.0
            )
            print(
                "warmup_successful_steps={}/{} attempted_steps={} "
                "steps_per_second={:.2f} "
                "episodes={} success_rate={:.3f} collisions={} "
                "timeouts={} mean_final_distance={:.4f}".format(
                    collected,
                    demonstration_steps,
                    attempted_steps,
                    attempted_steps / elapsed,
                    episode_count,
                    success_rate,
                    collision_count,
                    timeout_count,
                    (
                        float(np.mean(final_distances))
                        if final_distances else float("nan")
                    ),
                )
            )
            while next_log_attempt <= attempted_steps:
                next_log_attempt += max(log_interval, 1)

    observations_numpy = np.stack(observations).astype(np.float32)
    teacher_actions_numpy = np.stack(
        teacher_actions
    ).astype(np.float32)
    final_bc_loss = _fit_actor_to_teacher(
        model=model,
        optimizer=optimizer,
        normalizer=normalizer,
        raw_observations=observations_numpy,
        teacher_actions=teacher_actions_numpy,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        maximum_gradient_norm=maximum_gradient_norm,
        stage_label="warmup",
    )
    print(
        "teacher warm-up complete samples={} episodes={} successes={} "
        "collisions={} timeouts={} final_bc_loss={:.6f} "
        "mean_episode_reward={:.4f}".format(
            demonstration_steps,
            episode_count,
            success_count,
            collision_count,
            timeout_count,
            final_bc_loss,
            (
                float(np.mean(episode_rewards))
                if episode_rewards else float("nan")
            ),
        )
    )

    # PPO starts from a fresh episode so rollout boundaries stay correct.
    raw_observation = client.reset()
    normalizer.update(raw_observation)
    return (
        raw_observation,
        observations_numpy,
        teacher_actions_numpy,
    )


def _dagger_training_stage(
        client,
        model,
        optimizer,
        normalizer,
        raw_observation,
        device,
        initial_observations,
        initial_teacher_actions,
        total_steps,
        rounds,
        epochs,
        batch_size,
        beta_start,
        beta_end,
        recent_fraction,
        maximum_gradient_norm):
    """Aggregate DWA labels on states visited by the learned policy."""
    total_steps = int(total_steps)
    rounds = int(rounds)
    initial_observations = np.asarray(
        initial_observations,
        dtype=np.float32,
    )
    initial_teacher_actions = np.asarray(
        initial_teacher_actions,
        dtype=np.float32,
    )
    aggregated_observations = [
        row.copy() for row in initial_observations
    ]
    aggregated_teacher_actions = [
        row.copy() for row in initial_teacher_actions
    ]
    base_round_steps = total_steps // rounds
    remainder = total_steps % rounds

    print(
        "DAgger collecting {} steps in {} rounds "
        "beta={:.2f}->{:.2f} initial_samples={}".format(
            total_steps,
            rounds,
            beta_start,
            beta_end,
            len(aggregated_observations),
        )
    )
    for round_index in range(rounds):
        round_steps = base_round_steps + int(round_index < remainder)
        beta = _linear_schedule(
            beta_start,
            beta_end,
            round_index,
            rounds,
        )
        raw_observation = client.reset()
        normalizer.update(raw_observation)
        episode_count = 0
        success_count = 0
        collision_count = 0
        timeout_count = 0
        teacher_executed_count = 0
        teacher_valid_count = 0
        teacher_restart_count = 0
        round_observations = []
        round_teacher_actions = []
        reward_values = []
        distance_values = []
        learner_teacher_mse_values = []
        collection_start = time.time()

        for step_index in range(round_steps):
            current_observation = np.asarray(
                raw_observation,
                dtype=np.float32,
            ).copy()
            normalized_observation = normalizer.normalize(
                current_observation
            )
            observation_tensor = torch.from_numpy(
                normalized_observation
            ).to(device).unsqueeze(0)
            with torch.no_grad():
                learner_action = model.deterministic_action(
                    observation_tensor
                ).squeeze(0).cpu().numpy()
            execute_teacher = bool(
                np.random.random_sample() < beta
            )
            if execute_teacher:
                (
                    next_raw_observation,
                    reward,
                    done,
                    info,
                ) = client.teacher_step()
                teacher_executed_count += 1
            else:
                (
                    next_raw_observation,
                    reward,
                    done,
                    info,
                ) = client.step(learner_action)

            teacher_valid = bool(
                info.get("teacher_available", False)
            )
            if teacher_valid:
                teacher_action = np.asarray(
                    info["teacher_action"],
                    dtype=np.float32,
                )
                aggregated_observations.append(
                    current_observation
                )
                aggregated_teacher_actions.append(
                    teacher_action.copy()
                )
                round_observations.append(
                    current_observation.copy()
                )
                round_teacher_actions.append(
                    teacher_action.copy()
                )
                teacher_valid_count += 1
                learner_teacher_mse_values.append(float(
                    np.mean(np.square(
                        learner_action - teacher_action
                    ))
                ))
            teacher_restart_count += int(bool(
                info.get("teacher_goal_restarted", False)
            ))

            reward_values.append(float(reward))
            distance = float(info.get(
                "distance",
                np.linalg.norm(next_raw_observation[0:2]),
            ))
            if np.isfinite(distance):
                distance_values.append(distance)
            raw_observation = next_raw_observation
            normalizer.update(raw_observation)

            if done:
                episode_count += 1
                success_count += int(bool(
                    info.get("success", False)
                ))
                collision_count += int(bool(
                    info.get("collision", False)
                ))
                timeout_count += int(bool(
                    info.get("timeout", False)
                ))
                if step_index + 1 < round_steps:
                    raw_observation = client.reset()
                    normalizer.update(raw_observation)

        if not round_observations:
            raise RuntimeError(
                "DAgger round {} received no valid DWA labels".format(
                    round_index + 1
                )
            )
        aggregated_observations_numpy = np.stack(
            aggregated_observations
        ).astype(np.float32)
        aggregated_teacher_actions_numpy = np.stack(
            aggregated_teacher_actions
        ).astype(np.float32)
        round_observations_numpy = np.stack(
            round_observations
        ).astype(np.float32)
        round_teacher_actions_numpy = np.stack(
            round_teacher_actions
        ).astype(np.float32)
        (
            training_observations,
            training_teacher_actions,
        ) = _recent_balanced_dataset(
            aggregated_observations_numpy,
            aggregated_teacher_actions_numpy,
            round_observations_numpy,
            round_teacher_actions_numpy,
            recent_fraction,
        )
        supervised_loss = _fit_actor_to_teacher(
            model=model,
            optimizer=optimizer,
            normalizer=normalizer,
            raw_observations=training_observations,
            teacher_actions=training_teacher_actions,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            maximum_gradient_norm=maximum_gradient_norm,
            stage_label="dagger_{:02d}".format(round_index + 1),
        )
        recent_teacher_loss = _actor_teacher_loss(
            model=model,
            normalizer=normalizer,
            raw_observations=round_observations_numpy,
            teacher_actions=round_teacher_actions_numpy,
            device=device,
        )
        elapsed = max(time.time() - collection_start, 1e-6)
        print(
            "dagger_round={}/{} beta={:.3f} steps={} "
            "steps_per_second={:.2f} dataset={} training_samples={} "
            "recent_fraction={:.3f} teacher_valid={:.3f} "
            "teacher_executed={:.3f} episodes={} successes={} "
            "collisions={} timeouts={} teacher_restarts={} "
            "reward={:.4f} distance={:.4f} "
            "preupdate_teacher_mse={:.5f} "
            "balanced_teacher_loss={:.5f} "
            "postupdate_recent_teacher_loss={:.5f}".format(
                round_index + 1,
                rounds,
                beta,
                round_steps,
                round_steps / elapsed,
                len(aggregated_observations),
                int(training_observations.shape[0]),
                recent_fraction,
                float(teacher_valid_count) / float(max(round_steps, 1)),
                float(teacher_executed_count) / float(
                    max(round_steps, 1)
                ),
                episode_count,
                success_count,
                collision_count,
                timeout_count,
                teacher_restart_count,
                _mean_or_nan(reward_values),
                _mean_or_nan(distance_values),
                _mean_or_nan(learner_teacher_mse_values),
                supervised_loss,
                recent_teacher_loss,
            )
        )

    raw_observation = client.reset()
    normalizer.update(raw_observation)
    return (
        raw_observation,
        np.stack(aggregated_observations).astype(np.float32),
        np.stack(aggregated_teacher_actions).astype(np.float32),
    )


def _recent_balanced_dataset(
        aggregated_observations,
        aggregated_teacher_actions,
        recent_observations,
        recent_teacher_actions,
        recent_fraction):
    """Oversample the newest DAgger round to the requested exposure."""
    recent_fraction = float(recent_fraction)
    aggregate_count = int(aggregated_observations.shape[0])
    recent_count = int(recent_observations.shape[0])
    if recent_fraction <= 0.0 or recent_count >= aggregate_count:
        return (
            aggregated_observations,
            aggregated_teacher_actions,
        )
    required_extra = int(np.ceil(max(
        (
            recent_fraction * aggregate_count - recent_count
        ) / (1.0 - recent_fraction),
        0.0,
    )))
    if required_extra <= 0:
        return (
            aggregated_observations,
            aggregated_teacher_actions,
        )
    sampled_indices = np.random.choice(
        recent_count,
        size=required_extra,
        replace=True,
    )
    return (
        np.concatenate((
            aggregated_observations,
            recent_observations[sampled_indices],
        ), axis=0).astype(np.float32),
        np.concatenate((
            aggregated_teacher_actions,
            recent_teacher_actions[sampled_indices],
        ), axis=0).astype(np.float32),
    )


def _actor_teacher_loss(
        model,
        normalizer,
        raw_observations,
        teacher_actions,
        device):
    normalized_observations = np.stack([
        normalizer.normalize(observation)
        for observation in raw_observations
    ]).astype(np.float32)
    observation_tensor = torch.from_numpy(
        normalized_observations
    ).to(device)
    teacher_action_tensor = torch.from_numpy(
        np.asarray(teacher_actions, dtype=np.float32)
    ).to(device)
    with torch.no_grad():
        predictions = model.deterministic_action(
            observation_tensor
        )
        return float(torch.mean(torch.square(
            predictions - teacher_action_tensor
        )).item())


def _fit_actor_to_teacher(
        model,
        optimizer,
        normalizer,
        raw_observations,
        teacher_actions,
        device,
        epochs,
        batch_size,
        maximum_gradient_norm,
        stage_label):
    raw_observations = np.asarray(
        raw_observations,
        dtype=np.float32,
    )
    teacher_actions = np.asarray(
        teacher_actions,
        dtype=np.float32,
    )
    sample_count = int(raw_observations.shape[0])
    if (
            raw_observations.shape
            != (sample_count, model.OBS_DIM)):
        raise ValueError(
            "teacher observations must have shape (N, {})".format(
                model.OBS_DIM
            )
        )
    if teacher_actions.shape != (sample_count, model.ACTION_DIM):
        raise ValueError(
            "teacher actions must have shape (N, {})".format(
                model.ACTION_DIM
            )
        )
    if sample_count <= 0:
        raise ValueError("teacher dataset must not be empty")

    normalized_observations = np.stack([
        normalizer.normalize(observation)
        for observation in raw_observations
    ]).astype(np.float32)
    observation_tensor = torch.from_numpy(
        normalized_observations
    ).to(device)
    teacher_action_tensor = torch.from_numpy(
        teacher_actions
    ).to(device)
    model.train()
    for epoch in range(int(epochs)):
        indices = np.random.permutation(sample_count)
        total_loss = 0.0
        batch_count = 0
        for start in range(0, sample_count, int(batch_size)):
            batch_indices = torch.from_numpy(
                indices[start:start + int(batch_size)].astype(
                    np.int64
                )
            ).to(device)
            predicted_actions = model.deterministic_action(
                observation_tensor.index_select(0, batch_indices)
            )
            batch_teacher_actions = teacher_action_tensor.index_select(
                0,
                batch_indices,
            )
            loss = torch.mean(torch.square(
                predicted_actions - batch_teacher_actions
            ))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                maximum_gradient_norm,
            )
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1
        print(
            "{}_epoch={:03d}/{} teacher_loss={:.6f}".format(
                stage_label,
                epoch + 1,
                epochs,
                total_loss / float(max(batch_count, 1)),
            )
        )
    return _actor_teacher_loss(
        model=model,
        normalizer=normalizer,
        raw_observations=raw_observations,
        teacher_actions=teacher_actions,
        device=device,
    )


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
    data["advantages"] = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)
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
            unclipped_objective = probability_ratio * batch_advantages
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

            per_sample_teacher_loss = (
                (mean_actions - teacher_actions).pow(2)
            ).mean(dim=1)
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


def _window_mean(values, window_size):
    if not values:
        return float("nan")
    return float(np.mean(values[-int(window_size):]))


def _mean_or_nan(values):
    if not values:
        return float("nan")
    return float(np.mean(values))


def _linear_schedule(start, end, index, count):
    if int(count) <= 1:
        return float(start)
    fraction = float(index) / float(int(count) - 1)
    return float(start) + fraction * (float(end) - float(start))


def _set_policy_log_std(model, optimizer, log_std):
    log_std = float(log_std)
    with torch.no_grad():
        model.log_std.fill_(log_std)
    # Discard Adam momentum accumulated for the previous exploration scale.
    # Keeping it can immediately undo an explicit resume-time reset.
    optimizer.state.pop(model.log_std, None)


def _policy_exploration_std(model):
    with torch.no_grad():
        return float(torch.exp(torch.clamp(
            model.log_std.mean(),
            -5.0,
            1.0,
        )).item())


def _stage_checkpoint_path(output_path, stage):
    root, extension = os.path.splitext(output_path)
    if extension:
        return root + "." + str(stage) + extension
    return output_path + "." + str(stage) + ".pt"


def _warmup_checkpoint_path(output_path):
    return _stage_checkpoint_path(output_path, "warmup")


def _save_checkpoint(
        output_path,
        model,
        optimizer,
        normalizer,
        total_steps,
        update_index,
        teacher_warmup_completed,
        dagger_completed,
        args):
    output_path = os.path.abspath(os.path.expanduser(output_path))
    output_directory = os.path.dirname(output_path)
    if output_directory and not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    torch.save({
        "format_version": 2,
        "policy_type": "planar_base_ppo_dwa_teacher",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "total_steps": int(total_steps),
        "update_index": int(update_index),
        "teacher_warmup_completed": bool(
            teacher_warmup_completed
        ),
        "dagger_completed": bool(dagger_completed),
        "arguments": vars(args),
        "observation_dim": model.OBS_DIM,
        "action_dim": model.ACTION_DIM,
        "observation_semantics": (
            "body_target_xy,body_vw,scan5,previous_action2"
        ),
        "action_semantics": "normalized_linear_velocity_yaw_rate",
        "teacher_type": "move_base_dwa",
        "policy_log_std": model.log_std.detach().cpu().numpy(),
        "policy_exploration_std": np.asarray([
            _policy_exploration_std(model)
        ], dtype=np.float32),
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
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument(
        "--output",
        default="/tmp/mobile_arm_rl_training/planar_dwa_ppo.pt",
    )
    parser.add_argument(
        "--warmup-output",
        help=(
            "post-DAgger checkpoint; defaults to "
            "<output>.warmup.pt"
        ),
    )
    parser.add_argument(
        "--bc-output",
        help=(
            "pure behavior-cloning checkpoint; defaults to "
            "<output>.bc.pt"
        ),
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--resume-log-std",
        type=float,
        help="override checkpoint log_std and clear its Adam state",
    )
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--teacher-coefficient", type=float, default=2.0)
    parser.add_argument("--teacher-decay-steps", type=int, default=100000)
    parser.add_argument(
        "--teacher-warmup-steps",
        type=int,
        default=5000,
        help="DWA demonstration steps collected before PPO",
    )
    parser.add_argument(
        "--teacher-warmup-epochs",
        type=int,
        default=20,
        help="behavior-cloning epochs over DWA demonstrations",
    )
    parser.add_argument(
        "--teacher-warmup-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--teacher-warmup-log-interval",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--teacher-warmup-max-episodes",
        type=int,
        default=200,
        help="abort BC collection if DWA cannot supply enough successes",
    )
    parser.add_argument(
        "--post-warmup-log-std",
        type=float,
        default=-2.5,
        help="exploration log standard deviation used when PPO takes over",
    )
    parser.add_argument(
        "--dagger-steps",
        type=int,
        default=9000,
        help="state-label pairs collected across all DAgger rounds",
    )
    parser.add_argument(
        "--dagger-rounds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--dagger-epochs",
        type=int,
        default=5,
        help="supervised epochs over aggregated data after each round",
    )
    parser.add_argument(
        "--dagger-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dagger-beta-start",
        type=float,
        default=1.0,
        help="probability that DWA executes the action in round one",
    )
    parser.add_argument(
        "--dagger-beta-end",
        type=float,
        default=0.0,
        help="DWA execution probability in the final round",
    )
    parser.add_argument(
        "--dagger-recent-fraction",
        type=float,
        default=0.50,
        help=(
            "minimum training exposure assigned to the newest "
            "DAgger round"
        ),
    )
    parser.add_argument(
        "--dagger-only",
        action="store_true",
        help="save the DAgger checkpoint and exit before PPO",
    )
    parser.add_argument(
        "--maximum-gradient-norm",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[128, 128],
    )
    parser.add_argument("--initial-log-std", type=float, default=-1.5)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--reward-window", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _validate_arguments(args):
    positive_integer_names = (
        "total_steps",
        "rollout_steps",
        "batch_size",
        "ppo_epochs",
        "teacher_warmup_epochs",
        "teacher_warmup_batch_size",
        "teacher_warmup_max_episodes",
        "dagger_rounds",
        "dagger_epochs",
        "dagger_batch_size",
        "reward_window",
    )
    for name in positive_integer_names:
        if int(getattr(args, name)) <= 0:
            raise ValueError("--{} must be positive".format(
                name.replace("_", "-")
            ))
    if int(args.teacher_warmup_steps) < 0:
        raise ValueError("--teacher-warmup-steps cannot be negative")
    if int(args.dagger_steps) < 0:
        raise ValueError("--dagger-steps cannot be negative")
    if int(args.teacher_warmup_log_interval) < 0:
        raise ValueError(
            "--teacher-warmup-log-interval cannot be negative"
        )
    if float(args.maximum_gradient_norm) <= 0.0:
        raise ValueError("--maximum-gradient-norm must be positive")
    for name in ("dagger_beta_start", "dagger_beta_end"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError("--{} must be in [0, 1]".format(
                name.replace("_", "-")
            ))
    if float(args.dagger_beta_start) < float(args.dagger_beta_end):
        raise ValueError(
            "--dagger-beta-start must not be below --dagger-beta-end"
        )
    if not 0.0 <= float(args.dagger_recent_fraction) < 1.0:
        raise ValueError(
            "--dagger-recent-fraction must be in [0, 1)"
        )
    for name in ("post_warmup_log_std", "resume_log_std"):
        value = getattr(args, name)
        if value is None:
            continue
        if not np.isfinite(float(value)):
            raise ValueError("--{} must be finite".format(
                name.replace("_", "-")
            ))
        if not -5.0 <= float(value) <= 1.0:
            raise ValueError(
                "--{} must be in [-5, 1]".format(
                    name.replace("_", "-")
                )
            )


if __name__ == "__main__":
    main()
