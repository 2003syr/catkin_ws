#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train pose-guided tracked-base low-level control with PPO."""

import argparse
import csv
import os
import random
import sys
import time

import numpy as np

try:
    import torch
except ImportError as error:
    raise RuntimeError(
        "Direct planar PPO requires Python 3 with PyTorch"
    ) from error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.hrl4in_low_actor_critic import (
    RunningObservationNormalizer,
)
from training.planar_direct_env_client import (
    PlanarDirectEnvironmentClient,
)
from training.planar_subgoal_actor_critic import (
    PlanarSubgoalActorCritic,
)
from training.ppo_rollout import PPORolloutBuffer


METRIC_FIELDS = (
    "update",
    "steps",
    "steps_per_second",
    "mean_step_reward",
    "episodes",
    "success_rate",
    "cumulative_success_rate",
    "collision_rate",
    "timeout_rate",
    "free_success_rate",
    "offset_success_rate",
    "frontal_success_rate",
    "mean_episode_return",
    "curriculum_stage",
    "mean_distance",
    "mean_minimum_distance",
    "mean_distance_reduction",
    "mean_distance_ratio",
    "mean_progress",
    "mean_local_progress",
    "mean_cross_track_error",
    "mean_heading_error",
    "high_level_replan_rate",
    "mean_action_abs",
    "shield_intervention_rate",
    "policy_loss",
    "value_loss",
    "bc_anchor_loss",
    "bc_anchor_coefficient",
    "bc_anchor_gate_rate",
    "actor_frozen",
    "normalizer_frozen",
    "entropy",
    "exploration_std",
    "evaluation_success_rate",
    "evaluation_collisions",
    "evaluation_mean_distance_ratio",
    "evaluation_shield_rate",
    "evaluation_score",
)

SCAN_START = 6
SCAN_END = 42
SCAN_FIXED_MEAN = 0.5
SCAN_FIXED_VARIANCE = 1.0 / 12.0


def main():
    args = _parse_arguments()
    _validate_arguments(args)
    _set_random_seed(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    model = PlanarSubgoalActorCritic(
        hidden_sizes=args.hidden_sizes,
        initial_log_std=args.initial_log_std,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )
    # Normalize the continuous prefix while preserving the five binary
    # path-validity mask values at the end of the observation.
    normalizer = RunningObservationNormalizer(
        observation_dim=model.OBS_DIM,
        normalized_dim=57,
    )
    total_steps = 0
    update_index = 0
    best_success_rate = -1.0
    best_evaluation = None
    initialization_type = "random"
    initialization_checkpoint = ""
    preserve_fixed_scan_normalization = False
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("policy_type") != "planar_pose_guided_ppo":
            raise RuntimeError(
                "--resume requires a 62-D pose-guided PPO checkpoint"
            )
        if int(checkpoint.get("observation_dim", -1)) != model.OBS_DIM:
            raise RuntimeError("resume observation dimension mismatch")
        if int(checkpoint.get("format_version", 0)) < 3:
            raise RuntimeError(
                "legacy PPO checkpoint uses a shared actor/critic "
                "backbone and cannot be resumed safely"
            )
        if checkpoint.get("teacher_type", "none") != "none":
            raise RuntimeError(
                "cannot resume a teacher-assisted checkpoint"
            )
        model.load_compatible_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        normalizer.load_state_dict(checkpoint["normalizer"])
        preserve_fixed_scan_normalization = bool(
            checkpoint.get("scan_normalization")
            == "fixed_uniform_0_1"
        )
        total_steps = int(checkpoint.get("total_steps", 0))
        update_index = int(checkpoint.get("update_index", 0))
        best_success_rate = float(
            checkpoint.get("best_success_rate", -1.0)
        )
        best_evaluation = checkpoint.get("best_evaluation")
        print(
            "resumed checkpoint={} total_steps={} update={} "
            "exploration_std={:.5f}".format(
                args.resume,
                total_steps,
                update_index,
                _exploration_std(model),
            )
        )
        initialization_type = str(
            checkpoint.get("initialization_type", "resume")
        )
        initialization_checkpoint = args.resume
    elif args.bc_checkpoint:
        bc_checkpoint = torch.load(
            args.bc_checkpoint,
            map_location=device,
        )
        if bc_checkpoint.get("policy_type") not in (
                "planar_pose_guided_bc",):
            raise RuntimeError(
                "--bc-checkpoint is not a pose-guided tracking BC model"
            )
        bc_observation_dim = bc_checkpoint.get("observation_dim")
        if bc_observation_dim is None:
            bc_observation_dim = bc_checkpoint.get(
                "normalizer", {}
            ).get("observation_dim")
        if int(bc_observation_dim or -1) != model.OBS_DIM:
            raise RuntimeError("BC observation dimension mismatch")
        bc_action_dim = bc_checkpoint.get("action_dim")
        if (
                bc_action_dim is not None
                and int(bc_action_dim) != model.ACTION_DIM):
            raise RuntimeError("BC action dimension mismatch")
        bc_hidden_sizes = list(bc_checkpoint.get(
            "hidden_sizes",
            bc_checkpoint.get("arguments", {}).get(
                "hidden_sizes", []
            ),
        ))
        if bc_hidden_sizes != list(args.hidden_sizes):
            raise RuntimeError(
                "BC hidden sizes {} do not match PPO hidden sizes {}".format(
                    bc_hidden_sizes,
                    list(args.hidden_sizes),
                )
            )
        model.load_compatible_state_dict(
            bc_checkpoint["model"],
            actor_only=True,
        )
        normalizer.load_state_dict(bc_checkpoint["normalizer"])
        preserve_fixed_scan_normalization = bool(
            bc_checkpoint.get("scan_normalization")
            == "fixed_uniform_0_1"
        )
        bc_teacher_type = _decode_dataset_text(
            bc_checkpoint.get("teacher_type", "unknown")
        )
        initialization_type = (
            "teb_path_bc"
            if bc_teacher_type == "teb_local_plan_cmd_vel"
            else "observable_tracker_bc"
        )
        initialization_checkpoint = args.bc_checkpoint
        print(
            "initialized actor from BC checkpoint={} teacher_type={} "
            "validation_mse={} exploration_std={:.5f}; "
            "critic remains newly initialized".format(
                args.bc_checkpoint,
                bc_teacher_type,
                bc_checkpoint.get("validation_mse", "unknown"),
                _exploration_std(model),
            )
        )

    bc_raw_observations = None
    bc_actions = None
    if args.bc_dataset:
        bc_raw_observations, bc_actions = _load_bc_dataset(
            args.bc_dataset,
            model.OBS_DIM,
            model.ACTION_DIM,
        )
        print(
            "offline BC anchor loaded dataset={} samples={} "
            "coefficient={:.4f} decay_steps={}".format(
                args.bc_dataset,
                bc_raw_observations.shape[0],
                args.bc_coefficient,
                args.bc_decay_steps,
            )
        )

    buffer = PPORolloutBuffer(
        capacity=args.rollout_steps,
        observation_dim=model.OBS_DIM,
        action_dim=model.ACTION_DIM,
    )
    client = PlanarDirectEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )
    metrics_path = (
        args.metrics
        if args.metrics
        else _metrics_path(args.output)
    )
    metrics_writer, metrics_stream = _open_metrics(
        metrics_path,
        append=bool(args.resume),
    )
    zero_teacher_action = np.zeros(
        model.ACTION_DIM,
        dtype=np.float32,
    )
    episode_count = 0
    success_count = 0
    collision_count = 0
    timeout_count = 0
    current_episode_return = 0.0
    completed_episode_returns = []
    episode_minimum_distances = []
    episode_distance_reductions = []
    episode_distance_ratios = []
    episode_successes = []
    episode_collisions = []
    episode_timeouts = []
    episode_scenario_labels = []
    step_rewards = []
    distances = []
    progresses = []
    local_progresses = []
    cross_track_errors = []
    heading_errors = []
    high_level_replans = []
    action_magnitudes = []
    shield_events = []
    rollout_bc_gates = []
    start_time = time.time()
    run_start_steps = total_steps
    raw_observation = None
    last_done = True

    try:
        raw_observation, reset_info = client.reset(
            return_info=True,
            training_step=total_steps,
        )
        current_scenario_label = str(
            reset_info.get("scenario_label", "unspecified")
        )
        current_curriculum_stage = str(
            reset_info.get(
                "curriculum_stage",
                client.curriculum_stage,
            )
        )
        if total_steps > args.normalizer_freeze_steps:
            _update_normalizer(
                normalizer,
                raw_observation,
                preserve_fixed_scan_normalization,
            )
        while total_steps < args.total_steps:
            normalized_observation = normalizer.normalize(
                raw_observation
            )
            observation_tensor = torch.from_numpy(
                normalized_observation
            ).to(device).unsqueeze(0)
            with torch.no_grad():
                action_tensor, log_probability, value = model.act(
                    observation_tensor
                )
            action = action_tensor.squeeze(0).cpu().numpy()
            (
                next_raw_observation,
                reward,
                done,
                info,
            ) = client.step(action)
            buffer.add(
                observation=normalized_observation,
                action=action,
                teacher_action=zero_teacher_action,
                teacher_valid=False,
                log_probability=float(log_probability.item()),
                value=float(value.item()),
                reward=float(reward),
                done=done,
            )
            total_steps += 1
            current_episode_return += float(reward)
            step_rewards.append(float(reward))
            distances.append(float(info.get(
                "distance",
                np.linalg.norm(next_raw_observation[0:2]),
            )))
            progresses.append(float(info.get("progress", 0.0)))
            local_progresses.append(float(
                info.get("local_progress", 0.0)
            ))
            cross_track_errors.append(float(
                info.get("path_cross_track_error", 0.0)
            ))
            heading_errors.append(abs(float(
                info.get("path_heading_error", 0.0)
            )))
            high_level_replans.append(float(bool(
                info.get("high_level_replanned", False)
            )))
            action_magnitudes.append(float(np.mean(np.abs(action))))
            shield_events.append(float(bool(
                info.get("shield_intervened", False)
            )))
            rollout_bc_gates.append(float(
                float(info.get(
                    "shield_front_clearance",
                    float("inf"),
                )) >= args.bc_free_clearance
                and not bool(info.get("shield_front_blocked", False))
            ))
            last_done = bool(done)

            if done:
                episode_count += 1
                episode_success = int(bool(
                    info.get("success", False)
                ))
                episode_collision = int(bool(
                    info.get("collision", False)
                ))
                episode_timeout = int(bool(
                    info.get("timeout", False)
                ))
                success_count += episode_success
                collision_count += episode_collision
                timeout_count += episode_timeout
                episode_successes.append(episode_success)
                episode_collisions.append(episode_collision)
                episode_timeouts.append(episode_timeout)
                episode_scenario_labels.append(
                    current_scenario_label
                )
                completed_episode_returns.append(
                    current_episode_return
                )
                episode_minimum_distances.append(float(
                    info.get("minimum_distance", np.nan)
                ))
                episode_distance_reductions.append(float(
                    info.get("distance_reduction", 0.0)
                ))
                episode_distance_ratios.append(float(
                    info.get("distance_ratio", 1.0)
                ))
                current_episode_return = 0.0
                raw_observation, reset_info = client.reset(
                    return_info=True,
                    training_step=total_steps,
                )
                current_scenario_label = str(
                    reset_info.get(
                        "scenario_label",
                        "unspecified",
                    )
                )
                current_curriculum_stage = str(
                    reset_info.get(
                        "curriculum_stage",
                        client.curriculum_stage,
                    )
                )
            else:
                raw_observation = next_raw_observation
            if total_steps > args.normalizer_freeze_steps:
                _update_normalizer(
                    normalizer,
                    raw_observation,
                    preserve_fixed_scan_normalization,
                )

            if buffer.full:
                scheduled_bc_coefficient = _bc_anchor_coefficient(
                    args.bc_coefficient,
                    args.bc_decay_steps,
                    total_steps,
                )
                bc_gate_rate = _window_mean(
                    rollout_bc_gates,
                    len(rollout_bc_gates),
                )
                bc_coefficient = (
                    scheduled_bc_coefficient * bc_gate_rate
                )
                if bc_raw_observations is not None:
                    normalized_bc_observations = normalizer.normalize(
                        bc_raw_observations
                    )
                    bc_observation_tensor = torch.from_numpy(
                        normalized_bc_observations
                    ).to(device)
                    bc_action_tensor = torch.from_numpy(
                        bc_actions
                    ).to(device)
                else:
                    bc_observation_tensor = None
                    bc_action_tensor = None
                    bc_coefficient = 0.0
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
                losses = _ppo_update(
                    model=model,
                    optimizer=optimizer,
                    buffer=buffer,
                    device=device,
                    clip_ratio=args.clip_ratio,
                    value_coefficient=args.value_coefficient,
                    entropy_coefficient=args.entropy_coefficient,
                    maximum_gradient_norm=args.maximum_gradient_norm,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=args.batch_size,
                    bc_observations=bc_observation_tensor,
                    bc_actions=bc_action_tensor,
                    bc_coefficient=bc_coefficient,
                    actor_frozen=bool(
                        total_steps <= args.actor_freeze_steps
                    ),
                )
                buffer.clear()
                rollout_bc_gates = []
                update_index += 1
                evaluation = _empty_evaluation_metrics()
                if (
                        args.evaluation_interval > 0
                        and update_index % args.evaluation_interval == 0):
                    # Preserve the just-finished PPO update before the
                    # evaluation changes routes and resets the environment.
                    # A navigation/reset failure must not discard the latest
                    # training progress.
                    _save_checkpoint(
                        _stage_path(
                            args.output,
                            "pre_eval_update_{:04d}".format(
                                update_index
                            ),
                        ),
                        model,
                        optimizer,
                        normalizer,
                        total_steps,
                        update_index,
                        best_success_rate,
                        best_evaluation,
                        initialization_type,
                        initialization_checkpoint,
                        preserve_fixed_scan_normalization,
                        args,
                    )
                    evaluation = _deterministic_evaluation(
                        client=client,
                        model=model,
                        normalizer=normalizer,
                        device=device,
                        episodes=args.evaluation_episodes,
                        max_steps=args.evaluation_max_steps,
                    )
                    current_episode_return = 0.0
                    raw_observation, reset_info = client.reset(
                        return_info=True,
                        training_step=total_steps,
                    )
                    current_scenario_label = str(
                        reset_info.get(
                            "scenario_label",
                            "unspecified",
                        )
                    )
                    current_curriculum_stage = str(
                        reset_info.get(
                            "curriculum_stage",
                            client.curriculum_stage,
                        )
                    )
                    if total_steps > args.normalizer_freeze_steps:
                        _update_normalizer(
                            normalizer,
                            raw_observation,
                            preserve_fixed_scan_normalization,
                        )
                    last_done = True
                elapsed = max(time.time() - start_time, 1.0e-6)
                success_rate = _window_mean(
                    episode_successes, args.episode_window
                )
                metrics = {
                    "update": update_index,
                    "steps": total_steps,
                    "steps_per_second": (
                        total_steps - run_start_steps
                    ) / elapsed,
                    "mean_step_reward": _window_mean(
                        step_rewards, args.metric_window
                    ),
                    "episodes": episode_count,
                    "success_rate": success_rate,
                    "cumulative_success_rate": _rate(
                        success_count, episode_count
                    ),
                    "collision_rate": _window_mean(
                        episode_collisions, args.episode_window
                    ),
                    "timeout_rate": _window_mean(
                        episode_timeouts, args.episode_window
                    ),
                    "free_success_rate": _scenario_success_rate(
                        episode_scenario_labels,
                        episode_successes,
                        "free",
                        args.episode_window,
                    ),
                    "offset_success_rate": _scenario_success_rate(
                        episode_scenario_labels,
                        episode_successes,
                        "offset",
                        args.episode_window,
                    ),
                    "frontal_success_rate": _scenario_success_rate(
                        episode_scenario_labels,
                        episode_successes,
                        "frontal",
                        args.episode_window,
                    ),
                    "mean_episode_return": _window_mean(
                        completed_episode_returns,
                        args.episode_window,
                    ),
                    "curriculum_stage": current_curriculum_stage,
                    "mean_distance": _window_mean(
                        distances, args.metric_window
                    ),
                    "mean_minimum_distance": _window_mean(
                        episode_minimum_distances,
                        args.episode_window,
                    ),
                    "mean_distance_reduction": _window_mean(
                        episode_distance_reductions,
                        args.episode_window,
                    ),
                    "mean_distance_ratio": _window_mean(
                        episode_distance_ratios,
                        args.episode_window,
                    ),
                    "mean_progress": _window_mean(
                        progresses, args.metric_window
                    ),
                    "mean_local_progress": _window_mean(
                        local_progresses, args.metric_window
                    ),
                    "mean_cross_track_error": _window_mean(
                        cross_track_errors, args.metric_window
                    ),
                    "mean_heading_error": _window_mean(
                        heading_errors, args.metric_window
                    ),
                    "high_level_replan_rate": _window_mean(
                        high_level_replans, args.metric_window
                    ),
                    "mean_action_abs": _window_mean(
                        action_magnitudes, args.metric_window
                    ),
                    "shield_intervention_rate": _window_mean(
                        shield_events, args.metric_window
                    ),
                    "policy_loss": losses["policy_loss"],
                    "value_loss": losses["value_loss"],
                    "bc_anchor_loss": losses["bc_anchor_loss"],
                    "bc_anchor_coefficient": bc_coefficient,
                    "bc_anchor_gate_rate": bc_gate_rate,
                    "actor_frozen": int(
                        total_steps <= args.actor_freeze_steps
                    ),
                    "normalizer_frozen": int(
                        total_steps <= args.normalizer_freeze_steps
                    ),
                    "entropy": losses["entropy"],
                    "exploration_std": _exploration_std(model),
                    "evaluation_success_rate": evaluation[
                        "success_rate"
                    ],
                    "evaluation_collisions": evaluation["collisions"],
                    "evaluation_mean_distance_ratio": evaluation[
                        "mean_distance_ratio"
                    ],
                    "evaluation_shield_rate": evaluation[
                        "shield_rate"
                    ],
                    "evaluation_score": evaluation["score"],
                }
                metrics_writer.writerow(metrics)
                metrics_stream.flush()
                print(
                    "update={update} steps={steps} "
                    "steps_per_second={steps_per_second:.2f} "
                    "reward={mean_step_reward:.4f} "
                    "episodes={episodes} "
                    "success_rate={success_rate:.3f} "
                    "cumulative_success_rate="
                    "{cumulative_success_rate:.3f} "
                    "collision_rate={collision_rate:.3f} "
                    "timeout_rate={timeout_rate:.3f} "
                    "scenario_success=(free:{free_success_rate:.3f},"
                    "offset:{offset_success_rate:.3f},"
                    "frontal:{frontal_success_rate:.3f}) "
                    "stage={curriculum_stage} "
                    "distance={mean_distance:.4f} "
                    "min_distance={mean_minimum_distance:.4f} "
                    "distance_reduction={mean_distance_reduction:.4f} "
                    "distance_ratio={mean_distance_ratio:.3f} "
                    "progress={mean_progress:.5f} "
                    "local_progress={mean_local_progress:.5f} "
                    "cross_track={mean_cross_track_error:.4f} "
                    "heading_error={mean_heading_error:.4f} "
                    "high_replan={high_level_replan_rate:.3f} "
                    "shield={shield_intervention_rate:.3f} "
                    "action_abs={mean_action_abs:.4f} "
                    "policy_loss={policy_loss:.5f} "
                    "value_loss={value_loss:.5f} "
                    "bc_loss={bc_anchor_loss:.5f} "
                    "bc_coef={bc_anchor_coefficient:.5f} "
                    "bc_gate={bc_anchor_gate_rate:.3f} "
                    "actor_frozen={actor_frozen} "
                    "normalizer_frozen={normalizer_frozen} "
                    "entropy={entropy:.5f} "
                    "exploration_std={exploration_std:.5f} "
                    "eval_success={evaluation_success_rate:.3f} "
                    "eval_collisions={evaluation_collisions:.0f} "
                    "eval_ratio={evaluation_mean_distance_ratio:.3f} "
                    "eval_shield={evaluation_shield_rate:.3f} "
                    "eval_score={evaluation_score:.4f}".format(
                        **metrics
                    )
                )
                if (
                        np.isfinite(evaluation["score"])
                        and _evaluation_is_better(
                            evaluation,
                            best_evaluation,
                        )):
                    best_evaluation = dict(evaluation)
                    best_success_rate = float(
                        evaluation["success_rate"]
                    )
                    _save_checkpoint(
                        _stage_path(args.output, "best"),
                        model,
                        optimizer,
                        normalizer,
                        total_steps,
                        update_index,
                        best_success_rate,
                        best_evaluation,
                        initialization_type,
                        initialization_checkpoint,
                        preserve_fixed_scan_normalization,
                        args,
                    )
                if (
                        args.checkpoint_interval > 0
                        and update_index % args.checkpoint_interval == 0):
                    _save_checkpoint(
                        _stage_path(
                            args.output,
                            "update_{:04d}".format(update_index),
                        ),
                        model,
                        optimizer,
                        normalizer,
                        total_steps,
                        update_index,
                        best_success_rate,
                        best_evaluation,
                        initialization_type,
                        initialization_checkpoint,
                        preserve_fixed_scan_normalization,
                        args,
                    )
    finally:
        client.close()
        metrics_stream.close()

    _save_checkpoint(
        args.output,
        model,
        optimizer,
        normalizer,
        total_steps,
        update_index,
        best_success_rate,
        best_evaluation,
        initialization_type,
        initialization_checkpoint,
        preserve_fixed_scan_normalization,
        args,
    )
    print("model saved: {}".format(args.output))
    print("training metrics: {}".format(metrics_path))


def _ppo_update(
        model,
        optimizer,
        buffer,
        device,
        clip_ratio,
        value_coefficient,
        entropy_coefficient,
        maximum_gradient_norm,
        ppo_epochs,
        batch_size,
        bc_observations=None,
        bc_actions=None,
        bc_coefficient=0.0,
        actor_frozen=False):
    data = buffer.tensors(device)
    advantages = data["advantages"]
    data["advantages"] = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1.0e-8)
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "bc_anchor_loss": 0.0,
        "entropy": 0.0,
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
            (
                log_probabilities,
                entropy,
                values,
            ) = model.evaluate_actions(observations, actions)
            probability_ratio = torch.exp(
                log_probabilities - old_log_probabilities
            )
            unclipped = probability_ratio * batch_advantages
            clipped = torch.clamp(
                probability_ratio,
                1.0 - clip_ratio,
                1.0 + clip_ratio,
            ) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
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
            bc_anchor_loss = torch.zeros((), device=device)
            if (
                    bc_observations is not None
                    and bc_actions is not None
                    and bc_coefficient > 0.0):
                anchor_count = int(bc_observations.shape[0])
                anchor_batch_size = min(int(batch_size), anchor_count)
                anchor_indices = torch.randint(
                    low=0,
                    high=anchor_count,
                    size=(anchor_batch_size,),
                    device=device,
                )
                anchor_predictions = model.deterministic_action(
                    bc_observations.index_select(0, anchor_indices)
                )
                anchor_targets = bc_actions.index_select(
                    0,
                    anchor_indices,
                )
                bc_anchor_loss = (
                    anchor_predictions - anchor_targets
                ).pow(2).mean()
            if actor_frozen:
                # Actor and its observation contract have already passed the
                # deterministic TEB gate.  During critic warm-up, optimize
                # only the independent critic backbone/value head so early,
                # high-variance advantages cannot erase the BC behavior.
                total_loss = value_coefficient * value_loss
            else:
                total_loss = (
                    policy_loss
                    + value_coefficient * value_loss
                    + bc_coefficient * bc_anchor_loss
                    - entropy_coefficient * entropy_mean
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
            totals["bc_anchor_loss"] += float(
                bc_anchor_loss.item()
            )
            totals["entropy"] += float(entropy_mean.item())
            totals["updates"] += 1
    divisor = float(max(totals["updates"], 1))
    return {
        "policy_loss": totals["policy_loss"] / divisor,
        "value_loss": totals["value_loss"] / divisor,
        "bc_anchor_loss": totals["bc_anchor_loss"] / divisor,
        "entropy": totals["entropy"] / divisor,
    }


def _deterministic_evaluation(
        client,
        model,
        normalizer,
        device,
        episodes,
        max_steps):
    successes = 0
    collisions = 0
    shield_events = 0
    executed_steps = 0
    distance_ratios = []
    scenario_results = {}
    model.eval()
    try:
        for episode_index in range(int(episodes)):
            observation, reset_info = client.reset(
                return_info=True,
                evaluation_index=episode_index,
            )
            scenario = str(
                reset_info.get("scenario_label", "unspecified")
            )
            start_distance = float(reset_info.get(
                "initial_distance",
                np.linalg.norm(
                    np.asarray(
                        reset_info["subgoal"],
                        dtype=np.float32,
                    )[0:2]
                ),
            ))
            final_info = {}
            for unused_step in range(int(max_steps)):
                normalized = normalizer.normalize(observation)
                tensor = torch.from_numpy(
                    normalized
                ).to(device).unsqueeze(0)
                with torch.no_grad():
                    action = model.deterministic_action(
                        tensor
                    ).squeeze(0).cpu().numpy()
                observation, unused_reward, done, info = client.step(
                    action
                )
                executed_steps += 1
                shield_events += int(bool(
                    info.get("shield_intervened", False)
                ))
                final_info = info
                if done:
                    break
            success = int(bool(final_info.get("success", False)))
            collision = int(bool(final_info.get("collision", False)))
            final_distance = float(final_info.get(
                "distance",
                start_distance,
            ))
            if start_distance <= 1.0e-3:
                ratio = 0.0 if success else 1.0
            else:
                ratio = final_distance / start_distance
            successes += success
            collisions += collision
            distance_ratios.append(ratio)
            scenario_values = scenario_results.setdefault(
                scenario,
                {"successes": 0, "episodes": 0},
            )
            scenario_values["successes"] += success
            scenario_values["episodes"] += 1
    finally:
        model.train()

    success_rate = _rate(successes, episodes)
    collision_rate = _rate(collisions, episodes)
    mean_distance_ratio = float(np.mean(distance_ratios))
    shield_rate = _rate(shield_events, executed_steps)
    score = float(
        success_rate
        - 2.0 * collision_rate
        - 0.10 * mean_distance_ratio
        - 0.01 * shield_rate
    )
    scenario_text = ",".join(
        "{}:{:.3f}".format(
            name,
            _rate(values["successes"], values["episodes"]),
        )
        for name, values in sorted(scenario_results.items())
    )
    print(
        "deterministic_evaluation episodes={} success_rate={:.3f} "
        "collisions={} distance_ratio={:.3f} shield_rate={:.3f} "
        "score={:.4f} scenario_success=({})".format(
            episodes,
            success_rate,
            collisions,
            mean_distance_ratio,
            shield_rate,
            score,
            scenario_text,
        )
    )
    return {
        "success_rate": success_rate,
        "collisions": int(collisions),
        "mean_distance_ratio": mean_distance_ratio,
        "shield_rate": shield_rate,
        "score": score,
    }


def _empty_evaluation_metrics():
    return {
        "success_rate": float("nan"),
        "collisions": float("nan"),
        "mean_distance_ratio": float("nan"),
        "shield_rate": float("nan"),
        "score": float("nan"),
    }


def _evaluation_is_better(candidate, incumbent):
    if incumbent is None:
        return True
    candidate_key = (
        -int(candidate["collisions"]),
        float(candidate["success_rate"]),
        -float(candidate["mean_distance_ratio"]),
        -float(candidate["shield_rate"]),
    )
    incumbent_key = (
        -int(incumbent["collisions"]),
        float(incumbent["success_rate"]),
        -float(incumbent["mean_distance_ratio"]),
        -float(incumbent["shield_rate"]),
    )
    return candidate_key > incumbent_key


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


def _update_normalizer(
        normalizer,
        observation,
        preserve_fixed_scan_normalization):
    normalizer.update(observation)
    if preserve_fixed_scan_normalization:
        normalizer.mean[SCAN_START:SCAN_END] = SCAN_FIXED_MEAN
        normalizer.variance[SCAN_START:SCAN_END] = (
            SCAN_FIXED_VARIANCE
        )


def _save_checkpoint(
        output_path,
        model,
        optimizer,
        normalizer,
        total_steps,
        update_index,
        best_success_rate,
        best_evaluation,
        initialization_type,
        initialization_checkpoint,
        preserve_fixed_scan_normalization,
        args):
    output_path = os.path.abspath(os.path.expanduser(output_path))
    output_directory = os.path.dirname(output_path)
    if output_directory and not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    torch.save({
        "format_version": 5,
        "policy_type": "planar_pose_guided_ppo",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "total_steps": int(total_steps),
        "update_index": int(update_index),
        "best_success_rate": float(best_success_rate),
        "best_evaluation": best_evaluation,
        "initialization_type": str(initialization_type),
        "initialization_checkpoint": str(initialization_checkpoint),
        "scan_normalization": (
            "fixed_uniform_0_1"
            if preserve_fixed_scan_normalization
            else "running_statistics"
        ),
        "arguments": vars(args),
        "hidden_sizes": list(args.hidden_sizes),
        "observation_dim": model.OBS_DIM,
        "action_dim": model.ACTION_DIM,
        "observation_semantics": (
            "local_subgoal_xy2,subgoal_yaw_sin_cos2,body_vw2,"
            "scan36,path_preview_xy10,path_metrics3,"
            "previous_action2,path_mask5"
        ),
        "action_semantics": "normalized_linear_velocity_yaw_rate",
        "teacher_type": "none",
        "policy_log_std": model.log_std.detach().cpu().numpy(),
        "policy_exploration_std": np.asarray([
            _exploration_std(model)
        ], dtype=np.float32),
    }, output_path)


def _open_metrics(path, append):
    path = os.path.abspath(os.path.expanduser(path))
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    stream = open(path, "a" if append else "w", newline="")
    writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
    if not append or not exists:
        writer.writeheader()
        stream.flush()
    return writer, stream


def _load_bc_dataset(path, observation_dim, action_dim):
    dataset = np.load(
        os.path.abspath(os.path.expanduser(path)),
        allow_pickle=False,
    )
    try:
        observations = np.asarray(
            dataset["observations"],
            dtype=np.float32,
        )
        actions = np.asarray(
            dataset["teacher_actions"],
            dtype=np.float32,
        )
        teacher_type = _decode_dataset_text(
            dataset["teacher_type"][0]
        )
    finally:
        dataset.close()
    if observations.ndim != 2 or observations.shape[1] != observation_dim:
        raise ValueError("BC anchor observations have invalid shape")
    if actions.shape != (observations.shape[0], action_dim):
        raise ValueError("BC anchor actions have invalid shape")
    supported_teachers = (
        "observable_rule_based_planar_tracker",
        "teb_local_plan_cmd_vel",
    )
    if teacher_type not in supported_teachers:
        raise ValueError(
            "BC anchor dataset uses an unsupported teacher: {}".format(
                teacher_type
            )
        )
    if not np.all(np.isfinite(observations)):
        raise ValueError("BC anchor observations contain non-finite values")
    if not np.all(np.isfinite(actions)):
        raise ValueError("BC anchor actions contain non-finite values")
    return observations, actions


def _decode_dataset_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _bc_anchor_coefficient(initial, decay_steps, total_steps):
    initial = float(initial)
    if initial <= 0.0:
        return 0.0
    progress = min(
        1.0,
        float(total_steps) / float(max(int(decay_steps), 1)),
    )
    return initial * (1.0 - progress)


def _exploration_std(model):
    with torch.no_grad():
        return float(torch.exp(torch.clamp(
            model.log_std.mean(),
            -5.0,
            1.0,
        )).item())


def _window_mean(values, count):
    if not values:
        return 0.0
    return float(np.mean(values[-int(count):]))


def _scenario_success_rate(labels, successes, scenario, count):
    values = [
        success
        for label, success in zip(
            labels[-int(count):],
            successes[-int(count):],
        )
        if str(label) == str(scenario)
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _rate(numerator, denominator):
    if int(denominator) <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _metrics_path(output_path):
    root, unused_extension = os.path.splitext(output_path)
    return root + ".metrics.csv"


def _stage_path(output_path, stage):
    root, extension = os.path.splitext(output_path)
    if extension:
        return root + "." + str(stage) + extension
    return output_path + "." + str(stage) + ".pt"


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
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument(
        "--output",
        default="/tmp/mobile_arm_rl_training/planar_direct_ppo.pt",
    )
    parser.add_argument("--metrics")
    parser.add_argument("--resume")
    parser.add_argument(
        "--bc-checkpoint",
        help=(
            "initialize only the actor and normalizer from the "
            "observable rule-based or TEB path-guided BC model"
        ),
    )
    parser.add_argument(
        "--bc-dataset",
        help=(
            "offline observable-tracker or TEB dataset used only to "
            "anchor the actor during PPO updates"
        ),
    )
    parser.add_argument("--bc-coefficient", type=float, default=1.0)
    parser.add_argument("--bc-decay-steps", type=int, default=50000)
    parser.add_argument(
        "--actor-freeze-steps",
        type=int,
        default=0,
        help=(
            "train only the critic until this absolute step count; "
            "preserves a validated BC actor during critic warm-up"
        ),
    )
    parser.add_argument(
        "--normalizer-freeze-steps",
        type=int,
        default=0,
        help=(
            "keep the BC observation normalizer unchanged until this "
            "absolute step count"
        ),
    )
    parser.add_argument(
        "--bc-free-clearance",
        type=float,
        default=0.45,
        help=(
            "apply the offline BC anchor only in rollouts whose forward "
            "footprint clearance is at least this distance"
        ),
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
    parser.add_argument("--entropy-coefficient", type=float, default=0.003)
    parser.add_argument("--maximum-gradient-norm", type=float, default=0.5)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[128, 128],
    )
    parser.add_argument("--initial-log-std", type=float, default=-1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--metric-window", type=int, default=1000)
    parser.add_argument("--episode-window", type=int, default=20)
    parser.add_argument("--best-min-episodes", type=int, default=10)
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=20,
        help="run the fixed deterministic suite every N PPO updates; 0 disables",
    )
    parser.add_argument("--evaluation-episodes", type=int, default=9)
    parser.add_argument("--evaluation-max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _validate_arguments(args):
    if args.resume and args.bc_checkpoint:
        raise ValueError(
            "--resume and --bc-checkpoint are mutually exclusive"
        )
    if args.bc_dataset is None and args.bc_coefficient > 0.0:
        args.bc_coefficient = 0.0
    for name in (
            "total_steps",
            "rollout_steps",
            "batch_size",
            "ppo_epochs",
            "metric_window",
            "episode_window",
            "best_min_episodes"):
        if int(getattr(args, name)) <= 0:
            raise ValueError("--{} must be positive".format(
                name.replace("_", "-")
            ))
    if float(args.learning_rate) <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if float(args.maximum_gradient_norm) <= 0.0:
        raise ValueError("--maximum-gradient-norm must be positive")
    if float(args.bc_coefficient) < 0.0:
        raise ValueError("--bc-coefficient cannot be negative")
    if int(args.actor_freeze_steps) < 0:
        raise ValueError("--actor-freeze-steps cannot be negative")
    if int(args.normalizer_freeze_steps) < 0:
        raise ValueError(
            "--normalizer-freeze-steps cannot be negative"
        )
    if int(args.bc_decay_steps) <= 0:
        raise ValueError("--bc-decay-steps must be positive")
    if float(args.bc_free_clearance) <= 0.0:
        raise ValueError("--bc-free-clearance must be positive")
    if int(args.evaluation_interval) < 0:
        raise ValueError("--evaluation-interval cannot be negative")
    if (
            int(args.evaluation_episodes) <= 0
            or int(args.evaluation_max_steps) <= 0):
        raise ValueError(
            "--evaluation-episodes and --evaluation-max-steps "
            "must be positive"
        )
    if not 0.0 < float(args.gamma) <= 1.0:
        raise ValueError("--gamma must be in (0, 1]")
    if not 0.0 <= float(args.gae_lambda) <= 1.0:
        raise ValueError("--gae-lambda must be in [0, 1]")


if __name__ == "__main__":
    main()
