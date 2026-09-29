#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Teacher-guided PPO for the full fused 66-D/8-D low-level policy.

This is the non-residual alternative to ``train_fused_low_residual_ppo``.
The actor emits the complete tracked-base/arm action.  Besides teacher-guided
PPO, the trainer supports critic-only warm-up, a frozen reference actor, and
complete upper/lower detour rollouts with side-balanced optimization.
"""

import argparse
import collections
import copy
import os
import random
import sys
import time

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_low_actor_critic import FusedLowActorCritic  # noqa: E402
from training.fused_low_env_client import FusedLowEnvironmentClient  # noqa: E402
from training.fused_reach_scenarios import (  # noqa: E402
    FusedReachScenarioSampler,
    filter_scenario_categories,
    load_fused_reach_scenarios,
    scenario_category_counts,
    scenario_step_budget as scenario_budget_for,
    split_fused_reach_scenarios,
)
from training.hrl4in_low_actor_critic import (  # noqa: E402
    RunningObservationNormalizer,
)
from training.ppo_rollout import PPORolloutBuffer  # noqa: E402


def main():
    args = _parse_arguments()
    _set_random_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    model = FusedLowActorCritic(
        hidden_sizes=args.hidden_sizes,
        initial_log_std=args.initial_log_std,
    ).to(device)
    if args.freeze_log_std:
        model.log_std.requires_grad_(False)
    normalizer = RunningObservationNormalizer(
        observation_dim=model.OBS_DIM,
        normalized_dim=52,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
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
                args.resume, total_steps, update_index
            )
        )
    elif args.bc_checkpoint:
        checkpoint = torch.load(args.bc_checkpoint, map_location=device)
        if int(checkpoint.get("observation_dim", -1)) != model.OBS_DIM:
            raise ValueError("actor checkpoint observation_dim must be 66")
        if int(checkpoint.get("action_dim", -1)) != model.ACTION_DIM:
            raise ValueError("actor checkpoint action_dim must be 8")
        model.load_compatible_state_dict(
            checkpoint["model"],
            actor_only=True,
        )
        if checkpoint.get("normalizer") is not None:
            normalizer.load_state_dict(checkpoint["normalizer"])
        print(
            "initialized actor from checkpoint={} critic=new "
            "optimizer=new log_std=new teacher_type={}".format(
                args.bc_checkpoint,
                checkpoint.get("teacher_type", "unknown"),
            )
        )

    reference_model = _load_reference_model(model, args, device)
    training_start_steps = total_steps

    update_normalizer = bool(
        args.update_observation_normalizer
        or (not args.bc_checkpoint and not args.resume)
    )
    print(
        "observation_normalizer_updates={} source={}".format(
            update_normalizer,
            "online" if update_normalizer else "checkpoint_frozen",
        )
    )
    print(
        "box_action_gate_contract={}".format(
            "fixed_base_then_arm"
            if args.fixed_box_action_gate else "observation_mask"
        )
    )

    scenario_sampler = None
    scenario_set = None
    scenario_step_budget = None
    scenario_episode_counts = collections.Counter()
    scenario_success_counts = collections.Counter()
    if args.scenario_set:
        scenario_set = load_fused_reach_scenarios(args.scenario_set)
        eligible = filter_scenario_categories(
            scenario_set["scenarios"],
            args.scenario_categories,
        )
        scenarios = split_fused_reach_scenarios(
            eligible,
            split=args.scenario_split,
            validation_fraction=args.scenario_validation_fraction,
            test_fraction=args.scenario_test_fraction,
            seed=args.scenario_split_seed,
        )
        category_fractions = None
        if args.recoverable_failure_fraction > 0.0:
            category_fractions = {
                "recoverable_failure": args.recoverable_failure_fraction,
            }
        scenario_sampler = FusedReachScenarioSampler(
            scenarios,
            seed=args.seed,
            shuffle=True,
            category_fractions=category_fractions,
        )
        scenario_step_budget = (
            int(args.scenario_step_budget)
            if args.scenario_step_budget > 0
            else int(scenario_set["recommended_step_budget"])
        )
        print(
            "teacher_guided_scenarios path={} split={} scenarios={} "
            "step_budget={} categories={} sampling_fractions={}".format(
                args.scenario_set,
                args.scenario_split,
                len(scenarios),
                scenario_step_budget,
                scenario_category_counts(scenarios),
                dict(
                    (key, round(value, 3))
                    for key, value in
                    scenario_sampler.sampling_category_fractions().items()
                ),
            )
        )

    balanced_detour = bool(args.balanced_detour_episodes_per_side > 0)
    if balanced_detour and scenario_sampler is not None:
        raise ValueError(
            "--balanced-detour-episodes-per-side cannot be combined with "
            "--scenario-set"
        )
    if balanced_detour:
        print(
            "rollout_contract=balanced_complete_detour_episodes "
            "episodes_per_side={} max_steps={} sampling=equal_steps "
            "advantage_normalization=per_side".format(
                args.balanced_detour_episodes_per_side,
                args.balanced_detour_max_steps,
            )
        )
    else:
        print("rollout_contract=fixed_steps steps={}".format(
            args.rollout_steps
        ))
    print(
        "critic_warmup_steps={} teacher_action_weights={} "
        "reference_coefficient={} reference_action_weights={} "
        "log_std={} frozen={}".format(
            args.critic_warmup_steps,
            args.teacher_action_weights,
            args.reference_coefficient,
            args.reference_action_weights,
            [round(float(value), 4) for value in model.log_std.detach().cpu()],
            args.freeze_log_std,
        )
    )

    client = FusedLowEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )
    buffer_capacity = (
        2
        * args.balanced_detour_episodes_per_side
        * args.balanced_detour_max_steps
        if balanced_detour else args.rollout_steps
    )
    buffer = PPORolloutBuffer(
        capacity=buffer_capacity,
        observation_dim=model.OBS_DIM,
        action_dim=model.ACTION_DIM,
    )
    reward_window = []
    episode_window = collections.deque(maxlen=100)
    balanced_sides = (
        _balanced_detour_sides(
            args.balanced_detour_episodes_per_side,
            update_index,
        )
        if balanced_detour else []
    )
    balanced_cursor = 0
    balanced_episode_counts = collections.Counter()
    balanced_step_counts = collections.Counter()
    balanced_step_sides = []
    current_scenario = _next_scenario(
        scenario_sampler,
        balanced_sides,
        balanced_cursor,
    )
    raw_observation = client.reset(
        scenario=current_scenario,
        max_steps=(
            args.balanced_detour_max_steps
            if balanced_detour else
            scenario_budget_for(current_scenario, scenario_step_budget)
            if current_scenario is not None
            else scenario_step_budget
        ),
    )
    if update_normalizer:
        normalizer.update(raw_observation)
    start_time = time.time()
    rollout_start_total_steps = total_steps

    try:
        while (
                total_steps < args.total_steps
                or (balanced_detour and buffer.size > 0)):
            normalized = normalizer.normalize(raw_observation)
            observation_tensor = torch.from_numpy(normalized).to(device)
            observation_tensor = observation_tensor.unsqueeze(0)
            action_mask = _training_action_mask(observation_tensor, args)
            with torch.no_grad():
                sampled_action, masked_action, logp, value = model.act(
                    observation_tensor,
                    action_mask=action_mask,
                )
            action = sampled_action.squeeze(0).cpu().numpy()
            executed_action = masked_action.squeeze(0).cpu().numpy()
            teacher_action = np.asarray(
                client.last_teacher_action,
                dtype=np.float32,
            )
            teacher_valid = bool(client.last_teacher_available)
            next_observation, reward, done, info = client.step(
                executed_action
            )
            buffer.add(
                observation=normalized,
                action=action,
                teacher_action=teacher_action,
                teacher_valid=teacher_valid,
                log_probability=float(logp.item()),
                value=float(value.item()),
                reward=reward,
                done=done,
            )
            if balanced_detour:
                step_side = float(current_scenario["detour_side"])
                step_side_name = "upper" if step_side > 0.0 else "lower"
                balanced_step_sides.append(step_side)
                balanced_step_counts[step_side_name] += 1
            total_steps += 1
            reward_window.append(float(reward))

            balanced_rollout_complete = False
            if done:
                episode_window.append({
                    "success": bool(info.get("success", False)),
                    "collision": bool(info.get("collision", False)),
                    "timeout": bool(info.get("timeout", False)),
                    "side": float(info.get(
                        "box_detour_side",
                        current_scenario.get("detour_side", 0.0)
                        if current_scenario is not None else 0.0,
                    )),
                })
                if current_scenario is not None:
                    category = str(current_scenario.get("category", "unknown"))
                    scenario_episode_counts[category] += 1
                    scenario_success_counts[category] += int(
                        bool(info.get("success", False))
                    )
                if balanced_detour:
                    side = float(current_scenario["detour_side"])
                    balanced_episode_counts[
                        "upper" if side > 0.0 else "lower"
                    ] += 1
                    balanced_cursor += 1
                    balanced_rollout_complete = bool(
                        balanced_cursor >= len(balanced_sides)
                    )
                    if balanced_rollout_complete:
                        raw_observation = next_observation
                    else:
                        current_scenario = _next_scenario(
                            None,
                            balanced_sides,
                            balanced_cursor,
                        )
                        raw_observation = client.reset(
                            scenario=current_scenario,
                            max_steps=args.balanced_detour_max_steps,
                        )
                else:
                    current_scenario = (
                        scenario_sampler.next() if scenario_sampler else None
                    )
                    raw_observation = client.reset(
                        scenario=current_scenario,
                        max_steps=(
                            scenario_budget_for(
                                current_scenario,
                                scenario_step_budget,
                            )
                            if current_scenario is not None
                            else scenario_step_budget
                        ),
                    )
            else:
                raw_observation = next_observation
            if update_normalizer:
                normalizer.update(raw_observation)

            rollout_ready = (
                balanced_rollout_complete
                if balanced_detour else buffer.full
            )
            if not rollout_ready:
                continue
            last_value = _bootstrap(
                model,
                normalizer,
                raw_observation,
                done,
                device,
            )
            buffer.compute_returns_and_advantages(
                last_value,
                args.gamma,
                args.gae_lambda,
            )
            teacher_coefficient = _teacher_coefficient(
                args.teacher_coefficient,
                args.teacher_decay_steps,
                total_steps,
            )
            metrics = _ppo_update(
                model,
                optimizer,
                buffer,
                device,
                args,
                teacher_coefficient,
                reference_model=reference_model,
                critic_only=bool(
                    rollout_start_total_steps - training_start_steps
                    < args.critic_warmup_steps
                ),
                side_labels=(
                    balanced_step_sides if balanced_detour else None
                ),
            )
            buffer.clear()
            rollout_start_total_steps = total_steps
            update_index += 1
            elapsed = max(time.time() - start_time, 1.0e-6)
            print(
                "update={} steps={} steps_per_second={:.2f} reward={:.4f} "
                "policy_loss={:.5f} value_loss={:.5f} teacher_mse={:.6f} "
                "teacher_coef={:.5f} reference_mse={:.6f} "
                "reference_coef={:.5f} entropy={:.5f} action_abs={:.4f} "
                "actor_frozen={} rollout_steps={} detour_episodes={} "
                "detour_steps={} optimization_side_samples={} "
                "episode_rates={} detour_success={} teacher_action_mse={} "
                "reference_action_mse={} scenario_success={}".format(
                    update_index,
                    total_steps,
                    (total_steps - training_start_steps) / elapsed,
                    float(np.mean(reward_window[-args.reward_window:])),
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["teacher_loss"],
                    teacher_coefficient,
                    metrics["reference_loss"],
                    args.reference_coefficient,
                    metrics["entropy"],
                    metrics["action_abs"],
                    metrics["actor_frozen"],
                    metrics["rollout_steps"],
                    dict(balanced_episode_counts),
                    dict(balanced_step_counts),
                    metrics["optimization_side_samples"],
                    _episode_rates(episode_window),
                    _detour_success_rates(episode_window),
                    metrics["teacher_action_mse"],
                    metrics["reference_action_mse"],
                    _category_success_rates(
                        scenario_episode_counts,
                        scenario_success_counts,
                    ),
                )
            )
            if (
                    args.checkpoint_interval > 0
                    and update_index % args.checkpoint_interval == 0):
                checkpoint_path = _numbered_checkpoint_path(
                    args.output, update_index
                )
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    normalizer,
                    total_steps,
                    update_index,
                    args,
                )
                print("checkpoint saved: {}".format(checkpoint_path))

            if balanced_detour and total_steps < args.total_steps:
                balanced_sides = _balanced_detour_sides(
                    args.balanced_detour_episodes_per_side,
                    update_index,
                )
                balanced_cursor = 0
                balanced_episode_counts = collections.Counter()
                balanced_step_counts = collections.Counter()
                balanced_step_sides = []
                current_scenario = _next_scenario(
                    None,
                    balanced_sides,
                    balanced_cursor,
                )
                raw_observation = client.reset(
                    scenario=current_scenario,
                    max_steps=args.balanced_detour_max_steps,
                )
                if update_normalizer:
                    normalizer.update(raw_observation)

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
        args,
        teacher_coefficient,
        reference_model=None,
        critic_only=False,
        side_labels=None):
    data = buffer.tensors(device)
    data["advantages"] = _normalized_advantages(
        data["advantages"],
        side_labels,
    )
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "teacher_loss": 0.0,
        "reference_loss": 0.0,
        "entropy": 0.0,
        "action_abs": 0.0,
        "count": 0,
    }
    teacher_action_squared_error = np.zeros(model.ACTION_DIM, dtype=np.float64)
    teacher_action_count = np.zeros(model.ACTION_DIM, dtype=np.float64)
    reference_action_squared_error = np.zeros(
        model.ACTION_DIM,
        dtype=np.float64,
    )
    reference_action_count = np.zeros(model.ACTION_DIM, dtype=np.float64)
    optimization_side_samples = collections.Counter()
    side_labels_array = (
        np.asarray(side_labels, dtype=np.float32)
        if side_labels is not None else None
    )
    teacher_action_weights = torch.as_tensor(
        args.teacher_action_weights,
        dtype=torch.float32,
        device=device,
    )
    reference_action_weights = torch.as_tensor(
        args.reference_action_weights,
        dtype=torch.float32,
        device=device,
    )
    for _ in range(args.ppo_epochs):
        for indices_np in _ppo_mini_batch_indices(
                buffer,
                args.batch_size,
                side_labels):
            if side_labels_array is not None:
                selected_sides = side_labels_array[indices_np]
                optimization_side_samples["upper"] += int(
                    np.sum(selected_sides > 0.0)
                )
                optimization_side_samples["lower"] += int(
                    np.sum(selected_sides < 0.0)
                )
            indices = torch.from_numpy(indices_np.astype(np.int64)).to(device)
            obs = data["observations"].index_select(0, indices)
            actions = data["actions"].index_select(0, indices)
            old_logp = data["old_log_probabilities"].index_select(0, indices)
            old_values = data["old_values"].index_select(0, indices)
            returns = data["returns"].index_select(0, indices)
            batch_advantages = data["advantages"].index_select(0, indices)
            teacher_actions = data["teacher_actions"].index_select(0, indices)
            teacher_valid = data["teacher_valid"].index_select(0, indices)

            action_mask = _training_action_mask(obs, args)
            logp, entropy, values = model.evaluate_actions(
                obs,
                actions,
                action_mask=action_mask,
            )
            ratio = torch.exp(logp - old_logp)
            policy_loss = -torch.min(
                ratio * batch_advantages,
                torch.clamp(
                    ratio,
                    1.0 - args.clip_ratio,
                    1.0 + args.clip_ratio,
                ) * batch_advantages,
            ).mean()
            clipped_values = old_values + torch.clamp(
                values - old_values,
                -args.clip_ratio,
                args.clip_ratio,
            )
            value_loss = 0.5 * torch.max(
                (values - returns).pow(2),
                (clipped_values - returns).pow(2),
            ).mean()

            actor_features = model.actor_backbone(obs)
            mean_actions = torch.tanh(
                model.actor_mean(actor_features)
            )
            per_sample = _masked_weighted_action_mse(
                mean_actions,
                teacher_actions,
                action_mask,
                teacher_action_weights,
            )
            valid_count = torch.clamp(teacher_valid.sum(), min=1.0)
            teacher_loss = (per_sample * teacher_valid).sum() / valid_count
            teacher_loss = teacher_loss * (teacher_valid.sum() > 0).float()
            teacher_component_mask = (
                action_mask * teacher_valid.unsqueeze(1)
            )
            teacher_action_squared_error += (
                (mean_actions - teacher_actions).pow(2)
                * teacher_component_mask
            ).sum(dim=0).detach().cpu().numpy()
            teacher_action_count += teacher_component_mask.sum(
                dim=0
            ).detach().cpu().numpy()

            reference_loss = torch.zeros((), device=device)
            if reference_model is not None:
                with torch.no_grad():
                    reference_features = reference_model.actor_backbone(obs)
                    reference_actions = torch.tanh(
                        reference_model.actor_mean(reference_features)
                    )
                reference_loss = _masked_weighted_action_mse(
                    mean_actions,
                    reference_actions,
                    action_mask,
                    reference_action_weights,
                ).mean()
                reference_action_squared_error += (
                    (mean_actions - reference_actions).pow(2) * action_mask
                ).sum(dim=0).detach().cpu().numpy()
                reference_action_count += action_mask.sum(
                    dim=0
                ).detach().cpu().numpy()

            if critic_only:
                total_loss = args.value_coefficient * value_loss
            else:
                total_loss = (
                    policy_loss
                    + args.value_coefficient * value_loss
                    - args.entropy_coefficient * entropy.mean()
                    + teacher_coefficient * teacher_loss
                    + args.reference_coefficient * reference_loss
                )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.maximum_gradient_norm
            )
            optimizer.step()

            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["teacher_loss"] += float(teacher_loss.item())
            totals["reference_loss"] += float(reference_loss.item())
            totals["entropy"] += float(entropy.mean().item())
            active_action_count = torch.clamp(action_mask.sum(), min=1.0)
            totals["action_abs"] += float(
                (
                    mean_actions.abs() * action_mask
                ).sum().div(active_action_count).item()
            )
            totals["count"] += 1
    divisor = float(max(totals.pop("count"), 1))
    metrics = dict((key, value / divisor) for key, value in totals.items())
    metrics["teacher_action_mse"] = _component_mse(
        teacher_action_squared_error,
        teacher_action_count,
    )
    metrics["reference_action_mse"] = _component_mse(
        reference_action_squared_error,
        reference_action_count,
    )
    metrics["actor_frozen"] = bool(critic_only)
    metrics["rollout_steps"] = int(buffer.size)
    metrics["optimization_side_samples"] = dict(
        optimization_side_samples
    )
    return metrics


def _ppo_mini_batch_indices(buffer, batch_size, side_labels=None):
    if side_labels is None:
        for indices in buffer.mini_batch_indices(batch_size):
            yield indices
        return

    labels = np.asarray(side_labels, dtype=np.float32)
    if labels.shape != (buffer.size,):
        raise ValueError("side labels must match the rollout size")
    upper = np.flatnonzero(labels > 0.0)
    lower = np.flatnonzero(labels < 0.0)
    if upper.size <= 0 or lower.size <= 0:
        raise ValueError("balanced rollout must contain upper and lower steps")
    half_batch = int(batch_size) // 2
    if half_batch <= 0:
        raise ValueError("balanced PPO batch size must be at least 2")
    per_side_total = int(
        np.ceil(float(max(upper.size, lower.size)) / half_batch)
    ) * half_batch
    upper_samples = _repeat_shuffled_indices(upper, per_side_total)
    lower_samples = _repeat_shuffled_indices(lower, per_side_total)
    for start in range(0, per_side_total, half_batch):
        indices = np.concatenate((
            upper_samples[start:start + half_batch],
            lower_samples[start:start + half_batch],
        ))
        np.random.shuffle(indices)
        yield indices


def _normalized_advantages(advantages, side_labels=None):
    if side_labels is None:
        return (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1.0e-8)
    labels = torch.as_tensor(
        side_labels,
        dtype=torch.float32,
        device=advantages.device,
    )
    if labels.shape != advantages.shape:
        raise ValueError("side labels must match the advantage shape")
    normalized = torch.zeros_like(advantages)
    for selected in (labels > 0.0, labels < 0.0):
        if not bool(selected.any()):
            raise ValueError("balanced advantages require both detour sides")
        values = advantages[selected]
        normalized[selected] = (
            values - values.mean()
        ) / (values.std(unbiased=False) + 1.0e-8)
    return normalized


def _repeat_shuffled_indices(indices, target_count):
    samples = []
    remaining = int(target_count)
    while remaining > 0:
        shuffled = np.random.permutation(indices)
        take = min(remaining, shuffled.size)
        samples.append(shuffled[:take])
        remaining -= take
    return np.concatenate(samples).astype(np.int64, copy=False)


def _component_mse(squared_error, count):
    return [
        round(float(squared_error[index] / count[index]), 8)
        if count[index] > 0.0 else None
        for index in range(len(squared_error))
    ]


def _masked_weighted_action_mse(
        predicted_actions,
        target_actions,
        action_mask,
        action_weights):
    """Return per-sample MSE over actions that can actually be executed."""
    weights = action_mask * action_weights.reshape(1, -1)
    denominator = torch.clamp(weights.sum(dim=1), min=1.0)
    return (
        (predicted_actions - target_actions).pow(2) * weights
    ).sum(dim=1) / denominator


def _training_action_mask(observations, args):
    action_mask = torch.clamp(observations[..., 58:66], 0.0, 1.0)
    if not args.fixed_box_action_gate:
        return action_mask
    return _fixed_box_gate_action_mask(action_mask)


def _fixed_box_gate_action_mask(action_mask):
    """Match PPO optimization to the base-before-arm execution contract.

    Existing checkpoints keep the three-stage observation encoding where the
    COORDINATED phase advertises all eight action components.  The environment
    now treats that phase as base-only: the arm remains fixed until terminal
    base alignment is latched.  Keep the observation itself unchanged for
    checkpoint compatibility, but exclude those unexecuted arm components
    from sampling, log probabilities, entropy, and teacher supervision.
    """
    fixed_mask = action_mask.clone()
    base_active = action_mask[..., 0:2].sum(dim=-1) > 0.5
    arm_active = action_mask[..., 2:8].sum(dim=-1) > 0.5
    pre_alignment = base_active & arm_active
    fixed_mask[..., 2:8] = torch.where(
        pre_alignment.unsqueeze(-1),
        torch.zeros_like(fixed_mask[..., 2:8]),
        fixed_mask[..., 2:8],
    )
    return fixed_mask


def _bootstrap(model, normalizer, observation, done, device):
    if done:
        return 0.0
    tensor = torch.from_numpy(
        normalizer.normalize(observation)
    ).to(device).unsqueeze(0)
    with torch.no_grad():
        return float(model.get_value(tensor).item())


def _teacher_coefficient(initial, decay_steps, total_steps):
    initial = float(initial)
    decay_steps = int(decay_steps)
    if initial <= 0.0 or decay_steps <= 0:
        return max(initial, 0.0)
    return initial * (1.0 - min(float(total_steps) / decay_steps, 1.0))


def _category_success_rates(episode_counts, success_counts):
    return dict(
        (
            category,
            round(
                float(success_counts[category])
                / float(max(episode_counts[category], 1)),
                3,
            ),
        )
        for category in sorted(episode_counts)
    )


def _episode_rates(episodes):
    count = len(episodes)
    if count <= 0:
        return {
            "episodes": 0,
            "success": None,
            "collision": None,
            "timeout": None,
        }
    return {
        "episodes": int(count),
        "success": round(float(np.mean([
            episode["success"] for episode in episodes
        ])), 3),
        "collision": round(float(np.mean([
            episode["collision"] for episode in episodes
        ])), 3),
        "timeout": round(float(np.mean([
            episode["timeout"] for episode in episodes
        ])), 3),
    }


def _detour_success_rates(episodes):
    result = {}
    for name, predicate in (
            ("upper", lambda side: side > 0.0),
            ("lower", lambda side: side < 0.0)):
        matching = [
            episode for episode in episodes
            if predicate(float(episode.get("side", 0.0)))
        ]
        result[name] = {
            "episodes": len(matching),
            "success": (
                round(float(np.mean([
                    episode["success"] for episode in matching
                ])), 3)
                if matching else None
            ),
        }
    return result


def _balanced_detour_sides(episodes_per_side, update_index):
    """Build an equal upper/lower episode order and flip its first side."""
    episodes_per_side = int(episodes_per_side)
    if episodes_per_side <= 0:
        return []
    first_side = 1.0 if int(update_index) % 2 == 0 else -1.0
    sides = []
    for _ in range(episodes_per_side):
        sides.extend((first_side, -first_side))
    return sides


def _next_scenario(scenario_sampler, balanced_sides, balanced_cursor):
    if scenario_sampler is not None:
        return scenario_sampler.next()
    if not balanced_sides:
        return None
    side = float(balanced_sides[int(balanced_cursor)])
    side_name = "upper" if side > 0.0 else "lower"
    return {
        "scenario_id": "balanced_{}_{}".format(
            side_name,
            int(balanced_cursor),
        ),
        "category": "balanced_box_detour",
        "detour_side": side,
    }


def _load_reference_model(model, args, device):
    if args.reference_coefficient <= 0.0:
        return None
    checkpoint_path = args.reference_checkpoint or args.bc_checkpoint
    if not checkpoint_path:
        raise ValueError(
            "--reference-coefficient requires --reference-checkpoint or "
            "--bc-checkpoint"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if int(checkpoint.get("observation_dim", -1)) != model.OBS_DIM:
        raise ValueError("reference checkpoint observation_dim must be 66")
    if int(checkpoint.get("action_dim", -1)) != model.ACTION_DIM:
        raise ValueError("reference checkpoint action_dim must be 8")
    reference_model = copy.deepcopy(model)
    reference_model.load_compatible_state_dict(
        checkpoint["model"],
        actor_only=True,
    )
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    print(
        "reference_actor checkpoint={} coefficient={} frozen=True".format(
            checkpoint_path,
            args.reference_coefficient,
        )
    )
    return reference_model


def _numbered_checkpoint_path(path, update):
    root, extension = os.path.splitext(path)
    return "{}.update_{:04d}{}".format(root, int(update), extension)


def _save_checkpoint(path, model, optimizer, normalizer, total_steps, update, args):
    path = os.path.abspath(os.path.expanduser(path))
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    torch.save({
        "format_version": 1,
        "policy_type": "fused_low_teacher_ppo",
        "teacher_type": "coordinated_rule_teacher",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "total_steps": int(total_steps),
        "update_index": int(update),
        "arguments": vars(args),
        "observation_dim": model.OBS_DIM,
        "action_dim": model.ACTION_DIM,
    }, path)


def _set_random_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5562)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument(
        "--output",
        default="/tmp/mobile_arm_rl_training/fused_low_teacher_ppo.pt",
    )
    parser.add_argument("--resume")
    parser.add_argument("--bc-checkpoint")
    parser.add_argument(
        "--reference-checkpoint",
        help=(
            "Frozen actor used as a continual trust-region anchor. If it is "
            "omitted, --bc-checkpoint is used."
        ),
    )
    parser.add_argument(
        "--fixed-box-action-gate",
        action="store_true",
        help=(
            "Optimize only base actions before terminal base alignment and "
            "only arm actions afterward, matching the box-detour execution "
            "gate."
        ),
    )
    parser.add_argument(
        "--update-observation-normalizer",
        action="store_true",
        help=(
            "Update normalization statistics online. By default a BC/resume "
            "checkpoint keeps its observation normalizer frozen."
        ),
    )
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument(
        "--balanced-detour-episodes-per-side",
        type=int,
        default=0,
        help=(
            "Collect this many complete upper and lower box-detour episodes "
            "before every PPO update. Zero keeps fixed-step rollouts."
        ),
    )
    parser.add_argument(
        "--balanced-detour-max-steps",
        type=int,
        default=1500,
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=5e-4)
    parser.add_argument("--teacher-coefficient", type=float, default=1.0)
    parser.add_argument("--teacher-decay-steps", type=int, default=50000)
    parser.add_argument(
        "--teacher-action-weights",
        type=float,
        nargs=8,
        default=[1.0] * 8,
        metavar=("V", "OMEGA", "A2", "A3", "A4", "A5", "A6", "A7"),
    )
    parser.add_argument(
        "--reference-coefficient",
        type=float,
        default=0.0,
        help="Strength of MSE regularization to the frozen reference actor.",
    )
    parser.add_argument(
        "--reference-action-weights",
        type=float,
        nargs=8,
        default=[1.0] * 8,
        metavar=("V", "OMEGA", "A2", "A3", "A4", "A5", "A6", "A7"),
    )
    parser.add_argument(
        "--critic-warmup-steps",
        type=int,
        default=0,
        help=(
            "Update only the freshly initialized critic for this many new "
            "environment steps before allowing actor updates."
        ),
    )
    parser.add_argument("--maximum-gradient-norm", type=float, default=0.5)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[128, 128]
    )
    parser.add_argument("--initial-log-std", type=float, default=-2.5)
    parser.add_argument(
        "--freeze-log-std",
        action="store_true",
        help=(
            "Keep the newly selected exploration variance fixed instead of "
            "letting PPO inherit or rapidly alter it."
        ),
    )
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--reward-window", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--scenario-set")
    parser.add_argument("--scenario-split", default="train")
    parser.add_argument("--scenario-step-budget", type=int, default=0)
    parser.add_argument(
        "--scenario-categories", nargs="+",
        default=["simple_success", "hard_success", "recoverable_failure"],
    )
    parser.add_argument("--recoverable-failure-fraction", type=float, default=0.0)
    parser.add_argument("--scenario-validation-fraction", type=float, default=0.1)
    parser.add_argument("--scenario-test-fraction", type=float, default=0.1)
    parser.add_argument("--scenario-split-seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    _validate_arguments(args, parser)
    return args


def _validate_arguments(args, parser):
    if args.resume and args.bc_checkpoint:
        parser.error("--resume and --bc-checkpoint are mutually exclusive")
    if args.total_steps <= 0:
        parser.error("--total-steps must be positive")
    if args.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive")
    if args.balanced_detour_episodes_per_side < 0:
        parser.error("--balanced-detour-episodes-per-side cannot be negative")
    if args.balanced_detour_max_steps <= 0:
        parser.error("--balanced-detour-max-steps must be positive")
    if args.balanced_detour_episodes_per_side > 0 and args.scenario_set:
        parser.error(
            "--balanced-detour-episodes-per-side cannot be combined with "
            "--scenario-set"
        )
    if (
            args.balanced_detour_episodes_per_side > 0
            and (args.batch_size < 2 or args.batch_size % 2 != 0)):
        parser.error("balanced detour training requires an even batch size")
    if args.critic_warmup_steps < 0:
        parser.error("--critic-warmup-steps cannot be negative")
    if args.reference_coefficient < 0.0:
        parser.error("--reference-coefficient cannot be negative")
    if (
            args.reference_coefficient > 0.0
            and not args.reference_checkpoint
            and not args.bc_checkpoint):
        parser.error(
            "--reference-coefficient requires --reference-checkpoint or "
            "--bc-checkpoint"
        )
    for name, weights in (
            ("teacher", args.teacher_action_weights),
            ("reference", args.reference_action_weights)):
        if any(weight < 0.0 for weight in weights):
            parser.error("{} action weights cannot be negative".format(name))
        if sum(weights) <= 0.0:
            parser.error(
                "{} action weights must contain a positive value".format(name)
            )


if __name__ == "__main__":
    main()
