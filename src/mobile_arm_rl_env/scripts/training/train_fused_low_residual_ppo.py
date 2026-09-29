#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PPO trainer for bounded residuals around the fused rule teacher.

At initialization the actor mean is exactly zero.  A deterministic policy
therefore produces the existing teacher action through the residual adapter;
PPO only learns a bounded five-dimensional correction.
"""

import argparse
import collections
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

from training.fused_low_residual_env_client import (  # noqa: E402
    FusedLowResidualEnvironmentClient,
)
from training.fused_residual_actor_critic import (  # noqa: E402
    FusedResidualActorCritic,
)
from training.fused_reach_scenarios import (  # noqa: E402
    FusedReachScenarioSampler,
    filter_scenario_categories,
    load_fused_reach_scenarios,
    scenario_category_counts,
    scenario_step_budget,
    split_fused_reach_scenarios,
)
from training.hrl4in_low_actor_critic import (  # noqa: E402
    RunningObservationNormalizer,
)
from training.ppo_rollout import PPORolloutBuffer  # noqa: E402
from hrl.high_level_command import SubgoalType  # noqa: E402


TYPED_SUBGOAL_CONTEXT_CONTRACT = "three_class_in_subgoal_mask_v1"
REQUIRED_SUBGOAL_TYPES = (
    SubgoalType.DIRECT,
    SubgoalType.DETOUR,
    SubgoalType.TERMINAL,
)


def main():
    args = _parse_arguments()
    _seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = FusedResidualActorCritic(
        hidden_sizes=args.hidden_sizes,
        initial_log_std=args.initial_log_std,
    ).to(device)
    if args.freeze_log_std:
        model.log_std.requires_grad_(False)
    normalizer = RunningObservationNormalizer(
        observation_dim=model.OBS_DIM,
        normalized_dim=52,
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=args.learning_rate,
    )
    total_steps = 0
    update_index = 0
    resume_subgoal_context_contract = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        normalizer.load_state_dict(checkpoint["normalizer"])
        total_steps = int(checkpoint.get("total_steps", 0))
        update_index = int(checkpoint.get("update_index", 0))
        resume_subgoal_context_contract = str(checkpoint.get(
            "subgoal_context_contract",
            "legacy_all_ones_subgoal_mask",
        ))
        print("resumed checkpoint={} total_steps={} update={}".format(
            args.resume, total_steps, update_index
        ))

    client = FusedLowResidualEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )
    environment_subgoal_context_contract = str(client.metadata.get(
        "subgoal_context_contract",
        "legacy_all_ones_subgoal_mask",
    ))
    if (
            resume_subgoal_context_contract is not None
            and resume_subgoal_context_contract
            != environment_subgoal_context_contract):
        client.close()
        raise ValueError(
            "resume/environment subgoal context mismatch: {} != {}".format(
                resume_subgoal_context_contract,
                environment_subgoal_context_contract,
            )
        )
    print(
        "subgoal_context_contract={} subgoal_types={}".format(
            environment_subgoal_context_contract,
            client.metadata.get("subgoal_types", []),
        )
    )
    if (
            args.balanced_subgoal_types
            and environment_subgoal_context_contract
            != TYPED_SUBGOAL_CONTEXT_CONTRACT):
        client.close()
        raise ValueError(
            "balanced subgoal training requires {}, got {}".format(
                TYPED_SUBGOAL_CONTEXT_CONTRACT,
                environment_subgoal_context_contract,
            )
        )
    buffer_capacity = (
        args.maximum_rollout_steps
        if args.balanced_subgoal_types else args.rollout_steps
    )
    buffer = PPORolloutBuffer(
        capacity=buffer_capacity,
        observation_dim=model.OBS_DIM,
        action_dim=model.ACTION_DIM,
    )
    print(
        "rollout_contract={} target_steps={} capacity={} "
        "minimum_type_steps={}".format(
            "balanced_typed_complete_options"
            if args.balanced_subgoal_types else "fixed_steps",
            args.rollout_steps,
            buffer_capacity,
            args.minimum_subgoal_type_steps
            if args.balanced_subgoal_types else 0,
        )
    )
    scenario_sampler = None
    scenario_step_budget = None
    scenario_episode_counts = collections.Counter()
    scenario_success_counts = collections.Counter()
    if args.scenario_set:
        scenario_set = load_fused_reach_scenarios(args.scenario_set)
        eligible_scenarios = filter_scenario_categories(
            scenario_set["scenarios"],
            args.scenario_categories,
        )
        scenarios = split_fused_reach_scenarios(
            eligible_scenarios,
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
            "fused_curriculum_loaded path={} split={} scenarios={} "
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
    raw_observation, current_scenario = _reset_environment(
        client,
        scenario_sampler,
        scenario_step_budget,
    )
    normalizer.update(raw_observation)
    done = True
    last_transition_done = True
    start = time.time()
    reward_window = []
    subgoal_metric_window = []
    subgoal_step_counts = collections.Counter()
    subgoal_option_counts = collections.Counter()
    subgoal_completion_counts = collections.Counter()
    subgoal_timeout_counts = collections.Counter()
    subgoal_metric_sums = collections.Counter()
    rollout_subgoal_types = []
    rollout_subgoal_type_steps = collections.Counter()
    try:
        while total_steps < args.total_steps or buffer.size > 0:
            normalized = normalizer.normalize(raw_observation)
            observation_tensor = torch.from_numpy(normalized).to(device).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _, logp_tensor, value_tensor = model.act(
                    observation_tensor,
                    deterministic=args.deterministic_rollout,
                )
            action = action_tensor.squeeze(0).cpu().numpy()
            next_raw, reward, done, info = client.step(action)
            subgoal_option_done = bool(
                info.get("subgoal_option_done", False)
            )
            last_transition_done = bool(done or subgoal_option_done)
            buffer.add(
                observation=normalized,
                action=action,
                teacher_action=np.zeros(model.ACTION_DIM, dtype=np.float32),
                teacher_valid=True,
                log_probability=float(logp_tensor.item()),
                value=float(value_tensor.item()),
                reward=reward,
                done=last_transition_done,
            )
            total_steps += 1
            reward_window.append(reward)
            subgoal_type_name = str(
                info.get("subgoal_type_name", "UNKNOWN")
            )
            subgoal_type = int(info.get("subgoal_type", -1))
            if (
                    args.balanced_subgoal_types
                    and subgoal_type not in REQUIRED_SUBGOAL_TYPES):
                raise RuntimeError(
                    "typed rollout received invalid subgoal type: {}".format(
                        subgoal_type
                    )
                )
            rollout_subgoal_types.append(subgoal_type)
            rollout_subgoal_type_steps[subgoal_type_name] += 1
            subgoal_step_counts[subgoal_type_name] += 1
            if "subgoal_metric" in info:
                subgoal_metric = float(info["subgoal_metric"])
                subgoal_metric_window.append(
                    subgoal_metric
                )
                subgoal_metric_sums[subgoal_type_name] += subgoal_metric
            if subgoal_option_done:
                subgoal_option_counts[subgoal_type_name] += 1
                subgoal_completion_counts[subgoal_type_name] += int(
                    bool(info.get("subgoal_completed", False))
                )
                subgoal_timeout_counts[subgoal_type_name] += int(
                    bool(info.get("subgoal_timed_out", False))
                )
            if done:
                if current_scenario is not None:
                    category = str(
                        current_scenario.get("category", "unknown")
                    )
                    scenario_episode_counts[category] += 1
                    scenario_success_counts[category] += int(
                        bool(info.get("success", False))
                    )
                raw_observation, current_scenario = _reset_environment(
                    client,
                    scenario_sampler,
                    scenario_step_budget,
                )
            else:
                raw_observation = next_raw
            normalizer.update(raw_observation)
            rollout_ready = (
                _balanced_rollout_ready(
                    buffer,
                    rollout_subgoal_types,
                    args.rollout_steps,
                    args.minimum_subgoal_type_steps,
                    last_transition_done,
                )
                if args.balanced_subgoal_types else buffer.full
            )
            if buffer.full and not rollout_ready:
                raise RuntimeError(
                    "maximum rollout steps reached before all subgoal types "
                    "were represented: {}".format(
                        dict(rollout_subgoal_type_steps)
                    )
                )
            if not rollout_ready:
                continue
            last_value = _bootstrap(
                model,
                normalizer,
                raw_observation,
                last_transition_done,
                device,
            )
            buffer.compute_returns_and_advantages(
                last_value, args.gamma, args.gae_lambda
            )
            metrics = _ppo_update(
                model,
                optimizer,
                buffer,
                device,
                args,
                subgoal_type_labels=(
                    rollout_subgoal_types
                    if args.balanced_subgoal_types else None
                ),
                critic_only=bool(
                    total_steps <= args.critic_warmup_steps
                ),
            )
            buffer.clear()
            rollout_subgoal_types = []
            completed_rollout_type_steps = dict(
                rollout_subgoal_type_steps
            )
            rollout_subgoal_type_steps.clear()
            update_index += 1
            print(
                "update={} steps={} reward={:.4f} action_abs={:.4f} "
                "subgoal_error={:.5f} residual_mse={:.6f} "
                "policy_loss={:.5f} value_loss={:.5f} "
                "elapsed_s={:.1f} scenario_success={} "
                "subgoal_options={} rollout_type_steps={} "
                "optimization_type_samples={}".format(
                    update_index,
                    total_steps,
                    float(np.mean(reward_window[-args.reward_window:])),
                    float(np.mean(np.abs(action))),
                    float(np.mean(
                        subgoal_metric_window[-args.reward_window:]
                    )) if subgoal_metric_window else 0.0,
                    metrics["teacher_loss"],
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    time.time() - start,
                    _category_success_rates(
                        scenario_episode_counts,
                        scenario_success_counts,
                    ),
                    _subgoal_option_rates(
                        subgoal_step_counts,
                        subgoal_option_counts,
                        subgoal_completion_counts,
                        subgoal_timeout_counts,
                        subgoal_metric_sums,
                    ),
                    completed_rollout_type_steps,
                    metrics["optimization_subgoal_samples"],
                )
            )
            if metrics["actor_frozen"]:
                print(
                    "critic_warmup actor_frozen=True steps={}/{}".format(
                        total_steps,
                        args.critic_warmup_steps,
                    )
                )
            if args.checkpoint_interval > 0 and update_index % args.checkpoint_interval == 0:
                checkpoint_path = _numbered_checkpoint_path(
                    args.output, update_index
                )
                _save(
                    checkpoint_path,
                    model,
                    optimizer,
                    normalizer,
                    total_steps,
                    update_index,
                    args,
                    scenario_episode_counts,
                    scenario_success_counts,
                    client.metadata,
                    subgoal_step_counts,
                    subgoal_option_counts,
                    subgoal_completion_counts,
                    subgoal_timeout_counts,
                )
                _save(
                    args.output,
                    model,
                    optimizer,
                    normalizer,
                    total_steps,
                    update_index,
                    args,
                    scenario_episode_counts,
                    scenario_success_counts,
                    client.metadata,
                    subgoal_step_counts,
                    subgoal_option_counts,
                    subgoal_completion_counts,
                    subgoal_timeout_counts,
                )
                print("checkpoint saved: {}".format(checkpoint_path))
        _save(
            args.output,
            model,
            optimizer,
            normalizer,
            total_steps,
            update_index,
            args,
            scenario_episode_counts,
            scenario_success_counts,
            client.metadata,
            subgoal_step_counts,
            subgoal_option_counts,
            subgoal_completion_counts,
            subgoal_timeout_counts,
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
        subgoal_type_labels=None,
        critic_only=False):
    data = buffer.tensors(device)
    data["advantages"] = _normalized_advantages(
        data["advantages"],
        subgoal_type_labels,
    )
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "teacher_loss": 0.0, "count": 0}
    optimization_subgoal_samples = collections.Counter()
    labels_array = (
        np.asarray(subgoal_type_labels, dtype=np.int64)
        if subgoal_type_labels is not None else None
    )
    for _ in range(args.ppo_epochs):
        for indices_np in _ppo_mini_batch_indices(
                buffer,
                args.batch_size,
                subgoal_type_labels):
            if labels_array is not None:
                selected_types = labels_array[indices_np]
                for subgoal_type in REQUIRED_SUBGOAL_TYPES:
                    optimization_subgoal_samples[
                        SubgoalType.name(subgoal_type)
                    ] += int(np.sum(selected_types == subgoal_type))
            indices = torch.from_numpy(indices_np.astype(np.int64)).to(device)
            obs = data["observations"].index_select(0, indices)
            actions = data["actions"].index_select(0, indices)
            old_logp = data["old_log_probabilities"].index_select(0, indices)
            old_values = data["old_values"].index_select(0, indices)
            returns = data["returns"].index_select(0, indices)
            advantages = data["advantages"].index_select(0, indices)
            logp, entropy, values = model.evaluate_actions(obs, actions)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            clipped_values = old_values + torch.clamp(values - old_values, -args.clip_ratio, args.clip_ratio)
            value_loss = 0.5 * torch.max((values - returns).pow(2), (clipped_values - returns).pow(2)).mean()
            mean_action = torch.tanh(model.actor_mean(model.actor_backbone(obs)))
            teacher_loss = mean_action.pow(2).mean()
            if critic_only:
                total = args.value_coefficient * value_loss
            else:
                total = (
                    policy_loss
                    + args.value_coefficient * value_loss
                    - args.entropy_coefficient * entropy.mean()
                    + args.teacher_coefficient * teacher_loss
                )
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.maximum_gradient_norm)
            optimizer.step()
            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["entropy"] += float(entropy.mean().item())
            totals["teacher_loss"] += float(teacher_loss.item())
            totals["count"] += 1
    divisor = float(max(totals.pop("count"), 1))
    metrics = dict((key, value / divisor) for key, value in totals.items())
    metrics["optimization_subgoal_samples"] = dict(
        optimization_subgoal_samples
    )
    metrics["actor_frozen"] = bool(critic_only)
    return metrics


def _balanced_rollout_ready(
        buffer,
        subgoal_type_labels,
        target_steps,
        minimum_type_steps,
        option_done):
    """Require enough on-policy data from all upper-level option types."""
    if buffer.size < int(target_steps) or not bool(option_done):
        return False
    labels = np.asarray(subgoal_type_labels, dtype=np.int64)
    if labels.shape != (buffer.size,):
        raise ValueError("subgoal type labels must match rollout size")
    return all(
        int(np.sum(labels == subgoal_type)) >= int(minimum_type_steps)
        for subgoal_type in REQUIRED_SUBGOAL_TYPES
    )


def _ppo_mini_batch_indices(
        buffer,
        batch_size,
        subgoal_type_labels=None):
    if subgoal_type_labels is None:
        for indices in buffer.mini_batch_indices(batch_size):
            yield indices
        return

    labels = np.asarray(subgoal_type_labels, dtype=np.int64)
    if labels.shape != (buffer.size,):
        raise ValueError("subgoal type labels must match rollout size")
    per_type_batch = int(batch_size) // len(REQUIRED_SUBGOAL_TYPES)
    if per_type_batch <= 0:
        raise ValueError("balanced PPO batch size must be at least 3")
    type_indices = {}
    for subgoal_type in REQUIRED_SUBGOAL_TYPES:
        selected = np.flatnonzero(labels == subgoal_type)
        if selected.size <= 0:
            raise ValueError(
                "balanced rollout has no {} samples".format(
                    SubgoalType.name(subgoal_type)
                )
            )
        type_indices[subgoal_type] = selected
    per_type_total = int(np.ceil(
        float(max(len(value) for value in type_indices.values()))
        / float(per_type_batch)
    )) * per_type_batch
    repeated = dict(
        (
            subgoal_type,
            _repeat_shuffled_indices(indices, per_type_total),
        )
        for subgoal_type, indices in type_indices.items()
    )
    for start in range(0, per_type_total, per_type_batch):
        batch = np.concatenate(tuple(
            repeated[subgoal_type][start:start + per_type_batch]
            for subgoal_type in REQUIRED_SUBGOAL_TYPES
        ))
        np.random.shuffle(batch)
        yield batch


def _normalized_advantages(advantages, subgoal_type_labels=None):
    if subgoal_type_labels is None:
        return (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1.0e-8)
    labels = torch.as_tensor(
        subgoal_type_labels,
        dtype=torch.int64,
        device=advantages.device,
    )
    if labels.shape != advantages.shape:
        raise ValueError("subgoal type labels must match advantages")
    normalized = torch.zeros_like(advantages)
    for subgoal_type in REQUIRED_SUBGOAL_TYPES:
        selected = labels == int(subgoal_type)
        if not bool(selected.any()):
            raise ValueError(
                "advantages have no {} samples".format(
                    SubgoalType.name(subgoal_type)
                )
            )
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


def _bootstrap(model, normalizer, observation, done, device):
    if done:
        return 0.0
    tensor = torch.from_numpy(normalizer.normalize(observation)).to(device).unsqueeze(0)
    with torch.no_grad():
        return float(model.get_value(tensor).item())


def _reset_environment(client, scenario_sampler, step_budget):
    if scenario_sampler is None:
        observation = client.reset()
        # The subgoal residual server alternates upper/lower rule routes on
        # its own.  Preserve its reset metadata so PPO logs can report per-
        # scenario success instead of silently treating every episode as
        # unlabelled.
        metadata = dict(client.last_reset_info)
        if "category" not in metadata:
            metadata["category"] = metadata.get(
                "scenario_category", "unknown"
            )
        return observation, metadata
    scenario = scenario_sampler.next()
    current_step_budget = scenario_step_budget(scenario, step_budget)
    observation = client.reset(
        scenario=scenario,
        max_steps=current_step_budget,
    )
    return observation, scenario


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


def _subgoal_option_rates(
        step_counts,
        option_counts,
        completion_counts,
        timeout_counts,
        metric_sums):
    """Summarize each typed option independently of full-episode success."""
    return dict(
        (
            name,
            {
                "steps": int(step_counts[name]),
                "options": int(option_counts[name]),
                "completion": round(
                    float(completion_counts[name])
                    / float(max(option_counts[name], 1)),
                    3,
                ),
                "timeout": round(
                    float(timeout_counts[name])
                    / float(max(option_counts[name], 1)),
                    3,
                ),
                "mean_error": round(
                    float(metric_sums[name])
                    / float(max(step_counts[name], 1)),
                    5,
                ),
            },
        )
        for name in sorted(step_counts)
    )


def _numbered_checkpoint_path(path, update):
    expanded = os.path.abspath(os.path.expanduser(path))
    root, extension = os.path.splitext(expanded)
    if not extension:
        extension = ".pt"
    return "{}.update_{:04d}{}".format(root, int(update), extension)


def _save(
        path,
        model,
        optimizer,
        normalizer,
        steps,
        update,
        args,
        scenario_episode_counts=None,
        scenario_success_counts=None,
        environment_metadata=None,
        subgoal_step_counts=None,
        subgoal_option_counts=None,
        subgoal_completion_counts=None,
        subgoal_timeout_counts=None):
    directory = os.path.dirname(os.path.abspath(os.path.expanduser(path)))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    torch.save({
        "format_version": 1,
        "policy_type": FusedResidualActorCritic.POLICY_TYPE,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "total_steps": int(steps),
        "update_index": int(update),
        "observation_dim": model.OBS_DIM,
        "action_dim": model.ACTION_DIM,
        "nominal_action_dim": 8,
        "subgoal_context_contract": str(
            (environment_metadata or {}).get(
                "subgoal_context_contract",
                "legacy_all_ones_subgoal_mask",
            )
        ),
        "subgoal_types": list(
            (environment_metadata or {}).get("subgoal_types", [])
        ),
        "arguments": vars(args),
        "training_scenario_episodes": dict(
            scenario_episode_counts or {}
        ),
        "training_scenario_successes": dict(
            scenario_success_counts or {}
        ),
        "training_subgoal_steps": dict(subgoal_step_counts or {}),
        "training_subgoal_options": dict(subgoal_option_counts or {}),
        "training_subgoal_completions": dict(
            subgoal_completion_counts or {}
        ),
        "training_subgoal_timeouts": dict(
            subgoal_timeout_counts or {}
        ),
    }, path)


def _seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5563)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--scenario-set")
    parser.add_argument(
        "--scenario-split",
        choices=["all", "train", "validation", "test"],
        default="train",
    )
    parser.add_argument(
        "--scenario-validation-fraction", type=float, default=0.10
    )
    parser.add_argument(
        "--scenario-test-fraction", type=float, default=0.20
    )
    parser.add_argument("--scenario-split-seed", type=int, default=123)
    parser.add_argument("--scenario-step-budget", type=int, default=0)
    parser.add_argument(
        "--scenario-categories",
        nargs="+",
        help="optional category allow-list applied before train/validation/test split",
    )
    parser.add_argument(
        "--recoverable-failure-fraction",
        type=float,
        default=0.0,
        help=(
            "target training episode fraction for recoverable_failure; "
            "zero preserves the scenario set's natural distribution"
        ),
    )
    parser.add_argument("--output", default="/tmp/mobile_arm_rl_training/fused_low_residual_ppo.pt")
    parser.add_argument("--resume")
    parser.add_argument("--total-steps", type=int, default=500000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument(
        "--balanced-subgoal-types",
        action="store_true",
        help=(
            "balance DIRECT/DETOUR/TERMINAL samples and normalize "
            "advantages separately for each upper-level subgoal type"
        ),
    )
    parser.add_argument(
        "--minimum-subgoal-type-steps",
        type=int,
        default=128,
        help="minimum on-policy samples required from each subgoal type",
    )
    parser.add_argument(
        "--maximum-rollout-steps",
        type=int,
        default=8192,
        help=(
            "maximum typed rollout size while waiting for all three "
            "subgoal types"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--teacher-coefficient", type=float, default=1.0)
    parser.add_argument("--critic-warmup-steps", type=int, default=0)
    parser.add_argument("--maximum-gradient-norm", type=float, default=0.5)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--initial-log-std", type=float, default=-3.0)
    parser.add_argument("--freeze-log-std", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--reward-window", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic-rollout", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.recoverable_failure_fraction <= 1.0:
        parser.error("--recoverable-failure-fraction must be in [0, 1]")
    if args.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive")
    if args.minimum_subgoal_type_steps <= 0:
        parser.error("--minimum-subgoal-type-steps must be positive")
    if args.maximum_rollout_steps < args.rollout_steps:
        parser.error(
            "--maximum-rollout-steps must be at least --rollout-steps"
        )
    if args.balanced_subgoal_types and args.batch_size < 3:
        parser.error("balanced subgoal training requires batch-size >= 3")
    if args.critic_warmup_steps < 0:
        parser.error("--critic-warmup-steps must be non-negative")
    return args


if __name__ == "__main__":
    main()
