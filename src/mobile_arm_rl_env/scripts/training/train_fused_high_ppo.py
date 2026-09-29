#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Behavior-warm-started PPO for the joint-subgoal high-level policy."""

import argparse
import collections
import copy
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_high_actor_critic import FusedHighActorCritic  # noqa: E402
from training.fused_high_env_client import FusedHighEnvironmentClient  # noqa: E402
from training.fused_reach_scenarios import (  # noqa: E402
    FusedReachScenarioSampler,
    load_fused_reach_scenarios,
    scenario_category_counts,
    scenario_step_budget,
    split_fused_reach_scenarios,
)
from training.hrl4in_low_actor_critic import (  # noqa: E402
    RunningObservationNormalizer,
)
from training.ppo_rollout import PPORolloutBuffer  # noqa: E402


def main():
    args = _parse_arguments()
    _seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    scenario_set = load_fused_reach_scenarios(args.scenario_set)
    scenarios = split_fused_reach_scenarios(
        scenario_set["scenarios"],
        split=args.scenario_split,
        validation_fraction=args.scenario_validation_fraction,
        test_fraction=args.scenario_test_fraction,
        seed=args.scenario_split_seed,
    )
    sampler = FusedReachScenarioSampler(
        scenarios,
        seed=args.seed,
        shuffle=True,
    )
    gate_scenarios = split_fused_reach_scenarios(
        scenario_set["scenarios"],
        split=args.bc_gate_split,
        validation_fraction=args.scenario_validation_fraction,
        test_fraction=args.scenario_test_fraction,
        seed=args.scenario_split_seed,
    )
    gate_sampler = FusedReachScenarioSampler(
        gate_scenarios,
        seed=args.scenario_split_seed,
        shuffle=False,
    )
    step_budget = (
        int(args.scenario_step_budget)
        if args.scenario_step_budget > 0
        else int(scenario_set["recommended_step_budget"])
    )
    print(
        "high_curriculum path={} split={} scenarios={} budget={} categories={}".format(
            args.scenario_set,
            args.scenario_split,
            len(scenarios),
            step_budget,
            scenario_category_counts(scenarios),
        )
    )

    model = FusedHighActorCritic(
        hidden_sizes=args.hidden_sizes,
        initial_log_std=args.initial_log_std,
    ).to(device)
    if args.freeze_log_std:
        model.log_std.requires_grad_(False)
    normalizer = RunningObservationNormalizer(
        observation_dim=model.OBS_DIM,
        normalized_dim=model.NORMALIZED_DIM,
    )
    client = FusedHighEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )
    print(
        "high_contract={} low_policy={} interval={} obs={} action={}".format(
            client.metadata.get("reward_contract"),
            client.metadata.get("low_policy_contract"),
            client.metadata.get("high_level_interval"),
            model.OBS_DIM,
            model.ACTION_DIM,
        )
    )

    total_steps = 0
    total_low_steps = 0
    update_index = 0
    warmstart_metrics = {}
    resume_checkpoint = None
    created_warmstart = False
    try:
        if args.resume:
            resume_checkpoint = torch.load(args.resume, map_location=device)
            _validate_checkpoint(resume_checkpoint, model)
            model.load_state_dict(resume_checkpoint["model"])
            normalizer.load_state_dict(resume_checkpoint["normalizer"])
            total_steps = int(resume_checkpoint.get("total_steps", 0))
            total_low_steps = int(
                resume_checkpoint.get("total_low_steps", 0)
            )
            update_index = int(resume_checkpoint.get("update_index", 0))
            warmstart_metrics = dict(
                resume_checkpoint.get("warmstart_metrics", {})
            )
            print(
                "resumed high checkpoint={} steps={} update={}".format(
                    args.resume, total_steps, update_index
                )
            )
        elif args.bc_checkpoint:
            checkpoint = torch.load(args.bc_checkpoint, map_location=device)
            _validate_checkpoint(checkpoint, model)
            model.load_actor_state_dict(checkpoint["model"])
            normalizer.load_state_dict(checkpoint["normalizer"])
            warmstart_metrics = dict(checkpoint.get("warmstart_metrics", {}))
            print(
                "initialized high actor from={} critic=new optimizer=new "
                "log_std=new".format(args.bc_checkpoint)
            )
            # A checkpoint filename is not evidence that the cloned policy
            # is usable in closed loop.  Re-run the same held-out gate before
            # allowing PPO to update it, including for externally supplied
            # BC checkpoints.
            closed_loop_gate = _evaluate_closed_loop_gate(
                client,
                model,
                normalizer,
                gate_sampler,
                step_budget,
                args.bc_gate_episodes,
                device,
            )
            warmstart_metrics["closed_loop_gate"] = closed_loop_gate
            warmstart_metrics["closed_loop_gate_pass"] = (
                _closed_loop_gate_pass(closed_loop_gate, args)
            )
        else:
            warmstart_metrics = _teacher_warmstart(
                client,
                model,
                normalizer,
                sampler,
                gate_sampler,
                step_budget,
                args,
                device,
            )
            created_warmstart = True

        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad],
            lr=args.learning_rate,
        )
        if resume_checkpoint is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])

        if created_warmstart:
            warmstart_path = _warmstart_checkpoint_path(args.output)
            _save_checkpoint(
                warmstart_path,
                model,
                optimizer,
                normalizer,
                0,
                int(warmstart_metrics.get("low_steps", 0)),
                0,
                args,
                warmstart_metrics,
                client.metadata,
                {},
                {},
            )
            print("high BC checkpoint saved: {}".format(warmstart_path))
        if (
                (created_warmstart or args.bc_checkpoint)
                and not bool(warmstart_metrics.get(
                    "closed_loop_gate_pass", False
                ))):
            print(
                "high PPO refused: BC closed-loop gate did not pass {}".format(
                    warmstart_metrics.get("closed_loop_gate", {})
                )
            )
            return
        if args.warmstart_only:
            if not created_warmstart:
                raise ValueError(
                    "--warmstart-only requires online teacher warm-start"
                )
            return

        reference_model = copy.deepcopy(model).to(device)
        if resume_checkpoint is not None:
            reference_state = resume_checkpoint.get("reference_actor_state")
            if not isinstance(reference_state, dict):
                raise ValueError(
                    "resumed checkpoint is missing the frozen BC reference actor"
                )
            reference_model.load_actor_state_dict(reference_state)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)

        buffer = PPORolloutBuffer(
            capacity=args.maximum_rollout_steps,
            observation_dim=model.OBS_DIM,
            action_dim=model.ACTION_DIM,
        )
        raw_observation, current_scenario = _reset_environment(
            client, sampler, step_budget
        )
        current_teacher = client.teacher_action.copy()
        current_teacher_type = int(client.teacher_info.get("subgoal_type", 0))
        current_teacher_route_side = int(
            client.teacher_info.get("route_side", 0)
        )
        rollout_subgoal_types = []
        rollout_teacher_types = []
        rollout_route_sides = []
        rollout_teacher_route_sides = []
        last_done = True
        reward_window = []
        episode_counts = collections.Counter()
        success_counts = collections.Counter()
        collision_counts = collections.Counter()
        timeout_counts = collections.Counter()
        start_time = time.time()

        while total_steps < args.total_steps or buffer.size > 0:
            normalized = normalizer.normalize(raw_observation)
            observation_tensor = torch.from_numpy(normalized).to(
                device
            ).unsqueeze(0)
            with torch.no_grad():
                (
                    action_tensor,
                    type_tensor,
                    route_tensor,
                    logp_tensor,
                    value_tensor,
                ) = model.act(
                    observation_tensor,
                    deterministic=args.deterministic_rollout,
                )
            action = action_tensor.squeeze(0).cpu().numpy()
            subgoal_type = int(type_tensor.item())
            route_side = int(route_tensor.item())
            next_observation, reward, done, info = client.step(
                action,
                subgoal_type=subgoal_type,
                route_side=route_side,
            )
            buffer.add(
                observation=normalized,
                action=action,
                teacher_action=current_teacher,
                teacher_valid=True,
                log_probability=float(logp_tensor.item()),
                value=float(value_tensor.item()),
                reward=reward,
                done=done,
            )
            rollout_subgoal_types.append(subgoal_type)
            rollout_teacher_types.append(current_teacher_type)
            rollout_route_sides.append(route_side)
            rollout_teacher_route_sides.append(current_teacher_route_side)
            total_steps += 1
            total_low_steps += int(info.get("low_steps", 0))
            reward_window.append(float(reward))
            last_done = bool(done)
            if done:
                category = _scenario_category(current_scenario)
                episode_counts[category] += 1
                success_counts[category] += int(bool(info.get("success", False)))
                collision_counts[category] += int(bool(info.get("collision", False)))
                timeout_counts[category] += int(bool(info.get("timeout", False)))
                raw_observation, current_scenario = _reset_environment(
                    client, sampler, step_budget
                )
            else:
                raw_observation = next_observation
            current_teacher = client.teacher_action.copy()
            current_teacher_type = int(
                client.teacher_info.get("subgoal_type", 0)
            )
            current_teacher_route_side = int(
                client.teacher_info.get("route_side", 0)
            )

            rollout_ready = bool(
                buffer.size >= args.rollout_steps and last_done
            )
            training_complete = bool(total_steps >= args.total_steps)
            if training_complete and buffer.size > 0:
                rollout_ready = True
            if buffer.full and not rollout_ready:
                raise RuntimeError(
                    "high rollout reached maximum size before an episode boundary"
                )
            if not rollout_ready:
                continue

            last_value = _bootstrap(
                model,
                normalizer,
                raw_observation,
                last_done,
                device,
            )
            buffer.compute_returns_and_advantages(
                last_value,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            teacher_coefficient = _linear_coefficient(
                args.teacher_coefficient,
                args.teacher_final_coefficient,
                total_steps,
                args.teacher_decay_steps,
            )
            metrics = _ppo_update(
                model,
                optimizer,
                buffer,
                device,
                args,
                teacher_coefficient,
                rollout_subgoal_types,
                rollout_teacher_types,
                rollout_route_sides,
                rollout_teacher_route_sides,
                reference_model,
                actor_frozen=bool(total_steps <= args.critic_warmup_steps),
            )
            rollout_size = buffer.size
            buffer.clear()
            rollout_subgoal_types = []
            rollout_teacher_types = []
            rollout_route_sides = []
            rollout_teacher_route_sides = []
            update_index += 1
            print(
                "high_update={} high_steps={} low_steps={} rollout={} "
                "reward={:.4f} policy_loss={:.5f} value_loss={:.5f} "
                "teacher_mse={:.6f} teacher_type_loss={:.5f} "
                "teacher_type_accuracy={:.3f} teacher_route_loss={:.5f} "
                "teacher_route_accuracy={:.3f} teacher_option_loss={:.5f} "
                "teacher_option_accuracy={:.3f} reference_mse={:.6f} "
                "teacher_coef={:.4f} entropy={:.5f} "
                "action_abs={:.4f} actor_frozen={} episode_rates={} "
                "elapsed_s={:.1f}".format(
                    update_index,
                    total_steps,
                    total_low_steps,
                    rollout_size,
                    float(np.mean(reward_window[-args.reward_window:])),
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["teacher_loss"],
                    metrics["teacher_type_loss"],
                    metrics["teacher_type_accuracy"],
                    metrics["teacher_route_loss"],
                    metrics["teacher_route_accuracy"],
                    metrics["teacher_option_loss"],
                    metrics["teacher_option_accuracy"],
                    metrics["reference_loss"],
                    teacher_coefficient,
                    metrics["entropy"],
                    metrics["action_abs"],
                    metrics["actor_frozen"],
                    _episode_rates(
                        episode_counts,
                        success_counts,
                        collision_counts,
                        timeout_counts,
                    ),
                    time.time() - start_time,
                )
            )
            if (
                    args.checkpoint_interval > 0
                    and update_index % args.checkpoint_interval == 0):
                numbered_path = _numbered_checkpoint_path(
                    args.output, update_index
                )
                _save_checkpoint(
                    numbered_path,
                    model,
                    optimizer,
                    normalizer,
                    total_steps,
                    total_low_steps,
                    update_index,
                    args,
                    warmstart_metrics,
                    client.metadata,
                    episode_counts,
                    success_counts,
                    reference_model=reference_model,
                )
                _save_checkpoint(
                    args.output,
                    model,
                    optimizer,
                    normalizer,
                    total_steps,
                    total_low_steps,
                    update_index,
                    args,
                    warmstart_metrics,
                    client.metadata,
                    episode_counts,
                    success_counts,
                    reference_model=reference_model,
                )
                print("high checkpoint saved: {}".format(numbered_path))
        _save_checkpoint(
            args.output,
            model,
            optimizer,
            normalizer,
            total_steps,
            total_low_steps,
            update_index,
            args,
            warmstart_metrics,
            client.metadata,
            episode_counts,
            success_counts,
            reference_model=reference_model,
        )
        print("high model saved: {}".format(args.output))
    finally:
        client.close()


def _teacher_warmstart(
        client,
        model,
        normalizer,
        sampler,
        gate_sampler,
        step_budget,
        args,
        device):
    if args.teacher_warmup_steps <= 0:
        raise ValueError(
            "use --bc-checkpoint or a positive --teacher-warmup-steps; "
            "random high-level PPO is intentionally disabled"
        )
    random_state = np.random.RandomState(args.seed)
    successful_episodes, collection = _collect_successful_teacher_episodes(
        client, sampler, step_budget, args
    )
    training_episodes, validation_episodes = _episode_group_split(
        successful_episodes,
        args.warmstart_validation_fraction,
        random_state,
    )
    training_data = _flatten_behavior_episodes(training_episodes)
    validation_data = _flatten_behavior_episodes(validation_episodes)
    all_teacher_data = _flatten_behavior_episodes(successful_episodes)
    normalizer.update(all_teacher_data["observations"])
    fit_metrics = _fit_behavior_policy(
        model,
        normalizer,
        training_data,
        validation_data,
        args.warmstart_epochs,
        args,
        device,
        random_state,
        label="initial",
    )

    initial_selection = _evaluate_closed_loop_gate(
        client,
        model,
        normalizer,
        gate_sampler,
        step_budget,
        args.dagger_selection_episodes,
        device,
        label="initial",
    )
    best_actor_state = _clone_actor_state(model)
    best_selection = dict(initial_selection)
    best_selection_key = _closed_loop_selection_key(initial_selection)
    best_fit_metrics = dict(fit_metrics)
    best_round = 0
    selection_history = [{
        "round": 0,
        "gate": dict(initial_selection),
        "selected_as_best": True,
    }]

    dagger_episodes = []
    dagger_metrics = []
    _save_behavior_round_artifacts(
        args,
        "initial",
        model,
        normalizer,
        successful_episodes,
        dagger_episodes,
        {"fit": fit_metrics, "selection_gate": initial_selection},
        client.metadata,
    )
    for round_index in range(args.dagger_rounds):
        probability = _dagger_teacher_probability(
            round_index,
            args.dagger_rounds,
            args.dagger_pure_student_rounds,
            args.dagger_initial_teacher_probability,
            args.dagger_final_teacher_probability,
        )
        new_episodes, round_metrics = _collect_dagger_episodes(
            client,
            model,
            normalizer,
            sampler,
            step_budget,
            args.dagger_episodes_per_round,
            probability,
            device,
            random_state,
            round_index + 1,
            args,
        )
        dagger_episodes.extend(new_episodes)
        training_data = _flatten_behavior_episodes(
            training_episodes + dagger_episodes
        )
        fit_metrics = _fit_behavior_policy(
            model,
            normalizer,
            training_data,
            validation_data,
            args.dagger_epochs,
            args,
            device,
            random_state,
            label="dagger_{}".format(round_index + 1),
        )
        round_metrics.update(fit_metrics)
        selection = _evaluate_closed_loop_gate(
            client,
            model,
            normalizer,
            gate_sampler,
            step_budget,
            args.dagger_selection_episodes,
            device,
            label="dagger_{}".format(round_index + 1),
        )
        selection_key = _closed_loop_selection_key(selection)
        selected_as_best = bool(selection_key > best_selection_key)
        if selected_as_best:
            best_selection_key = selection_key
            best_selection = dict(selection)
            best_actor_state = _clone_actor_state(model)
            best_fit_metrics = dict(fit_metrics)
            best_round = round_index + 1
        round_metrics["selection_gate"] = dict(selection)
        round_metrics["selected_as_best"] = selected_as_best
        selection_history.append({
            "round": int(round_index + 1),
            "gate": dict(selection),
            "selected_as_best": selected_as_best,
        })
        dagger_metrics.append(round_metrics)
        _save_behavior_round_artifacts(
            args,
            "dagger_{:02d}".format(round_index + 1),
            model,
            normalizer,
            successful_episodes,
            dagger_episodes,
            round_metrics,
            client.metadata,
        )

    # Open-loop loss is only a diagnostic.  Restore the actor that achieved
    # the strongest deterministic closed-loop outcome before applying the
    # final, larger acceptance gate.
    model.load_actor_state_dict(best_actor_state)
    fit_metrics = best_fit_metrics

    if args.warmstart_dataset:
        _save_behavior_dataset(
            args.warmstart_dataset,
            successful_episodes,
            dagger_episodes,
        )

    closed_loop_gate = _evaluate_closed_loop_gate(
        client,
        model,
        normalizer,
        gate_sampler,
        step_budget,
        args.bc_gate_episodes,
        device,
        label="final_selected_round_{}".format(best_round),
    )
    gate_pass = _closed_loop_gate_pass(closed_loop_gate, args)
    metrics = {
        "samples": int(all_teacher_data["observations"].shape[0]),
        "successful_episodes": int(len(successful_episodes)),
        "attempted_episodes": int(collection["attempted_episodes"]),
        "teacher_success_rate": float(collection["success_rate"]),
        "discarded_failed_episodes": int(collection["failed_episodes"]),
        "low_steps": int(collection["low_steps"]),
        "training_episode_count": int(len(training_episodes)),
        "validation_episode_count": int(len(validation_episodes)),
        "validation_loss": float(fit_metrics["validation_loss"]),
        "validation_type_accuracy": float(
            fit_metrics["validation_type_accuracy"]
        ),
        "validation_route_accuracy": float(
            fit_metrics["validation_route_accuracy"]
        ),
        "validation_option_accuracy": float(
            fit_metrics["validation_option_accuracy"]
        ),
        "dagger_rounds": dagger_metrics,
        "closed_loop_selection_history": selection_history,
        "selected_dagger_round": int(best_round),
        "selected_closed_loop_gate": best_selection,
        "closed_loop_gate": closed_loop_gate,
        "closed_loop_gate_pass": gate_pass,
    }
    print("high_bc_complete {}".format(metrics))
    return metrics


def _collect_successful_teacher_episodes(client, sampler, step_budget, args):
    kept = []
    attempted = 0
    low_steps = 0
    kept_samples = 0
    while (
            kept_samples < args.teacher_warmup_steps
            or len(kept) < args.teacher_warmup_episodes):
        if attempted >= args.teacher_warmup_max_episodes:
            raise RuntimeError(
                "teacher warm-start exhausted {} attempts with {} successful "
                "episodes and {} samples".format(
                    attempted, len(kept), kept_samples
                )
            )
        observation, scenario = _reset_environment(
            client, sampler, step_budget
        )
        episode = _new_behavior_episode(scenario, source="teacher")
        final_info = {}
        while True:
            teacher_action, teacher_type, teacher_route = _teacher_label(client)
            _append_behavior_sample(
                episode,
                observation,
                teacher_action,
                teacher_type,
                teacher_route,
            )
            observation, unused_reward, done, final_info = client.step(
                teacher_action,
                subgoal_type=teacher_type,
                route_side=teacher_route,
            )
            low_steps += int(final_info.get("low_steps", 0))
            if done:
                break
        attempted += 1
        success = bool(final_info.get("success", False))
        if success:
            kept.append(episode)
            kept_samples += len(episode["observations"])
        print(
            "high_teacher_episode={} scenario_id={} episode_samples={} "
            "kept_samples={} success={} kept={} collision={} timeout={}".format(
                attempted,
                scenario.get("scenario_id"),
                len(episode["observations"]),
                kept_samples,
                success,
                success,
                bool(final_info.get("collision", False)),
                bool(final_info.get("timeout", False)),
            )
        )
    return kept, {
        "attempted_episodes": attempted,
        "failed_episodes": attempted - len(kept),
        "success_rate": float(len(kept)) / float(max(attempted, 1)),
        "low_steps": low_steps,
    }


def _collect_dagger_episodes(
        client,
        model,
        normalizer,
        sampler,
        step_budget,
        episode_count,
        teacher_probability,
        device,
        random_state,
        round_index,
        args):
    episodes = []
    successes = 0
    collisions = 0
    teacher_steps = 0
    total_steps = 0
    risk_steps = 0
    intervention_events = 0
    risk_reason_counts = collections.Counter()
    for episode_index in range(int(episode_count)):
        observation, scenario = _reset_environment(
            client, sampler, step_budget
        )
        episode = _new_behavior_episode(
            scenario, source="dagger_{}".format(round_index)
        )
        final_info = {}
        previous_info = {}
        stall_count = 0
        intervention_remaining = 0
        intervention_active = False
        recovery_stable_count = 0
        while True:
            teacher_action, teacher_type, teacher_route = _teacher_label(client)
            tensor = torch.from_numpy(
                normalizer.normalize(observation)
            ).to(device).unsqueeze(0)
            with torch.no_grad():
                student_action, student_type, student_route = (
                    model.deterministic_decision(tensor)
                )
            student_action = student_action.squeeze(0).cpu().numpy()
            student_type = int(student_type.item())
            student_route = int(student_route.item())
            risk_reasons = _dagger_risk_reasons(
                observation,
                student_action,
                student_type,
                student_route,
                teacher_action,
                teacher_type,
                teacher_route,
                previous_info,
                stall_count,
                args,
            )
            risky = bool(risk_reasons)
            if risky:
                risk_steps += 1
                risk_reason_counts.update(risk_reasons)
            trigger = bool(
                risky
                and not intervention_active
                and teacher_probability > 0.0
                and random_state.rand() < teacher_probability
            )
            if trigger:
                intervention_active = True
                intervention_remaining = int(
                    args.dagger_intervention_steps
                )
                recovery_stable_count = 0
                intervention_events += 1
                _mark_recent_samples(
                    episode,
                    args.dagger_pre_intervention_steps,
                    args.dagger_intervention_weight,
                )
            use_teacher = bool(intervention_active)
            sample_weight = 1.0
            if risky:
                sample_weight = max(
                    sample_weight,
                    float(args.dagger_risk_sample_weight),
                )
            if use_teacher:
                sample_weight = max(
                    sample_weight,
                    float(args.dagger_intervention_weight),
                )
            _append_behavior_sample(
                episode,
                observation,
                teacher_action,
                teacher_type,
                teacher_route,
                sample_weight=sample_weight,
                intervention=use_teacher,
                risky=risky,
            )
            if use_teacher:
                action = teacher_action
                subgoal_type = teacher_type
                route_side = teacher_route
                teacher_steps += 1
                intervention_remaining = max(
                    intervention_remaining - 1, 0
                )
                if intervention_remaining <= 0:
                    if risky:
                        recovery_stable_count = 0
                    else:
                        recovery_stable_count += 1
            else:
                action = student_action
                subgoal_type = student_type
                route_side = student_route
            observation, unused_reward, done, final_info = client.step(
                action,
                subgoal_type=subgoal_type,
                route_side=route_side,
            )
            total_steps += 1
            reward_info = final_info.get("high_reward", {})
            progress = max(
                float(reward_info.get("path_progress", 0.0)),
                float(reward_info.get("goal_progress", 0.0)),
                float(reward_info.get("base_subgoal_progress", 0.0)),
                float(reward_info.get("ee_subgoal_progress", 0.0)),
            )
            if progress < float(args.dagger_minimum_progress):
                stall_count += 1
            else:
                stall_count = 0
            previous_info = dict(final_info)
            if (
                    intervention_active
                    and intervention_remaining <= 0
                    and recovery_stable_count
                    >= args.dagger_recovery_stable_steps):
                intervention_active = False
                recovery_stable_count = 0
            if done:
                break
        success = bool(final_info.get("success", False))
        if not success:
            _mark_recent_samples(
                episode,
                args.dagger_pre_intervention_steps,
                args.dagger_risk_sample_weight,
            )
            # A failed teacher rollout is useful for locating recovery
            # states, but its labels are not a successful demonstration.
            # Keep it in DAgger without allowing it to dominate successful
            # complete-option data.
            _cap_sample_weights(
                episode, args.dagger_failed_sample_weight
            )
        raw_sample_count = len(episode["observations"])
        episode = _compact_behavior_episode(
            episode, args.dagger_max_samples_per_episode
        )
        successes += int(success)
        collisions += int(bool(final_info.get("collision", False)))
        episodes.append(episode)
        print(
            "high_dagger_round={} episode={} scenario_id={} samples={} "
            "raw_samples={} teacher_probability={:.3f} risk_steps={} "
            "interventions={} "
            "success={} collision={} timeout={}".format(
                round_index,
                episode_index + 1,
                scenario.get("scenario_id"),
                len(episode["observations"]),
                raw_sample_count,
                teacher_probability,
                int(sum(episode["risk_flags"])),
                int(sum(episode["interventions"])),
                success,
                bool(final_info.get("collision", False)),
                bool(final_info.get("timeout", False)),
            )
        )
    return episodes, {
        "round": int(round_index),
        "episodes": int(episode_count),
        "successes": int(successes),
        "collisions": int(collisions),
        "teacher_probability": float(teacher_probability),
        "teacher_execution_rate": float(teacher_steps) / float(max(total_steps, 1)),
        "pure_student_execution": bool(teacher_probability <= 0.0),
        "risk_rate": float(risk_steps) / float(max(total_steps, 1)),
        "intervention_events": int(intervention_events),
        "risk_reasons": dict(risk_reason_counts),
    }


def _new_behavior_episode(scenario, source):
    return {
        "scenario_id": int(scenario.get("scenario_id", -1)),
        "source": str(source),
        "observations": [],
        "actions": [],
        "types": [],
        "routes": [],
        "sample_weights": [],
        "interventions": [],
        "risk_flags": [],
    }


def _append_behavior_sample(
        episode,
        observation,
        action,
        subgoal_type,
        route_side,
        sample_weight=1.0,
        intervention=False,
        risky=False):
    episode["observations"].append(
        np.asarray(observation, dtype=np.float32).copy()
    )
    episode["actions"].append(
        np.asarray(action, dtype=np.float32).copy()
    )
    episode["types"].append(int(subgoal_type))
    episode["routes"].append(int(route_side))
    episode["sample_weights"].append(float(sample_weight))
    episode["interventions"].append(bool(intervention))
    episode["risk_flags"].append(bool(risky))


def _mark_recent_samples(episode, count, minimum_weight):
    count = max(int(count), 0)
    minimum_weight = float(minimum_weight)
    if count <= 0 or not episode["sample_weights"]:
        return
    start = max(0, len(episode["sample_weights"]) - count)
    for index in range(start, len(episode["sample_weights"])):
        episode["sample_weights"][index] = max(
            float(episode["sample_weights"][index]), minimum_weight
        )
        episode["risk_flags"][index] = True


def _cap_sample_weights(episode, maximum_weight):
    maximum_weight = max(float(maximum_weight), 1e-6)
    episode["sample_weights"] = [
        min(float(value), maximum_weight)
        for value in episode["sample_weights"]
    ]


def _compact_behavior_episode(episode, maximum_samples):
    """Bound long failed rollouts while retaining temporal/risk coverage.

    Each temporal bin contributes one sample. Within a bin, intervention and
    risk states outrank ordinary repeated states, preventing a 3200-step
    timeout from overwhelming complete successful option demonstrations.
    """
    maximum_samples = int(maximum_samples)
    sample_count = len(episode["observations"])
    if maximum_samples <= 0 or sample_count <= maximum_samples:
        return episode
    selected = []
    for bin_index in range(maximum_samples):
        start = int(math.floor(
            float(bin_index) * sample_count / maximum_samples
        ))
        end = int(math.floor(
            float(bin_index + 1) * sample_count / maximum_samples
        ))
        end = max(end, start + 1)
        candidates = range(start, min(end, sample_count))
        selected.append(max(
            candidates,
            key=lambda index: (
                int(episode["interventions"][index]),
                int(episode["risk_flags"][index]),
                float(episode["sample_weights"][index]),
                index,
            ),
        ))
    compact = dict(episode)
    for name in (
            "observations",
            "actions",
            "types",
            "routes",
            "sample_weights",
            "interventions",
            "risk_flags"):
        compact[name] = [episode[name][index] for index in selected]
    return compact


def _dagger_teacher_probability(
        round_index,
        round_count,
        pure_student_rounds,
        initial_probability,
        final_probability):
    round_index = int(round_index)
    round_count = int(round_count)
    pure_student_rounds = int(pure_student_rounds)
    active_rounds = max(round_count - pure_student_rounds, 0)
    if round_index >= active_rounds:
        return 0.0
    if active_rounds <= 1:
        return float(initial_probability)
    return _linear_coefficient(
        initial_probability,
        final_probability,
        round_index,
        active_rounds - 1,
    )


def _dagger_risk_reasons(
        observation,
        student_action,
        student_type,
        student_route,
        teacher_action,
        teacher_type,
        teacher_route,
        previous_info,
        stall_count,
        args):
    observation = np.asarray(observation, dtype=np.float32)
    student_action = np.asarray(student_action, dtype=np.float32)
    teacher_action = np.asarray(teacher_action, dtype=np.float32)
    reasons = []
    remembered_route = int(np.argmax(observation[35:38]))
    route_locked = remembered_route in (1, 2)
    alternative_unlocked_route = bool(
        not route_locked
        and int(student_type) == 1
        and int(teacher_type) == 1
        and int(student_route) != int(teacher_route)
    )
    if int(student_type) != int(teacher_type):
        reasons.append("subgoal_type_disagreement")
    if (
            route_locked
            and int(student_type) == 1
            and int(student_route) != remembered_route):
        reasons.append("latched_route_disagreement")
    deviation = float(np.max(np.abs(student_action - teacher_action)))
    if (
            not alternative_unlocked_route
            and deviation > float(args.dagger_action_deviation_threshold)):
        reasons.append("teacher_action_deviation")
    low_steps = int(previous_info.get("low_steps", 0))
    safety_steps = int(previous_info.get("safety_steps", 0))
    safety_rate = float(safety_steps) / float(max(low_steps, 1))
    if (
            low_steps > 0
            and safety_rate >= float(args.dagger_safety_rate_threshold)):
        reasons.append("safety_intervention")
    reward_info = previous_info.get("high_reward", {})
    if bool(reward_info.get("subgoal_failed", False)):
        reasons.append("subgoal_failed")
    option_termination = str(previous_info.get(
        "option_termination", ""
    ))
    if option_termination in ("stalled", "safety_risk"):
        reasons.append("option_{}".format(option_termination))
    if int(stall_count) >= int(args.dagger_progress_stall_steps):
        reasons.append("progress_stall")
    return sorted(set(reasons))


def _teacher_label(client):
    return (
        client.teacher_action.copy(),
        int(client.teacher_info.get("subgoal_type", 0)),
        int(client.teacher_info.get("route_side", 0)),
    )


def _episode_group_split(episodes, validation_fraction, random_state):
    if len(episodes) < 2:
        raise ValueError("at least two successful teacher episodes are required")
    order = random_state.permutation(len(episodes))
    validation_count = max(1, int(round(
        len(episodes) * float(validation_fraction)
    )))
    validation_count = min(validation_count, len(episodes) - 1)
    validation_set = set(int(value) for value in order[:validation_count])
    training = [
        episode for index, episode in enumerate(episodes)
        if index not in validation_set
    ]
    validation = [
        episode for index, episode in enumerate(episodes)
        if index in validation_set
    ]
    return training, validation


def _flatten_behavior_episodes(episodes):
    observations = []
    actions = []
    types = []
    routes = []
    episode_ids = []
    scenario_ids = []
    sources = []
    sample_weights = []
    interventions = []
    risk_flags = []
    for episode_id, episode in enumerate(episodes):
        count = len(episode["observations"])
        observations.extend(episode["observations"])
        actions.extend(episode["actions"])
        types.extend(episode["types"])
        routes.extend(episode["routes"])
        episode_ids.extend([episode_id] * count)
        scenario_ids.extend([episode["scenario_id"]] * count)
        sources.extend([episode["source"]] * count)
        sample_weights.extend(
            episode.get("sample_weights", [1.0] * count)
        )
        interventions.extend(
            episode.get("interventions", [False] * count)
        )
        risk_flags.extend(
            episode.get("risk_flags", [False] * count)
        )
    if not observations:
        raise ValueError("behavior dataset contains no samples")
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "types": np.asarray(types, dtype=np.int64),
        "routes": np.asarray(routes, dtype=np.int64),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "scenario_ids": np.asarray(scenario_ids, dtype=np.int64),
        "sources": np.asarray(sources),
        "sample_weights": np.asarray(sample_weights, dtype=np.float32),
        "interventions": np.asarray(interventions, dtype=np.bool_),
        "risk_flags": np.asarray(risk_flags, dtype=np.bool_),
    }


def _balanced_behavior_epoch_indices(types, routes, random_state):
    """Return one epoch with equal samples for every observed option.

    The continuous actor has a distinct head for each (subgoal type, route)
    option.  Uniformly shuffling raw trajectories would otherwise let long
    DETOUR phases starve DIRECT, TERMINAL, and the less frequent route side.
    """
    types = np.asarray(types, dtype=np.int64)
    routes = np.asarray(routes, dtype=np.int64)
    if types.shape != routes.shape or types.ndim != 1:
        raise ValueError("behavior type and route labels must be 1-D peers")
    groups = []
    for subgoal_type, route_side in sorted(set(zip(types, routes))):
        indices = np.flatnonzero(
            (types == int(subgoal_type)) & (routes == int(route_side))
        )
        if indices.size:
            groups.append(indices)
    if not groups:
        raise ValueError("behavior dataset contains no option groups")
    target_count = max(int(indices.size) for indices in groups)
    balanced = []
    for indices in groups:
        if int(indices.size) < target_count:
            selected = random_state.choice(
                indices, size=target_count, replace=True
            )
        else:
            selected = indices.copy()
        balanced.append(selected)
    order = np.concatenate(balanced).astype(np.int64, copy=False)
    random_state.shuffle(order)
    return order


def _fit_behavior_policy(
        model,
        normalizer,
        training_data,
        validation_data,
        epochs,
        args,
        device,
        random_state,
        label):
    training_observations = normalizer.normalize(
        training_data["observations"]
    )
    validation_observations = normalizer.normalize(
        validation_data["observations"]
    )
    actor_parameters = (
        list(model.actor_backbone.parameters())
        + list(model.actor_mean.parameters())
        + list(model.option_head.parameters())
    )
    optimizer = torch.optim.Adam(
        actor_parameters, lr=args.warmstart_learning_rate
    )
    action_weights = torch.as_tensor(
        args.teacher_action_weights,
        dtype=torch.float32,
        device=device,
    ).view(1, -1)
    training_options = _encode_option_labels(
        training_data["types"], training_data["routes"]
    )
    validation_options = _encode_option_labels(
        validation_data["types"], validation_data["routes"]
    )
    option_weights, option_count_values = _balanced_class_weights(
        training_options, model.OPTION_COUNT, device, "unified option"
    )
    type_counts = np.bincount(
        training_data["types"], minlength=model.SUBGOAL_TYPE_COUNT
    ).astype(int).tolist()
    route_counts = np.bincount(
        training_data["routes"], minlength=model.ROUTE_SIDE_COUNT
    ).astype(int).tolist()
    option_counts = collections.Counter(
        (int(subgoal_type), int(route_side))
        for subgoal_type, route_side in zip(
            training_data["types"], training_data["routes"]
        )
    )
    print(
        "high_bc_balance label={} type_counts={} route_counts={} "
        "option_counts={} option_class_counts={} risk_samples={} "
        "interventions={}".format(
            label,
            type_counts,
            route_counts,
            dict(option_counts),
            option_count_values,
            int(np.sum(training_data["risk_flags"])),
            int(np.sum(training_data["interventions"])),
        )
    )
    best_validation = float("inf")
    best_state = None
    best_type_accuracy = 0.0
    best_route_accuracy = 0.0
    best_option_accuracy = 0.0
    for epoch in range(1, int(epochs) + 1):
        order = _balanced_behavior_epoch_indices(
            training_data["types"],
            training_data["routes"],
            random_state,
        )
        model.train()
        train_losses = []
        for start in range(0, len(order), args.warmstart_batch_size):
            indices = order[start:start + args.warmstart_batch_size]
            observations = torch.from_numpy(
                training_observations[indices]
            ).to(device)
            targets = torch.from_numpy(
                training_data["actions"][indices]
            ).to(device)
            type_targets = torch.from_numpy(
                training_data["types"][indices]
            ).to(device)
            route_targets = torch.from_numpy(
                training_data["routes"][indices]
            ).to(device)
            option_targets = torch.from_numpy(
                training_options[indices]
            ).to(device)
            sample_weights = torch.from_numpy(
                training_data["sample_weights"][indices]
            ).to(device)
            sample_weights = sample_weights / torch.clamp(
                sample_weights.mean(), min=1e-6
            )
            predicted = model.deterministic_action(
                observations, type_targets, route_targets
            )
            action_loss_per_sample = (
                (predicted - targets).pow(2) * action_weights
            ).mean(dim=-1)
            action_loss = (
                action_loss_per_sample * sample_weights
            ).mean()
            option_loss_per_sample = F.cross_entropy(
                model.option_logits(observations),
                option_targets,
                weight=option_weights,
                reduction="none",
            )
            option_loss = (
                option_loss_per_sample * sample_weights
            ).mean()
            loss = (
                action_loss
                + args.warmstart_option_coefficient * option_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor_parameters, args.maximum_gradient_norm
            )
            optimizer.step()
            train_losses.append(float(loss.item()))
        validation = _behavior_validation_metrics(
            model,
            validation_observations,
            validation_data,
            validation_options,
            action_weights,
            option_weights,
            args,
            device,
        )
        if validation["loss"] < best_validation:
            best_validation = validation["loss"]
            best_type_accuracy = validation["type_accuracy"]
            best_route_accuracy = validation["route_accuracy"]
            best_option_accuracy = validation["option_accuracy"]
            best_state = dict(
                (name, value.detach().cpu().clone())
                for name, value in model.actor_state_dict().items()
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            print(
                "high_bc_epoch label={} epoch={}/{} train_loss={:.7f} "
                "validation_loss={:.7f} type_accuracy={:.3f} "
                "route_accuracy={:.3f} option_accuracy={:.3f}".format(
                    label,
                    epoch,
                    epochs,
                    float(np.mean(train_losses)),
                    validation["loss"],
                    validation["type_accuracy"],
                    validation["route_accuracy"],
                    validation["option_accuracy"],
                )
            )
    model.load_actor_state_dict(best_state)
    model.train()
    return {
        "validation_loss": float(best_validation),
        "validation_type_accuracy": float(best_type_accuracy),
        "validation_route_accuracy": float(best_route_accuracy),
        "validation_option_accuracy": float(best_option_accuracy),
        "training_risk_samples": int(np.sum(training_data["risk_flags"])),
        "training_interventions": int(
            np.sum(training_data["interventions"])
        ),
    }


def _behavior_validation_metrics(
        model,
        observations,
        data,
        option_targets,
        action_weights,
        option_weights,
        args,
        device):
    model.eval()
    with torch.no_grad():
        observation_tensor = torch.from_numpy(observations).to(device)
        target = torch.from_numpy(data["actions"]).to(device)
        type_target = torch.from_numpy(data["types"]).to(device)
        route_target = torch.from_numpy(data["routes"]).to(device)
        option_target = torch.from_numpy(option_targets).to(device)
        prediction = model.deterministic_action(
            observation_tensor, type_target, route_target
        )
        type_logits = model.subgoal_type_logits(observation_tensor)
        route_logits = model.route_side_logits(observation_tensor)
        option_logits = model.option_logits(observation_tensor)
        action_loss = ((prediction - target).pow(2) * action_weights).mean()
        option_loss = F.cross_entropy(
            option_logits, option_target, weight=option_weights
        )
        loss = (
            action_loss
            + args.warmstart_option_coefficient * option_loss
        )
        return {
            "loss": float(loss.item()),
            "type_accuracy": float((
                torch.argmax(type_logits, dim=-1) == type_target
            ).float().mean().item()),
            "route_accuracy": float((
                torch.argmax(route_logits, dim=-1) == route_target
            ).float().mean().item()),
            "option_accuracy": float((
                torch.argmax(option_logits, dim=-1) == option_target
            ).float().mean().item()),
        }


def _encode_option_labels(types, routes):
    """Encode the only four legal (semantic type, route) pairs."""
    types = np.asarray(types, dtype=np.int64)
    routes = np.asarray(routes, dtype=np.int64)
    if types.shape != routes.shape:
        raise ValueError("behavior type and route labels must be peers")
    options = np.full(types.shape, -1, dtype=np.int64)
    options[(types == 0) & (routes == 0)] = 0
    options[(types == 1) & (routes == 1)] = 1
    options[(types == 1) & (routes == 2)] = 2
    options[(types == 2) & (routes == 0)] = 3
    if np.any(options < 0):
        invalid = sorted(set(zip(
            types[options < 0].tolist(),
            routes[options < 0].tolist(),
        )))
        raise ValueError(
            "behavior dataset contains invalid high options: {}".format(
                invalid
            )
        )
    return options


def _balanced_class_weights(labels, count, device, name):
    counts = np.bincount(labels, minlength=count).astype(np.float64)
    if np.any(counts <= 0.0):
        raise RuntimeError(
            "behavior dataset is missing {} class: {}".format(
                name, counts.astype(int).tolist()
            )
        )
    weights = np.sqrt(float(len(labels)) / (float(count) * counts))
    return (
        torch.as_tensor(weights, dtype=torch.float32, device=device),
        counts.astype(int).tolist(),
    )


def _save_behavior_dataset(path, teacher_episodes, dagger_episodes):
    data = _flatten_behavior_episodes(teacher_episodes + dagger_episodes)
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    np.savez_compressed(
        path,
        observations=data["observations"],
        teacher_actions=data["actions"],
        teacher_subgoal_types=data["types"],
        teacher_route_sides=data["routes"],
        episode_ids=data["episode_ids"],
        scenario_ids=data["scenario_ids"],
        sources=data["sources"],
        sample_weights=data["sample_weights"],
        interventions=data["interventions"],
        risk_flags=data["risk_flags"],
    )


def _save_behavior_round_artifacts(
        args,
        label,
        model,
        normalizer,
        teacher_episodes,
        dagger_episodes,
        metrics,
        environment_metadata):
    """Persist every BC/DAgger round so an interruption loses no progress."""
    checkpoint_path = _labeled_artifact_path(
        args.output, label, ".bc.pt"
    )
    directory = os.path.dirname(os.path.abspath(checkpoint_path))
    os.makedirs(directory, exist_ok=True)
    torch.save({
        "format_version": 3,
        "checkpoint_kind": "high_behavior_round",
        "round_label": str(label),
        "model": model.state_dict(),
        "normalizer": normalizer.state_dict(),
        "hidden_sizes": list(args.hidden_sizes),
        "arguments": vars(args),
        "observation_dim": model.OBS_DIM,
        "action_dim": model.ACTION_DIM,
        "subgoal_type_count": model.SUBGOAL_TYPE_COUNT,
        "route_side_count": model.ROUTE_SIDE_COUNT,
        "option_count": model.OPTION_COUNT,
        "policy_type": model.POLICY_TYPE,
        "warmstart_metrics": dict(metrics),
        "environment_metadata": dict(environment_metadata),
    }, checkpoint_path)
    dataset_path = None
    if args.warmstart_dataset:
        dataset_path = _labeled_artifact_path(
            args.warmstart_dataset, label, ".npz"
        )
        _save_behavior_dataset(
            dataset_path, teacher_episodes, dagger_episodes
        )
    print(
        "high_behavior_round_saved label={} checkpoint={} dataset={}".format(
            label, checkpoint_path, dataset_path
        )
    )


def _labeled_artifact_path(path, label, suffix):
    path = os.path.abspath(path)
    if path.endswith(suffix):
        stem = path[:-len(suffix)]
    else:
        stem = os.path.splitext(path)[0]
    return "{}.{}.{}".format(
        stem,
        str(label),
        suffix.lstrip("."),
    )
    print("high behavior dataset saved: {}".format(path))


def _evaluate_closed_loop_gate(
        client,
        model,
        normalizer,
        sampler,
        step_budget,
        episode_count,
        device,
        label="gate"):
    successes = 0
    collisions = 0
    timeouts = 0
    direct_episodes = 0
    direct_successes = 0
    final_distances = []
    low_steps = 0
    safety_steps = 0
    for episode_index in range(int(episode_count)):
        observation, scenario = _reset_environment(
            client, sampler, step_budget
        )
        final_info = {}
        while True:
            tensor = torch.from_numpy(
                normalizer.normalize(observation)
            ).to(device).unsqueeze(0)
            with torch.no_grad():
                action, subgoal_type, route_side = (
                    model.deterministic_decision(tensor)
                )
            observation, unused_reward, done, final_info = client.step(
                action.squeeze(0).cpu().numpy(),
                subgoal_type=int(subgoal_type.item()),
                route_side=int(route_side.item()),
            )
            low_steps += int(final_info.get("low_steps", 0))
            safety_steps += int(final_info.get("safety_steps", 0))
            if done:
                break
        success = bool(final_info.get("success", False))
        collision = bool(final_info.get("collision", False))
        timeout = bool(final_info.get("timeout", False))
        direct = bool(scenario.get("no_obstacle", False))
        successes += int(success)
        collisions += int(collision)
        timeouts += int(timeout)
        direct_episodes += int(direct)
        direct_successes += int(direct and success)
        final_distance = float(final_info.get("dist", float("inf")))
        if np.isfinite(final_distance):
            final_distances.append(final_distance)
        print(
            "high_bc_gate_episode={} label={} scenario_id={} success={} collision={} "
            "timeout={} final_distance={:.4f}".format(
                episode_index + 1,
                label,
                scenario.get("scenario_id"),
                success,
                collision,
                timeout,
                final_distance,
            )
        )
    metrics = {
        "episodes": int(episode_count),
        "successes": int(successes),
        "success_rate": float(successes) / float(max(episode_count, 1)),
        "collisions": int(collisions),
        "timeouts": int(timeouts),
        "direct_episodes": int(direct_episodes),
        "direct_successes": int(direct_successes),
        "direct_success_rate": float(direct_successes)
        / float(max(direct_episodes, 1)),
        "mean_final_distance": float(np.mean(final_distances))
        if final_distances else float("inf"),
        "low_steps": int(low_steps),
        "safety_steps": int(safety_steps),
        "safety_rate": float(safety_steps) / float(max(low_steps, 1)),
    }
    print("high_bc_closed_loop label={} {}".format(label, metrics))
    return metrics


def _clone_actor_state(model):
    return dict(
        (name, value.detach().cpu().clone())
        for name, value in model.actor_state_dict().items()
    )


def _closed_loop_selection_key(metrics):
    """Rank checkpoints by task outcome, not open-loop imitation loss."""
    final_distance = float(metrics.get("mean_final_distance", float("inf")))
    if not np.isfinite(final_distance):
        final_distance = float("inf")
    return (
        int(metrics.get("successes", 0)),
        int(metrics.get("direct_successes", 0)),
        -int(metrics.get("collisions", 0)),
        -int(metrics.get("timeouts", 0)),
        -final_distance,
        -float(metrics.get("safety_rate", 1.0)),
    )


def _closed_loop_gate_pass(metrics, args):
    return bool(
        metrics["success_rate"] >= args.bc_min_success_rate
        and metrics["collisions"] <= args.bc_max_collisions
        and metrics["direct_success_rate"]
        >= args.bc_min_direct_success_rate
    )


def _ppo_update(
        model,
        optimizer,
        buffer,
        device,
        args,
        teacher_coefficient,
        subgoal_types,
        teacher_subgoal_types,
        route_sides,
        teacher_route_sides,
        reference_model,
        actor_frozen=False):
    data = buffer.tensors(device)
    advantages = data["advantages"]
    data["advantages"] = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)
    weights = torch.as_tensor(
        args.teacher_action_weights,
        dtype=torch.float32,
        device=device,
    ).view(1, -1)
    subgoal_types = torch.as_tensor(
        subgoal_types,
        dtype=torch.int64,
        device=device,
    )
    if tuple(subgoal_types.shape) != (buffer.size,):
        raise ValueError("rollout subgoal types must match the buffer")
    teacher_subgoal_types = torch.as_tensor(
        teacher_subgoal_types,
        dtype=torch.int64,
        device=device,
    )
    if tuple(teacher_subgoal_types.shape) != (buffer.size,):
        raise ValueError("teacher subgoal types must match the buffer")
    route_sides = torch.as_tensor(
        route_sides,
        dtype=torch.int64,
        device=device,
    )
    teacher_route_sides = torch.as_tensor(
        teacher_route_sides,
        dtype=torch.int64,
        device=device,
    )
    if tuple(route_sides.shape) != (buffer.size,):
        raise ValueError("rollout route sides must match the buffer")
    if tuple(teacher_route_sides.shape) != (buffer.size,):
        raise ValueError("teacher route sides must match the buffer")
    teacher_option_weights = torch.as_tensor(
        args.teacher_option_weights,
        dtype=torch.float32,
        device=device,
    )
    totals = collections.Counter()
    count = 0
    for _ in range(args.ppo_epochs):
        for indices_np in buffer.mini_batch_indices(args.batch_size):
            indices = torch.from_numpy(indices_np.astype(np.int64)).to(device)
            observations = data["observations"].index_select(0, indices)
            actions = data["actions"].index_select(0, indices)
            teachers = data["teacher_actions"].index_select(0, indices)
            selected_types = subgoal_types.index_select(0, indices)
            selected_routes = route_sides.index_select(0, indices)
            teacher_type_target = teacher_subgoal_types.index_select(
                0, indices
            )
            teacher_route_target = teacher_route_sides.index_select(
                0, indices
            )
            teacher_option_target = model.encode_options(
                teacher_type_target, teacher_route_target
            )
            old_logp = data["old_log_probabilities"].index_select(0, indices)
            old_values = data["old_values"].index_select(0, indices)
            returns = data["returns"].index_select(0, indices)
            batch_advantages = data["advantages"].index_select(0, indices)
            (
                logp,
                entropy,
                values,
                mean_action,
                type_logits,
                route_logits,
            ) = model.evaluate_actions(
                observations,
                actions,
                selected_types,
                selected_routes,
            )
            ratio = torch.exp(logp - old_logp)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
            )
            policy_loss = -torch.min(
                ratio * batch_advantages,
                clipped_ratio * batch_advantages,
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
            teacher_prediction = model.deterministic_action(
                observations,
                teacher_type_target,
                teacher_route_target,
            )
            teacher_loss = (
                (teacher_prediction - teachers).pow(2) * weights
            ).mean()
            teacher_type_loss = F.cross_entropy(
                type_logits,
                teacher_type_target,
            )
            teacher_type_accuracy = (
                torch.argmax(type_logits, dim=-1) == teacher_type_target
            ).float().mean()
            teacher_route_loss = F.cross_entropy(
                route_logits,
                teacher_route_target,
            )
            teacher_route_accuracy = (
                torch.argmax(route_logits, dim=-1) == teacher_route_target
            ).float().mean()
            option_logits = model.option_logits(observations)
            teacher_option_loss = F.cross_entropy(
                option_logits,
                teacher_option_target,
                weight=teacher_option_weights,
            )
            teacher_option_accuracy = (
                torch.argmax(option_logits, dim=-1)
                == teacher_option_target
            ).float().mean()
            with torch.no_grad():
                (
                    reference_action,
                    reference_type,
                    reference_route,
                ) = reference_model.deterministic_decision(observations)
            reference_prediction = model.deterministic_action(
                observations,
                reference_type,
                reference_route,
            )
            reference_loss = (
                (reference_prediction - reference_action).pow(2) * weights
            ).mean()
            reference_option = model.encode_options(
                reference_type, reference_route
            )
            reference_option_loss = F.cross_entropy(
                option_logits, reference_option
            )
            reference_type_loss = F.cross_entropy(
                type_logits, reference_type
            )
            reference_route_loss = F.cross_entropy(
                route_logits, reference_route
            )
            if actor_frozen:
                loss = args.value_coefficient * value_loss
            else:
                loss = (
                    policy_loss
                    + args.value_coefficient * value_loss
                    - args.entropy_coefficient * entropy.mean()
                    + float(teacher_coefficient) * (
                        teacher_loss
                        + args.teacher_option_coefficient
                        * teacher_option_loss
                    )
                    + args.reference_coefficient * (
                        reference_loss
                        + args.reference_discrete_coefficient
                        * reference_option_loss
                    )
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.maximum_gradient_norm
            )
            optimizer.step()
            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["teacher_loss"] += float(teacher_loss.item())
            totals["teacher_type_loss"] += float(
                teacher_type_loss.item()
            )
            totals["teacher_type_accuracy"] += float(
                teacher_type_accuracy.item()
            )
            totals["teacher_route_loss"] += float(
                teacher_route_loss.item()
            )
            totals["teacher_route_accuracy"] += float(
                teacher_route_accuracy.item()
            )
            totals["teacher_option_loss"] += float(
                teacher_option_loss.item()
            )
            totals["teacher_option_accuracy"] += float(
                teacher_option_accuracy.item()
            )
            totals["reference_loss"] += float(reference_loss.item())
            totals["entropy"] += float(entropy.mean().item())
            totals["action_abs"] += float(mean_action.abs().mean().item())
            count += 1
    metrics = dict(
        (name, value / float(max(count, 1)))
        for name, value in totals.items()
    )
    metrics["actor_frozen"] = bool(actor_frozen)
    return metrics


def _reset_environment(client, sampler, default_budget):
    scenario = sampler.next()
    budget = scenario_step_budget(scenario, default_budget)
    observation, unused_info = client.reset_with_info(
        scenario=scenario,
        max_steps=budget,
    )
    return observation, scenario


def _bootstrap(model, normalizer, observation, done, device):
    if done:
        return 0.0
    tensor = torch.from_numpy(
        normalizer.normalize(observation)
    ).to(device).unsqueeze(0)
    with torch.no_grad():
        return float(model.get_value(tensor).item())


def _scenario_category(scenario):
    category = str(scenario.get("category", "unknown"))
    if bool(scenario.get("no_obstacle", False)):
        return "direct"
    side = float(scenario.get("detour_side", 0.0))
    return "{}_{}".format(category, "upper" if side >= 0.0 else "lower")


def _episode_rates(episodes, successes, collisions, timeouts):
    return dict(
        (
            category,
            {
                "episodes": int(count),
                "success": round(
                    float(successes[category]) / float(max(count, 1)), 3
                ),
                "collision": round(
                    float(collisions[category]) / float(max(count, 1)), 3
                ),
                "timeout": round(
                    float(timeouts[category]) / float(max(count, 1)), 3
                ),
            },
        )
        for category, count in sorted(episodes.items())
    )


def _linear_coefficient(start, end, step, decay_steps):
    if decay_steps <= 0:
        return float(end)
    fraction = min(max(float(step) / float(decay_steps), 0.0), 1.0)
    return float(start) + fraction * (float(end) - float(start))


def _validate_checkpoint(checkpoint, model):
    if int(checkpoint.get("observation_dim", -1)) != model.OBS_DIM:
        raise ValueError("high checkpoint observation dimension mismatch")
    if int(checkpoint.get("action_dim", -1)) != model.ACTION_DIM:
        raise ValueError("high checkpoint action dimension mismatch")
    if int(checkpoint.get("subgoal_type_count", -1)) != (
            model.SUBGOAL_TYPE_COUNT):
        raise ValueError("high checkpoint subgoal-type dimension mismatch")
    if int(checkpoint.get("route_side_count", -1)) != model.ROUTE_SIDE_COUNT:
        raise ValueError("high checkpoint route-side dimension mismatch")
    if int(checkpoint.get("option_count", -1)) != model.OPTION_COUNT:
        raise ValueError("high checkpoint unified-option dimension mismatch")
    if str(checkpoint.get("policy_type")) != model.POLICY_TYPE:
        raise ValueError("checkpoint is not a fused high-level policy")


def _numbered_checkpoint_path(path, update):
    expanded = os.path.abspath(os.path.expanduser(path))
    root, extension = os.path.splitext(expanded)
    if not extension:
        extension = ".pt"
    return "{}.update_{:04d}{}".format(root, int(update), extension)


def _warmstart_checkpoint_path(path):
    expanded = os.path.abspath(os.path.expanduser(path))
    root, extension = os.path.splitext(expanded)
    if not extension:
        extension = ".pt"
    return "{}.bc{}".format(root, extension)


def _save_checkpoint(
        path,
        model,
        optimizer,
        normalizer,
        total_steps,
        total_low_steps,
        update_index,
        args,
        warmstart_metrics,
        environment_metadata,
        episode_counts,
        success_counts,
        reference_model=None):
    directory = os.path.dirname(os.path.abspath(os.path.expanduser(path)))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    reference_actor_state = (
        model.actor_state_dict()
        if reference_model is None
        else reference_model.actor_state_dict()
    )
    torch.save({
        "format_version": 3,
        "policy_type": model.POLICY_TYPE,
        "model": model.state_dict(),
        "reference_actor_state": reference_actor_state,
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "observation_dim": model.OBS_DIM,
        "action_dim": model.ACTION_DIM,
        "subgoal_type_count": model.SUBGOAL_TYPE_COUNT,
        "route_side_count": model.ROUTE_SIDE_COUNT,
        "option_count": model.OPTION_COUNT,
        "total_steps": int(total_steps),
        "total_low_steps": int(total_low_steps),
        "update_index": int(update_index),
        "arguments": vars(args),
        "warmstart_metrics": dict(warmstart_metrics),
        "environment_metadata": dict(environment_metadata),
        "training_scenario_episodes": dict(episode_counts),
        "training_scenario_successes": dict(success_counts),
    }, path)


def _seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5565)
    parser.add_argument("--socket-timeout", type=float, default=180.0)
    parser.add_argument(
        "--scenario-set",
        default=(
            "src/mobile_arm_rl_env/config/"
            "fused_box_subgoal_randomized_scenarios.json"
        ),
    )
    parser.add_argument(
        "--scenario-split",
        choices=["all", "train", "validation", "test"],
        default="train",
    )
    parser.add_argument("--scenario-validation-fraction", type=float, default=0.10)
    parser.add_argument("--scenario-test-fraction", type=float, default=0.20)
    parser.add_argument("--scenario-split-seed", type=int, default=123)
    parser.add_argument("--scenario-step-budget", type=int, default=0)
    parser.add_argument(
        "--bc-gate-split",
        choices=["validation", "test"],
        default="validation",
    )
    parser.add_argument(
        "--output",
        default="/tmp/mobile_arm_rl_training/fused_high_ppo_seed789.pt",
    )
    parser.add_argument("--resume")
    parser.add_argument("--bc-checkpoint")
    parser.add_argument("--teacher-warmup-steps", type=int, default=4096)
    parser.add_argument("--teacher-warmup-max-episodes", type=int, default=200)
    parser.add_argument(
        "--teacher-warmup-episodes",
        type=int,
        default=0,
        help="minimum complete teacher episodes collected before BC",
    )
    parser.add_argument("--warmstart-dataset")
    parser.add_argument("--warmstart-only", action="store_true")
    parser.add_argument("--warmstart-epochs", type=int, default=100)
    parser.add_argument("--warmstart-batch-size", type=int, default=256)
    parser.add_argument("--warmstart-learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmstart-type-coefficient", type=float, default=0.5)
    parser.add_argument("--warmstart-route-coefficient", type=float, default=0.25)
    parser.add_argument(
        "--warmstart-option-coefficient", type=float, default=0.75
    )
    parser.add_argument("--warmstart-validation-fraction", type=float, default=0.20)
    parser.add_argument("--dagger-rounds", type=int, default=5)
    parser.add_argument(
        "--dagger-pure-student-rounds",
        type=int,
        default=2,
        help="final DAgger rounds executed without teacher takeover",
    )
    parser.add_argument("--dagger-episodes-per-round", type=int, default=20)
    parser.add_argument("--dagger-epochs", type=int, default=50)
    parser.add_argument(
        "--dagger-selection-episodes",
        type=int,
        default=10,
        help="deterministic held-out episodes used to select each DAgger actor",
    )
    parser.add_argument(
        "--dagger-initial-teacher-probability", type=float, default=0.70
    )
    parser.add_argument(
        "--dagger-final-teacher-probability", type=float, default=0.20
    )
    parser.add_argument("--dagger-intervention-steps", type=int, default=3)
    parser.add_argument(
        "--dagger-recovery-stable-steps",
        type=int,
        default=5,
        help="risk-free high decisions required before teacher releases",
    )
    parser.add_argument(
        "--dagger-max-samples-per-episode",
        type=int,
        default=200,
        help="temporal/risk-aware cap for each aggregated DAgger episode",
    )
    parser.add_argument(
        "--dagger-pre-intervention-steps", type=int, default=5
    )
    parser.add_argument(
        "--dagger-intervention-weight", type=float, default=8.0
    )
    parser.add_argument(
        "--dagger-risk-sample-weight", type=float, default=5.0
    )
    parser.add_argument(
        "--dagger-failed-sample-weight",
        type=float,
        default=0.50,
        help="maximum label weight retained from an unsuccessful episode",
    )
    parser.add_argument(
        "--dagger-action-deviation-threshold", type=float, default=0.20
    )
    parser.add_argument(
        "--dagger-safety-rate-threshold", type=float, default=0.10
    )
    parser.add_argument(
        "--dagger-minimum-progress", type=float, default=0.005
    )
    parser.add_argument(
        "--dagger-progress-stall-steps", type=int, default=3
    )
    parser.add_argument("--bc-gate-episodes", type=int, default=10)
    parser.add_argument("--bc-min-success-rate", type=float, default=0.70)
    parser.add_argument("--bc-max-collisions", type=int, default=1)
    parser.add_argument(
        "--bc-min-direct-success-rate", type=float, default=0.90
    )
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--maximum-rollout-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.10)
    parser.add_argument("--value-coefficient", type=float, default=0.50)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0001)
    parser.add_argument("--teacher-coefficient", type=float, default=2.0)
    parser.add_argument("--teacher-final-coefficient", type=float, default=0.20)
    parser.add_argument("--teacher-type-coefficient", type=float, default=0.5)
    parser.add_argument("--teacher-route-coefficient", type=float, default=0.25)
    parser.add_argument(
        "--teacher-option-coefficient", type=float, default=0.75
    )
    parser.add_argument(
        "--teacher-subgoal-type-weights",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 4.0],
    )
    parser.add_argument("--teacher-decay-steps", type=int, default=30000)
    parser.add_argument(
        "--teacher-action-weights",
        type=float,
        nargs=6,
        default=[2.0, 2.0, 4.0, 1.0, 1.0, 1.0],
    )
    parser.add_argument(
        "--teacher-route-side-weights",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
    )
    parser.add_argument(
        "--teacher-option-weights",
        type=float,
        nargs=4,
        default=[1.0, 1.0, 1.0, 4.0],
    )
    parser.add_argument("--reference-coefficient", type=float, default=1.0)
    parser.add_argument(
        "--reference-discrete-coefficient", type=float, default=0.25
    )
    parser.add_argument("--critic-warmup-steps", type=int, default=2048)
    parser.add_argument("--maximum-gradient-norm", type=float, default=0.50)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--initial-log-std", type=float, default=-3.5)
    parser.add_argument("--freeze-log-std", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--reward-window", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--deterministic-rollout", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.resume and args.bc_checkpoint:
        parser.error("--resume and --bc-checkpoint are mutually exclusive")
    if args.total_steps <= 0:
        parser.error("--total-steps must be positive")
    if args.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive")
    if args.maximum_rollout_steps < args.rollout_steps:
        parser.error("--maximum-rollout-steps must be at least rollout-steps")
    if not 0.0 < args.warmstart_validation_fraction < 1.0:
        parser.error("--warmstart-validation-fraction must be in (0, 1)")
    if args.teacher_decay_steps < 0 or args.critic_warmup_steps < 0:
        parser.error("decay and warm-up steps must be non-negative")
    if args.teacher_warmup_episodes < 0:
        parser.error("--teacher-warmup-episodes must be non-negative")
    if args.teacher_warmup_max_episodes <= 0:
        parser.error("--teacher-warmup-max-episodes must be positive")
    if args.dagger_rounds < 0 or args.dagger_episodes_per_round <= 0:
        parser.error("invalid DAgger round configuration")
    if not 0 <= args.dagger_pure_student_rounds <= args.dagger_rounds:
        parser.error(
            "--dagger-pure-student-rounds must be between 0 and dagger-rounds"
        )
    if args.dagger_epochs <= 0:
        parser.error("--dagger-epochs must be positive")
    if args.dagger_selection_episodes <= 0:
        parser.error("--dagger-selection-episodes must be positive")
    if not (
            0.0 <= args.dagger_final_teacher_probability <= 1.0
            and 0.0 <= args.dagger_initial_teacher_probability <= 1.0):
        parser.error("DAgger teacher probabilities must be in [0, 1]")
    if (
            args.dagger_intervention_steps <= 0
            or args.dagger_recovery_stable_steps <= 0
            or args.dagger_max_samples_per_episode <= 0
            or args.dagger_pre_intervention_steps < 0
            or args.dagger_progress_stall_steps <= 0):
        parser.error("invalid DAgger intervention/stall configuration")
    if min(
            args.dagger_intervention_weight,
            args.dagger_risk_sample_weight,
            args.dagger_failed_sample_weight,
        ) <= 0.0:
        parser.error("DAgger sample weights must be positive")
    if args.dagger_action_deviation_threshold < 0.0:
        parser.error("--dagger-action-deviation-threshold must be non-negative")
    if not 0.0 <= args.dagger_safety_rate_threshold <= 1.0:
        parser.error("--dagger-safety-rate-threshold must be in [0, 1]")
    if args.dagger_minimum_progress < 0.0:
        parser.error("--dagger-minimum-progress must be non-negative")
    if args.bc_gate_episodes <= 0:
        parser.error("--bc-gate-episodes must be positive")
    if not 0.0 <= args.bc_min_success_rate <= 1.0:
        parser.error("--bc-min-success-rate must be in [0, 1]")
    if not 0.0 <= args.bc_min_direct_success_rate <= 1.0:
        parser.error("--bc-min-direct-success-rate must be in [0, 1]")
    if min(args.teacher_subgoal_type_weights) <= 0.0:
        parser.error("--teacher-subgoal-type-weights must be positive")
    if min(args.teacher_route_side_weights) <= 0.0:
        parser.error("--teacher-route-side-weights must be positive")
    if min(args.teacher_option_weights) <= 0.0:
        parser.error("--teacher-option-weights must be positive")
    if args.warmstart_option_coefficient <= 0.0:
        parser.error("--warmstart-option-coefficient must be positive")
    if args.teacher_option_coefficient <= 0.0:
        parser.error("--teacher-option-coefficient must be positive")
    return args


if __name__ == "__main__":
    main()
