#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train the minimal high-level PPO over a frozen low-level PPO.

The high action contains exactly two parts: one of three semantic stages and
one six-dimensional relative subgoal.  A guarded teacher warm-start prevents
blind random exploration.  Closed-loop KL-DAgger then labels states induced
by the student before PPO is allowed to start.  There is no route output,
route latch, graph, diffuser or reachability model in the learned hierarchy.
"""

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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_basic_high_actor_critic import (  # noqa: E402
    FusedBasicHighActorCritic,
)
from training.fused_basic_high_env_client import (  # noqa: E402
    FusedBasicHighEnvironmentClient,
)
from training.fused_reach_scenarios import (  # noqa: E402
    FusedReachScenarioSampler,
    load_fused_reach_scenarios,
    scenario_category_counts,
    split_fused_reach_scenarios,
)
from training.hrl4in_low_actor_critic import (  # noqa: E402
    RunningObservationNormalizer,
)
from training.ppo_rollout import PPORolloutBuffer  # noqa: E402


DAGGER_STAGE_NAMES = ("DIRECT", "DETOUR", "TERMINAL")
TERMINAL_STAGE = 2
ARM_SUBGOAL_SLICE = slice(3, 6)


def main():
    args = _parse_arguments()
    terminal_blend_curriculum = _new_terminal_blend_curriculum(args)
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
    scenarios = _basic_scenarios(scenarios, args.fixed_detour_side)
    sampler = FusedReachScenarioSampler(
        scenarios, seed=args.seed, shuffle=True
    )
    validation_scenarios = split_fused_reach_scenarios(
        scenario_set["scenarios"],
        split="validation",
        validation_fraction=args.scenario_validation_fraction,
        test_fraction=args.scenario_test_fraction,
        seed=args.scenario_split_seed,
    )
    validation_scenarios = _basic_scenarios(
        validation_scenarios, args.fixed_detour_side
    )
    gate_sampler = FusedReachScenarioSampler(
        validation_scenarios,
        seed=args.scenario_split_seed,
        shuffle=False,
    )
    step_budget = (
        int(args.scenario_step_budget)
        if args.scenario_step_budget > 0
        else int(scenario_set["recommended_step_budget"])
    )
    print(
        "basic_high_curriculum path={} split={} scenarios={} budget={} "
        "categories={}".format(
            args.scenario_set,
            args.scenario_split,
            len(scenarios),
            step_budget,
            scenario_category_counts(scenarios),
        )
    )
    print(
        "basic_high_kl_contract teacher=forward_kl_on_student_states "
        "teacher_std={:.4f} reference=frozen_bc_forward_kl "
        "reference_target={:.4f} policy_target={:.4f} "
        "policy_guard=full_rollout_after_actor_minibatch "
        "critic_updates=independent".format(
            args.teacher_kl_std,
            args.reference_kl_target,
            args.target_policy_kl,
        )
    )
    print(
        "basic_high_dagger_contract rounds={} episodes_per_round={} "
        "teacher_probability={:.3f}->{:.3f} "
        "risk_takeover=stage_disagreement_or_terminal_deviation "
        "base_action_deviation_takeover={} "
        "stage_thresholds={} terminal_arm_threshold={:.3f} "
        "skip_pre_dagger_gate={} dataset={}".format(
            args.dagger_rounds,
            args.dagger_episodes,
            args.dagger_initial_teacher_probability,
            args.dagger_final_teacher_probability,
            bool(args.dagger_base_action_deviation_takeover),
            dict(zip(
                DAGGER_STAGE_NAMES,
                _dagger_action_deviation_thresholds(args),
            )),
            args.dagger_terminal_arm_deviation_threshold,
            bool(args.skip_pre_dagger_gate),
            args.dagger_dataset_output,
        )
    )
    print(
        "basic_high_ppo_sampling stage_balanced_advantages={} "
        "terminal_teacher_weight={:.3f} "
        "minibatches=balanced_complete_rollout "
        "actor_critic_phases=separate "
        "value_clip_range={:.6f} value_clip_enabled={}".format(
            bool(args.stage_balanced_advantages),
            args.teacher_terminal_weight,
            _resolved_value_clip_range(args),
            bool(_resolved_value_clip_range(args) > 0.0),
        )
    )
    print(
        "basic_high_checkpoint_contract recovery_interval={} recovery={} "
        "numbered_selection_interval={}".format(
            args.recovery_checkpoint_interval,
            _recovery_checkpoint_path(args.output),
            args.checkpoint_interval,
        )
    )

    model = FusedBasicHighActorCritic(
        hidden_sizes=args.hidden_sizes,
        initial_log_std=args.initial_log_std,
    ).to(device)
    normalizer = RunningObservationNormalizer(
        observation_dim=model.OBS_DIM,
        normalized_dim=model.NORMALIZED_DIM,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )
    total_steps = 0
    total_low_steps = 0
    update_index = 0
    reference_model = None
    source_discount_mode = None
    reference_kl_coefficient = float(args.reference_kl_coefficient)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        source_discount_mode = _checkpoint_discount_mode(checkpoint)
        _validate_checkpoint(checkpoint, model)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        normalizer.load_state_dict(checkpoint["normalizer"])
        reference_model = copy.deepcopy(model).to(device)
        reference_model.load_state_dict(
            checkpoint.get("reference_model", checkpoint["model"])
        )
        _freeze_reference(reference_model)
        reference_kl_coefficient = float(checkpoint.get(
            "reference_kl_coefficient", args.reference_kl_coefficient
        ))
        total_steps = int(checkpoint.get("total_steps", 0))
        total_low_steps = int(checkpoint.get("total_low_steps", 0))
        update_index = int(checkpoint.get("update_index", 0))
        random_state_restored = _restore_random_state(
            checkpoint.get("random_state")
        )
        terminal_blend_curriculum = _restore_terminal_blend_curriculum(
            checkpoint.get("terminal_blend_curriculum"), args
        )
        print(
            "resumed basic high checkpoint={} steps={} update={} "
            "reference_kl_coef={:.6f} random_state_restored={}".format(
                args.resume,
                total_steps,
                update_index,
                reference_kl_coefficient,
                random_state_restored,
            )
        )
    elif args.ppo_initial_checkpoint:
        checkpoint = torch.load(
            args.ppo_initial_checkpoint, map_location=device
        )
        source_discount_mode = _checkpoint_discount_mode(checkpoint)
        _initialize_ppo_from_checkpoint(
            checkpoint, model, normalizer
        )
        reference_model = copy.deepcopy(model).to(device)
        _freeze_reference(reference_model)
        reference_kl_coefficient = float(
            args.reference_kl_coefficient
        )
        print(
            "initialized basic high PPO policy={} actor=checkpoint "
            "critic=new optimizer=new steps=0 update=0 "
            "reference=frozen_initial_policy reference_kl_coef={:.6f}".format(
                args.ppo_initial_checkpoint,
                reference_kl_coefficient,
            )
        )

    client = FusedBasicHighEnvironmentClient(
        host=args.host, port=args.port, timeout=args.socket_timeout
    )
    discount_reference_low_steps = _discount_reference_low_steps(
        args, client.metadata
    )
    low_step_gamma = _low_step_gamma(
        args.gamma,
        args.discount_mode,
        discount_reference_low_steps,
    )
    print(
        "basic_high_discount_contract mode={} option_reward={} "
        "gamma_per_reference={:.8f} reference_low_steps={:.3f} "
        "low_step_gamma={:.10f} gae_lambda_per_option={:.6f} "
        "duration_source=info.low_steps source_checkpoint_mode={}".format(
            args.discount_mode,
            "environment_aggregate_option_reward",
            args.gamma,
            discount_reference_low_steps,
            low_step_gamma,
            args.gae_lambda,
            source_discount_mode,
        )
    )
    active_terminal_blend = _active_terminal_blend(
        terminal_blend_curriculum
    )
    client.set_terminal_student_blend(active_terminal_blend)
    print(
        "basic_high_terminal_blend_curriculum enabled={} schedule={} "
        "stage_index={} active_blend={} gate_success_rate={:.3f} "
        "maximum_collisions={} required_gates={} completed={}".format(
            bool(terminal_blend_curriculum["enabled"]),
            terminal_blend_curriculum["schedule"],
            terminal_blend_curriculum["stage_index"],
            active_terminal_blend,
            args.terminal_blend_minimum_success_rate,
            args.terminal_blend_maximum_collisions,
            args.terminal_blend_required_gates,
            bool(terminal_blend_curriculum["completed"]),
        )
    )
    print(
        "basic_high_diagnostics_contract={}".format(
            client.metadata.get("diagnostics_contract", "unavailable")
        )
    )
    buffer = PPORolloutBuffer(
        capacity=args.maximum_rollout_steps,
        observation_dim=model.OBS_DIM,
        action_dim=model.ACTION_DIM,
    )
    rollout_stages = []
    rollout_teacher_stages = []
    episode_counts = collections.Counter()
    success_counts = collections.Counter()
    collision_counts = collections.Counter()
    timeout_counts = collections.Counter()
    stage_counts = collections.Counter()
    terminal_contract_counts = collections.Counter()
    reward_term_sums = collections.Counter()
    rollout_requested_stage_counts = collections.Counter()
    rollout_executed_stage_counts = collections.Counter()
    rollout_option_termination_counts = collections.Counter()
    rollout_option_termination_by_stage = collections.Counter()
    rollout_stage_option_counts = collections.Counter()
    rollout_stage_low_steps = collections.Counter()
    rollout_stage_progress_sums = collections.Counter()
    rollout_stage_safety_steps = collections.Counter()
    rollout_stage_overlap_steps = collections.Counter()
    rollout_stage_residual_abs_sums = collections.Counter()
    rollout_stage_safety_projection_sums = collections.Counter()
    rollout_stage_base_projection_sums = collections.Counter()
    rollout_stage_ee_projection_sums = collections.Counter()
    rollout_stage_reward_term_sums = collections.Counter()
    rollout_safety_reasons = collections.Counter()
    rollout_safety_steps = 0
    rollout_overlap_steps = 0
    rollout_safety_projection_max = 0.0
    rollout_terminal_projection_sum = 0.0
    rollout_terminal_projection_max = 0.0
    rollout_terminal_student_alignment_sum = 0.0
    rollout_terminal_student_alignment_count = 0
    rollout_terminal_alignment_sum = 0.0
    rollout_terminal_alignment_count = 0
    rollout_terminal_pose_stage_counts = collections.Counter()
    reward_window = []
    start_time = time.time()
    best_gate = None
    best_gate_stage = -1

    try:
        if not args.resume and not args.ppo_initial_checkpoint:
            if args.skip_pre_dagger_gate:
                print(
                    "basic_high_teacher_gate skipped=True reason="
                    "skip_pre_dagger_gate"
                )
            else:
                teacher_gate = _evaluate_policy(
                    client,
                    None,
                    normalizer,
                    gate_sampler,
                    step_budget,
                    args.gate_episodes,
                    device,
                )
                print("basic_high_teacher_gate={}".format(teacher_gate))
                if not _gate_pass(teacher_gate, args):
                    raise RuntimeError(
                        "rule high teacher failed the prerequisite gate; "
                        "basic PPO was not started"
                    )
            completed_dagger_round = 0
            if args.resume_dagger_dataset:
                dataset, completed_dagger_round = _load_dagger_dataset(
                    args.resume_dagger_dataset
                )
                if completed_dagger_round < 0:
                    remaining_successes = max(
                        int(args.teacher_episodes)
                        - int(dataset["successful_episodes"]),
                        0,
                    )
                    remaining_attempts = max(
                        int(args.teacher_max_attempts)
                        - int(dataset["attempted_episodes"]),
                        remaining_successes,
                    )
                    if remaining_successes > 0:
                        partial_initial_dataset = dataset
                        additional_dataset = _collect_teacher_dataset(
                            client,
                            sampler,
                            step_budget,
                            remaining_successes,
                            remaining_attempts,
                            progress_callback=(
                                lambda partial, unused_attempt,
                                base_dataset=partial_initial_dataset: (
                                    _save_dagger_dataset(
                                        args.dagger_dataset_output,
                                        _merge_teacher_datasets(
                                            base_dataset, partial
                                        ),
                                        completed_dagger_round=-1,
                                        label="initial_partial",
                                    )
                                )
                            ),
                        )
                        dataset = _merge_teacher_datasets(
                            partial_initial_dataset, additional_dataset
                        )
                    completed_dagger_round = 0
                    _save_dagger_dataset(
                        args.dagger_dataset_output,
                        dataset,
                        completed_dagger_round=0,
                        label="initial",
                    )
                fit_label = "resume_round_{}".format(
                    completed_dagger_round
                )
                fit_epochs = args.warmstart_epochs
                print(
                    "basic_high_dagger_resume path={} samples={} "
                    "completed_round={}".format(
                        args.resume_dagger_dataset,
                        len(dataset["observations"]),
                        completed_dagger_round,
                    )
                )
            else:
                dataset = _collect_teacher_dataset(
                    client,
                    sampler,
                    step_budget,
                    args.teacher_episodes,
                    args.teacher_max_attempts,
                    progress_callback=lambda partial, unused_attempt: (
                        _save_dagger_dataset(
                            args.dagger_dataset_output,
                            partial,
                            completed_dagger_round=-1,
                            label="initial_partial",
                        )
                    ),
                )
                _save_dagger_dataset(
                    args.dagger_dataset_output,
                    dataset,
                    completed_dagger_round=0,
                    label="initial",
                )
                fit_label = "initial"
                fit_epochs = args.warmstart_epochs
            warmstart_optimizer = _fit_teacher_warmstart(
                model,
                normalizer,
                dataset,
                device,
                args,
                epochs=fit_epochs,
                label=fit_label,
                seed_offset=completed_dagger_round,
            )
            _save_checkpoint(
                _dagger_fit_checkpoint_path(args.output, fit_label),
                model,
                warmstart_optimizer,
                normalizer,
                0,
                int(dataset["low_steps"]),
                0,
                args,
                client.metadata,
                reference_kl_coefficient=reference_kl_coefficient,
                training_phase="kl_bc_before_dagger",
                completed_dagger_round=completed_dagger_round,
                optimizer_role="kl_bc",
            )
            if (
                    args.skip_pre_dagger_gate
                    and completed_dagger_round < args.dagger_rounds):
                student_gate = None
                print(
                    "basic_high_student_gate round={} skipped=True "
                    "next_dagger_round={}".format(
                        fit_label, completed_dagger_round + 1
                    )
                )
            else:
                student_gate = _evaluate_policy(
                    client,
                    model,
                    normalizer,
                    gate_sampler,
                    step_budget,
                    args.gate_episodes,
                    device,
                )
                print(
                    "basic_high_student_gate round={} metrics={}".format(
                        fit_label, student_gate
                    )
                )
            for dagger_round in range(
                    completed_dagger_round + 1, args.dagger_rounds + 1):
                if (
                        student_gate is not None
                        and _gate_pass(student_gate, args)):
                    break
                teacher_probability = _dagger_teacher_probability(
                    dagger_round,
                    args.dagger_rounds,
                    args.dagger_initial_teacher_probability,
                    args.dagger_final_teacher_probability,
                )
                dagger_dataset = _collect_dagger_dataset(
                    client,
                    model,
                    normalizer,
                    sampler,
                    step_budget,
                    args.dagger_episodes,
                    device,
                    teacher_probability,
                    _dagger_action_deviation_thresholds(args),
                    args.seed + 1009 * dagger_round,
                    dagger_round,
                    terminal_arm_deviation_threshold=(
                        args.dagger_terminal_arm_deviation_threshold
                    ),
                    base_action_deviation_takeover=(
                        args.dagger_base_action_deviation_takeover
                    ),
                    progress_callback=(
                        lambda partial, unused_episode,
                        base_dataset=dataset,
                        active_round=dagger_round: _save_dagger_dataset(
                            args.dagger_dataset_output,
                            _merge_teacher_datasets(
                                base_dataset, partial
                            ),
                            completed_dagger_round=active_round - 1,
                            label="round_{:04d}_partial".format(
                                active_round
                            ),
                        )
                    ),
                )
                dataset = _merge_teacher_datasets(
                    dataset, dagger_dataset
                )
                print(
                    "basic_high_dagger_aggregate round={} samples={} "
                    "teacher_episodes={} attempted_episodes={} "
                    "dagger_samples={} risk_interventions={}".format(
                        dagger_round,
                        len(dataset["observations"]),
                        dataset["successful_episodes"],
                        dataset["attempted_episodes"],
                        dataset.get("dagger_samples", 0),
                        dataset.get("risk_intervention_count", 0),
                    )
                )
                _save_dagger_dataset(
                    args.dagger_dataset_output,
                    dataset,
                    completed_dagger_round=dagger_round,
                    label="round_{:04d}".format(dagger_round),
                )
                warmstart_optimizer = _fit_teacher_warmstart(
                    model,
                    normalizer,
                    dataset,
                    device,
                    args,
                    epochs=args.dagger_epochs,
                    label="dagger_{}".format(dagger_round),
                    seed_offset=dagger_round,
                )
                _save_checkpoint(
                    _dagger_fit_checkpoint_path(
                        args.output,
                        "dagger_round_{:04d}".format(dagger_round),
                    ),
                    model,
                    warmstart_optimizer,
                    normalizer,
                    0,
                    int(dataset["low_steps"]),
                    0,
                    args,
                    client.metadata,
                    reference_kl_coefficient=reference_kl_coefficient,
                    training_phase="kl_bc_after_dagger",
                    completed_dagger_round=dagger_round,
                    optimizer_role="kl_bc",
                )
                student_gate = _evaluate_policy(
                    client,
                    model,
                    normalizer,
                    gate_sampler,
                    step_budget,
                    args.gate_episodes,
                    device,
                )
                print(
                    "basic_high_student_gate round=dagger_{} "
                    "metrics={}".format(dagger_round, student_gate)
                )
            reference_model = copy.deepcopy(model).to(device)
            _freeze_reference(reference_model)
            _save_checkpoint(
                _warmstart_checkpoint_path(args.output),
                model,
                optimizer,
                normalizer,
                0,
                int(dataset["low_steps"]),
                0,
                args,
                client.metadata,
                reference_model=reference_model,
                reference_kl_coefficient=reference_kl_coefficient,
                terminal_blend_curriculum=terminal_blend_curriculum,
            )
            if student_gate is None or not _gate_pass(student_gate, args):
                raise RuntimeError(
                    "basic high student failed the closed-loop gate; "
                    "PPO was not started"
                )
            best_gate = student_gate
            best_gate_stage = int(
                terminal_blend_curriculum["stage_index"]
            )
            # The selected output starts as the gate-passing BC policy.  PPO
            # can replace it only after a strictly better deterministic gate,
            # so a later unstable update cannot erase the working baseline.
            _save_checkpoint(
                args.output,
                model,
                optimizer,
                normalizer,
                0,
                int(dataset["low_steps"]),
                0,
                args,
                client.metadata,
                reference_model=reference_model,
                reference_kl_coefficient=reference_kl_coefficient,
                terminal_blend_curriculum=terminal_blend_curriculum,
            )

        if reference_model is None:
            raise RuntimeError("basic high reference policy was not initialized")

        observation, scenario = _reset(client, sampler, step_budget)
        last_done = True
        training_episode_index = 0
        episode_reward = 0.0
        episode_high_steps = 0
        episode_low_steps = 0
        episode_requested_stages = collections.Counter()
        episode_executed_stages = collections.Counter()
        episode_terminations = collections.Counter()
        episode_reward_terms = collections.Counter()
        episode_safety_reasons = collections.Counter()
        episode_safety_steps = 0
        episode_overlap_steps = 0
        episode_residual_abs_sum = 0.0
        episode_safety_projection_sum = 0.0
        episode_safety_projection_max = 0.0
        episode_base_projection_sum = 0.0
        episode_ee_projection_sum = 0.0
        episode_terminal_steps = 0
        episode_terminal_projection_sum = 0.0
        episode_terminal_projection_max = 0.0
        episode_terminal_student_alignment_sum = 0.0
        episode_terminal_student_alignment_count = 0
        episode_terminal_alignment_sum = 0.0
        episode_terminal_alignment_count = 0
        episode_terminal_pose_stage_counts = collections.Counter()
        episode_terminal_forced = 0
        episode_terminal_fallbacks = 0
        episode_terminal_norm_clips = 0
        episode_invalid_terminal = 0
        while total_steps < args.total_steps or buffer.size > 0:
            if args.update_normalizer:
                normalizer.update(observation)
            normalized = normalizer.normalize(observation)
            observation_tensor = torch.from_numpy(normalized).to(
                device
            ).unsqueeze(0)
            with torch.no_grad():
                action_tensor, stage_tensor, logp_tensor, value_tensor = (
                    model.act(observation_tensor)
                )
            action = action_tensor.squeeze(0).cpu().numpy()
            stage = int(stage_tensor.item())
            teacher_action = client.teacher_action.copy()
            teacher_stage = int(client.teacher_info.get("subgoal_type", 0))
            next_observation, reward, done, info = client.step(
                action, subgoal_type=stage
            )
            option_low_steps = int(info.get("low_steps", 0))
            if option_low_steps <= 0:
                raise RuntimeError(
                    "high environment returned invalid option duration: "
                    "low_steps={}".format(option_low_steps)
                )
            buffer.add(
                observation=normalized,
                action=action,
                teacher_action=teacher_action,
                teacher_valid=True,
                log_probability=float(logp_tensor.item()),
                value=float(value_tensor.item()),
                reward=reward,
                done=done,
                duration=option_low_steps,
            )
            rollout_stages.append(stage)
            rollout_teacher_stages.append(teacher_stage)
            stage_counts[stage] += 1
            requested_stage = int(info.get(
                "requested_subgoal_type", stage
            ))
            executed_stage = int(info.get("subgoal_type", stage))
            requested_stage_name = _stage_name(requested_stage)
            executed_stage_name = _stage_name(executed_stage)
            option_termination = str(info.get(
                "option_termination", "unknown"
            ))
            option_progress = float(info.get("option_progress", 0.0))
            option_safety_steps = int(info.get("safety_steps", 0))
            option_overlap_steps = int(info.get("overlap_steps", 0))
            option_residual_abs = _mean_abs(
                info.get("low_residual_mean_abs", [])
            )
            option_safety_projection = _mean_abs(
                info.get("low_safety_projection_mean_abs", [])
            )
            option_safety_projection_max = float(info.get(
                "low_safety_projection_max", 0.0
            ))
            option_base_projection = float(info.get(
                "base_command_projection", 0.0
            ))
            option_ee_projection = float(info.get(
                "ee_command_projection", 0.0
            ))
            rollout_requested_stage_counts[requested_stage_name] += 1
            rollout_executed_stage_counts[executed_stage_name] += 1
            rollout_option_termination_counts[option_termination] += 1
            rollout_option_termination_by_stage[
                "{}:{}".format(executed_stage_name, option_termination)
            ] += 1
            rollout_stage_option_counts[executed_stage_name] += 1
            rollout_stage_low_steps[executed_stage_name] += option_low_steps
            rollout_stage_progress_sums[
                executed_stage_name
            ] += option_progress
            rollout_stage_safety_steps[
                executed_stage_name
            ] += option_safety_steps
            rollout_stage_overlap_steps[
                executed_stage_name
            ] += option_overlap_steps
            rollout_stage_residual_abs_sums[
                executed_stage_name
            ] += option_residual_abs * option_low_steps
            rollout_stage_safety_projection_sums[
                executed_stage_name
            ] += option_safety_projection * option_low_steps
            rollout_stage_base_projection_sums[
                executed_stage_name
            ] += option_base_projection
            rollout_stage_ee_projection_sums[
                executed_stage_name
            ] += option_ee_projection
            rollout_safety_steps += option_safety_steps
            rollout_overlap_steps += option_overlap_steps
            rollout_safety_projection_max = max(
                rollout_safety_projection_max,
                option_safety_projection_max,
            )
            for reason, count in info.get("safety_reasons", {}).items():
                rollout_safety_reasons[str(reason)] += int(count)
                episode_safety_reasons[str(reason)] += int(count)
            terminal_contract_counts["executed"] += int(
                executed_stage == TERMINAL_STAGE
            )
            terminal_contract_counts["latched"] += int(
                bool(info.get("terminal_option_latched", False))
            )
            terminal_contract_counts["forced"] += int(
                bool(info.get("terminal_stage_forced", False))
            )
            terminal_contract_counts["invalid"] += int(
                bool(info.get("invalid_terminal", False))
            )
            terminal_contract_counts["eligible"] += int(
                bool(info.get("terminal_eligible", False))
            )
            terminal_contract_counts["progress_fallback"] += int(
                bool(info.get("terminal_ee_progress_fallback", False))
            )
            terminal_contract_counts["norm_clipped"] += int(
                bool(info.get("terminal_ee_norm_clipped", False))
            )
            terminal_contract_counts["persistent_target"] += int(
                bool(info.get("terminal_ee_persistent_target", False))
            )
            terminal_contract_counts["stalled"] += int(
                option_termination
                in ("stalled", "safety_risk")
            )
            terminal_projection = float(info.get(
                "terminal_ee_projection", 0.0
            ))
            rollout_terminal_projection_sum += terminal_projection
            rollout_terminal_projection_max = max(
                rollout_terminal_projection_max, terminal_projection
            )
            if executed_stage == TERMINAL_STAGE:
                terminal_pose_stage = str(info.get(
                    "terminal_pose_stage", "UNKNOWN"
                ))
                rollout_terminal_pose_stage_counts[
                    terminal_pose_stage
                ] += 1
                episode_terminal_pose_stage_counts[
                    terminal_pose_stage
                ] += 1
                student_alignment = float(info.get(
                    "terminal_ee_student_alignment_cosine", 0.0
                ))
                executed_alignment = float(info.get(
                    "terminal_ee_executed_alignment_cosine",
                    info.get("terminal_ee_alignment_cosine", 0.0),
                ))
                if np.isfinite(student_alignment):
                    rollout_terminal_student_alignment_sum += (
                        student_alignment
                    )
                    rollout_terminal_student_alignment_count += 1
                    episode_terminal_student_alignment_sum += (
                        student_alignment
                    )
                    episode_terminal_student_alignment_count += 1
                if np.isfinite(executed_alignment):
                    rollout_terminal_alignment_sum += executed_alignment
                    rollout_terminal_alignment_count += 1
                    episode_terminal_alignment_sum += executed_alignment
                    episode_terminal_alignment_count += 1
            reward_terms = info.get("high_reward", {}).get("terms", {})
            for term_name in (
                    "goal_progress",
                    "final_base_progress",
                    "final_yaw_progress",
                    "terminal_goal_progress",
                    "low_step_cost",
                    "option_stall",
                    "invalid_terminal"):
                reward_term_sums[term_name] += float(
                    reward_terms.get(term_name, 0.0)
                )
                rollout_stage_reward_term_sums[
                    "{}:{}".format(executed_stage_name, term_name)
                ] += float(reward_terms.get(term_name, 0.0))
                episode_reward_terms[term_name] += float(
                    reward_terms.get(term_name, 0.0)
                )
            reward_window.append(float(reward))
            episode_reward += float(reward)
            episode_high_steps += 1
            episode_low_steps += option_low_steps
            episode_requested_stages[requested_stage_name] += 1
            episode_executed_stages[executed_stage_name] += 1
            episode_terminations[option_termination] += 1
            episode_safety_steps += option_safety_steps
            episode_overlap_steps += option_overlap_steps
            episode_residual_abs_sum += (
                option_residual_abs * option_low_steps
            )
            episode_safety_projection_sum += (
                option_safety_projection * option_low_steps
            )
            episode_safety_projection_max = max(
                episode_safety_projection_max,
                option_safety_projection_max,
            )
            episode_base_projection_sum += option_base_projection
            episode_ee_projection_sum += option_ee_projection
            episode_terminal_steps += int(
                executed_stage == TERMINAL_STAGE
            )
            episode_terminal_projection_sum += terminal_projection
            episode_terminal_projection_max = max(
                episode_terminal_projection_max, terminal_projection
            )
            episode_terminal_forced += int(bool(info.get(
                "terminal_stage_forced", False
            )))
            episode_terminal_fallbacks += int(bool(info.get(
                "terminal_ee_progress_fallback", False
            )))
            episode_terminal_norm_clips += int(bool(info.get(
                "terminal_ee_norm_clipped", False
            )))
            episode_invalid_terminal += int(bool(info.get(
                "invalid_terminal", False
            )))
            total_steps += 1
            total_low_steps += option_low_steps
            last_done = bool(done)
            if done:
                category = str(scenario.get("category", "unknown"))
                episode_counts[category] += 1
                success_counts[category] += int(bool(info.get("success", False)))
                collision_counts[category] += int(bool(info.get("collision", False)))
                timeout_counts[category] += int(bool(info.get("timeout", False)))
                training_episode_index += 1
                print(
                    "basic_high_train_episode={} scenario_id={} category={} "
                    "success={} collision={} timeout={} tf_ok={} reward={:.4f} "
                    "high_steps={} low_steps={} mean_low_steps={:.2f} "
                    "final_distance={:.4f} path_complete={} "
                    "base_position_error={:.4f} base_yaw_error={:.4f} "
                    "terminal_pose_stage={} stable_count={} "
                    "requested_stages={} executed_stages={} terminations={} "
                    "safety_steps={} safety_rate={:.4f} safety_reasons={} "
                    "overlap_rate={:.4f} residual_abs={:.6f} "
                    "safety_projection_abs={:.6f} "
                    "safety_projection_max={:.6f} "
                    "base_command_projection_mean={:.6f} "
                    "ee_command_projection_mean={:.6f} "
                    "terminal_steps={} terminal_forced={} "
                    "invalid_terminal={} terminal_fallbacks={} "
                    "terminal_norm_clips={} terminal_projection_mean={:.6f} "
                    "terminal_projection_max={:.6f} "
                    "terminal_pose_stages={} "
                    "terminal_student_alignment_mean={:.6f} "
                    "terminal_executed_alignment_mean={:.6f} "
                    "reward_terms={}".format(
                        training_episode_index,
                        scenario.get("scenario_id"),
                        category,
                        bool(info.get("success", False)),
                        bool(info.get("collision", False)),
                        bool(info.get("timeout", False)),
                        bool(info.get("tf_ok", False)),
                        episode_reward,
                        episode_high_steps,
                        episode_low_steps,
                        float(episode_low_steps) / float(max(
                            episode_high_steps, 1
                        )),
                        _info_float(info, "dist"),
                        bool(info.get(
                            "path_complete",
                            info.get("final_base_pose_aligned", False),
                        )),
                        _info_float(info, "final_base_position_error"),
                        _info_float(info, "final_base_yaw_error"),
                        info.get("terminal_pose_stage", "UNKNOWN"),
                        int(info.get("terminal_pose_stable_count", 0)),
                        dict(episode_requested_stages),
                        dict(episode_executed_stages),
                        dict(episode_terminations),
                        episode_safety_steps,
                        float(episode_safety_steps) / float(max(
                            episode_low_steps, 1
                        )),
                        dict(episode_safety_reasons),
                        float(episode_overlap_steps) / float(max(
                            episode_low_steps, 1
                        )),
                        episode_residual_abs_sum / float(max(
                            episode_low_steps, 1
                        )),
                        episode_safety_projection_sum / float(max(
                            episode_low_steps, 1
                        )),
                        episode_safety_projection_max,
                        episode_base_projection_sum / float(max(
                            episode_high_steps, 1
                        )),
                        episode_ee_projection_sum / float(max(
                            episode_high_steps, 1
                        )),
                        episode_terminal_steps,
                        episode_terminal_forced,
                        episode_invalid_terminal,
                        episode_terminal_fallbacks,
                        episode_terminal_norm_clips,
                        episode_terminal_projection_sum / float(max(
                            episode_terminal_steps, 1
                        )),
                        episode_terminal_projection_max,
                        dict(episode_terminal_pose_stage_counts),
                        episode_terminal_student_alignment_sum / float(max(
                            episode_terminal_student_alignment_count, 1
                        )),
                        episode_terminal_alignment_sum / float(max(
                            episode_terminal_alignment_count, 1
                        )),
                        dict(episode_reward_terms),
                    )
                )
                observation, scenario = _reset(
                    client, sampler, step_budget
                )
                episode_reward = 0.0
                episode_high_steps = 0
                episode_low_steps = 0
                episode_requested_stages.clear()
                episode_executed_stages.clear()
                episode_terminations.clear()
                episode_reward_terms.clear()
                episode_safety_reasons.clear()
                episode_safety_steps = 0
                episode_overlap_steps = 0
                episode_residual_abs_sum = 0.0
                episode_safety_projection_sum = 0.0
                episode_safety_projection_max = 0.0
                episode_base_projection_sum = 0.0
                episode_ee_projection_sum = 0.0
                episode_terminal_steps = 0
                episode_terminal_projection_sum = 0.0
                episode_terminal_projection_max = 0.0
                episode_terminal_student_alignment_sum = 0.0
                episode_terminal_student_alignment_count = 0
                episode_terminal_alignment_sum = 0.0
                episode_terminal_alignment_count = 0
                episode_terminal_pose_stage_counts.clear()
                episode_terminal_forced = 0
                episode_terminal_fallbacks = 0
                episode_terminal_norm_clips = 0
                episode_invalid_terminal = 0
            else:
                observation = next_observation

            ready = bool(buffer.size >= args.rollout_steps and last_done)
            if total_steps >= args.total_steps and buffer.size > 0:
                ready = True
            if buffer.full and not ready:
                raise RuntimeError(
                    "basic high rollout filled before episode boundary"
                )
            if not ready:
                continue

            last_value = _bootstrap(
                model, normalizer, observation, last_done, device
            )
            buffer.compute_returns_and_advantages(
                last_value,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                duration_discount_reference=(
                    discount_reference_low_steps
                    if args.discount_mode == "smdp"
                    else None
                ),
            )
            rollout_size = buffer.size
            discount_diagnostics = _rollout_discount_diagnostics(
                buffer,
                args.discount_mode,
                args.gamma,
                args.gae_lambda,
                discount_reference_low_steps,
            )
            reference_kl_coefficient_used = float(
                reference_kl_coefficient
            )
            metrics = _ppo_update(
                model,
                optimizer,
                buffer,
                rollout_stages,
                rollout_teacher_stages,
                device,
                args,
                _linear_coefficient(
                    args.teacher_coefficient,
                    args.teacher_final_coefficient,
                    total_steps,
                    args.teacher_decay_steps,
                ),
                reference_model,
                reference_kl_coefficient_used,
            )
            reference_kl_coefficient = _adapt_kl_coefficient(
                reference_kl_coefficient,
                metrics["reference_kl"],
                args.reference_kl_target,
                args.reference_kl_min_coefficient,
                args.reference_kl_max_coefficient,
                args.reference_kl_adaptation_factor,
                args.reference_kl_tolerance,
            )
            buffer.clear()
            rollout_stages = []
            rollout_teacher_stages = []
            update_index += 1
            if (
                    args.recovery_checkpoint_interval > 0
                    and update_index
                    % args.recovery_checkpoint_interval == 0):
                recovery_path = _recovery_checkpoint_path(args.output)
                _save_checkpoint(
                    recovery_path,
                    model,
                    optimizer,
                    normalizer,
                    total_steps,
                    total_low_steps,
                    update_index,
                    args,
                    client.metadata,
                    reference_model=reference_model,
                    reference_kl_coefficient=reference_kl_coefficient,
                    terminal_blend_curriculum=terminal_blend_curriculum,
                )
                print(
                    "basic_high_recovery_checkpoint update={} path={}".format(
                        update_index, recovery_path
                    )
                )
            mean_reward_terms = dict(
                (name, float(value) / float(max(rollout_size, 1)))
                for name, value in reward_term_sums.items()
            )
            option_diagnostics = {
                "requested_stages": dict(rollout_requested_stage_counts),
                "executed_stages": dict(rollout_executed_stage_counts),
                "terminations": dict(rollout_option_termination_counts),
                "terminations_by_stage": dict(
                    rollout_option_termination_by_stage
                ),
                "mean_low_steps_by_stage": _counter_means(
                    rollout_stage_low_steps,
                    rollout_stage_option_counts,
                ),
                "mean_progress_by_stage": _counter_means(
                    rollout_stage_progress_sums,
                    rollout_stage_option_counts,
                ),
                "safety_rate_by_stage": _counter_means(
                    rollout_stage_safety_steps,
                    rollout_stage_low_steps,
                ),
                "overlap_rate_by_stage": _counter_means(
                    rollout_stage_overlap_steps,
                    rollout_stage_low_steps,
                ),
                "mean_residual_abs_by_stage": _counter_means(
                    rollout_stage_residual_abs_sums,
                    rollout_stage_low_steps,
                ),
                "mean_safety_projection_abs_by_stage": _counter_means(
                    rollout_stage_safety_projection_sums,
                    rollout_stage_low_steps,
                ),
                "mean_base_command_projection_by_stage": _counter_means(
                    rollout_stage_base_projection_sums,
                    rollout_stage_option_counts,
                ),
                "mean_ee_command_projection_by_stage": _counter_means(
                    rollout_stage_ee_projection_sums,
                    rollout_stage_option_counts,
                ),
                "mean_reward_terms_by_stage": _stage_term_means(
                    rollout_stage_reward_term_sums,
                    rollout_stage_option_counts,
                ),
                "safety_steps": int(rollout_safety_steps),
                "safety_rate": float(rollout_safety_steps) / float(max(
                    sum(rollout_stage_low_steps.values()), 1
                )),
                "safety_reasons": dict(rollout_safety_reasons),
                "safety_projection_max": float(
                    rollout_safety_projection_max
                ),
                "overlap_steps": int(rollout_overlap_steps),
                "overlap_rate": float(rollout_overlap_steps) / float(max(
                    sum(rollout_stage_low_steps.values()), 1
                )),
            }
            terminal_diagnostics = dict(terminal_contract_counts)
            terminal_diagnostics.update({
                "pose_stage_counts": dict(
                    rollout_terminal_pose_stage_counts
                ),
                "projection_mean": (
                    rollout_terminal_projection_sum / float(max(
                        terminal_contract_counts.get("executed", 0), 1
                    ))
                ),
                "projection_max": float(rollout_terminal_projection_max),
                "student_alignment_cosine_mean": (
                    rollout_terminal_student_alignment_sum / float(max(
                        rollout_terminal_student_alignment_count, 1
                    ))
                ),
                "executed_alignment_cosine_mean": (
                    rollout_terminal_alignment_sum / float(max(
                        rollout_terminal_alignment_count, 1
                    ))
                ),
                "alignment_cosine_mean": (
                    rollout_terminal_alignment_sum / float(max(
                        rollout_terminal_alignment_count, 1
                    ))
                ),
            })
            ppo_diagnostics = {
                "epochs_entered": int(metrics["epochs_completed"]),
                "full_epochs_completed": int(
                    metrics["full_epochs_completed"]
                ),
                "minibatches_completed": int(
                    metrics["minibatches_completed"]
                ),
                "minibatches_expected": int(
                    metrics["minibatches_expected"]
                ),
                "update_fraction": float(metrics["update_fraction"]),
                "critic_full_epochs_completed": int(
                    metrics["critic_full_epochs_completed"]
                ),
                "critic_minibatches_completed": int(
                    metrics["critic_minibatches_completed"]
                ),
                "critic_minibatches_expected": int(
                    metrics["critic_minibatches_expected"]
                ),
                "critic_update_fraction": float(
                    metrics["critic_update_fraction"]
                ),
                "policy_kl_mean": float(metrics["policy_kl"]),
                "policy_kl_max": float(metrics["policy_kl_max"]),
                "policy_kl_batch_mean": float(
                    metrics["policy_kl_batch_mean"]
                ),
                "policy_kl_batch_max": float(
                    metrics["policy_kl_batch_max"]
                ),
                "policy_kl_epoch_values": list(
                    metrics["policy_kl_epoch_values"]
                ),
                "policy_kl_minibatch_values": list(
                    metrics["policy_kl_minibatch_values"]
                ),
                "policy_kl_check_count": int(
                    metrics["policy_kl_check_count"]
                ),
                "policy_kl_stop_value": float(
                    metrics["policy_kl_stop_value"]
                ),
                "policy_kl_stop_epoch": int(
                    metrics["policy_kl_stop_epoch"]
                ),
                "policy_kl_stop_minibatch": int(
                    metrics["policy_kl_stop_minibatch"]
                ),
                "policy_kl_stop_global_minibatch": int(
                    metrics["policy_kl_stop_global_minibatch"]
                ),
                "policy_kl_threshold": float(
                    metrics["policy_kl_threshold"]
                ),
                "clip_fraction": float(metrics["clip_fraction"]),
                "value_clip_fraction": float(
                    metrics["value_clip_fraction"]
                ),
                "value_target_mean": float(
                    metrics["value_target_mean"]
                ),
                "value_target_std": float(metrics["value_target_std"]),
                "value_rmse": float(metrics["value_rmse"]),
                "value_explained_variance_before": float(
                    metrics["value_explained_variance_before"]
                ),
                "value_explained_variance_after": float(
                    metrics["value_explained_variance_after"]
                ),
                "actor_grad_norm": float(metrics["actor_grad_norm"]),
                "critic_grad_norm": float(metrics["critic_grad_norm"]),
                "global_grad_norm": float(metrics["global_grad_norm"]),
                "reference_kl_coef_used": reference_kl_coefficient_used,
                "reference_kl_coef_next": float(
                    reference_kl_coefficient
                ),
                "reference_kl_update_mean": float(
                    metrics["reference_kl_update_mean"]
                ),
            }
            print(
                "basic_high_update={} high_steps={} low_steps={} rollout={} "
                "terminal_blend={} reward={:.4f} policy_loss={:.5f} "
                "value_loss={:.5f} "
                "teacher_kl={:.6f} reference_kl={:.6f} "
                "reference_kl_coef={:.6f} policy_kl={:.6f} "
                "kl_early_stop={} entropy={:.5f} action_abs={:.4f} stages={} "
                "duration_mean={:.2f} transition_discount_mean={:.6f} "
                "terminal_contract={} reward_terms={} episode_rates={} "
                "elapsed_s={:.1f}".format(
                    update_index,
                    total_steps,
                    total_low_steps,
                    rollout_size,
                    active_terminal_blend,
                    float(np.mean(reward_window[-args.reward_window:])),
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["teacher_kl"],
                    metrics["reference_kl"],
                    reference_kl_coefficient,
                    metrics["policy_kl"],
                    bool(metrics["kl_early_stop"]),
                    metrics["entropy"],
                    metrics["action_abs"],
                    dict(stage_counts),
                    discount_diagnostics["duration_mean"],
                    discount_diagnostics["transition_discount_mean"],
                    dict(terminal_contract_counts),
                    mean_reward_terms,
                    _episode_rates(
                        episode_counts,
                        success_counts,
                        collision_counts,
                        timeout_counts,
                    ),
                    time.time() - start_time,
                )
            )
            print(
                "basic_high_update_diagnostics update={} ppo={} option={} "
                "terminal={} discount={}".format(
                    update_index,
                    ppo_diagnostics,
                    option_diagnostics,
                    terminal_diagnostics,
                    discount_diagnostics,
                )
            )
            terminal_contract_counts.clear()
            reward_term_sums.clear()
            rollout_requested_stage_counts.clear()
            rollout_executed_stage_counts.clear()
            rollout_option_termination_counts.clear()
            rollout_option_termination_by_stage.clear()
            rollout_stage_option_counts.clear()
            rollout_stage_low_steps.clear()
            rollout_stage_progress_sums.clear()
            rollout_stage_safety_steps.clear()
            rollout_stage_overlap_steps.clear()
            rollout_stage_residual_abs_sums.clear()
            rollout_stage_safety_projection_sums.clear()
            rollout_stage_base_projection_sums.clear()
            rollout_stage_ee_projection_sums.clear()
            rollout_stage_reward_term_sums.clear()
            rollout_safety_reasons.clear()
            rollout_safety_steps = 0
            rollout_overlap_steps = 0
            rollout_safety_projection_max = 0.0
            rollout_terminal_projection_sum = 0.0
            rollout_terminal_projection_max = 0.0
            rollout_terminal_student_alignment_sum = 0.0
            rollout_terminal_student_alignment_count = 0
            rollout_terminal_alignment_sum = 0.0
            rollout_terminal_alignment_count = 0
            rollout_terminal_pose_stage_counts.clear()
            if (
                    args.checkpoint_interval > 0
                    and update_index % args.checkpoint_interval == 0):
                _save_checkpoint(
                    _numbered_checkpoint_path(args.output, update_index),
                    model,
                    optimizer,
                    normalizer,
                    total_steps,
                    total_low_steps,
                    update_index,
                    args,
                    client.metadata,
                    reference_model=reference_model,
                    reference_kl_coefficient=reference_kl_coefficient,
                    terminal_blend_curriculum=terminal_blend_curriculum,
                )
                gate = _evaluate_policy(
                    client,
                    model,
                    normalizer,
                    gate_sampler,
                    step_budget,
                    args.selection_episodes,
                    device,
                )
                print(
                    "basic_high_selection update={} terminal_blend={} "
                    "stage_index={} gate={}".format(
                        update_index,
                        active_terminal_blend,
                        terminal_blend_curriculum["stage_index"],
                        gate,
                    )
                )
                selection_stage = int(
                    terminal_blend_curriculum["stage_index"]
                )
                if (
                        _gate_pass(gate, args)
                        and (
                            best_gate is None
                            or selection_stage > best_gate_stage
                            or (
                                selection_stage == best_gate_stage
                                and _gate_key(gate) > _gate_key(best_gate)
                            )
                        )):
                    best_gate = gate
                    best_gate_stage = selection_stage
                    _save_checkpoint(
                        args.output,
                        model,
                        optimizer,
                        normalizer,
                        total_steps,
                        total_low_steps,
                        update_index,
                        args,
                        client.metadata,
                        reference_model=reference_model,
                        reference_kl_coefficient=reference_kl_coefficient,
                        terminal_blend_curriculum=(
                            terminal_blend_curriculum
                        ),
                    )
                    print(
                        "basic_high_best_gate stage_index={} "
                        "terminal_blend={} gate={}".format(
                            best_gate_stage,
                            active_terminal_blend,
                            best_gate,
                        )
                    )
                anneal_event = _update_terminal_blend_curriculum(
                    terminal_blend_curriculum, gate, args
                )
                print(
                    "basic_high_terminal_blend_gate update={} event={}".format(
                        update_index, anneal_event
                    )
                )
                if anneal_event["stage_completed"]:
                    milestone_curriculum = copy.deepcopy(
                        terminal_blend_curriculum
                    )
                    milestone_curriculum["stage_index"] = selection_stage
                    milestone_curriculum["active_blend"] = float(
                        anneal_event["completed_blend"]
                    )
                    milestone_curriculum["gate_streak"] = 0
                    milestone_curriculum["completed"] = bool(
                        selection_stage
                        == len(milestone_curriculum["schedule"]) - 1
                    )
                    if anneal_event["advanced"]:
                        reference_model = copy.deepcopy(model).to(device)
                        _freeze_reference(reference_model)
                        reference_kl_coefficient = float(
                            args.reference_kl_coefficient
                        )
                        active_terminal_blend = _active_terminal_blend(
                            terminal_blend_curriculum
                        )
                        client.set_terminal_student_blend(
                            active_terminal_blend
                        )
                    milestone_path = _terminal_blend_checkpoint_path(
                        args.output, anneal_event["completed_blend"]
                    )
                    _save_checkpoint(
                        milestone_path,
                        model,
                        optimizer,
                        normalizer,
                        total_steps,
                        total_low_steps,
                        update_index,
                        args,
                        client.metadata,
                        reference_model=reference_model,
                        reference_kl_coefficient=(
                            reference_kl_coefficient
                        ),
                        terminal_blend_curriculum=(
                            milestone_curriculum
                        ),
                    )
                    _save_checkpoint(
                        _recovery_checkpoint_path(args.output),
                        model,
                        optimizer,
                        normalizer,
                        total_steps,
                        total_low_steps,
                        update_index,
                        args,
                        client.metadata,
                        reference_model=reference_model,
                        reference_kl_coefficient=(
                            reference_kl_coefficient
                        ),
                        terminal_blend_curriculum=(
                            terminal_blend_curriculum
                        ),
                    )
                    print(
                        "basic_high_terminal_blend_transition update={} "
                        "completed_blend={:.3f} next_blend={} "
                        "reference_refreshed={} milestone={}".format(
                            update_index,
                            anneal_event["completed_blend"],
                            active_terminal_blend,
                            bool(anneal_event["advanced"]),
                            milestone_path,
                        )
                    )
                # Validation resets Gazebo, so the next training observation
                # must be obtained from a fresh training reset.
                observation, scenario = _reset(
                    client, sampler, step_budget
                )
                last_done = True

        final_gate = _evaluate_policy(
            client,
            model,
            normalizer,
            gate_sampler,
            step_budget,
            args.selection_episodes,
            device,
        )
        print(
            "basic_high_final_selection terminal_blend={} gate={}".format(
                active_terminal_blend, final_gate
            )
        )
        final_stage = int(terminal_blend_curriculum["stage_index"])
        if (
                _gate_pass(final_gate, args)
                and (
                    best_gate is None
                    or final_stage > best_gate_stage
                    or (
                        final_stage == best_gate_stage
                        and _gate_key(final_gate) > _gate_key(best_gate)
                    )
                )):
            best_gate = final_gate
            best_gate_stage = final_stage
            _save_checkpoint(
                args.output,
                model,
                optimizer,
                normalizer,
                total_steps,
                total_low_steps,
                update_index,
                args,
                client.metadata,
                reference_model=reference_model,
                reference_kl_coefficient=reference_kl_coefficient,
                terminal_blend_curriculum=terminal_blend_curriculum,
            )
        _save_checkpoint(
            _last_checkpoint_path(args.output),
            model,
            optimizer,
            normalizer,
            total_steps,
            total_low_steps,
            update_index,
            args,
            client.metadata,
            reference_model=reference_model,
            reference_kl_coefficient=reference_kl_coefficient,
            terminal_blend_curriculum=terminal_blend_curriculum,
        )
        print(
            "basic_high_complete best={} last={} high_steps={} low_steps={} "
            "updates={} best_gate={}".format(
                args.output,
                _last_checkpoint_path(args.output),
                total_steps,
                total_low_steps,
                update_index,
                best_gate,
            )
        )
    finally:
        client.close()


def _evaluate_policy(
        client,
        model,
        normalizer,
        sampler,
        default_budget,
        episodes,
        device):
    counts = collections.Counter()
    final_distances = []
    requested_stage_counts = collections.Counter()
    executed_stage_counts = collections.Counter()
    option_termination_counts = collections.Counter()
    stage_option_counts = collections.Counter()
    stage_low_steps = collections.Counter()
    stage_safety_steps = collections.Counter()
    stage_overlap_steps = collections.Counter()
    stage_residual_abs_sums = collections.Counter()
    stage_safety_projection_sums = collections.Counter()
    stage_base_projection_sums = collections.Counter()
    stage_ee_projection_sums = collections.Counter()
    safety_reasons = collections.Counter()
    terminal_pose_stage_counts = collections.Counter()
    terminal_student_alignment_sum = 0.0
    terminal_student_alignment_count = 0
    terminal_executed_alignment_sum = 0.0
    terminal_executed_alignment_count = 0
    for unused_episode in range(int(episodes)):
        observation, scenario = _reset(client, sampler, default_budget)
        done = False
        last_stage = -1
        last_teacher_stage = -1
        episode_high_steps = 0
        episode_low_steps = 0
        episode_safety_steps = 0
        episode_overlap_steps = 0
        episode_residual_abs_sum = 0.0
        episode_safety_projection_sum = 0.0
        episode_safety_projection_max = 0.0
        episode_base_projection_sum = 0.0
        episode_ee_projection_sum = 0.0
        episode_terminal_steps = 0
        episode_terminal_projection = 0.0
        episode_terminal_pose_stage_counts = collections.Counter()
        episode_terminal_student_alignment_sum = 0.0
        episode_terminal_student_alignment_count = 0
        episode_terminal_executed_alignment_sum = 0.0
        episode_terminal_executed_alignment_count = 0
        episode_requested_stages = collections.Counter()
        episode_executed_stages = collections.Counter()
        episode_terminations = collections.Counter()
        while not done:
            last_teacher_stage = int(
                client.teacher_info.get("subgoal_type", -1)
            )
            if model is None:
                action = client.teacher_action.copy()
                stage = max(last_teacher_stage, 0)
            else:
                normalized = normalizer.normalize(observation)
                tensor = torch.from_numpy(normalized).to(device).unsqueeze(0)
                with torch.no_grad():
                    action_tensor, stage_tensor = (
                        model.deterministic_decision(tensor)
                    )
                action = action_tensor.squeeze(0).cpu().numpy()
                stage = int(stage_tensor.item())
            last_stage = int(stage)
            observation, unused_reward, done, info = client.step(
                action, subgoal_type=stage
            )
            requested_stage = int(info.get(
                "requested_subgoal_type", stage
            ))
            executed_stage = int(info.get("subgoal_type", stage))
            termination = str(info.get(
                "option_termination", "unknown"
            ))
            option_low_steps = int(info.get("low_steps", 0))
            requested_stage_name = _stage_name(requested_stage)
            executed_stage_name = _stage_name(executed_stage)
            option_safety_steps = int(info.get("safety_steps", 0))
            option_overlap_steps = int(info.get("overlap_steps", 0))
            option_residual_abs = _mean_abs(
                info.get("low_residual_mean_abs", [])
            )
            option_safety_projection = _mean_abs(
                info.get("low_safety_projection_mean_abs", [])
            )
            option_safety_projection_max = float(info.get(
                "low_safety_projection_max", 0.0
            ))
            option_base_projection = float(info.get(
                "base_command_projection", 0.0
            ))
            option_ee_projection = float(info.get(
                "ee_command_projection", 0.0
            ))
            requested_stage_counts[requested_stage_name] += 1
            executed_stage_counts[executed_stage_name] += 1
            option_termination_counts[termination] += 1
            stage_option_counts[executed_stage_name] += 1
            stage_low_steps[executed_stage_name] += option_low_steps
            stage_safety_steps[executed_stage_name] += option_safety_steps
            stage_overlap_steps[executed_stage_name] += option_overlap_steps
            stage_residual_abs_sums[executed_stage_name] += (
                option_residual_abs * option_low_steps
            )
            stage_safety_projection_sums[executed_stage_name] += (
                option_safety_projection * option_low_steps
            )
            stage_base_projection_sums[
                executed_stage_name
            ] += option_base_projection
            stage_ee_projection_sums[
                executed_stage_name
            ] += option_ee_projection
            episode_requested_stages[requested_stage_name] += 1
            episode_executed_stages[executed_stage_name] += 1
            episode_terminations[termination] += 1
            episode_high_steps += 1
            episode_low_steps += option_low_steps
            episode_safety_steps += option_safety_steps
            episode_overlap_steps += option_overlap_steps
            episode_residual_abs_sum += (
                option_residual_abs * option_low_steps
            )
            episode_safety_projection_sum += (
                option_safety_projection * option_low_steps
            )
            episode_safety_projection_max = max(
                episode_safety_projection_max,
                option_safety_projection_max,
            )
            episode_base_projection_sum += option_base_projection
            episode_ee_projection_sum += option_ee_projection
            episode_terminal_steps += int(
                executed_stage == TERMINAL_STAGE
            )
            episode_terminal_projection += float(info.get(
                "terminal_ee_projection", 0.0
            ))
            if executed_stage == TERMINAL_STAGE:
                terminal_pose_stage = str(info.get(
                    "terminal_pose_stage", "UNKNOWN"
                ))
                terminal_pose_stage_counts[terminal_pose_stage] += 1
                episode_terminal_pose_stage_counts[
                    terminal_pose_stage
                ] += 1
                student_alignment = _info_float(
                    info,
                    "terminal_ee_student_alignment_cosine",
                )
                executed_alignment = _info_float(
                    info,
                    "terminal_ee_executed_alignment_cosine",
                    _info_float(info, "terminal_ee_alignment_cosine"),
                )
                if np.isfinite(student_alignment):
                    terminal_student_alignment_sum += student_alignment
                    terminal_student_alignment_count += 1
                    episode_terminal_student_alignment_sum += (
                        student_alignment
                    )
                    episode_terminal_student_alignment_count += 1
                if np.isfinite(executed_alignment):
                    terminal_executed_alignment_sum += executed_alignment
                    terminal_executed_alignment_count += 1
                    episode_terminal_executed_alignment_sum += (
                        executed_alignment
                    )
                    episode_terminal_executed_alignment_count += 1
            for reason, count in info.get("safety_reasons", {}).items():
                safety_reasons[str(reason)] += int(count)
        counts["episodes"] += 1
        counts["successes"] += int(bool(info.get("success", False)))
        counts["collisions"] += int(bool(info.get("collision", False)))
        counts["timeouts"] += int(bool(info.get("timeout", False)))
        counts["high_steps"] += episode_high_steps
        counts["low_steps"] += episode_low_steps
        counts["safety_steps"] += episode_safety_steps
        counts["overlap_steps"] += episode_overlap_steps
        counts["residual_abs_sum"] += episode_residual_abs_sum
        counts["safety_projection_sum"] += episode_safety_projection_sum
        counts["base_projection_sum"] += episode_base_projection_sum
        counts["ee_projection_sum"] += episode_ee_projection_sum
        counts["safety_projection_max"] = max(
            float(counts["safety_projection_max"]),
            episode_safety_projection_max,
        )
        counts["terminal_steps"] += episode_terminal_steps
        counts["terminal_projection"] += episode_terminal_projection
        final_distances.append(float(info.get("dist", float("inf"))))
        print(
            "basic_high_gate_episode={} policy={} scenario_id={} "
            "success={} collision={} timeout={} tf_ok={} "
            "final_distance={:.4f} path_complete={} "
            "base_position_error={:.4f} base_yaw_error={:.4f} "
            "terminal_pose_stage={} stable_count={} "
            "student_stage={} teacher_stage={} high_steps={} low_steps={} "
            "mean_low_steps_per_high={:.2f} terminations={} "
            "requested_stages={} executed_stages={} safety_rate={:.4f} "
            "overlap_rate={:.4f} residual_abs={:.6f} "
            "safety_projection_abs={:.6f} "
            "safety_projection_max={:.6f} "
            "base_command_projection_mean={:.6f} "
            "ee_command_projection_mean={:.6f} terminal_steps={} "
            "terminal_projection_mean={:.5f} terminal_pose_stages={} "
            "terminal_student_alignment_mean={:.5f} "
            "terminal_executed_alignment_mean={:.5f}".format(
                counts["episodes"],
                "teacher" if model is None else "student",
                scenario.get("scenario_id"),
                bool(info.get("success", False)),
                bool(info.get("collision", False)),
                bool(info.get("timeout", False)),
                bool(info.get("tf_ok", False)),
                final_distances[-1],
                bool(info.get(
                    "path_complete",
                    info.get("final_base_pose_aligned", False),
                )),
                _info_float(info, "final_base_position_error"),
                _info_float(info, "final_base_yaw_error"),
                info.get("terminal_pose_stage", "UNKNOWN"),
                int(info.get("terminal_pose_stable_count", 0)),
                last_stage,
                last_teacher_stage,
                episode_high_steps,
                episode_low_steps,
                float(episode_low_steps) / float(max(
                    episode_high_steps, 1
                )),
                dict(episode_terminations),
                dict(episode_requested_stages),
                dict(episode_executed_stages),
                float(episode_safety_steps) / float(max(
                    episode_low_steps, 1
                )),
                float(episode_overlap_steps) / float(max(
                    episode_low_steps, 1
                )),
                episode_residual_abs_sum / float(max(
                    episode_low_steps, 1
                )),
                episode_safety_projection_sum / float(max(
                    episode_low_steps, 1
                )),
                episode_safety_projection_max,
                episode_base_projection_sum / float(max(
                    episode_high_steps, 1
                )),
                episode_ee_projection_sum / float(max(
                    episode_high_steps, 1
                )),
                episode_terminal_steps,
                episode_terminal_projection / float(max(
                    episode_terminal_steps, 1
                )),
                dict(episode_terminal_pose_stage_counts),
                episode_terminal_student_alignment_sum / float(max(
                    episode_terminal_student_alignment_count, 1
                )),
                episode_terminal_executed_alignment_sum / float(max(
                    episode_terminal_executed_alignment_count, 1
                )),
            )
        )
    return {
        "episodes": int(counts["episodes"]),
        "successes": int(counts["successes"]),
        "success_rate": float(counts["successes"])
        / float(max(counts["episodes"], 1)),
        "collisions": int(counts["collisions"]),
        "timeouts": int(counts["timeouts"]),
        "mean_final_distance": float(np.mean(final_distances)),
        "high_steps": int(counts["high_steps"]),
        "low_steps": int(counts["low_steps"]),
        "mean_low_steps_per_high": float(counts["low_steps"])
        / float(max(counts["high_steps"], 1)),
        "requested_stage_counts": dict(requested_stage_counts),
        "executed_stage_counts": dict(executed_stage_counts),
        "option_termination_counts": dict(option_termination_counts),
        "safety_steps": int(counts["safety_steps"]),
        "safety_rate": float(counts["safety_steps"])
        / float(max(counts["low_steps"], 1)),
        "safety_reasons": dict(safety_reasons),
        "overlap_steps": int(counts["overlap_steps"]),
        "overlap_rate": float(counts["overlap_steps"])
        / float(max(counts["low_steps"], 1)),
        "mean_residual_abs": float(counts["residual_abs_sum"])
        / float(max(counts["low_steps"], 1)),
        "mean_safety_projection_abs": float(
            counts["safety_projection_sum"]
        ) / float(max(counts["low_steps"], 1)),
        "safety_projection_max": float(counts["safety_projection_max"]),
        "mean_base_command_projection": float(counts["base_projection_sum"])
        / float(max(counts["high_steps"], 1)),
        "mean_ee_command_projection": float(counts["ee_projection_sum"])
        / float(max(counts["high_steps"], 1)),
        "safety_rate_by_stage": _counter_means(
            stage_safety_steps, stage_low_steps
        ),
        "overlap_rate_by_stage": _counter_means(
            stage_overlap_steps, stage_low_steps
        ),
        "mean_residual_abs_by_stage": _counter_means(
            stage_residual_abs_sums, stage_low_steps
        ),
        "mean_safety_projection_abs_by_stage": _counter_means(
            stage_safety_projection_sums, stage_low_steps
        ),
        "mean_base_command_projection_by_stage": _counter_means(
            stage_base_projection_sums, stage_option_counts
        ),
        "mean_ee_command_projection_by_stage": _counter_means(
            stage_ee_projection_sums, stage_option_counts
        ),
        "terminal_steps": int(counts["terminal_steps"]),
        "terminal_projection_mean": float(counts["terminal_projection"])
        / float(max(counts["terminal_steps"], 1)),
        "terminal_pose_stage_counts": dict(terminal_pose_stage_counts),
        "terminal_student_alignment_cosine_mean": float(
            terminal_student_alignment_sum
        ) / float(max(terminal_student_alignment_count, 1)),
        "terminal_executed_alignment_cosine_mean": float(
            terminal_executed_alignment_sum
        ) / float(max(terminal_executed_alignment_count, 1)),
        # Backward-compatible alias: historical alignment measured the
        # environment-executed action after projection/blending.
        "terminal_alignment_cosine_mean": float(
            terminal_executed_alignment_sum
        ) / float(max(terminal_executed_alignment_count, 1)),
    }


def _info_float(info, name, default=float("nan")):
    try:
        return float(info.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _build_collected_dataset(**values):
    """Convert collection buffers to the persistent KL-DAgger contract."""
    dtypes = {
        "observations": np.float32,
        "actions": np.float32,
        "stages": np.int64,
        "scenario_ids": np.int64,
        "episode_ids": np.int64,
        "dagger_rounds": np.int16,
        "episode_success": np.bool_,
        "episode_collision": np.bool_,
        "episode_timeout": np.bool_,
        "episode_tf_ok": np.bool_,
        "final_distance": np.float32,
        "path_complete": np.bool_,
        "final_base_position_error": np.float32,
        "final_base_yaw_error": np.float32,
        "terminal_pose_stage": "<U16",
        "terminal_pose_stable_count": np.int32,
        "teacher_executed": np.bool_,
        "risk_intervention": np.bool_,
        "student_actions": np.float32,
        "student_stages": np.int64,
        "executed_actions": np.float32,
        "executed_stages": np.int64,
        "action_deviations": np.float32,
        "stage_disagreements": np.bool_,
    }
    result = {
        name: np.asarray(values[name], dtype=dtype)
        for name, dtype in dtypes.items()
    }
    for name in (
            "successful_episodes",
            "attempted_episodes",
            "low_steps",
            "dagger_samples",
            "teacher_execution_steps",
            "risk_intervention_count"):
        result[name] = int(values[name])
    return result


def _collect_teacher_dataset(
        client,
        sampler,
        default_budget,
        required_successes,
        maximum_attempts,
        progress_callback=None):
    observations = []
    actions = []
    stages = []
    scenario_ids = []
    episode_ids = []
    dagger_rounds = []
    episode_success = []
    episode_collision = []
    episode_timeout = []
    episode_tf_ok = []
    final_distance = []
    path_complete = []
    final_base_position_error = []
    final_base_yaw_error = []
    terminal_pose_stage = []
    terminal_pose_stable_count = []
    teacher_executed = []
    risk_intervention = []
    student_actions = []
    student_stages = []
    executed_actions = []
    executed_stages = []
    action_deviations = []
    stage_disagreements = []
    successful_episodes = 0
    attempted_episodes = 0
    low_steps = 0
    while (
            successful_episodes < int(required_successes)
            and attempted_episodes < int(maximum_attempts)):
        episode_observations = []
        episode_actions = []
        episode_stages = []
        observation, scenario = _reset(client, sampler, default_budget)
        done = False
        while not done:
            action = client.teacher_action.copy()
            stage = int(client.teacher_info.get("subgoal_type", 0))
            episode_observations.append(observation.copy())
            episode_actions.append(action.copy())
            episode_stages.append(stage)
            observation, unused_reward, done, info = client.step(
                action, subgoal_type=stage
            )
            low_steps += int(info.get("low_steps", 0))
        attempted_episodes += 1
        success = bool(info.get("success", False))
        if success:
            observations.extend(episode_observations)
            actions.extend(episode_actions)
            stages.extend(episode_stages)
            sample_count = len(episode_observations)
            scenario_id = int(scenario.get("scenario_id", -1))
            scenario_ids.extend([scenario_id] * sample_count)
            episode_ids.extend([attempted_episodes] * sample_count)
            dagger_rounds.extend([0] * sample_count)
            episode_success.extend([True] * sample_count)
            episode_collision.extend([
                bool(info.get("collision", False))
            ] * sample_count)
            episode_timeout.extend([
                bool(info.get("timeout", False))
            ] * sample_count)
            episode_tf_ok.extend([
                bool(info.get("tf_ok", False))
            ] * sample_count)
            final_distance.extend([
                _info_float(info, "dist")
            ] * sample_count)
            path_complete.extend([
                bool(info.get(
                    "path_complete",
                    info.get("final_base_pose_aligned", False),
                ))
            ] * sample_count)
            final_base_position_error.extend([
                _info_float(info, "final_base_position_error")
            ] * sample_count)
            final_base_yaw_error.extend([
                _info_float(info, "final_base_yaw_error")
            ] * sample_count)
            terminal_pose_stage.extend([
                str(info.get("terminal_pose_stage", "UNKNOWN"))
            ] * sample_count)
            terminal_pose_stable_count.extend([
                int(info.get("terminal_pose_stable_count", 0))
            ] * sample_count)
            teacher_executed.extend([True] * sample_count)
            risk_intervention.extend([False] * sample_count)
            student_actions.extend([
                np.full(
                    FusedBasicHighActorCritic.ACTION_DIM,
                    np.nan,
                    dtype=np.float32,
                )
                for unused_index in range(sample_count)
            ])
            student_stages.extend([-1] * sample_count)
            executed_actions.extend(episode_actions)
            executed_stages.extend(episode_stages)
            action_deviations.extend([np.nan] * sample_count)
            stage_disagreements.extend([False] * sample_count)
            successful_episodes += 1
            if progress_callback is not None:
                progress_callback(
                    _build_collected_dataset(
                        observations=observations,
                        actions=actions,
                        stages=stages,
                        scenario_ids=scenario_ids,
                        episode_ids=episode_ids,
                        dagger_rounds=dagger_rounds,
                        episode_success=episode_success,
                        episode_collision=episode_collision,
                        episode_timeout=episode_timeout,
                        episode_tf_ok=episode_tf_ok,
                        final_distance=final_distance,
                        path_complete=path_complete,
                        final_base_position_error=(
                            final_base_position_error
                        ),
                        final_base_yaw_error=final_base_yaw_error,
                        terminal_pose_stage=terminal_pose_stage,
                        terminal_pose_stable_count=(
                            terminal_pose_stable_count
                        ),
                        teacher_executed=teacher_executed,
                        risk_intervention=risk_intervention,
                        student_actions=student_actions,
                        student_stages=student_stages,
                        executed_actions=executed_actions,
                        executed_stages=executed_stages,
                        action_deviations=action_deviations,
                        stage_disagreements=stage_disagreements,
                        successful_episodes=successful_episodes,
                        attempted_episodes=attempted_episodes,
                        low_steps=low_steps,
                        dagger_samples=0,
                        teacher_execution_steps=len(observations),
                        risk_intervention_count=0,
                    ),
                    attempted_episodes,
                )
        print(
            "basic_high_teacher_episode={} scenario_id={} success={} "
            "kept={} samples={} tf_ok={} path_complete={} "
            "base_position_error={:.4f} base_yaw_error={:.4f} "
            "terminal_pose_stage={} stable_count={}".format(
                attempted_episodes,
                scenario.get("scenario_id"),
                success,
                success,
                len(episode_observations),
                bool(info.get("tf_ok", False)),
                bool(info.get(
                    "path_complete",
                    info.get("final_base_pose_aligned", False),
                )),
                _info_float(info, "final_base_position_error"),
                _info_float(info, "final_base_yaw_error"),
                info.get("terminal_pose_stage", "UNKNOWN"),
                int(info.get("terminal_pose_stable_count", 0)),
            )
        )
    if successful_episodes < int(required_successes):
        raise RuntimeError(
            "teacher produced only {}/{} required successful episodes".format(
                successful_episodes, required_successes
            )
        )
    return _build_collected_dataset(
        observations=observations,
        actions=actions,
        stages=stages,
        scenario_ids=scenario_ids,
        episode_ids=episode_ids,
        dagger_rounds=dagger_rounds,
        episode_success=episode_success,
        episode_collision=episode_collision,
        episode_timeout=episode_timeout,
        episode_tf_ok=episode_tf_ok,
        final_distance=final_distance,
        path_complete=path_complete,
        final_base_position_error=final_base_position_error,
        final_base_yaw_error=final_base_yaw_error,
        terminal_pose_stage=terminal_pose_stage,
        terminal_pose_stable_count=terminal_pose_stable_count,
        teacher_executed=teacher_executed,
        risk_intervention=risk_intervention,
        student_actions=student_actions,
        student_stages=student_stages,
        executed_actions=executed_actions,
        executed_stages=executed_stages,
        action_deviations=action_deviations,
        stage_disagreements=stage_disagreements,
        successful_episodes=successful_episodes,
        attempted_episodes=attempted_episodes,
        low_steps=low_steps,
        dagger_samples=0,
        teacher_execution_steps=len(observations),
        risk_intervention_count=0,
    )


def _collect_dagger_dataset(
        client,
        model,
        normalizer,
        sampler,
        default_budget,
        episodes,
        device,
        teacher_probability,
        action_deviation_threshold,
        seed,
        dagger_round,
        terminal_arm_deviation_threshold=None,
        base_action_deviation_takeover=False,
        progress_callback=None):
    """Label states induced by the student with the live rule teacher.

    Every visited state is stored with the rule action and stage, regardless
    of which policy is executed.  Scheduled teacher mixing keeps early DAgger
    rollouts recoverable, while mandatory takeover on a stage disagreement or
    terminal action deviation prevents an invalid arm-finish command from
    being sent to the low layer.  DIRECT/DETOUR continuous deviations are
    diagnostic by default so the student can actually visit and label its
    own compounded-error states.
    """
    observations = []
    actions = []
    stages = []
    scenario_ids = []
    episode_ids = []
    dagger_round_values = []
    episode_success = []
    episode_collision = []
    episode_timeout = []
    episode_tf_ok = []
    final_distance = []
    path_complete = []
    final_base_position_error = []
    final_base_yaw_error = []
    terminal_pose_stage = []
    terminal_pose_stable_count = []
    teacher_executed = []
    risk_intervention = []
    student_actions = []
    student_stages = []
    executed_actions = []
    executed_stages = []
    action_deviations = []
    stage_disagreement_flags = []
    random_state = np.random.RandomState(int(seed))
    successful_episodes = 0
    low_steps = 0
    teacher_execution_steps = 0
    student_execution_steps = 0
    risk_interventions = 0
    stage_disagreements = 0
    maximum_action_deviation = 0.0
    stage_sample_counts = collections.Counter()
    stage_teacher_counts = collections.Counter()
    stage_risk_counts = collections.Counter()
    stage_component_deviation_sums = dict(
        (stage, np.zeros(FusedBasicHighActorCritic.ACTION_DIM))
        for stage in range(FusedBasicHighActorCritic.STAGE_COUNT)
    )
    was_training = bool(model.training)
    model.eval()
    try:
        for episode_index in range(1, int(episodes) + 1):
            observation, scenario = _reset(client, sampler, default_budget)
            done = False
            episode_steps = 0
            episode_teacher_steps = 0
            episode_risk_steps = 0
            while not done:
                normalized = normalizer.normalize(observation)
                observation_tensor = torch.from_numpy(normalized).to(
                    device
                ).unsqueeze(0)
                with torch.no_grad():
                    student_action_tensor, student_stage_tensor = (
                        model.deterministic_decision(observation_tensor)
                    )
                student_action = (
                    student_action_tensor.squeeze(0).cpu().numpy()
                )
                student_stage = int(student_stage_tensor.item())
                teacher_action = client.teacher_action.copy()
                teacher_stage = int(
                    client.teacher_info.get("subgoal_type", 0)
                )
                observations.append(observation.copy())
                actions.append(teacher_action.copy())
                stages.append(teacher_stage)

                risk, stage_disagreement, action_deviation = (
                    _dagger_intervention_required(
                        student_action,
                        student_stage,
                        teacher_action,
                        teacher_stage,
                        action_deviation_threshold,
                        terminal_arm_deviation_threshold,
                        base_action_deviation_takeover,
                    )
                )
                use_teacher = bool(
                    risk or random_state.rand() < teacher_probability
                )
                if use_teacher:
                    executed_action = teacher_action
                    executed_stage = teacher_stage
                    teacher_execution_steps += 1
                    episode_teacher_steps += 1
                else:
                    executed_action = student_action
                    executed_stage = student_stage
                    student_execution_steps += 1
                if risk:
                    risk_interventions += 1
                    episode_risk_steps += 1
                stage_disagreements += int(stage_disagreement)
                stage_sample_counts[teacher_stage] += 1
                stage_teacher_counts[teacher_stage] += int(use_teacher)
                stage_risk_counts[teacher_stage] += int(risk)
                stage_component_deviation_sums[teacher_stage] += np.abs(
                    student_action - teacher_action
                )
                maximum_action_deviation = max(
                    maximum_action_deviation, action_deviation
                )
                scenario_ids.append(int(scenario.get("scenario_id", -1)))
                episode_ids.append(int(episode_index))
                dagger_round_values.append(int(dagger_round))
                teacher_executed.append(bool(use_teacher))
                risk_intervention.append(bool(risk))
                student_actions.append(student_action.copy())
                student_stages.append(int(student_stage))
                executed_actions.append(executed_action.copy())
                executed_stages.append(int(executed_stage))
                action_deviations.append(float(action_deviation))
                stage_disagreement_flags.append(bool(stage_disagreement))
                observation, unused_reward, done, info = client.step(
                    executed_action, subgoal_type=executed_stage
                )
                low_steps += int(info.get("low_steps", 0))
                episode_steps += 1
            success = bool(info.get("success", False))
            episode_success.extend([success] * episode_steps)
            episode_collision.extend([
                bool(info.get("collision", False))
            ] * episode_steps)
            episode_timeout.extend([
                bool(info.get("timeout", False))
            ] * episode_steps)
            episode_tf_ok.extend([
                bool(info.get("tf_ok", False))
            ] * episode_steps)
            final_distance.extend([
                _info_float(info, "dist")
            ] * episode_steps)
            path_complete.extend([
                bool(info.get(
                    "path_complete",
                    info.get("final_base_pose_aligned", False),
                ))
            ] * episode_steps)
            final_base_position_error.extend([
                _info_float(info, "final_base_position_error")
            ] * episode_steps)
            final_base_yaw_error.extend([
                _info_float(info, "final_base_yaw_error")
            ] * episode_steps)
            terminal_pose_stage.extend([
                str(info.get("terminal_pose_stage", "UNKNOWN"))
            ] * episode_steps)
            terminal_pose_stable_count.extend([
                int(info.get("terminal_pose_stable_count", 0))
            ] * episode_steps)
            successful_episodes += int(success)
            if progress_callback is not None:
                progress_callback(
                    _build_collected_dataset(
                        observations=observations,
                        actions=actions,
                        stages=stages,
                        scenario_ids=scenario_ids,
                        episode_ids=episode_ids,
                        dagger_rounds=dagger_round_values,
                        episode_success=episode_success,
                        episode_collision=episode_collision,
                        episode_timeout=episode_timeout,
                        episode_tf_ok=episode_tf_ok,
                        final_distance=final_distance,
                        path_complete=path_complete,
                        final_base_position_error=(
                            final_base_position_error
                        ),
                        final_base_yaw_error=final_base_yaw_error,
                        terminal_pose_stage=terminal_pose_stage,
                        terminal_pose_stable_count=(
                            terminal_pose_stable_count
                        ),
                        teacher_executed=teacher_executed,
                        risk_intervention=risk_intervention,
                        student_actions=student_actions,
                        student_stages=student_stages,
                        executed_actions=executed_actions,
                        executed_stages=executed_stages,
                        action_deviations=action_deviations,
                        stage_disagreements=(
                            stage_disagreement_flags
                        ),
                        successful_episodes=successful_episodes,
                        attempted_episodes=episode_index,
                        low_steps=low_steps,
                        dagger_samples=len(observations),
                        teacher_execution_steps=(
                            teacher_execution_steps
                        ),
                        risk_intervention_count=risk_interventions,
                    ),
                    episode_index,
                )
            print(
                "basic_high_dagger_episode round={} episode={}/{} "
                "scenario_id={} samples={} teacher_probability={:.3f} "
                "teacher_steps={} risk_interventions={} success={} "
                "collision={} timeout={} tf_ok={} path_complete={} "
                "base_position_error={:.4f} base_yaw_error={:.4f} "
                "terminal_pose_stage={} stable_count={}".format(
                    dagger_round,
                    episode_index,
                    episodes,
                    scenario.get("scenario_id"),
                    episode_steps,
                    teacher_probability,
                    episode_teacher_steps,
                    episode_risk_steps,
                    success,
                    bool(info.get("collision", False)),
                    bool(info.get("timeout", False)),
                    bool(info.get("tf_ok", False)),
                    bool(info.get(
                        "path_complete",
                        info.get("final_base_pose_aligned", False),
                    )),
                    _info_float(info, "final_base_position_error"),
                    _info_float(info, "final_base_yaw_error"),
                    info.get("terminal_pose_stage", "UNKNOWN"),
                    int(info.get("terminal_pose_stable_count", 0)),
                )
            )
    finally:
        model.train(was_training)

    total_execution_steps = teacher_execution_steps + student_execution_steps
    stage_diagnostics = {}
    for stage, name in enumerate(DAGGER_STAGE_NAMES):
        sample_count = int(stage_sample_counts[stage])
        stage_diagnostics[name] = {
            "samples": sample_count,
            "teacher_rate": round(
                float(stage_teacher_counts[stage])
                / float(max(sample_count, 1)),
                4,
            ),
            "risk_rate": round(
                float(stage_risk_counts[stage])
                / float(max(sample_count, 1)),
                4,
            ),
            "component_mae": np.round(
                stage_component_deviation_sums[stage]
                / float(max(sample_count, 1)),
                5,
            ).tolist(),
        }
    print(
        "basic_high_dagger_complete round={} episodes={} successes={} "
        "samples={} teacher_probability={:.3f} "
        "teacher_execution_rate={:.3f} risk_interventions={} "
        "stage_disagreements={} max_action_deviation={:.4f}".format(
            dagger_round,
            episodes,
            successful_episodes,
            len(observations),
            teacher_probability,
            float(teacher_execution_steps)
            / float(max(total_execution_steps, 1)),
            risk_interventions,
            stage_disagreements,
            maximum_action_deviation,
        )
    )
    print(
        "basic_high_dagger_stage_diagnostics round={} {}".format(
            dagger_round, stage_diagnostics
        )
    )
    return _build_collected_dataset(
        observations=observations,
        actions=actions,
        stages=stages,
        scenario_ids=scenario_ids,
        episode_ids=episode_ids,
        dagger_rounds=dagger_round_values,
        episode_success=episode_success,
        episode_collision=episode_collision,
        episode_timeout=episode_timeout,
        episode_tf_ok=episode_tf_ok,
        final_distance=final_distance,
        path_complete=path_complete,
        final_base_position_error=final_base_position_error,
        final_base_yaw_error=final_base_yaw_error,
        terminal_pose_stage=terminal_pose_stage,
        terminal_pose_stable_count=terminal_pose_stable_count,
        teacher_executed=teacher_executed,
        risk_intervention=risk_intervention,
        student_actions=student_actions,
        student_stages=student_stages,
        executed_actions=executed_actions,
        executed_stages=executed_stages,
        action_deviations=action_deviations,
        stage_disagreements=stage_disagreement_flags,
        successful_episodes=successful_episodes,
        attempted_episodes=episodes,
        low_steps=low_steps,
        dagger_samples=len(observations),
        teacher_execution_steps=teacher_execution_steps,
        risk_intervention_count=risk_interventions,
    )


def _dagger_intervention_required(
        student_action,
        student_stage,
        teacher_action,
        teacher_stage,
        action_deviation_threshold,
        terminal_arm_deviation_threshold=None,
        base_action_deviation_takeover=True):
    student_action = np.asarray(student_action, dtype=np.float32)
    teacher_action = np.asarray(teacher_action, dtype=np.float32)
    if student_action.shape != teacher_action.shape:
        raise ValueError("student and teacher actions must have equal shape")
    action_deviation = float(np.max(np.abs(
        student_action - teacher_action
    )))
    action_deviation_threshold = _dagger_stage_threshold(
        action_deviation_threshold, teacher_stage
    )
    terminal_arm_deviation = float(np.max(np.abs(
        student_action[ARM_SUBGOAL_SLICE]
        - teacher_action[ARM_SUBGOAL_SLICE]
    )))
    stage_disagreement = bool(int(student_stage) != int(teacher_stage))
    terminal_arm_intervention = bool(
        int(teacher_stage) == TERMINAL_STAGE
        and terminal_arm_deviation_threshold is not None
        and terminal_arm_deviation
        > float(terminal_arm_deviation_threshold)
    )
    stage_action_intervention = bool(
        action_deviation > float(action_deviation_threshold)
        and (
            int(teacher_stage) == TERMINAL_STAGE
            or bool(base_action_deviation_takeover)
        )
    )
    intervention = bool(
        stage_disagreement
        or stage_action_intervention
        or terminal_arm_intervention
    )
    return intervention, stage_disagreement, action_deviation


def _dagger_stage_threshold(action_deviation_thresholds, teacher_stage):
    thresholds = np.asarray(
        action_deviation_thresholds, dtype=np.float32
    ).reshape(-1)
    if len(thresholds) == 1:
        return float(thresholds[0])
    if len(thresholds) != len(DAGGER_STAGE_NAMES):
        raise ValueError(
            "DAgger action thresholds must be scalar or DIRECT/DETOUR/"
            "TERMINAL"
        )
    teacher_stage = int(teacher_stage)
    if teacher_stage not in range(len(DAGGER_STAGE_NAMES)):
        raise ValueError("unknown teacher stage: {}".format(teacher_stage))
    return float(thresholds[teacher_stage])


def _dagger_action_deviation_thresholds(args):
    if args.dagger_action_deviation_threshold is not None:
        return (float(args.dagger_action_deviation_threshold),) * 3
    return (
        float(args.dagger_direct_action_deviation_threshold),
        float(args.dagger_detour_action_deviation_threshold),
        float(args.dagger_terminal_action_deviation_threshold),
    )


def _fit_teacher_warmstart(
        model,
        normalizer,
        dataset,
        device,
        args,
        epochs,
        label,
        seed_offset=0):
    observations = dataset["observations"]
    actions = dataset["actions"]
    stages = dataset["stages"]
    episode_ids = np.asarray(dataset.get(
        "episode_ids", np.arange(len(observations))
    ), dtype=np.int64)
    _refit_normalizer(normalizer, observations)
    normalized = normalizer.normalize(observations)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.warmstart_learning_rate
    )
    random_state = np.random.RandomState(
        int(args.seed) + 7919 * int(seed_offset)
    )
    print(
        "basic_high_bc_sampling label={} contract="
        "stage_balanced_episode_uniform samples={} episodes={}".format(
            label,
            len(observations),
            len(np.unique(episode_ids)),
        )
    )
    for epoch in range(int(epochs)):
        indices = _balanced_stage_episode_indices(
            stages, episode_ids, random_state
        )
        totals = collections.Counter()
        stage_absolute_error_sums = dict(
            (stage, np.zeros(FusedBasicHighActorCritic.ACTION_DIM))
            for stage in range(FusedBasicHighActorCritic.STAGE_COUNT)
        )
        stage_error_counts = collections.Counter()
        count = 0
        for start in range(0, len(indices), args.warmstart_batch_size):
            batch = indices[start:start + args.warmstart_batch_size]
            observation_tensor = torch.from_numpy(normalized[batch]).to(device)
            action_tensor = torch.from_numpy(actions[batch]).to(device)
            stage_tensor = torch.from_numpy(stages[batch]).to(device)
            action_kl, stage_kl = model.teacher_kl(
                observation_tensor,
                action_tensor,
                stage_tensor,
                teacher_standard_deviation=args.teacher_kl_std,
            )
            action_kl_mean = action_kl.mean()
            stage_kl_mean = stage_kl.mean()
            predicted_actions = model.deterministic_action(
                observation_tensor, stage_tensor
            )
            absolute_errors = torch.abs(
                predicted_actions.detach() - action_tensor
            ).cpu().numpy()
            batch_stages = stages[batch]
            for stage in range(FusedBasicHighActorCritic.STAGE_COUNT):
                stage_mask = batch_stages == stage
                if np.any(stage_mask):
                    stage_absolute_error_sums[stage] += np.sum(
                        absolute_errors[stage_mask], axis=0
                    )
                    stage_error_counts[stage] += int(np.sum(stage_mask))
            terminal_mask = stage_tensor == TERMINAL_STAGE
            if bool(torch.any(terminal_mask).item()):
                terminal_arm_mse = (
                    predicted_actions[terminal_mask, ARM_SUBGOAL_SLICE]
                    - action_tensor[terminal_mask, ARM_SUBGOAL_SLICE]
                ).pow(2).mean()
            else:
                terminal_arm_mse = action_kl_mean * 0.0
            loss = (
                action_kl_mean
                + args.warmstart_stage_coefficient * stage_kl_mean
                + args.warmstart_terminal_arm_coefficient
                * terminal_arm_mse
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.maximum_gradient_norm
            )
            optimizer.step()
            batch_count = int(len(batch))
            totals["action_kl"] += float(
                action_kl_mean.item()
            ) * batch_count
            totals["stage_kl"] += float(
                stage_kl_mean.item()
            ) * batch_count
            totals["terminal_arm_mse"] += float(
                terminal_arm_mse.item()
            ) * batch_count
            with torch.no_grad():
                all_means, stage_logits, unused_values = model._outputs(
                    observation_tensor
                )
            totals["accuracy"] += float((
                torch.argmax(stage_logits, dim=-1) == stage_tensor
            ).float().mean().item()) * batch_count
            count += batch_count
        if epoch == 0 or (epoch + 1) % 10 == 0:
            stage_action_mae = dict(
                (
                    DAGGER_STAGE_NAMES[stage],
                    np.round(
                        stage_absolute_error_sums[stage]
                        / float(max(stage_error_counts[stage], 1)),
                        5,
                    ).tolist(),
                )
                for stage in range(FusedBasicHighActorCritic.STAGE_COUNT)
            )
            print(
                "basic_high_bc_epoch label={} epoch={}/{} "
                "teacher_action_kl={:.7f} "
                "teacher_stage_kl={:.6f} terminal_arm_mse={:.7f} "
                "stage_accuracy={:.3f} stage_action_mae={}".format(
                    label,
                    epoch + 1,
                    epochs,
                    totals["action_kl"] / max(count, 1),
                    totals["stage_kl"] / max(count, 1),
                    totals["terminal_arm_mse"] / max(count, 1),
                    totals["accuracy"] / max(count, 1),
                    stage_action_mae,
                )
            )
    return optimizer


def _refit_normalizer(normalizer, observations):
    """Fit normalization once to the complete aggregated DAgger dataset."""
    normalizer.count = 1e-4
    normalizer.mean = np.zeros(
        normalizer.normalized_dim, dtype=np.float64
    )
    normalizer.variance = np.ones(
        normalizer.normalized_dim, dtype=np.float64
    )
    normalizer.update(observations)


def _balanced_stage_indices(stages, random_state):
    groups = [
        np.flatnonzero(stages == stage)
        for stage in range(FusedBasicHighActorCritic.STAGE_COUNT)
    ]
    if any(len(group) == 0 for group in groups):
        raise RuntimeError(
            "teacher dataset must contain DIRECT, DETOUR and TERMINAL"
        )
    target = max(len(group) for group in groups)
    balanced = np.concatenate([
        random_state.choice(group, size=target, replace=len(group) < target)
        for group in groups
    ])
    return balanced[random_state.permutation(len(balanced))]


def _balanced_stage_episode_indices(stages, episode_ids, random_state):
    """Balance semantic stages, then sample episodes uniformly per stage.

    A timeout can contain hundreds of highly correlated high-level samples,
    while a successful episode may contain only a few dozen.  Plain sample
    balancing therefore lets one failed trajectory dominate a KL-BC epoch.
    This sampler preserves the previous equal-stage contract but gives each
    episode equal probability inside its stage.
    """
    stages = np.asarray(stages, dtype=np.int64)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    if stages.shape != episode_ids.shape:
        raise ValueError("stages and episode_ids must have equal shape")
    stage_groups = [
        np.flatnonzero(stages == stage)
        for stage in range(FusedBasicHighActorCritic.STAGE_COUNT)
    ]
    if any(len(group) == 0 for group in stage_groups):
        raise RuntimeError(
            "teacher dataset must contain DIRECT, DETOUR and TERMINAL"
        )
    target = max(len(group) for group in stage_groups)
    selected = []
    for group in stage_groups:
        grouped_by_episode = {}
        for sample_index in group:
            grouped_by_episode.setdefault(
                int(episode_ids[sample_index]), []
            ).append(int(sample_index))
        episode_keys = np.asarray(
            sorted(grouped_by_episode), dtype=np.int64
        )
        chosen_episodes = random_state.choice(
            episode_keys, size=target, replace=True
        )
        selected.append(np.asarray([
            random_state.choice(grouped_by_episode[int(episode_id)])
            for episode_id in chosen_episodes
        ], dtype=np.int64))
    balanced = np.concatenate(selected)
    return balanced[random_state.permutation(len(balanced))]


def _merge_teacher_datasets(existing, new):
    if existing is None:
        return _ensure_dagger_sample_fields(new)
    existing = _ensure_dagger_sample_fields(existing)
    new = _ensure_dagger_sample_fields(new)
    episode_offset = (
        int(np.max(existing["episode_ids"])) + 1
        if len(existing["episode_ids"]) else 0
    )
    new = dict(new)
    new["episode_ids"] = new["episode_ids"] + episode_offset
    merged = {
        "successful_episodes": int(existing["successful_episodes"])
        + int(new["successful_episodes"]),
        "attempted_episodes": int(existing["attempted_episodes"])
        + int(new["attempted_episodes"]),
        "low_steps": int(existing["low_steps"]) + int(new["low_steps"]),
    }
    for name in _dagger_sample_field_names():
        merged[name] = np.concatenate((
            existing[name], new[name]
        ), axis=0)
    for name in (
            "dagger_samples",
            "teacher_execution_steps",
            "risk_intervention_count"):
        merged[name] = int(existing.get(name, 0)) + int(new.get(name, 0))
    return merged


def _dagger_sample_field_names():
    return (
        "observations",
        "actions",
        "stages",
        "scenario_ids",
        "episode_ids",
        "dagger_rounds",
        "episode_success",
        "episode_collision",
        "episode_timeout",
        "episode_tf_ok",
        "final_distance",
        "path_complete",
        "final_base_position_error",
        "final_base_yaw_error",
        "terminal_pose_stage",
        "terminal_pose_stable_count",
        "teacher_executed",
        "risk_intervention",
        "student_actions",
        "student_stages",
        "executed_actions",
        "executed_stages",
        "action_deviations",
        "stage_disagreements",
    )


def _ensure_dagger_sample_fields(dataset):
    result = dict(dataset)
    observations = np.asarray(result["observations"], dtype=np.float32)
    actions = np.asarray(result["actions"], dtype=np.float32)
    stages = np.asarray(result["stages"], dtype=np.int64)
    sample_count = int(len(observations))
    if observations.shape != (
            sample_count, FusedBasicHighActorCritic.OBS_DIM):
        raise ValueError("invalid DAgger observation array shape")
    if actions.shape != (
            sample_count, FusedBasicHighActorCritic.ACTION_DIM):
        raise ValueError("invalid DAgger teacher action array shape")
    if stages.shape != (sample_count,):
        raise ValueError("invalid DAgger teacher stage array shape")
    result["observations"] = observations
    result["actions"] = actions
    result["stages"] = stages
    scalar_defaults = {
        "scenario_ids": (-1, np.int64),
        "episode_ids": (0, np.int64),
        "dagger_rounds": (0, np.int16),
        "episode_success": (False, np.bool_),
        "episode_collision": (False, np.bool_),
        "episode_timeout": (False, np.bool_),
        "episode_tf_ok": (True, np.bool_),
        "final_distance": (np.nan, np.float32),
        "path_complete": (False, np.bool_),
        "final_base_position_error": (np.nan, np.float32),
        "final_base_yaw_error": (np.nan, np.float32),
        "terminal_pose_stage": ("UNKNOWN", "<U16"),
        "terminal_pose_stable_count": (0, np.int32),
        "teacher_executed": (True, np.bool_),
        "risk_intervention": (False, np.bool_),
        "student_stages": (-1, np.int64),
        "executed_stages": (0, np.int64),
        "action_deviations": (np.nan, np.float32),
        "stage_disagreements": (False, np.bool_),
    }
    for name, (default, dtype) in scalar_defaults.items():
        value = result.get(name)
        if value is None:
            if name == "executed_stages":
                value = stages.copy()
            else:
                value = np.full(sample_count, default, dtype=dtype)
        value = np.asarray(value, dtype=dtype)
        if value.shape != (sample_count,):
            raise ValueError("invalid DAgger {} array shape".format(name))
        result[name] = value
    vector_defaults = {
        "student_actions": np.full(
            (sample_count, FusedBasicHighActorCritic.ACTION_DIM),
            np.nan,
            dtype=np.float32,
        ),
        "executed_actions": actions.copy(),
    }
    for name, default in vector_defaults.items():
        value = np.asarray(result.get(name, default), dtype=np.float32)
        if value.shape != (
                sample_count, FusedBasicHighActorCritic.ACTION_DIM):
            raise ValueError("invalid DAgger {} array shape".format(name))
        result[name] = value
    for name in (
            "successful_episodes",
            "attempted_episodes",
            "low_steps",
            "dagger_samples",
            "teacher_execution_steps",
            "risk_intervention_count"):
        result[name] = int(result.get(name, 0))
    return result


def _save_dagger_dataset(
        path,
        dataset,
        completed_dagger_round,
        label):
    dataset = _ensure_dagger_sample_fields(dataset)
    payload = {
        "format_version": np.asarray(2, dtype=np.int64),
        "observation_contract": np.asarray(
            "physical_state_plus_active_path_context_v2"
        ),
        "completed_dagger_round": np.asarray(
            int(completed_dagger_round), dtype=np.int64
        ),
        "observations": dataset["observations"],
        "teacher_actions": dataset["actions"],
        "teacher_stages": dataset["stages"],
    }
    for name in _dagger_sample_field_names():
        if name not in ("observations", "actions", "stages"):
            payload[name] = dataset[name]
    for name in (
            "successful_episodes",
            "attempted_episodes",
            "low_steps",
            "dagger_samples",
            "teacher_execution_steps",
            "risk_intervention_count"):
        payload[name] = np.asarray(dataset.get(name, 0), dtype=np.int64)
    latest_path = os.path.abspath(os.path.expanduser(path))
    snapshot_path = _dagger_dataset_snapshot_path(latest_path, label)
    _write_npz_atomic(latest_path, payload)
    if snapshot_path != latest_path:
        _write_npz_atomic(snapshot_path, payload)
    print(
        "basic_high_dagger_dataset_saved path={} snapshot={} samples={} "
        "completed_round={}".format(
            latest_path,
            snapshot_path,
            len(dataset["observations"]),
            completed_dagger_round,
        )
    )


def _load_dagger_dataset(path):
    resolved = os.path.abspath(os.path.expanduser(path))
    with np.load(resolved, allow_pickle=False) as archive:
        if int(archive["format_version"].item()) != 2:
            raise ValueError(
                "unsupported KL-DAgger dataset format; the 86-D Markov "
                "path-context contract requires a new dataset"
            )
        dataset = {
            "observations": archive["observations"].astype(np.float32),
            "actions": archive["teacher_actions"].astype(np.float32),
            "stages": archive["teacher_stages"].astype(np.int64),
        }
        for name in _dagger_sample_field_names():
            if (
                    name not in ("observations", "actions", "stages")
                    and name in archive.files):
                dataset[name] = archive[name].copy()
        for name in (
                "successful_episodes",
                "attempted_episodes",
                "low_steps",
                "dagger_samples",
                "teacher_execution_steps",
                "risk_intervention_count"):
            dataset[name] = (
                int(archive[name].item()) if name in archive.files else 0
            )
        completed_round = int(archive["completed_dagger_round"].item())
    return _ensure_dagger_sample_fields(dataset), completed_round


def _write_npz_atomic(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    temporary_path = path + ".tmp.npz"
    np.savez_compressed(temporary_path, **payload)
    os.replace(temporary_path, path)


def _dagger_dataset_snapshot_path(path, label):
    root, extension = os.path.splitext(path)
    if root.endswith(".latest"):
        root = root[:-len(".latest")]
    return "{}.{}{}".format(root, str(label), extension or ".npz")


def _dagger_teacher_probability(
        dagger_round,
        dagger_rounds,
        initial_probability,
        final_probability):
    if int(dagger_rounds) <= 1:
        return float(initial_probability)
    fraction = (
        float(int(dagger_round) - 1)
        / float(max(int(dagger_rounds) - 1, 1))
    )
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return float(initial_probability) + fraction * (
        float(final_probability) - float(initial_probability)
    )


def _gate_pass(metrics, args):
    return bool(
        metrics["success_rate"] >= args.minimum_gate_success_rate
        and metrics["collisions"] <= args.maximum_gate_collisions
    )


def _gate_key(metrics):
    return (
        int(metrics["successes"]),
        -int(metrics["collisions"]),
        -int(metrics["timeouts"]),
        -float(metrics["mean_final_distance"]),
    )


def _new_terminal_blend_curriculum(args):
    schedule = [float(value) for value in args.terminal_blend_schedule]
    enabled = bool(args.terminal_blend_annealing)
    return {
        "enabled": enabled,
        "schedule": schedule,
        "stage_index": 0,
        "active_blend": schedule[0] if enabled else None,
        "gate_streak": 0,
        "completed_stage_index": -1,
        "completed": False,
    }


def _restore_terminal_blend_curriculum(saved, args):
    """Restore an exact annealing stage while supporting legacy checkpoints."""
    fresh = _new_terminal_blend_curriculum(args)
    if not isinstance(saved, dict):
        return fresh
    # An explicit annealing request can start a new curriculum from a legacy
    # fixed-blend checkpoint.  Once a checkpoint contains an enabled
    # curriculum, its schedule and active stage are authoritative on resume.
    if not bool(saved.get("enabled", False)):
        return fresh
    schedule = [float(value) for value in saved.get("schedule", [])]
    _validate_terminal_blend_schedule(schedule)
    stage_index = int(saved.get("stage_index", 0))
    if stage_index < 0 or stage_index >= len(schedule):
        raise ValueError(
            "checkpoint terminal blend stage index is out of range"
        )
    completed_stage_index = int(saved.get("completed_stage_index", -1))
    completed_stage_index = max(
        -1, min(completed_stage_index, len(schedule) - 1)
    )
    return {
        "enabled": True,
        "schedule": schedule,
        "stage_index": stage_index,
        "active_blend": schedule[stage_index],
        "gate_streak": max(int(saved.get("gate_streak", 0)), 0),
        "completed_stage_index": completed_stage_index,
        "completed": bool(saved.get("completed", False)),
    }


def _active_terminal_blend(curriculum):
    if not bool(curriculum.get("enabled", False)):
        return None
    return float(curriculum["schedule"][curriculum["stage_index"]])


def _terminal_blend_gate_pass(metrics, args):
    return bool(
        metrics["success_rate"]
        >= args.terminal_blend_minimum_success_rate
        and metrics["collisions"]
        <= args.terminal_blend_maximum_collisions
    )


def _update_terminal_blend_curriculum(curriculum, metrics, args):
    """Apply one deterministic ability gate to the annealing state."""
    active_blend = _active_terminal_blend(curriculum)
    event = {
        "enabled": bool(curriculum.get("enabled", False)),
        "gate_pass": False,
        "stage_completed": False,
        "advanced": False,
        "completed_blend": None,
        "next_blend": active_blend,
        "gate_streak": int(curriculum.get("gate_streak", 0)),
        "required_gates": int(args.terminal_blend_required_gates),
        "curriculum_completed": bool(curriculum.get("completed", False)),
    }
    if not event["enabled"]:
        return event
    passed = _terminal_blend_gate_pass(metrics, args)
    curriculum["gate_streak"] = (
        int(curriculum.get("gate_streak", 0)) + 1 if passed else 0
    )
    event["gate_pass"] = bool(passed)
    event["gate_streak"] = int(curriculum["gate_streak"])
    if (
            not passed
            or curriculum["gate_streak"]
            < int(args.terminal_blend_required_gates)):
        return event

    stage_index = int(curriculum["stage_index"])
    already_completed = bool(
        int(curriculum.get("completed_stage_index", -1)) >= stage_index
    )
    curriculum["completed_stage_index"] = max(
        int(curriculum.get("completed_stage_index", -1)), stage_index
    )
    curriculum["gate_streak"] = 0
    event["completed_blend"] = float(
        curriculum["schedule"][stage_index]
    )
    event["stage_completed"] = not already_completed
    if stage_index + 1 < len(curriculum["schedule"]):
        curriculum["stage_index"] = stage_index + 1
        curriculum["active_blend"] = float(
            curriculum["schedule"][stage_index + 1]
        )
        event["advanced"] = True
    else:
        curriculum["completed"] = True
        curriculum["active_blend"] = float(
            curriculum["schedule"][stage_index]
        )
    event["next_blend"] = _active_terminal_blend(curriculum)
    event["curriculum_completed"] = bool(curriculum["completed"])
    return event


def _validate_terminal_blend_schedule(schedule):
    if not schedule:
        raise ValueError("terminal blend schedule must not be empty")
    previous = -1.0
    for value in schedule:
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "terminal blend schedule values must be in [0, 1]"
            )
        if value <= previous:
            raise ValueError(
                "terminal blend schedule must be strictly increasing"
            )
        previous = value


def _terminal_blend_checkpoint_path(path, blend):
    root, extension = os.path.splitext(path)
    label = "{:.2f}".format(float(blend)).replace(".", "p")
    return "{}.terminal_blend_{}_passed{}".format(
        root, label, extension or ".pt"
    )


def _reset(client, sampler, default_budget):
    scenario = sampler.next()
    budget = int(scenario.get("step_budget") or default_budget)
    observation, unused_info = client.reset_with_info(
        scenario=scenario, max_steps=budget
    )
    return observation, scenario


def _basic_scenarios(scenarios, fixed_detour_side):
    """Remove the unobserved upper/lower label from the basic benchmark.

    A single deterministic passage direction makes the same observed state
    map to one continuous target.  Generalizing the passage direction is a
    later experiment; it is not part of the minimum hierarchy contract.
    """
    result = []
    for source in scenarios:
        scenario = dict(source)
        if not bool(scenario.get("no_obstacle", False)):
            scenario["detour_side"] = float(fixed_detour_side)
        result.append(scenario)
    return result


def _bootstrap(model, normalizer, observation, done, device):
    if done:
        return 0.0
    normalized = normalizer.normalize(observation)
    tensor = torch.from_numpy(normalized).to(device).unsqueeze(0)
    with torch.no_grad():
        return float(model.get_value(tensor).item())


def _ppo_update(
        model,
        optimizer,
        buffer,
        stages,
        teacher_stages,
        device,
        args,
        teacher_coefficient,
        reference_model,
        reference_coefficient):
    data = buffer.tensors(device)
    stages = torch.as_tensor(stages, dtype=torch.int64, device=device)
    if tuple(stages.shape) != (buffer.size,):
        raise ValueError("stage labels must match the rollout buffer")
    data["advantages"] = _normalize_ppo_advantages(
        data["advantages"],
        stages if args.stage_balanced_advantages else None,
    )
    teacher_stages = torch.as_tensor(
        teacher_stages, dtype=torch.int64, device=device
    )
    if tuple(teacher_stages.shape) != (buffer.size,):
        raise ValueError("teacher stages must match the rollout buffer")
    value_targets = data["returns"].detach()
    values_before = data["old_values"].detach()
    value_explained_variance_before = _explained_variance(
        values_before, value_targets
    )
    value_rmse_before = float(torch.sqrt(torch.mean(
        (values_before - value_targets).pow(2)
    )).item())
    actor_parameters = (
        list(model.actor_backbone.parameters())
        + list(model.actor_mean.parameters())
        + list(model.stage_head.parameters())
        + [model.log_std]
    )
    critic_parameters = (
        list(model.critic_backbone.parameters())
        + list(model.critic.parameters())
    )
    actor_totals = collections.Counter()
    critic_totals = collections.Counter()
    actor_count = 0
    critic_count = 0
    stopped_early = False
    epochs_completed = 0
    full_epochs_completed = 0
    minibatches_completed = 0
    critic_full_epochs_completed = 0
    critic_minibatches_completed = 0
    value_clip_range = _resolved_value_clip_range(args)
    policy_kl_max = 0.0
    policy_kl_batch_max = 0.0
    policy_kl_stop_value = 0.0
    policy_kl_epoch_values = []
    policy_kl_minibatch_values = []
    policy_kl_stop_epoch = 0
    policy_kl_stop_minibatch = 0
    policy_kl_stop_global_minibatch = 0

    # Optimize the actor first.  A minibatch KL is useful as a diagnostic but
    # is too local to define a trust region.  After every actor minibatch,
    # therefore, measure KL on the complete rollout and stop before another
    # actor update can compound an overshoot.  Critic training remains
    # independent and always runs below.
    for epoch_index in range(args.ppo_epochs):
        epochs_completed = epoch_index + 1
        epoch_completed = True
        epoch_minibatches = list(_balanced_ppo_minibatch_indices(
            buffer.size, args.batch_size
        ))
        for minibatch_index, indices_np in enumerate(epoch_minibatches):
            indices = torch.from_numpy(
                indices_np.astype(np.int64)
            ).to(device)
            observations = data["observations"].index_select(0, indices)
            actions = data["actions"].index_select(0, indices)
            teacher_actions = data["teacher_actions"].index_select(
                0, indices
            )
            selected_stages = stages.index_select(0, indices)
            selected_teacher_stages = teacher_stages.index_select(
                0, indices
            )
            old_logp = data["old_log_probabilities"].index_select(0, indices)
            batch_advantages = data["advantages"].index_select(0, indices)
            logp, entropy, unused_values, mean_action, stage_logits = (
                model.evaluate_actions(
                    observations, actions, selected_stages
                )
            )
            log_ratio = logp - old_logp
            ratio = torch.exp(log_ratio)
            approximate_policy_kl = (
                (ratio - 1.0) - log_ratio
            ).mean()
            policy_kl_value = float(approximate_policy_kl.item())
            policy_kl_batch_max = max(
                policy_kl_batch_max, policy_kl_value
            )
            clipped_ratio = torch.clamp(
                ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
            )
            clip_fraction = (
                torch.abs(ratio - 1.0) > args.clip_ratio
            ).float().mean()
            policy_loss = -torch.min(
                ratio * batch_advantages,
                clipped_ratio * batch_advantages,
            ).mean()
            entropy_mean = entropy.mean()
            teacher_action_kl, teacher_stage_kl = model.teacher_kl(
                observations,
                teacher_actions,
                selected_teacher_stages,
                teacher_standard_deviation=args.teacher_kl_std,
            )
            teacher_weights = torch.ones_like(teacher_action_kl)
            teacher_weights = torch.where(
                selected_teacher_stages == TERMINAL_STAGE,
                teacher_weights * args.teacher_terminal_weight,
                teacher_weights,
            )
            teacher_weight_sum = torch.clamp(
                teacher_weights.sum(), min=1.0
            )
            teacher_action_kl_mean = (
                teacher_action_kl * teacher_weights
            ).sum() / teacher_weight_sum
            teacher_stage_kl_mean = (
                teacher_stage_kl * teacher_weights
            ).sum() / teacher_weight_sum
            teacher_kl = (
                teacher_action_kl_mean
                + args.teacher_stage_coefficient * teacher_stage_kl_mean
            )
            reference_action_kl, reference_stage_kl = model.reference_kl(
                observations, reference_model
            )
            reference_action_kl_mean = reference_action_kl.mean()
            reference_stage_kl_mean = reference_stage_kl.mean()
            reference_kl = (
                reference_action_kl_mean
                + args.reference_stage_coefficient
                * reference_stage_kl_mean
            )
            actor_loss = (
                policy_loss
                - args.entropy_coefficient * entropy_mean
                + teacher_coefficient * teacher_kl
                + reference_coefficient * reference_kl
            )
            optimizer.zero_grad()
            actor_loss.backward()
            _clear_parameter_gradients(critic_parameters)
            actor_grad_norm = _parameter_gradient_norm(actor_parameters)
            torch.nn.utils.clip_grad_norm_(
                actor_parameters, args.maximum_gradient_norm
            )
            optimizer.step()
            batch_count = int(indices.shape[0])
            actor_totals["policy_loss"] += (
                float(policy_loss.item()) * batch_count
            )
            actor_totals["entropy"] += (
                float(entropy_mean.item()) * batch_count
            )
            actor_totals["teacher_kl"] += float(
                teacher_kl.item()
            ) * batch_count
            actor_totals["teacher_action_kl"] += float(
                teacher_action_kl_mean.item()
            ) * batch_count
            actor_totals["teacher_stage_kl"] += float(
                teacher_stage_kl_mean.item()
            ) * batch_count
            actor_totals["reference_kl"] += float(
                reference_kl.item()
            ) * batch_count
            actor_totals["reference_action_kl"] += float(
                reference_action_kl_mean.item()
            ) * batch_count
            actor_totals["reference_stage_kl"] += float(
                reference_stage_kl_mean.item()
            ) * batch_count
            actor_totals["policy_kl_batch"] += float(
                approximate_policy_kl.item()
            ) * batch_count
            actor_totals["clip_fraction"] += float(
                clip_fraction.item()
            ) * batch_count
            actor_totals["actor_grad_norm"] += (
                actor_grad_norm * batch_count
            )
            actor_totals["action_abs"] += float(
                mean_action.abs().mean().item()
            ) * batch_count
            actor_count += batch_count
            minibatches_completed += 1

            rollout_policy_kl = _full_rollout_policy_kl(
                model,
                data["observations"],
                data["actions"],
                stages,
                data["old_log_probabilities"],
                args.batch_size,
            )
            policy_kl_minibatch_values.append(rollout_policy_kl)
            policy_kl_max = max(policy_kl_max, rollout_policy_kl)
            if (
                    args.target_policy_kl > 0.0
                    and rollout_policy_kl
                    > args.target_policy_kl
                    * args.policy_kl_stop_multiplier):
                stopped_early = True
                epoch_completed = (
                    minibatch_index + 1 == len(epoch_minibatches)
                )
                policy_kl_stop_value = rollout_policy_kl
                policy_kl_stop_epoch = epoch_index + 1
                policy_kl_stop_minibatch = minibatch_index + 1
                policy_kl_stop_global_minibatch = minibatches_completed
                break

        if epoch_completed:
            full_epochs_completed += 1
            policy_kl_epoch_values.append(
                policy_kl_minibatch_values[-1]
            )
        if stopped_early:
            break

    # The value function has an independent optimization phase.  Even when
    # the actor reaches its KL trust-region boundary, every critic epoch is
    # completed so that GAE does not remain tied to an untrained baseline.
    for unused_epoch_index in range(args.ppo_epochs):
        for indices_np in _balanced_ppo_minibatch_indices(
                buffer.size, args.batch_size):
            indices = torch.from_numpy(
                indices_np.astype(np.int64)
            ).to(device)
            observations = data["observations"].index_select(0, indices)
            old_values = data["old_values"].index_select(0, indices)
            returns = data["returns"].index_select(0, indices)
            values = model.get_value(observations)
            value_loss, value_clip_fraction = _critic_value_loss(
                values,
                old_values,
                returns,
                value_clip_range,
            )

            optimizer.zero_grad()
            (args.value_coefficient * value_loss).backward()
            _clear_parameter_gradients(actor_parameters)
            critic_grad_norm = _parameter_gradient_norm(critic_parameters)
            torch.nn.utils.clip_grad_norm_(
                critic_parameters, args.maximum_gradient_norm
            )
            optimizer.step()

            batch_count = int(indices.shape[0])
            critic_totals["value_loss"] += (
                float(value_loss.item()) * batch_count
            )
            critic_totals["value_clip_fraction"] += (
                float(value_clip_fraction.item()) * batch_count
            )
            critic_totals["critic_grad_norm"] += (
                critic_grad_norm * batch_count
            )
            critic_count += batch_count
            critic_minibatches_completed += 1
        critic_full_epochs_completed += 1

    metrics = dict(
        (name, float(value) / float(max(actor_count, 1)))
        for name, value in actor_totals.items()
    )
    metrics.update(dict(
        (name, float(value) / float(max(critic_count, 1)))
        for name, value in critic_totals.items()
    ))
    final_policy_kl = _full_rollout_policy_kl(
        model,
        data["observations"],
        data["actions"],
        stages,
        data["old_log_probabilities"],
        args.batch_size,
    )
    policy_kl_max = max(policy_kl_max, final_policy_kl)
    metrics["policy_kl"] = float(final_policy_kl)
    metrics["policy_kl_batch_mean"] = float(
        metrics.pop("policy_kl_batch", 0.0)
    )
    metrics["reference_kl_update_mean"] = float(
        metrics.get("reference_kl", 0.0)
    )
    metrics["reference_action_kl_update_mean"] = float(
        metrics.get("reference_action_kl", 0.0)
    )
    metrics["reference_stage_kl_update_mean"] = float(
        metrics.get("reference_stage_kl", 0.0)
    )
    (
        final_reference_action_kl,
        final_reference_stage_kl,
        final_reference_kl,
    ) = _full_rollout_reference_kl(
        model,
        reference_model,
        data["observations"],
        args.reference_stage_coefficient,
        args.batch_size,
    )
    # Adapt the reference coefficient from the final actor, not an average
    # over transient policies observed during minibatch optimization.
    metrics["reference_action_kl"] = final_reference_action_kl
    metrics["reference_stage_kl"] = final_reference_stage_kl
    metrics["reference_kl"] = final_reference_kl
    actor_grad_norm = float(metrics.get("actor_grad_norm", 0.0))
    critic_grad_norm = float(metrics.get("critic_grad_norm", 0.0))
    metrics["global_grad_norm"] = (
        actor_grad_norm * float(actor_count)
        + critic_grad_norm * float(critic_count)
    ) / float(max(actor_count + critic_count, 1))
    for name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "teacher_kl",
            "teacher_action_kl",
            "teacher_stage_kl",
            "reference_kl",
            "reference_action_kl",
            "reference_stage_kl",
            "policy_kl",
            "policy_kl_batch_mean",
            "clip_fraction",
            "value_clip_fraction",
            "actor_grad_norm",
            "critic_grad_norm",
            "global_grad_norm",
            "action_abs"):
        metrics.setdefault(name, 0.0)
    with torch.no_grad():
        values_after = model.get_value(data["observations"]).detach()
    metrics["value_target_mean"] = float(value_targets.mean().item())
    metrics["value_target_std"] = float(
        value_targets.std(unbiased=False).item()
    )
    metrics["value_prediction_mean_before"] = float(
        values_before.mean().item()
    )
    metrics["value_prediction_mean_after"] = float(
        values_after.mean().item()
    )
    metrics["value_rmse_before"] = value_rmse_before
    metrics["value_rmse"] = float(torch.sqrt(torch.mean(
        (values_after - value_targets).pow(2)
    )).item())
    metrics["value_explained_variance_before"] = (
        value_explained_variance_before
    )
    metrics["value_explained_variance_after"] = _explained_variance(
        values_after, value_targets
    )
    metrics["kl_early_stop"] = float(stopped_early)
    metrics["epochs_completed"] = float(epochs_completed)
    metrics["full_epochs_completed"] = float(full_epochs_completed)
    metrics["minibatches_completed"] = float(minibatches_completed)
    minibatches_expected = (
        int(math.ceil(float(buffer.size) / float(args.batch_size)))
        * int(args.ppo_epochs)
    )
    metrics["minibatches_expected"] = float(minibatches_expected)
    metrics["update_fraction"] = float(minibatches_completed) / float(max(
        minibatches_expected, 1
    ))
    metrics["critic_full_epochs_completed"] = float(
        critic_full_epochs_completed
    )
    metrics["critic_minibatches_completed"] = float(
        critic_minibatches_completed
    )
    metrics["critic_minibatches_expected"] = float(
        minibatches_expected
    )
    metrics["critic_update_fraction"] = float(
        critic_minibatches_completed
    ) / float(max(minibatches_expected, 1))
    metrics["value_clip_range"] = float(value_clip_range)
    metrics["value_clip_enabled"] = float(value_clip_range > 0.0)
    metrics["policy_kl_max"] = float(policy_kl_max)
    metrics["policy_kl_batch_max"] = float(policy_kl_batch_max)
    metrics["policy_kl_stop_value"] = float(policy_kl_stop_value)
    metrics["policy_kl_epoch_values"] = list(policy_kl_epoch_values)
    metrics["policy_kl_minibatch_values"] = list(
        policy_kl_minibatch_values
    )
    metrics["policy_kl_check_count"] = float(
        len(policy_kl_minibatch_values)
    )
    metrics["policy_kl_stop_epoch"] = float(policy_kl_stop_epoch)
    metrics["policy_kl_stop_minibatch"] = float(
        policy_kl_stop_minibatch
    )
    metrics["policy_kl_stop_global_minibatch"] = float(
        policy_kl_stop_global_minibatch
    )
    metrics["policy_kl_threshold"] = float(
        args.target_policy_kl * args.policy_kl_stop_multiplier
    )
    return metrics


def _full_rollout_policy_kl(
        model,
        observations,
        actions,
        stages,
        old_log_probabilities,
        batch_size):
    """Measure the current-vs-rollout policy KL on every rollout sample."""
    total = 0.0
    count = 0
    batch_size = max(int(batch_size), 1)
    with torch.no_grad():
        for start in range(0, int(observations.shape[0]), batch_size):
            stop = min(start + batch_size, int(observations.shape[0]))
            logp, unused_entropy, unused_values, unused_mean, unused_logits = (
                model.evaluate_actions(
                    observations[start:stop],
                    actions[start:stop],
                    stages[start:stop],
                )
            )
            log_ratio = logp - old_log_probabilities[start:stop]
            ratio = torch.exp(log_ratio)
            approximate_kl = (ratio - 1.0) - log_ratio
            total += float(approximate_kl.sum().item())
            count += int(approximate_kl.shape[0])
    return total / float(max(count, 1))


def _full_rollout_reference_kl(
        model,
        reference_model,
        observations,
        stage_coefficient,
        batch_size):
    """Measure frozen-reference KL using the final actor on all samples."""
    action_total = 0.0
    stage_total = 0.0
    count = 0
    batch_size = max(int(batch_size), 1)
    with torch.no_grad():
        for start in range(0, int(observations.shape[0]), batch_size):
            stop = min(start + batch_size, int(observations.shape[0]))
            action_kl, stage_kl = model.reference_kl(
                observations[start:stop], reference_model
            )
            action_total += float(action_kl.sum().item())
            stage_total += float(stage_kl.sum().item())
            count += int(action_kl.shape[0])
    action_mean = action_total / float(max(count, 1))
    stage_mean = stage_total / float(max(count, 1))
    combined = action_mean + float(stage_coefficient) * stage_mean
    return action_mean, stage_mean, combined


def _balanced_ppo_minibatch_indices(size, batch_size):
    """Shuffle once and split into complete minibatches of similar size."""
    size = int(size)
    batch_size = int(batch_size)
    if size <= 0:
        raise ValueError("PPO minibatch size source must be positive")
    if batch_size <= 0:
        raise ValueError("PPO batch size must be positive")
    batch_count = int(math.ceil(float(size) / float(batch_size)))
    permutation = np.random.permutation(size)
    for indices in np.array_split(permutation, batch_count):
        if len(indices) > 0:
            yield indices


def _clear_parameter_gradients(parameters):
    """Keep inactive Adam parameters out of a phase-specific optimizer step."""
    for parameter in parameters:
        parameter.grad = None


def _parameter_gradient_norm(parameters):
    """Return the pre-clipping L2 norm without changing any gradient."""
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        squared_norm += float(torch.sum(gradient * gradient).item())
    return float(math.sqrt(max(squared_norm, 0.0)))


def _resolved_value_clip_range(args):
    """Return the critic clip range without coupling it to policy clipping.

    Older commands and checkpoints did not define ``value_clip_range``.  For
    them, retain the legacy behavior by falling back to ``clip_ratio``.  A
    configured value of zero explicitly disables value clipping.
    """
    configured = getattr(args, "value_clip_range", None)
    if configured is None:
        return float(args.clip_ratio)
    return float(configured)


def _critic_value_loss(values, old_values, returns, value_clip_range):
    """Compute PPO critic loss and the fraction affected by value clipping.

    Policy ratios are dimensionless, whereas values are expressed in return
    units.  Reusing the policy clip ratio (for example 0.1) for targets whose
    standard deviation is tens of reward units can nearly freeze the critic.
    A non-positive range therefore selects ordinary squared-error regression.
    """
    clip_range = float(value_clip_range)
    if clip_range <= 0.0:
        loss = 0.5 * (values - returns).pow(2).mean()
        clip_fraction = torch.zeros(
            (), dtype=values.dtype, device=values.device
        )
        return loss, clip_fraction
    clipped_values = old_values + torch.clamp(
        values - old_values,
        -clip_range,
        clip_range,
    )
    clip_fraction = (
        torch.abs(values - old_values) > clip_range
    ).float().mean()
    loss = 0.5 * torch.max(
        (values - returns).pow(2),
        (clipped_values - returns).pow(2),
    ).mean()
    return loss, clip_fraction


def _explained_variance(predictions, targets):
    """Return 1 - Var[target - prediction] / Var[target]."""
    target_variance = float(targets.var(unbiased=False).item())
    if target_variance <= 1.0e-12:
        return 0.0
    residual_variance = float(
        (targets - predictions).var(unbiased=False).item()
    )
    return float(1.0 - residual_variance / target_variance)


def _normalize_ppo_advantages(advantages, stages=None):
    """Normalize returns globally or independently within semantic stages."""
    if stages is None:
        return (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)
    if tuple(stages.shape) != tuple(advantages.shape):
        raise ValueError("advantage stages must match advantage shape")
    normalized = torch.zeros_like(advantages)
    global_normalized = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)
    for stage in range(len(DAGGER_STAGE_NAMES)):
        mask = stages == stage
        count = int(mask.sum().item())
        if count < 2:
            normalized[mask] = global_normalized[mask]
            continue
        values = advantages[mask]
        normalized[mask] = (
            values - values.mean()
        ) / (values.std(unbiased=False) + 1e-8)
    return normalized


def _linear_coefficient(initial, final, step, decay_steps):
    if int(decay_steps) <= 0:
        return float(final)
    fraction = min(max(float(step) / float(decay_steps), 0.0), 1.0)
    return float(initial) + fraction * (float(final) - float(initial))


def _freeze_reference(reference_model):
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)


def _adapt_kl_coefficient(
        coefficient,
        measured_kl,
        target_kl,
        minimum_coefficient,
        maximum_coefficient,
        adaptation_factor,
        tolerance):
    coefficient = float(coefficient)
    measured_kl = float(measured_kl)
    target_kl = float(target_kl)
    if target_kl <= 0.0:
        return coefficient
    if measured_kl > target_kl * float(tolerance):
        coefficient *= float(adaptation_factor)
    elif measured_kl < target_kl / float(tolerance):
        coefficient /= float(adaptation_factor)
    return float(np.clip(
        coefficient,
        float(minimum_coefficient),
        float(maximum_coefficient),
    ))


def _stage_name(stage):
    stage = int(stage)
    if 0 <= stage < len(DAGGER_STAGE_NAMES):
        return DAGGER_STAGE_NAMES[stage]
    return "UNKNOWN_{}".format(stage)


def _counter_means(sums, counts):
    return dict(
        (
            name,
            float(sums.get(name, 0.0)) / float(max(count, 1)),
        )
        for name, count in counts.items()
    )


def _stage_term_means(sums, stage_counts):
    means = {}
    for key, value in sums.items():
        stage_name, unused_separator, term_name = str(key).partition(":")
        means.setdefault(stage_name, {})[term_name] = (
            float(value) / float(max(stage_counts.get(stage_name, 0), 1))
        )
    return means


def _mean_abs(value):
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return 0.0
    return float(np.mean(np.abs(array)))


def _episode_rates(episodes, successes, collisions, timeouts):
    total = int(sum(episodes.values()))
    if total <= 0:
        return {
            "episodes": 0,
            "success": None,
            "collision": None,
            "timeout": None,
        }
    return {
        "episodes": total,
        "success": round(float(sum(successes.values())) / total, 3),
        "collision": round(float(sum(collisions.values())) / total, 3),
        "timeout": round(float(sum(timeouts.values())) / total, 3),
    }


def _checkpoint_discount_mode(checkpoint):
    contract = checkpoint.get("discount_contract", {})
    if isinstance(contract, dict) and contract.get("mode"):
        return str(contract["mode"])
    return "legacy_high_step"


def _discount_reference_low_steps(args, environment_metadata):
    if str(args.discount_mode) != "smdp":
        return 1.0
    configured = float(args.smdp_discount_reference_low_steps)
    if configured > 0.0:
        return configured
    reference = float(environment_metadata.get("high_level_interval", 0.0))
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError(
            "SMDP discounting requires a positive high_level_interval "
            "or --smdp-discount-reference-low-steps"
        )
    return reference


def _low_step_gamma(gamma, discount_mode, reference_low_steps):
    gamma = float(gamma)
    if str(discount_mode) != "smdp":
        return gamma
    return gamma ** (1.0 / float(reference_low_steps))


def _rollout_discount_diagnostics(
        buffer,
        discount_mode,
        gamma,
        gae_lambda,
        reference_low_steps):
    if buffer.size <= 0:
        raise RuntimeError("cannot diagnose an empty rollout")
    active = slice(0, buffer.size)
    durations = np.asarray(buffer.durations[active], dtype=np.float64)
    discounts = np.asarray(
        buffer.transition_discounts[active], dtype=np.float64
    )
    return {
        "mode": str(discount_mode),
        "option_reward": "environment_aggregate_option_reward",
        "gamma_per_reference": float(gamma),
        "reference_low_steps": float(reference_low_steps),
        "low_step_gamma": float(_low_step_gamma(
            gamma, discount_mode, reference_low_steps
        )),
        "gae_lambda_per_option": float(gae_lambda),
        "duration_mean": float(np.mean(durations)),
        "duration_std": float(np.std(durations)),
        "duration_min": float(np.min(durations)),
        "duration_max": float(np.max(durations)),
        "transition_discount_mean": float(np.mean(discounts)),
        "transition_discount_std": float(np.std(discounts)),
        "transition_discount_min": float(np.min(discounts)),
        "transition_discount_max": float(np.max(discounts)),
    }


def _save_checkpoint(
        path,
        model,
        optimizer,
        normalizer,
        total_steps,
        total_low_steps,
        update_index,
        args,
        environment_metadata,
        reference_model=None,
        reference_kl_coefficient=None,
        training_phase="ppo",
        completed_dagger_round=None,
        optimizer_role="ppo",
        terminal_blend_curriculum=None):
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    random_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        random_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    checkpoint = {
        "model": model.state_dict(),
        "reference_model": (
            model.state_dict()
            if reference_model is None
            else reference_model.state_dict()
        ),
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "observation_dim": model.OBS_DIM,
        "action_dim": model.ACTION_DIM,
        "subgoal_type_count": model.STAGE_COUNT,
        "policy_type": model.POLICY_TYPE,
        "hidden_sizes": list(args.hidden_sizes),
        "total_steps": int(total_steps),
        "total_low_steps": int(total_low_steps),
        "update_index": int(update_index),
        "training_phase": str(training_phase),
        "completed_dagger_round": (
            None
            if completed_dagger_round is None
            else int(completed_dagger_round)
        ),
        "optimizer_role": str(optimizer_role),
        "ppo_initial_checkpoint": (
            None
            if not getattr(args, "ppo_initial_checkpoint", None)
            else str(args.ppo_initial_checkpoint)
        ),
        "random_state": random_state,
        "reference_kl_coefficient": float(
            args.reference_kl_coefficient
            if reference_kl_coefficient is None
            else reference_kl_coefficient
        ),
        "checkpoint_contract": {
            "recovery_interval": int(
                getattr(args, "recovery_checkpoint_interval", 0)
            ),
            "numbered_selection_interval": int(args.checkpoint_interval),
            "recovery_path": _recovery_checkpoint_path(args.output),
        },
        "terminal_blend_curriculum": copy.deepcopy(
            terminal_blend_curriculum
            if terminal_blend_curriculum is not None
            else _new_terminal_blend_curriculum(args)
        ),
        "environment_metadata": dict(environment_metadata),
        "training_contract": (
            "markov_kl_dagger_two_layer_smdp_ppo_terminal_blend_annealing_v9"
        ),
        "discount_contract": {
            "mode": str(getattr(args, "discount_mode", "high_step")),
            "option_reward": "environment_aggregate_option_reward",
            "duration_source": "info.low_steps",
            "gamma_per_reference": float(args.gamma),
            "reference_low_steps": float(_discount_reference_low_steps(
                args, environment_metadata
            )),
            "low_step_gamma": float(_low_step_gamma(
                args.gamma,
                getattr(args, "discount_mode", "high_step"),
                _discount_reference_low_steps(args, environment_metadata),
            )),
            "gae_lambda_per_option": float(args.gae_lambda),
            "td_rule": "R_option+gamma_low**n*V_next-V",
            "gae_rule": "delta+gamma_low**n*lambda*next_advantage",
        },
        "kl_contract": {
            "teacher": "narrow_gaussian_forward_kl_on_student_states",
            "reference": "frozen_bc_forward_kl",
            "policy_update": (
                "full_rollout_kl_after_each_actor_minibatch"
            ),
            "critic_update": "independent_complete_epochs",
            "critic_value_clip_range": float(
                _resolved_value_clip_range(args)
            ),
            "critic_value_clip_enabled": bool(
                _resolved_value_clip_range(args) > 0.0
            ),
            "gradient_update": "separate_actor_critic_steps",
            "terminal_arm_bc_coefficient": float(
                args.warmstart_terminal_arm_coefficient
            ),
        },
        "dagger_contract": {
            "state_distribution": "mixed_student_teacher_rollout",
            "label_policy": "live_rule_teacher",
            "risk_takeover": (
                "stage_disagreement_or_terminal_action_deviation"
            ),
            "base_action_deviation_takeover": bool(
                args.dagger_base_action_deviation_takeover
            ),
            "bc_sampling": "stage_balanced_episode_uniform",
            "persistent_dataset": str(args.dagger_dataset_output),
            "persistence_interval": "each_completed_episode",
            "rounds": int(args.dagger_rounds),
            "episodes_per_round": int(args.dagger_episodes),
            "initial_teacher_probability": float(
                args.dagger_initial_teacher_probability
            ),
            "final_teacher_probability": float(
                args.dagger_final_teacher_probability
            ),
            "action_deviation_thresholds": dict(zip(
                DAGGER_STAGE_NAMES,
                _dagger_action_deviation_thresholds(args),
            )),
            "terminal_arm_deviation_threshold": float(
                args.dagger_terminal_arm_deviation_threshold
            ),
        },
    }
    temporary_path = "{}.tmp.{}".format(path, os.getpid())
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)
    print("basic high checkpoint saved: {}".format(path))


def _validate_checkpoint(checkpoint, model):
    expected = {
        "observation_dim": model.OBS_DIM,
        "action_dim": model.ACTION_DIM,
        "subgoal_type_count": model.STAGE_COUNT,
        "policy_type": model.POLICY_TYPE,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError("basic high checkpoint {} mismatch".format(key))


def _initialize_ppo_from_checkpoint(checkpoint, model, normalizer):
    """Load an imitation actor while retaining a fresh PPO critic.

    DAgger checkpoints contain an optimizer used by KL-BC and a critic that
    was never trained by returns.  A direct PPO initialization must therefore
    copy only actor/stage/exploration parameters plus the observation
    normalizer.  The model's freshly initialized critic and the caller's fresh
    Adam optimizer are deliberately preserved.
    """
    _validate_checkpoint(checkpoint, model)
    source_state = checkpoint["model"]
    target_state = model.state_dict()
    critic_prefixes = ("critic_backbone.", "critic.")
    for name, value in source_state.items():
        if name.startswith(critic_prefixes):
            continue
        if name not in target_state:
            raise ValueError(
                "basic high checkpoint model parameter {} mismatch".format(
                    name
                )
            )
        target_state[name] = value
    model.load_state_dict(target_state)
    normalizer.load_state_dict(checkpoint["normalizer"])


def _numbered_checkpoint_path(path, update_index):
    root, extension = os.path.splitext(path)
    return "{}.update_{:04d}{}".format(
        root, int(update_index), extension or ".pt"
    )


def _recovery_checkpoint_path(path):
    root, extension = os.path.splitext(path)
    return "{}.recovery_latest{}".format(root, extension or ".pt")


def _warmstart_checkpoint_path(path):
    root, extension = os.path.splitext(path)
    return "{}.bc{}".format(root, extension or ".pt")


def _dagger_fit_checkpoint_path(path, label):
    root, extension = os.path.splitext(path)
    safe_label = str(label).replace("/", "_").replace("\\", "_")
    return "{}.{}{}".format(root, safe_label, extension or ".pt")


def _last_checkpoint_path(path):
    root, extension = os.path.splitext(path)
    return "{}.last{}".format(root, extension or ".pt")


def _restore_random_state(random_state):
    if not random_state:
        return False
    required = ("python", "numpy", "torch")
    if any(name not in random_state for name in required):
        return False
    random.setstate(random_state["python"])
    np.random.set_state(random_state["numpy"])
    torch_state = random_state["torch"]
    if hasattr(torch_state, "cpu"):
        torch_state = torch_state.cpu()
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "torch_cuda" in random_state:
        torch.cuda.set_rng_state_all(random_state["torch_cuda"])
    return True


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
        "--fixed-detour-side", type=float, choices=[-1.0, 1.0], default=1.0
    )
    parser.add_argument(
        "--output",
        default="/tmp/mobile_arm_rl_training/fused_basic_high_ppo_seed789.pt",
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--ppo-initial-checkpoint",
        help=(
            "start PPO directly from an imitation checkpoint: load actor "
            "and normalizer, create a fresh critic and optimizer, and skip "
            "teacher collection, KL-BC, DAgger, and prerequisite gates"
        ),
    )
    parser.add_argument("--teacher-episodes", type=int, default=30)
    parser.add_argument("--teacher-max-attempts", type=int, default=60)
    parser.add_argument(
        "--warmstart-rounds",
        type=int,
        default=None,
        help="deprecated alias: total rounds including initial BC",
    )
    parser.add_argument("--warmstart-epochs", type=int, default=150)
    parser.add_argument("--warmstart-batch-size", type=int, default=256)
    parser.add_argument("--warmstart-learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmstart-stage-coefficient", type=float, default=0.5)
    parser.add_argument(
        "--warmstart-terminal-arm-coefficient", type=float, default=2.0
    )
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-episodes", type=int, default=20)
    parser.add_argument("--dagger-epochs", type=int, default=75)
    parser.add_argument(
        "--dagger-initial-teacher-probability", type=float, default=0.80
    )
    parser.add_argument(
        "--dagger-final-teacher-probability", type=float, default=0.20
    )
    parser.add_argument(
        "--dagger-action-deviation-threshold",
        type=float,
        default=None,
        help="deprecated scalar override for all three stage thresholds",
    )
    parser.add_argument(
        "--dagger-direct-action-deviation-threshold",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--dagger-detour-action-deviation-threshold",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--dagger-terminal-action-deviation-threshold",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--dagger-terminal-arm-deviation-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--dagger-base-action-deviation-takeover",
        action="store_true",
        help=(
            "also force teacher takeover on DIRECT/DETOUR continuous-action "
            "deviation; disabled by default so DAgger labels student-induced "
            "base tracking errors"
        ),
    )
    parser.add_argument(
        "--dagger-dataset-output",
        help=(
            "persistent latest .npz dataset; defaults below ~/"
            "mobile_arm_rl_training/datasets"
        ),
    )
    parser.add_argument(
        "--resume-dagger-dataset",
        help="load a previously saved aggregate dataset before KL fitting",
    )
    parser.add_argument(
        "--skip-pre-dagger-gate",
        action="store_true",
        help=(
            "skip the prerequisite teacher gate and the student gate before "
            "the first pending DAgger round; post-DAgger gates are retained"
        ),
    )
    parser.add_argument("--gate-episodes", type=int, default=10)
    parser.add_argument("--selection-episodes", type=int, default=10)
    parser.add_argument("--minimum-gate-success-rate", type=float, default=0.80)
    parser.add_argument("--maximum-gate-collisions", type=int, default=0)
    parser.add_argument(
        "--terminal-blend-annealing",
        action="store_true",
        help=(
            "increase TERMINAL student authority only after deterministic "
            "ability gates pass"
        ),
    )
    parser.add_argument(
        "--terminal-blend-schedule",
        type=float,
        nargs="+",
        default=[0.40, 0.60, 0.80, 1.00],
        help=(
            "strictly increasing TERMINAL student blend stages; 1.0 removes "
            "the nominal goal-direction blend"
        ),
    )
    parser.add_argument(
        "--terminal-blend-minimum-success-rate",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--terminal-blend-maximum-collisions",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--terminal-blend-required-gates",
        type=int,
        default=1,
        help=(
            "consecutive deterministic gates required before advancing one "
            "blend stage"
        ),
    )
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--maximum-rollout-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--discount-mode",
        choices=("high_step", "smdp"),
        default="high_step",
        help=(
            "high_step applies one gamma per upper-level decision; smdp "
            "applies gamma^(low_steps/reference_low_steps)"
        ),
    )
    parser.add_argument(
        "--smdp-discount-reference-low-steps",
        type=float,
        default=0.0,
        help=(
            "number of low-level steps represented by --gamma in SMDP "
            "mode; 0 uses environment high_level_interval"
        ),
    )
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument(
        "--value-clip-range",
        type=float,
        default=None,
        help=(
            "critic value clipping range in raw return units; omitted "
            "preserves the legacy --clip-ratio behavior and 0 disables "
            "critic value clipping"
        ),
    )
    parser.add_argument("--value-coefficient", type=float, default=0.50)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--teacher-coefficient", type=float, default=0.50)
    parser.add_argument("--teacher-final-coefficient", type=float, default=0.05)
    parser.add_argument("--teacher-stage-coefficient", type=float, default=0.20)
    parser.add_argument(
        "--teacher-terminal-weight",
        type=float,
        default=1.0,
        help=(
            "relative teacher-KL weight for TERMINAL samples during PPO"
        ),
    )
    parser.add_argument("--teacher-kl-std", type=float, default=0.10)
    parser.add_argument("--teacher-decay-steps", type=int, default=100000)
    parser.add_argument(
        "--reference-kl-coefficient", type=float, default=0.10
    )
    parser.add_argument(
        "--reference-stage-coefficient", type=float, default=0.20
    )
    parser.add_argument("--reference-kl-target", type=float, default=0.02)
    parser.add_argument(
        "--reference-kl-min-coefficient", type=float, default=0.01
    )
    parser.add_argument(
        "--reference-kl-max-coefficient", type=float, default=10.0
    )
    parser.add_argument(
        "--reference-kl-adaptation-factor", type=float, default=2.0
    )
    parser.add_argument(
        "--reference-kl-tolerance", type=float, default=1.5
    )
    parser.add_argument("--target-policy-kl", type=float, default=0.02)
    parser.add_argument(
        "--policy-kl-stop-multiplier", type=float, default=1.5
    )
    parser.add_argument("--maximum-gradient-norm", type=float, default=0.50)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--initial-log-std", type=float, default=-2.5)
    parser.add_argument(
        "--recovery-checkpoint-interval",
        type=int,
        default=1,
        help=(
            "atomically overwrite recovery_latest after this many PPO "
            "updates; 0 disables crash-recovery saves"
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help=(
            "save a numbered checkpoint and run deterministic selection "
            "after this many PPO updates; independent of recovery saves"
        ),
    )
    parser.add_argument("--reward-window", type=int, default=1000)
    parser.add_argument(
        "--stage-balanced-advantages",
        action="store_true",
        help=(
            "normalize PPO advantages independently for DIRECT, DETOUR, "
            "and TERMINAL so the long terminal phase cannot set the scale"
        ),
    )
    parser.add_argument("--update-normalizer", action="store_true")
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.dagger_dataset_output is None:
        output_stem = os.path.splitext(os.path.basename(args.output))[0]
        args.dagger_dataset_output = os.path.expanduser(os.path.join(
            "~",
            "mobile_arm_rl_training",
            "datasets",
            output_stem + ".dataset.latest.npz",
        ))
    else:
        args.dagger_dataset_output = os.path.abspath(
            os.path.expanduser(args.dagger_dataset_output)
        )
    if args.resume_dagger_dataset:
        args.resume_dagger_dataset = os.path.abspath(
            os.path.expanduser(args.resume_dagger_dataset)
        )
        if not os.path.isfile(args.resume_dagger_dataset):
            parser.error("--resume-dagger-dataset does not exist")
    initialization_inputs = [
        bool(args.resume),
        bool(args.ppo_initial_checkpoint),
        bool(args.resume_dagger_dataset),
    ]
    if sum(initialization_inputs) > 1:
        parser.error(
            "--resume, --ppo-initial-checkpoint, and "
            "--resume-dagger-dataset are mutually exclusive"
        )
    if args.ppo_initial_checkpoint:
        args.ppo_initial_checkpoint = os.path.abspath(
            os.path.expanduser(args.ppo_initial_checkpoint)
        )
        if not os.path.isfile(args.ppo_initial_checkpoint):
            parser.error("--ppo-initial-checkpoint does not exist")
    if args.warmstart_rounds is not None:
        if args.warmstart_rounds <= 0:
            parser.error("--warmstart-rounds must be positive")
        args.dagger_rounds = max(int(args.warmstart_rounds) - 1, 0)
    if args.total_steps <= 0:
        parser.error("--total-steps must be positive")
    if args.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive")
    if args.maximum_rollout_steps < args.rollout_steps:
        parser.error("--maximum-rollout-steps must be at least rollout-steps")
    if args.recovery_checkpoint_interval < 0:
        parser.error("--recovery-checkpoint-interval must be non-negative")
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint-interval must be non-negative")
    if args.batch_size <= 0 or args.ppo_epochs <= 0:
        parser.error("batch size and PPO epochs must be positive")
    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must be in (0, 1]")
    if not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("--gae-lambda must be in [0, 1]")
    if args.smdp_discount_reference_low_steps < 0.0:
        parser.error(
            "--smdp-discount-reference-low-steps must be non-negative"
        )
    if args.value_clip_range is not None and args.value_clip_range < 0.0:
        parser.error("--value-clip-range must be non-negative")
    if (
            args.teacher_episodes <= 0
            or args.teacher_max_attempts < args.teacher_episodes
            or args.warmstart_epochs <= 0
            or args.warmstart_batch_size <= 0
            or args.dagger_rounds < 0
            or args.dagger_episodes <= 0
            or args.dagger_epochs <= 0
            or args.gate_episodes <= 0
            or args.selection_episodes <= 0):
        parser.error("invalid warm-start or gate configuration")
    if not (
            0.0 <= args.dagger_final_teacher_probability <= 1.0
            and 0.0 <= args.dagger_initial_teacher_probability <= 1.0):
        parser.error("DAgger teacher probabilities must be in [0, 1]")
    if (
            args.dagger_initial_teacher_probability
            < args.dagger_final_teacher_probability):
        parser.error("DAgger teacher probability must not increase")
    dagger_thresholds = list(_dagger_action_deviation_thresholds(args)) + [
        args.dagger_terminal_arm_deviation_threshold
    ]
    if min(dagger_thresholds) <= 0.0:
        parser.error("DAgger action deviation thresholds must be positive")
    if not 0.0 <= args.minimum_gate_success_rate <= 1.0:
        parser.error("--minimum-gate-success-rate must be in [0, 1]")
    try:
        _validate_terminal_blend_schedule(args.terminal_blend_schedule)
    except ValueError as error:
        parser.error(str(error))
    if not 0.0 <= args.terminal_blend_minimum_success_rate <= 1.0:
        parser.error(
            "--terminal-blend-minimum-success-rate must be in [0, 1]"
        )
    if args.terminal_blend_maximum_collisions < 0:
        parser.error(
            "--terminal-blend-maximum-collisions must be non-negative"
        )
    if args.terminal_blend_required_gates <= 0:
        parser.error("--terminal-blend-required-gates must be positive")
    if min(
            args.teacher_coefficient,
            args.teacher_final_coefficient,
            args.teacher_stage_coefficient,
            args.teacher_terminal_weight,
            args.reference_kl_coefficient,
            args.reference_stage_coefficient,
            args.reference_kl_target,
            args.reference_kl_min_coefficient,
            args.reference_kl_max_coefficient,
            args.target_policy_kl,
            args.warmstart_terminal_arm_coefficient) < 0.0:
        parser.error("KL regularization values must be non-negative")
    if args.teacher_terminal_weight <= 0.0:
        parser.error("--teacher-terminal-weight must be positive")
    if args.teacher_kl_std <= 0.0:
        parser.error("--teacher-kl-std must be positive")
    if args.teacher_decay_steps < 0:
        parser.error("--teacher-decay-steps must be non-negative")
    if (
            args.reference_kl_min_coefficient
            > args.reference_kl_max_coefficient):
        parser.error("reference KL coefficient bounds are invalid")
    if args.reference_kl_adaptation_factor <= 1.0:
        parser.error("--reference-kl-adaptation-factor must exceed 1")
    if args.reference_kl_tolerance <= 1.0:
        parser.error("--reference-kl-tolerance must exceed 1")
    if args.policy_kl_stop_multiplier <= 1.0:
        parser.error("--policy-kl-stop-multiplier must exceed 1")
    return args


if __name__ == "__main__":
    main()
