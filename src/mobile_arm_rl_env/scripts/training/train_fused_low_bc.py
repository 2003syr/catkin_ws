#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Behavior-clone the coordinated 66-D/8-D base-arm teacher."""

import argparse
import copy
import os
import random
import sys

import numpy as np

try:
    import torch
except ImportError as error:
    raise RuntimeError(
        "Fused low-level BC requires Python 3 with PyTorch"
    ) from error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_low_actor_critic import (
    FusedLowActorCritic,
    HRL4INArmTransferredFusedPolicy,
)
from training.fused_low_dataset import (
    ACTION_DIM,
    DEFAULT_ACTION_LOSS_WEIGHTS,
    OBSERVATION_DIM,
    PHASE_NAMES,
    episode_train_validation_split,
    phase_balanced_epoch_indices,
    validate_action_mask_contract,
    validate_action_loss_weights,
)
from training.hrl4in_low_actor_critic import (
    RunningObservationNormalizer,
)


NORMALIZED_DIM = 52
SUPPORTED_TEACHERS = {"coordinated_rule_teacher"}
SUPPORTED_POLICY_TARGETS = {"safe_teacher_actions"}


def main():
    args = _parse_arguments()
    _set_seed(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    (
        observations,
        actions,
        episode_ids,
        phase_ids,
        teacher_type,
        policy_target,
        action_names,
        observation_semantics,
        action_semantics,
        dataset_paths,
        dataset_sample_counts,
    ) = _load_datasets(args.dataset)
    action_mask_contract = _validate_dataset(
        observations,
        actions,
        episode_ids,
        phase_ids,
        teacher_type,
        policy_target,
    )
    structured_action_masks = bool(
        action_mask_contract["mode"] == "three_stage"
    )
    action_loss_weights = validate_action_loss_weights(
        args.action_loss_weights
    )
    (
        train_indices,
        validation_indices,
        train_episode_ids,
        validation_episode_ids,
    ) = episode_train_validation_split(
        episode_ids,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    normalizer = RunningObservationNormalizer(
        observation_dim=OBSERVATION_DIM,
        normalized_dim=NORMALIZED_DIM,
    )
    normalizer.update(observations[train_indices])
    normalized = normalizer.normalize(observations)
    observation_tensor = torch.from_numpy(normalized).to(device)
    action_tensor = torch.from_numpy(actions).to(device)
    weight_tensor = torch.from_numpy(action_loss_weights).to(device)

    arm_checkpoint_path = None
    arm_source_steps = None
    arm_hidden_sizes = None
    arm_normalizer_state = None
    arm_anchor = []
    if args.arm_checkpoint:
        arm_checkpoint_path = os.path.abspath(
            os.path.expanduser(args.arm_checkpoint)
        )
        arm_checkpoint = torch.load(
            arm_checkpoint_path,
            map_location="cpu",
        )
        model = HRL4INArmTransferredFusedPolicy.from_hrl4in_checkpoint(
            checkpoint=arm_checkpoint,
            base_hidden_sizes=args.hidden_sizes,
            fused_normalizer_state=normalizer.state_dict(),
            arm_start_distance=args.arm_start_distance,
            base_stop_distance=args.base_stop_distance,
            arm_max_subgoal=args.arm_max_subgoal,
        ).to(device)
        base_parameters = model.trainable_base_parameters()
        if args.finetune_transferred_arm:
            model.unfreeze_arm()
            arm_parameters = model.trainable_arm_parameters()
            actor_parameters = base_parameters + arm_parameters
            optimizer_parameter_groups = [
                {
                    "params": base_parameters,
                    "lr": args.learning_rate,
                },
                {
                    "params": arm_parameters,
                    "lr": args.arm_learning_rate,
                },
            ]
            optimization_weight_tensor = weight_tensor
            arm_anchor = [
                (parameter, parameter.detach().clone())
                for parameter in arm_parameters
            ]
        else:
            actor_parameters = base_parameters
            optimizer_parameter_groups = actor_parameters
            optimization_weight_tensor = weight_tensor.clone()
            optimization_weight_tensor[2:8] = 0.0
        arm_source_steps = int(arm_checkpoint.get("total_steps", 0))
        arm_hidden_sizes = list(model.arm_hidden_sizes)
        arm_normalizer_state = arm_checkpoint["normalizer"]
        policy_type = model.POLICY_TYPE
        print(
            "transferred_hrl4in_arm checkpoint={} source_steps={} "
            "arm_hidden_sizes={} frozen_parameters={} "
            "base_hidden_sizes={} arm_blend={:.3f}->{:.3f} "
            "arm_max_subgoal={:.3f} arm_finetune={} arm_lr={:.2e} "
            "arm_anchor={:.2e}".format(
                arm_checkpoint_path,
                arm_source_steps,
                arm_hidden_sizes,
                sum(
                    parameter.numel()
                    for parameter in model.arm_backbone.parameters()
                ) + sum(
                    parameter.numel()
                    for parameter in model.arm_mean.parameters()
                ),
                list(args.hidden_sizes),
                args.arm_start_distance,
                args.base_stop_distance,
                args.arm_max_subgoal,
                str(args.finetune_transferred_arm),
                args.arm_learning_rate,
                args.arm_anchor_coefficient,
            )
        )
    else:
        model = FusedLowActorCritic(
            hidden_sizes=args.hidden_sizes,
            initial_log_std=args.initial_log_std,
        ).to(device)
        actor_parameters = (
            list(model.actor_backbone.parameters())
            + list(model.actor_mean.parameters())
        )
        optimizer_parameter_groups = actor_parameters
        optimization_weight_tensor = weight_tensor
        policy_type = "fused_low_bc"
    optimizer = torch.optim.Adam(
        optimizer_parameter_groups,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_selection_loss = float("inf")
    best_state = None
    best_epoch = 0
    sampling_random = np.random.RandomState(args.seed + 1701)

    print(
        "fused_bc_datasets={} dataset_samples={} samples={} episodes={} "
        "train_episodes={} "
        "validation_episodes={} teacher={} target={} device={} "
        "action_mask_contract={} phase_sampling_exponent={:.2f}".format(
            dataset_paths,
            dataset_sample_counts,
            observations.shape[0],
            np.unique(episode_ids).size,
            train_episode_ids.size,
            validation_episode_ids.size,
            teacher_type,
            policy_target,
            device,
            action_mask_contract["mode"],
            (
                args.phase_sampling_exponent
                if structured_action_masks else 0.0
            ),
        )
    )
    print(
        "fused_bc_action_weights={}".format(
            np.round(action_loss_weights, 4).tolist()
        )
    )
    if args.arm_checkpoint:
        print(
            "fused_bc_optimization_weights={} "
            "(arm_mode={}; full weights remain in validation)"
            .format(
                np.round(
                    optimization_weight_tensor.detach().cpu().numpy(),
                    4,
                ).tolist(),
                (
                    "low_lr_finetune"
                    if args.finetune_transferred_arm
                    else "frozen"
                ),
            )
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        if structured_action_masks:
            order = phase_balanced_epoch_indices(
                train_indices,
                phase_ids,
                args.phase_sampling_exponent,
                sampling_random,
            )
        else:
            order = np.random.permutation(train_indices)
        batch_losses = []
        batch_anchor_losses = []
        for start in range(0, order.size, args.batch_size):
            batch_indices = order[start:start + args.batch_size]
            index_tensor = torch.from_numpy(
                np.asarray(batch_indices, dtype=np.int64)
            ).to(device)
            predicted = model.deterministic_action(
                observation_tensor.index_select(0, index_tensor)
            )
            target = action_tensor.index_select(0, index_tensor)
            loss = _weighted_mse_tensor(
                predicted,
                target,
                optimization_weight_tensor,
            )
            anchor_loss = _parameter_anchor_loss(
                arm_anchor,
                observation_tensor,
            )
            loss = (
                loss
                + args.arm_anchor_coefficient * anchor_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor_parameters,
                args.maximum_gradient_norm,
            )
            optimizer.step()
            batch_losses.append(float(loss.item()))
            batch_anchor_losses.append(float(anchor_loss.item()))

        validation_loss, validation_per_action = _evaluate_loss(
            model,
            observation_tensor,
            action_tensor,
            validation_indices,
            weight_tensor,
        )
        validation_phase_losses = _evaluate_phase_losses(
            model,
            observation_tensor,
            action_tensor,
            validation_indices,
            phase_ids,
            weight_tensor,
        )
        selection_loss = (
            _finite_mean(validation_phase_losses)
            if structured_action_masks else validation_loss
        )
        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if (
                epoch == 1
                or epoch == args.epochs
                or epoch % args.log_interval == 0):
            print(
                "fused_bc_epoch={:03d}/{:03d} train_weighted_mse={:.8f} "
                "arm_anchor_mse={:.8f} "
                "validation_weighted_mse={:.8f} "
                "validation_phase_mse={} "
                "selection_mse={:.8f} validation_action_mse={}".format(
                    epoch,
                    args.epochs,
                    float(np.mean(batch_losses)),
                    float(np.mean(batch_anchor_losses)),
                    validation_loss,
                    _phase_loss_dict(validation_phase_losses),
                    selection_loss,
                    np.round(validation_per_action, 7).tolist(),
                )
            )

    if best_state is None:
        raise RuntimeError("BC training did not produce a model")
    model.load_state_dict(best_state)
    train_loss, train_per_action = _evaluate_loss(
        model,
        observation_tensor,
        action_tensor,
        train_indices,
        weight_tensor,
    )
    validation_loss, validation_per_action = _evaluate_loss(
        model,
        observation_tensor,
        action_tensor,
        validation_indices,
        weight_tensor,
    )
    train_phase_losses = _evaluate_phase_losses(
        model,
        observation_tensor,
        action_tensor,
        train_indices,
        phase_ids,
        weight_tensor,
    )
    validation_phase_losses = _evaluate_phase_losses(
        model,
        observation_tensor,
        action_tensor,
        validation_indices,
        phase_ids,
        weight_tensor,
    )
    phase_balanced_validation_loss = _finite_mean(
        validation_phase_losses
    )

    output = os.path.abspath(os.path.expanduser(args.output))
    output_directory = os.path.dirname(output)
    if output_directory and not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    torch.save({
        "format_version": 1,
        "policy_type": policy_type,
        "model": model.state_dict(),
        "normalizer": normalizer.state_dict(),
        "arguments": vars(args),
        "hidden_sizes": list(args.hidden_sizes),
        "initial_log_std": float(args.initial_log_std),
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "normalized_dim": NORMALIZED_DIM,
        "teacher_type": teacher_type,
        "policy_target": policy_target,
        "critic_trained": False,
        "sample_count": int(observations.shape[0]),
        "episode_count": int(np.unique(episode_ids).size),
        "train_episode_ids": train_episode_ids,
        "validation_episode_ids": validation_episode_ids,
        "best_epoch": int(best_epoch),
        "action_loss_weights": action_loss_weights,
        "action_mask_contract": action_mask_contract["mode"],
        "phase_sampling_exponent": float(
            args.phase_sampling_exponent
            if structured_action_masks else 0.0
        ),
        "train_phase_weighted_mse": train_phase_losses,
        "validation_phase_weighted_mse": validation_phase_losses,
        "validation_phase_balanced_mse": float(
            phase_balanced_validation_loss
        ),
        "train_weighted_mse": float(train_loss),
        "validation_weighted_mse": float(validation_loss),
        "train_action_mse": train_per_action,
        "validation_action_mse": validation_per_action,
        "action_names": action_names,
        "observation_semantics": observation_semantics,
        "action_semantics": action_semantics,
        "dataset_paths": dataset_paths,
        "dataset_sample_counts": dataset_sample_counts,
        "arm_source_checkpoint": arm_checkpoint_path,
        "arm_source_total_steps": arm_source_steps,
        "arm_hidden_sizes": arm_hidden_sizes,
        "arm_normalizer": arm_normalizer_state,
        "arm_transferred_frozen": bool(
            args.arm_checkpoint
            and not args.finetune_transferred_arm
        ),
        "arm_finetuned": bool(
            args.arm_checkpoint
            and args.finetune_transferred_arm
        ),
        "arm_learning_rate": float(args.arm_learning_rate),
        "arm_anchor_coefficient": float(
            args.arm_anchor_coefficient
        ),
        "arm_start_distance": float(args.arm_start_distance),
        "base_stop_distance": float(args.base_stop_distance),
        "arm_max_subgoal": float(args.arm_max_subgoal),
    }, output)
    print(
        "fused_bc_model={} best_epoch={} samples={} "
        "train_weighted_mse={:.8f} validation_weighted_mse={:.8f} "
        "train_action_mse={} validation_action_mse={}".format(
            output,
            best_epoch,
            observations.shape[0],
            train_loss,
            validation_loss,
            np.round(train_per_action, 8).tolist(),
            np.round(validation_per_action, 8).tolist(),
        )
    )


def _load_dataset(path):
    dataset = np.load(
        os.path.abspath(os.path.expanduser(path)),
        allow_pickle=False,
    )
    try:
        return (
            np.asarray(dataset["observations"], dtype=np.float32),
            np.asarray(dataset["safe_teacher_actions"], dtype=np.float32),
            np.asarray(dataset["episode_ids"], dtype=np.int32),
            np.asarray(dataset["phase_ids"], dtype=np.int8),
            _decode_text(dataset["teacher_type"][0]),
            _decode_text(dataset["policy_target"][0]),
            _decode_text_array(dataset["action_names"]),
            _decode_text(dataset["observation_semantics"][0]),
            _decode_text(dataset["action_semantics"][0]),
        )
    finally:
        dataset.close()


def _load_datasets(paths):
    if not paths:
        raise ValueError("at least one dataset is required")
    loaded = []
    for path in paths:
        loaded.append(_load_dataset(path))

    reference = loaded[0]
    observations = []
    actions = []
    episode_ids = []
    phase_ids = []
    sample_counts = []
    episode_offset = 0
    for index, values in enumerate(loaded):
        (
            dataset_observations,
            dataset_actions,
            dataset_episode_ids,
            dataset_phase_ids,
            teacher_type,
            policy_target,
            action_names,
            observation_semantics,
            action_semantics,
        ) = values
        if (
                teacher_type != reference[4]
                or policy_target != reference[5]
                or action_names != reference[6]
                or observation_semantics != reference[7]
                or action_semantics != reference[8]):
            raise ValueError(
                "dataset {} contract differs from the first dataset".format(
                    paths[index]
                )
            )
        unique_episodes = np.unique(dataset_episode_ids)
        episode_remap = dict(
            (int(value), episode_offset + remapped)
            for remapped, value in enumerate(unique_episodes)
        )
        remapped_episode_ids = np.asarray([
            episode_remap[int(value)]
            for value in dataset_episode_ids
        ], dtype=np.int32)
        episode_offset += unique_episodes.size
        observations.append(dataset_observations)
        actions.append(dataset_actions)
        episode_ids.append(remapped_episode_ids)
        phase_ids.append(dataset_phase_ids)
        sample_counts.append(int(dataset_observations.shape[0]))

    dataset_paths = [
        os.path.abspath(os.path.expanduser(path))
        for path in paths
    ]
    return (
        np.concatenate(observations, axis=0),
        np.concatenate(actions, axis=0),
        np.concatenate(episode_ids, axis=0),
        np.concatenate(phase_ids, axis=0),
        reference[4],
        reference[5],
        reference[6],
        reference[7],
        reference[8],
        dataset_paths,
        sample_counts,
    )


def _validate_dataset(
        observations,
        actions,
        episode_ids,
        phase_ids,
        teacher_type,
        policy_target):
    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIM:
        raise ValueError("observations must have shape (N, 66)")
    sample_count = observations.shape[0]
    if actions.shape != (sample_count, ACTION_DIM):
        raise ValueError("safe_teacher_actions must have shape (N, 8)")
    if episode_ids.shape != (sample_count,):
        raise ValueError("episode_ids must have shape (N,)")
    if phase_ids.shape != (sample_count,):
        raise ValueError("phase_ids must have shape (N,)")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations contain non-finite values")
    if not np.all(np.isfinite(actions)):
        raise ValueError("safe_teacher_actions contain non-finite values")
    if np.max(np.abs(actions)) > 1.0001:
        raise ValueError("safe_teacher_actions must remain in [-1, 1]")
    if np.unique(episode_ids).size < 2:
        raise ValueError("dataset must contain at least two episodes")
    if teacher_type not in SUPPORTED_TEACHERS:
        raise ValueError(
            "unsupported fused teacher: {}".format(teacher_type)
        )
    if policy_target not in SUPPORTED_POLICY_TARGETS:
        raise ValueError(
            "unsupported fused policy target: {}".format(policy_target)
        )
    return validate_action_mask_contract(
        observations,
        actions,
        phase_ids,
    )


def _weighted_mse_tensor(predicted, target, weights):
    squared = (predicted - target).pow(2)
    return (
        squared * weights.reshape(1, -1)
    ).sum(dim=-1).mean() / torch.clamp(weights.sum(), min=1.0e-8)


def _parameter_anchor_loss(anchor, reference_tensor):
    if not anchor:
        return reference_tensor.new_tensor(0.0)
    total = reference_tensor.new_tensor(0.0)
    count = 0
    for parameter, initial_value in anchor:
        total = total + (parameter - initial_value).pow(2).sum()
        count += parameter.numel()
    return total / float(max(count, 1))


def _evaluate_loss(
        model,
        observations,
        actions,
        indices,
        weights):
    model.eval()
    index_tensor = torch.from_numpy(
        np.asarray(indices, dtype=np.int64)
    ).to(observations.device)
    with torch.no_grad():
        predicted = model.deterministic_action(
            observations.index_select(0, index_tensor)
        )
        target = actions.index_select(0, index_tensor)
        loss = _weighted_mse_tensor(predicted, target, weights)
        per_action = (
            (predicted - target).pow(2).mean(dim=0).cpu().numpy()
        )
    return float(loss.item()), per_action.astype(np.float32)


def _evaluate_phase_losses(
        model,
        observations,
        actions,
        indices,
        phase_ids,
        weights):
    losses = np.full(len(PHASE_NAMES), float("nan"), dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    phase_ids = np.asarray(phase_ids, dtype=np.int32)
    for phase_index in range(len(PHASE_NAMES)):
        phase_indices = indices[phase_ids[indices] == phase_index]
        if phase_indices.size == 0:
            continue
        phase_loss, unused_per_action = _evaluate_loss(
            model,
            observations,
            actions,
            phase_indices,
            weights,
        )
        losses[phase_index] = phase_loss
    return losses


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("phase losses contain no finite values")
    return float(np.mean(finite))


def _phase_loss_dict(values):
    return dict(
        (name, None if not np.isfinite(values[index]) else round(
            float(values[index]), 8
        ))
        for index, name in enumerate(PHASE_NAMES)
    )


def _decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _decode_text_array(values):
    return [_decode_text(value) for value in np.asarray(values).reshape(-1)]


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a 66-D/8-D fused low-level BC policy.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        nargs="+",
        help="one or more compatible teacher/DAgger NPZ datasets",
    )
    parser.add_argument(
        "--output",
        default="/tmp/mobile_arm_rl_training/fused_low_bc.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 256],
    )
    parser.add_argument("--initial-log-std", type=float, default=-2.5)
    parser.add_argument(
        "--arm-checkpoint",
        help=(
            "optional 68-D/10-D HRL4IN checkpoint; its six arm outputs "
            "are embedded and frozen while the tracked-base branch trains"
        ),
    )
    parser.add_argument("--arm-start-distance", type=float, default=0.30)
    parser.add_argument("--base-stop-distance", type=float, default=0.04)
    parser.add_argument("--arm-max-subgoal", type=float, default=0.08)
    parser.add_argument(
        "--finetune-transferred-arm",
        action="store_true",
        help=(
            "fine-tune the transferred arm at a separate low learning rate "
            "against the current limit-aware teacher"
        ),
    )
    parser.add_argument("--arm-learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--arm-anchor-coefficient",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--phase-sampling-exponent",
        type=float,
        default=0.5,
        help=(
            "inverse-frequency exponent for structured three-stage data; "
            "0 disables balancing and 1 gives equal expected phase mass"
        ),
    )
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    parser.add_argument(
        "--action-loss-weights",
        type=float,
        nargs=ACTION_DIM,
        default=DEFAULT_ACTION_LOSS_WEIGHTS.tolist(),
        metavar=(
            "V",
            "OMEGA",
            "J1",
            "J2",
            "J3",
            "J4",
            "J5",
            "J6",
        ),
    )
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be non-negative")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in (0, 1)")
    if not 0.0 <= args.phase_sampling_exponent <= 1.0:
        raise ValueError("phase-sampling-exponent must be in [0, 1]")
    if args.maximum_gradient_norm <= 0.0:
        raise ValueError("maximum-gradient-norm must be positive")
    if args.arm_start_distance <= args.base_stop_distance:
        raise ValueError(
            "arm-start-distance must exceed base-stop-distance"
        )
    if args.arm_max_subgoal <= 0.0:
        raise ValueError("arm-max-subgoal must be positive")
    if args.arm_learning_rate <= 0.0:
        raise ValueError("arm-learning-rate must be positive")
    if args.arm_anchor_coefficient < 0.0:
        raise ValueError("arm-anchor-coefficient must be non-negative")
    if args.finetune_transferred_arm and not args.arm_checkpoint:
        raise ValueError(
            "finetune-transferred-arm requires arm-checkpoint"
        )
    if args.log_interval <= 0:
        raise ValueError("log-interval must be positive")
    return args


if __name__ == "__main__":
    main()
