#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Behavior-clone the pose-guided tracked-base subgoal controller."""

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
        "Planar tracking BC requires Python 3 with PyTorch"
    ) from error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.hrl4in_low_actor_critic import (
    RunningObservationNormalizer,
)
from training.planar_subgoal_actor_critic import (
    PlanarSubgoalActorCritic,
)

SCAN_START = 6
SCAN_END = 42
SCAN_DIM = SCAN_END - SCAN_START
SCAN_FIXED_MEAN = 0.5
SCAN_FIXED_VARIANCE = 1.0 / 12.0


def main():
    args = _parse_arguments()
    _set_seed(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    dataset = np.load(args.dataset, allow_pickle=False)
    try:
        observations = np.asarray(
            dataset["observations"], dtype=np.float32
        )
        teacher_actions = np.asarray(
            dataset["teacher_actions"], dtype=np.float32
        )
        episode_ids = np.asarray(
            dataset["episode_ids"], dtype=np.int32
        )
        episode_goal_indices = (
            np.asarray(
                dataset["accepted_goal_indices"],
                dtype=np.int32,
            )
            if "accepted_goal_indices" in dataset.files
            else None
        )
        teacher_type = _decode_dataset_text(
            dataset["teacher_type"][0]
        )
    finally:
        dataset.close()
    _validate_dataset(
        observations,
        teacher_actions,
        episode_ids,
        teacher_type,
        episode_goal_indices,
    )
    scan_augmentation_enabled = (
        teacher_type == "observable_rule_based_planar_tracker"
    )
    effective_augmentation_probability = (
        args.scan_augmentation_probability
        if scan_augmentation_enabled
        else 0.0
    )
    effective_lidar_validation_weight = (
        args.lidar_validation_weight
        if scan_augmentation_enabled
        else 0.0
    )
    print(
        "teacher_type={} samples={} episodes={} routes={} "
        "scan_augmentation_enabled={}".format(
            teacher_type,
            observations.shape[0],
            np.unique(episode_ids).size,
            (
                np.unique(episode_goal_indices).size
                if episode_goal_indices is not None
                else "unknown"
            ),
            scan_augmentation_enabled,
        )
    )

    normalizer = RunningObservationNormalizer(
        observation_dim=PlanarSubgoalActorCritic.OBS_DIM,
        normalized_dim=57,
    )
    normalizer.update(observations)
    _configure_scan_normalization(normalizer)
    normalized = normalizer.normalize(observations)
    train_indices, validation_indices = _episode_split(
        episode_ids,
        args.validation_fraction,
        args.seed,
        episode_goal_indices=episode_goal_indices,
    )
    model = PlanarSubgoalActorCritic(
        hidden_sizes=args.hidden_sizes,
        initial_log_std=args.initial_log_std,
    ).to(device)
    actor_parameters = (
        list(model.actor_backbone.parameters())
        + list(model.actor_mean.parameters())
    )
    optimizer = torch.optim.Adam(
        actor_parameters,
        lr=args.learning_rate,
    )
    observation_tensor = torch.from_numpy(normalized).to(device)
    action_tensor = torch.from_numpy(teacher_actions).to(device)
    robust_validation_observations = _augment_scan_observations(
        observations[validation_indices],
        np.random.RandomState(args.seed + 100000),
        probability=(
            1.0 if scan_augmentation_enabled else 0.0
        ),
        minimum_range=args.scan_augmentation_minimum_range,
        maximum_range=args.scan_augmentation_maximum_range,
        maximum_obstacles=args.scan_augmentation_maximum_obstacles,
        maximum_half_width=args.scan_augmentation_maximum_half_width,
        noise_standard_deviation=(
            args.scan_augmentation_noise_standard_deviation
        ),
    )
    robust_validation_tensor = torch.from_numpy(
        normalizer.normalize(robust_validation_observations)
    ).to(device)
    validation_action_tensor = torch.from_numpy(
        teacher_actions[validation_indices]
    ).to(device)
    best_selection_loss = float("inf")
    best_clean_validation_loss = float("inf")
    best_robust_validation_loss = float("inf")
    best_state = None
    augmentation_rng = np.random.RandomState(args.seed + 1)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(train_indices)
        losses = []
        for start in range(0, order.size, args.batch_size):
            batch_numpy = order[start:start + args.batch_size]
            batch_observations = _augment_scan_observations(
                observations[batch_numpy],
                augmentation_rng,
                probability=effective_augmentation_probability,
                minimum_range=args.scan_augmentation_minimum_range,
                maximum_range=args.scan_augmentation_maximum_range,
                maximum_obstacles=(
                    args.scan_augmentation_maximum_obstacles
                ),
                maximum_half_width=(
                    args.scan_augmentation_maximum_half_width
                ),
                noise_standard_deviation=(
                    args.scan_augmentation_noise_standard_deviation
                ),
            )
            normalized_batch = torch.from_numpy(
                normalizer.normalize(batch_observations)
            ).to(device)
            target = torch.from_numpy(
                teacher_actions[batch_numpy]
            ).to(device)
            predicted = model.deterministic_action(
                normalized_batch
            )
            loss = (predicted - target).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor_parameters,
                args.maximum_gradient_norm,
            )
            optimizer.step()
            losses.append(float(loss.item()))
        validation_loss = _mse(
            model,
            observation_tensor,
            action_tensor,
            validation_indices,
        )
        robust_validation_loss = _all_mse(
            model,
            robust_validation_tensor,
            validation_action_tensor,
        )
        selection_loss = (
            (1.0 - effective_lidar_validation_weight) * validation_loss
            + effective_lidar_validation_weight
            * robust_validation_loss
        )
        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_clean_validation_loss = validation_loss
            best_robust_validation_loss = robust_validation_loss
            best_state = copy.deepcopy(model.state_dict())
        print(
            "bc_epoch={:03d}/{:03d} train_mse={:.8f} "
            "validation_mse={:.8f} lidar_validation_mse={:.8f} "
            "selection_mse={:.8f}".format(
                epoch,
                args.epochs,
                float(np.mean(losses)),
                validation_loss,
                robust_validation_loss,
                selection_loss,
            )
        )

    model.load_state_dict(best_state)
    train_loss = _mse(
        model,
        observation_tensor,
        action_tensor,
        train_indices,
    )
    output = os.path.abspath(os.path.expanduser(args.output))
    directory = os.path.dirname(output)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    torch.save({
        "format_version": 6,
        "policy_type": "planar_pose_guided_bc",
        "model": model.state_dict(),
        "normalizer": normalizer.state_dict(),
        "arguments": vars(args),
        "hidden_sizes": list(args.hidden_sizes),
        "observation_dim": PlanarSubgoalActorCritic.OBS_DIM,
        "action_dim": 2,
        "teacher_type": teacher_type,
        "runtime_teacher_type": "none",
        "train_mse": float(train_loss),
        "validation_mse": float(best_clean_validation_loss),
        "lidar_augmented_validation_mse": float(
            best_robust_validation_loss
        ),
        "selection_validation_mse": float(best_selection_loss),
        "scan_normalization": "fixed_uniform_0_1",
        "scan_normalization_mean": SCAN_FIXED_MEAN,
        "scan_normalization_variance": SCAN_FIXED_VARIANCE,
        "scan_augmentation": {
            "enabled": scan_augmentation_enabled,
            "probability": effective_augmentation_probability,
            "minimum_range": args.scan_augmentation_minimum_range,
            "maximum_range": args.scan_augmentation_maximum_range,
            "maximum_obstacles": (
                args.scan_augmentation_maximum_obstacles
            ),
            "maximum_half_width": (
                args.scan_augmentation_maximum_half_width
            ),
            "noise_standard_deviation": (
                args.scan_augmentation_noise_standard_deviation
            ),
        },
        "sample_count": int(observations.shape[0]),
        "episode_count": int(np.unique(episode_ids).size),
        "observation_semantics": (
            "local_subgoal_xy2,subgoal_yaw_sin_cos2,body_vw2,"
            "scan36,path_preview_xy10,path_metrics3,"
            "previous_action2,path_mask5"
        ),
        "action_semantics": "normalized_linear_velocity_yaw_rate",
    }, output)
    print(
        "bc_model={} samples={} train_mse={:.8f} "
        "validation_mse={:.8f} lidar_validation_mse={:.8f} "
        "selection_mse={:.8f}".format(
            output,
            observations.shape[0],
            train_loss,
            best_clean_validation_loss,
            best_robust_validation_loss,
            best_selection_loss,
        )
    )


def _mse(model, observations, actions, indices):
    model.eval()
    index_tensor = torch.from_numpy(
        np.asarray(indices, dtype=np.int64)
    ).to(observations.device)
    with torch.no_grad():
        predicted = model.deterministic_action(
            observations.index_select(0, index_tensor)
        )
        target = actions.index_select(0, index_tensor)
        return float((predicted - target).pow(2).mean().item())


def _all_mse(model, observations, actions):
    model.eval()
    with torch.no_grad():
        predicted = model.deterministic_action(observations)
        return float((predicted - actions).pow(2).mean().item())


def _configure_scan_normalization(normalizer):
    """Use a stable [0, 1] LiDAR scale even for obstacle-free datasets.

    Obstacle-free demonstrations contain scans close to one everywhere.
    Estimating their variance from that dataset makes real obstacle ranges
    saturate at the normalizer clip during deployment.  Uniform [0, 1]
    moments keep both clear and occupied ranges in a compact fixed scale.
    """
    normalizer.mean[SCAN_START:SCAN_END] = SCAN_FIXED_MEAN
    normalizer.variance[SCAN_START:SCAN_END] = SCAN_FIXED_VARIANCE


def _augment_scan_observations(
        observations,
        rng,
        probability,
        minimum_range,
        maximum_range,
        maximum_obstacles,
        maximum_half_width,
        noise_standard_deviation):
    """Insert circular, sector-shaped LiDAR obstacles for BC robustness.

    The simple tracking teacher intentionally ignores LiDAR and follows the
    high-level path preview.  Randomizing only the scan therefore teaches
    that obstacle geometry must not corrupt path tracking before PPO learns
    a task-specific avoidance residual.
    """
    augmented = np.asarray(observations, dtype=np.float32).copy()
    scans = augmented[:, SCAN_START:SCAN_END]
    selected = rng.uniform(size=scans.shape[0]) < float(probability)
    bin_indices = np.arange(SCAN_DIM, dtype=np.int32)

    for row in np.flatnonzero(selected):
        synthetic = scans[row].copy()
        obstacle_count = int(rng.randint(1, maximum_obstacles + 1))
        for unused_obstacle in range(obstacle_count):
            center = int(rng.randint(0, SCAN_DIM))
            half_width = int(rng.randint(1, maximum_half_width + 1))
            center_range = float(rng.uniform(
                minimum_range,
                maximum_range,
            ))
            distance = np.abs(bin_indices - center)
            circular_distance = np.minimum(
                distance,
                SCAN_DIM - distance,
            )
            inside = circular_distance <= half_width
            edge_fraction = (
                circular_distance.astype(np.float32)
                / float(half_width + 1)
            )
            profile = (
                center_range
                + (1.0 - center_range) * edge_fraction
            )
            synthetic[inside] = np.minimum(
                synthetic[inside],
                profile[inside],
            )
        if noise_standard_deviation > 0.0:
            synthetic += rng.normal(
                loc=0.0,
                scale=noise_standard_deviation,
                size=SCAN_DIM,
            ).astype(np.float32)
        scans[row] = np.clip(synthetic, 0.0, 1.0)
    return augmented


def _episode_split(
        episode_ids,
        validation_fraction,
        seed,
        episode_goal_indices=None):
    episodes = np.unique(episode_ids)
    rng = np.random.RandomState(seed)
    validation_episodes = set()
    if episode_goal_indices is not None:
        if episode_goal_indices.shape != (episodes.size,):
            raise ValueError(
                "accepted_goal_indices must contain one value per episode"
            )
        for route_index in np.unique(episode_goal_indices):
            route_episodes = episodes[
                episode_goal_indices == route_index
            ].copy()
            rng.shuffle(route_episodes)
            if route_episodes.size < 2:
                continue
            route_validation_count = min(
                route_episodes.size - 1,
                max(
                    1,
                    int(round(
                        route_episodes.size * validation_fraction
                    )),
                ),
            )
            validation_episodes.update(
                route_episodes[
                    :route_validation_count
                ].tolist()
            )
    if not validation_episodes:
        shuffled = episodes.copy()
        rng.shuffle(shuffled)
        validation_count = max(
            1,
            int(round(episodes.size * validation_fraction)),
        )
        validation_episodes.update(
            shuffled[:validation_count].tolist()
        )
    validation_mask = np.asarray([
        value in validation_episodes for value in episode_ids
    ])
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if train_indices.size == 0 or validation_indices.size == 0:
        raise ValueError("dataset needs multiple episodes for splitting")
    return train_indices, validation_indices


def _validate_dataset(
        observations,
        actions,
        episode_ids,
        teacher_type,
        episode_goal_indices=None):
    if (
            observations.ndim != 2
            or observations.shape[1]
            != PlanarSubgoalActorCritic.OBS_DIM):
        raise ValueError("observations must have shape (N, 62)")
    if actions.shape != (observations.shape[0], 2):
        raise ValueError("teacher_actions must have shape (N, 2)")
    if episode_ids.shape != (observations.shape[0],):
        raise ValueError("episode_ids must have shape (N,)")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations contain non-finite values")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contain non-finite values")
    supported_teachers = {
        "observable_rule_based_planar_tracker",
        "teb_local_plan_cmd_vel",
    }
    if teacher_type not in supported_teachers:
        raise ValueError(
            "unsupported planar tracking teacher: {}".format(
                teacher_type
            )
        )
    if np.unique(episode_ids).size < 2:
        raise ValueError("dataset must contain at least two episodes")
    if (
            episode_goal_indices is not None
            and episode_goal_indices.shape
            != (np.unique(episode_ids).size,)):
        raise ValueError(
            "accepted_goal_indices must have shape (episode_count,)"
        )


def _decode_dataset_text(value):
    """Return text consistently for NumPy byte strings on Python 3.6."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output",
        default="/tmp/mobile_arm_rl_training/planar_tracking_bc.pt",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[128, 128],
    )
    parser.add_argument("--initial-log-std", type=float, default=-1.5)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    parser.add_argument(
        "--scan-augmentation-probability",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--scan-augmentation-minimum-range",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--scan-augmentation-maximum-range",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--scan-augmentation-maximum-obstacles",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--scan-augmentation-maximum-half-width",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--scan-augmentation-noise-standard-deviation",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--lidar-validation-weight",
        type=float,
        default=0.50,
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in (0, 1)")
    if not 0.0 <= args.scan_augmentation_probability <= 1.0:
        raise ValueError(
            "scan-augmentation-probability must be in [0, 1]"
        )
    if not (
            0.0 <= args.scan_augmentation_minimum_range
            < args.scan_augmentation_maximum_range <= 1.0):
        raise ValueError(
            "scan augmentation ranges must satisfy 0 <= min < max <= 1"
        )
    if (
            args.scan_augmentation_maximum_obstacles <= 0
            or args.scan_augmentation_maximum_half_width <= 0):
        raise ValueError(
            "scan augmentation obstacle counts must be positive"
        )
    if args.scan_augmentation_noise_standard_deviation < 0.0:
        raise ValueError(
            "scan augmentation noise cannot be negative"
        )
    if not 0.0 <= args.lidar_validation_weight <= 1.0:
        raise ValueError("lidar-validation-weight must be in [0, 1]")
    return args


if __name__ == "__main__":
    main()
