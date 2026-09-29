#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a 46-to-6 behavior-cloning actor and export NumPy weights."""

import argparse
import copy
import os
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as error:
    raise RuntimeError(
        "Behavior-cloning training requires Python 3 with PyTorch"
    ) from error


class BehaviorCloningActor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes):
        super(BehaviorCloningActor, self).__init__()
        layers = []
        input_dim = observation_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, observation):
        return self.network(observation)


def main():
    args = _parse_arguments()
    _set_random_seed(args.seed)

    dataset = np.load(args.dataset)
    try:
        observations = np.asarray(
            dataset["observations"],
            dtype=np.float32,
        )
        actions = np.asarray(dataset[args.action_key], dtype=np.float32)
        episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)
        episode_success = np.asarray(
            dataset["episode_success"],
            dtype=np.bool_,
        )
        safety_interventions = np.asarray(
            dataset["safety_interventions"],
            dtype=np.int8,
        )
    finally:
        dataset.close()

    _validate_dataset(
        observations,
        actions,
        episode_ids,
        episode_success,
        safety_interventions,
    )
    original_sample_count = observations.shape[0]
    selection_mask = np.ones(original_sample_count, dtype=np.bool_)
    if args.successful_only:
        selection_mask = np.logical_and(selection_mask, episode_success)
    if args.safe_only:
        safe_mask = np.logical_not(
            np.any(safety_interventions != 0, axis=1)
        )
        selection_mask = np.logical_and(selection_mask, safe_mask)

    observations = observations[selection_mask]
    actions = actions[selection_mask]
    episode_ids = episode_ids[selection_mask]
    if observations.shape[0] == 0:
        raise RuntimeError(
            "No transitions remain after applying dataset filters"
        )
    selected_episode_count = np.unique(episode_ids).shape[0]
    print(
        "dataset samples={}->{} episodes={} successful_only={} "
        "safe_only={}".format(
            original_sample_count,
            observations.shape[0],
            selected_episode_count,
            args.successful_only,
            args.safe_only,
        )
    )

    train_indices, validation_indices, test_indices = _episode_split(
        episode_ids,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    observation_mean = observations[train_indices].mean(axis=0)
    observation_std = observations[train_indices].std(axis=0)
    observation_std = np.maximum(observation_std, 1e-6).astype(np.float32)
    observation_mean = observation_mean.astype(np.float32)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    actor = BehaviorCloningActor(
        observation_dim=observations.shape[1],
        action_dim=actions.shape[1],
        hidden_sizes=args.hidden_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()

    best_state = copy.deepcopy(actor.state_dict())
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    training_losses = []
    validation_losses = []

    for epoch in range(args.epochs):
        actor.train()
        shuffled = np.random.permutation(train_indices)
        accumulated_loss = 0.0
        accumulated_samples = 0

        for start in range(0, shuffled.shape[0], args.batch_size):
            batch_indices = shuffled[start:start + args.batch_size]
            batch_observations = _observation_tensor(
                observations[batch_indices],
                observation_mean,
                observation_std,
                device,
            )
            batch_actions = torch.from_numpy(
                actions[batch_indices]
            ).to(device)

            optimizer.zero_grad()
            predictions = actor(batch_observations)
            loss = loss_function(predictions, batch_actions)
            loss.backward()
            optimizer.step()

            batch_count = batch_indices.shape[0]
            accumulated_loss += float(loss.item()) * batch_count
            accumulated_samples += batch_count

        training_loss = accumulated_loss / float(accumulated_samples)
        validation_loss = _evaluate_mse(
            actor,
            observations,
            actions,
            validation_indices,
            observation_mean,
            observation_std,
            args.batch_size,
            device,
        )
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        print(
            "epoch={:03d} train_mse={:.8f} val_mse={:.8f}".format(
                epoch + 1,
                training_loss,
                validation_loss,
            )
        )

        if validation_loss < best_validation_loss - args.minimum_delta:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(actor.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("early_stopping_epoch={}".format(epoch + 1))
                break

    actor.load_state_dict(best_state)
    test_loss = _evaluate_mse(
        actor,
        observations,
        actions,
        test_indices,
        observation_mean,
        observation_std,
        args.batch_size,
        device,
    )
    _export_numpy_model(
        output_path=args.output,
        actor=actor,
        observation_mean=observation_mean,
        observation_std=observation_std,
        training_losses=training_losses,
        validation_losses=validation_losses,
        best_validation_loss=best_validation_loss,
        test_loss=test_loss,
        sample_counts=(
            train_indices.shape[0],
            validation_indices.shape[0],
            test_indices.shape[0],
        ),
        successful_only=args.successful_only,
        safe_only=args.safe_only,
        selected_episode_count=selected_episode_count,
    )
    print(
        "model={} best_val_mse={:.8f} test_mse={:.8f}".format(
            args.output,
            best_validation_loss,
            test_loss,
        )
    )


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--action-key",
        choices=("teacher_actions", "executed_actions"),
        default="teacher_actions",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--minimum-delta", type=float, default=1e-7)
    parser.add_argument("--successful-only", action="store_true")
    parser.add_argument(
        "--include-failed-episodes",
        action="store_false",
        dest="successful_only",
    )
    parser.add_argument("--safe-only", action="store_true")
    parser.add_argument(
        "--include-filtered",
        action="store_false",
        dest="safe_only",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.set_defaults(successful_only=True, safe_only=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    if args.learning_rate <= 0.0:
        parser.error("learning-rate must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("validation-fraction must be between zero and one")
    if not 0.0 < args.test_fraction < 1.0:
        parser.error("test-fraction must be between zero and one")
    if args.validation_fraction + args.test_fraction >= 1.0:
        parser.error("validation and test fractions must sum to less than one")
    if not args.hidden_sizes or any(size <= 0 for size in args.hidden_sizes):
        parser.error("hidden-sizes must contain positive integers")
    return args


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_dataset(
        observations,
        actions,
        episode_ids,
        episode_success,
        safety_interventions):
    if observations.ndim != 2 or observations.shape[1] != 46:
        raise ValueError(
            "observations must have shape (N, 46), got {}".format(
                observations.shape
            )
        )
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise ValueError(
            "actions must have shape (N, 6), got {}".format(actions.shape)
        )
    if not (
            observations.shape[0] == actions.shape[0]
            == episode_ids.shape[0] == episode_success.shape[0]
            == safety_interventions.shape[0]):
        raise ValueError("dataset arrays have inconsistent sample counts")
    if episode_success.ndim != 1:
        raise ValueError("episode_success must have shape (N,)")
    if (
            safety_interventions.ndim != 2
            or safety_interventions.shape[1] != 6):
        raise ValueError(
            "safety_interventions must have shape (N, 6), got {}".format(
                safety_interventions.shape
            )
        )
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations contain non-finite values")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contain non-finite values")


def _episode_split(episode_ids, validation_fraction, test_fraction, seed):
    unique_episodes = np.unique(episode_ids)
    if unique_episodes.shape[0] < 3:
        raise ValueError("at least three episodes are required")
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_episodes)

    validation_count = max(
        1,
        int(round(unique_episodes.shape[0] * validation_fraction)),
    )
    test_count = max(
        1,
        int(round(unique_episodes.shape[0] * test_fraction)),
    )
    if validation_count + test_count >= unique_episodes.shape[0]:
        validation_count = 1
        test_count = 1

    test_episodes = unique_episodes[:test_count]
    validation_episodes = unique_episodes[
        test_count:test_count + validation_count
    ]
    train_episodes = unique_episodes[test_count + validation_count:]

    train_indices = np.flatnonzero(np.isin(episode_ids, train_episodes))
    validation_indices = np.flatnonzero(
        np.isin(episode_ids, validation_episodes)
    )
    test_indices = np.flatnonzero(np.isin(episode_ids, test_episodes))
    return train_indices, validation_indices, test_indices


def _observation_tensor(observations, mean, std, device):
    normalized = (observations - mean) / std
    return torch.from_numpy(normalized.astype(np.float32)).to(device)


def _evaluate_mse(
        actor,
        observations,
        actions,
        indices,
        mean,
        std,
        batch_size,
        device):
    actor.eval()
    squared_error = 0.0
    element_count = 0
    with torch.no_grad():
        for start in range(0, indices.shape[0], batch_size):
            batch_indices = indices[start:start + batch_size]
            inputs = _observation_tensor(
                observations[batch_indices],
                mean,
                std,
                device,
            )
            targets = torch.from_numpy(actions[batch_indices]).to(device)
            predictions = actor(inputs)
            squared_error += float(
                torch.sum((predictions - targets) ** 2).item()
            )
            element_count += int(targets.numel())
    return squared_error / float(element_count)


def _export_numpy_model(
        output_path,
        actor,
        observation_mean,
        observation_std,
        training_losses,
        validation_losses,
        best_validation_loss,
        test_loss,
        sample_counts,
        successful_only,
        safe_only,
        selected_episode_count):
    output_path = os.path.abspath(os.path.expanduser(output_path))
    output_directory = os.path.dirname(output_path)
    if output_directory and not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    linear_layers = [
        layer for layer in actor.network
        if isinstance(layer, nn.Linear)
    ]
    arrays = {
        "format_version": np.asarray([1], dtype=np.int32),
        "observation_dim": np.asarray(
            [observation_mean.shape[0]],
            dtype=np.int32,
        ),
        "action_dim": np.asarray(
            [linear_layers[-1].out_features],
            dtype=np.int32,
        ),
        "layer_count": np.asarray([len(linear_layers)], dtype=np.int32),
        "observation_mean": observation_mean.astype(np.float32),
        "observation_std": observation_std.astype(np.float32),
        "training_losses": np.asarray(training_losses, dtype=np.float32),
        "validation_losses": np.asarray(
            validation_losses,
            dtype=np.float32,
        ),
        "best_validation_loss": np.asarray(
            [best_validation_loss],
            dtype=np.float32,
        ),
        "test_loss": np.asarray([test_loss], dtype=np.float32),
        "sample_counts": np.asarray(sample_counts, dtype=np.int64),
        "successful_only": np.asarray(
            [int(successful_only)],
            dtype=np.int8,
        ),
        "safe_only": np.asarray([int(safe_only)], dtype=np.int8),
        "selected_episode_count": np.asarray(
            [selected_episode_count],
            dtype=np.int32,
        ),
    }
    for index, layer in enumerate(linear_layers):
        arrays["weight_{}".format(index)] = (
            layer.weight.detach().cpu().numpy().astype(np.float32)
        )
        arrays["bias_{}".format(index)] = (
            layer.bias.detach().cpu().numpy().astype(np.float32)
        )

    temporary_path = output_path + ".tmp"
    with open(temporary_path, "wb") as output_file:
        np.savez_compressed(output_file, **arrays)
    if os.path.exists(output_path):
        os.remove(output_path)
    os.replace(temporary_path, output_path)


if __name__ == "__main__":
    main()
