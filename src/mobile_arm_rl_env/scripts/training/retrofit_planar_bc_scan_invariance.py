#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Make an obstacle-free planar BC actor invariant to LiDAR at startup."""

import argparse
import os

import numpy as np

try:
    import torch
except ImportError as error:
    raise RuntimeError(
        "Planar BC checkpoint retrofit requires PyTorch"
    ) from error


SCAN_START = 6
SCAN_END = 42
SCAN_FIXED_MEAN = 0.5
SCAN_FIXED_VARIANCE = 1.0 / 12.0


def main():
    args = _parse_arguments()
    source = os.path.abspath(os.path.expanduser(args.checkpoint))
    output = os.path.abspath(os.path.expanduser(args.output))
    checkpoint = torch.load(source, map_location="cpu")
    _validate_checkpoint(checkpoint)

    model_state = checkpoint["model"]
    weight_key, bias_key = _actor_input_keys(model_state)
    weight = model_state[weight_key].clone()
    bias = model_state[bias_key].clone()
    normalizer = checkpoint["normalizer"]

    old_mean = np.asarray(
        normalizer["mean"], dtype=np.float64
    ).copy()
    old_variance = np.asarray(
        normalizer["variance"], dtype=np.float64
    ).copy()
    old_clip = float(normalizer.get("clip", 10.0))
    reference_scan = np.ones(
        SCAN_END - SCAN_START,
        dtype=np.float64,
    )
    reference_normalized = (
        reference_scan - old_mean[SCAN_START:SCAN_END]
    ) / np.sqrt(
        old_variance[SCAN_START:SCAN_END] + 1.0e-8
    )
    reference_normalized = np.clip(
        reference_normalized,
        -old_clip,
        old_clip,
    )
    reference_tensor = torch.from_numpy(
        reference_normalized
    ).to(dtype=weight.dtype)

    lidar_weight = weight[:, SCAN_START:SCAN_END].clone()
    bias = bias + torch.mv(lidar_weight, reference_tensor)
    weight[:, SCAN_START:SCAN_END] = 0.0
    model_state[weight_key] = weight
    model_state[bias_key] = bias

    normalizer["mean"] = old_mean
    normalizer["variance"] = old_variance
    normalizer["mean"][SCAN_START:SCAN_END] = SCAN_FIXED_MEAN
    normalizer["variance"][SCAN_START:SCAN_END] = (
        SCAN_FIXED_VARIANCE
    )

    checkpoint["format_version"] = max(
        int(checkpoint.get("format_version", 0)),
        6,
    )
    checkpoint["scan_normalization"] = "fixed_uniform_0_1"
    checkpoint["scan_normalization_mean"] = SCAN_FIXED_MEAN
    checkpoint["scan_normalization_variance"] = (
        SCAN_FIXED_VARIANCE
    )
    checkpoint["actor_scan_initialization"] = (
        "zero_weight_bias_preserving_clear_scan"
    )
    checkpoint["retrofit_source_checkpoint"] = source
    checkpoint["actor_scan_weight_l2_before"] = float(
        torch.norm(lidar_weight).item()
    )
    checkpoint["actor_scan_weight_l2_after"] = float(
        torch.norm(
            weight[:, SCAN_START:SCAN_END]
        ).item()
    )

    directory = os.path.dirname(output)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    torch.save(checkpoint, output)
    print(
        "retrofit_model={} source={} scan_weight_l2={:.8f}->{:.8f} "
        "scan_normalization={} observation_dim={}".format(
            output,
            source,
            checkpoint["actor_scan_weight_l2_before"],
            checkpoint["actor_scan_weight_l2_after"],
            checkpoint["scan_normalization"],
            checkpoint.get("observation_dim"),
        )
    )


def _actor_input_keys(model_state):
    candidates = (
        (
            "actor_backbone.0.weight",
            "actor_backbone.0.bias",
        ),
        (
            "backbone.0.weight",
            "backbone.0.bias",
        ),
    )
    for weight_key, bias_key in candidates:
        if weight_key in model_state and bias_key in model_state:
            return weight_key, bias_key
    raise RuntimeError("checkpoint has no compatible actor input layer")


def _validate_checkpoint(checkpoint):
    if checkpoint.get("policy_type") not in (
            "planar_pose_guided_bc",):
        raise RuntimeError("checkpoint is not a planar BC policy")
    observation_dim = checkpoint.get("observation_dim")
    if observation_dim is None:
        observation_dim = checkpoint.get(
            "normalizer", {}
        ).get("observation_dim")
    if int(observation_dim or -1) != 62:
        raise RuntimeError("checkpoint observation dimension is not 62")
    normalizer = checkpoint.get("normalizer")
    if not isinstance(normalizer, dict):
        raise RuntimeError("checkpoint has no normalizer state")
    if int(normalizer.get("normalized_dim", -1)) < SCAN_END:
        raise RuntimeError("normalizer does not include the LiDAR slice")
    mean = np.asarray(normalizer.get("mean"))
    variance = np.asarray(normalizer.get("variance"))
    if mean.shape[0] < SCAN_END or variance.shape[0] < SCAN_END:
        raise RuntimeError("normalizer LiDAR statistics are incomplete")


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
