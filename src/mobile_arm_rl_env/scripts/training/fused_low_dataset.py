#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ROS-independent contracts and statistics for fused teacher datasets."""

from __future__ import division

import numpy as np


OBSERVATION_DIM = 66
ACTION_DIM = 8
ACTION_MASK_SLICE = slice(58, 66)
PHASE_NAMES = (
    "BASE_APPROACH",
    "COORDINATED",
    "ARM_FINISH",
    "UNKNOWN",
)
PHASE_TO_ID = dict(
    (name, index) for index, name in enumerate(PHASE_NAMES)
)
DEFAULT_ACTION_LOSS_WEIGHTS = np.asarray(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25],
    dtype=np.float32,
)
PHASE_ACTION_MASKS = {
    "BASE_APPROACH": np.asarray(
        [1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32
    ),
    "COORDINATED": np.ones(ACTION_DIM, dtype=np.float32),
    "ARM_FINISH": np.asarray(
        [0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32
    ),
}


def phase_id(name):
    return PHASE_TO_ID.get(str(name), PHASE_TO_ID["UNKNOWN"])


def episode_train_validation_split(
        episode_ids,
        validation_fraction=0.20,
        seed=123):
    """Split complete episodes so adjacent samples never leak across sets."""
    episode_ids = np.asarray(episode_ids, dtype=np.int32)
    if episode_ids.ndim != 1:
        raise ValueError("episode_ids must be one-dimensional")
    episodes = np.unique(episode_ids)
    if episodes.size < 2:
        raise ValueError("dataset must contain at least two episodes")
    validation_fraction = float(validation_fraction)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")

    shuffled = episodes.copy()
    np.random.RandomState(int(seed)).shuffle(shuffled)
    validation_count = min(
        episodes.size - 1,
        max(1, int(round(episodes.size * validation_fraction))),
    )
    validation_episodes = shuffled[:validation_count]
    validation_mask = np.isin(episode_ids, validation_episodes)
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if train_indices.size == 0 or validation_indices.size == 0:
        raise ValueError("episode split produced an empty partition")
    return (
        train_indices.astype(np.int64),
        validation_indices.astype(np.int64),
        np.unique(episode_ids[~validation_mask]).astype(np.int32),
        np.unique(episode_ids[validation_mask]).astype(np.int32),
    )


def validate_action_loss_weights(weights):
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (ACTION_DIM,):
        raise ValueError(
            "action loss weights must have shape ({},)".format(
                ACTION_DIM
            )
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("action loss weights contain non-finite values")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError(
            "action loss weights must be non-negative with one positive"
        )
    return weights


def validate_action_mask_contract(
        observations,
        actions,
        phase_ids,
        tolerance=1.0e-5):
    """Validate legacy or three-stage action-mask dataset semantics.

    Obstacle-free datasets historically used an all-enabled mask for every
    sample; that contract remains supported.  Once any structured mask is
    present, every sample must match its phase and every inactive target
    action must be zero.  Mixing legacy and structured samples is rejected so
    a BC run cannot silently learn contradictory meanings for dimensions
    58..65.
    """
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    phase_ids = np.asarray(phase_ids, dtype=np.int32)
    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIM:
        raise ValueError("observations must have shape (N, 66)")
    sample_count = observations.shape[0]
    if actions.shape != (sample_count, ACTION_DIM):
        raise ValueError("actions must have shape (N, 8)")
    if phase_ids.shape != (sample_count,):
        raise ValueError("phase_ids must have shape (N,)")

    masks = observations[:, ACTION_MASK_SLICE]
    all_enabled = np.ones((sample_count, ACTION_DIM), dtype=np.float32)
    legacy_rows = np.all(
        np.isclose(masks, all_enabled, atol=tolerance),
        axis=1,
    )
    if np.all(legacy_rows):
        return {
            "mode": "legacy_all_enabled",
            "phase_counts": np.bincount(
                phase_ids, minlength=len(PHASE_NAMES)
            ).astype(np.int64),
            "mask_mismatch_count": 0,
            "inactive_action_mismatch_count": 0,
        }

    if np.any(legacy_rows):
        # COORDINATED legitimately uses all ones, so only reject all-enabled
        # rows whose phase says a structured mask should be active.
        structured_phase = np.logical_or(
            phase_ids == PHASE_TO_ID["BASE_APPROACH"],
            phase_ids == PHASE_TO_ID["ARM_FINISH"],
        )
        if np.any(np.logical_and(legacy_rows, structured_phase)):
            raise ValueError(
                "dataset mixes legacy all-enabled and three-stage masks"
            )

    expected = np.zeros_like(masks)
    known_phase = np.zeros(sample_count, dtype=np.bool_)
    for phase, action_mask in PHASE_ACTION_MASKS.items():
        matches = phase_ids == PHASE_TO_ID[phase]
        expected[matches] = action_mask
        known_phase = np.logical_or(known_phase, matches)
    if not np.all(known_phase):
        raise ValueError(
            "three-stage dataset contains UNKNOWN phase samples"
        )

    mask_matches = np.all(
        np.isclose(masks, expected, atol=tolerance),
        axis=1,
    )
    if not np.all(mask_matches):
        raise ValueError(
            "action masks disagree with phase_ids for {} samples".format(
                int(np.sum(np.logical_not(mask_matches)))
            )
        )
    inactive = expected < 0.5
    inactive_action_mismatch = np.logical_and(
        inactive,
        np.abs(actions) > tolerance,
    )
    mismatch_count = int(np.sum(inactive_action_mismatch))
    if mismatch_count:
        raise ValueError(
            "{} inactive target-action components are non-zero".format(
                mismatch_count
            )
        )
    return {
        "mode": "three_stage",
        "phase_counts": np.bincount(
            phase_ids, minlength=len(PHASE_NAMES)
        ).astype(np.int64),
        "mask_mismatch_count": 0,
        "inactive_action_mismatch_count": 0,
    }


def phase_balanced_epoch_indices(
        train_indices,
        phase_ids,
        exponent,
        random_state):
    """Resample structured stages without duplicating rare data excessively."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    phase_ids = np.asarray(phase_ids, dtype=np.int32)
    exponent = float(exponent)
    if not 0.0 <= exponent <= 1.0:
        raise ValueError("phase sampling exponent must be in [0, 1]")
    train_phases = phase_ids[train_indices]
    counts = np.bincount(
        train_phases,
        minlength=len(PHASE_NAMES),
    ).astype(np.float64)
    if np.any(counts[:3] <= 0.0):
        missing = [
            PHASE_NAMES[index]
            for index in range(3)
            if counts[index] <= 0.0
        ]
        raise ValueError(
            "structured training split is missing phases: {}".format(
                missing
            )
        )
    weights = np.power(counts[train_phases], -exponent)
    probabilities = weights / np.sum(weights)
    return random_state.choice(
        train_indices,
        size=train_indices.size,
        replace=True,
        p=probabilities,
    ).astype(np.int64)


def validate_sample_arrays(
        observations,
        teacher_actions,
        safe_teacher_actions,
        phase_ids):
    observations = np.asarray(observations)
    teacher_actions = np.asarray(teacher_actions)
    safe_teacher_actions = np.asarray(safe_teacher_actions)
    phase_ids = np.asarray(phase_ids)
    sample_count = observations.shape[0]
    expected = {
        "observations": (sample_count, OBSERVATION_DIM),
        "teacher_actions": (sample_count, ACTION_DIM),
        "safe_teacher_actions": (sample_count, ACTION_DIM),
        "phase_ids": (sample_count,),
    }
    actual = {
        "observations": observations.shape,
        "teacher_actions": teacher_actions.shape,
        "safe_teacher_actions": safe_teacher_actions.shape,
        "phase_ids": phase_ids.shape,
    }
    invalid = [
        "{}={} expected={}".format(name, actual[name], shape)
        for name, shape in expected.items()
        if actual[name] != shape
    ]
    if invalid:
        raise ValueError(
            "invalid fused dataset arrays: {}".format(", ".join(invalid))
        )
    for name, values in (
            ("observations", observations),
            ("teacher_actions", teacher_actions),
            ("safe_teacher_actions", safe_teacher_actions)):
        if not np.all(np.isfinite(values)):
            raise ValueError("{} contains non-finite values".format(name))
    if np.any(phase_ids < 0) or np.any(phase_ids >= len(PHASE_NAMES)):
        raise ValueError("phase_ids contain an unknown phase index")


def summarize_samples(
        observations,
        teacher_actions,
        safe_teacher_actions,
        phase_ids,
        safety_filter_active):
    validate_sample_arrays(
        observations,
        teacher_actions,
        safe_teacher_actions,
        phase_ids,
    )
    teacher_actions = np.asarray(teacher_actions, dtype=np.float64)
    safe_teacher_actions = np.asarray(
        safe_teacher_actions,
        dtype=np.float64,
    )
    phase_ids = np.asarray(phase_ids, dtype=np.int32)
    safety_filter_active = np.asarray(
        safety_filter_active,
        dtype=np.bool_,
    )
    if safety_filter_active.shape != (teacher_actions.shape[0],):
        raise ValueError(
            "safety_filter_active must contain one value per sample"
        )

    counts = np.bincount(
        phase_ids,
        minlength=len(PHASE_NAMES),
    ).astype(np.int64)
    action_delta = safe_teacher_actions - teacher_actions
    return {
        "sample_count": int(teacher_actions.shape[0]),
        "phase_counts": counts,
        "phase_fractions": (
            counts.astype(np.float64) / max(int(np.sum(counts)), 1)
        ),
        "teacher_action_mean": np.mean(teacher_actions, axis=0),
        "teacher_action_std": np.std(teacher_actions, axis=0),
        "teacher_action_min": np.min(teacher_actions, axis=0),
        "teacher_action_max": np.max(teacher_actions, axis=0),
        "safe_action_mean": np.mean(safe_teacher_actions, axis=0),
        "safe_action_std": np.std(safe_teacher_actions, axis=0),
        "mean_absolute_filter_delta": np.mean(
            np.abs(action_delta),
            axis=0,
        ),
        "maximum_filter_delta": np.max(np.abs(action_delta), axis=0),
        "safety_intervention_rate": float(
            np.mean(safety_filter_active.astype(np.float64))
        ),
    }
