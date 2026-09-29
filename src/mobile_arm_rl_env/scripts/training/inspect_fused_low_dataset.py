#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validate and print the coverage of a fused low-level NPZ dataset."""

from __future__ import print_function

import argparse
import collections

import numpy as np

from fused_low_dataset import (
    PHASE_NAMES,
    summarize_samples,
    validate_action_mask_contract,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    data = np.load(args.dataset, allow_pickle=False)
    try:
        summary = summarize_samples(
            data["observations"],
            data["teacher_actions"],
            data["safe_teacher_actions"],
            data["phase_ids"],
            data["safety_filter_active"],
        )
        action_mask_contract = validate_action_mask_contract(
            data["observations"],
            data["safe_teacher_actions"],
            data["phase_ids"],
        )
        print("dataset:", args.dataset)
        print("observations:", data["observations"].shape)
        print("teacher_actions:", data["teacher_actions"].shape)
        print("safe_teacher_actions:", data["safe_teacher_actions"].shape)
        print(
            "action_mask_contract:",
            action_mask_contract["mode"],
        )
        print(
            "episodes: successful={} attempted={} collisions={} "
            "timeouts={}".format(
                int(data["successful_episodes"][0]),
                int(data["attempted_episodes"][0]),
                int(data["collision_episodes"][0]),
                int(data["timeout_episodes"][0]),
            )
        )
        print("phases:")
        for index, name in enumerate(PHASE_NAMES):
            print(
                "  {}: {} ({:.3f})".format(
                    name,
                    int(summary["phase_counts"][index]),
                    float(summary["phase_fractions"][index]),
                )
            )
        if data["observations"].shape[1] >= 66:
            action_masks = np.asarray(
                data["observations"][:, 58:66],
                dtype=np.float32,
            )
            expected_masks = (
                ("BASE_APPROACH", np.asarray(
                    [1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32
                )),
                ("COORDINATED", np.ones(8, dtype=np.float32)),
                ("ARM_FINISH", np.asarray(
                    [0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32
                )),
            )
            print("action_mask_stages:")
            known = np.zeros(action_masks.shape[0], dtype=np.bool_)
            for name, expected in expected_masks:
                matches = np.all(
                    np.isclose(action_masks, expected[None, :]),
                    axis=1,
                )
                known = np.logical_or(known, matches)
                print(
                    "  {}: {} ({:.3f})".format(
                        name,
                        int(np.sum(matches)),
                        float(np.mean(matches)),
                    )
                )
            unknown = np.logical_not(known)
            print(
                "  UNKNOWN: {} ({:.3f})".format(
                    int(np.sum(unknown)),
                    float(np.mean(unknown)),
                )
            )
        print(
            "safe_action_mean:",
            np.round(summary["safe_action_mean"], 4).tolist(),
        )
        print(
            "safe_action_std:",
            np.round(summary["safe_action_std"], 4).tolist(),
        )
        print(
            "safe_action_min:",
            np.round(
                np.min(data["safe_teacher_actions"], axis=0),
                4,
            ).tolist(),
        )
        print(
            "safe_action_max:",
            np.round(
                np.max(data["safe_teacher_actions"], axis=0),
                4,
            ).tolist(),
        )
        print(
            "mean_absolute_filter_delta:",
            np.round(
                summary["mean_absolute_filter_delta"],
                5,
            ).tolist(),
        )
        print(
            "maximum_filter_delta:",
            np.round(summary["maximum_filter_delta"], 5).tolist(),
        )
        print(
            "safety_intervention_rate: {:.4f}".format(
                summary["safety_intervention_rate"]
            )
        )
        teacher_limit_active = np.asarray(
            data["teacher_limit_active"],
            dtype=np.float64,
        )
        print(
            "teacher_joint_limit_rate: {:.4f}".format(
                float(np.mean(teacher_limit_active))
            )
        )
        blocked_counts = collections.Counter(
            _text(value) if _text(value) else "NONE"
            for value in data["teacher_limit_blocked"]
        )
        print("teacher_joint_limit_blocked:")
        for reason, count in blocked_counts.most_common():
            print("  {}: {}".format(reason, count))
        overlap = np.asarray(data["overlap_active"], dtype=np.float64)
        print("overlap_rate: {:.4f}".format(float(np.mean(overlap))))
        if "teacher_interventions" in data.files:
            interventions = np.asarray(
                data["teacher_interventions"],
                dtype=np.float64,
            )
            print(
                "dagger_teacher_intervention_rate: {:.4f}".format(
                    float(np.mean(interventions))
                )
            )
            print(
                "dagger_student_teacher_max_deviation: "
                "mean={:.4f} max={:.4f}".format(
                    float(np.mean(
                        data["student_teacher_max_deviation"]
                    )),
                    float(np.max(
                        data["student_teacher_max_deviation"]
                    )),
                )
            )
            execution_safety = np.asarray(
                data["execution_safety_filter_active"],
                dtype=np.float64,
            )
            print(
                "dagger_execution_safety_rate: {:.4f}".format(
                    float(np.mean(execution_safety))
                )
            )
        final_ee = _final_values(
            data,
            "accepted_final_ee_distances",
            "episode_final_ee_distances",
        )
        final_base = _final_values(
            data,
            "accepted_final_base_distances",
            "episode_final_base_distances",
        )
        print(
            "final_ee_distance: mean={:.4f} max={:.4f}".format(
                float(np.mean(final_ee)),
                float(np.max(final_ee)),
            )
        )
        print(
            "final_base_distance: mean={:.4f} max={:.4f}".format(
                float(np.mean(final_base)),
                float(np.max(final_base)),
            )
        )
        print("dataset_contract_pass=True")
    finally:
        data.close()


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _final_values(data, accepted_name, episode_name):
    values = np.asarray(data[accepted_name], dtype=np.float32)
    if values.size == 0 and episode_name in data.files:
        values = np.asarray(data[episode_name], dtype=np.float32)
    if values.size == 0:
        return np.asarray([float("nan")], dtype=np.float32)
    return values


if __name__ == "__main__":
    main()
