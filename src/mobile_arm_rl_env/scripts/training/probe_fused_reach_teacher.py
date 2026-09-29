#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Probe random fused-reach tasks with the zero-residual rule teacher.

The probe does not train a policy.  It records sampled task parameters and
measures which episode horizon would place teacher success near a requested
range.  A later curriculum builder can replay the recorded target parameters.
"""

import argparse
import json
import os
import sys

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_low_residual_env_client import (  # noqa: E402
    FusedLowResidualEnvironmentClient,
)
from training.fused_reach_scenarios import (  # noqa: E402
    DEFAULT_ARM_JOINT_SPEEDS,
    DEFAULT_BASE_SPEED,
    DEFAULT_CONTROL_DT,
    DEFAULT_YAW_SPEED,
    kinematic_lower_bound_steps,
)


def main():
    args = _parse_arguments()
    budgets = sorted(set(int(value) for value in args.step_budgets))
    maximum_steps = max(max(budgets), int(args.max_probe_steps))
    records = []
    client = FusedLowResidualEnvironmentClient(
        host=args.host,
        port=args.port,
        timeout=args.socket_timeout,
    )
    try:
        for episode in range(1, args.episodes + 1):
            observation, reset_info = client.reset_with_info()
            initial_distance = float(
                reset_info.get("initial_ee_distance", float("nan"))
            )
            lower_bound = _record_lower_bound(reset_info, args)
            final_distance = initial_distance
            success_step = None
            collision = False
            tf_ok = True
            safety_steps = 0
            reward_sum = 0.0
            final_info = {}

            for step in range(1, maximum_steps + 1):
                observation, reward, done, info = client.step(
                    np.zeros(client.ACTION_DIM, dtype=np.float32)
                )
                reward_sum += float(reward)
                final_info = dict(info)
                final_distance = float(
                    info.get("ee_distance", final_distance)
                )
                collision = bool(info.get("collision", False))
                tf_ok = bool(info.get("tf_ok", True))
                projection = np.asarray(
                    info.get("projection", np.zeros(8)),
                    dtype=np.float32,
                )
                safety_steps += int(
                    float(np.max(np.abs(projection)))
                    > args.projection_tolerance
                )
                if bool(info.get("success", False)):
                    success_step = step
                    break
                if done:
                    break

            record = {
                "episode": episode,
                "reset_info": reset_info,
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "progress": initial_distance - final_distance,
                "success_step": success_step,
                "observed_steps": step,
                "collision": collision,
                "tf_ok": tf_ok,
                "environment_timeout": bool(
                    final_info.get("timeout", False)
                ),
                "safety_steps": safety_steps,
                "reward": reward_sum,
                "kinematic_lower_bound_steps": lower_bound,
            }
            records.append(record)
            print(
                "teacher_probe episode={}/{} success_step={} "
                "distance={:.4f}->{:.4f} progress={:.4f} "
                "collision={} tf_ok={} safety_steps={}".format(
                    episode,
                    args.episodes,
                    success_step if success_step is not None else "none",
                    initial_distance,
                    final_distance,
                    initial_distance - final_distance,
                    collision,
                    tf_ok,
                    safety_steps,
                )
            )
    finally:
        # A reset publishes zero velocity before the socket is closed.  This
        # avoids leaving the Gazebo velocity controllers on their last action.
        try:
            client.reset()
        except (OSError, RuntimeError):
            pass
        client.close()

    dynamic_available = all(
        record["kinematic_lower_bound_steps"] is not None
        for record in records
    )
    if args.require_dynamic_budget and not dynamic_available:
        raise RuntimeError(
            "dynamic budget metadata missing from reset_info; restart the "
            "updated fused environment server before probing"
        )
    if dynamic_available:
        budget_rates = _dynamic_budget_success_rates(
            records, args.budget_multipliers, args
        )
        selected_multiplier = _select_budget(
            budget_rates,
            args.target_min_success_rate,
            args.target_max_success_rate,
        )
        selected_budget = int(np.median([
            _dynamic_budget(record, selected_multiplier, args)
            for record in records
        ]))
        for record in records:
            record["budget_multiplier"] = float(selected_multiplier)
            record["step_budget"] = _dynamic_budget(
                record, selected_multiplier, args
            )
    else:
        budget_rates = _budget_success_rates(records, budgets)
        selected_multiplier = None
        selected_budget = _select_budget(
            budget_rates,
            args.target_min_success_rate,
            args.target_max_success_rate,
        )
        for record in records:
            record["step_budget"] = int(selected_budget)
    _annotate_budget_normalization(records)
    categories = _classify_records(
        records,
        selected_budget,
        args.recoverable_distance,
    )
    category_counts = dict(
        (name, sum(item["category"] == name for item in categories))
        for name in (
            "simple_success",
            "hard_success",
            "recoverable_failure",
            "other_failure",
            "invalid",
        )
    )
    selected_rate = float(
        budget_rates[
            selected_multiplier
            if selected_multiplier is not None else selected_budget
        ]
    )
    target_reached = bool(
        args.target_min_success_rate
        <= selected_rate
        <= args.target_max_success_rate
    )
    output = {
        "schema_version": 2 if dynamic_available else 1,
        "teacher": "coordinated_rule_teacher_zero_residual",
        "episodes": int(args.episodes),
        "max_probe_steps": int(maximum_steps),
        "step_budgets": budgets,
        "budget_success_rates": dict(
            (str(key), float(value))
            for key, value in budget_rates.items()
        ),
        "target_success_rate_range": [
            float(args.target_min_success_rate),
            float(args.target_max_success_rate),
        ],
        "recommended_step_budget": int(selected_budget),
        "budget_mode": "dynamic" if dynamic_available else "fixed",
        "recommended_budget_multiplier": selected_multiplier,
        "control_dt": float(args.control_dt),
        "base_speed": float(args.base_speed),
        "yaw_speed": float(args.yaw_speed),
        "arm_joint_speeds": [
            float(value) for value in args.arm_joint_speeds
        ],
        "recommended_teacher_success_rate": selected_rate,
        "target_range_reached": target_reached,
        "teacher_time_ratio_summary": _time_ratio_summary(
            records, "teacher_time_ratio"
        ),
        "observed_time_ratio_summary": _time_ratio_summary(
            records, "observed_time_ratio"
        ),
        "category_counts": category_counts,
        "records": categories,
    }
    output_directory = os.path.dirname(os.path.abspath(args.output))
    if output_directory and not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    with open(args.output, "w") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)

    print("teacher_probe_budget_rates={}".format(
        dict((key, round(value, 3)) for key, value in budget_rates.items())
    ))
    print(
        "teacher_probe_selected budget={} multiplier={} "
        "success_rate={:.3f} target_range=[{:.3f},{:.3f}] "
        "target_reached={} categories={}".format(
            selected_budget,
            selected_multiplier if selected_multiplier is not None else "fixed",
            selected_rate,
            args.target_min_success_rate,
            args.target_max_success_rate,
            target_reached,
            category_counts,
        )
    )
    print("teacher_probe_time_ratio_summary={}".format(
        _time_ratio_summary(records, "teacher_time_ratio")
    ))
    print("teacher_probe_observed_ratio_summary={}".format(
        _time_ratio_summary(records, "observed_time_ratio")
    ))
    print("teacher_probe_saved={}".format(args.output))


def _budget_success_rates(records, budgets):
    denominator = float(max(len(records), 1))
    return dict(
        (
            budget,
            sum(
                record["success_step"] is not None
                and int(record["success_step"]) <= budget
                for record in records
            ) / denominator,
        )
        for budget in budgets
    )


def _dynamic_budget_success_rates(records, multipliers, args):
    denominator = float(max(len(records), 1))
    return dict(
        (
            float(multiplier),
            sum(
                record["success_step"] is not None
                and int(record["success_step"]) <= _dynamic_budget(
                    record, multiplier, args
                )
                for record in records
            ) / denominator,
        )
        for multiplier in multipliers
    )


def _dynamic_budget(record, multiplier, args):
    lower_bound = int(record["kinematic_lower_bound_steps"])
    return int(min(
        args.max_dynamic_budget,
        max(
            args.min_dynamic_budget,
            int(np.ceil(float(multiplier) * float(lower_bound))),
        ),
    ))


def _annotate_budget_normalization(records):
    """Attach horizon diagnostics that separate geometry from controller speed."""
    for record in records:
        lower_bound = record.get("kinematic_lower_bound_steps")
        if lower_bound is None or int(lower_bound) <= 0:
            record["budget_to_lower_bound_ratio"] = None
            record["teacher_time_ratio"] = None
            record["observed_time_ratio"] = None
            continue
        lower_bound = float(lower_bound)
        budget = float(record["step_budget"])
        success_step = record.get("success_step")
        record["budget_to_lower_bound_ratio"] = budget / lower_bound
        record["teacher_time_ratio"] = (
            None if success_step is None
            else float(success_step) / lower_bound
        )
        record["observed_time_ratio"] = (
            float(record["observed_steps"]) / lower_bound
        )


def _time_ratio_summary(records, key):
    values = [
        float(record[key])
        for record in records
        if record.get(key) is not None
    ]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "max": float(np.max(values)),
    }


def _record_lower_bound(reset_info, args):
    initial_arm = reset_info.get("initial_arm_positions")
    target_arm = reset_info.get("target_arm_positions")
    displacement = reset_info.get("base_displacement_body")
    if initial_arm is None or target_arm is None or displacement is None:
        return None
    try:
        return kinematic_lower_bound_steps(
            displacement,
            initial_arm,
            target_arm,
            control_dt=args.control_dt,
            base_speed=args.base_speed,
            yaw_speed=args.yaw_speed,
            arm_joint_speeds=args.arm_joint_speeds,
        )
    except ValueError:
        return None


def _select_budget(rates, minimum_rate, maximum_rate):
    target = 0.5 * (float(minimum_rate) + float(maximum_rate))
    inside = [
        budget
        for budget, rate in rates.items()
        if minimum_rate <= rate <= maximum_rate
    ]
    candidates = inside if inside else list(rates.keys())
    return min(candidates, key=lambda budget: (
        abs(float(rates[budget]) - target),
        budget,
    ))


def _classify_records(records, step_budget, recoverable_distance):
    classified = []
    for source in records:
        record = dict(source)
        record_budget = int(source.get("step_budget", step_budget))
        simple_limit = max(1, int(round(0.70 * float(record_budget))))
        success_step = record["success_step"]
        if record["collision"] or not record["tf_ok"]:
            category = "invalid"
        elif success_step is not None and success_step <= simple_limit:
            category = "simple_success"
        elif success_step is not None and success_step <= record_budget:
            category = "hard_success"
        elif (
                success_step is not None
                or (
                    record["progress"] > 0.0
                    and record["final_distance"] <= recoverable_distance
                )):
            category = "recoverable_failure"
        else:
            category = "other_failure"
        record["category"] = category
        record["teacher_success_at_recommended_budget"] = bool(
            success_step is not None and success_step <= record_budget
        )
        classified.append(record)
    return classified


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5563)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--step-budgets",
        type=int,
        nargs="+",
        default=[80, 100, 120, 140],
    )
    parser.add_argument("--max-probe-steps", type=int, default=160)
    parser.add_argument(
        "--budget-multipliers",
        type=float,
        nargs="+",
        default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0],
    )
    parser.add_argument("--control-dt", type=float, default=DEFAULT_CONTROL_DT)
    parser.add_argument("--base-speed", type=float, default=DEFAULT_BASE_SPEED)
    parser.add_argument("--yaw-speed", type=float, default=DEFAULT_YAW_SPEED)
    parser.add_argument(
        "--arm-joint-speeds",
        type=float,
        nargs=6,
        default=DEFAULT_ARM_JOINT_SPEEDS.tolist(),
    )
    parser.add_argument("--min-dynamic-budget", type=int, default=40)
    parser.add_argument("--max-dynamic-budget", type=int, default=240)
    parser.add_argument(
        "--require-dynamic-budget",
        action="store_true",
        help="fail instead of silently falling back to fixed horizons",
    )
    parser.add_argument("--target-min-success-rate", type=float, default=0.70)
    parser.add_argument("--target-max-success-rate", type=float, default=0.85)
    parser.add_argument("--recoverable-distance", type=float, default=0.15)
    parser.add_argument("--projection-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        default=(
            "/tmp/mobile_arm_rl_training/"
            "fused_reach_teacher_probe.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
