#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate residual checkpoints on a held-out split and select the best."""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUATOR = os.path.join(SCRIPT_DIR, "evaluate_fused_low_residual.py")


def main():
    args = _parse_arguments()
    checkpoints = _resolve_checkpoints(args.checkpoints)
    if not checkpoints:
        raise ValueError("no residual checkpoints matched")

    metrics_directory = os.path.abspath(os.path.expanduser(
        args.metrics_directory
    ))
    if not os.path.isdir(metrics_directory):
        os.makedirs(metrics_directory)

    results = []
    for index, checkpoint in enumerate(checkpoints, 1):
        stem = os.path.splitext(os.path.basename(checkpoint))[0]
        metrics_path = os.path.join(
            metrics_directory,
            "{:03d}_{}.json".format(index, stem),
        )
        command = [
            sys.executable,
            EVALUATOR,
            "--checkpoint", checkpoint,
            "--host", args.host,
            "--port", str(args.port),
            "--socket-timeout", str(args.socket_timeout),
            "--scenario-set", args.scenario_set,
            "--scenario-split", args.scenario_split,
            "--scenario-validation-fraction",
            str(args.scenario_validation_fraction),
            "--scenario-test-fraction", str(args.scenario_test_fraction),
            "--scenario-split-seed", str(args.scenario_split_seed),
            "--scenario-step-budget", str(args.scenario_step_budget),
            "--episodes", "0",
            "--min-success-rate", "0.0",
            "--max-collisions", "1000000",
            "--max-timeouts", "1000000",
            "--max-safety-rate", "1.0",
            "--metrics-output", metrics_path,
        ]
        if args.scenario_categories:
            command.append("--scenario-categories")
            command.extend(args.scenario_categories)
        print(
            "checkpoint_evaluation index={}/{} checkpoint={}".format(
                index, len(checkpoints), checkpoint
            )
        )
        return_code = subprocess.call(command)
        if not os.path.isfile(metrics_path):
            print(
                "checkpoint_skipped checkpoint={} return_code={} "
                "reason=no_metrics".format(checkpoint, return_code)
            )
            continue
        with open(metrics_path, "r") as stream:
            metrics = json.load(stream)
        score = _selection_score(metrics, args.maximum_safety_rate)
        results.append({
            "checkpoint": checkpoint,
            "metrics_path": metrics_path,
            "return_code": int(return_code),
            "score": list(score),
            "metrics": metrics,
        })
        print(
            "checkpoint_result checkpoint={} steps={} success_rate={:.3f} "
            "protected_rate={:.3f} recoverable_rate={:.3f} "
            "collisions={} tf_failures={} safety_rate={:.4f}".format(
                checkpoint,
                metrics.get("checkpoint_steps"),
                float(metrics.get("success_rate", 0.0)),
                _protected_success_rate(metrics),
                _category_success_rate(
                    metrics, "recoverable_failure", 0.0
                ),
                int(metrics.get("collisions", 0)),
                int(metrics.get("tf_failures", 0)),
                float(metrics.get("safety_rate", 0.0)),
            )
        )

    if not results:
        raise RuntimeError("no checkpoint produced evaluation metrics")
    best = max(results, key=lambda item: tuple(item["score"]))
    if best["score"][0] < 1.0 and not args.allow_unsafe_selection:
        raise RuntimeError(
            "all evaluated checkpoints violated collision, TF, or safety "
            "requirements; no model was selected"
        )

    output = os.path.abspath(os.path.expanduser(args.output))
    output_directory = os.path.dirname(output)
    if output_directory and not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    if os.path.abspath(best["checkpoint"]) != output:
        shutil.copy2(best["checkpoint"], output)

    manifest = {
        "format_version": 1,
        "selection_split": args.scenario_split,
        "scenario_set": args.scenario_set,
        "selected_checkpoint": best["checkpoint"],
        "output": output,
        "selection_score": best["score"],
        "selected_metrics": best["metrics"],
        "candidates": results,
    }
    manifest_path = output + ".selection.json"
    with open(manifest_path, "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print(
        "checkpoint_selected source={} output={} manifest={}".format(
            best["checkpoint"], output, manifest_path
        )
    )


def _resolve_checkpoints(patterns):
    resolved = []
    for pattern in patterns:
        expanded = os.path.abspath(os.path.expanduser(pattern))
        matches = glob.glob(expanded)
        if not matches and os.path.isfile(expanded):
            matches = [expanded]
        resolved.extend(matches)
    return sorted(set(
        path for path in resolved if os.path.isfile(path)
    ))


def _category_success_rate(metrics, category, default):
    record = metrics.get("categories", {}).get(category)
    if not record:
        return float(default)
    return float(record.get("success_rate", default))


def _protected_success_rate(metrics):
    rates = []
    for category in ("simple_success", "hard_success"):
        if category in metrics.get("categories", {}):
            rates.append(_category_success_rate(metrics, category, 0.0))
    return min(rates) if rates else float(metrics.get("success_rate", 0.0))


def _selection_score(metrics, maximum_safety_rate):
    safe = float(
        int(metrics.get("collisions", 0)) == 0
        and int(metrics.get("tf_failures", 0)) == 0
        and float(metrics.get("safety_rate", 0.0))
        <= float(maximum_safety_rate)
    )
    return (
        safe,
        _protected_success_rate(metrics),
        _category_success_rate(metrics, "recoverable_failure", 0.0),
        float(metrics.get("success_rate", 0.0)),
        float(metrics.get("mean_reward", float("-inf"))),
        -float(metrics.get("mean_final_distance", float("inf"))),
        -float(metrics.get("mean_residual_abs", float("inf"))),
    )


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--scenario-set", required=True)
    parser.add_argument(
        "--scenario-split",
        choices=["validation", "test"],
        default="validation",
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
        default=[
            "simple_success",
            "hard_success",
            "recoverable_failure",
        ],
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5563)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--maximum-safety-rate", type=float, default=0.05)
    parser.add_argument("--allow-unsafe-selection", action="store_true")
    parser.add_argument(
        "--metrics-directory",
        default=(
            "/tmp/mobile_arm_rl_training/"
            "fused_low_residual_checkpoint_metrics"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "/tmp/mobile_arm_rl_training/"
            "fused_low_residual_validation_best.pt"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
