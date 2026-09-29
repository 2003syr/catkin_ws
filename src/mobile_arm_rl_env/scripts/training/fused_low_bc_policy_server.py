#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Serve deterministic fused low-level actions to the ROS evaluator."""

from __future__ import print_function

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

try:
    import torch
except ImportError as error:
    raise RuntimeError(
        "PyTorch is required. Run this server inside mobile_arm_bc."
    ) from error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_low_actor_critic import (
    FusedLowActorCritic,
    HRL4INArmTransferredFusedPolicy,
)
from training.hrl4in_low_actor_critic import (
    RunningObservationNormalizer,
)


class FusedLowBcPolicyServer(object):

    def __init__(self, checkpoint_path, host, port, device):
        self.checkpoint_path = os.path.abspath(
            os.path.expanduser(checkpoint_path)
        )
        self.host = str(host)
        self.port = int(port)
        self.device = torch.device(device)
        self.shutdown_requested = False

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )
        self.policy_type = str(checkpoint.get("policy_type", ""))
        if self.policy_type not in (
                "fused_low_bc",
                "fused_low_teacher_ppo",
                HRL4INArmTransferredFusedPolicy.POLICY_TYPE):
            raise ValueError(
                "unsupported checkpoint policy_type: {}".format(
                    self.policy_type
                )
            )
        self.observation_dim = int(
            checkpoint.get("observation_dim", -1)
        )
        self.action_dim = int(checkpoint.get("action_dim", -1))
        if self.observation_dim != FusedLowActorCritic.OBS_DIM:
            raise ValueError(
                "fused checkpoint observation_dim must be {}, got {}".format(
                    FusedLowActorCritic.OBS_DIM,
                    self.observation_dim,
                )
            )
        if self.action_dim != FusedLowActorCritic.ACTION_DIM:
            raise ValueError(
                "fused checkpoint action_dim must be {}, got {}".format(
                    FusedLowActorCritic.ACTION_DIM,
                    self.action_dim,
                )
            )

        arguments = checkpoint.get("arguments", {})
        hidden_sizes = checkpoint.get(
            "hidden_sizes",
            arguments.get("hidden_sizes", [256, 256]),
        )
        initial_log_std = checkpoint.get(
            "initial_log_std",
            arguments.get("initial_log_std", -2.5),
        )
        normalizer_state = checkpoint.get("normalizer")
        if normalizer_state is None:
            raise ValueError("checkpoint does not contain a normalizer")
        if (
                self.policy_type
                == HRL4INArmTransferredFusedPolicy.POLICY_TYPE):
            arm_normalizer_state = checkpoint.get("arm_normalizer")
            if arm_normalizer_state is None:
                # The transferred model buffers are sufficient for inference,
                # but constructor validation requires the original contract.
                arm_normalizer_state = {
                    "observation_dim": 68,
                    "normalized_dim": 52,
                    "clip": 10.0,
                    "count": 1.0,
                    "mean": np.asarray(
                        checkpoint["model"]["arm_mean_normalizer"].cpu(),
                        dtype=np.float64,
                    ),
                    "variance": np.asarray(
                        checkpoint["model"]["arm_std"].cpu(),
                        dtype=np.float64,
                    ) ** 2,
                }
            self.model = HRL4INArmTransferredFusedPolicy(
                base_hidden_sizes=hidden_sizes,
                arm_hidden_sizes=checkpoint["arm_hidden_sizes"],
                fused_normalizer_state=normalizer_state,
                arm_normalizer_state=arm_normalizer_state,
                arm_start_distance=checkpoint.get(
                    "arm_start_distance",
                    0.30,
                ),
                base_stop_distance=checkpoint.get(
                    "base_stop_distance",
                    0.04,
                ),
                arm_max_subgoal=checkpoint.get(
                    "arm_max_subgoal",
                    0.08,
                ),
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.model.freeze_arm()
        else:
            self.model = FusedLowActorCritic(
                hidden_sizes=hidden_sizes,
                initial_log_std=initial_log_std,
            ).to(self.device)
            self.model.load_compatible_state_dict(
                checkpoint["model"],
                actor_only=True,
            )
        self.model.eval()

        self.normalizer = RunningObservationNormalizer(
            observation_dim=self.observation_dim,
            normalized_dim=int(normalizer_state["normalized_dim"]),
        )
        self.normalizer.load_state_dict(normalizer_state)
        self.teacher_type = str(
            checkpoint.get("teacher_type", "unknown")
        )
        self.validation_weighted_mse = float(
            checkpoint.get("validation_weighted_mse", float("nan"))
        )
        self.arm_source_checkpoint = checkpoint.get(
            "arm_source_checkpoint"
        )
        self.arm_source_total_steps = checkpoint.get(
            "arm_source_total_steps"
        )
        self.arm_transferred_frozen = bool(
            checkpoint.get("arm_transferred_frozen", False)
        )
        self.arm_finetuned = bool(
            checkpoint.get("arm_finetuned", False)
        )
        self.arm_max_subgoal = checkpoint.get("arm_max_subgoal")

    def serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        print(
            "Fused low policy server listening on {}:{} checkpoint={} "
            "device={} validation_weighted_mse={:.8f}".format(
                self.host,
                self.port,
                self.checkpoint_path,
                self.device,
                self.validation_weighted_mse,
            ),
            flush=True,
        )
        try:
            while not self.shutdown_requested:
                connection, address = server.accept()
                print(
                    "fused policy client connected: {}:{}".format(
                        address[0],
                        address[1],
                    ),
                    flush=True,
                )
                try:
                    self._serve_connection(connection)
                finally:
                    connection.close()
        finally:
            server.close()

    def _serve_connection(self, connection):
        reader = connection.makefile("r")
        try:
            while not self.shutdown_requested:
                line = reader.readline()
                if not line:
                    return
                try:
                    request = json.loads(line)
                    response = self._handle_request(request)
                except Exception as error:
                    response = {
                        "ok": False,
                        "error": "{}: {}".format(
                            type(error).__name__,
                            error,
                        ),
                    }
                payload = json.dumps(response, separators=(",", ":"))
                connection.sendall((payload + "\n").encode("utf-8"))
        finally:
            reader.close()

    def _handle_request(self, request):
        command = str(request.get("command", ""))
        if command == "ping":
            return {
                "ok": True,
                "policy_type": self.policy_type,
                "teacher_type": self.teacher_type,
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "checkpoint": self.checkpoint_path,
                "validation_weighted_mse": (
                    self.validation_weighted_mse
                ),
                "arm_source_checkpoint": self.arm_source_checkpoint,
                "arm_source_total_steps": self.arm_source_total_steps,
                "arm_transferred_frozen": self.arm_transferred_frozen,
                "arm_finetuned": self.arm_finetuned,
                "arm_max_subgoal": self.arm_max_subgoal,
            }
        if command == "predict":
            started = time.time()
            observation = np.asarray(
                request.get("observation", []),
                dtype=np.float32,
            ).reshape(-1)
            if observation.shape != (self.observation_dim,):
                raise ValueError(
                    "observation length must be {}, got {}".format(
                        self.observation_dim,
                        observation.shape[0],
                    )
                )
            if not np.all(np.isfinite(observation)):
                raise ValueError("observation contains non-finite values")
            normalized = self.normalizer.normalize(observation)
            tensor = torch.as_tensor(
                normalized,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            with torch.no_grad():
                action = self.model.deterministic_action(tensor)
            action = np.clip(
                action.squeeze(0).cpu().numpy(),
                -1.0,
                1.0,
            )
            return {
                "ok": True,
                "action": action.astype(float).tolist(),
                "inference_ms": 1000.0 * (time.time() - started),
            }
        if command == "close":
            return {"ok": True}
        if command == "shutdown":
            self.shutdown_requested = True
            return {"ok": True}
        raise ValueError("unknown command: {}".format(command))


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Serve a trained 66-D/8-D fused low-level policy.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5563)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    return parser.parse_args()


def main():
    args = _parse_arguments()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    server = FusedLowBcPolicyServer(
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port,
        device=args.device,
    )
    server.serve()


if __name__ == "__main__":
    main()
