#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Serve a fused residual PPO actor on arbitrary 66-D subgoal observations.

This is deliberately a policy-only server.  The ROS HRL process owns the
coordinated rule teacher and composes the returned five-dimensional residual
with it, so the residual actor can be tested on fixed joint subgoals without
duplicating ROS/KDL dependencies in Python 3.
"""

from __future__ import print_function

import argparse
import json
import os
import socket
import time

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
import sys
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.fused_residual_actor_critic import (  # noqa: E402
    FusedResidualActorCritic,
)
from training.hrl4in_low_actor_critic import (  # noqa: E402
    RunningObservationNormalizer,
)


class FusedResidualPolicyServer(object):

    OBS_DIM = 66
    ACTION_DIM = 5

    def __init__(self, checkpoint_path, host, port, device,
                 zero_residual=False):
        self.zero_residual = bool(zero_residual)
        self.checkpoint_path = (
            None if checkpoint_path is None else os.path.abspath(
                os.path.expanduser(checkpoint_path)
            )
        )
        self.host = str(host)
        self.port = int(port)
        self.device = torch.device(device)
        self.shutdown_requested = False

        if self.zero_residual:
            self.model = None
            self.normalizer = None
            self.metadata = {
                "policy_type": "fused_low_residual_zero",
                "checkpoint": None,
                "observation_dim": self.OBS_DIM,
                "action_dim": self.ACTION_DIM,
                "nominal_action_dim": 8,
                "subgoal_context_contract": (
                    "legacy_all_ones_subgoal_mask"
                ),
                "total_steps": 0,
                "update_index": 0,
            }
        else:
            if self.checkpoint_path is None:
                raise ValueError(
                    "--checkpoint is required unless --zero-residual is used"
                )
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )
            policy_type = str(checkpoint.get("policy_type", ""))
            if policy_type != FusedResidualActorCritic.POLICY_TYPE:
                raise ValueError(
                    "unsupported residual checkpoint policy_type: {}".format(
                        policy_type
                    )
                )
            arguments = checkpoint.get("arguments", {})
            hidden_sizes = checkpoint.get(
                "hidden_sizes",
                arguments.get("hidden_sizes", [128, 128]),
            )
            initial_log_std = checkpoint.get(
                "initial_log_std",
                arguments.get("initial_log_std", -3.0),
            )
            self.model = FusedResidualActorCritic(
                hidden_sizes=hidden_sizes,
                initial_log_std=initial_log_std,
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.model.eval()

            normalizer_state = checkpoint.get("normalizer")
            if normalizer_state is None:
                raise ValueError("residual checkpoint has no normalizer")
            self.normalizer = RunningObservationNormalizer(
                observation_dim=self.model.OBS_DIM,
                normalized_dim=int(normalizer_state["normalized_dim"]),
            )
            self.normalizer.load_state_dict(normalizer_state)
            self.metadata = {
                "policy_type": policy_type,
                "checkpoint": self.checkpoint_path,
                "observation_dim": self.model.OBS_DIM,
                "action_dim": self.model.ACTION_DIM,
                "nominal_action_dim": 8,
                "subgoal_context_contract": str(checkpoint.get(
                    "subgoal_context_contract",
                    "legacy_all_ones_subgoal_mask",
                )),
                "total_steps": int(checkpoint.get("total_steps", 0)),
                "update_index": int(checkpoint.get("update_index", 0)),
            }

    def serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        print(
            "Fused residual policy listening on {}:{} checkpoint={} "
            "steps={} update={}".format(
                self.host,
                self.port,
                self.checkpoint_path,
                self.metadata["total_steps"],
                self.metadata["update_index"],
            ),
            flush=True,
        )
        try:
            while not self.shutdown_requested:
                connection, _ = server.accept()
                try:
                    self._serve_connection(connection)
                finally:
                    connection.close()
        finally:
            server.close()

    def _serve_connection(self, connection):
        reader = connection.makefile("r")
        try:
            for line in reader:
                try:
                    request = json.loads(line)
                    response = self._handle_request(request)
                except Exception as error:
                    response = {"ok": False, "error": str(error)}
                connection.sendall(
                    (json.dumps(response, separators=(",", ":")) + "\n")
                    .encode("utf-8")
                )
                if response.get("close", False):
                    return
        finally:
            reader.close()

    def _handle_request(self, request):
        command = str(request.get("command", ""))
        if command == "ping":
            return dict({"ok": True}, **self.metadata)
        if command == "predict":
            started = time.time()
            observation = np.asarray(
                request.get("observation", []),
                dtype=np.float32,
            ).reshape(-1)
            if observation.shape != (self.OBS_DIM,):
                raise ValueError(
                    "observation length must be {}, got {}".format(
                        self.OBS_DIM,
                        observation.shape[0],
                    )
                )
            if not np.all(np.isfinite(observation)):
                raise ValueError("observation contains non-finite values")
            if self.zero_residual:
                action = np.zeros(self.ACTION_DIM, dtype=np.float32)
            else:
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
            return {"ok": True, "close": True}
        if command == "shutdown":
            self.shutdown_requested = True
            return {"ok": True, "close": True}
        raise ValueError("unknown command: {}".format(command))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--zero-residual", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5563)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if not args.zero_residual and not args.checkpoint:
        parser.error("--checkpoint is required unless --zero-residual is used")
    FusedResidualPolicyServer(
        args.checkpoint,
        args.host,
        args.port,
        args.device,
        zero_residual=args.zero_residual,
    ).serve()


if __name__ == "__main__":
    main()
