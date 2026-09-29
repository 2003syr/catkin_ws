#!/usr/bin/env python3

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
        "PyTorch is required. Run this server with the mobile_arm_bc "
        "Python 3 virtual environment."
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


class TebBcPolicyServer(object):
    def __init__(self, checkpoint_path, host, port, device):
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.host = str(host)
        self.port = int(port)
        self.device = torch.device(device)
        self.shutdown_requested = False

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )
        policy_type = str(checkpoint.get("policy_type", ""))
        if policy_type not in ("planar_pose_guided_bc",):
            raise ValueError(
                "unsupported checkpoint policy_type: {}".format(policy_type)
            )

        self.observation_dim = int(checkpoint.get("observation_dim", -1))
        self.action_dim = int(checkpoint.get("action_dim", -1))
        if self.observation_dim != 62:
            raise ValueError(
                "TEB BC checkpoint observation_dim must be 62, got {}".format(
                    self.observation_dim
                )
            )
        if self.action_dim != 2:
            raise ValueError(
                "TEB BC checkpoint action_dim must be 2, got {}".format(
                    self.action_dim
                )
            )

        checkpoint_arguments = checkpoint.get("arguments", {})
        hidden_sizes = checkpoint.get(
            "hidden_sizes",
            checkpoint_arguments.get("hidden_sizes", [128, 128]),
        )
        self.model = PlanarSubgoalActorCritic(
            hidden_sizes=hidden_sizes,
            initial_log_std=checkpoint_arguments.get(
                "initial_log_std", -1.0
            ),
        ).to(self.device)
        self.model.load_compatible_state_dict(
            checkpoint["model"],
            actor_only=True,
        )
        self.model.eval()

        normalizer_state = checkpoint.get("normalizer")
        if normalizer_state is None:
            raise ValueError("checkpoint does not contain a normalizer")
        normalized_dim = int(
            normalizer_state.get("normalized_dim", self.observation_dim)
        )
        self.normalizer = RunningObservationNormalizer(
            observation_dim=self.observation_dim,
            normalized_dim=normalized_dim,
        )
        self.normalizer.load_state_dict(normalizer_state)

        self.policy_type = policy_type
        self.teacher_type = str(checkpoint.get("teacher_type", "unknown"))

    def serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        print(
            "TEB BC policy server listening on {}:{} checkpoint={} "
            "device={}".format(
                self.host,
                self.port,
                self.checkpoint_path,
                self.device,
            ),
            flush=True,
        )

        try:
            while not self.shutdown_requested:
                connection, address = server.accept()
                print(
                    "policy client connected: {}:{}".format(
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
            }
        if command == "predict":
            started = time.time()
            observation = np.asarray(
                request.get("observation", []),
                dtype=np.float32,
            ).reshape(-1)
            if observation.shape[0] != self.observation_dim:
                raise ValueError(
                    "observation length must be {}, got {}".format(
                        self.observation_dim,
                        observation.shape[0],
                    )
                )

            normalized = self.normalizer.normalize(observation)
            tensor = torch.as_tensor(
                normalized,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            with torch.no_grad():
                action = self.model.deterministic_action(tensor)
            action = action.squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve a trained 62-D TEB path-guided BC policy.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5559)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    server = TebBcPolicyServer(
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port,
        device=args.device,
    )
    server.serve()


if __name__ == "__main__":
    main()
