#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Serve deterministic HRL4IN PPO actions to the ROS Python-2 runner."""

import argparse
import json
import os
import socket
import sys
import time
import traceback

import numpy as np

try:
    import torch
except ImportError as error:
    raise RuntimeError(
        "HRL4IN PPO inference requires Python 3 with PyTorch"
    ) from error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from training.hrl4in_low_actor_critic import (
    HRL4INLowActorCritic,
    RunningObservationNormalizer,
)


class HRL4INLowPolicyServer(object):
    """Load one checkpoint and answer newline-delimited JSON requests."""

    OBS_DIM = HRL4INLowActorCritic.OBS_DIM
    ACTION_DIM = HRL4INLowActorCritic.ACTION_DIM

    def __init__(
            self,
            checkpoint_path,
            host="127.0.0.1",
            port=5556,
            backlog=1,
            socket_timeout=1.0,
            torch_threads=2):
        self.checkpoint_path = os.path.abspath(
            os.path.expanduser(checkpoint_path)
        )
        self.host = str(host)
        self.port = int(port)
        self.backlog = int(backlog)
        self.socket_timeout = float(socket_timeout)
        self.running = True
        self.server_socket = None

        if not os.path.isfile(self.checkpoint_path):
            raise IOError(
                "PPO checkpoint not found: {}".format(
                    self.checkpoint_path
                )
            )
        if self.backlog <= 0:
            raise ValueError("backlog must be positive")
        if self.socket_timeout <= 0.0:
            raise ValueError("socket_timeout must be positive")

        torch.set_num_threads(max(int(torch_threads), 1))
        self.device = torch.device("cpu")
        self.model, self.normalizer, self.checkpoint = (
            self._load_checkpoint()
        )

    def _load_checkpoint(self):
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )
        saved_arguments = checkpoint.get("arguments", {})
        hidden_sizes = saved_arguments.get(
            "hidden_sizes",
            [256, 256],
        )
        initial_log_std = saved_arguments.get(
            "initial_log_std",
            -0.5,
        )
        observation_dim = int(
            checkpoint.get("observation_dim", self.OBS_DIM)
        )
        action_dim = int(
            checkpoint.get("action_dim", self.ACTION_DIM)
        )
        if observation_dim != self.OBS_DIM:
            raise ValueError(
                "checkpoint observation dim is {}, expected {}".format(
                    observation_dim,
                    self.OBS_DIM,
                )
            )
        if action_dim != self.ACTION_DIM:
            raise ValueError(
                "checkpoint action dim is {}, expected {}".format(
                    action_dim,
                    self.ACTION_DIM,
                )
            )

        model = HRL4INLowActorCritic(
            hidden_sizes=hidden_sizes,
            initial_log_std=initial_log_std,
        ).to(self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        normalizer = RunningObservationNormalizer()
        normalizer.load_state_dict(checkpoint["normalizer"])
        return model, normalizer, checkpoint

    def serve_forever(self):
        self.server_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        self.server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.backlog)
        self.server_socket.settimeout(self.socket_timeout)
        print(
            "HRL4IN PPO policy server listening on {}:{} "
            "checkpoint={} total_steps={}".format(
                self.host,
                self.port,
                self.checkpoint_path,
                int(self.checkpoint.get("total_steps", 0)),
            )
        )
        sys.stdout.flush()

        try:
            while self.running:
                try:
                    connection, address = self.server_socket.accept()
                except socket.timeout:
                    continue
                print(
                    "PPO policy client connected from {}:{}".format(
                        address[0],
                        address[1],
                    )
                )
                sys.stdout.flush()
                try:
                    self._serve_connection(connection)
                finally:
                    try:
                        connection.close()
                    except socket.error:
                        pass
                    print("PPO policy client disconnected")
                    sys.stdout.flush()
        finally:
            self.close()

    def _serve_connection(self, connection):
        connection.settimeout(self.socket_timeout)
        receive_buffer = b""
        while self.running:
            try:
                chunk = connection.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                return
            receive_buffer += chunk

            while b"\n" in receive_buffer:
                raw_request, receive_buffer = receive_buffer.split(
                    b"\n",
                    1,
                )
                if not raw_request.strip():
                    continue
                should_close = False
                try:
                    request = json.loads(raw_request.decode("utf-8"))
                    result, should_close = self._handle_request(request)
                    response = {"ok": True, "result": result}
                except Exception as error:
                    traceback.print_exc()
                    response = {"ok": False, "error": str(error)}

                payload = (
                    json.dumps(response, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                connection.sendall(payload)
                if should_close:
                    return

    def _handle_request(self, request):
        command = str(request.get("command", ""))
        if command == "ping":
            return {
                "observation_dim": self.OBS_DIM,
                "action_dim": self.ACTION_DIM,
                "checkpoint": self.checkpoint_path,
                "total_steps": int(
                    self.checkpoint.get("total_steps", 0)
                ),
            }, False

        if command == "predict":
            observation = np.asarray(
                request["observation"],
                dtype=np.float32,
            )
            if observation.shape != (self.OBS_DIM,):
                raise ValueError(
                    "observation must have shape ({},), got {}".format(
                        self.OBS_DIM,
                        observation.shape,
                    )
                )
            if not np.all(np.isfinite(observation)):
                raise ValueError("observation contains non-finite values")

            start_time = time.time()
            normalized = self.normalizer.normalize(observation)
            observation_tensor = torch.from_numpy(
                normalized
            ).to(self.device).unsqueeze(0)
            with torch.no_grad():
                action = self.model.deterministic_action(
                    observation_tensor
                ).squeeze(0).cpu().numpy()
            inference_ms = (time.time() - start_time) * 1000.0
            if not np.all(np.isfinite(action)):
                raise RuntimeError("model returned a non-finite action")
            return {
                "action": np.asarray(
                    action,
                    dtype=np.float32,
                ).tolist(),
                "inference_ms": float(inference_ms),
            }, False

        if command == "close":
            return {"closed": True}, True

        if command == "shutdown":
            self.running = False
            return {"shutdown": True}, True

        raise ValueError("unknown command: {}".format(command))

    def close(self):
        self.running = False
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except socket.error:
                pass
            self.server_socket = None


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--backlog", type=int, default=1)
    parser.add_argument("--socket-timeout", type=float, default=1.0)
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def main():
    args = _parse_arguments()
    server = HRL4INLowPolicyServer(
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port,
        backlog=args.backlog,
        socket_timeout=args.socket_timeout,
        torch_threads=args.torch_threads,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
