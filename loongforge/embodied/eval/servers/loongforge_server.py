# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Run a LoongForge-backed policy server for standalone eval."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from loongforge.embodied.eval.orchestrator.config import load_config
from loongforge.embodied.eval.servers.loongforge_policy import LoongForgePI05Policy
from loongforge.embodied.eval.transport.rpc_server import PolicyServer


class HealthHandler(BaseHTTPRequestHandler):
    """Provide HealthHandler behavior."""

    ckpt_path = ""

    def do_GET(self) -> None:
        """Run do_GET."""
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ready": True, "ckpt_path": self.ckpt_path}).encode())
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Run log_message."""
        return


def start_health_server(port: int, ckpt_path: str) -> None:
    """Run start_health_server."""
    HealthHandler.ckpt_path = ckpt_path
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    """Run _apply_config."""
    if not args.config:
        return args
    config = load_config(args.config)
    model = config.get("model") or {}
    server = config.get("server") or {}
    env = config.get("env") or {}
    if not isinstance(model, dict) or not isinstance(server, dict) or not isinstance(env, dict):
        raise ValueError("model, server, and env sections must be mappings")

    args.model_type = str(model.get("model_type", args.model_type)).lower()
    args.ckpt_path = model.get("ckpt_path") or model.get("pretrained_path") or args.ckpt_path
    args.dataset_statistics_path = (
        model.get("dataset_statistics_path") or env.get("dataset_statistics_path") or args.dataset_statistics_path
    )
    args.tokenizer_path = model.get("tokenizer_path") or args.tokenizer_path
    args.action_dim = int(model.get("action_dim", args.action_dim))
    args.state_dim = int(model.get("state_dim", args.state_dim))
    args.action_horizon = int(model.get("action_horizon", model.get("action_chunk_size", args.action_horizon)))
    args.max_action_dim = int(model.get("max_action_dim", args.max_action_dim))
    args.max_state_dim = int(model.get("max_state_dim", args.max_state_dim))
    args.use_bf16 = bool(model.get("use_bf16", args.use_bf16))
    args.compile_model = bool(model.get("compile_model", args.compile_model))
    args.compile_mode = model.get("compile_mode") or args.compile_mode
    args.random_init = bool(model.get("random_init", args.random_init))
    args.loongforge_root = env.get("loongforge_root") or args.loongforge_root
    if server.get("port") is not None:
        args.port = int(server["port"])
    if server.get("health_port") is not None:
        args.health_port = int(server["health_port"])
    return args


def build_argparser() -> argparse.ArgumentParser:
    """Run build_argparser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--model-type", default="pi05")
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument("--dataset-statistics-path", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--health-port", type=int, default=10094)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="use_bf16")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument("--max-state-dim", type=int, default=32)
    parser.add_argument("--compile-model", action="store_true", default=False)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--random-init", action="store_true", default=False)
    parser.add_argument("--loongforge-root", default="")
    return parser


def main() -> None:
    """Run main."""
    args = _apply_config(build_argparser().parse_args())
    if args.model_type != "pi05":
        raise SystemExit(f"unsupported LoongForge model_type: {args.model_type!r}")
    if args.loongforge_root:
        os.chdir(args.loongforge_root)
    if not args.ckpt_path and not args.random_init:
        raise SystemExit("checkpoint must be set by model.ckpt_path in YAML unless model.random_init is true")

    logging.basicConfig(level=logging.INFO, force=True)
    start_health_server(args.health_port, args.ckpt_path)
    policy = LoongForgePI05Policy(
        ckpt_path=str(Path(args.ckpt_path)),
        loongforge_root=args.loongforge_root,
        device=args.device,
        use_bf16=args.use_bf16,
        dataset_statistics_path=args.dataset_statistics_path,
        tokenizer_path=args.tokenizer_path,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        action_horizon=args.action_horizon,
        max_action_dim=args.max_action_dim,
        max_state_dim=args.max_state_dim,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        random_init=args.random_init,
    )
    PolicyServer(policy=policy, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
