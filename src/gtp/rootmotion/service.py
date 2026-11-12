"""HTTP service the engine calls for root-motion plans.

    POST /plan   world-space request  ->  TrajectoryPlan JSON
    GET  /health

Standard library only, so the Unity project needs nothing but this process.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

import torch

from gtp.rootmotion.planner import (
    SOURCE_HOLD, SOURCE_NAMES, PlannerConfig, RootMotionPlanner,
)
from gtp.rootmotion.request import RequestError, build_batch, build_plan

MAX_BODY_BYTES = 1_000_000


class PlanService:
    """Thread-safe wrapper: one model, one request at a time."""

    def __init__(self, planner: RootMotionPlanner) -> None:
        self.planner = planner
        self.lock = threading.Lock()

    def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        batch, frame = build_batch(request)
        half_width = request.get("corridor_half_width")
        if half_width is not None:
            half_width = float(half_width)
            if not 0.05 <= half_width <= 5.0:
                raise RequestError("corridor_half_width must be between 0.05 and 5 metres")
        candidates = request.get("candidates")
        if candidates is not None:
            candidates = int(candidates)
            if not 1 <= candidates <= 64:
                raise RequestError("candidates must be between 1 and 64")
        generator = None
        if request.get("seed") is not None:
            generator = torch.Generator(device=self.planner.device).manual_seed(int(request["seed"]))
        started = time.perf_counter()
        with self.lock:
            result = self.planner.plan_batch(batch, generator, half_width, candidates)
        source = int(result["source"][0])
        rejected = int(result["candidate_invalid"][0].sum())
        plan = build_plan(
            result["deltas"][0].cpu().tolist(), frame,
            fallback=source == SOURCE_HOLD, score=float(rejected),
        ).to_dict()
        # Extra fields; clients that only know the base contract ignore them.
        plan["source"] = SOURCE_NAMES[source]
        plan["rejected_candidates"] = rejected
        plan["candidates"] = int(result["candidate_invalid"].shape[1])
        plan["latency_ms"] = (time.perf_counter() - started) * 1_000.0
        return plan


def make_handler(service: PlanService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            if self.path == "/health":
                self._send(200, {"status": "ok", "candidates": service.planner.config.candidates})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/plan":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise RequestError("request body is empty or too large")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(request, dict):
                    raise RequestError("request body must be a JSON object")
                self._send(200, service.plan(request))
            except (RequestError, json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
                self._send(400, {"error": str(error)})
            except Exception as error:  # the engine must get an answer, not a dropped socket
                self._send(500, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return Handler


def serve(planner: RootMotionPlanner, host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(PlanService(planner)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--half-width", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = PlannerConfig(
        candidates=args.candidates, flow_steps=args.flow_steps, half_width=args.half_width
    )
    planner = RootMotionPlanner.from_checkpoint(args.checkpoint, config, args.device)
    server = serve(planner, args.host, args.port)
    print(f"root-motion planner listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
