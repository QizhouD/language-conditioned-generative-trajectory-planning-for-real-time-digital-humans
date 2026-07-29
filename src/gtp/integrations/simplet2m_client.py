"""HTTP client for SimpleText2Motion's binary VRMA endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gtp.integrations.http_transport import (
    ServiceError,
    Transport,
    default_transport,
)
from urllib.request import Request


@dataclass(frozen=True)
class MotionAsset:
    prompt: str
    duration: float
    seed: int | None
    content_type: str
    data: bytes

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.data)
        return output


class SimpleText2MotionClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        *,
        timeout: float = 120.0,
        transport: Transport = default_transport,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def generate(
        self, prompt: str, *, duration: float = 3.0, seed: int | None = None
    ) -> MotionAsset:
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not 1.0 <= duration <= 10.0:
            raise ValueError("duration must be between 1 and 10 seconds")
        payload: dict[str, object] = {"prompt": prompt, "duration": duration}
        if seed is not None:
            if not 0 <= seed <= 0xFFFFFFFF:
                raise ValueError("seed must be an unsigned 32-bit integer")
            payload["seed"] = seed
        request = Request(
            f"{self.base_url}/t2m",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "model/gltf-binary",
            },
            method="POST",
        )
        result = self.transport(request, self.timeout)
        if not 200 <= result.status < 300:
            detail = result.body.decode("utf-8", errors="replace").strip()
            raise ServiceError(
                "SimpleText2Motion", detail or "generation failed", result.status
            )
        content_type = next(
            (
                value
                for key, value in result.headers.items()
                if key.lower() == "content-type"
            ),
            "",
        ).split(";", 1)[0]
        if content_type != "model/gltf-binary":
            raise ServiceError(
                "SimpleText2Motion",
                f"unexpected content type: {content_type or 'missing'}",
                result.status,
            )
        if len(result.body) < 4 or result.body[:4] != b"glTF":
            raise ServiceError(
                "SimpleText2Motion", "response is not a GLB/VRMA asset", result.status
            )
        return MotionAsset(prompt, duration, seed, content_type, result.body)
