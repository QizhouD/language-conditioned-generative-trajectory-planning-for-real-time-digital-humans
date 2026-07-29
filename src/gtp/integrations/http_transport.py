"""Small injectable standard-library HTTP transport for local model services."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: Mapping[str, str]
    body: bytes


class ServiceError(RuntimeError):
    def __init__(self, service: str, message: str, status: int | None = None) -> None:
        self.service = service
        self.status = status
        prefix = f"{service} HTTP {status}" if status is not None else service
        super().__init__(f"{prefix}: {message}")


Transport = Callable[[Request, float], HttpResult]


def default_transport(request: Request, timeout: float) -> HttpResult:
    try:
        with urlopen(request, timeout=timeout) as response:
            return HttpResult(
                status=response.status,
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        return HttpResult(error.code, dict(error.headers.items()), detail.encode())
    except URLError as error:
        raise ServiceError("HTTP transport", str(error.reason)) from error


def json_request(
    url: str,
    payload: Mapping[str, Any] | None,
    *,
    timeout: float,
    transport: Transport,
    method: str = "POST",
) -> HttpResult:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    return transport(Request(url, data=body, headers=headers, method=method), timeout)


def decode_json(service: str, result: HttpResult) -> dict[str, Any]:
    if not 200 <= result.status < 300:
        detail = result.body.decode("utf-8", errors="replace").strip()
        raise ServiceError(service, detail or "request failed", result.status)
    try:
        document = json.loads(result.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ServiceError(service, "response is not valid JSON", result.status) from error
    if not isinstance(document, dict):
        raise ServiceError(service, "JSON response must be an object", result.status)
    return document
