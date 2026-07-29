"""HTTP client for the upstream SimpleTool v2 function-calling service."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from gtp.integrations.http_transport import (
    ServiceError,
    Transport,
    decode_json,
    default_transport,
    json_request,
)

DIGITAL_HUMAN_SYSTEM_PROMPT = """You are the behavior controller for a real-time digital human.
Select exactly one available function. Resolve target IDs from the supplied scene state.
Use idle when the instruction is ambiguous or already completed."""


def _tool(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
            },
        },
    }


DIGITAL_HUMAN_TOOLS = [
    _tool(
        "move_to",
        "Walk to a scene object and stop at a safe distance.",
        {
            "target_id": {"type": "string"},
            "distance": {
                "type": "number",
                "description": "Desired stopping distance in meters.",
            },
        },
        ["target_id"],
    ),
    _tool(
        "approach",
        "Approach a scene object while maintaining social distance.",
        {
            "target_id": {"type": "string"},
            "distance": {"type": "number"},
        },
        ["target_id"],
    ),
    _tool(
        "greet_person",
        "Approach and greet a person with a hand gesture.",
        {
            "target_id": {"type": "string"},
            "distance": {
                "type": "number",
                "description": "Preferred social distance in meters.",
            },
            "hand": {"type": "string", "enum": ["left", "right"]},
            "emotion": {"type": "string"},
        },
        ["target_id"],
    ),
    _tool(
        "look_at",
        "Turn toward and look at a scene object.",
        {"target_id": {"type": "string"}},
        ["target_id"],
    ),
    _tool("idle", "Remain in place with a natural idle motion.", {}),
]


class SimpleToolClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8899",
        *,
        timeout: float = 30.0,
        transport: Transport = default_transport,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def health(self) -> dict[str, Any]:
        result = json_request(
            f"{self.base_url}/health",
            None,
            timeout=self.timeout,
            transport=self.transport,
            method="GET",
        )
        return decode_json("SimpleTool", result)

    def function_call(
        self,
        query: str,
        *,
        tools: Sequence[Mapping[str, Any]] = DIGITAL_HUMAN_TOOLS,
        system: str = DIGITAL_HUMAN_SYSTEM_PROMPT,
        history: Sequence[str] = (),
        environment: Sequence[str] = (),
    ) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        payload = {
            "messages": [{"role": "user", "content": query}],
            "tools": list(tools),
            "system": system,
            "history": list(history),
            "environment": list(environment),
            "temperature": 0.0,
            "max_tokens": 32,
        }
        document = decode_json(
            "SimpleTool",
            json_request(
                f"{self.base_url}/v1/function_call",
                payload,
                timeout=self.timeout,
                transport=self.transport,
            ),
        )
        if not document.get("success"):
            raise ServiceError(
                "SimpleTool",
                str(document.get("error") or "no function selected"),
            )
        function = document.get("function")
        arguments = document.get("args", {})
        if not isinstance(function, str) or not function:
            raise ServiceError("SimpleTool", "response has no function name")
        if not isinstance(arguments, dict):
            raise ServiceError("SimpleTool", "response args must be an object")
        return {
            "function": function,
            "arguments": arguments,
            "latency_ms": float(document.get("latency_ms", 0.0)),
        }
