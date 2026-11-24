"""Async OpenAI-compatible client tailored for vLLM / SGLang multimodal serving.

Key optimization: builds every tile request with a byte-identical prefix
(`system` prompt + `PinnedContext` image(s) such as a geological map legend or
global overview) before the per-tile crop. When vLLM runs with
`--enable-prefix-caching`, the visual encoder KV cache for the pinned context
is computed once and reused across all concurrent tile requests.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .tiler import PinnedContext, Tile


@dataclass(frozen=True)
class VLMResponse:
    """Parsed response from one VLM chat-completion call."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    raw: dict[str, Any] | None = None


class VLMBackend(Protocol):
    """Protocol implemented by `AsyncVLLMClient` and mock backends in tests."""

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> VLMResponse: ...


def build_tile_messages(
    *,
    prompt: str,
    tile: Tile,
    pinned_contexts: list[PinnedContext] | None = None,
    system_prompt: str | None = None,
    image_format: str = "JPEG",
    image_quality: int = 90,
) -> list[dict[str, Any]]:
    """Construct prefix-cache-friendly OpenAI vision messages for one tile.

    Message layout (stable prefix first -> tile-specific content last):
      1. Optional `system` message (identical across all tiles)
      2. Optional pinned-context `user` turn (exact same base64 legend/overview
         bytes across all tiles -> 100% prefix-cache hit on vLLM/SGLang)
      3. Tile-specific `user` message carrying the cropped high-res tile + query
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if pinned_contexts:
        prefix_parts: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Shared reference context for all zoomed regions of this high-resolution "
                    "image (use this legend / overview to interpret symbols, colors, and codes):"
                ),
            }
        ]
        for ctx in pinned_contexts:
            prefix_parts.append({"type": "text", "text": f"[{ctx.label}]"})
            prefix_parts.append({"type": "image_url", "image_url": {"url": ctx.data_url}})
        messages.append({"role": "user", "content": prefix_parts})
        messages.append(
            {
                "role": "assistant",
                "content": "Understood. I will reference the shared context when inspecting tiles.",
            }
        )

    tile_url = tile.to_data_url(fmt=image_format, quality=image_quality)  # type: ignore[arg-type]
    tile_header = (
        f"Zoomed tile `{tile.spec.tile_id}` "
        f"(pixel region x=[{int(tile.spec.box.x1)}, {int(tile.spec.box.x2)}], "
        f"y=[{int(tile.spec.box.y1)}, {int(tile.spec.box.y2)}]):\n{prompt}"
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": tile_url}},
                {"type": "text", "text": tile_header},
            ],
        }
    )
    return messages


class AsyncVLLMClient:
    """Lightweight async client for OpenAI-compatible vLLM / SGLang endpoints."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        api_key: str = "EMPTY",
        max_concurrency: int = 8,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_concurrency = max(1, max_concurrency)
        self.timeout = timeout
        self._sem = asyncio.Semaphore(self.max_concurrency)

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> VLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with self._sem, httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]["message"]["content"]
        text = choice if isinstance(choice, str) else str(choice)
        usage = data.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        return VLMResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cached_tokens=int(prompt_details.get("cached_tokens", 0)),
            raw=data,
        )
