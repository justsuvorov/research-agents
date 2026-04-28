"""
GeminiAdapter — Anthropic-API-compatible facade over google-genai.

Exposes only the slice of the Anthropic client surface used by EngineeringCalculator
(and other data-agent components):
    client.messages.create(model=, max_tokens=, system=[...], messages=[...])
        → response.content[0].text

Gemini API errors are wrapped as anthropic.APIError so that the calculator's
existing exception handler works without modification.
"""

from __future__ import annotations

import time

import anthropic
from google import genai
from google.genai import types
from loguru import logger


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_TextBlock(text)]


class _Messages:
    def __init__(self, client: genai.Client, default_model: str) -> None:
        self._client = client
        self._default_model = default_model

    def create(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        system: list | str | None = None,
        messages: list | None = None,
        **_: object,
    ) -> _Response:
        system_text = self._system_text(system)
        contents = self._contents(messages or [])

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system_text or None,
        )

        last_exc: Exception | None = None
        for attempt in range(1, 5):
            try:
                response = self._client.models.generate_content(
                    model=self._default_model,
                    contents=contents,
                    config=config,
                )
                text = response.text or ""
                return _Response(text)
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                transient = any(code in msg for code in ("503", "429", "500", "UNAVAILABLE"))
                if not transient or attempt == 4:
                    break
                backoff = 2 ** attempt
                logger.warning(
                    "[GeminiAdapter] transient error (attempt {}): {} — retrying in {}s",
                    attempt, msg.split('\n', 1)[0], backoff,
                )
                time.sleep(backoff)

        raise anthropic.APIError(
            f"Gemini API error: {last_exc}", request=None, body=None
        ) from last_exc

    @staticmethod
    def _system_text(system: list | str | None) -> str:
        if system is None:
            return ""
        if isinstance(system, str):
            return system
        parts: list[str] = []
        for blk in system:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(str(blk.get("text", "")))
            elif isinstance(blk, str):
                parts.append(blk)
        return "\n".join(parts)

    @staticmethod
    def _contents(messages: list) -> list[types.Content]:
        out: list[types.Content] = []
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content", "")
            if isinstance(content, list):
                text = "".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = str(content)
            out.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
        return out


class GeminiAdapter:
    """Drop-in replacement for `anthropic.Anthropic` exposing `.messages.create()`."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self.messages = _Messages(self._client, model)
