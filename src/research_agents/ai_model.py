from __future__ import annotations

from abc import ABC, abstractmethod

import anthropic
from google import genai
from google.genai import types


class AIModel(ABC):
    @abstractmethod
    def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        ...


class GeminiModel(AIModel):

    def __init__(self, client: genai.Client, model: str = "gemini-2.5-flash") -> None:
        self._client = client
        self._model = model

    def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text


class AnthropicModel(AIModel):

    def __init__(self, client: anthropic.Anthropic, model: str = "claude-sonnet-4-6") -> None:
        self._client = client
        self._model = model

    def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
