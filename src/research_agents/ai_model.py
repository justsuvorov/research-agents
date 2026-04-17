from __future__ import annotations

from abc import ABC, abstractmethod

import anthropic


class AIModel(ABC):
    @abstractmethod
    def complete(self, *, system: str, user: str, max_tokens: int) -> str:
        ...


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
