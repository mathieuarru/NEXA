"""Helpers for LangChain message payloads."""

from __future__ import annotations

from typing import Any
import json


def normalize_message_content(content: Any) -> str:
    """Return the text of an LLM message ``content`` field.

    Content is either a string or a list of blocks such as
    ``[{"type": "text", "text": "..."}]`` (Anthropic). Tool-use and delta
    blocks are dropped.
    """

    def _convert(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)

        if isinstance(value, dict):
            block_type = value.get("type")
            if isinstance(block_type, str):
                lowered = block_type.lower()
                if lowered == "tool_use" or lowered.startswith("tool_") or lowered.endswith("_delta"):
                    return ""

            text_value = value.get("text")
            if isinstance(text_value, str):
                return text_value

            if "content" in value:
                return _convert(value["content"])
            if "parts" in value and isinstance(value["parts"], list):
                return "".join(_convert(part) for part in value["parts"])

            return json.dumps(value, ensure_ascii=False)

        if isinstance(value, list):
            return "".join(_convert(item) for item in value)

        return str(value)

    return _convert(content)
