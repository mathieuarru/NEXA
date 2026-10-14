"""Token-budget helpers: context-window lookup, token counting and the KG result budget.

Optional environment variables:
    NEXA_LLM_CONTEXT_WINDOW   override the context window of the current model
    NEXA_OUTPUT_RESERVE       tokens reserved for the generated response (default 4000)
    NEXA_OTHER_RESERVE        tokens reserved for system prompt, schema and tool definitions (default 5000)
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages.utils import count_tokens_approximately

from nexa.config import params


def get_model_name(llm: Any) -> str:
    """Return the model name of a LangChain chat model, or "" if none is found.

    The attribute depends on the provider: ``model_name``, ``model``,
    ``deployment_name`` or ``azure_deployment``.
    """
    if llm is None:
        return ""
    for attr in ("model_name", "model", "deployment_name", "azure_deployment"):
        val = getattr(llm, attr, None)
        if val:
            return str(val)
    return ""


def get_context_window(model_name: str) -> int:
    """Return the context window, in tokens, for ``model_name``.

    Uses ``NEXA_LLM_CONTEXT_WINDOW`` if set, else the first key of
    ``params._CONTEXT_WINDOWS`` found in the lowercased name, else
    ``params._DEFAULT_CONTEXT_WINDOW``.
    """
    override = os.getenv("NEXA_LLM_CONTEXT_WINDOW")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    name = (model_name or "").lower()
    for key, window in params._CONTEXT_WINDOWS.items():
        if key in name:
            return window
    return params._DEFAULT_CONTEXT_WINDOW


def count_tokens(text: str, model: str = "") -> int:
    """Count the tokens in ``text``.

    Uses tiktoken with the encoding of ``model`` when known, else ``o200k_base``
    (exact for OpenAI models). LangChain's approximate counter undercounts dense
    tabular text, so it is only a fallback, followed by ``len(text) / 3``.
    """
    if not text:
        return 0
    s = str(text)

    try:
        import tiktoken
        if model:
            try:
                enc = tiktoken.encoding_for_model(model)
                return len(enc.encode(s))
            except KeyError:
                pass
        # o200k_base covers gpt-4o and later OpenAI models.
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(s))
    except Exception:
        pass

    try:
        return count_tokens_approximately([{"role": "user", "content": s}])
    except Exception:
        return int(len(s) / 3)


def compute_kg_budget(context_window: int) -> int:
    """Return the token budget for the KG result block in the final LLM call.

    budget = window - output_reserve - other_reserve - max(1000, window // 20),
    with a floor of 2000. The reserves come from NEXA_OUTPUT_RESERVE and
    NEXA_OTHER_RESERVE.
    """
    try:
        output_reserve = int(os.getenv("NEXA_OUTPUT_RESERVE", "4000"))
    except ValueError:
        output_reserve = 4000
    try:
        other_reserve = int(os.getenv("NEXA_OTHER_RESERVE", "5000"))
    except ValueError:
        other_reserve = 5000

    safety = max(1000, context_window // 20)
    return max(2000, context_window - output_reserve - other_reserve - safety)


def with_output_cap(llm: Any, max_tokens: int) -> Any:
    """Return a copy of ``llm`` capped at ``max_tokens`` generated tokens (Ollama models only).

    Bounds repetition loops on calls whose answer is short. Other providers are
    returned unchanged: they enforce their own output limit, and a low cap on a
    hosted reasoning model can leave no tokens for the answer. The shared
    instance is not modified.
    """
    if llm is None or "num_predict" not in type(llm).model_fields:
        return llm
    return llm.model_copy(update={"num_predict": max_tokens})
