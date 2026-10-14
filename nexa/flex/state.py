"""State schema for the flex StateGraph."""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class FlexState(TypedDict, total=False):
    """Minimal state for the flex agent. Persisted via the LangGraph Postgres
    checkpointer; one row per (thread_id, checkpoint).

    Keep it small: heavy data (result rows) lives in nexa_v2_results, not here.
    """

    # `add_messages` appends new messages and deduplicates them by id.
    messages: Annotated[list, add_messages]

    # Set on the first turn and kept afterwards.
    title: Optional[str]

    # Stored so the API can read the thread's KG profile from the checkpoint.
    kg_profile_name: Optional[str]
