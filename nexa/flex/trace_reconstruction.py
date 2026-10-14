"""Rebuild a flex trace from a thread's persisted LangGraph checkpoint.

Produces the same event shape as the live `[TOOL_SUMMARY]` stream, so the trace
panel renders identically live and when a thread is reopened. Used by the API
endpoint `GET /threads/{thread_id}/trace`.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nexa.flex.trace_format import (
    coerce_tool_payload,
    safe_args,
    summarize_tool_result,
)
from nexa.tools.output_tools import RESPOND_TO_USER_TOOL_NAME


# Prefix of the reminder injected by `graph.remind_respond`. These
# HumanMessages are not user turns.
_INTERNAL_REMINDER_MARKER = "[SYSTEM REMINDER — internal]"


def reconstruct_trace_from_messages(messages: list) -> dict:
    """Build a trace dict from a thread's messages.

    Returns:
        {
          "events": [ {name, summary, result_id, args, turn}, ... ],
          "last_final": {answer_text, cited_result_ids, highlight_rows, graph_spec} | None,
        }
    """
    events: list[dict] = []
    pending_calls: dict[str, dict] = {}  # tool_call_id -> {"name", "args"}
    last_final: Optional[dict] = None
    turn_index = 0

    for msg in messages or []:
        if isinstance(msg, HumanMessage):
            # Internal reminders are not user turns.
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.startswith(_INTERNAL_REMINDER_MARKER):
                continue
            turn_index += 1
            continue

        if isinstance(msg, AIMessage):
            for tc in (getattr(msg, "tool_calls", None) or []):
                tc_id = tc.get("id") or tc.get("name")
                if not tc_id:
                    continue
                pending_calls[tc_id] = {
                    "name": tc.get("name"),
                    "args": tc.get("args") or {},
                }
            continue

        if isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", None) or msg.name
            meta = pending_calls.pop(tc_id, None) or {"name": msg.name, "args": {}}
            args = meta.get("args") or {}
            summary, result_id = summarize_tool_result(msg, args)
            event = {
                "name": msg.name,
                "summary": summary,
                "result_id": result_id,
                "args": safe_args(args),
                "turn": turn_index,
            }
            # Completeness verdict, when assessed. An absent key means not
            # assessed, never complete.
            payload = coerce_tool_payload(msg.content) or {}
            if isinstance(payload.get("complete"), bool):
                event["complete"] = payload["complete"]
                if payload.get("completeness"):
                    event["completeness"] = payload["completeness"]
            events.append(event)

            if msg.name == RESPOND_TO_USER_TOOL_NAME:
                payload = coerce_tool_payload(msg.content) or {}
                # Forward both `answer_text` (free-form) and `answer` (structured).
                last_final = {
                    "answer_text": payload.get("answer_text"),
                    "answer": payload.get("answer"),
                    "cited_result_ids": payload.get("cited_result_ids") or [],
                    "highlight_rows": payload.get("highlight_rows"),
                    "graph_spec": payload.get("graph_spec"),
                }
            continue

    return {"events": events, "last_final": last_final}


def reconstruct_trace_from_checkpoint(checkpoint_data: Optional[dict]) -> dict:
    """Build a trace dict from a LangGraph checkpoint (see `reconstruct_trace_from_messages`)."""
    if not checkpoint_data:
        return {"events": [], "last_final": None}
    channel_values = checkpoint_data.get("channel_values") or {}
    messages = channel_values.get("messages") or []
    return reconstruct_trace_from_messages(messages)
