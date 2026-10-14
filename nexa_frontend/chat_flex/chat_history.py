"""Flex chat history rendering.

A flex turn ends with a `respond_to_user` tool call, so assistant bubbles are
rebuilt from its result: `answer_text`, or `answer` for structured output.
Plain-text assistant messages are shown as a fallback. Each turn's trace is
shown between the user message and the reply, grouped by the `turn` index set
by the backend.
"""

from __future__ import annotations

import json
from collections import defaultdict

import streamlit as st

from nexa_frontend.chat_flex.trace import render_agent_trace_flex
from nexa_frontend.ui_components import assistant_avatar, format_structured_output


# Persisted by the backend (`STOP_MARKER`) when the user stops a stream.
_STOP_MARKER_TEXT = "_(Generation stopped by user.)_"


def render_chat_history_flex(title: str, messages: list) -> None:
    """Render the persisted flex chat, with each turn's trace before its reply."""
    st.title(f"{title}")

    if not messages:
        return

    # Group persisted events by turn. Live events (no `turn` yet) are
    # rendered by handle_streaming.
    events_by_turn: dict[int, list[dict]] = defaultdict(list)
    for ev in st.session_state.get("current_trace") or []:
        t = ev.get("turn")
        if isinstance(t, int):
            events_by_turn[t].append(ev)

    user_turn_index = 0
    pending_trace_rendered_for_turn = False

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            user_turn_index += 1
            pending_trace_rendered_for_turn = False
            with st.chat_message("user"):
                st.markdown(content if isinstance(content, str) else str(content), unsafe_allow_html=True)
            continue

        if role == "tool":
            payload = _safe_json_load(content) if isinstance(content, str) else content
            tool_name = msg.get("name")
            looks_like_final = isinstance(payload, dict) and (
                "answer_text" in payload or "answer" in payload
            )
            if tool_name == "respond_to_user" or looks_like_final:
                if not pending_trace_rendered_for_turn:
                    _render_turn_trace(events_by_turn.get(user_turn_index))
                    pending_trace_rendered_for_turn = True
                with st.chat_message("assistant", avatar=assistant_avatar()):
                    _render_respond_payload(payload)
            # Other tool messages appear in the trace only.
            continue

        if role == "assistant":
            # Plain-text reply (respond_to_user not called) or the stop marker.
            has_tool_calls = bool(msg.get("tool_calls"))
            text = content if isinstance(content, str) else ""
            if text.strip() == _STOP_MARKER_TEXT:
                if not pending_trace_rendered_for_turn:
                    _render_turn_trace(events_by_turn.get(user_turn_index))
                    pending_trace_rendered_for_turn = True
                _render_stopped_badge()
                continue
            if not has_tool_calls and text.strip():
                if not pending_trace_rendered_for_turn:
                    _render_turn_trace(events_by_turn.get(user_turn_index))
                    pending_trace_rendered_for_turn = True
                with st.chat_message("assistant", avatar=assistant_avatar()):
                    st.markdown(text, unsafe_allow_html=True)
            # Messages with only tool calls are skipped.
            continue


def _render_stopped_badge() -> None:
    """Render the badge of a turn stopped by the user, styled like the Stop button."""
    with st.chat_message("assistant", avatar=assistant_avatar()):
        st.markdown(
            "<span style='display:inline-block; padding:0.35em 0.85em; "
            "border-radius:0.4rem; background:rgba(255, 75, 75, 0.12); "
            "color:#ff4b4b; border:1px solid rgba(255, 75, 75, 0.35); "
            "font-size:0.9em; font-weight:600;'>"
            "⏹ Generation stopped by user"
            "</span>",
            unsafe_allow_html=True,
        )


def _render_turn_trace(events: list[dict] | None) -> None:
    """Render a turn's trace events; render nothing when there are none."""
    if not events:
        return
    render_agent_trace_flex(trace_events=events, empty_caption="")


def _render_respond_payload(payload) -> None:
    """Render a respond_to_user payload, structured (`answer`) or free-form (`answer_text`)."""
    if not isinstance(payload, dict):
        return
    # Prefer the structured answer when present.
    structured = payload.get("answer")
    if isinstance(structured, dict) and structured:
        formatted = format_structured_output(json.dumps(structured, ensure_ascii=False))
        if formatted:
            st.markdown(formatted, unsafe_allow_html=True)
            with st.expander("📦 Raw JSON", expanded=False):
                st.json(structured)
        else:
            st.json(structured)
        return
    # Free-form variant.
    answer_text = payload.get("answer_text")
    if isinstance(answer_text, str) and answer_text:
        st.markdown(answer_text, unsafe_allow_html=True)


def _safe_json_load(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None
