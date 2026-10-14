"""Flex agent trace panel.

Lists each tool step with its summary. Steps that produced a `result_id` get an
eye button that focuses the data panel on that result, cited or not.
"""

from __future__ import annotations

from typing import Iterable, Optional

import streamlit as st


_TOOL_ICONS = {
    "execute_cypher":     "⚡",
    "read_kg_schema":     "🗺️",
    "lookup_property":    "🔍",
    "list_results":       "📋",
    "read_result":        "📄",
    "filter_result":      "🪣",
    "summarize_result":   "📊",
    "aggregate_result":   "Σ",
    "write_todos":        "📝",
    "respond_to_user":    "✅",
    "validate_cypher":    "🧪",
}


_TODO_STATUS_ICONS = {
    "completed":   "✅",
    "in_progress": "🔄",
    "pending":     "☐",
}


def _render_todo_checklist(todos: list) -> None:
    """Render write_todos items as a checklist.

    In-progress items are shown in bold, completed items struck through.
    """
    lines: list[str] = []
    for t in todos:
        if not isinstance(t, dict):
            continue
        status = (t.get("status") or "pending").lower()
        content = (t.get("content") or "").strip()
        icon = _TODO_STATUS_ICONS.get(status, "☐")
        if status == "completed":
            lines.append(f"{icon} ~~{content}~~")
        elif status == "in_progress":
            lines.append(f"{icon} **{content}**")
        else:
            lines.append(f"{icon} {content}")
    if lines:
        st.markdown("\n\n".join(lines))


def render_agent_trace_flex(
    trace_events: Optional[Iterable[dict]] = None,
    *,
    live_only: bool = False,
    empty_caption: Optional[str] = None,
) -> None:
    """Render the flex trace in an expander, one row per tool step.

    Args:
        trace_events: events to render; defaults to
            `st.session_state["current_trace"]`.
        live_only: with the default events, render only those without a
            `turn` (appended live during the current turn), expanded and
            without eye buttons.
        empty_caption: placeholder when there are no events; None uses the
            default text, "" renders nothing.
    """
    if trace_events is None:
        trace_events = st.session_state.get("current_trace") or []
        if live_only:
            trace_events = [e for e in trace_events if e.get("turn") is None]
    trace_events = list(trace_events)

    if not trace_events:
        if empty_caption is None:
            st.caption("The agent trace will appear here once you ask a question.")
        elif empty_caption:
            st.caption(empty_caption)
        return

    focused = st.session_state.get("focused_result_id")
    # A click while streaming would rerun the script and abort the stream.
    is_streaming = bool(
        st.session_state.get("just_posted_message")
        or st.session_state.get("just_created_thread")
    )

    # `current_graph_kg_profile` is unset during a new thread's first stream;
    # the sidebar selection is the same profile.
    profile = (
        st.session_state.get("current_graph_kg_profile")
        or st.session_state.get("selected_kg_profile_name")
    )
    title = f"Agent trace · {len(trace_events)} step(s)"
    if profile:
        title += f"  ·  {profile}"
    # Expanded only for the live trace; persisted history stays collapsed.
    with st.expander(title, expanded=live_only):
        for i, event in enumerate(trace_events, 1):
            name = event.get("name") or "tool"
            summary = event.get("summary") or ""
            result_id = event.get("result_id")
            icon = _TOOL_ICONS.get(name, "🛠️")
            is_focused = bool(result_id) and result_id == focused

            label = f"{icon}  **{i}. {name}**  —  {summary}"
            if is_focused:
                label += "  · _active view_"
            if result_id:
                label += f"  ·  `{result_id[:8]}…`"

            text_col, btn_col = st.columns([6, 1])
            with text_col:
                st.markdown(label)
            with btn_col:
                # No button in the live trace: it is re-rendered once per event
                # in the same script run, and re-creating a keyed widget raises
                # StreamlitDuplicateElementKey.
                if result_id and not live_only:
                    if st.button(
                        "👁",
                        key=f"trace_view_{i}_{result_id}",
                        help=(
                            "Available once generation finishes"
                            if is_streaming
                            else "View the data for this step"
                        ),
                        disabled=is_streaming,
                        use_container_width=True,
                    ):
                        st.session_state["focused_result_id"] = result_id
                        st.rerun()

            args = event.get("args") or {}
            if name == "write_todos":
                # Show the plan inline.
                _render_todo_checklist(args.get("todos") or [])
            elif args:
                with st.expander("details", expanded=False):
                    if isinstance(args.get("query"), str):
                        st.code(args["query"], language="cypher")
                        rest = {k: v for k, v in args.items() if k != "query"}
                        if rest:
                            st.json(rest)
                    elif isinstance(args.get("answer_text"), str):
                        st.markdown(args["answer_text"])
                        rest = {k: v for k, v in args.items() if k != "answer_text"}
                        if rest:
                            st.json(rest)
                    else:
                        st.json(args)
