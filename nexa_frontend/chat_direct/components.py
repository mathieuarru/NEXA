"""Direct chat components: history rendering and the result data tables.

They read the direct data shape (`st.session_state.data = {"nodes": [...],
"edges": [...]}`) and the tool messages of the direct pipeline
(`retrieve_data_from_kg`).
"""

import json

import pandas as pd
import streamlit as st

from nexa_frontend.ui_components import assistant_avatar, format_structured_output, parse_thinking_content


# Persisted by the backend (`STOP_MARKER`) when the user stops a stream.
_STOP_MARKER_TEXT = "_(Generation stopped by user.)_"


def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Cast object columns to str so st.dataframe can serialize them to Arrow.

    Some KGs mix ints and strings in one column, which PyArrow rejects. Only
    the displayed copy is changed.
    """
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str)
    return out


def render_chat_history(title: str, messages: list):
    """Render a thread's messages, grouping thinking and tool calls in expanders.

    The workflow steps of the latest live turn are shown before the last
    assistant message.
    """
    live_callback_thinking_history = st.session_state.pop("last_callback_history", None)
    st.title(f"{title}")

    if not messages:
        if live_callback_thinking_history:
            with st.expander("📚 Workflow Thinking Steps", expanded=True):
                st.markdown("\n\n".join(live_callback_thinking_history), unsafe_allow_html=True)
        return

    reasoning_buffer_for_display = []
    chat_container = st.container()

    def flush_reasoning_expander(container_to_draw_in):
        nonlocal reasoning_buffer_for_display
        if reasoning_buffer_for_display:
            with container_to_draw_in.expander("🛠️ External Tool Calls", expanded=False):
                st.markdown("\n\n".join(reasoning_buffer_for_display), unsafe_allow_html=True)
            reasoning_buffer_for_display = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", ""))
        is_last_message_in_history = (i == len(messages) - 1)

        if role == "user":
            flush_reasoning_expander(chat_container)
            with chat_container.chat_message(role):
                st.markdown(content, unsafe_allow_html=True)

        elif role == "tool":
            tool_display_content = "Tool Output : Processed."
            try:
                tool_data = json.loads(content)
                if "data" in tool_data and isinstance(tool_data["data"], dict):
                    num_nodes = len(tool_data["data"].get("nodes", []))
                    num_edges = len(tool_data["data"].get("edges", []))
                    tool_display_content = f"Tool Output : Retrieved {num_nodes} nodes, {num_edges} edges."
            except (json.JSONDecodeError, TypeError):
                tool_display_content = f"Tool Output : {content[:200]}"
            reasoning_buffer_for_display.append(tool_display_content)

        elif role == "assistant":
            thinking_content, message_content = parse_thinking_content(content)

            if thinking_content:
                reasoning_buffer_for_display.append(f"🧠 <i>{thinking_content}</i>")

            for tc in msg.get("tool_calls") or []:
                tool_input = (tc.get("args") or {}).get("user_query", "")
                reasoning_buffer_for_display.append(f"Tool Input : {tool_input}")

            # Render the stop marker as a badge, not as an answer.
            if message_content.strip() == _STOP_MARKER_TEXT:
                flush_reasoning_expander(chat_container)
                with chat_container.chat_message(role, avatar=assistant_avatar()):
                    st.markdown(
                        "<span style='display:inline-block; padding:0.35em 0.85em; "
                        "border-radius:0.4rem; background:rgba(255, 75, 75, 0.12); "
                        "color:#ff4b4b; border:1px solid rgba(255, 75, 75, 0.35); "
                        "font-size:0.9em; font-weight:600;'>"
                        "⏹ Generation stopped by user"
                        "</span>",
                        unsafe_allow_html=True,
                    )
                continue

            if message_content.strip():
                flush_reasoning_expander(chat_container)

                if is_last_message_in_history and live_callback_thinking_history:
                    with chat_container.expander("📚 Workflow Thinking Steps", expanded=False):
                        st.markdown("\n\n".join(live_callback_thinking_history), unsafe_allow_html=True)

                with chat_container.chat_message(role, avatar=assistant_avatar()):
                    formatted_output = format_structured_output(message_content)
                    if formatted_output:
                        st.markdown(formatted_output, unsafe_allow_html=True)
                        with st.expander("📦 Raw JSON", expanded=False):
                            st.json(json.loads(message_content))
                    else:
                        st.markdown(message_content, unsafe_allow_html=True)

            elif is_last_message_in_history:
                flush_reasoning_expander(chat_container)
                if live_callback_thinking_history:
                    with chat_container.expander("📚 Workflow Thinking Steps", expanded=False):
                        st.markdown("\n\n".join(live_callback_thinking_history), unsafe_allow_html=True)

def render_data_display():
    """Render `st.session_state.data` as combined, node and edge tables."""
    data = st.session_state.get("data", {})

    if data and isinstance(data, dict) and ("nodes" in data or "edges" in data):
        nodes_list = data.get("nodes", [])
        edges_list = data.get("edges", [])

        df_nodes = _arrow_safe(pd.DataFrame(nodes_list) if nodes_list else pd.DataFrame())
        df_edges = _arrow_safe(pd.DataFrame(edges_list) if edges_list else pd.DataFrame())

        if df_nodes.empty and df_edges.empty:
            return

        st.subheader("🗂️ Result Data")
        tab_all, tab_nodes, tab_edges = st.tabs(["All", "Nodes", "Edges"])

        with tab_nodes:
            st.dataframe(df_nodes, width="stretch")

        with tab_edges:
            st.dataframe(df_edges, width="stretch")

        with tab_all:
            df_combined = pd.DataFrame()
            if not df_nodes.empty:
                df_nodes_copy = df_nodes.copy()
                df_nodes_copy["_kind_"] = "node"
                df_combined = pd.concat([df_combined, df_nodes_copy], ignore_index=True)
            if not df_edges.empty:
                df_edges_copy = df_edges.copy()
                df_edges_copy["_kind_"] = "edge"
                df_combined = pd.concat([df_combined, df_edges_copy], ignore_index=True)

            NODE_ACCENT = "#7E57C2"
            NODE_BG = "rgba(126, 87, 194, 0.16)"
            EDGE_ACCENT = "#F59E0B"
            EDGE_BG = "rgba(245, 158, 11, 0.12)"

            STYLE = {
                "node": f"background-color: {NODE_BG}; border-left: 4px solid {NODE_ACCENT};",
                "edge": f"background-color: {EDGE_BG}; border-left: 4px solid {EDGE_ACCENT};",
            }

            df_display = df_combined.drop(columns=["_kind_"]) if "_kind_" in df_combined.columns else df_combined

            def _style_row(row):
                kind = df_combined.loc[row.name, "_kind_"] if "_kind_" in df_combined.columns else None
                style = STYLE.get(kind, "")
                return [style] * len(row)

            styled_df = df_display.style.apply(_style_row, axis=1)
            st.dataframe(styled_df, width="stretch")
