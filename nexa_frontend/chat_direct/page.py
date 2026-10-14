"""Direct chat page.

Renders the chat and, when the thread's last tool result holds data
(`st.session_state.data = {"nodes": [...], "edges": [...]}`), its tables,
graph and Cypher query.
"""

import json
from typing import Tuple

import streamlit as st

from nexa_frontend.api_utils import get_thread_history
from nexa_frontend.chat_direct.components import render_chat_history, render_data_display
from nexa_frontend.graph_utils import render_cypher_query_and_link, render_graph
from nexa_frontend.ui_components import handle_streaming, render_user_input, render_welcome_logo


def render_chat_page_direct(is_model_ready: bool) -> None:
    """Render the direct chat page."""
    processed_messages = []
    initial_data = None
    last_cypher_query = None
    thread_title = "New Conversation"
    kg_profile_for_current_graph = None

    current_thread_id = st.session_state.get("current_thread_id")

    if current_thread_id and not st.session_state.just_created_thread:
        thread_title, processed_messages, initial_data, last_cypher_query, kg_profile_for_current_graph = (
            get_processed_thread_data(current_thread_id)
        )
        st.session_state.current_graph_kg_profile = kg_profile_for_current_graph
    else:
        st.session_state.current_graph_kg_profile = None

    st.session_state.data = initial_data if initial_data else {}
    has_data_to_display = bool(st.session_state.data)

    if has_data_to_display:
        col1, col2 = st.columns(2)
        with col1:
            if current_thread_id and not st.session_state.just_created_thread:
                render_chat_history(thread_title, processed_messages)
            handle_streaming(is_model_ready)
            # Keep the chat_input visible (disabled) during streaming so
            # the Stop overlay can replace its send button in place.
            _streaming = st.session_state.just_created_thread or st.session_state.just_posted_message
            render_user_input(is_model_ready, disabled=_streaming)
            if not _streaming:
                from nexa_frontend.streaming_control import render_stop_button
                render_stop_button(current_thread_id, streaming=False)
        with col2:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            with st.container():
                render_data_display()
            with st.container():
                if "edges" in st.session_state.get("data", {}):
                    render_graph()

            render_cypher_query_and_link(last_cypher_query, kg_profile_for_current_graph)

    else:
        if current_thread_id and not st.session_state.just_created_thread:
            render_chat_history(thread_title, processed_messages)
        elif not current_thread_id:
            render_welcome_logo()

        handle_streaming(is_model_ready)
        _streaming = st.session_state.just_created_thread or st.session_state.just_posted_message
        render_user_input(is_model_ready, disabled=_streaming)
        if not _streaming:
            from nexa_frontend.streaming_control import render_stop_button
            render_stop_button(current_thread_id, streaming=False)


def get_processed_thread_data(thread_id: str) -> Tuple[str, list, dict | None, str | None, str | None]:
    """Fetch a thread's history.

    Returns (title, messages, last tool data, last Cypher query, KG profile name).
    """
    title = f"Conversation {thread_id[:8]}"
    messages = []
    last_tool_data = None
    last_cypher_query = None
    kg_profile_name_from_thread = None
    try:
        thread_history = get_thread_history(thread_id)
        if thread_history:
            title = thread_history.get("title", title)
            kg_profile_name_from_thread = thread_history.get("kg_profile_name")
            if isinstance(thread_history.get("messages"), list):
                messages = thread_history["messages"]
                for msg in reversed(messages):
                    if msg.get("role") == "tool":
                        try:
                            tool_content = json.loads(msg.get("content", "{}"))

                            if last_tool_data is None:
                                potential_data = tool_content.get("data")
                                if isinstance(potential_data, list) and potential_data:
                                    last_tool_data = {"nodes": potential_data}
                                elif isinstance(potential_data, dict) and potential_data:
                                    last_tool_data = potential_data

                            if last_cypher_query is None:
                                potential_query = tool_content.get("cypher_query")
                                if isinstance(potential_query, str) and potential_query:
                                    last_cypher_query = potential_query

                            if last_tool_data is not None and last_cypher_query is not None:
                                break
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        st.sidebar.error(f"Error processing thread data for {thread_id}: {e}")
        return title, [], None, None, None

    return title, messages, last_tool_data, last_cypher_query, kg_profile_name_from_thread
