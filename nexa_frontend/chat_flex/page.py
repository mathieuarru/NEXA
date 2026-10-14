"""Flex chat page.

The chat is on the left and, once a result is cited or focused, the data panel
is on the right. Each turn's agent trace is shown inline between the user
message and the reply; a step's eye button focuses the data panel on it.
"""

import streamlit as st

from nexa_frontend.chat_direct.page import get_processed_thread_data
from nexa_frontend.chat_flex.chat_history import render_chat_history_flex
from nexa_frontend.chat_flex.data_panel import render_data_panel_flex
from nexa_frontend.chat_flex.state import (
    hydrate_trace_from_backend,
    init_flex_state,
    reset_flex_buffers_for_new_turn,
)
from nexa_frontend.ui_components import handle_streaming, render_user_input, render_welcome_logo


def render_chat_page_flex(is_model_ready: bool) -> None:
    """Render the flex chat page."""
    init_flex_state()
    reset_flex_buffers_for_new_turn()

    processed_messages = []
    thread_title = "New Conversation"
    current_thread_id = st.session_state.get("current_thread_id")

    if current_thread_id and not st.session_state.just_created_thread:
        # Shared with the direct page; tool messages carry the `name` needed
        # to find respond_to_user.
        thread_title, processed_messages, _initial_data, _last_cypher, kg_profile_for_current_graph = (
            get_processed_thread_data(current_thread_id)
        )
        st.session_state.current_graph_kg_profile = kg_profile_for_current_graph
    else:
        st.session_state.current_graph_kg_profile = None

    # Always hydrate: it clears stale state on a thread switch and is a no-op
    # while streaming or when the trace is already loaded.
    hydrate_trace_from_backend(current_thread_id)

    # Flex data comes from the result store, not from `st.session_state.data`.
    st.session_state.data = {}

    # Placeholders don't survive reruns; handle_streaming recreates it while streaming.
    st.session_state.pop("_flex_trace_placeholder", None)

    # Two columns only when there is data to show (a cited or focused result).
    _last_final = st.session_state.get("last_final") or {}
    _cited = list(_last_final.get("cited_result_ids") or [])
    _focused = st.session_state.get("focused_result_id")
    has_data_to_display = bool(current_thread_id) and (bool(_cited) or bool(_focused))

    def _render_chat_column() -> None:
        if current_thread_id and not st.session_state.just_created_thread:
            render_chat_history_flex(thread_title, processed_messages)
        elif not current_thread_id:
            render_welcome_logo()
        handle_streaming(is_model_ready)
        # Keep the chat input visible (disabled) while streaming; the Stop
        # overlay replaces its send button.
        _streaming = st.session_state.just_created_thread or st.session_state.just_posted_message
        render_user_input(is_model_ready, disabled=_streaming)
        if not _streaming:
            # Remove a Stop button left in the DOM by the previous turn.
            from nexa_frontend.streaming_control import render_stop_button
            render_stop_button(current_thread_id, streaming=False)

    if has_data_to_display:
        col1, col2 = st.columns(2)
        with col1:
            _render_chat_column()
        with col2:
            with st.container():
                render_data_panel_flex()
    else:
        _render_chat_column()
