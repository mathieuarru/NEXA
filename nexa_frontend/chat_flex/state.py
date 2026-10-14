"""Flex session state and trace hydration.

During streaming, `StreamlitCallback` appends events to
`st.session_state["current_trace"]`. When a thread is opened, or after a turn
completes, `hydrate_trace_from_backend` replaces them with the trace rebuilt
from the checkpoint by `GET /threads/{id}/trace`.
"""

import streamlit as st

from nexa_frontend.api_utils import fetch_thread_trace


# Flex keys, written while streaming and read by the flex page components.
# Mutable defaults are factories, so that sessions never share one object.
_FLEX_KEYS = {
    "current_trace": list,        # list[dict]: tool_summary events (live or hydrated)
    "last_final": None,           # dict | None: payload of the latest respond_to_user [FINAL]
    "focused_result_id": None,    # str | None: result of the trace step the user clicked
    "graph_cache": dict,          # dict[result_id, {nodes, edges, error?}]: graphs fetched on demand
    "hydrated_thread_id": None,   # str | None: thread whose trace is loaded (avoids refetch loops)
}


def init_flex_state() -> None:
    """Ensure flex-specific keys exist with safe defaults. Idempotent."""
    for k, default in _FLEX_KEYS.items():
        if k not in st.session_state:
            st.session_state[k] = default() if callable(default) else default


def reset_flex_buffers_for_new_turn() -> None:
    """Clear the focused result when a new message is being streamed.

    `current_trace` and `last_final` are kept: the live stream appends to them
    and hydration replaces them after the rerun.
    """
    if st.session_state.get("just_created_thread") or st.session_state.get("just_posted_message"):
        st.session_state["focused_result_id"] = None


def hydrate_trace_from_backend(thread_id: str | None, force: bool = False) -> None:
    """Load the persisted trace of `thread_id` into session state.

    Clears the flex state on a thread switch (including to or from no thread).
    Skips the fetch while streaming, since the callback owns the trace, and
    when the thread is already hydrated unless `force` is set.
    """
    _SENTINEL = "__never_seen__"
    previous = st.session_state.get("last_seen_thread_id", _SENTINEL)

    # 1. Thread switch: clear the previous thread's state. The sentinel
    #    skips this on the first call.
    if previous != _SENTINEL and previous != thread_id:
        st.session_state["current_trace"] = []
        st.session_state["last_final"] = None
        st.session_state["focused_result_id"] = None
        st.session_state["hydrated_thread_id"] = None
    st.session_state["last_seen_thread_id"] = thread_id

    # 2. No active thread.
    if not thread_id:
        return

    # 3. Streaming: the callback owns the trace.
    if st.session_state.get("just_created_thread") or st.session_state.get("just_posted_message"):
        return

    # 4. Already hydrated.
    if not force and st.session_state.get("hydrated_thread_id") == thread_id:
        return

    # 5. Fetch and install.
    payload = fetch_thread_trace(thread_id)
    if payload is None:
        return
    st.session_state["current_trace"] = payload.get("events") or []
    st.session_state["last_final"] = payload.get("last_final")
    st.session_state["hydrated_thread_id"] = thread_id


def invalidate_trace_hydration() -> None:
    """Mark the trace as stale so the next render fetches it again.

    Called after a streamed turn so the rerun loads the persisted events.
    """
    st.session_state["hydrated_thread_id"] = None
