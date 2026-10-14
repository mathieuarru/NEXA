"""Flex graph view: fetch and render the subgraph behind a stored result.

`/results/{id}/graph` rewrites the stored Cypher to return its entities. The
subgraph is rendered with `graph_utils.render_graph` and cached per result_id
in `st.session_state.graph_cache`.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

from nexa_frontend.graph_utils import render_graph


API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")


def fetch_graph_for_result(result_id: str) -> dict[str, Any]:
    """Fetch a result's subgraph from `/results/{id}/graph`, with caching.

    The entry always has `error` (None on success), `nodes`, `edges`,
    `rewritten_query`, `kg_profile_name` and `note`.
    """
    cache: dict[str, Any] = st.session_state.setdefault("graph_cache", {})
    if result_id in cache:
        return cache[result_id]

    entry: dict[str, Any] = {
        "error": None,
        "nodes": [],
        "edges": [],
        "rewritten_query": None,
        "kg_profile_name": None,
        "note": None,
    }
    try:
        resp = requests.get(f"{API_BASE_URL}/results/{result_id}/graph", timeout=60)
    except requests.exceptions.RequestException as e:
        entry["error"] = f"API connection error: {e}"
        cache[result_id] = entry
        return entry

    if resp.status_code == 200:
        payload = resp.json()
        entry["nodes"] = payload.get("nodes") or []
        entry["edges"] = payload.get("edges") or []
        entry["rewritten_query"] = payload.get("rewritten_query")
        entry["kg_profile_name"] = payload.get("kg_profile_name")
        entry["note"] = payload.get("note")
    elif resp.status_code == 422:
        # No entities to return (e.g. `RETURN count(*)`), or a derived result
        # without its own Cypher.
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            detail = resp.text
        entry["error"] = detail or "No entities to visualize for this result."
    elif resp.status_code == 404:
        entry["error"] = "Result not found on the server."
    else:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        entry["error"] = f"Server error ({resp.status_code}): {detail}"

    cache[result_id] = entry
    return entry


def render_graph_for_result(result_id: str) -> None:
    """Render the subgraph of a stored result with `graph_utils.render_graph`."""
    entry = fetch_graph_for_result(result_id)

    if entry["error"]:
        st.info(entry["error"])
        return

    if not entry["nodes"]:
        st.caption(entry.get("note") or "No graph entities to visualize for this result.")
        if entry.get("rewritten_query"):
            with st.expander("Rewritten query", expanded=False):
                st.code(entry["rewritten_query"], language="cypher")
        return

    # render_graph reads `data` and `current_graph_kg_profile` from session
    # state: set them for this call, then restore them.
    prev_data = st.session_state.get("data", {})
    prev_profile = st.session_state.get("current_graph_kg_profile")
    st.session_state.data = {"nodes": entry["nodes"], "edges": entry["edges"]}
    if entry.get("kg_profile_name"):
        st.session_state.current_graph_kg_profile = entry["kg_profile_name"]

    try:
        render_graph()
    finally:
        st.session_state.data = prev_data
        st.session_state.current_graph_kg_profile = prev_profile

    if entry.get("rewritten_query"):
        with st.expander("Rewritten query for the graph", expanded=False):
            st.code(entry["rewritten_query"], language="cypher")
