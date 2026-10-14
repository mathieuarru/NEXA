"""Flex data panel.

Shows one tab per result cited in the latest `respond_to_user` payload or, when
a trace step is focused, that single result with a back button. Each result
shows its rows, its subgraph, and its Cypher query with a Neo4j Browser link.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from nexa_frontend.api_utils import fetch_result_by_id
from nexa_frontend.chat_flex.graph_view import render_graph_for_result
from nexa_frontend.chat_direct.components import _arrow_safe
from nexa_frontend.graph_utils import render_cypher_query_and_link


def _flatten_rows_for_display(rows: list) -> pd.DataFrame:
    """Flatten stored rows into a DataFrame with one column per property.

    Node/relationship cells (e.g. from ``RETURN s, r, t``) are expanded into
    ``alias.property`` columns, the dotted names also used by the result tools.
    Scalar columns pass through; nested values are JSON-encoded so the frame
    stays Arrow-compatible.
    """
    flat_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            flat_rows.append({"value": row})
            continue
        flat: dict[str, Any] = {}
        for col, val in row.items():
            if isinstance(val, dict):
                # Expand node/relationship dicts one level into alias.prop.
                for sub_k, sub_v in val.items():
                    key = f"{col}.{sub_k}"
                    flat[key] = json.dumps(sub_v, ensure_ascii=False) if isinstance(sub_v, (dict, list)) else sub_v
            elif isinstance(val, list):
                flat[col] = json.dumps(val, ensure_ascii=False)
            else:
                flat[col] = val
        flat_rows.append(flat)
    return _arrow_safe(pd.DataFrame(flat_rows))


def render_data_panel_flex() -> None:
    """Render the focused result, or the results cited by the latest answer."""
    focused = st.session_state.get("focused_result_id")
    final = st.session_state.get("last_final") or {}
    cited = list(final.get("cited_result_ids") or [])

    if focused:
        _render_drill_down(focused, cited)
        return

    if not cited:
        st.caption(
            "Results cited by the agent (`respond_to_user`) will appear here. "
            "You can also click 👁 on a step in the trace to inspect an "
            "intermediate result."
        )
        return

    if len(cited) == 1:
        _render_single_result(cited[0])
        return

    tabs = st.tabs([f"Result {idx + 1}" for idx, _ in enumerate(cited)])
    for tab, rid in zip(tabs, cited):
        with tab:
            _render_single_result(rid)


def _render_drill_down(result_id: str, cited: list[str]) -> None:
    is_cited = result_id in cited
    badge = "✅ cited" if is_cited else "🔎 intermediate"
    cols = st.columns([4, 1])
    with cols[0]:
        st.markdown(f"### 📊 Focused result · {badge}")
    with cols[1]:
        if st.button("← Back", key="drill_down_back", use_container_width=True):
            st.session_state["focused_result_id"] = None
            st.rerun()
    _render_single_result(result_id)


def _render_single_result(result_id: str) -> None:
    record = fetch_result_by_id(result_id)
    if record is None:
        st.warning(f"Result `{result_id[:8]}…` not found.")
        return

    # Same layout as the direct page: rows, subgraph, then the Cypher query.
    st.subheader("🗂️ Result Data")
    rows = record.get("data") or []
    if rows:
        try:
            df = _flatten_rows_for_display(rows)
            st.dataframe(df, width="stretch")
        except Exception:
            st.json(rows)
    else:
        st.caption("(no rows)")

    render_graph_for_result(result_id)

    cypher = record.get("cypher_query")
    if cypher:
        kg_profile = st.session_state.get("current_graph_kg_profile")
        render_cypher_query_and_link(cypher, kg_profile)
