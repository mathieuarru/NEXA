"""Graph and Cypher query rendering shared by the direct and flex pages."""

import streamlit as st
import json
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle
from urllib.parse import quote
from typing import Optional, List, Dict, Any
from nexa_frontend.api_utils import get_kg_profile, get_label_color_mapping
import logging

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Display-name properties in priority order, as in `nexa.core.nexa_agent`. No
# single one is universal: AlzKB uses `commonName`/`geneSymbol`, Hetionet and
# PrimeKG use `name`.
CAPTION_KEY_PRIORITY = ["name", "commonName", "geneSymbol", "symbol", "title"]


def get_node_caption_value(node_data: Dict[str, Any], caption_key: str = "name") -> str:
    """Return the best human-readable caption, falling back to the label and the id.

    `caption_key` is tried first.
    """
    for key in [caption_key, *CAPTION_KEY_PRIORITY]:
        caption = node_data.get(key)
        if caption:
            return str(caption)
    # No display property: use the label with the id.
    label = node_data.get("label")
    node_id = node_data.get("id", "N/A")
    return f"{label} {node_id}" if label else str(node_id)

def render_graph():
    """Render `st.session_state.data` as a graph, colored by the KG profile's label colors."""
    st.subheader("🕸️ Knowledge Subgraph")
    graph_render_data = st.session_state.get("data", {})

    profile_name_for_styles = st.session_state.get("current_graph_kg_profile")

    if not profile_name_for_styles:
        profile_name_for_styles = st.session_state.get("selected_kg_profile_name")
        if st.session_state.get("current_graph_kg_profile") is None and graph_render_data and graph_render_data.get("nodes"):
            logger.debug("No KG profile specific to the graph data. Using current session profile for styles.")
        elif not profile_name_for_styles and graph_render_data and graph_render_data.get("nodes"):
            logger.debug("No graph-specific KG profile and no selected session profile. Using fallback colors only.")

    try:
        transformed_data = json.loads(json.dumps(graph_render_data))
    except (TypeError, json.JSONDecodeError) as e:
        st.error(f"Error copying/preparing graph data: {e}.")
        st.json(graph_render_data)
        return

    if transformed_data and isinstance(transformed_data, dict) and transformed_data.get("nodes"):
        try:
            if "edges" in transformed_data and transformed_data["edges"] is not None:
                transformed_edges = []
                for edge in transformed_data["edges"]:
                    edge_data = edge.copy()

                    if 'source' in edge_data:
                        edge_data['source_property'] = edge_data.pop('source')
                    if 'target' in edge_data:
                        edge_data['target_property'] = edge_data.pop('target')

                    if "source_node" in edge_data:
                        edge_data["source"] = str(edge_data.pop("source_node"))
                    if "target_node" in edge_data:
                        edge_data["target"] = str(edge_data.pop("target_node"))
                    transformed_edges.append(edge_data)
                transformed_data["edges"] = transformed_edges
            else:
                transformed_data["edges"] = []

            caption_key_for_nodes = "name"

            label_color_map: Dict[str, str] = {}
            if profile_name_for_styles:
                fetched_map = get_label_color_mapping(profile_name_for_styles)
                if fetched_map:
                    label_color_map = fetched_map
                else:
                    st.warning(f"No color mapping via API for '{profile_name_for_styles}'. Using fallback colors.")
            else:
                st.info("No KG profile (graph-specific or session) defined. Using fallback colors.")

            generated_node_styles: List[NodeStyle] = []
            current_node_labels = set()
            if "nodes" in transformed_data and transformed_data["nodes"] is not None:
                for node in transformed_data["nodes"]:
                    label = node.get("label")
                    if label:
                        current_node_labels.add(label)

            fallback_color_idx = 0
            for label_text in sorted(list(current_node_labels)):
                color = label_color_map.get(label_text)
                if not color:
                    color = DEFAULT_FALLBACK_COLOR_PALETTE[fallback_color_idx % len(DEFAULT_FALLBACK_COLOR_PALETTE)]
                    fallback_color_idx += 1

                generated_node_styles.append(
                    NodeStyle(label=label_text, color=color, caption=caption_key_for_nodes)
                )

            # Every node gets a resolved caption under this key below, so the
            # fallback style uses it too.
            default_global_style = NodeStyle(label=None, color="#D3D3D3", caption=caption_key_for_nodes)
            final_node_styles = generated_node_styles + [default_global_style]

            edge_styles = [EdgeStyle(None, caption='label', directed=True)]

            if "nodes" in transformed_data and transformed_data["nodes"] is not None:
                for node_data_content in transformed_data["nodes"]:
                    node_data_content[caption_key_for_nodes] = get_node_caption_value(node_data_content, caption_key_for_nodes)
                    if "id" in node_data_content and isinstance(node_data_content.get("id"), int):
                            node_data_content["id"] = str(node_data_content["id"])

            st_elements = {
                "nodes": [{"data": n} for n in transformed_data.get("nodes", [])],
                "edges": [{"data": e} for e in transformed_data.get("edges", [])]
            }
            st_link_analysis(st_elements, "cose", final_node_styles, edge_styles)

        except KeyError as e:
             st.error(f"Missing data (e.g., 'label' or 'name' in graph data): {e}")
             st.json(transformed_data)
        except Exception as e:
            st.warning(f"Unexpected error while displaying the graph: {e}")
            st.json(transformed_data)
    else:
        st.write("No graph to display (no data or no nodes).")


def render_cypher_query_and_link(last_cypher_query: Optional[str], profile_name_for_browser_link: Optional[str]):
    """Render the Cypher query and a link that opens it in Neo4j Browser."""
    if not last_cypher_query:
        return

    st.subheader("🔎 Cypher Query")
    st.code(last_cypher_query, language="cypher")

    # The user's browser runs on the host, where the Neo4j ports are exposed.
    neo4j_host_browser = "localhost"
    http_port_browser = "7474"

    if profile_name_for_browser_link:
        profile_data_browser = get_kg_profile(profile_name_for_browser_link)
        if profile_data_browser:
            http_port_browser = profile_data_browser.get("browser_port", "7474")
        else:
            st.warning(f"Unable to retrieve profile '{profile_name_for_browser_link}' details for Neo4j Browser link. Using defaults.")
    else:
        logger.debug("No KG profile provided for Neo4j Browser link; using defaults.")

    neo4j_browser_base_url = f"http://{neo4j_host_browser}:{http_port_browser}/browser/"
    encoded_query = quote(last_cypher_query)
    browser_link = f"{neo4j_browser_base_url}?cmd=edit&arg={encoded_query}"
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 10px;">
            <a href="{browser_link}" target="_blank" style="text-decoration: none;">
                <button style="padding: 0.6em 1.2em; font-size: 1.1em; cursor: pointer; border-radius: 8px; background-color: #0099ff; color: white; border: none;">
                    🔍 Open Query in Neo4j
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
