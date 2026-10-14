"""Top-level KG question-answering graph.

``GraphQueryAgent`` chains the generation and correction subgraphs and runs a
question against Neo4j in ``decomposed``, ``one_shot`` or ``one_shot_fallback``
mode.
"""

from nexa.knowledge_graph.neo4j_handler import Neo4jHandler
from nexa.knowledge_graph.read_only import policy_notification

from langchain_core.language_models import BaseChatModel
from typing import TypedDict, Annotated, Union
from langgraph.graph import add_messages
from nexa.models.enums import Status
from nexa.utils.utils import reduce_attempts_info
from langchain_core.messages import HumanMessage

from nexa.knowledge_graph.query_generation import build_generation_graph
from nexa.knowledge_graph.query_correction import build_correction_graph, execute_cypher_query

import time
from functools import partial
from langgraph.graph import START, END, StateGraph

from nexa.utils.logger import get_logger
from nexa.config import params

logger = get_logger(__name__)


class GraphQueryState(TypedDict):
    """Shared state passed through the full generation-correction-execution graph."""
    messages: Annotated[list, add_messages]
    nodes: list
    relationships: list
    rel_properties: list
    user_queries: list
    cypher_queries: list
    thinking: str
    data: Union[list, None]
    error: Union[str, None]
    status: Status
    parsed_properties: dict
    suggested_properties: list
    selected_labels: list
    selected_relationships: list
    attempts_info: Annotated[dict, reduce_attempts_info]  # Format: {"global_": {"current": 0, "max": 3}, "property": {...}, "neo4j_error": {...}}
    nodes_descriptions: dict
    relationships_descriptions: dict
    dynamic_examples: dict
    identifying_properties: dict
    # one_shot_fallback mode only: True when an empty one-shot result sent the
    # question to the decomposed branch. None in the other modes.
    fallback_triggered: Union[bool, None]


GENERATION_MODES = ("decomposed", "one_shot", "one_shot_fallback")


def _is_data_empty(data) -> bool:
    """Return True for None, an empty list, or a dict whose values are all empty.

    Same rule as ``query_correction.check_attempts``.
    """
    if data is None:
        return True
    if isinstance(data, dict):
        return not any(data.values())
    if isinstance(data, list):
        return not data
    return False


# Keyed by connection. Module-level because a GraphQueryAgent is built per question.
_SCHEMA_CACHE: dict = {}


def clear_schema_cache() -> None:
    """Drop every cached schema (for tests, or after reloading a graph)."""
    _SCHEMA_CACHE.clear()


def _get_schema_cached(handler: Neo4jHandler):
    """Return ``(nodes, relationships, rel_properties)``, cached per connection.

    The schema reads are full graph scans (``apoc.meta.data()`` is not used
    because it samples and can miss relationship triples). A cached schema is
    reused while ``schema_fingerprint()`` is unchanged. ``NEXA_SCHEMA_CACHE_TTL``
    also forces a re-read, for changes the fingerprint cannot see, such as a new
    property key on existing nodes.
    """
    ttl = params.NEXA_SCHEMA_CACHE_TTL
    key = (getattr(handler, "db_uri", None) or id(handler), handler.db_name)
    now = time.monotonic()
    cached = _SCHEMA_CACHE.get(key)

    if cached:
        stamp, schema, fingerprint = cached
        fresh_enough = ttl <= 0 or (now - stamp) < ttl
        if fresh_enough:
            try:
                current = handler.schema_fingerprint()
            except Exception as e:  # never fail a question over a cache check
                logger.warning(f"Schema fingerprint check failed ({e}); reusing cached schema.")
                return schema
            if current is not None and current == fingerprint:
                logger.debug(f"Schema cache hit for {key}")
                return schema
            logger.info(f"Schema fingerprint changed for {key}; re-reading the schema.")

    schema = (
        handler.get_nodes(),
        handler.get_relationships(),
        handler.get_relationships_properties(),
    )
    try:
        fingerprint = handler.schema_fingerprint()
    except Exception as e:
        logger.warning(f"Could not compute schema fingerprint ({e}); caching without one.")
        fingerprint = None
    _SCHEMA_CACHE[key] = (now, schema, fingerprint)
    logger.info(
        f"Schema read and cached for {key} "
        f"({len(schema[0])} labels, {len(schema[1])} relationship triples)"
    )
    return schema


def _execute_one_shot(state, driver: Neo4jHandler):
    """Execute the one-shot query and record in ``fallback_triggered`` whether it came back empty."""
    update = execute_cypher_query(state, driver)
    update["fallback_triggered"] = (
        not policy_notification(update.get("notifications")) and _is_data_empty(update.get("data"))
    )
    return update


def _route_after_one_shot(state) -> str:
    """Route to the decomposed branch when the one-shot query returned no data.

    Recomputes the decision when ``fallback_triggered`` is missing, so a missing
    flag never disables the fallback.
    """
    if policy_notification(state.get("notifications")):
        return "done"
    triggered = state.get("fallback_triggered")
    if triggered is None:
        triggered = _is_data_empty(state.get("data"))
    return "fallback" if triggered else "done"


class GraphQueryAgent:
    """Run user questions through the KG pipeline against a Neo4j database.

    Empty connection arguments fall back to the values in ``params``.
    """
    def __init__(self,
                 llm: BaseChatModel,
                 rephrasing=True,
                 neo4j_host: str = None,
                 neo4j_port: str = None,
                 neo4j_username: str = None,
                 neo4j_password: str = None,
                 neo4j_db_name: str = "neo4j",
                 nodes_descriptions: dict = None,
                 relationships_descriptions: dict = None,
                 dynamic_examples: dict = None,
                 identifying_properties: dict = None,
                 generation_mode: str = "one_shot_fallback"
                 ):
        self.llm = llm

        actual_host = neo4j_host if neo4j_host and neo4j_host.strip() else params.NEO4J_HOST
        actual_port = neo4j_port if neo4j_port and neo4j_port.strip() else params.NEO4J_PORT
        actual_username = neo4j_username if neo4j_username and neo4j_username.strip() else params.NEO4J_USERNAME
        actual_password = neo4j_password if neo4j_password and neo4j_password.strip() else params.NEO4J_PASSWORD
        actual_db_name = neo4j_db_name if neo4j_db_name and neo4j_db_name.strip() else params.NEO4J_DB_NAME

        logger.info(f"GraphQueryAgent Init: Received - host='{neo4j_host}', port='{neo4j_port}', user='{neo4j_username}', db_name='{neo4j_db_name}'")
        logger.info(f"GraphQueryAgent Init: Effective for Neo4jHandler - host='{actual_host}', port='{actual_port}', user='{actual_username}', db_name='{actual_db_name}'")

        self.neo4j_handler = Neo4jHandler(
            host=actual_host,
            port=actual_port,
            username=actual_username,
            password=actual_password,
            db_name=actual_db_name
        )
        self.rephrasing = rephrasing
        self.nodes_descriptions = nodes_descriptions
        self.relationships_descriptions = relationships_descriptions
        self.dynamic_examples = dynamic_examples or {}
        self.identifying_properties = identifying_properties or {}

        if generation_mode not in GENERATION_MODES:
            raise ValueError(f"Unknown generation_mode={generation_mode!r}; expected one of {GENERATION_MODES}")
        self.generation_mode = generation_mode

        self.correction_graph = build_correction_graph(self.llm, self.neo4j_handler)
        self.entire_graph = self._build_entire_graph()


    def _build_entire_graph(self):
        """Build the top-level graph for the configured mode; the fallback is a conditional edge."""
        builder = StateGraph(GraphQueryState)
        builder.add_node("correction", self.correction_graph)

        if self.generation_mode in ("decomposed", "one_shot"):
            generation_subgraph = build_generation_graph(
                self.llm, self.rephrasing, mode=self.generation_mode
            )
            builder.add_node("generation", generation_subgraph)
            builder.add_edge(START, "generation")
            builder.add_edge("generation", "correction")
            builder.add_edge("correction", END)

        elif self.generation_mode == "one_shot_fallback":
            one_shot_subgraph = build_generation_graph(
                self.llm, self.rephrasing, mode="one_shot"
            )
            decomposed_subgraph = build_generation_graph(
                self.llm, self.rephrasing, mode="decomposed"
            )
            builder.add_node("one_shot_generation", one_shot_subgraph)
            builder.add_node(
                "execute_one_shot",
                partial(_execute_one_shot, driver=self.neo4j_handler),
            )
            builder.add_node("decomposed_generation", decomposed_subgraph)

            builder.add_edge(START, "one_shot_generation")
            builder.add_edge("one_shot_generation", "execute_one_shot")
            builder.add_conditional_edges(
                "execute_one_shot",
                _route_after_one_shot,
                {
                    "done": END,
                    "fallback": "decomposed_generation",
                },
            )
            builder.add_edge("decomposed_generation", "correction")
            builder.add_edge("correction", END)

        return builder.compile()


    def _build_initial_state(self, user_query: str, nodes, relationships, rel_properties) -> GraphQueryState:
        return GraphQueryState(
            messages=[HumanMessage(content=user_query)],
            user_queries=[user_query],
            nodes=nodes,
            relationships=relationships,
            rel_properties=rel_properties,
            attempts_info={
                "global_":     {"current": 0, "max": 3},
                "property":    {"current": 0, "max": 3},
                "neo4j_error": {"current": 0, "max": 3},
            },
            data=None,
            status=Status.INIT,
            nodes_descriptions=self.nodes_descriptions,
            relationships_descriptions=self.relationships_descriptions,
            dynamic_examples=self.dynamic_examples,
            identifying_properties=self.identifying_properties,
        )


    def run_entire_graph(self, user_query: str, stop_check=None):
        """Run the full pipeline (generation, correction, execution) and return the final state.

        If ``stop_check`` is given, it is called after every step, subgraph steps
        included; any exception it raises stops the run and propagates.
        """
        logger.info(f"Running entire graph for user query: {user_query} (generation_mode={self.generation_mode})...")

        nodes, relationships, rel_properties = _get_schema_cached(self.neo4j_handler)

        state = self._build_initial_state(user_query, nodes, relationships, rel_properties)

        if stop_check is None:
            result_state = self.entire_graph.invoke(state, config={"recursion_limit": 200})
        else:
            # Stream subgraph steps too, so stop_check runs between every node.
            result_state = None
            for _ns, update in self.entire_graph.stream(
                state,
                config={"recursion_limit": 200},
                stream_mode="values",
                subgraphs=True,
            ):
                # With stream_mode="values", the last snapshot is the final state.
                result_state = update
                stop_check()
        logger.debug(f"Entire graph result state: {result_state}")
        return result_state
