"""Query-correction subgraph.

Executes the generated Cypher query and, while it returns no data or raises
Neo4j errors, repairs it: property literals are matched against the graph (one
subgraph per property) and Neo4j errors are fixed by the LLM, within bounded
attempt limits.
"""

from typing import TypedDict, Union, Literal, Optional
from pydantic import BaseModel, Field

from nexa.config import params
from nexa.models.enums import Status
from nexa.models.prompts import (
    get_query_correction_prompt,
    get_neo4j_error_correction_prompt,
)

from typing import Annotated
from nexa.knowledge_graph.neo4j_handler import Neo4jHandler
from nexa.knowledge_graph.read_only import policy_notification
from langchain_core.language_models import BaseChatModel

from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import START, StateGraph, END
from langgraph.types import Send, Command

from functools import partial

from nexa.utils.utils import reduce_list, reduce_status, reduce_parsed_properties, reduce_thinking, reduce_error, reduce_dict, reduce_attempts_info
from nexa.knowledge_graph.properties_correction import parse_cypher_query
from nexa.utils.cypher_utils import ensure_entities_in_return
from nexa.knowledge_graph.property_processing_subgraph import create_property_processing_subgraph

from nexa.utils.logger import get_logger

logger = get_logger(__name__)

############################
# State Definitions
############################

class CorrectionState(TypedDict):
    """Shared state threaded through the query-correction subgraph."""
    messages: Annotated[list, add_messages]
    thinking: Annotated[str, reduce_thinking]
    user_queries: Annotated[list, reduce_list]
    cypher_queries: Annotated[list, reduce_list]
    nodes: list
    relationships: list
    rel_properties: list
    parsed_properties: Annotated[dict, reduce_parsed_properties]
    parsed_property_match: Optional[dict]
    parsed_property: Optional[dict]
    reason: Optional[str]
    data: Union[list, None]
    notifications: Union[str, None]
    error: Annotated[str, reduce_error]
    status: Annotated[Status, reduce_status]
    attempts_info: Annotated[dict, reduce_attempts_info]  # Format: {"global_": {"current": 0, "max": 3}, "property": {...}, "neo4j_error": {...}}
    nodes_descriptions: Annotated[dict, reduce_dict]
    relationships_descriptions: Annotated[dict, reduce_dict]
    dynamic_examples: dict
    identifying_properties: dict


############################
# Pydantic Models
############################

class CorrectedQuery(BaseModel):
    """Represents a Cypher query that has been corrected to match the user's intent and the graph database schema."""
    query: str = Field(
        description=("The final Cypher query after corrections. "
                    "This query should retrieve the most relevant information from the graph database based on the user's initial intent. ")
    )
    thinking: str = Field(
        description=("A detailed, step-by-step explanation of the corrections made to the query. "
                    "This includes the reasoning behind the changes, how the adjustments align with the user's intent, "
                    "and any modifications required to comply with the graph schema.")
    )

    @property
    def full_description(self) -> str:
        """Return a human-readable summary of the corrected query and its reasoning."""
        return f"Corrected Query: {self.query}\nCorrection Reasoning: {self.thinking}"


class Neo4jErrorCorrectedQuery(BaseModel):
    """Represents a Cypher query that has been corrected to address specific Neo4j errors or notifications."""
    query: str = Field(
        description="The corrected Cypher query that addresses the Neo4j errors or notifications."
    )
    thinking: str = Field(
        description="A detailed explanation of what errors were identified and how they were fixed in the query."
    )

    @property
    def full_description(self) -> str:
        """Return a human-readable summary of the Neo4j-error-corrected query and its reasoning."""
        return f"Neo4j Error Corrected Query: {self.query}\nError Correction Reasoning: {self.thinking}"


############################
# Main Workflow Nodes
############################

def execute_cypher_query(state: CorrectionState, driver: Neo4jHandler):
    """Execute the latest Cypher query and store its data and notifications."""
    logger.info("Executing Cypher query on the Neo4j database...")

    cypher_query = state.get("cypher_queries")[-1]
    elements, notifications = driver.execute_query_as_elements(cypher_query)
    logger.debug(f"Elements returned by the graph: {elements}")

    violation = policy_notification(notifications)
    if violation:
        return {"data": None, "thinking": cypher_query, "notifications": notifications,
                "error": f"Read-only policy: {violation['description']}", "status": Status.ERROR}

    return {'data' : elements, 'thinking': cypher_query, 'notifications': notifications, "status": Status.EXECUTING}


def check_attempts(state: CorrectionState) -> Command[Literal["__end__", "extract_properties", "correct_neo4j_error"]]:
    """Route after execution.

    A policy violation ends with an error. An empty result once the global
    attempt limit is reached is accepted as the answer. Neo4j notifications go
    to error correction (up to its own limit), an empty result goes to property
    correction, and any data ends the loop.
    """
    data = state.get("data")
    notifications = state.get("notifications")
    violation = policy_notification(notifications)
    if violation:
        return Command(
            update={"error": f"Read-only policy: {violation['description']}", "status": Status.ERROR},
            goto=END, graph=Command.PARENT,
        )
    attempts_info = state["attempts_info"]
    global_attempts = attempts_info["global_"]
    neo4j_error_attempts = attempts_info["neo4j_error"]

    logger.info(f"Global attempts: {global_attempts['current']}/{global_attempts['max']}")
    logger.info(f"Neo4j error correction attempts: {neo4j_error_attempts['current']}/{neo4j_error_attempts['max']}")


    if data is None:
        no_data_available = True
    elif isinstance(data, dict):
        no_data_available = not any(data.values())
    elif isinstance(data, list):
        no_data_available = not data
    else:
        no_data_available = False

    has_neo4j_notifications = notifications is not None and len(notifications) > 0
    # Ablation: without the dedicated repair step, a failing query is handled
    # like an empty one (the rewrite loop below, unless that is ablated too).
    if params.NEXA_ABLATE_SYNTAX_CORRECTION:
        has_neo4j_notifications = False

    if no_data_available and global_attempts["current"] >= global_attempts["max"]:
        # Not an error: an empty result after every correction attempt is read
        # downstream as absence of data.
        logger.info(
            f"Max correction attempts ({global_attempts['max']}) reached with empty results; "
            "treating this as absence of data, not as a system error."
        )
        return Command(
            update={"status": Status.DONE},
            goto=END,
            graph=Command.PARENT
        )

    if has_neo4j_notifications:
        logger.info(f"Neo4j notifications detected: {notifications[0] if isinstance(notifications, list) and notifications else notifications}...")
        if neo4j_error_attempts["current"] >= neo4j_error_attempts["max"]:
            logger.error("Max Neo4j error correction attempts reached.")
            return Command(
                update={"error": "Max Neo4j error correction attempts reached.", "status": Status.ERROR},
                goto=END,
                graph=Command.PARENT
            )
        neo4j_error_attempts["current"] += 1
        return Command(
            goto="correct_neo4j_error"
        )

    if no_data_available and params.NEXA_ABLATE_EMPTY_RESULT_LOOP:
        logger.info("Empty result accepted as final (NEXA_ABLATE_EMPTY_RESULT_LOOP).")
        return Command(
            update={"status": Status.DONE},
            goto=END,
            graph=Command.PARENT
        )

    if no_data_available:
        logger.info("Continue to correction because no data available.")
        global_attempts["current"] += 1
        return Command(
            goto="extract_properties"
        )

    logger.info("Data received successfully.")
    return Command(
            update={"status": Status.DONE},
            goto=END,
    )


def extract_properties(state: CorrectionState):
    """Extract the property literals of the latest Cypher query."""
    logger.info("Extracting properties from the Cypher query...")
    cypher_query = state.get("cypher_queries")[-1]
    parsed_properties = parse_cypher_query(cypher_query)
    return {
        "parsed_properties": {
            "reset": True,
            "value": parsed_properties
        },
        "status": Status.EXTRACTING_PROPERTIES
    }


def start_property_subgraphs(state: CorrectionState):
    """Router: send each extracted property to its own subgraph, run in parallel."""

    suggested_properties = state.get("parsed_properties", {})

    if not suggested_properties:
        logger.warning("No properties found to process")
        return "collect_all_properties"

    if params.NEXA_ABLATE_DB_MATCHING:
        # Ablation: no candidate values from the database. The rewrite step
        # then receives no suggested replacement for any literal.
        logger.info("Skipping database value matching (NEXA_ABLATE_DB_MATCHING).")
        return "collect_all_properties"

    sends = []
    for key, value in suggested_properties.items():
        original_value = ""
        if value.get("suggested_values"):
            first_suggestion = value["suggested_values"][0]
            if first_suggestion:
                original_value = next(iter(first_suggestion.keys()))

        subgraph_initial_state = {
            "property_key": key,
            "parsed_properties": {key: value},
            "status": Status.INIT,
            "thinking": "",
            "error": None,
            "user_queries": state.get("user_queries", []),
            "cypher_queries": state.get("cypher_queries", []),
            "nodes_descriptions": state.get("nodes_descriptions", {}),
            "relationships_descriptions": state.get("relationships_descriptions", {}),
            "attempts_info": {
                "property": {"current": 0, "max": 3},
                "synonym": {"current": 0, "max": 3}
            },
            "original_property_value": original_value,
            "current_property_value": original_value
        }

        sends.append(Send("property_processing_subgraph", subgraph_initial_state))

    return sends


def collect_all_properties(state: CorrectionState):
    """Keep only the properties for which a replacement value was selected.

    The subgraph results are already merged into ``parsed_properties`` by its reducer.
    """
    logger.info("Collecting results from all property processing subgraphs...")

    parsed_properties = state.get("parsed_properties", {})

    if not parsed_properties:
        logger.warning("No parsed properties found after subgraph processing")
        return {
            "status": Status.SUGGESTING_PROPERTIES,
            "thinking": "No properties were processed"
        }

    all_completed = True
    incomplete_properties = []
    completed_properties = {}

    for prop_key, prop_data in parsed_properties.items():
        suggested_values = prop_data.get("suggested_values", [])

        if not suggested_values:
            all_completed = False
            incomplete_properties.append(prop_key)
            continue

        last_suggestion = suggested_values[-1]
        if ("selectionned_value" in last_suggestion and
            last_suggestion["selectionned_value"] is not None):
            completed_properties[prop_key] = prop_data
            logger.info(f"Property {prop_key} completed with selection: {last_suggestion['selectionned_value']}")
        else:
            all_completed = False
            incomplete_properties.append(prop_key)

    if all_completed:
        logger.info("All properties have been processed successfully by subgraphs")
        return {
            "parsed_properties": {
                "reset": True,
                "value": completed_properties
            },
            "status": Status.SUGGESTING_PROPERTIES,
            "thinking": f"Successfully processed {len(completed_properties)} properties via subgraphs"
        }
    else:
        logger.warning(f"Some properties incomplete after subgraph processing: {incomplete_properties}")
        return {
            "parsed_properties": {
                "reset": True,
                "value": completed_properties
            },
            "status": Status.SUGGESTING_PROPERTIES,
            "thinking": f"Processed {len(completed_properties)} properties, {len(incomplete_properties)} incomplete"
        }


def correct_cypher_query(state: CorrectionState, llm: BaseChatModel):
    """Ask the LLM to rewrite the Cypher query using the suggested property values."""

    logger.info("Correcting the Cypher query...")

    user_query = state.get("user_queries")[-1]
    cypher_query = state.get("cypher_queries")[-1]
    nodes = state.get("nodes")
    relationships = state.get("relationships")
    notifications = state.get("notifications")
    parsed_properties = state.get("parsed_properties", {})
    nodes_descriptions = state.get("nodes_descriptions")
    relationships_descriptions = state.get("relationships_descriptions")

    structured_llm = llm.with_structured_output(CorrectedQuery)

    dynamic_examples = state.get("dynamic_examples") or {}
    identifying_properties = state.get("identifying_properties") or {}
    system_message, user_message = get_query_correction_prompt(question=user_query, nodes=nodes, relationships=relationships, notifications=notifications, cypher_query=cypher_query, parsed_properties=parsed_properties, nodes_descriptions_dict=nodes_descriptions, relationships_descriptions_dict=relationships_descriptions, examples=dynamic_examples.get("query_correction"), identifying_properties=identifying_properties)

    llm_response = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=user_message)])
    logger.debug(f"Corrected Cypher Query: {llm_response.full_description}")

    return {"cypher_queries": [llm_response.query], "thinking": llm_response.thinking, "status": Status.CORRECTING_CYPHER}


def correct_neo4j_error(state: CorrectionState, llm: BaseChatModel):
    """Ask the LLM to fix the Neo4j errors or warnings raised by the Cypher query.

    Only syntax and schema issues are addressed, not property values.
    """
    logger.info("Correcting Neo4j errors or notifications in the Cypher query...")

    cypher_query = state.get("cypher_queries")[-1]
    nodes = state.get("nodes")
    relationships = state.get("relationships")
    notifications = state.get("notifications", "")
    nodes_descriptions = state.get("nodes_descriptions")
    relationships_descriptions = state.get("relationships_descriptions")

    structured_llm = llm.with_structured_output(Neo4jErrorCorrectedQuery)

    system_message, user_message = get_neo4j_error_correction_prompt(
        nodes=nodes,
        relationships=relationships,
        notifications=notifications,
        cypher_query=cypher_query,
        nodes_descriptions_dict=nodes_descriptions,
        relationships_descriptions_dict=relationships_descriptions
    )
    llm_response = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=user_message)])
    logger.debug(f"Neo4j Error Corrected Query: {llm_response.query}")

    return {"cypher_queries": [llm_response.query], "thinking": llm_response.thinking, "status": Status.CORRECTING_NEO4J_ERRORS}


############################
# Graph Builder
############################

def build_correction_graph(llm: BaseChatModel, neo4j_handler: Neo4jHandler):
    """Build the correction graph, which loops until data is found or attempt limits are hit."""
    builder = StateGraph(CorrectionState)

    builder.add_node("execute_cypher_query", partial(execute_cypher_query, driver=neo4j_handler))
    builder.add_node("check_attempts", check_attempts)
    builder.add_node("extract_properties", extract_properties)

    property_subgraph = create_property_processing_subgraph(neo4j_handler, llm)
    builder.add_node("property_processing_subgraph", property_subgraph)

    builder.add_node("collect_all_properties", collect_all_properties)

    builder.add_node("correct_cypher_query", partial(correct_cypher_query, llm=llm))
    builder.add_node("correct_neo4j_error", partial(correct_neo4j_error, llm=llm))
    builder.add_node("ensure_entities_in_return", ensure_entities_in_return)

    builder.add_edge(START, "execute_cypher_query")
    builder.add_edge("execute_cypher_query", "check_attempts")

    builder.add_conditional_edges("extract_properties", start_property_subgraphs, ["property_processing_subgraph", "collect_all_properties"])
    builder.add_edge("property_processing_subgraph", "collect_all_properties")
    builder.add_edge("collect_all_properties", "correct_cypher_query")

    builder.add_edge("correct_cypher_query", "ensure_entities_in_return")
    builder.add_edge("correct_neo4j_error", "execute_cypher_query")
    builder.add_edge("ensure_entities_in_return", "execute_cypher_query")

    return builder.compile()
