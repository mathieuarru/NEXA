"""Per-property correction subgraph.

For one property literal of a Cypher query: match the value against the graph,
let the LLM pick the best existing value, and when none fits, ask for a synonym
and retry a bounded number of times. One subgraph runs per property, so
properties are corrected in parallel.
"""

from typing import TypedDict, Optional, Annotated, Literal
from pydantic import BaseModel, Field
from functools import partial

from nexa.config import params
from nexa.models.enums import Status
from nexa.models.prompts import (
    get_properties_suggestion_prompt,
    get_synonym_suggestion_prompt,
)
from nexa.utils.token_utils import with_output_cap

from nexa.knowledge_graph.neo4j_handler import Neo4jHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import START, StateGraph, END
from langgraph.types import Command

from nexa.utils.utils import parse_key, reduce_parsed_properties
from nexa.knowledge_graph.properties_correction import find_top_properties_for_single_key

from nexa.utils.logger import get_logger


logger = get_logger(__name__)


############################
# State Definition
############################

class PropertyProcessingState(TypedDict):
    """Isolated state for each property processing subgraph.
    Enables parallel processing with granular status updates.
    """
    property_key: str
    parsed_properties: Annotated[dict, reduce_parsed_properties]
    status: Status
    thinking: str
    error: Optional[str]
    user_queries: list
    cypher_queries: list
    nodes_descriptions: dict
    relationships_descriptions: dict
    attempts_info: dict
    original_property_value: str
    current_property_value: str


############################
# Pydantic Models
############################

class SuggestedProperty(BaseModel):
    """
    Represents the output when asked to pick one property from a list
    or return None if no property is suitable.
    """
    suggested_property: Optional[str] = Field(
        description=("The property that best match the user's intent. "
                     "These property are chosen to align with the query's purpose and improve its relevance."
                     "or None if none is suitable.")
    )
    thinking: str = Field(
        description=("The reasoning about why it picked this property or why none was suitable.")
    )

    @property
    def full_description(self) -> str:
        """Return a human-readable summary of the suggested property and its reasoning."""
        return f"Suggested Property: {self.suggested_property}\nReasoning: {self.thinking}"


class SynonymSuggestion(BaseModel):
    """
    Represents a brand-new synonym proposed by the LLM
    when no existing property was deemed suitable.
    """
    new_property_value: str = Field(
        description="The new or synonymous value suggested by the LLM."
    )
    thinking: str = Field(
        description="Reasoning behind proposing this new value."
    )

    @property
    def full_description(self) -> str:
        """Return a human-readable summary of the proposed synonym and its reasoning."""
        return f"New Value: {self.new_property_value}\nReasoning: {self.thinking}"


############################
# Subgraph Nodes
############################

def match_property_subgraph(state: PropertyProcessingState, neo4j_handler: Neo4jHandler):
    """Subgraph node: find graph values similar to the current value (original or synonym)."""
    property_key = state["property_key"]
    current_value = state.get("current_property_value", state.get("original_property_value", ""))

    logger.info(f"Subgraph Step 1: Matching property {property_key} with value '{current_value}' in database...")

    temp_property_data = {
        "suggested_values": [
            {current_value: []}
        ]
    }

    parsed_property = {property_key: temp_property_data}

    matched_property = find_top_properties_for_single_key(neo4j_handler, parsed_property)

    property_content = matched_property[property_key]
    suggested_values_list = property_content.get("suggested_values", [])

    if not suggested_values_list or not suggested_values_list[0][current_value]:
        logger.warning(f"No suggested values found for property {property_key} with value '{current_value}'")
        return {
            "parsed_properties": {property_key: {"suggested_values": [{"": [], "needs_synonym": True}]}},
            "status": Status.CHECKING_PROPERTIES,
            "thinking": f"No matches found for {property_key} with value '{current_value}', will need synonym"
        }

    logger.info(f"Found {len(suggested_values_list[0][current_value])} matches for {property_key} with value '{current_value}'")

    return {
        "parsed_properties": {property_key: property_content},
        "status": Status.CHECKING_PROPERTIES,
        "thinking": f"Found {len(suggested_values_list[0][current_value])} potential matches for {property_key} with value '{current_value}'"
    }


def suggest_property_subgraph(state: PropertyProcessingState, llm: BaseChatModel):
    """Subgraph node: let the LLM pick the best value among the matches."""
    logger.info(f"Subgraph Step 2: LLM suggesting best property for {state['property_key']}...")

    property_key = state["property_key"]
    parsed_properties = state["parsed_properties"]
    property_data = parsed_properties.get(property_key, {})
    suggested_values_list = property_data.get("suggested_values", [])

    if suggested_values_list and suggested_values_list[0].get("needs_synonym"):
        logger.info(f"Property {property_key} needs synonym, skipping suggestion step")
        return {
            "status": Status.SUGGESTING_PROPERTIES,
            "thinking": f"Property {property_key} needs synonym, skipping LLM suggestion"
        }

    if not suggested_values_list:
        raise ValueError(f"No suggested values for property {property_key}")

    last_item = suggested_values_list[-1]
    (current_val, top_k_matches) = next(iter(last_item.items()))

    if params.NEXA_ABLATE_LLM_SELECTION:
        # Ablation: the best-ranked database candidate is taken as is.
        chosen = top_k_matches[0] if top_k_matches else None
        last_item["selectionned_value"] = chosen
        return {
            "parsed_properties": {property_key: {"suggested_values": suggested_values_list}},
            "status": Status.SUGGESTING_PROPERTIES,
            "thinking": f"Top-ranked candidate '{chosen}' taken for {property_key} (NEXA_ABLATE_LLM_SELECTION)",
        }

    parsed_key = parse_key(property_key)
    if len(parsed_key) >= 3:
        element_type, label_or_rel, property_name = parsed_key[0], parsed_key[1], parsed_key[2]
    else:
        raise ValueError(f"Invalid property key format: {property_key}")

    # Picking one of ten candidates needs few tokens; the cap stops repetition
    # loops (see NEXA_LLM_MAX_PROPERTY_TOKENS).
    capped_llm = with_output_cap(llm, params.NEXA_LLM_MAX_PROPERTY_TOKENS)
    structured_llm = capped_llm.with_structured_output(SuggestedProperty)

    system_message, user_message = get_properties_suggestion_prompt(element_type, property_name, label_or_rel, current_val, top_k_matches)
    llm_response = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=user_message)])

    if llm_response is None:
        llm_response = SuggestedProperty(
            suggested_property=None,
            thinking="The model did not provide a structured response. Assuming no suitable property was found."
        )

    last_item["selectionned_value"] = llm_response.suggested_property

    updated_property_data = {"suggested_values": suggested_values_list}

    return {
        "parsed_properties": {property_key: updated_property_data},
        "status": Status.SUGGESTING_PROPERTIES,
        "thinking": f"LLM selected '{llm_response.suggested_property}' for {property_key}: {llm_response.thinking}"
    }


def check_completion_subgraph(state: PropertyProcessingState):
    """Router: go to synonym suggestion when no value was selected, else end."""
    property_key = state["property_key"]
    parsed_properties = state["parsed_properties"]
    property_data = parsed_properties.get(property_key, {})
    suggested_values_list = property_data.get("suggested_values", [])

    if not suggested_values_list:
        logger.error(f"No suggested values for property {property_key}")
        return END

    if params.NEXA_ABLATE_LLM_SELECTION:
        # Ablation: no LLM-proposed synonym either; the property keeps
        # whatever the top-ranked candidate gave, possibly nothing.
        return END

    if suggested_values_list[0].get("needs_synonym"):
        logger.info(f"Property {property_key} needs synonym")
        return "suggest_synonym_subgraph"

    last_item = suggested_values_list[-1]
    selected_value = last_item.get("selectionned_value")

    if selected_value is None:
        logger.info(f"Property {property_key} needs synonym (no suitable match found)")
        return "suggest_synonym_subgraph"
    return END


def suggest_synonym_subgraph(state: PropertyProcessingState, llm: BaseChatModel):
    """Subgraph node: ask the LLM for a synonym when no existing value fits."""
    logger.info(f"Subgraph Step 3: LLM suggesting synonym for {state['property_key']}...")

    property_key = state["property_key"]
    parsed_properties = state["parsed_properties"]
    property_data = parsed_properties.get(property_key, {})
    suggested_values_list = property_data.get("suggested_values", [])

    if not suggested_values_list:
        raise ValueError(f"No suggested values for property {property_key}")

    parsed_key = parse_key(property_key)
    if len(parsed_key) >= 3:
        label_or_rel, property_name = parsed_key[1], parsed_key[2]
    else:
        raise ValueError(f"Invalid property key format: {property_key}")

    existing_synonyms = {
        k for suggestion_dict in suggested_values_list
        for k in suggestion_dict
        if k not in ["selectionned_value", "needs_synonym"]
    }

    current_val = state.get("current_property_value", property_name)

    structured_llm = llm.with_structured_output(SynonymSuggestion)
    system_message, user_message = get_synonym_suggestion_prompt(property_name, label_or_rel, current_val, existing_synonyms)
    synonym_response = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=user_message)])

    logger.info(f"Synonym suggested: {synonym_response.new_property_value} for {property_key}")

    return {
        "current_property_value": synonym_response.new_property_value,
        "status": Status.SUGGESTING_SYNONYMS,
        "thinking": f"Synonym '{synonym_response.new_property_value}' suggested for {property_key}: {synonym_response.thinking}"
    }


def check_synonym_attempts(state: PropertyProcessingState) -> Command[Literal["match_property_subgraph", "__end__"]]:
    """Subgraph node: retry matching with the new synonym unless the attempt limit is reached."""
    property_key = state["property_key"]
    attempts_info = state["attempts_info"]
    synonym_attempts = attempts_info["synonym"]

    logger.info(f"Checking synonym attempts for {property_key}: {synonym_attempts['current']}/{synonym_attempts['max']}")

    if synonym_attempts["current"] >= synonym_attempts["max"]:
        logger.warning(f"Max synonym attempts reached for {property_key}")
        return Command(
            goto=END
        )

    synonym_attempts["current"] += 1

    logger.info(f"Retrying synonym matching for {property_key} (attempt {synonym_attempts['current']})")
    return Command(
        goto="match_property_subgraph"
        )


############################
# Subgraph Factory
############################

def create_property_processing_subgraph(neo4j_handler: Neo4jHandler, llm: BaseChatModel):
    """Build the per-property subgraph: match, select, and retry with synonyms."""
    subgraph_builder = StateGraph(PropertyProcessingState)

    subgraph_builder.add_node("match_property_subgraph", partial(match_property_subgraph, neo4j_handler=neo4j_handler))
    subgraph_builder.add_node("suggest_property_subgraph", partial(suggest_property_subgraph, llm=llm))
    subgraph_builder.add_node("suggest_synonym_subgraph", partial(suggest_synonym_subgraph, llm=llm))
    subgraph_builder.add_node("check_synonym_attempts", check_synonym_attempts)

    subgraph_builder.add_edge(START, "match_property_subgraph")
    subgraph_builder.add_edge("match_property_subgraph", "suggest_property_subgraph")
    subgraph_builder.add_conditional_edges("suggest_property_subgraph", check_completion_subgraph, ["suggest_synonym_subgraph", END])
    subgraph_builder.add_edge("suggest_synonym_subgraph", "check_synonym_attempts")

    return subgraph_builder.compile()
