"""Query-generation subgraph: natural-language question to Cypher.

Two modes: ``decomposed`` (optional rephrasing, then label selection,
relationship selection and generation) and ``one_shot`` (a single LLM call with
the full schema). Both end by ensuring the queried entities are returned.
"""

from typing import List, TypedDict, Union, Annotated
from pydantic import BaseModel, Field

from nexa.models.enums import Status
from nexa.models.prompts import (
    get_rephrase_user_query_prompt,
    get_labels_selection_prompt,
    get_relationships_selection_prompt,
    get_query_generation_prompt
)
from nexa.utils.neo4j_utils import (
    filter_nodes_by_labels,
    filter_relationships_by_nodes,
    filter_relationships_by_type,
    filter_relationships_props_by_type
)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from langgraph.graph import START, END, StateGraph

from functools import partial

from nexa.utils.utils import reduce_list
from nexa.utils.cypher_utils import ensure_entities_in_return

from nexa.utils.logger import get_logger

logger = get_logger(__name__)


class GenerationState(TypedDict):
    """Shared state threaded through the query-generation subgraph."""
    user_queries: Annotated[list, reduce_list]
    rephrased_query: str
    cypher_queries: Annotated[list, reduce_list]
    thinking: str
    nodes: list
    relationships: list
    rel_properties: list
    selected_labels: list
    selected_relationships: list
    error: Union[str, None]
    status: Status
    nodes_descriptions: dict
    relationships_descriptions: dict
    dynamic_examples: dict
    identifying_properties: dict


class RephrasedQuery(BaseModel):
    """Represents the rephrased user query to enhance clarity and relevance."""

    rephrased_query: str = Field(
        description="The reformulated version of the user's query for improved clarity and specificity."
    )
    thinking: str = Field(
        description="Detailed reasoning or steps followed to reformulate the query, explaining how it enhances clarity or relevance."
    )

    @property
    def full_description(self) -> str:
        """Return a human-readable summary of the rephrased query and its reasoning."""
        return f"Rephrased Query: {self.rephrased_query}\nReasoning: {self.thinking}\n"


def rephrase_user_query(llm: BaseChatModel, state: GenerationState):
    """Rephrase the latest user query for clarity, using the graph schema."""

    logger.info("Rephrasing User Query...")

    user_query = state.get("user_queries")[-1]
    nodes = state.get("nodes")
    relationships = state.get("relationships")
    nodes_descriptions = state.get("nodes_descriptions")
    relationships_descriptions = state.get("relationships_descriptions")

    structured_llm = llm.with_structured_output(RephrasedQuery)

    system_message, user_message = get_rephrase_user_query_prompt(user_query, nodes, relationships, nodes_descriptions, relationships_descriptions)

    llm_response = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=user_message)])

    return {"user_queries": [llm_response.rephrased_query], "thinking": llm_response.thinking, "status": Status.REPHRASING}


class SelectedLabels(BaseModel):
    """Represents the labels of nodes that are most relevant to the user query."""

    selected_labels: List[str] = Field(
        description="A list of node labels deemed most relevant to the user's query."
    )
    thinking: str = Field(
        description="A detailed explanation of the thought process or steps taken to select these labels."
    )

    @property
    def full_description(self) -> str:
        """Return a human-readable summary of the selected node labels and reasoning."""
        return f"Selected Labels: {self.selected_labels}\nThinking: {self.thinking}\n"

def select_nodes_labels(llm: BaseChatModel, state: GenerationState):
    """Select the node labels relevant to the latest user query."""

    logger.info("Selecting Nodes Labels...")

    user_query = state.get("user_queries")[-1]
    nodes = state.get("nodes")
    nodes_descriptions = state.get("nodes_descriptions")

    structured_llm = llm.with_structured_output(SelectedLabels, include_raw=False)

    dynamic_examples = state.get("dynamic_examples") or {}
    system_message, user_message = get_labels_selection_prompt(
        user_query, nodes, nodes_descriptions,
        examples=dynamic_examples.get("labels_selection"),
    )
    llm_response = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=user_message)])
    return {"selected_labels": llm_response.selected_labels, "thinking": llm_response.thinking, "status": Status.NODES_SELECTION}


class SelectedRelationships(BaseModel):
    """Represents the relationships most relevant to the user query."""

    selected_relationships: List[str] = Field(
        description="A list of relationships' names deemed most relevant to the user's query."
    )
    thinking: str = Field(
        description="A detailed explanation of the thought process or steps taken to select these relationships."
    )

    @property
    def full_description(self) -> str:
        """Return a human-readable summary of the selected relationships and reasoning."""
        return f"Selected Relationships: {self.selected_relationships}\nReasoning: {self.thinking}\n"

def select_relationships(llm: BaseChatModel, state: GenerationState):
    """Select the relationships relevant to the query among those linking the selected labels."""

    logger.info("Selecting Relationships...")

    user_query = state.get("user_queries")[-1]
    selected_nodes = state.get("selected_labels")
    relationships = state.get("relationships")
    nodes_descriptions = state.get("nodes_descriptions")
    relationships_descriptions = state.get("relationships_descriptions")

    filtered_relationships = filter_relationships_by_nodes(relationships, selected_nodes)
    structured_llm = llm.with_structured_output(SelectedRelationships)

    dynamic_examples = state.get("dynamic_examples") or {}
    system_message, user_message = get_relationships_selection_prompt(user_query=user_query,
                                                                      filtered_relationships=filtered_relationships,
                                                                      selected_nodes=selected_nodes,
                                                                      nodes_descriptions_dict=nodes_descriptions,
                                                                      relationships_descriptions_dict=relationships_descriptions,
                                                                      examples=dynamic_examples.get("relationships_selection"))


    llm_response = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=user_message)])
    return {"selected_relationships": llm_response.selected_relationships, "thinking": llm_response.thinking, "status": Status.RELATIONSHIPS_SELECTION}


class GeneratedQuery(BaseModel):
    """Represents the generated Cypher query based on user input."""

    query: str = Field(
        description="The generated Cypher query to be executed on the graph database."
    )
    thinking: str = Field(
        description="Detailed reasoning or steps followed to arrive at the generated query."
    )

    @property
    def full_description(self) -> str:
        """Return a human-readable summary of the generated query and its reasoning."""
        return f"Query: {self.query}\nThinking: {self.thinking}\n"


def generate_cypher_query(llm: BaseChatModel, state: GenerationState) -> dict:
    """Generate a Cypher query from the selected labels and relationships."""

    logger.info("Generating Cypher Query...")

    user_query = state.get("user_queries")[-1]
    nodes = state.get("nodes")
    relationships = state.get("relationships")
    relationships_properties = state.get("rel_properties")
    nodes_selected = filter_nodes_by_labels(nodes, state.get("selected_labels"))
    relationships_selected = filter_relationships_by_type(relationships, state.get("selected_relationships"))
    rel_properties_selected = filter_relationships_props_by_type(relationships_properties, state.get("selected_relationships"))
    nodes_descriptions = state.get("nodes_descriptions")
    relationships_descriptions = state.get("relationships_descriptions")

    structured_llm = llm.with_structured_output(GeneratedQuery, include_raw=False)

    dynamic_examples = state.get("dynamic_examples") or {}
    identifying_properties = state.get("identifying_properties") or {}
    system_message, user_message = get_query_generation_prompt(user_query=user_query,
                                                               nodes=nodes_selected,
                                                               relationships=relationships_selected,
                                                               rel_properties=rel_properties_selected,
                                                               nodes_descriptions_dict=nodes_descriptions,
                                                               relationships_descriptions_dict=relationships_descriptions,
                                                               examples=dynamic_examples.get("query_generation"),
                                                               identifying_properties=identifying_properties)

    llm_response = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=user_message)])
    logger.info(f"Generated Cypher Query: {llm_response.query}")
    return {"cypher_queries": [llm_response.query], "thinking": llm_response.thinking, "status": Status.CYPHER_GENERATION}


def generate_cypher_query_one_shot(llm: BaseChatModel, state: GenerationState) -> dict:
    """Generate a Cypher query in a single LLM call, exposing the full schema.

    Skips label and relationship selection, like the GraphCypherQAChain baseline.
    """
    logger.info("Generating Cypher Query (one-shot, full schema)...")

    user_query = state.get("user_queries")[-1]
    nodes = state.get("nodes")
    relationships = state.get("relationships")
    relationships_properties = state.get("rel_properties")
    nodes_descriptions = state.get("nodes_descriptions")
    relationships_descriptions = state.get("relationships_descriptions")

    structured_llm = llm.with_structured_output(GeneratedQuery, include_raw=False)

    dynamic_examples = state.get("dynamic_examples") or {}
    identifying_properties = state.get("identifying_properties") or {}
    system_message, user_message = get_query_generation_prompt(
        user_query=user_query,
        nodes=nodes,
        relationships=relationships,
        rel_properties=relationships_properties,
        nodes_descriptions_dict=nodes_descriptions,
        relationships_descriptions_dict=relationships_descriptions,
        examples=dynamic_examples.get("query_generation"),
        identifying_properties=identifying_properties,
    )

    llm_response = structured_llm.invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content=user_message)]
    )
    logger.info(f"Generated Cypher Query (one-shot): {llm_response.query}")

    # Report the full schema as the selection, so later steps see every pattern.
    all_labels = [
        (n['node']['labels'][0] if isinstance(n, dict) and 'node' in n and n['node'].get('labels') else str(n))
        for n in (nodes or [])
    ]
    all_rel_types = list({r['relationship'] for r in (relationships or []) if isinstance(r, dict) and 'relationship' in r})

    return {
        "cypher_queries": [llm_response.query],
        "thinking": llm_response.thinking,
        "status": Status.CYPHER_GENERATION,
        "selected_labels": all_labels,
        "selected_relationships": all_rel_types,
    }


def build_generation_graph(llm: BaseChatModel, rephrasing: bool, mode: str = "decomposed") -> StateGraph:
    """Build the generation subgraph.

    Modes:
      - "decomposed": [rephrase] -> nodes_selection -> relationships_selection
        -> cypher_query_generation -> ensure_entities_in_return
      - "one_shot": [rephrase] -> cypher_query_generation_one_shot -> ensure_entities_in_return
    """
    if mode not in ("decomposed", "one_shot"):
        raise ValueError(f"Unknown generation mode: {mode!r}")

    builder = StateGraph(GenerationState)

    if rephrasing:
        builder.add_node("user_query_reformulation", partial(rephrase_user_query, llm))

    if mode == "decomposed":
        builder.add_node("nodes_selection", partial(select_nodes_labels, llm))
        builder.add_node("relationships_selection", partial(select_relationships, llm))
        builder.add_node("cypher_query_generation", partial(generate_cypher_query, llm))
        builder.add_edge("nodes_selection", "relationships_selection")
        builder.add_edge("relationships_selection", "cypher_query_generation")
        first_gen_node, last_gen_node = "nodes_selection", "cypher_query_generation"
    else:
        builder.add_node("cypher_query_generation_one_shot", partial(generate_cypher_query_one_shot, llm))
        first_gen_node = last_gen_node = "cypher_query_generation_one_shot"

    builder.add_node("ensure_entities_in_return", ensure_entities_in_return)

    if rephrasing:
        builder.add_edge(START, "user_query_reformulation")
        builder.add_edge("user_query_reformulation", first_gen_node)
    else:
        builder.add_edge(START, first_gen_node)

    builder.add_edge(last_gen_node, "ensure_entities_in_return")
    builder.add_edge("ensure_entities_in_return", END)

    return builder.compile()
