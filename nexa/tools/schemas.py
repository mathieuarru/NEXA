"""Pydantic input schemas for the flex tools.

The LLM sees these schemas when calling a tool, so field descriptions say what
to put in each argument. Guidance on when to use a tool belongs in the tool's
own description.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# KG tools
# ---------------------------------------------------------------------------


class ExecuteCypherInput(BaseModel):
    query: str = Field(
        description=(
            "A complete Cypher query to run against the active knowledge graph. "
            "Must be syntactically valid. Read-only queries are expected; write "
            "queries are not supported in flex and may be rejected by the database."
        )
    )


class ReadKgSchemaInput(BaseModel):
    pass


class ValidateCypherInput(BaseModel):
    query: str = Field(
        description=(
            "The Cypher query to validate against the active KG schema. "
            "Lightweight syntactic + label/relationship/property existence "
            "check — the query is NOT executed."
        )
    )


class LookupPropertyInput(BaseModel):
    element_type: Literal["node", "relationship"] = Field(
        description='Whether to search property values on nodes or relationships.'
    )
    label_or_rel: str = Field(
        description=(
            "The node label (e.g. 'Gene', 'Drug') or relationship type (e.g. "
            "'TARGETS') that owns the property. Must match the schema exactly."
        )
    )
    query: str = Field(
        description=(
            "The candidate value you want to match against existing KG values. "
            "Fuzzy matching (case-insensitive, token overlap, substring) is applied."
        )
    )
    property: Optional[str] = Field(
        default=None,
        description=(
            "Property name to match values on (e.g. 'geneSymbol', 'commonName'). "
            "OMIT it to search across ALL properties of the label/rel and "
            "discover which property holds the value (entity-resolution mode)."
        ),
    )
    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="How many top matches to return (1-50, default 10).",
    )


# ---------------------------------------------------------------------------
# Result manipulation tools
# ---------------------------------------------------------------------------


class ListResultsInput(BaseModel):
    pass


class ReadResultInput(BaseModel):
    result_id: str = Field(description="The result_id returned by a previous tool call.")
    limit: int = Field(
        default=20, ge=1, le=200,
        description="Maximum number of rows to return in this page.",
    )
    offset: int = Field(
        default=0, ge=0,
        description="Zero-based row offset for pagination.",
    )


class FilterCondition(BaseModel):
    column: str = Field(
        description=(
            "Column to filter on. Use dot notation to reach inside a "
            "node/relationship cell returned whole (e.g. 'g.geneSymbol' when the "
            "query did `RETURN g`); use the plain name for scalar columns."
        )
    )
    op: Literal["eq", "neq", "gt", "gte", "lt", "lte", "in", "contains", "is_null", "is_not_null"] = Field(
        description=(
            "Comparison operator. 'eq'/'neq' for equality, 'gt'/'gte'/'lt'/'lte' "
            "for numeric, 'in' expects a list value, 'contains' is case-insensitive "
            "substring, 'is_null'/'is_not_null' ignore the value field."
        )
    )
    value: Optional[object] = Field(
        default=None,
        description="Comparison value. Type depends on op. Omit for is_null/is_not_null.",
    )


class FilterResultInput(BaseModel):
    result_id: str = Field(description="The result_id to filter.")
    conditions: list[FilterCondition] = Field(
        description="Conditions combined with logical AND. At least one required.",
        min_length=1,
    )


class SummarizeResultInput(BaseModel):
    result_id: str = Field(description="The result_id to summarize.")
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Number of top values to return per column.",
    )


class AggregateOp(BaseModel):
    column: str = Field(
        description="Column to aggregate. Dot notation reaches into node cells (e.g. 'g.geneSymbol')."
    )
    op: Literal["count", "sum", "avg", "min", "max"] = Field(
        description="Aggregation operator."
    )


class AggregateResultInput(BaseModel):
    result_id: str = Field(description="The result_id to aggregate.")
    group_by: list[str] = Field(
        default_factory=list,
        description=(
            "Columns to group by (dot notation allowed, e.g. 'g.geneSymbol'). "
            "Empty list means global aggregation (one output row)."
        ),
    )
    aggregations: list[AggregateOp] = Field(
        description="One or more aggregations to compute.",
        min_length=1,
    )


# ---------------------------------------------------------------------------
# Reasoning / output tools
# ---------------------------------------------------------------------------


class TodoItem(BaseModel):
    content: str = Field(description="Imperative description of the step (e.g. 'Find drugs targeting APOE').")
    status: Literal["pending", "in_progress", "completed"] = Field(
        default="pending", description="Current status of the step."
    )


class WriteTodosInput(BaseModel):
    todos: list[TodoItem] = Field(
        description=(
            "The full updated todo list. Send the complete list every time — "
            "this tool replaces the previous list, it does not append."
        )
    )


class GraphSpec(BaseModel):
    """Optional rendering hint for the frontend graph viz.

    The actual nodes/edges are NEVER provided by the LLM — they are fetched
    server-side from the cited result_ids. This object only conveys display
    preferences (which result_id to render as a graph, colors, etc.).
    """

    primary_result_id: str = Field(description="result_id to render as an interactive graph.")
    node_label_field: Optional[str] = Field(
        default=None,
        description="Column to use as the node label in the rendering (e.g. 'symbol').",
    )


class RespondToUserInput(BaseModel):
    answer_text: str = Field(
        description=(
            "The narrative answer for the user. May describe and synthesize the "
            "findings, but MUST NOT contain raw tabular data — cite result_ids "
            "instead so the frontend can render the actual data."
        )
    )
    cited_result_ids: list[str] = Field(
        default_factory=list,
        description=(
            "result_ids the user should see as tables/graphs. Pass an empty list "
            "only when the answer is purely conversational (no data backing it)."
        ),
    )
    highlight_rows: Optional[dict[str, list[int]]] = Field(
        default=None,
        description=(
            "Optional map {result_id: [row_indices]} to highlight specific rows "
            "in the frontend display."
        ),
    )
    graph_spec: Optional[GraphSpec] = Field(
        default=None,
        description="Optional graph rendering preferences (see GraphSpec).",
    )
