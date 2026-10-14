"""Direct-mode fallback when the LLM runs out of output tokens mid-list.

The entity names are recovered from the KG result and returned in the shape of
the caller's Pydantic schema; a schema that is not an entity list is declined.
"""
import json

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from nexa.core.nexa_agent import NexaAgent
from nexa.models.structured_answers import format_entity_answer


class ListAnswer(BaseModel):
    answer: list[str]


class TextAnswer(BaseModel):
    answer: str
    thinking: str


class BooleanAnswer(BaseModel):
    answer: bool


# ---------------------------------------------------------------------------
# format_entity_answer: the shape comes from the schema
# ---------------------------------------------------------------------------

def test_list_schema_keeps_names_whole():
    """A list[str] schema keeps comma-bearing names whole."""
    names = ["A, B", "C"]
    assert format_entity_answer(ListAnswer, names, "fallback").answer == names


def test_str_schema_keeps_the_csv_contract():
    result = format_entity_answer(TextAnswer, ["APOE", "APP"], "fallback")
    assert result.answer == "APOE,APP"
    assert result.thinking == "fallback"


def test_thinking_is_filled_only_when_the_schema_declares_it():
    assert not hasattr(format_entity_answer(ListAnswer, ["APOE"], "why"), "thinking")


def test_schemas_that_cannot_be_satisfied_are_declined():
    """Schemas the names cannot satisfy as-is get None, not a coerced answer."""

    class Constrained(BaseModel):
        answer: list[str] = Field(max_length=1)

    class ExtraRequired(BaseModel):
        answer: list[str]
        evidence: str

    names = ["APOE", "APP"]
    for schema in (Constrained, ExtraRequired, BooleanAnswer):
        assert format_entity_answer(schema, names, "fallback") is None


# ---------------------------------------------------------------------------
# The fallback itself
# ---------------------------------------------------------------------------

def _agent(identifying=None):
    agent = NexaAgent.__new__(NexaAgent)
    agent._identifying_properties = identifying or {"Gene": "geneSymbol"}
    return agent


def _call(agent, error, kg_data, schema=ListAnswer):
    return agent._handle_structured_output_length_overflow(
        error,
        ToolMessage(content=json.dumps({"data": kg_data}), tool_call_id="test"),
        schema,
    )


_LENGTH_ERROR = type("LengthFinishReasonError", (Exception,), {})()

_GRAPH = {
    "nodes": [
        {"id": "n0", "label": "Gene", "geneSymbol": "APOE"},
        {"id": "n1", "label": "Gene", "geneSymbol": "APP"},
    ],
    "edges": [],
}


def test_recovers_the_entity_list_on_a_length_stop():
    response = _call(_agent(), _LENGTH_ERROR, _GRAPH)
    assert json.loads(response.content)["answer"] == ["APOE", "APP"]


def test_the_same_result_adapts_to_a_csv_schema():
    response = _call(_agent(), _LENGTH_ERROR, _GRAPH, schema=TextAnswer)
    assert json.loads(response.content)["answer"] == "APOE,APP"


def test_a_non_length_error_is_not_rescued():
    """Errors other than a length stop return None, so the caller re-raises."""
    assert _call(_agent(), ValueError("connection reset"), _GRAPH) is None


def test_a_non_entity_schema_is_not_rescued():
    assert _call(_agent(), _LENGTH_ERROR, _GRAPH, schema=BooleanAnswer) is None


def test_no_extractable_names_is_not_rescued():
    assert _call(_agent(), _LENGTH_ERROR, {"nodes": [], "edges": []}) is None


def test_identifying_properties_pick_the_property_the_gold_answer_uses():
    """The label's identifying property (geneSymbol for AlzKB genes) is used, not the display name."""
    nodes = [{"id": "n0", "label": "Gene", "commonName": "apolipoprotein E",
              "geneSymbol": "APOE"}]
    response = _call(_agent({"Gene": "geneSymbol"}), _LENGTH_ERROR,
                     {"nodes": nodes, "edges": []})
    assert json.loads(response.content)["answer"] == ["APOE"]


def test_a_property_colliding_with_a_reserved_key_is_read_from_its_namespace():
    """A property named like a reserved key is read from `<key>_property`."""
    nodes = [{"id": "n0", "label": "Gene", "label_property": "APOE"}]
    response = _call(_agent({"Gene": "label"}), _LENGTH_ERROR,
                     {"nodes": nodes, "edges": []})
    assert json.loads(response.content)["answer"] == ["APOE"]


def test_duplicate_names_are_deduplicated_in_order():
    nodes = [{"id": f"n{i}", "label": "Gene", "geneSymbol": s}
             for i, s in enumerate(["APP", "APOE", "APP"])]
    response = _call(_agent(), _LENGTH_ERROR, {"nodes": nodes, "edges": []})
    assert json.loads(response.content)["answer"] == ["APP", "APOE"]
