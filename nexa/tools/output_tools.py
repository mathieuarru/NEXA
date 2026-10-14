"""Planning and output tools for the flex agent.

`write_todos` records the agent's plan. `respond_to_user` delivers the final
answer and ends the turn (the graph routes to END after it). It has two
variants, registered under the same name:
  - standard (default): free-form `answer_text`.
  - structured (when an `output_schema` Pydantic model is given): an `answer`
    object validated against that model, used for parseable benchmark answers.
"""

from __future__ import annotations

from typing import Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel as PydBaseModel, Field as PydField, create_model as pyd_create_model

from nexa.result_store import ResultStore
from nexa.tools.schemas import (
    GraphSpec,
    RespondToUserInput,
    WriteTodosInput,
)
from nexa.utils.logger import get_logger

logger = get_logger(__name__)


RESPOND_TO_USER_TOOL_NAME = "respond_to_user"
WRITE_TODOS_TOOL_NAME = "write_todos"


# ---------------------------------------------------------------------------
# write_todos
# ---------------------------------------------------------------------------


_WRITE_TODOS_DESCRIPTION = """\
Record or update the agent's todo list for the current question. Send the
FULL updated list every time — this tool replaces the previous list, it does
not append.

When to use:
- Multi-step questions (2+ Cypher queries needed, multi-hop reasoning).
- When the user gives several requests in one message.
- Update statuses (in_progress -> completed) as you make progress.

When NOT to use:
- Trivial single-query lookups ("how many drugs are in the KG?"). One-shot
  questions don't need a todo list.

Each todo: {content: imperative description, status: pending|in_progress|completed}.
"""


def _build_write_todos() -> StructuredTool:
    def _write_todos(todos: list) -> dict:
        # Items may arrive as TodoItem models or plain dicts.
        norm: list[dict] = []
        for t in todos:
            if hasattr(t, "model_dump"):
                norm.append(t.model_dump())
            elif isinstance(t, dict):
                norm.append(
                    {
                        "content": t.get("content", ""),
                        "status": t.get("status", "pending"),
                    }
                )
        return {"todos": norm, "count": len(norm)}

    return StructuredTool.from_function(
        func=_write_todos,
        name=WRITE_TODOS_TOOL_NAME,
        description=_WRITE_TODOS_DESCRIPTION,
        args_schema=WriteTodosInput,
    )


# ---------------------------------------------------------------------------
# respond_to_user
# ---------------------------------------------------------------------------


_RESPOND_TO_USER_DESCRIPTION = """\
Deliver the final answer to the user. This is the ONLY way to end the
conversation turn with a complete response.

Rules:
- `answer_text` is your narrative explanation. It may describe, synthesize,
  and interpret findings. It MUST NOT contain raw tabular rows — cite
  result_ids instead, so the frontend renders the actual stored data and the
  user can trust what they see.
- `cited_result_ids` lists the result_ids the user should see as tables/graphs.
  Pass an empty list ONLY when the answer is purely conversational (e.g. a
  clarification question to the user, or "I cannot answer that with this KG").
- Never invent or fabricate data in answer_text. Defer to the cited results.
- Call this exactly once when you have enough information to answer.
"""


def _build_respond_to_user(store: ResultStore, thread_id: str) -> StructuredTool:
    def _respond_to_user(
        answer_text: str,
        cited_result_ids: Optional[list[str]] = None,
        highlight_rows: Optional[dict[str, list[int]]] = None,
        graph_spec: Optional[dict] = None,
    ) -> dict:
        cited_result_ids = cited_result_ids or []

        # Keep only cited ids that exist in this thread; report the others.
        unknown: list[str] = []
        wrong_thread: list[str] = []
        validated: list[str] = []
        for rid in cited_result_ids:
            row = store.get(rid)
            if row is None:
                unknown.append(rid)
                continue
            if row["thread_id"] != thread_id:
                wrong_thread.append(rid)
                continue
            validated.append(rid)

        if hasattr(graph_spec, "model_dump"):
            graph_spec_dict = graph_spec.model_dump()
        elif isinstance(graph_spec, dict):
            graph_spec_dict = graph_spec
        else:
            graph_spec_dict = None

        result = {
            "answer_text": answer_text,
            "cited_result_ids": validated,
            "highlight_rows": highlight_rows,
            "graph_spec": graph_spec_dict,
            "final": True,
        }
        if unknown:
            result["unknown_result_ids"] = unknown
            logger.warning(
                f"respond_to_user: stripped unknown result_ids={unknown} for thread={thread_id}"
            )
        if wrong_thread:
            result["wrong_thread_result_ids"] = wrong_thread
            logger.warning(
                f"respond_to_user: stripped cross-thread result_ids={wrong_thread} "
                f"(thread={thread_id})"
            )
        return result

    return StructuredTool.from_function(
        func=_respond_to_user,
        name=RESPOND_TO_USER_TOOL_NAME,
        description=_RESPOND_TO_USER_DESCRIPTION,
        args_schema=RespondToUserInput,
    )


# ---------------------------------------------------------------------------
# respond_to_user, structured variant (user-supplied answer schema)
# ---------------------------------------------------------------------------


_RESPOND_TO_USER_STRUCTURED_DESCRIPTION = """\
Deliver the FINAL structured answer to the user. This call is the only way to
end the conversation turn.

This is a STRUCTURED variant: the `answer` field is NOT free-form text but
must conform exactly to the JSON Schema declared by the caller (typically a
benchmark question type — MCQ, True/False, OpenEnded). The LLM-side
validation will reject malformed calls; fix the structure and retry.

Other rules (unchanged):
- Cite the result_ids that back your answer in `cited_result_ids`.
- Never fabricate values — every assertion in `answer` must be supported by a
  cited result.
- Call this exactly once at the end of the turn.
"""


def _make_structured_respond_input(answer_model: Type[PydBaseModel]) -> Type[PydBaseModel]:
    """Build the structured variant's input model.

    `answer_model` becomes the `answer` field; the other fields match
    `RespondToUserInput`.
    """
    return pyd_create_model(
        "RespondToUserStructuredInput",
        __base__=PydBaseModel,
        answer=(
            answer_model,
            PydField(
                description=(
                    "The structured answer. Must conform exactly to the "
                    "schema. Each field must be filled — do not omit "
                    "required fields."
                )
            ),
        ),
        cited_result_ids=(
            list[str],
            PydField(
                default_factory=list,
                description=(
                    "result_ids that back the structured answer. Pass an "
                    "empty list only if the answer is conversational."
                ),
            ),
        ),
        highlight_rows=(
            Optional[dict[str, list[int]]],
            PydField(default=None, description="Optional row highlights per result_id."),
        ),
        graph_spec=(
            Optional[GraphSpec],
            PydField(default=None, description="Optional graph rendering hints."),
        ),
    )


def _build_respond_to_user_structured(
    store: ResultStore,
    thread_id: str,
    answer_model: Type[PydBaseModel],
) -> StructuredTool:
    StructuredInput = _make_structured_respond_input(answer_model)

    def _respond_to_user_structured(
        answer,
        cited_result_ids: Optional[list[str]] = None,
        highlight_rows: Optional[dict[str, list[int]]] = None,
        graph_spec: Optional[dict] = None,
    ) -> dict:
        cited_result_ids = cited_result_ids or []

        # Validate cited result_ids (same logic as the standard variant).
        unknown: list[str] = []
        wrong_thread: list[str] = []
        validated: list[str] = []
        for rid in cited_result_ids:
            row = store.get(rid)
            if row is None:
                unknown.append(rid)
                continue
            if row["thread_id"] != thread_id:
                wrong_thread.append(rid)
                continue
            validated.append(rid)

        # Serialize `answer` to a plain dict.
        if hasattr(answer, "model_dump"):
            answer_dict = answer.model_dump()
        elif isinstance(answer, dict):
            answer_dict = answer
        else:
            answer_dict = {"value": str(answer)}

        if hasattr(graph_spec, "model_dump"):
            graph_spec_dict = graph_spec.model_dump()
        elif isinstance(graph_spec, dict):
            graph_spec_dict = graph_spec
        else:
            graph_spec_dict = None

        result = {
            "answer": answer_dict,
            "cited_result_ids": validated,
            "highlight_rows": highlight_rows,
            "graph_spec": graph_spec_dict,
            "final": True,
        }
        if unknown:
            result["unknown_result_ids"] = unknown
            logger.warning(
                f"respond_to_user_structured: stripped unknown result_ids={unknown} "
                f"for thread={thread_id}"
            )
        if wrong_thread:
            result["wrong_thread_result_ids"] = wrong_thread
            logger.warning(
                f"respond_to_user_structured: stripped cross-thread result_ids="
                f"{wrong_thread} (thread={thread_id})"
            )
        return result

    # Same tool name as the standard variant.
    return StructuredTool.from_function(
        func=_respond_to_user_structured,
        name=RESPOND_TO_USER_TOOL_NAME,
        description=_RESPOND_TO_USER_STRUCTURED_DESCRIPTION,
        args_schema=StructuredInput,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_output_tools(
    store: ResultStore,
    thread_id: str,
    output_schema: Optional[Type[PydBaseModel]] = None,
) -> list[StructuredTool]:
    """Build the planning and output tools.

    With `output_schema`, `respond_to_user` takes `answer: output_schema`
    instead of `answer_text`; its name is unchanged.
    """
    if output_schema is not None:
        respond_tool = _build_respond_to_user_structured(store, thread_id, output_schema)
    else:
        respond_tool = _build_respond_to_user(store, thread_id)
    return [
        _build_write_todos(),
        respond_tool,
    ]
