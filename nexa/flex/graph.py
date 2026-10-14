"""StateGraph builder for the flex agent.

An explicit ReAct loop. `respond_to_user` is a terminal sentinel: the turn ends
once it has run, rather than when the model stops calling tools.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from nexa.flex.state import FlexState
from nexa.tools.output_tools import RESPOND_TO_USER_TOOL_NAME
from nexa.utils.logger import get_logger

logger = get_logger(__name__)


# Prefix of the injected reminder, also used to cap reminders at one per turn.
_REMINDER_MARKER = "[SYSTEM REMINDER — internal]"
_REMINDER_TEXT = (
    _REMINDER_MARKER
    + " You replied with plain text but did not call the `respond_to_user` "
    "tool. Plain-text replies are NOT delivered to the user — the frontend "
    "only renders responses sent via `respond_to_user`. Please call "
    "`respond_to_user` now, passing your narrative as `answer_text` and the "
    "result_ids you used as `cited_result_ids` (use the result_ids you already "
    "obtained from your tool calls in this turn)."
)


def is_reminder(msg: Any) -> bool:
    """Whether `msg` is the internal reminder injected by `remind_respond`."""
    return (
        isinstance(msg, HumanMessage)
        and isinstance(msg.content, str)
        and msg.content.startswith(_REMINDER_MARKER)
    )


def current_turn(messages: list) -> list:
    """Return the messages after the latest user message.

    Internal reminders are not user messages.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage) and not is_reminder(messages[i]):
            return messages[i + 1:]
    return list(messages)


def build_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Compile the flex StateGraph.

    Nodes: `call_model` (tool-bound LLM), `call_tools` (ToolNode) and
    `remind_respond` (one reminder when the model answers in plain text).
    A turn ends after `respond_to_user` runs, or when the model answers in
    plain text again after the reminder.
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def call_model(state: FlexState) -> dict[str, Any]:
        """Invoke the tool-bound LLM with the current system prompt prepended."""
        messages = list(state.get("messages") or [])
        # The system prompt is added at call time, not stored in checkpoints; a
        # stored SystemMessage is replaced by the current prompt.
        if messages and isinstance(messages[0], SystemMessage):
            full = [SystemMessage(content=system_prompt)] + messages[1:]
        else:
            full = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(full)
        return {"messages": [response]}

    def remind_respond(state: FlexState) -> dict[str, Any]:
        """Ask the model to call `respond_to_user` (routed at most once per turn)."""
        return {"messages": [HumanMessage(content=_REMINDER_TEXT)]}

    # Both checks look at the current turn only, so earlier turns don't count.
    def _has_called_respond(msgs: list) -> bool:
        return any(
            isinstance(m, ToolMessage) and m.name == RESPOND_TO_USER_TOOL_NAME
            for m in current_turn(msgs)
        )

    def _reminder_count(msgs: list) -> int:
        return sum(1 for m in current_turn(msgs) if is_reminder(m))

    def route_after_model(state: FlexState) -> str:
        """Route to the tools, to a single reminder, or to END after a model step."""
        msgs = state.get("messages") or []
        if not msgs:
            return END
        last = msgs[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "call_tools"
        # Plain-text reply: end if already answered this turn, else remind once.
        if _has_called_respond(msgs):
            return END
        if _reminder_count(msgs) >= 1:
            logger.warning(
                "flex: model still produced plain text after one reminder; "
                "ending turn; the API will synthesize a final answer from the text."
            )
            return END
        logger.info("flex: model returned plain text without respond_to_user; sending one reminder.")
        return "remind_respond"

    def route_after_tools(state: FlexState) -> str:
        """End the turn if `respond_to_user` ran in the latest tool batch, else call the model."""
        msgs = state.get("messages") or []
        for msg in reversed(msgs):
            if isinstance(msg, ToolMessage):
                if msg.name == RESPOND_TO_USER_TOOL_NAME:
                    return END
                continue
            break  # End of the latest tool batch.
        return "call_model"

    graph = StateGraph(FlexState)
    graph.add_node("call_model", call_model)
    graph.add_node("call_tools", tool_node)
    graph.add_node("remind_respond", remind_respond)

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model",
        route_after_model,
        {"call_tools": "call_tools", "remind_respond": "remind_respond", END: END},
    )
    graph.add_conditional_edges(
        "call_tools",
        route_after_tools,
        {"call_model": "call_model", END: END},
    )
    graph.add_edge("remind_respond", "call_model")

    return graph.compile(checkpointer=checkpointer)
