"""NexaFlex: entry point of the flex agent.

Exposes the same methods as the direct `NexaAgent` (`query`, `stream_agent`,
`persist_stop_marker`), so the API and the benchmark can switch on
`agent_mode`. The LangGraph StateGraph is compiled per request, with its tools
bound to the thread.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Generator, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from psycopg_pool import ConnectionPool

from nexa.flex.graph import build_graph, current_turn, is_reminder
from nexa.flex.system_prompt import render_system_prompt
from nexa.flex.trace_format import (
    coerce_tool_payload as _coerce_tool_payload,
    extract_partial_string_field,
    safe_args as _safe_args,
    summarize_tool_result as _summarize_tool_result,
)
from nexa.flex.trace_reconstruction import reconstruct_trace_from_messages
from nexa.api.profile_manager import get_profile_with_password
from nexa.config import params
from nexa.core.nexa_agent import get_llm
from nexa.knowledge_graph.neo4j_handler import Neo4jHandler
from nexa.result_store import ResultStore
from nexa.tools import build_kg_tools, build_output_tools, build_result_tools
from nexa.tools.output_tools import RESPOND_TO_USER_TOOL_NAME
from nexa.utils.logger import get_logger

logger = get_logger(__name__)


class NexaFlex:
    """Per-request flex agent. Construction is cheap; build one per HTTP call."""

    def __init__(
        self,
        llm: BaseChatModel,
        neo4j_driver: Neo4jHandler,
        result_store: ResultStore,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        kg_profile_name: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        output_schema_model=None,
        output_schema_json: Optional[str] = None,
    ):
        self._llm = llm
        self._driver = neo4j_driver
        self._store = result_store
        self._checkpointer = checkpointer
        self._kg_profile_name = kg_profile_name
        self._custom_system_prompt = custom_system_prompt
        # When set, `respond_to_user` takes a structured `answer` of this model
        # instead of a free-form `answer_text`.
        self._output_schema_model = output_schema_model
        self._output_schema_json = output_schema_json

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        db_uri: str,
        model_provider: str,
        model_name: str,
        connection_pool: ConnectionPool,
        checkpointer: Optional[BaseCheckpointSaver],
        kg_profile_name: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        output_schema_model=None,
        output_schema_json: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
    ) -> "NexaFlex":
        """Build a NexaFlex from connection settings and model selection.

        Resolves the Neo4j connection from the named KG profile (falling back to
        environment defaults), opens a driver and result store, and instantiates
        the chat model for the given provider. API keys may be passed explicitly
        or resolved from the environment.
        """
        if kg_profile_name:
            profile = get_profile_with_password(kg_profile_name)
            if not profile:
                raise ValueError(f"KG Profile '{kg_profile_name}' not found.")
            host = profile.get("host") or params.NEO4J_HOST
            port = profile.get("port") or params.NEO4J_PORT
            username = profile.get("username") or params.NEO4J_USERNAME
            password = profile.get("password") or params.NEO4J_PASSWORD
            db_name = profile.get("db_name") or params.NEO4J_DB_NAME
        else:
            host, port = params.NEO4J_HOST, params.NEO4J_PORT
            username, password = params.NEO4J_USERNAME, params.NEO4J_PASSWORD
            db_name = params.NEO4J_DB_NAME

        driver = Neo4jHandler(
            host=host,
            port=str(port),
            username=username,
            password=password,
            db_name=db_name,
        )

        store = ResultStore(connection_pool)

        llm = get_llm(
            provider=model_provider,
            name=model_name,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            google_api_key=google_api_key,
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
        )

        return cls(
            llm=llm,
            neo4j_driver=driver,
            result_store=store,
            checkpointer=checkpointer,
            kg_profile_name=kg_profile_name,
            custom_system_prompt=custom_system_prompt,
            output_schema_model=output_schema_model,
            output_schema_json=output_schema_json,
        )

    # ---------------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------------

    @staticmethod
    def _parse_loose_json(text: str) -> Any:
        """Best-effort JSON parse of model output; None if nothing parses.

        Tolerates markdown fences, surrounding prose and unbalanced trailing braces.
        """
        s = text.strip()
        if s.startswith("```"):
            newline = s.find("\n")
            s = s[newline + 1:] if newline != -1 else s[3:]
            if s.rstrip().endswith("```"):
                s = s.rstrip()[:-3]
            s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        # Walk to the first balanced object, ignoring braces inside strings.
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(s)):
            char = s[i]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = not in_string
            elif not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(s[start:i + 1])
                        except Exception:
                            return None
        return None

    def _reshape_structured_answer(self, malformed: Any) -> Optional[dict]:
        """Re-emit an already-given answer in the declared output schema.

        Ollama does not constrain tool-call arguments, so a local model may pass
        `answer` as a (possibly malformed) JSON string instead of an object.
        Tries a tolerant parse, then constrained decoding with
        `with_structured_output`. Neither pass re-derives the answer: only its
        shape is fixed, so flex gets no extra attempt that direct does not.

        Returns None when neither pass yields an object.
        """
        text = malformed if isinstance(malformed, str) else json.dumps(
            malformed, ensure_ascii=False, default=str
        )

        parsed = self._parse_loose_json(text)
        if isinstance(parsed, dict):
            logger.info("reshape: recovered by parsing, no model call.")
            return parsed

        instruction = (
            "An assistant produced an answer that does not match the required "
            "structure. Re-emit the SAME answer in the required form. Do not "
            "re-derive it, do not add or drop entities, do not change the "
            "decision — fix only the shape."
        )
        try:
            structured = self._llm.with_structured_output(
                self._output_schema_model
            ).invoke([SystemMessage(content=instruction), HumanMessage(content=text)])
        except Exception as e:
            logger.warning(f"reshape: constrained pass failed: {type(e).__name__}: {e}")
            return None
        if not hasattr(structured, "model_dump"):
            return None
        logger.info(f"reshape: re-emitted a {self._output_schema_model.__name__}.")
        return structured.model_dump()

    def _build_tools_for_thread(self, thread_id: str):
        return [
            *build_kg_tools(self._driver, self._store, thread_id),
            *build_result_tools(self._store, thread_id),
            *build_output_tools(
                self._store, thread_id, output_schema=self._output_schema_model
            ),
        ]

    def _compile_for_thread(self, thread_id: str):
        tools = self._build_tools_for_thread(thread_id)
        system_prompt = render_system_prompt(
            kg_profile_name=self._kg_profile_name,
            custom_system_prompt=self._custom_system_prompt,
            output_schema_json=self._output_schema_json,
        )
        return build_graph(
            llm=self._llm,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=self._checkpointer,
        )

    # ---------------------------------------------------------------------
    # User-requested stop persistence
    # ---------------------------------------------------------------------

    STOP_MARKER = "_(Generation stopped by user.)_"

    def persist_stop_marker(self, thread_id: str) -> None:
        """Append the user-stop AIMessage to the thread's persisted state.

        `update_state` goes through the `add_messages` reducer, so the marker is
        appended. The frontend renders it as a "stopped" badge.
        """
        try:
            graph = self._compile_for_thread(thread_id)
            graph.update_state(
                {"configurable": {"thread_id": thread_id}},
                {"messages": [AIMessage(content=self.STOP_MARKER)]},
            )
        except Exception as e:
            logger.warning(
                f"persist_stop_marker failed for thread {thread_id}: {e}",
                exc_info=True,
            )

    def run(
        self,
        user_query: str,
        thread_id: str,
        recursion_limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run one turn synchronously and return the final state."""
        graph = self._compile_for_thread(thread_id)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit or params.NEXA_AGENT_RECURSION_LIMIT,
        }
        state_in = {
            "messages": [HumanMessage(content=user_query)],
            "kg_profile_name": self._kg_profile_name,
        }
        final_state = graph.invoke(state_in, config=config)
        return final_state

    def query(
        self,
        user_query: str,
        kg_mode: str = "Force",
        custom_system_prompt: Optional[str] = None,
        output_schema=None,
        recursion_limit: Optional[int] = None,
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Synchronous entry point used by the benchmark.

        Without `thread_id` each call uses a fresh thread whose stored results
        are deleted on return; passing the same `thread_id` again continues
        that conversation. Returns a `NexaAgent.query`-shaped dict plus
        flex-only telemetry and the turn's tool trace.

        `kg_mode` is accepted for signature compatibility and ignored.
        `output_schema` and `custom_system_prompt` are stored on the instance,
        so they also apply to later calls.
        """
        # Only a thread created here has its stored results deleted on return.
        owns_thread = thread_id is None
        thread_id = thread_id or f"bench-{uuid.uuid4()}"

        if output_schema is not None:
            self._output_schema_model = output_schema
            try:
                self._output_schema_json = json.dumps(output_schema.model_json_schema())
            except Exception:
                self._output_schema_json = None
        if custom_system_prompt and custom_system_prompt.strip():
            self._custom_system_prompt = custom_system_prompt

        limit = (
            recursion_limit
            or getattr(self, "_benchmark_recursion_limit", None)
            or params.NEXA_AGENT_RECURSION_LIMIT
        )

        recursion_hit = False
        final_state: dict[str, Any] = {}
        try:
            final_state = self.run(user_query, thread_id=thread_id, recursion_limit=limit)
        except GraphRecursionError:
            recursion_hit = True
            logger.warning(
                f"query: GraphRecursionError at recursion_limit={limit} for thread={thread_id}"
            )
            # Reload the state persisted before the limit to keep the partial trace.
            try:
                snap = self._compile_for_thread(thread_id).get_state(
                    {"configurable": {"thread_id": thread_id}}
                )
                final_state = dict(snap.values) if snap and snap.values else {}
            except Exception as reload_e:
                logger.warning(
                    f"query: could not reload state after recursion error: {reload_e}"
                )

        messages = final_state.get("messages") or []
        thread_events = (reconstruct_trace_from_messages(messages) or {}).get("events") or []
        # Telemetry and the answer cover the current turn only.
        turn = sum(1 for m in messages if isinstance(m, HumanMessage) and not is_reminder(m))
        events = [e for e in thread_events if e.get("turn") == turn]

        n_tool_calls = len(events)
        n_execute_cypher = sum(1 for e in events if e.get("name") == "execute_cypher")
        cypher_queries = [
            e["args"].get("query")
            for e in events
            if e.get("name") == "execute_cypher"
            and isinstance(e.get("args"), dict)
            and e["args"].get("query")
        ]
        distinct_result_ids = {e.get("result_id") for e in events if e.get("result_id")}

        # Completeness guard telemetry: only `complete is False` counts; an
        # absent key means not assessed.
        incomplete_result_ids = {
            e["result_id"] for e in events
            if e.get("complete") is False and e.get("result_id")
        }
        n_truncated_cypher = sum(
            1 for e in events
            if e.get("name") == "execute_cypher" and e.get("complete") is False
        )

        payload = extract_final_response({"messages": current_turn(messages)})
        answer_text = (payload.get("answer_text") or "") if payload else ""
        cited_result_ids = (payload.get("cited_result_ids") if payload else None) or []

        data: Any = None
        if payload:
            ans = payload.get("answer")
            if isinstance(ans, dict):
                data = ans
            elif ans is not None:
                data = ans

        diag = (
            f"recursion_hit={recursion_hit}, tool_calls={n_tool_calls}, "
            f"execute_cypher={n_execute_cypher}, result_ids={len(distinct_result_ids)}"
        )
        # Reshape an answer that missed the declared schema. A missing answer is
        # left missing.
        answer_rescued = False
        if (
            data is not None
            and not isinstance(data, dict)
            and self._output_schema_model is not None
            and params.NEXA_FLEX_ANSWER_RESCUE
        ):
            reshaped = self._reshape_structured_answer(data)
            if reshaped is not None:
                data = reshaped
                answer_rescued = True

        error: Optional[str] = None
        if data is None:
            error = (
                "flex: no structured answer (respond_to_user not called or no "
                "`answer`); " + diag
            )
        elif not any(e.get("name") == "execute_cypher" for e in thread_events):
            # Thread-wide: a follow-up may answer from an earlier turn's results.
            error = (
                "flex: answer produced WITHOUT any execute_cypher (possible "
                "fabrication from priors); " + diag
            )
        if recursion_hit:
            error = f"recursion_limit hit at {limit}; " + (error or diag)

        # The answer is extracted; drop the stored rows.
        if owns_thread:
            try:
                self._store.delete_by_thread(thread_id)
            except Exception as cleanup_e:
                logger.warning(f"query: delete_by_thread failed for {thread_id}: {cleanup_e}")

        return {
            "answer": json.dumps(data, ensure_ascii=False) if data is not None else answer_text,
            "cypher_query": cypher_queries[-1] if cypher_queries else None,
            "cypher_first_attempt": cypher_queries[0] if cypher_queries else None,
            "n_correction_iterations": max(0, n_execute_cypher - 1),
            "thinking": [answer_text] if answer_text else [],
            "data": data,
            "kg_results": None,
            "error": error,
            # flex-only extras, persisted by the benchmark harness:
            "trace": events,
            "n_tool_calls": n_tool_calls,
            "n_execute_cypher": n_execute_cypher,
            "n_distinct_result_ids": len(distinct_result_ids),
            "recursion_hit": recursion_hit,
            "cited_result_ids": cited_result_ids,
            # True when `_reshape_structured_answer` produced the answer.
            "answer_rescued": answer_rescued,
            # Completeness guard: truncated queries, incomplete results, and
            # incomplete results cited by the answer.
            "n_truncated_cypher": n_truncated_cypher,
            "n_incomplete_results": len(incomplete_result_ids),
            "n_cited_incomplete_results": len(
                incomplete_result_ids & set(cited_result_ids)
            ),
        }

    # ---------------------------------------------------------------------
    # API-compatible streaming surface
    # ---------------------------------------------------------------------

    def stream_agent(
        self,
        user_query: str,
        thread_id: str,
        kg_mode: str = "Auto",
        title: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        recursion_limit: Optional[int] = None,
        **_ignored_kwargs,
    ) -> Generator[dict[str, Any], None, None]:
        """Yield chunk dicts for the FastAPI streaming layer.

        Chunk shapes:
          {"type": "messages", "content"}: answer text, streamed token by token
          {"type": "tool_call", "name", "args", "summary", "result_id"}: one per
              executed tool
          {"type": "final", "answer_text", "answer", "cited_result_ids",
              "highlight_rows", "graph_spec"}: the final payload, emitted once

        `kg_mode` and the direct agent's extra keyword arguments are accepted
        and ignored.
        """
        # Allow the API to override custom_system_prompt at stream time.
        if custom_system_prompt and custom_system_prompt.strip():
            self._custom_system_prompt = custom_system_prompt

        graph = self._compile_for_thread(thread_id)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit or params.NEXA_AGENT_RECURSION_LIMIT,
        }
        state_in: dict[str, Any] = {
            "messages": [HumanMessage(content=user_query)],
            "kg_profile_name": self._kg_profile_name,
        }
        if title is not None:
            state_in["title"] = title

        pending_tool_inputs: dict[str, dict[str, Any]] = {}
        final_emitted = False
        last_ai_text: Optional[str] = None
        result_ids_in_turn: list[str] = []
        recursion_hit = False

        # Tool-call args arrive in token chunks: accumulate them per tool call
        # and forward only the new part of `respond_to_user.answer_text`.
        streaming_tool_names: dict[Any, str] = {}      # call id/index -> tool name
        streaming_args_buffer: dict[Any, str] = {}     # call id/index -> accumulated args JSON
        streaming_answer_emitted: dict[Any, int] = {}  # call id/index -> chars already emitted

        try:
            for stream_event in graph.stream(
                state_in, config=config, stream_mode=["messages", "updates"]
            ):
                # With list stream_mode, items are (mode_name, value) tuples.
                if not isinstance(stream_event, tuple) or len(stream_event) != 2:
                    continue
                mode, value = stream_event

                if mode == "messages":
                    # value = (AIMessageChunk, metadata_dict)
                    chunk, _metadata = value if isinstance(value, tuple) and len(value) == 2 else (value, {})
                    if not isinstance(chunk, AIMessageChunk):
                        continue

                    # 1. Plain-text content.
                    if isinstance(chunk.content, str) and chunk.content:
                        yield {"type": "messages", "content": chunk.content}
                    elif isinstance(chunk.content, list):
                        for block in chunk.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text") or ""
                                if text:
                                    yield {"type": "messages", "content": text}

                    # 2. In-flight `answer_text` of respond_to_user. Other
                    #    tools' args are shown in the trace only.
                    for tc_chunk in (getattr(chunk, "tool_call_chunks", None) or []):
                        key = tc_chunk.get("id") or tc_chunk.get("index")
                        if key is None:
                            continue
                        # `name` is only present on a tool call's first chunk.
                        if tc_chunk.get("name"):
                            streaming_tool_names[key] = tc_chunk["name"]
                        tool_name = streaming_tool_names.get(key)
                        args_delta = tc_chunk.get("args") or ""
                        if args_delta:
                            streaming_args_buffer[key] = streaming_args_buffer.get(key, "") + args_delta
                        if tool_name == "respond_to_user":
                            current = extract_partial_string_field(
                                streaming_args_buffer.get(key, ""), "answer_text"
                            )
                            already = streaming_answer_emitted.get(key, 0)
                            if len(current) > already:
                                yield {"type": "messages", "content": current[already:]}
                                streaming_answer_emitted[key] = len(current)
                    continue

                # mode == "updates": trace events and the final payload.
                if not isinstance(value, dict):
                    continue
                for _node_name, partial in value.items():
                    new_messages = (partial or {}).get("messages") or []
                    logger.debug(
                        f"stream_agent: node={_node_name} -> {len(new_messages)} msg(s) "
                        f"({[type(m).__name__ for m in new_messages]})"
                    )
                    for msg in new_messages:
                        if isinstance(msg, AIMessage):
                            # Fallback answer if respond_to_user is never called.
                            if isinstance(msg.content, str) and msg.content.strip():
                                last_ai_text = msg.content
                            elif isinstance(msg.content, list):
                                joined = "".join(
                                    (block.get("text") if isinstance(block, dict) and block.get("type") == "text" else "")
                                    for block in msg.content
                                )
                                if joined.strip():
                                    last_ai_text = joined
                            # Remember tool inputs to pair them with their ToolMessages.
                            for tc in (msg.tool_calls or []):
                                tc_id = tc.get("id") or tc.get("name")
                                pending_tool_inputs[tc_id] = {
                                    "name": tc.get("name"),
                                    "args": tc.get("args") or {},
                                }
                        elif isinstance(msg, ToolMessage):
                            # Pair with the matching tool-call input.
                            tc_id = getattr(msg, "tool_call_id", None) or msg.name
                            meta = pending_tool_inputs.pop(tc_id, None)
                            args = (meta or {}).get("args") or {}
                            summary, result_id = _summarize_tool_result(msg, args)
                            if result_id and result_id not in result_ids_in_turn:
                                result_ids_in_turn.append(result_id)
                            yield {
                                "type": "tool_call",
                                "name": msg.name,
                                "args": _safe_args(args),
                                "summary": summary,
                                "result_id": result_id,
                            }

                            # Emit the final payload once respond_to_user has run.
                            if msg.name == "respond_to_user" and not final_emitted:
                                payload = _coerce_tool_payload(msg.content) or {}
                                final_chunk = {
                                    "type": "final",
                                    "answer_text": payload.get("answer_text") or args.get("answer_text"),
                                    "answer": payload.get("answer") or args.get("answer"),
                                    "cited_result_ids": payload.get("cited_result_ids") or args.get("cited_result_ids") or [],
                                    "highlight_rows": payload.get("highlight_rows") or args.get("highlight_rows"),
                                    "graph_spec": payload.get("graph_spec") or args.get("graph_spec"),
                                }
                                yield final_chunk
                                final_emitted = True
        except GraphRecursionError:
            # Out of iterations before END: emit a partial answer instead of failing.
            recursion_hit = True
            logger.warning(
                f"stream_agent: GraphRecursionError after recursion_limit="
                f"{config['recursion_limit']}; emitting partial final"
            )
            note = (
                "⚠️ Iteration limit reached before the reasoning finished "
                f"(recursion_limit={config['recursion_limit']}). Partial answer. "
                "Increase NEXA_AGENT_RECURSION_LIMIT or split your question into "
                "several shorter messages."
            )
            yield {"type": "messages", "content": "\n\n" + note}

        # respond_to_user was never called: synthesize a final from the last
        # plain text, citing every result_id produced during the turn.
        if not final_emitted:
            text = last_ai_text or ("(answer interrupted)" if recursion_hit else "(no answer produced)")
            logger.warning(
                "stream_agent: respond_to_user never called; emitting synthesized final"
            )
            yield {"type": "messages", "content": text}
            yield {
                "type": "final",
                "answer_text": text,
                "cited_result_ids": result_ids_in_turn,
                "highlight_rows": None,
                "graph_spec": None,
            }


# ---------------------------------------------------------------------------
# Final-answer extraction
# ---------------------------------------------------------------------------


def extract_final_response(final_state: dict) -> Optional[dict]:
    """Return the payload of the latest `respond_to_user` call, or None.

    Starts from the tool's output and fills `answer`, `answer_text`,
    `highlight_rows` and `graph_spec` from the call's arguments when missing.
    """
    messages = final_state.get("messages") or []
    last_call_args: Optional[dict] = None
    last_tool_output: Optional[dict] = None

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == RESPOND_TO_USER_TOOL_NAME:
            last_tool_output = _coerce_tool_payload(msg.content)
            break

    # The call's own arguments, from the AIMessage.
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", None) or []:
                if tc.get("name") == RESPOND_TO_USER_TOOL_NAME:
                    last_call_args = tc.get("args") or {}
                    break
            if last_call_args is not None:
                break

    if last_tool_output is None and last_call_args is None:
        return None
    payload = last_tool_output if isinstance(last_tool_output, dict) else {}
    if last_call_args:
        for key in ("answer", "answer_text", "highlight_rows", "graph_spec"):
            if key not in payload and key in last_call_args:
                payload[key] = last_call_args[key]
    return payload
