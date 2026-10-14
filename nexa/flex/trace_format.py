"""Tool-event formatting shared by live streaming and trace reconstruction.

`stream_agent` and `trace_reconstruction` both use these helpers, so a live
trace and one rebuilt from a checkpoint have identical summaries.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import ToolMessage


def coerce_tool_payload(content: Any) -> Optional[dict]:
    """Parse ToolMessage content (a JSON string, a dict or a list of text blocks)."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except Exception:
            return None
    if isinstance(content, list):
        joined = "".join(
            (block.get("text") if isinstance(block, dict) and block.get("type") == "text" else "")
            for block in content
        )
        try:
            return json.loads(joined)
        except Exception:
            return {"answer_text": joined}
    return None


def summarize_tool_result(msg: ToolMessage, input_args: dict) -> tuple[str, Optional[str]]:
    """Return a short summary of one tool execution and its result_id, if any."""
    payload = coerce_tool_payload(msg.content) or {}
    name = msg.name or "tool"
    result_id: Optional[str] = None

    if name == "execute_cypher":
        err = payload.get("error")
        if err:
            return (f"Cypher failed: {str(err)[:120]}", None)
        result_id = payload.get("result_id")
        row_count = payload.get("row_count", 0)
        if row_count == 0:
            return ("Cypher returned 0 rows", result_id)
        # Only `complete is False` means truncated; a missing key means not assessed.
        if payload.get("complete") is False:
            limit = (payload.get("completeness") or {}).get("limit", "?")
            return (
                f"Cypher executed → {row_count} row(s) stored, TRUNCATED at LIMIT {limit}",
                result_id,
            )
        return (f"Cypher executed → {row_count} row(s) stored", result_id)

    if name == "read_kg_schema":
        nodes = payload.get("nodes") or {}
        rels = payload.get("relationships") or []
        return (f"Schema read: {len(nodes)} labels, {len(rels)} relationships", None)

    if name == "lookup_property":
        candidates = payload.get("candidates") or []
        prop = input_args.get("property") or "?"
        return (f"lookup_property({prop}) → {len(candidates)} candidate(s)", None)

    if name == "list_results":
        count = payload.get("count", 0)
        return (f"{count} stored result(s) in thread", None)

    if name == "read_result":
        if "error" in payload:
            return (f"read_result failed: {payload['error']}", None)
        rid = input_args.get("result_id") or payload.get("result_id")
        total = payload.get("total", 0)
        rows = payload.get("rows") or []
        return (f"Read {len(rows)}/{total} rows from result", rid)

    if name == "filter_result":
        if "error" in payload:
            return (f"filter_result failed: {payload['error']}", None)
        result_id = payload.get("result_id")
        suffix = " (from a TRUNCATED result)" if payload.get("complete") is False else ""
        return (f"Filtered → {payload.get('row_count', 0)} row(s){suffix}", result_id)

    if name == "summarize_result":
        if "error" in payload:
            return (f"summarize_result failed: {payload['error']}", None)
        n_cols = len(payload.get("columns") or {})
        return (f"Summary on {payload.get('row_count', 0)} row(s), {n_cols} column(s)", None)

    if name == "aggregate_result":
        if "error" in payload:
            return (f"aggregate_result failed: {payload['error']}", None)
        result_id = payload.get("result_id")
        return (f"Aggregated → {payload.get('row_count', 0)} group(s)", result_id)

    if name == "validate_cypher":
        if payload.get("valid"):
            return ("Cypher validated (no schema issues)", None)
        n_err = len(payload.get("errors") or [])
        return (f"Cypher invalid: {n_err} schema issue(s)", None)

    if name == "write_todos":
        count = payload.get("count", 0)
        return (f"Todo list updated ({count} item(s))", None)

    if name == "respond_to_user":
        cited = payload.get("cited_result_ids") or input_args.get("cited_result_ids") or []
        return (f"Final answer ({len(cited)} citation(s))", None)

    return (f"{name} executed", payload.get("result_id"))


def safe_args(args: dict) -> dict:
    """Make tool args JSON-serializable, truncating long strings for transport."""
    safe: dict[str, Any] = {}
    for k, v in (args or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            if isinstance(v, str) and len(v) > 5000:
                safe[k] = v[:5000] + "...<truncated>"
            else:
                safe[k] = v
        else:
            try:
                json.dumps(v, default=str)
                safe[k] = v
            except Exception:
                safe[k] = str(v)
    return safe


def extract_partial_string_field(partial_json: str, field_name: str) -> str:
    """Return the value received so far for a string field of a partial JSON object.

    Streamed tool-call arguments arrive as growing prefixes of the final JSON
    (e.g. `{"answer_text": "Hel`). Returns the characters after the opening
    quote, up to the closing quote once it has arrived, with JSON escapes
    decoded. Returns "" if the field has not started yet.
    """
    if not partial_json:
        return ""
    needle = '"' + field_name + '"'
    idx = partial_json.find(needle)
    if idx < 0:
        return ""
    # Skip the field name, the colon, whitespace and the opening quote.
    i = idx + len(needle)
    n = len(partial_json)
    while i < n and partial_json[i] in " \t\r\n":
        i += 1
    if i >= n or partial_json[i] != ":":
        return ""
    i += 1
    while i < n and partial_json[i] in " \t\r\n":
        i += 1
    if i >= n or partial_json[i] != '"':
        return ""
    i += 1
    # Walk characters, honoring escapes, until end-of-buffer or unescaped `"`.
    out: list[str] = []
    while i < n:
        c = partial_json[i]
        if c == "\\":
            if i + 1 >= n:
                # Incomplete escape: wait for more data.
                break
            esc = partial_json[i + 1]
            if esc == "n":
                out.append("\n")
            elif esc == "t":
                out.append("\t")
            elif esc == "r":
                out.append("\r")
            elif esc == '"':
                out.append('"')
            elif esc == "\\":
                out.append("\\")
            elif esc == "/":
                out.append("/")
            elif esc == "u" and i + 5 < n:
                try:
                    out.append(chr(int(partial_json[i + 2 : i + 6], 16)))
                    i += 6
                    continue
                except ValueError:
                    out.append(esc)
            else:
                out.append(esc)
            i += 2
            continue
        if c == '"':
            break  # closing quote
        out.append(c)
        i += 1
    return "".join(out)
