"""Result-store tools for the flex agent.

They let the agent list, page through, filter, summarize and aggregate stored
query results without loading all rows into its context. filter_result and
aggregate_result create derived results that record their parent.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import StructuredTool

from nexa.result_store import ResultStore, infer_columns_schema, resolve_cell
from nexa.tools.schemas import (
    AggregateResultInput,
    FilterResultInput,
    ListResultsInput,
    ReadResultInput,
    SummarizeResultInput,
)
from nexa.utils.logger import get_logger

logger = get_logger(__name__)


_PREVIEW_ROWS = 5


# ---------------------------------------------------------------------------
# Completeness lineage
# ---------------------------------------------------------------------------
#
# A result derived from a truncated one is itself incomplete, so an empty
# filtered child must not be read as a true absence.

_DERIVED_INCOMPLETE_DIRECTIVE = (
    "INCOMPLETE RESULT — derived from result {parent}, which was itself cut "
    "short by its query's LIMIT. An empty or short result here does NOT mean "
    "the rows are absent from the graph; they may simply never have been "
    "retrieved. Re-query the graph directly instead of concluding from this."
)

_TRUNCATED_INCOMPLETE_DIRECTIVE = (
    "INCOMPLETE RESULT — this result was cut short by its query's LIMIT of "
    "{limit}, so the graph holds rows it does not contain. `total` below is "
    "what was stored, NOT what exists. Do not infer any absence from it: "
    "re-run the query without the LIMIT, or ask the specific question with a "
    "targeted `WHERE ... IN [...]`, `count(...)` or `EXISTS {{...}}` query."
)


def _derive_completeness(parent_full: dict) -> Optional[dict]:
    """Return the completeness verdict a derived result inherits from its parent.

    An unassessed parent (None) yields an unassessed child.
    """
    parent = parent_full.get("completeness")
    if not isinstance(parent, dict):
        return None
    if parent.get("complete"):
        return {"complete": True, "inherited_from": parent_full["result_id"]}
    return {
        "complete": False,
        "reason": "derived_from_incomplete",
        "inherited_from": parent_full["result_id"],
    }


def _attach_completeness(payload: dict, completeness: Optional[dict]) -> dict:
    """Add completeness fields, and a warning when incomplete, to a payload in place.

    No-op when the result was never assessed.
    """
    if completeness is None:
        return payload
    payload["complete"] = completeness["complete"]
    payload["completeness"] = completeness
    if not completeness["complete"]:
        if completeness.get("reason") == "derived_from_incomplete":
            description = _DERIVED_INCOMPLETE_DIRECTIVE.format(
                parent=completeness.get("inherited_from") or "an earlier query"
            )
        else:
            description = _TRUNCATED_INCOMPLETE_DIRECTIVE.format(
                limit=completeness.get("limit", "?")
            )
        payload.setdefault("warnings", []).insert(0, {
            "code": "Nexa.IncompleteResult",
            "severity": "WARNING",
            "description": description,
        })
    return payload


# ---------------------------------------------------------------------------
# list_results
# ---------------------------------------------------------------------------


_LIST_RESULTS_DESCRIPTION = """\
Return the metadata of every stored result in the current conversation thread
(no row data, only id/cypher/row_count/source/columns).

Use this when you have lost track of which result_ids exist, or to confirm a
previous query is still available before re-running it.
"""


def _build_list_results(store: ResultStore, thread_id: str) -> StructuredTool:
    def _list_results() -> dict:
        items = store.list_by_thread(thread_id)
        return {"count": len(items), "results": items}

    return StructuredTool.from_function(
        func=_list_results,
        name="list_results",
        description=_LIST_RESULTS_DESCRIPTION,
        args_schema=ListResultsInput,
    )


# ---------------------------------------------------------------------------
# read_result
# ---------------------------------------------------------------------------


_READ_RESULT_DESCRIPTION = """\
Return a paginated slice of a stored result by result_id.

Use this to inspect actual rows beyond the 5-row preview returned by
execute_cypher / filter_result. Always specify a limit appropriate to the task
— do NOT default to large limits just to dump everything; the data is already
safe in the store, and full dumps will overwhelm your context.

Returns: {result_id, rows, total, columns_schema, offset, limit}.
Returns an error object {"error": "..."} if the result_id is unknown.
"""


def _build_read_result(store: ResultStore) -> StructuredTool:
    def _read_result(result_id: str, limit: int = 20, offset: int = 0) -> dict:
        page = store.read(result_id, limit=limit, offset=offset)
        if page is None:
            return {"error": f"Unknown result_id: {result_id}"}
        # Restate the verdict on every page: `total` is the stored row count,
        # not what the graph holds.
        return _attach_completeness(page, page.get("completeness"))

    return StructuredTool.from_function(
        func=_read_result,
        name="read_result",
        description=_READ_RESULT_DESCRIPTION,
        args_schema=ReadResultInput,
    )


# ---------------------------------------------------------------------------
# filter_result
# ---------------------------------------------------------------------------


_FILTER_RESULT_DESCRIPTION = """\
Apply structured filter conditions (AND-combined) to an existing stored
result. Creates a new derived result and returns its result_id + preview.
The parent result_id is preserved in the new record's lineage.

Each condition is {column, op, value}. Supported ops:
  eq, neq, gt, gte, lt, lte    — straightforward comparisons (value typed)
  in                            — value must be a list; row matches if cell is in it
  contains                      — case-insensitive substring match on string cells
  is_null, is_not_null          — value is ignored

Use this instead of re-running a Cypher when you just need a subset of rows
already in memory.

Column references support DOT NOTATION to reach into node/relationship cells:
if a query did `RETURN g` (a whole Gene node), filter on `g.geneSymbol`, not
`geneSymbol`. Plain column names (from a `RETURN g.geneSymbol AS gene`) work as
before. Call `summarize_result` first if unsure which dotted fields exist.
"""


def _build_filter_result(store: ResultStore, thread_id: str) -> StructuredTool:
    def _filter_result(result_id: str, conditions: list[dict]) -> dict:
        full = store.get(result_id)
        if full is None:
            return {"error": f"Unknown result_id: {result_id}"}

        # Conditions may arrive as FilterCondition models or plain dicts.
        norm_conds = []
        for c in conditions:
            if hasattr(c, "model_dump"):
                norm_conds.append(c.model_dump())
            elif isinstance(c, dict):
                norm_conds.append(c)
            else:
                return {"error": f"Bad condition object: {c!r}"}

        try:
            kept = [row for row in full["data"] if _row_matches_all(row, norm_conds)]
        except Exception as e:
            logger.exception("filter_result failed during evaluation")
            return {"error": f"Filter evaluation error: {e}"}

        # Filtering keeps the parent's column types.
        columns_schema = full.get("columns_schema") or infer_columns_schema(kept)
        completeness = _derive_completeness(full)
        new_id = store.create(
            thread_id=thread_id,
            source_tool="filter_result",
            rows=kept,
            columns_schema=columns_schema,
            parent_result_id=full["result_id"],
            completeness=completeness,
        )
        return _attach_completeness({
            "result_id": new_id,
            "parent_result_id": full["result_id"],
            "row_count": len(kept),
            "columns_schema": columns_schema,
            "preview": kept[:_PREVIEW_ROWS],
        }, completeness)

    return StructuredTool.from_function(
        func=_filter_result,
        name="filter_result",
        description=_FILTER_RESULT_DESCRIPTION,
        args_schema=FilterResultInput,
    )


def _row_matches_all(row: dict, conditions: list[dict]) -> bool:
    if not isinstance(row, dict):
        return False
    for cond in conditions:
        if not _row_matches_one(row, cond):
            return False
    return True


def _row_matches_one(row: dict, cond: dict) -> bool:
    op = cond.get("op")
    col = cond.get("column")
    val = cond.get("value")
    # Dotted names (e.g. "g.geneSymbol") reach into node/relationship cells.
    cell = resolve_cell(row, col)

    if op == "is_null":
        return cell is None
    if op == "is_not_null":
        return cell is not None

    # All remaining ops treat NULL cells as non-matching (SQL semantics).
    if cell is None:
        return False

    if op == "eq":
        return cell == val
    if op == "neq":
        return cell != val
    if op == "gt":
        return _safe_compare(cell, val, lambda a, b: a > b)
    if op == "gte":
        return _safe_compare(cell, val, lambda a, b: a >= b)
    if op == "lt":
        return _safe_compare(cell, val, lambda a, b: a < b)
    if op == "lte":
        return _safe_compare(cell, val, lambda a, b: a <= b)
    if op == "in":
        if not isinstance(val, (list, tuple, set)):
            return False
        return cell in val
    if op == "contains":
        if cell is None or val is None:
            return False
        return str(val).lower() in str(cell).lower()
    return False


def _safe_compare(a: Any, b: Any, fn) -> bool:
    try:
        return fn(a, b)
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# summarize_result
# ---------------------------------------------------------------------------


_SUMMARIZE_RESULT_DESCRIPTION = """\
Compute per-column summary statistics on a stored result without pulling the
rows into your context.

For each column, returns: distinct value count, null count, dtype hint, and
the top-k most frequent values (for low-cardinality / categorical columns).
Numeric columns also get min/max/mean.

Node/relationship columns (from `RETURN g`) are flattened into dotted
sub-columns ("g.geneSymbol", "g.commonName", ...) so you can see — and then
filter/aggregate on — the actual scalar fields they contain.

Use when:
- You want a feel for distributions before deciding how to filter or query.
- The user asks "how varied is X" / "what are the most common Y" — answer
  from the summary instead of dumping all rows.
"""


def _build_summarize_result(store: ResultStore) -> StructuredTool:
    def _summarize_result(result_id: str, top_k: int = 5) -> dict:
        full = store.get(result_id)
        if full is None:
            return {"error": f"Unknown result_id: {result_id}"}
        rows = full.get("data") or []
        if not rows:
            return {"result_id": result_id, "row_count": 0, "columns": {}}

        columns_schema = full.get("columns_schema") or {}
        # Node/relationship cells are flattened one level into dotted
        # sub-columns ("g.geneSymbol", ...) that the agent can filter on.
        per_col_values: dict[str, list[Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            for k, v in row.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (dict, list)):
                            continue  # only surface scalar leaf fields
                        per_col_values.setdefault(f"{k}.{sub_k}", []).append(sub_v)
                else:
                    per_col_values.setdefault(k, []).append(v)

        summary: dict[str, dict[str, Any]] = {}
        for col, values in per_col_values.items():
            non_null = [v for v in values if v is not None]
            null_count = len(values) - len(non_null)
            distinct = _distinct_count_safe(non_null)
            entry: dict[str, Any] = {
                "dtype": columns_schema.get(col, _infer_dtype(non_null)),
                "null_count": null_count,
                "distinct_count": distinct,
            }
            # Numeric stats
            numeric = [v for v in non_null if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if numeric and len(numeric) == len(non_null):
                entry["min"] = min(numeric)
                entry["max"] = max(numeric)
                entry["mean"] = sum(numeric) / len(numeric)
            # Top-k values, for low-cardinality columns only
            if distinct and distinct <= max(top_k * 4, 100):
                counts: dict[Any, int] = {}
                for v in non_null:
                    key = v if isinstance(v, (str, int, float, bool)) else str(v)
                    counts[key] = counts.get(key, 0) + 1
                top = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:top_k]
                entry["top_values"] = [{"value": v, "count": c} for v, c in top]
            summary[col] = entry

        return {
            "result_id": result_id,
            "row_count": full.get("row_count", len(rows)),
            "columns": summary,
        }

    return StructuredTool.from_function(
        func=_summarize_result,
        name="summarize_result",
        description=_SUMMARIZE_RESULT_DESCRIPTION,
        args_schema=SummarizeResultInput,
    )


def _distinct_count_safe(values: list) -> int:
    seen: set = set()
    for v in values:
        if isinstance(v, (str, int, float, bool)):
            seen.add(v)
        else:
            seen.add(str(v))
    return len(seen)


def _infer_dtype(values: list) -> str:
    for v in values:
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "string"
        if isinstance(v, list):
            return "list"
        if isinstance(v, dict):
            return "dict"
    return "unknown"


# ---------------------------------------------------------------------------
# aggregate_result
# ---------------------------------------------------------------------------


_AGGREGATE_RESULT_DESCRIPTION = """\
Group-by + aggregate on a stored result, creating a new derived result.

Use for "how many X per Y" / "average score by category" style questions
that you've already pulled the data for. Cheaper than a new Cypher round-trip.

Supports: count, sum, avg, min, max. Group_by may be empty for a global
aggregation (one output row). Column references (both `group_by` and the
aggregation columns) support DOT NOTATION into node cells, e.g. group by
`g.geneSymbol` when the query returned whole `g` nodes.

Returns: {result_id, parent_result_id, row_count, columns_schema, preview}.
"""


def _build_aggregate_result(store: ResultStore, thread_id: str) -> StructuredTool:
    def _aggregate_result(
        result_id: str,
        group_by: Optional[list[str]] = None,
        aggregations: Optional[list] = None,
    ) -> dict:
        full = store.get(result_id)
        if full is None:
            return {"error": f"Unknown result_id: {result_id}"}
        rows = full.get("data") or []
        group_by = list(group_by or [])

        # Normalize aggregations (Pydantic objects or dicts).
        agg_specs: list[dict] = []
        for a in (aggregations or []):
            if hasattr(a, "model_dump"):
                agg_specs.append(a.model_dump())
            elif isinstance(a, dict):
                agg_specs.append(a)
        if not agg_specs:
            return {"error": "aggregations is required and must be non-empty"}

        # Group rows by the tuple of group_by values.
        groups: dict[tuple, list[dict]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            # Dotted paths descend into node dicts (e.g. group by "g.geneSymbol").
            key = tuple(resolve_cell(row, g) for g in group_by)
            groups.setdefault(key, []).append(row)

        out_rows: list[dict] = []
        for key, group_rows in groups.items():
            out: dict[str, Any] = {col: val for col, val in zip(group_by, key)}
            for spec in agg_specs:
                col = spec.get("column")
                op = spec.get("op")
                out_col = f"{op}_{col}"
                values = [resolve_cell(r, col) for r in group_rows]
                numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if op == "count":
                    out[out_col] = sum(1 for v in values if v is not None)
                elif op == "sum":
                    out[out_col] = sum(numeric) if numeric else None
                elif op == "avg":
                    out[out_col] = (sum(numeric) / len(numeric)) if numeric else None
                elif op == "min":
                    out[out_col] = min(numeric) if numeric else None
                elif op == "max":
                    out[out_col] = max(numeric) if numeric else None
                else:
                    out[out_col] = None
            out_rows.append(out)

        # Sort by group values.
        if group_by:
            out_rows.sort(key=lambda r: tuple(str(r.get(g)) for g in group_by))

        columns_schema = infer_columns_schema(out_rows)
        # An aggregate over a truncated result is itself incomplete.
        completeness = _derive_completeness(full)
        new_id = store.create(
            thread_id=thread_id,
            source_tool="aggregate_result",
            rows=out_rows,
            columns_schema=columns_schema,
            parent_result_id=result_id,
            completeness=completeness,
        )
        return _attach_completeness({
            "result_id": new_id,
            "parent_result_id": result_id,
            "row_count": len(out_rows),
            "columns_schema": columns_schema,
            "preview": out_rows[:_PREVIEW_ROWS],
        }, completeness)

    return StructuredTool.from_function(
        func=_aggregate_result,
        name="aggregate_result",
        description=_AGGREGATE_RESULT_DESCRIPTION,
        args_schema=AggregateResultInput,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_result_tools(store: ResultStore, thread_id: str) -> list[StructuredTool]:
    """Build the flex result tools (list, read, filter, summarize, aggregate) for a thread."""
    return [
        _build_list_results(store, thread_id),
        _build_read_result(store),
        _build_filter_result(store, thread_id),
        _build_summarize_result(store),
        _build_aggregate_result(store, thread_id),
    ]
