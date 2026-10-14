"""PostgreSQL store for Flex query results.

Full KG results, and results derived from them, are stored by id so the LLM
only sees a result id and a preview.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from nexa.utils.logger import get_logger

logger = get_logger(__name__)

_SCHEMA_SQL_PATH = Path(__file__).parent / "schema.sql"


class ResultStore:
    """Tabular results keyed by UUID and thread, with an optional parent result for lineage."""

    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def setup(self) -> None:
        """Create the tables and indexes if missing (idempotent)."""
        sql = _SCHEMA_SQL_PATH.read_text()
        # Strip full-line `--` comments before the naive split on ";", so a
        # semicolon inside a comment cannot cut a statement in two.
        sql = "\n".join(
            line for line in sql.splitlines() if not line.strip().startswith("--")
        )
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                for stmt in statements:
                    cur.execute(stmt)
        logger.info("ResultStore: schema ready.")

    def create(
        self,
        thread_id: str,
        source_tool: str,
        rows: list[dict],
        columns_schema: dict[str, str],
        cypher_query: Optional[str] = None,
        parent_result_id: Optional[str] = None,
        completeness: Optional[dict] = None,
    ) -> str:
        """Insert a result and return its UUID as a string.

        ``completeness`` is the completeness guard's verdict; None means
        "not assessed", not "complete".
        """
        result_id = str(uuid.uuid4())
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO nexa_v2_results
                        (result_id, thread_id, cypher_query, source_tool,
                         parent_result_id, columns_schema, row_count, data,
                         completeness)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        result_id,
                        thread_id,
                        cypher_query,
                        source_tool,
                        parent_result_id,
                        Jsonb(columns_schema),
                        len(rows),
                        Jsonb(rows),
                        Jsonb(completeness) if completeness is not None else None,
                    ),
                )
        return result_id

    def get(self, result_id: str) -> Optional[dict]:
        """Return the full record, including data, or None if not found."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT result_id, thread_id, created_at, cypher_query, source_tool,
                           parent_result_id, columns_schema, row_count, data,
                           completeness
                    FROM nexa_v2_results
                    WHERE result_id = %s
                    """,
                    (result_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return _row_to_full_dict(row)

    def read(
        self,
        result_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> Optional[dict]:
        """Return one page: {result_id, rows, total, columns_schema, offset, limit[, completeness]}.

        Loads the full data and slices it in Python.
        """
        full = self.get(result_id)
        if full is None:
            return None
        rows = full["data"][offset : offset + limit]
        page = {
            "result_id": full["result_id"],
            "rows": rows,
            "total": full["row_count"],
            "columns_schema": full["columns_schema"],
            "offset": offset,
            "limit": limit,
        }
        # Omitted when not assessed, so "not assessed" is distinct from "complete".
        if full.get("completeness") is not None:
            page["completeness"] = full["completeness"]
        return page

    def list_by_thread(self, thread_id: str) -> list[dict]:
        """List the metadata of a thread's results, without their data."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT result_id, created_at, cypher_query, source_tool,
                           parent_result_id, columns_schema, row_count,
                           completeness
                    FROM nexa_v2_results
                    WHERE thread_id = %s
                    ORDER BY created_at ASC
                    """,
                    (thread_id,),
                )
                rows = cur.fetchall()
        return [
            {
                "result_id": str(r[0]),
                "created_at": r[1].isoformat() if isinstance(r[1], datetime) else r[1],
                "cypher_query": r[2],
                "source_tool": r[3],
                "parent_result_id": str(r[4]) if r[4] else None,
                "columns_schema": r[5],
                "row_count": r[6],
                **({"completeness": r[7]} if r[7] is not None else {}),
            }
            for r in rows
        ]

    def delete_by_thread(self, thread_id: str) -> int:
        """Delete all results of a thread and return how many were deleted."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM nexa_v2_results WHERE thread_id = %s",
                    (thread_id,),
                )
                return cur.rowcount


def _row_to_full_dict(row: tuple) -> dict:
    return {
        "result_id": str(row[0]),
        "thread_id": row[1],
        "created_at": row[2].isoformat() if isinstance(row[2], datetime) else row[2],
        "cypher_query": row[3],
        "source_tool": row[4],
        "parent_result_id": str(row[5]) if row[5] else None,
        "columns_schema": row[6],
        "row_count": row[7],
        "data": row[8],
        "completeness": row[9],
    }


_MISSING = object()


def resolve_cell(row: dict, column: str) -> Any:
    """Return the value of a possibly dotted column in ``row``, or None if it does not resolve.

    Cells holding whole nodes are nested dicts, so a dotted path descends into
    them. A literal key equal to the full column name takes precedence.

        resolve_cell({"g": {"geneSymbol": "APOE"}}, "g.geneSymbol")  -> "APOE"
        resolve_cell({"gene": "APOE"}, "gene")                        -> "APOE"
    """
    if not isinstance(row, dict):
        return None
    # Also matches literal dotted keys, which some drivers emit.
    if column in row:
        return row[column]
    if "." not in column:
        return None
    cur: Any = row
    for segment in column.split("."):
        if isinstance(cur, dict):
            nxt = cur.get(segment, _MISSING)
            if nxt is _MISSING:
                return None
            cur = nxt
        else:
            return None
    return cur


def infer_columns_schema(rows: list[dict]) -> dict[str, str]:
    """Infer each column's type from its first non-null value; all-null columns are "unknown"."""
    schema: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if key in schema:
                continue
            if value is None:
                continue
            schema[key] = _python_type_name(value)
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row:
            schema.setdefault(key, "unknown")
    return schema


def _python_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__
