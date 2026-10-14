"""KG tools for the flex agent.

Each builder returns a StructuredTool wrapping Neo4jHandler methods or the
property fuzzy matcher. Tools return JSON-serializable dicts; execute_cypher
stores the full result in the ResultStore and returns a preview and a result_id.
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.tools import StructuredTool

from nexa.config import params
from nexa.knowledge_graph.neo4j_handler import Neo4jHandler
from nexa.knowledge_graph.read_only import ReadOnlyViolation, admit_read_query, policy_notification
from nexa.knowledge_graph.properties_correction import find_top_k_similar_values
from nexa.result_store import ResultStore, infer_columns_schema
from nexa.tools.schemas import (
    ExecuteCypherInput,
    LookupPropertyInput,
    ReadKgSchemaInput,
    ValidateCypherInput,
)
from nexa.utils.cypher_utils import parse_trailing_limit, replace_trailing_limit
from nexa.utils.logger import get_logger
from nexa.utils.utils import tokenize_for_matching

logger = get_logger(__name__)


_PREVIEW_ROWS = 5


# ---------------------------------------------------------------------------
# execute_cypher
# ---------------------------------------------------------------------------


_EXECUTE_CYPHER_DESCRIPTION = """\
Execute a Cypher query against the active knowledge graph.

The FULL result set is stored server-side in the result store; you receive
only a preview (up to 5 rows), the column schema, the row count, and a
`result_id`. Use `read_result`, `filter_result`, or `summarize_result` with
that `result_id` to drill into the data. NEVER restate raw rows in your
reasoning — reference the result_id instead.

When to use:
- The user asks a question that requires querying the KG.
- You need fresh data (you have not stored an equivalent result this turn).

Before calling:
- If your query contains a hardcoded property value in WHERE/property pattern
  (e.g. `WHERE g.symbol = 'APOE'`), first call `lookup_property` to confirm
  the value exists in that exact form.

On error: the response includes `error: <message>`. Fix the Cypher and retry,
unless `retryable` is false. A read-only policy refusal is terminal: report it
to the user and do not try another query to perform the refused operation.
"""


_INCOMPLETE_DIRECTIVE = (
    "INCOMPLETE RESULT — your query returned exactly its own LIMIT of {limit} "
    "row(s) and the graph holds at least one more. Rows were left behind, so "
    "this result does NOT license any closed-world conclusion: you may not "
    "infer that something is absent because it is missing here. Either re-run "
    "the query without the LIMIT, or answer the specific question directly "
    "with a targeted `WHERE ... IN [...]`, `count(...)` or `EXISTS {{...}}` "
    "query instead of enumerating and scanning."
)


def _assess_completeness(driver: Neo4jHandler, query: str, row_count: int) -> Optional[dict]:
    """Tell whether the query's own trailing LIMIT cut the result short.

    Returns None when not assessed (guard off, or no outermost integer
    ``LIMIT``); None is never a completeness claim. The check re-runs the query
    with ``LIMIT n+1``: the cost stays bounded and only the integer is
    rewritten, unlike wrapping an arbitrary query in a ``count(*)``.
    """
    if not params.NEXA_COMPLETENESS_GUARD:
        return None

    limit = parse_trailing_limit(query)
    if limit is None:
        return None
    # Fewer rows than the limit: the result is whole.
    if row_count < limit:
        return {"complete": True, "limit": limit, "rows_returned": row_count}

    probe = replace_trailing_limit(query, limit + 1)
    if probe is None:  # unreachable while parse_trailing_limit matched
        return None

    probe_rows, probe_warnings = driver.execute_query(probe)
    if probe_rows is None:
        # Probe failed: leave the result unassessed.
        logger.warning(
            f"completeness probe failed (limit={limit}); leaving result unassessed: "
            f"{probe_warnings}"
        )
        return None

    if len(probe_rows) > row_count:
        logger.info(
            f"completeness guard: result truncated at LIMIT {limit} "
            f"({row_count} rows returned, more available)"
        )
        return {
            "complete": False,
            "limit": limit,
            "rows_returned": row_count,
            "reason": "trailing_limit_saturated",
        }
    return {"complete": True, "limit": limit, "rows_returned": row_count}


def _build_execute_cypher(
    driver: Neo4jHandler,
    store: ResultStore,
    thread_id: str,
) -> StructuredTool:
    def _execute_cypher(query: str) -> dict:
        data, warnings = driver.execute_query(query)
        if data is None:
            err = (
                warnings[0]["description"]
                if warnings
                else "Unknown error from Neo4j driver."
            )
            logger.warning(f"execute_cypher error for thread={thread_id}: {err}")
            return {
                "result_id": None,
                "row_count": 0,
                "columns_schema": {},
                "preview": [],
                "error": err,
                "error_code": warnings[0].get("code") if warnings else None,
                "retryable": not bool(policy_notification(warnings)),
                "warnings": warnings,
            }

        columns = infer_columns_schema(data)
        completeness = _assess_completeness(driver, query, len(data))
        result_id = store.create(
            thread_id=thread_id,
            source_tool="execute_cypher",
            rows=data,
            columns_schema=columns,
            cypher_query=query,
            completeness=completeness,
        )
        out_warnings = list(warnings or [])
        if completeness is not None and not completeness["complete"]:
            out_warnings.insert(0, {
                "code": "Nexa.IncompleteResult",
                "severity": "WARNING",
                "description": _INCOMPLETE_DIRECTIVE.format(limit=completeness["limit"]),
            })
        payload = {
            "result_id": result_id,
            "row_count": len(data),
            "columns_schema": columns,
            "preview": data[:_PREVIEW_ROWS],
            "error": None,
            "warnings": out_warnings,
        }
        if completeness is not None:
            payload["complete"] = completeness["complete"]
            payload["completeness"] = completeness
        return payload

    return StructuredTool.from_function(
        func=_execute_cypher,
        name="execute_cypher",
        description=_EXECUTE_CYPHER_DESCRIPTION,
        args_schema=ExecuteCypherInput,
    )


# ---------------------------------------------------------------------------
# read_kg_schema
# ---------------------------------------------------------------------------


_READ_KG_SCHEMA_DESCRIPTION = """\
Return the active knowledge graph's schema: node labels with their properties,
relationship types with source/target labels, and relationship properties.

Call this BEFORE writing any Cypher against an unfamiliar profile, or when
your previous Cypher failed with an "unknown label/relationship/property"
error. The schema is small and cheap to retrieve — prefer reading it once
upfront over guessing at structure.
"""


def _build_read_kg_schema(driver: Neo4jHandler) -> StructuredTool:
    def _read_kg_schema() -> dict:
        nodes_raw = driver.get_nodes() or []
        rels_raw = driver.get_relationships() or []
        rel_props_raw = driver.get_relationships_properties() or []

        # Reshape into LLM-friendly maps.
        nodes: dict[str, list[str]] = {}
        for entry in nodes_raw:
            node = entry.get("node") if isinstance(entry, dict) else None
            if not isinstance(node, dict):
                continue
            label = node.get("labels")
            props = node.get("properties") or []
            if not label:
                continue
            existing = nodes.setdefault(label, [])
            for p in props:
                if p not in existing:
                    existing.append(p)

        rel_properties: dict[str, list[str]] = {}
        for entry in rel_props_raw:
            rel = entry.get("relationship") if isinstance(entry, dict) else None
            if not isinstance(rel, dict):
                continue
            rtype = rel.get("type")
            props = rel.get("properties") or []
            if not rtype:
                continue
            existing = rel_properties.setdefault(rtype, [])
            for p in props:
                if p not in existing:
                    existing.append(p)

        relationships = [
            {
                "type": r.get("relationship"),
                "source": r.get("source"),
                "target": r.get("target"),
            }
            for r in rels_raw
            if isinstance(r, dict) and r.get("relationship")
        ]

        return {
            "nodes": nodes,
            "relationships": relationships,
            "relationship_properties": rel_properties,
        }

    return StructuredTool.from_function(
        func=_read_kg_schema,
        name="read_kg_schema",
        description=_READ_KG_SCHEMA_DESCRIPTION,
        args_schema=ReadKgSchemaInput,
    )


# ---------------------------------------------------------------------------
# lookup_property
# ---------------------------------------------------------------------------


_LOOKUP_PROPERTY_DESCRIPTION = """\
Fuzzy-match a candidate value against the actual values that exist in the
knowledge graph for a given label or relationship type. Ranked by a composite
score (exact match, case-insensitive equality, substring containment, token
overlap).

Two modes, selected by whether you pass `property`:
- TARGETED (you pass `property`): match only that property's values. Use when
  you already know the identifying property (e.g. `geneSymbol` for Gene).
  Returns `{candidates: [value, ...], warning}`. Fast — single property scan.
- ENTITY RESOLUTION (you OMIT `property`): search across ALL scalar properties
  of the label/rel in one pass and report which (property, value) pairs match.
  Use when you don't know which property holds the name, or to discover the
  identifying property of an entity. Returns
  `{candidates: [{property, value, score}, ...], warning}`.

When to use:
- BEFORE writing a Cypher that pins a value to a literal (e.g.
  `WHERE g.geneSymbol = 'APOE'`). Confirm the value exists in that exact
  case/form so your query doesn't silently return zero rows.
- When a previous execute_cypher returned 0 rows and a misspelled/miscased
  literal is the likely cause.
- When the user gives a name that may be a synonym/abbreviation/loose spelling
  (e.g. "alzheimer s" vs "Alzheimer's Disease").

An empty candidates list means nothing matched — the label is wrong, or the
value simply isn't in the KG.
"""


def _build_lookup_property(driver: Neo4jHandler) -> StructuredTool:
    def _lookup_property(
        element_type: str,
        label_or_rel: str,
        query: str,
        property: Optional[str] = None,
        k: int = 10,
    ) -> dict:
        # Neo4jHandler expects "nodes" / "relationships".
        internal_el = "nodes" if element_type == "node" else "relationships"

        tokens = tokenize_for_matching(query)

        # Entity-resolution mode: no property given, so search all scalar
        # properties of the label/rel and report matching (property, value) pairs.
        if not property:
            candidates, warning = driver.find_value_across_properties(
                element_type=internal_el,
                label_or_rel=label_or_rel,
                query_value=query,
                tokens=tokens,
                limit=max(k, 10),
            )
            warning_msg_b: Optional[str] = None
            if isinstance(warning, list) and warning:
                warning_msg_b = warning[0].get("description") if isinstance(warning[0], dict) else str(warning[0])
            elif isinstance(warning, dict):
                warning_msg_b = warning.get("description") or str(warning)
            elif isinstance(warning, str):
                warning_msg_b = warning
            return {"candidates": candidates[:k], "warning": warning_msg_b}
        candidates, warning = driver.find_property_value_candidates(
            element_type=internal_el,
            label_or_rel=label_or_rel,
            property_name=property,
            query_value=query,
            tokens=tokens,
            limit=50,
        )

        warning_msg: Optional[str] = None
        if isinstance(warning, list) and warning:
            warning_msg = warning[0].get("description") if isinstance(warning[0], dict) else str(warning[0])
        elif isinstance(warning, dict):
            warning_msg = warning.get("description") or str(warning)
        elif isinstance(warning, str):
            warning_msg = warning

        if not candidates:
            # Fallback: pull all distinct values for this property, rerank in Python.
            if internal_el == "nodes":
                existing, w2 = driver.get_property_values_for_node(label_or_rel, property)
            else:
                existing, w2 = driver.get_property_values_for_relationship(label_or_rel, property)
            if not existing:
                if w2 and isinstance(w2, list) and w2:
                    warning_msg = warning_msg or (
                        w2[0].get("description") if isinstance(w2[0], dict) else str(w2[0])
                    )
                return {"candidates": [], "warning": warning_msg}
            ranked = find_top_k_similar_values(query, existing, k)
            return {"candidates": ranked, "warning": warning_msg}

        ranked = find_top_k_similar_values(query, candidates, k)
        return {"candidates": ranked, "warning": warning_msg}

    return StructuredTool.from_function(
        func=_lookup_property,
        name="lookup_property",
        description=_LOOKUP_PROPERTY_DESCRIPTION,
        args_schema=LookupPropertyInput,
    )


# ---------------------------------------------------------------------------
# validate_cypher
# ---------------------------------------------------------------------------


_VALIDATE_CYPHER_DESCRIPTION = """\
Lightweight pre-flight check on a Cypher query: parses node/relationship
patterns and verifies each label, relationship type, and property reference
exists in the active KG schema. Also checks the local read-only admission
policy. The query is NOT executed; mandatory server classification still runs
inside execute_cypher, independently of this optional tool.

Returns: {valid: bool, errors: [...], warnings: [...]}.

Cheap and offline (just reads the schema once). Use when:
- You wrote a non-trivial Cypher and want a sanity check before paying the
  round-trip to Neo4j.
- A previous execute_cypher failed with a label/relationship/property error
  and you want to confirm the new version is clean.
- Multi-hop / multi-relation queries where a single typo blocks the whole
  thing.

Skip for trivial queries (`MATCH (n:Drug) RETURN count(n)` etc.) — it costs
more in tokens than just running.
"""


# Regex approximations of the common pattern forms, not a full Cypher parser.
_LABEL_PATTERN = re.compile(r"\(\s*\w*\s*:\s*`?(\w+)`?")
_REL_PATTERN = re.compile(r"\[\s*\w*\s*:\s*`?(\w+)`?")
# Inline {prop: ...} in a node pattern
_INLINE_NODE_PROP = re.compile(r"\(\s*\w*\s*:\s*`?(\w+)`?\s*\{([^}]+)\}\s*\)")


def _build_validate_cypher(driver: Neo4jHandler) -> StructuredTool:
    def _validate_cypher(query: str) -> dict:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            admit_read_query(query)
        except ReadOnlyViolation as e:
            return {"valid": False, "errors": [f"Read-only policy: {e}"], "warnings": []}
        try:
            nodes_raw = driver.get_nodes() or []
            rels_raw = driver.get_relationships() or []
        except Exception as e:
            return {"valid": False, "errors": [f"Could not load schema: {e}"], "warnings": []}

        node_labels: dict[str, set[str]] = {}
        for entry in nodes_raw:
            node = entry.get("node") if isinstance(entry, dict) else None
            if not isinstance(node, dict):
                continue
            label = node.get("labels")
            if not label:
                continue
            node_labels.setdefault(label, set()).update(node.get("properties") or [])

        rel_types: set[str] = {
            r.get("relationship") for r in rels_raw
            if isinstance(r, dict) and r.get("relationship")
        }

        # 1) Labels referenced in node patterns
        for label in set(_LABEL_PATTERN.findall(query)):
            if label not in node_labels:
                close = _did_you_mean(label, node_labels.keys())
                msg = f"Unknown node label: '{label}'."
                if close:
                    msg += f" Did you mean: {close}?"
                errors.append(msg)

        # 2) Relationship types referenced in relation patterns
        for rel in set(_REL_PATTERN.findall(query)):
            if rel not in rel_types:
                close = _did_you_mean(rel, rel_types)
                msg = f"Unknown relationship type: '{rel}'."
                if close:
                    msg += f" Did you mean: {close}?"
                errors.append(msg)

        # 3) Inline {prop: value} in node patterns
        for label, props_blob in _INLINE_NODE_PROP.findall(query):
            if label not in node_labels:
                continue  # already reported above
            for kv in props_blob.split(","):
                if ":" not in kv:
                    continue
                prop = kv.split(":", 1)[0].strip().strip("`").strip()
                if prop and prop not in node_labels[label]:
                    close = _did_you_mean(prop, node_labels[label])
                    msg = f"Unknown property '{prop}' on label '{label}'."
                    if close:
                        msg += f" Did you mean: {close}?"
                    errors.append(msg)

        # 4) alias.property references whose alias resolves to a label.
        alias_to_label: dict[str, str] = {}
        for m in re.finditer(r"\(\s*(\w+)\s*:\s*`?(\w+)`?", query):
            alias, label = m.group(1), m.group(2)
            if alias:
                alias_to_label[alias] = label
        for m in re.finditer(r"\b(\w+)\.([\w`]+)", query):
            alias, prop = m.group(1), m.group(2).strip("`")
            if alias in alias_to_label and prop:
                label = alias_to_label[alias]
                if label in node_labels and prop not in node_labels[label]:
                    close = _did_you_mean(prop, node_labels[label])
                    msg = f"Unknown property '{prop}' on label '{label}' (via alias '{alias}')."
                    if close:
                        msg += f" Did you mean: {close}?"
                    if msg not in errors:
                        errors.append(msg)

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        }

    return StructuredTool.from_function(
        func=_validate_cypher,
        name="validate_cypher",
        description=_VALIDATE_CYPHER_DESCRIPTION,
        args_schema=ValidateCypherInput,
    )


def _did_you_mean(value: str, candidates) -> Optional[str]:
    """Suggest up to three candidates matching `value` by case-insensitive prefix or substring."""
    if not value:
        return None
    lower = value.lower()
    best: list[str] = []
    for cand in candidates:
        c_lower = cand.lower()
        if c_lower == lower:
            return cand
        if c_lower.startswith(lower) or lower.startswith(c_lower) or lower in c_lower or c_lower in lower:
            best.append(cand)
    if not best:
        return None
    return ", ".join(sorted(best, key=lambda s: abs(len(s) - len(value)))[:3])


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_kg_tools(
    driver: Neo4jHandler,
    store: ResultStore,
    thread_id: str,
) -> list[StructuredTool]:
    """Build the four KG tools bound to a driver, result store and thread.

    Cheap to construct; rebuild per request.
    """
    return [
        _build_execute_cypher(driver, store, thread_id),
        _build_read_kg_schema(driver),
        _build_lookup_property(driver),
        _build_validate_cypher(driver),
    ]
