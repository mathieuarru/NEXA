"""Deterministic Cypher rewrites applied after query generation.

:func:`ensure_entities_in_return` aliases anonymous patterns and makes the query
return whole nodes and relationships. The trailing-LIMIT helpers support the
completeness guard.
"""

import re
from typing import Any, Dict, List, Optional

from nexa.models.enums import Status
from nexa.utils.logger import get_logger

logger = get_logger(__name__)

# Word boundaries rather than spaces, so a UNION on its own line is also caught.
_UNION_RE = re.compile(r"\bUNION\b", re.IGNORECASE)
_RETURN_RE = re.compile(r"\bRETURN\b", re.IGNORECASE)
_SUFFIX_RE = re.compile(r"\b(ORDER\s+BY|LIMIT|SKIP)\b", re.IGNORECASE)
# count() can be recovered from the returned entities; value aggregates cannot.
_COUNT_AGG_RE = re.compile(r"\bcount\s*\(", re.IGNORECASE)
_VALUE_AGG_RE = re.compile(
    r"\b(collect|sum|avg|min|max|stdev|stdevp|percentile\w*)\s*\(", re.IGNORECASE
)
_DISTINCT_RE = re.compile(r"^\s*DISTINCT\b", re.IGNORECASE)

# Function and keyword names the alias regexes can match (e.g. `count(d)`) but
# that are never aliases.
_NOT_ALIASES = {
    "count", "collect", "sum", "avg", "min", "max", "stdev", "stdevp",
    "size", "length", "type", "id", "labels", "keys", "properties", "exists",
    "coalesce", "tolower", "toupper", "tostring", "tointeger", "tofloat",
    "toboolean", "trim", "ltrim", "rtrim", "replace", "split", "substring",
    "abs", "ceil", "floor", "round", "sqrt", "distinct", "case", "when",
    "then", "else", "end", "and", "or", "not", "xor", "in", "is", "null",
    "true", "false", "where", "with", "return", "match", "optional", "call",
    "yield", "unwind", "as", "order", "by", "limit", "skip", "union", "all",
    "any", "none", "single", "reduce", "range", "head", "last", "tail",
    "nodes", "relationships", "startnode", "endnode", "apoc", "date",
    "datetime", "duration", "point", "rand", "sign", "log", "exp",
}


# Integer `LIMIT n` at the very end of the query, i.e. the one bounding the whole
# result. Limits inside subqueries or WITH stages, and `LIMIT $param`, do not match.
_TRAILING_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\s*;?\s*$", re.IGNORECASE)


def parse_trailing_limit(query: str) -> Optional[int]:
    """Return ``n`` from the query's trailing ``LIMIT n``, or None.

    None means completeness is unknown (no limit, a parameter, or a nested
    limit); callers must never read it as "complete".
    """
    if not query:
        return None
    match = _TRAILING_LIMIT_RE.search(query)
    return int(match.group(1)) if match else None


def replace_trailing_limit(query: str, new_limit: int) -> Optional[str]:
    """Return ``query`` with its trailing ``LIMIT n`` set to ``new_limit``.

    Returns None when :func:`parse_trailing_limit` finds no limit to replace.
    """
    if parse_trailing_limit(query) is None:
        return None
    return _TRAILING_LIMIT_RE.sub(f"LIMIT {int(new_limit)}", query, count=1)


def _collect_aliases(query: str) -> List[str]:
    """Alias names declared in node/relationship patterns, function names excluded."""
    node_aliases: List[str] = []
    rel_aliases: List[str] = []
    for i, part in enumerate(re.split(r"(\".*?\"|'.*?')", query)):
        if i % 2 == 1:  # inside a string literal
            continue
        node_aliases.extend(re.findall(r"\(\s*([a-zA-Z_][a-zA-Z0-9_]*)", part))
        rel_aliases.extend(re.findall(r"\[\s*([a-zA-Z_][a-zA-Z0-9_]*)", part))
    return sorted({
        a for a in node_aliases + rel_aliases if a.lower() not in _NOT_ALIASES
    })


def _add_missing_aliases(query: str) -> str:
    """Give every anonymous `(:Label)` / `[:REL]` pattern a unique alias."""
    parts = re.split(r"(\".*?\"|'.*?')", query)
    out: List[str] = []
    rel_counter = node_counter = 0

    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
            continue

        def rel_alias_adder(_match):
            nonlocal rel_counter
            alias = f"rel{rel_counter}"
            rel_counter += 1
            return f"[{alias}:"

        def node_alias_adder(_match):
            nonlocal node_counter
            alias = f"node{node_counter}"
            node_counter += 1
            return f"({alias}:"

        part = re.sub(r"\[\s*:", rel_alias_adder, part)
        part = re.sub(r"\(\s*:", node_alias_adder, part)
        out.append(part)

    return "".join(out)


def ensure_entities_in_return(state: Dict[str, Any]) -> Dict[str, Any]:
    """Alias every pattern element and make sure the queried entities are returned.

    - ``UNION`` queries are left untouched; ``RETURN *`` only gets pattern aliases.
    - Value aggregates (``collect``, ``sum``, ...) are kept as written, since they
      cannot be recomputed from the entities.
    - ``count`` or ``DISTINCT`` projections are replaced by the entity aliases,
      because adding variables to an aggregate changes its grouping. A suffix
      that refers to a removed alias is dropped.
    - Otherwise the missing aliases are appended to the projection.
    """
    logger.info("Ensuring all entities have aliases and are in RETURN clause...")
    cypher_query = state.get("cypher_queries", [])[-1]

    if _UNION_RE.search(cypher_query):
        logger.warning("Query contains 'UNION', skipping alias modification to avoid breaking it.")
        return {}

    modified_query = _add_missing_aliases(cypher_query)
    aliases = _collect_aliases(modified_query)

    if not aliases:
        logger.warning("No aliases found in the query. Returning it as is.")
        return {"cypher_queries": [modified_query]}

    return_match = _RETURN_RE.search(modified_query)
    if not return_match:
        new_query = f"{modified_query.strip()} RETURN {', '.join(aliases)}"
        return _result(cypher_query, new_query)

    head = modified_query[: return_match.start()].rstrip()
    tail = modified_query[return_match.end():]

    # Split the RETURN body from any ORDER BY / LIMIT / SKIP suffix.
    suffix_match = _SUFFIX_RE.search(tail)
    projection = (tail[: suffix_match.start()] if suffix_match else tail).strip()
    suffix = tail[suffix_match.start():] if suffix_match else ""

    if projection == "*":
        return _result(cypher_query, f"{head} RETURN *{(' ' + suffix.strip()) if suffix else ''}")

    if _VALUE_AGG_RE.search(projection):
        logger.info("Projection carries a value aggregate; preserving it as written.")
        return _result(cypher_query, modified_query)

    if _COUNT_AGG_RE.search(projection) or _DISTINCT_RE.search(projection):
        # Neo4j rejects an ORDER BY on an alias that is no longer returned.
        new_projection = ", ".join(aliases)
        if suffix:
            defined = set(aliases)
            referenced = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", suffix))
            referenced -= {"ORDER", "BY", "LIMIT", "SKIP", "ASC", "DESC"}
            referenced = {r for r in referenced if r.lower() not in _NOT_ALIASES}
            if referenced and not referenced.issubset(defined):
                logger.info(
                    "Dropping '%s': it references %s, which the rewritten RETURN no longer defines.",
                    suffix.strip(), sorted(referenced - defined),
                )
                suffix = ""
        new_query = f"{head} RETURN {new_projection}{(' ' + suffix.strip()) if suffix else ''}"
        return _result(cypher_query, new_query)

    # Plain projection: append the aliases it does not already return.
    existing_terms = {t.strip() for t in projection.split(",")}
    missing = [a for a in aliases if a not in existing_terms]
    new_projection = ", ".join([projection] + missing) if projection else ", ".join(aliases)
    new_query = f"{head} RETURN {new_projection}{(' ' + suffix.strip()) if suffix else ''}"
    return _result(cypher_query, new_query)


def _result(original: str, new_query: str) -> Dict[str, Any]:
    logger.info(f"Original query: {original}")
    logger.info(f"Modified query: {new_query}")
    return {
        "cypher_queries": [new_query],
        "thinking": "Ensuring all entities have aliases and are in RETURN clause.",
        "status": Status.ENSURE_ENTITIES_IN_RETURN,
    }
