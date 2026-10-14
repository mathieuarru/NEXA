"""Rewrite a projection Cypher query so it returns graph entities.

Used by the graph view: a query such as `RETURN g.symbol, d.name` is rewritten
to return the underlying nodes and relationships, which
`Neo4jHandler.execute_query_as_elements` turns into cytoscape elements.

Steps:
  1. Alias anonymous relationship patterns (`__r1`, `__r2`, ...).
  2. Alias anonymous node patterns (`__n1`, `__n2`, ...), so every edge keeps
     both endpoints in the view.
  3. Replace the last RETURN with `RETURN DISTINCT <aliases>`, keeping LIMIT and
     dropping ORDER BY.
"""

from __future__ import annotations

import re

from nexa.knowledge_graph.properties_correction import _build_alias_map


# RETURN body, with an optional trailing ORDER BY / SKIP / LIMIT clause.
_RETURN_RE = re.compile(
    r"\bRETURN\b(?P<body>.*?)(?P<tail>\b(?:ORDER\s+BY|SKIP|LIMIT)\b.*)?$",
    flags=re.IGNORECASE | re.DOTALL,
)

_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b", flags=re.IGNORECASE)

# Relationship bracket between dashes, so list literals and index access are
# left alone. Groups: prefix dash, bracket content, suffix dash.
_REL_BRACKET_RE = re.compile(r"(<-|-)\[([^\]]*)\](->|-)")

# Content of a node pattern: optional alias, labels and property map. Function
# calls, grouping and other expressions do not match.
_NODE_CONTENT_RE = re.compile(
    r"^\s*"
    r"(?P<alias>\w*)\s*"
    r"(?P<labels>(?::\s*\w+)+)?\s*"
    r"(?P<props>\{[^{}]*\})?"
    r"\s*$"
)

# LIMIT used when the query has none, to keep the graph view small.
_DEFAULT_GRAPH_LIMIT = 200


def rewrite_for_graph(query: str, max_limit: int = _DEFAULT_GRAPH_LIMIT) -> tuple[str, list[str]]:
    """Rewrite a projection query to return every node and relationship alias.

    Anonymous patterns are aliased first. ``max_limit`` is used as LIMIT when
    the query has none. Returns ``(rewritten_query, aliases)``.

    Raises:
        ValueError: if there is nothing to visualize or no usable RETURN clause.
    """
    if not query or not query.strip():
        raise ValueError("empty query")

    # Existing aliases, so generated ones cannot collide with them.
    initial = _build_alias_map(query)
    existing_aliases: set[str] = set(initial.keys())

    # Both helpers add their generated aliases to existing_aliases.
    query_w_rels, auto_rel_aliases = _alias_anonymous_rels(query, existing_aliases)
    query_w_aliases, auto_node_aliases = _alias_anonymous_nodes(query_w_rels, existing_aliases)

    # _build_alias_map only sees aliases followed by a label or type; append
    # the generated ones it misses.
    full_alias_map = _build_alias_map(query_w_aliases)
    aliases: list[str] = list(full_alias_map.keys())
    for a in auto_rel_aliases + auto_node_aliases:
        if a not in aliases:
            aliases.append(a)

    if not aliases:
        raise ValueError(
            "no entity aliases found in query; nothing to visualize. "
            "Add an alias to your MATCH (e.g. `MATCH (g:Gene)` rather than `MATCH (:Gene)`)."
        )

    # Only the final RETURN: earlier ones may belong to CALL { ... } subqueries.
    last_return_idx = _find_last_return(query_w_aliases)
    if last_return_idx < 0:
        raise ValueError("no RETURN clause found in query")

    before_return = query_w_aliases[:last_return_idx]
    after_return = query_w_aliases[last_return_idx:]

    m = _RETURN_RE.search(after_return)
    if not m:
        raise ValueError("could not parse RETURN clause")

    # Keep LIMIT; drop ORDER BY, which may reference removed projections.
    tail_raw = m.group("tail") or ""
    limit_match = _LIMIT_RE.search(tail_raw)
    limit_clause = limit_match.group(0) if limit_match else f"LIMIT {max_limit}"

    new_return = f"RETURN DISTINCT {', '.join(aliases)} {limit_clause}"
    rewritten = (before_return + new_return).strip()
    return rewritten, aliases


def _alias_anonymous_rels(query: str, existing_aliases: set[str]) -> tuple[str, list[str]]:
    """Alias anonymous relationship patterns (`-[]-`, `-[:REL]->`, `-[*1..3]-`).

    Adds the generated names to ``existing_aliases`` and returns
    ``(new_query, new_aliases)`` in source order.
    """
    auto: list[str] = []
    counter = [1]

    def _next_alias() -> str:
        while True:
            candidate = f"__r{counter[0]}"
            counter[0] += 1
            if candidate not in existing_aliases:
                existing_aliases.add(candidate)
                auto.append(candidate)
                return candidate

    def _replace(match: "re.Match") -> str:
        prefix, content, suffix = match.group(1), match.group(2), match.group(3)
        stripped = content.lstrip()
        if not stripped:
            # Empty: `-[]-` becomes `-[__r1]-`.
            return f"{prefix}[{_next_alias()}]{suffix}"
        first = stripped[0]
        if first == ":" or first == "*":
            # Anonymous typed (`-[:Rel]-`) or variable-length (`-[*1..3]-`)
            return f"{prefix}[{_next_alias()}{stripped}]{suffix}"
        # Already aliased (`[r]`, `[r:Rel]`, `[r*1..3]`).
        return match.group(0)

    new_query = _REL_BRACKET_RE.sub(_replace, query)
    return new_query, auto


def _alias_anonymous_nodes(query: str, existing_aliases: set[str]) -> tuple[str, list[str]]:
    """Alias anonymous node patterns (`()`, `(:Label)`, `(:Label {...})`).

    Parentheses that are not node patterns are left unchanged. Adds the
    generated names to ``existing_aliases`` and returns
    ``(new_query, new_aliases)``.
    """
    auto: list[str] = []
    counter = [1]

    def _next_alias() -> str:
        while True:
            candidate = f"__n{counter[0]}"
            counter[0] += 1
            if candidate not in existing_aliases:
                existing_aliases.add(candidate)
                auto.append(candidate)
                return candidate

    def _replace(match: "re.Match") -> str:
        content = match.group(1)
        m = _NODE_CONTENT_RE.match(content)
        if not m:
            # Not a node pattern (function call, grouping, ...).
            return match.group(0)
        alias_part = (m.group("alias") or "").strip()
        if alias_part:
            return match.group(0)
        labels_part = (m.group("labels") or "").strip()
        props_part = (m.group("props") or "").strip()
        rebuilt = _next_alias()
        if labels_part:
            rebuilt += labels_part
        if props_part:
            rebuilt += " " + props_part
        return f"({rebuilt})"

    # Innermost parentheses only, which is the granularity of node patterns.
    new_query = re.sub(r"\(([^()]*)\)", _replace, query)
    return new_query, auto


def _find_last_return(query: str) -> int:
    """Return the index of the last `RETURN` keyword, or -1 if there is none."""
    last = -1
    for m in re.finditer(r"\bRETURN\b", query, flags=re.IGNORECASE):
        last = m.start()
    return last
