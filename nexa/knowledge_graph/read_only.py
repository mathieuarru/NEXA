"""Read-only admission policy for Cypher queries.

A small lexer (not a parser) rejects writes, extra statements and
non-allowlisted procedures or functions before a query reaches Neo4j. It
understands strings, comments and escaped identifiers so these cannot hide
anything. Neo4j's EXPLAIN classification makes the final decision, and
server-side read-only permissions remain required as defence in depth.
"""

from dataclasses import dataclass


POLICY_ERROR_CODE = "Nexa.ReadOnlyViolation"

# Exact names only. Do not add dynamic-Cypher, import, export or administrative
# procedures here, even if a plugin declares them READ.
READ_PROCEDURES = frozenset({
    "db.labels", "db.relationshipTypes", "db.propertyKeys",
    "db.schema.nodeTypeProperties", "db.schema.relTypeProperties",
    "db.schema.visualization", "db.index.fulltext.queryNodes",
    "db.index.fulltext.queryRelationships", "db.index.vector.queryNodes",
    "db.index.vector.queryRelationships", "apoc.meta.stats", "apoc.meta.data",
})
READ_FUNCTIONS = frozenset("""
    abs acos all allReduce any asin atan atan2 avg btrim ceil ceiling char_length
    character_length coalesce collect cos cot count degrees e elementId endNode
    exp floor haversin head id isEmpty isNaN keys labels last left length log
    log10 lower ltrim max min nodes none normalize nullIf percentileCont
    percentileDisc pi properties radians rand range reduce relationships replace
    reverse right round rtrim sign sin single size split sqrt startNode stDev
    stDevP substring sum tail tan timestamp toBoolean toBooleanList toBooleanOrNull
    toFloat toFloatList toFloatOrNull toInteger toIntegerList toIntegerOrNull
    toLower toString toStringList toStringOrNull toUpper trim type upper valueType
    exists shortestPath allShortestPaths date datetime duration localdatetime
    localtime time point apoc.text.replace point.distance point.withinBBox
    duration.between duration.inDays duration.inMonths duration.inSeconds
""".split()) | frozenset(
    f"{kind}.{clock}" for kind in ("date", "datetime", "localdatetime", "localtime", "time")
    for clock in ("realtime", "statement", "transaction", "truncate")
) | {"datetime.fromepoch", "datetime.fromepochmillis"}
READ_FUNCTIONS = frozenset(name.lower() for name in READ_FUNCTIONS)

_STARTS = {"MATCH", "OPTIONAL", "WITH", "UNWIND", "RETURN", "CALL"}
_CLAUSES_WITH_PARENS = {"MATCH", "WHERE", "RETURN", "WITH", "IN", "NOT", "AND", "OR", "XOR", "CALL", "WHEN", "THEN", "ELSE", "DISTINCT", "ON"}
_FORBIDDEN = {
    "CREATE", "MERGE", "DELETE", "DETACH", "NODETACH", "SET", "REMOVE",
    "FOREACH", "DROP", "ALTER", "GRANT", "DENY", "REVOKE", "TERMINATE",
    "START", "STOP", "ENABLE", "DISABLE", "RENAME", "INSERT", "LOAD",
    "USE", "SHOW", "EXPLAIN", "PROFILE", "CYPHER", "TRANSACTIONS",
}


class ReadOnlyViolation(ValueError):
    """Terminal policy error: the query must not be repaired or retried."""


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    start: int


def _tokens(query: str) -> list[_Token]:
    """Split a query into words, quoted values and symbols, skipping comments."""
    tokens = []
    i = 0
    while i < len(query):
        ch = query[i]
        if ch.isspace():
            i += 1
            continue
        if query.startswith("//", i):
            while i < len(query) and query[i] not in "\r\n":
                i += 1
            continue
        if query.startswith("/*", i):
            # Support nested comments; the server still checks actual syntax.
            depth = 1
            i += 2
            while i < len(query) and depth:
                if query.startswith("/*", i):
                    depth += 1
                    i += 2
                elif query.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise ReadOnlyViolation("Unterminated Cypher comment.")
            continue
        start = i
        if ch in "'\"`":
            quote = ch
            i += 1
            value = ""
            while i < len(query):
                if query[i] == "\\":
                    # Rejected rather than risk decoding them differently from Neo4j.
                    if quote == "`":
                        raise ReadOnlyViolation("Backslash escapes in identifiers are not supported.")
                    i += 2
                elif query[i] == quote:
                    if i + 1 < len(query) and query[i + 1] == quote:
                        value += quote
                        i += 2
                    else:
                        i += 1
                        break
                else:
                    value += query[i]
                    i += 1
            else:
                raise ReadOnlyViolation("Unterminated Cypher string or identifier.")
            tokens.append(_Token("quoted" if quote == "`" else "string", value, start))
        elif ch.isalpha() or ch == "_":
            i += 1
            while i < len(query) and (query[i].isalnum() or query[i] == "_"):
                i += 1
            tokens.append(_Token("word", query[start:i], start))
        else:
            tokens.append(_Token("symbol", ch, start))
            i += 1
    return tokens


def admit_read_query(query: str) -> str:
    """Return the query as one admissible read statement, or raise ReadOnlyViolation."""
    if not isinstance(query, str) or not query.strip():
        raise ReadOnlyViolation("A non-empty Cypher query is required.")
    tokens = _tokens(query)
    if tokens and tokens[-1].value == ";" and tokens[-1].kind == "symbol":
        query = query[:tokens[-1].start]
        tokens.pop()
    if not tokens or tokens[0].kind != "word" or tokens[0].value.upper() not in _STARTS:
        raise ReadOnlyViolation("Only retrieval queries (MATCH, WITH, UNWIND, RETURN, CALL) are allowed.")
    for i, token in enumerate(tokens):
        if token.kind == "symbol" and token.value == ";":
            raise ReadOnlyViolation("Only one Cypher statement is allowed.")
        if token.kind not in {"word", "quoted"}:
            continue
        previous = tokens[i - 1].value if i else ""
        upper = token.value.upper()
        if token.kind == "word" and upper in _FORBIDDEN and previous not in {".", ":", "$"}:
            raise ReadOnlyViolation(f"Cypher operation {upper} is not allowed.")
        if token.kind == "word" and upper == "CALL" and previous not in {".", ":", "$"}:
            if i + 1 == len(tokens):
                raise ReadOnlyViolation("Incomplete CALL clause.")
            if tokens[i + 1].value not in {"{", "("}:
                name, _ = _routine_name(tokens, i + 1)
                if name not in READ_PROCEDURES:
                    raise ReadOnlyViolation(f"Procedure {name!r} is not permitted by the read-only policy.")
        # Function calls, qualified or not. `CALL (...) { ... }` is a subquery.
        if previous in {".", "$"}:
            continue
        name, end = _routine_name(tokens, i)
        if end < len(tokens) and tokens[end].value == "(":
            is_call = i > 0 and tokens[i - 1].kind == "word" and previous.upper() == "CALL"
            is_clause = token.kind == "word" and name.upper() in _CLAUSES_WITH_PARENS
            if not is_call and not is_clause and name.lower() not in READ_FUNCTIONS:
                raise ReadOnlyViolation(f"Function {name!r} is not permitted by the read-only policy.")
    return query.strip()


def _routine_name(tokens: list[_Token], start: int) -> tuple[str, int]:
    """Read a dotted name starting at ``start``; return it and the next index."""
    parts = []
    i = start
    while i < len(tokens) and tokens[i].kind in {"word", "quoted"}:
        parts.append(tokens[i].value)
        i += 1
        if i + 1 < len(tokens) and tokens[i].value == ".":
            i += 1
        else:
            break
    return ".".join(parts), i


def policy_notification(notifications):
    """Return the read-only policy notification from ``notifications``, if any."""
    return next((n for n in notifications or [] if isinstance(n, dict) and n.get("code") == POLICY_ERROR_CODE), None)
