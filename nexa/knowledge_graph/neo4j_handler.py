"""Neo4j access layer.

Runs read-only Cypher queries, reads the graph schema, and looks up stored
property values for the query-correction step.
"""

import re
from neo4j import GraphDatabase, READ_ACCESS, exceptions
from neo4j.graph import Node, Relationship

from nexa.utils.logger import get_logger
from nexa.knowledge_graph.read_only import (
    POLICY_ERROR_CODE, ReadOnlyViolation, admit_read_query,
)

logger = get_logger(__name__)


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_for_matching(value: str) -> str:
    """Lowercase, replace non-alphanumeric runs with a space, and trim.

    Same normalization as the Cypher side of `find_property_value_candidates`.
    """
    if not value:
        return ""
    return _NORMALIZE_RE.sub(" ", str(value).lower()).strip()


class Neo4jHandler:
    """Neo4j driver wrapper for read-only queries, schema reads and value lookups."""
    def __init__(self, host="localhost", port="7687", username="neo4j", password="password", db_name="neo4j"):
        db_uri = f"bolt://{host}:{port}"
        self.driver = GraphDatabase.driver(db_uri, auth=(username, password))
        # Cache key for the connection; handlers are rebuilt per question.
        self.db_uri = db_uri
        self.db_name = db_name

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    def execute_query(self, query, parameters=None):
        """Run a read-only query; return ``(rows, warnings)``."""
        return self._execute_read(query, parameters, lambda result: result.data())


    def execute_query_as_elements(self, query, parameters=None):
        """Run a read-only query; return ``(elements, warnings)``.

        ``elements`` is ``{"nodes": [...], "edges": [...]}`` when the result holds
        graph entities, else a list of row dicts.
        """
        return self._execute_read(query, parameters, self._build_cytoscape_elements)

    def _execute_read(self, query, parameters, convert):
        """Read-only gate shared by every query path.

        The query must pass ``admit_read_query``, then be classified read-only
        (``query_type == 'r'``) by EXPLAIN in the same transaction before it runs.
        ``READ_ACCESS`` only affects routing, not authorization. Errors are
        returned as notifications; nothing is retried.
        """
        try:
            query = admit_read_query(query)
            database = self.db_name or "neo4j"
            if database.lower() == "system":
                raise ReadOnlyViolation("NEXA cannot query the system database.")
            bound = dict(parameters or {})
            with self.driver.session(database=database, default_access_mode=READ_ACCESS) as session:
                with session.begin_transaction() as tx:
                    summary = tx.run("EXPLAIN " + query, bound).consume()
                    if summary.query_type != "r":
                        raise ReadOnlyViolation(
                            f"Neo4j classified this query as {summary.query_type!r}; only read-only queries are allowed."
                        )
                    result = tx.run(query, bound)
                    data = convert(result)
                    summary = result.consume()
                    if summary.query_type != "r" or summary.counters.contains_updates or summary.counters.contains_system_updates:
                        raise ReadOnlyViolation("Unexpected write classification; the transaction was rolled back.")
                    warnings = [
                        {"code": n.get("code"), "description": n.get("description"), "title": n.get("title")}
                        for n in summary.notifications or [] if n.get("severity") == "WARNING"
                    ]
                    tx.commit()
                    return data, warnings
        except ReadOnlyViolation as e:
            return None, [{"code": POLICY_ERROR_CODE, "description": str(e)}]
        except exceptions.Neo4jError as e:
            # Permission and read-only errors are terminal, not for the correction loop.
            code = getattr(e, "code", "") or ""
            policy_error = (
                "Security.Forbidden" in code or "ReadOnly" in code
                or "ForbiddenOnReadOnlyDatabase" in code
                or "AccessMode" in code
            )
            return None, [{"code": POLICY_ERROR_CODE if policy_error else "Neo4jError", "description": str(e)}]

    def _build_cytoscape_elements(self, result):
        """Convert records to cytoscape nodes and edges, or to row dicts if they hold no graph entities."""
        records = list(result)
        if not records:
            return {"nodes": [], "edges": []}

        has_graph_elements = any(isinstance(value, (Node, Relationship)) for value in records[0].values())

        if not has_graph_elements:
            return [dict(record) for record in records]

        # Short aliases (n0, n1, ...) join edges to nodes. They derive from
        # element_id, never from a stored property, which can repeat across
        # entities (e.g. PrimeKG's `id`). Valid only within this result set.
        aliases: dict[str, str] = {}

        def alias_for(entity) -> str:
            key = entity.element_id
            if key not in aliases:
                aliases[key] = f"n{len(aliases)}"
            return aliases[key]

        nodes_dict = {}
        edges_dict = {}

        for record in records:
            for alias, value in record.items():
                if isinstance(value, Node):
                    node_alias = alias_for(value)
                    if node_alias in nodes_dict:
                        continue
                    node_labels = list(value.labels)
                    nodes_dict[node_alias] = {
                        # Stored properties first; reserved names become `<key>_property`.
                        **self._namespace_properties(value, self._RESERVED_NODE_KEYS),
                        "id": node_alias,
                        "element_id": value.element_id,
                        "label": node_labels[0] if node_labels else "UNKNOWN",
                    }

                elif isinstance(value, Relationship):
                    edge_alias = alias_for(value)
                    if edge_alias in edges_dict:
                        continue
                    edges_dict[edge_alias] = {
                        **self._namespace_properties(value, self._RESERVED_EDGE_KEYS),
                        "id": edge_alias,
                        "element_id": value.element_id,
                        "label": value.type,
                        "source_node": alias_for(value.start_node),
                        "target_node": alias_for(value.end_node),
                    }

        return {
            "nodes": list(nodes_dict.values()),
            "edges": list(edges_dict.values())
        }

    # Keys owned by the element format. Stored properties with these names are
    # kept under `<key>_property` (e.g. PrimeKG's node `id`).
    _RESERVED_NODE_KEYS = frozenset({"id", "element_id", "label"})
    _RESERVED_EDGE_KEYS = frozenset({"id", "element_id", "label", "source_node", "target_node"})

    @staticmethod
    def _namespace_properties(entity, reserved: frozenset) -> dict:
        """Return the entity's properties with reserved names moved to `<key>_property`."""
        return {
            f"{key}_property" if key in reserved else key: value
            for key, value in dict(entity).items()
        }

    def get_nodes(self) -> list:
        """Return each node label with the distinct property keys observed on it."""
        nodes_query = """
            MATCH (n)
            UNWIND labels(n) AS label
            UNWIND keys(n) AS prop
            WITH label, collect(DISTINCT prop) AS properties
            RETURN {labels: label, properties: properties} AS node
        """
        result, _ = self.execute_query(nodes_query)
        return result

    def get_relationships(self) -> list:
        """Return the distinct (source label, relationship type, target label) triples in the graph."""
        rel_query = """
            MATCH (a)-[r]->(b)
            RETURN DISTINCT {source: labels(a)[0], relationship: type(r), target: labels(b)[0]} AS relationship
        """
        result, _ = self.execute_query(rel_query)
        return [{"relationship": r["relationship"]["relationship"],
                "source": r["relationship"]["source"],
                "target": r["relationship"]["target"]} for r in result]

    def schema_fingerprint(self):
        """Return a cheap schema signature for cache invalidation, or None.

        The tuple ``(node count, relationship count, labels, relationship types)``
        changes whenever nodes, relationships, labels or relationship types are
        added or removed, and is much cheaper than the full schema scans.
        """
        query = """
        CALL apoc.meta.stats() YIELD nodeCount, relCount
        CALL db.labels() YIELD label
        WITH nodeCount, relCount, collect(label) AS labels
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN nodeCount, relCount, labels, collect(relationshipType) AS types
        """
        data, _ = self.execute_query(query)
        if not data:
            return None
        row = data[0]
        return (
            row.get("nodeCount"),
            row.get("relCount"),
            tuple(sorted(row.get("labels") or [])),
            tuple(sorted(row.get("types") or [])),
        )

    def get_relationships_properties(self) -> list:
        """Return each relationship type with its property keys, via ``apoc.meta.data()``."""
        rel_properties_query = """
        CALL apoc.meta.data()
        YIELD label, other, elementType, type, property
        WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
        WITH label AS nodeLabels, collect(property) AS properties
        RETURN {type: nodeLabels, properties: properties} AS relationship
        """
        result, _ = self.execute_query(rel_properties_query)
        return result

    def get_property_values_for_node(self, label, property_name) -> list:
        """Return ``(values, warning)`` with the distinct values of a node property."""
        query = f"""
        MATCH (n:{label})
        WHERE n.{property_name} IS NOT NULL
        RETURN DISTINCT n.{property_name} AS value
        """
        result, warning = self.execute_query(query)
        return [record["value"] for record in result], warning

    def get_property_values_for_relationship(self, rel_type, property_name) -> list:
        """Return ``(values, warning)`` with the distinct values of a relationship property."""
        query = f"""
        MATCH ()-[r:{rel_type}]->()
        WHERE r.{property_name} IS NOT NULL
        RETURN DISTINCT r.{property_name} AS value
        """
        result, warning = self.execute_query(query)
        return [record["value"] for record in result], warning

    def find_property_value_candidates(
        self,
        element_type: str,
        label_or_rel: str,
        property_name: str,
        query_value: str,
        tokens: list = None,
        limit: int = 50,
    ):
        """Return ``(values, warning)``: stored values that fuzzily match ``query_value``.

        Candidates are scored in Cypher so that ``LIMIT`` keeps the best ones:
          4: exact match (case- and punctuation-insensitive)
          3: the candidate contains the query
          2: the query contains the candidate
          1: the candidate contains one of ``tokens``
        Ties go to the shorter candidate; the caller reranks the result.
        ``label_or_rel`` and ``property_name`` are interpolated into the query,
        so they must come from the schema.
        """
        lower_query = str(query_value).lower()
        norm_query = _normalize_for_matching(query_value)
        params = {
            "lower_query": lower_query,
            "norm_query": norm_query,
            "tokens": [t for t in (tokens or []) if t],
            "limit": int(limit),
        }

        # Levels 4 to 2 also compare the punctuation-insensitive norm_value.
        score_expr = (
            "CASE "
            "WHEN lower_value = $lower_query OR norm_value = $norm_query THEN 4 "
            "WHEN lower_value CONTAINS $lower_query OR norm_value CONTAINS $norm_query THEN 3 "
            "WHEN $lower_query CONTAINS lower_value OR $norm_query CONTAINS norm_value THEN 2 "
            "WHEN ANY(tok IN $tokens WHERE lower_value CONTAINS tok) THEN 1 "
            "ELSE 0 END"
        )

        # Requires APOC. Without it the query fails, and
        # properties_correction._retrieve_candidates falls back to every distinct value.
        norm_expr_node = (
            "trim(apoc.text.replace(toLower(toString(n.`{prop}`)), '[^a-z0-9]+', ' '))"
        ).format(prop=property_name)
        norm_expr_rel = (
            "trim(apoc.text.replace(toLower(toString(r.`{prop}`)), '[^a-z0-9]+', ' '))"
        ).format(prop=property_name)

        if element_type == "nodes":
            cypher = (
                f"MATCH (n:`{label_or_rel}`) "
                f"WHERE n.`{property_name}` IS NOT NULL "
                f"WITH n.`{property_name}` AS value, "
                f"     toLower(toString(n.`{property_name}`)) AS lower_value, "
                f"     {norm_expr_node} AS norm_value "
                f"WITH value, lower_value, norm_value, {score_expr} AS score "
                f"WHERE score > 0 "
                f"RETURN DISTINCT value, score, size(lower_value) AS len "
                f"ORDER BY score DESC, len ASC "
                f"LIMIT $limit"
            )
        elif element_type == "relationships":
            cypher = (
                f"MATCH ()-[r:`{label_or_rel}`]->() "
                f"WHERE r.`{property_name}` IS NOT NULL "
                f"WITH r.`{property_name}` AS value, "
                f"     toLower(toString(r.`{property_name}`)) AS lower_value, "
                f"     {norm_expr_rel} AS norm_value "
                f"WITH value, lower_value, norm_value, {score_expr} AS score "
                f"WHERE score > 0 "
                f"RETURN DISTINCT value, score, size(lower_value) AS len "
                f"ORDER BY score DESC, len ASC "
                f"LIMIT $limit"
            )
        else:
            return [], f"unknown element_type: {element_type}"

        result, warning = self.execute_query(cypher, parameters=params)
        if result is None:
            return [], warning
        return [record["value"] for record in result], warning

    def find_value_across_properties(
        self,
        element_type: str,
        label_or_rel: str,
        query_value: str,
        tokens: list = None,
        limit: int = 50,
        max_value_len: int = 200,
    ):
        """Find the properties of a label or relationship type whose values match ``query_value``.

        A single scan scores every property with the same levels as
        ``find_property_value_candidates``; values longer than ``max_value_len``
        are skipped. Returns ``([{property, value, score}, ...], warning)``, best first.
        """
        lower_query = str(query_value).lower()
        norm_query = _normalize_for_matching(query_value)
        params = {
            "lower_query": lower_query,
            "norm_query": norm_query,
            "tokens": [t for t in (tokens or []) if t],
            "limit": int(limit),
            "max_len": int(max_value_len),
        }

        score_expr = (
            "CASE "
            "WHEN lower_value = $lower_query OR norm_value = $norm_query THEN 4 "
            "WHEN lower_value CONTAINS $lower_query OR norm_value CONTAINS $norm_query THEN 3 "
            "WHEN $lower_query CONTAINS lower_value OR $norm_query CONTAINS norm_value THEN 2 "
            "WHEN ANY(tok IN $tokens WHERE lower_value CONTAINS tok) THEN 1 "
            "ELSE 0 END"
        )
        norm_value_expr = "trim(apoc.text.replace(toLower(raw), '[^a-z0-9]+', ' '))"

        if element_type == "nodes":
            match_clause = f"MATCH (e:`{label_or_rel}`)"
        elif element_type == "relationships":
            match_clause = f"MATCH ()-[e:`{label_or_rel}`]->()"
        else:
            return [], f"unknown element_type: {element_type}"

        cypher = (
            f"{match_clause} "
            "UNWIND keys(e) AS prop "
            "WITH prop, e[prop] AS rawval "
            "WHERE rawval IS NOT NULL "
            "WITH prop, toString(rawval) AS raw "
            "WHERE size(raw) <= $max_len AND size(trim(raw)) > 0 "
            f"WITH prop, raw AS value, toLower(raw) AS lower_value, {norm_value_expr} AS norm_value "
            f"WITH prop, value, {score_expr} AS score "
            "WHERE score > 0 "
            "RETURN DISTINCT prop AS property, value, score "
            "ORDER BY score DESC, size(value) ASC "
            "LIMIT $limit"
        )

        result, warning = self.execute_query(cypher, parameters=params)
        if result is None:
            return [], warning
        return (
            [
                {"property": r["property"], "value": r["value"], "score": r["score"]}
                for r in result
            ],
            warning,
        )
