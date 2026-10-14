"""Graph-element identity: no live database or LLM required.

`execute_query_as_elements` reserves `id`, `element_id` and `label` on nodes
(and `source_node`/`target_node` on edges). A stored KG property with one of
these names must not replace them: PrimeKG reuses `id` values across entity
types, so distinct nodes would merge.
"""
from nexa.core.nexa_agent import NexaAgent
from nexa.knowledge_graph.neo4j_handler import Neo4jHandler


class FakeNode:
    """Stands in for `neo4j.graph.Node` (mapping access + labels + element_id)."""

    def __init__(self, element_id, labels, **props):
        self.element_id = element_id
        self.labels = labels
        self._props = props

    def __iter__(self):
        return iter(self._props)

    def __getitem__(self, key):
        return self._props[key]

    def keys(self):
        return self._props.keys()


class FakeRel:
    def __init__(self, element_id, rel_type, start, end, **props):
        self.element_id = element_id
        self.type = rel_type
        self.start_node = start
        self.end_node = end
        self._props = props

    def __iter__(self):
        return iter(self._props)

    def __getitem__(self, key):
        return self._props[key]

    def keys(self):
        return self._props.keys()


def build(records):
    """Run the element builder over fake records, bypassing the driver."""
    import nexa.knowledge_graph.neo4j_handler as handler_module

    # __new__ skips __init__, so no Bolt connection is opened.
    handler = Neo4jHandler.__new__(Neo4jHandler)
    handler_module.Node = FakeNode
    handler_module.Relationship = FakeRel
    try:
        return handler._build_cytoscape_elements(records)
    finally:
        from neo4j.graph import Node, Relationship

        handler_module.Node = Node
        handler_module.Relationship = Relationship


def test_colliding_id_property_keeps_nodes_distinct():
    """PrimeKG's real collision: MONDO 1945 and HPO 1945 are different entities."""
    disease = FakeNode("4:db:1", ["Disease"], id="1945", name="postencephalitic Parkinson disease")
    fever = FakeNode("4:db:2", ["EffectPhenotype"], id="1945", name="Fever")
    rel = FakeRel("5:db:9", "DISEASE_PHENOTYPE_POSITIVE", disease, fever)

    elements = build([{"d": disease, "r": rel, "p": fever}])
    nodes, edges = elements["nodes"], elements["edges"]

    assert len(nodes) == 2
    assert len({n["id"] for n in nodes}) == 2, "the shared `id` property collapsed two entities"
    assert edges[0]["source_node"] != edges[0]["target_node"], "a real relation became a self-loop"

    by_alias = {n["id"]: n for n in nodes}
    assert by_alias[edges[0]["source_node"]]["name"] == "postencephalitic Parkinson disease"
    assert by_alias[edges[0]["target_node"]]["name"] == "Fever"


def test_reserved_keys_survive_and_properties_are_preserved():
    node = FakeNode("4:db:1", ["Disease"], id="1945", label="not-a-label", name="X")
    elements = build([{"d": node}])
    emitted = elements["nodes"][0]

    assert emitted["label"] == "Disease", "a stored `label` property overwrote the node label"
    assert emitted["id"] != "1945"
    assert emitted["element_id"] == "4:db:1"
    # Displaced properties stay readable under the `<key>_property` convention.
    assert emitted["id_property"] == "1945"
    assert emitted["label_property"] == "not-a-label"
    assert emitted["name"] == "X"


def test_edge_reserved_keys_survive():
    """Reserved edge keys win over properties, and Hetionet's `source` property is kept."""
    a = FakeNode("4:db:1", ["Compound"], name="Clomifene")
    b = FakeNode("4:db:2", ["Disease"], name="breast cancer")
    rel = FakeRel("5:db:9", "TREATS", a, b, source="DrugBank", label="nope", source_node="spoof")

    edge = build([{"a": a, "r": rel, "b": b}])["edges"][0]

    assert edge["label"] == "TREATS"
    assert edge["source_node"] == build([{"a": a}])["nodes"][0]["id"]
    assert edge["label_property"] == "nope"
    assert edge["source_node_property"] == "spoof"
    assert edge["source"] == "DrugBank", "Hetionet's `source` property must reach the graph view"


def test_endpoints_resolve_to_exactly_one_node():
    nodes = [FakeNode(f"4:db:{i}", ["Gene"], id="same", name=f"G{i}") for i in range(5)]
    rels = [FakeRel(f"5:db:{i}", "INTERACTS", nodes[i], nodes[i + 1]) for i in range(4)]
    records = [{"a": nodes[i], "r": rels[i], "b": nodes[i + 1]} for i in range(4)]

    elements = build(records)
    index = {}
    for n in elements["nodes"]:
        index.setdefault(n["id"], []).append(n)

    assert len(elements["nodes"]) == 5
    for edge in elements["edges"]:
        assert len(index[edge["source_node"]]) == 1
        assert len(index[edge["target_node"]]) == 1


def test_element_id_is_not_rendered_to_the_llm():
    """`element_id` is kept on elements but not rendered into the LLM prompt."""
    node = FakeNode("4:db:1", ["Disease"], id="1945", name="Parkinson disease")
    kg_data = {"data": build([{"d": node}]), "cypher_query": "MATCH (d:Disease) RETURN d"}

    rendered = NexaAgent._kg_level_0_full(None, kg_data)

    assert "4:db:1" not in rendered
    assert "element_id" not in rendered
    assert "Parkinson disease" in rendered
    assert "1945" in rendered, "the KG's own `id` value is real data and must stay visible"
