"""Read-only query enforcement: no live database or LLM required."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nexa.knowledge_graph.read_only import POLICY_ERROR_CODE, ReadOnlyViolation, admit_read_query
from nexa.knowledge_graph.neo4j_handler import Neo4jHandler
from nexa.knowledge_graph.query_correction import check_attempts, execute_cypher_query
from nexa.knowledge_graph.graph_query import _execute_one_shot, _route_after_one_shot
from nexa.models.enums import Status
from nexa.tools.kg_tools import _build_execute_cypher, _build_validate_cypher


@pytest.mark.parametrize("query", [
    "MATCH (n:Drug) RETURN n LIMIT 2",
    "MATCH (n) RETURN COUNT(n)",
    "MATCH (n) WHERE n.name = $name RETURN n",
    "RETURN 'DELETE; CALL apoc.cypher.doIt(\"CREATE ()\", {})' AS text",
    'RETURN "MERGE (n)" AS text',
    "RETURN 'it\\'s a DELETE' AS text",
    "// DELETE everything\nMATCH (n) RETURN n; // trailing comment",
    "MATCH (n) /* CALL dbms.setConfigValue() */ RETURN n",
    "MATCH (n:`DELETE`) RETURN n.`SET`, n.set",
    "RETURN apoc.text.replace('A-B', '-', ' ') AS value",
    "CALL apoc.meta.stats() YIELD nodeCount RETURN nodeCount",
    "CALL `db`.`labels`() YIELD label RETURN label",
    "CALL { MATCH (n) RETURN n LIMIT 1 } RETURN n",
    "MATCH (n) CALL (n) { RETURN n AS x } RETURN x",
    "RETURN COUNT { MATCH (n) RETURN n } AS c",
    "MATCH (n) WHERE EXISTS { MATCH (n)-->(m) } RETURN n",
    "UNWIND [1,2] AS x RETURN x UNION RETURN 3 AS x",
    "MATCH (n) WHERE (n.name IS NOT NULL) RETURN (n.name)",
])
def test_admits_reads_and_ignores_literals(query):
    assert admit_read_query(query)


@pytest.mark.parametrize("query", [
    "MATCH (n:Drug) DETACH DELETE n",
    "MATCH (n) SET n.name = 'changed' RETURN n",
    "MATCH (n) REMOVE n.name RETURN n",
    "CREATE (:Drug)", "MERGE (:Drug)", "DROP INDEX some_index",
    "CREATE USER reader SET PASSWORD 'password'", "SHOW USERS",
    "CALL dbms.setConfigValue('server.databases.read_only', '')",
    "CALL apoc.cypher.run('MATCH (n) DETACH DELETE n', {})",
    "CALL apoc.cypher.doIt($query, {})",
    "CALL apoc.periodic.iterate($read, $write, {})",
    "CALL apoc.create.node(['Drug'], {})",
    "CALL apoc.load.json('https://example.com')",
    "CALL apoc.export.json.all('/tmp/graph.json', {})",
    "CALL `apoc`.`cypher`.`doIt`('CREATE ()', {})",
    "CALL apoc /* comment */ . cypher . doIt('CREATE ()', {})",
    "CALL { MATCH (n) DELETE n } RETURN 1",
    "CALL () { CREATE (:Drug) } RETURN 1",
    "CALL { RETURN 1 AS n } IN TRANSACTIONS RETURN n",
    "RETURN 1; MATCH (n) DELETE n", "RETURN 1;;",
    "RETURN 1 UNION MATCH (n) SET n.x = 1 RETURN n",
    "MATCH (n) FOREACH (x IN [1] | DELETE n)",
    "LOAD CSV FROM 'file:///data.csv' AS row RETURN row",
    "USE system RETURN 1", "CYPHER runtime=slotted RETURN 1",
    "EXPLAIN MATCH (n) DELETE n", "PROFILE MATCH (n) DELETE n",
    "RETURN apoc.cypher.runFirstColumn('CREATE ()', {}, true)",
    "RETURN customFunction()", "RETURN custom.function()",
    "RETURN `apoc\\u002ecypher`.runFirstColumn($q, {}, true)",
    "RETURN 'unterminated", "RETURN 1 /* unterminated", "", "// only comment",
])
def test_refuses_writes_dynamic_execution_and_admin(query):
    with pytest.raises(ReadOnlyViolation):
        admit_read_query(query)


def handler_fixture(query_type="r", updates=False):
    handler = Neo4jHandler.__new__(Neo4jHandler)
    handler.db_name = "chosen_graph"
    handler.driver = MagicMock()
    session = handler.driver.session.return_value.__enter__.return_value
    tx = session.begin_transaction.return_value.__enter__.return_value
    summary = SimpleNamespace(query_type=query_type, notifications=[], counters=SimpleNamespace(
        contains_updates=updates, contains_system_updates=False,
    ))
    tx.run.return_value.consume.return_value = summary
    tx.run.return_value.data.return_value = [{"answer": 42}]
    return handler, tx


@pytest.mark.parametrize("method", ["execute_query", "execute_query_as_elements"])
def test_both_entry_points_block_before_opening_session(method):
    handler, tx = handler_fixture()
    data, warnings = getattr(handler, method)("MATCH (n) DETACH DELETE n")
    assert data is None and warnings[0]["code"] == POLICY_ERROR_CODE
    handler.driver.session.assert_not_called()
    tx.run.assert_not_called()


@pytest.mark.parametrize("query_type", ["w", "rw", "s", None, "unknown"])
def test_non_read_explain_never_executes_original_query(query_type):
    handler, tx = handler_fixture(query_type)
    data, warnings = handler.execute_query("RETURN 42 AS answer")
    assert data is None and warnings[0]["code"] == POLICY_ERROR_CODE
    assert tx.run.call_count == 1
    assert tx.run.call_args.args[0].startswith("EXPLAIN ")
    tx.commit.assert_not_called()


def test_read_uses_explicit_database_and_same_bound_query_in_transaction():
    handler, tx = handler_fixture()
    data, warnings = handler.execute_query("RETURN $value AS answer;", {"value": 42})
    assert data == [{"answer": 42}] and warnings == []
    handler.driver.session.assert_called_once_with(database="chosen_graph", default_access_mode="READ")
    assert [c.args for c in tx.run.call_args_list] == [
        ("EXPLAIN RETURN $value AS answer", {"value": 42}),
        ("RETURN $value AS answer", {"value": 42}),
    ]
    tx.commit.assert_called_once()


def test_unexpected_updates_do_not_commit():
    handler, tx = handler_fixture(updates=True)
    data, warnings = handler.execute_query("RETURN 42")
    assert data is None and warnings[0]["code"] == POLICY_ERROR_CODE
    tx.commit.assert_not_called()


def test_system_database_is_never_opened():
    handler, tx = handler_fixture()
    handler.db_name = "system"
    assert handler.execute_query("RETURN 1")[1][0]["code"] == POLICY_ERROR_CODE
    handler.driver.session.assert_not_called()


def refused_driver():
    driver = MagicMock()
    refusal = (None, [{"code": POLICY_ERROR_CODE, "description": "Writing is forbidden."}])
    driver.execute_query.return_value = refusal
    driver.execute_query_as_elements.return_value = refusal
    return driver


def test_direct_policy_error_stops_correction_even_at_attempt_limit():
    state = {"cypher_queries": ["MATCH (n) DELETE n"]}
    update = execute_cypher_query(state, refused_driver())
    assert update["status"] == Status.ERROR
    # No attempts_info is needed: policy handling precedes correction counters.
    command = check_attempts(update)
    assert command.goto == "__end__"
    assert command.update["status"] == Status.ERROR


def test_direct_policy_error_does_not_trigger_decomposed_fallback():
    update = _execute_one_shot({"cypher_queries": ["CREATE ()"]}, refused_driver())
    assert update["fallback_triggered"] is False
    assert update["status"] == Status.ERROR
    assert _route_after_one_shot(update) == "done"


def test_flex_does_not_store_refused_query_as_empty_result():
    store = MagicMock()
    payload = _build_execute_cypher(refused_driver(), store, "security-test").invoke({"query": "CREATE ()"})
    assert payload["error_code"] == POLICY_ERROR_CODE
    assert payload["retryable"] is False
    assert payload["result_id"] is None
    store.create.assert_not_called()


def test_optional_validator_also_refuses_writes_before_schema_reads():
    driver = MagicMock()
    result = _build_validate_cypher(driver).invoke({"query": "MATCH (n:Drug) DETACH DELETE n"})
    assert result["valid"] is False
    driver.get_nodes.assert_not_called()


@pytest.mark.parametrize("refused_on_rewrite", [False, True])
def test_graph_view_reports_policy_refusal_without_further_rewrites(monkeypatch, refused_on_rewrite):
    # The API sets up PostgreSQL at import time: stub that before importing it.
    from nexa.core.nexa_agent import NexaAgent
    from langgraph.checkpoint.postgres import PostgresSaver
    from nexa.result_store import ResultStore
    from nexa.config import params
    monkeypatch.setattr(params, "DB_URI", "postgresql://unused/offline-test")
    monkeypatch.setattr(NexaAgent, "setup_postgres_pool", lambda *_: MagicMock())
    monkeypatch.setattr(PostgresSaver, "setup", lambda *_: None)
    monkeypatch.setattr(ResultStore, "setup", lambda *_: None)
    from nexa.api import agent_api
    driver = refused_driver()
    refusal = driver.execute_query_as_elements.return_value
    if refused_on_rewrite:
        driver.execute_query_as_elements.side_effect = [([], []), refusal]
    store = MagicMock()
    store.get.return_value = {"cypher_query": "MATCH (n) RETURN n.name", "thread_id": None}
    monkeypatch.setattr(agent_api, "ResultStore", lambda _: store)
    monkeypatch.setattr(agent_api, "Neo4jHandler", lambda **_: driver)
    rewrite = MagicMock(return_value=("MATCH (n) RETURN n", ["n"]))
    monkeypatch.setattr(agent_api, "rewrite_for_graph", rewrite)
    payload = agent_api.get_result_graph_endpoint("test-result")
    assert payload["warnings"][0]["code"] == POLICY_ERROR_CODE
    assert payload["note"].startswith("Read-only policy:")
    assert driver.execute_query_as_elements.call_count == (2 if refused_on_rewrite else 1)
    assert rewrite.call_count == int(refused_on_rewrite)
    driver.close.assert_called_once()
