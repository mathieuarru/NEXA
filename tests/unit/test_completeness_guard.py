"""Completeness guard of execute_cypher: no live database required.

Covers the trailing-LIMIT parser (limits it cannot attribute yield
"unassessed", never "complete"), the saturation probe, and how an incomplete
verdict carries into derived results. The stub driver records its queries, so
the tests also check that the non-saturated path costs no extra query.
"""

import pytest

from nexa.config import params
from nexa.tools.kg_tools import _assess_completeness
from nexa.tools.result_tools import _attach_completeness, _derive_completeness
from nexa.utils.cypher_utils import parse_trailing_limit, replace_trailing_limit


# ---------------------------------------------------------------------------
# Trailing-LIMIT parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("MATCH (g:Gene) RETURN g LIMIT 100", 100),
    ("MATCH (g:Gene) RETURN g LIMIT 100;", 100),
    ("MATCH (g:Gene) RETURN g LIMIT 100  \n ", 100),
    ("MATCH (g) RETURN g.geneSymbol AS s ORDER BY s LIMIT 25", 25),
    ("match (g) return g limit 10", 10),
    ("MATCH (g:Gene)-[:BINDS]->(d)\nRETURN d.commonName\nLIMIT 200", 200),
])
def test_parses_outermost_integer_limit(query, expected):
    assert parse_trailing_limit(query) == expected


@pytest.mark.parametrize("query", [
    "MATCH (g:Gene) RETURN g",                                  # no limit at all
    "MATCH (g) RETURN g LIMIT $k",                               # parameterized
    "CALL { MATCH (g) RETURN g LIMIT 10 } RETURN count(*) AS n",  # nested in a subquery
    "MATCH (a) WITH a LIMIT 50 MATCH (a)-->(b) RETURN b",         # staging limit, not the result bound
])
def test_refuses_limits_it_cannot_attribute(query):
    """None means 'completeness unknown', never 'complete'."""
    assert parse_trailing_limit(query) is None
    assert replace_trailing_limit(query, 999) is None


def test_probe_rewrites_only_the_trailing_limit():
    assert replace_trailing_limit("MATCH (g) RETURN g LIMIT 100", 101) == (
        "MATCH (g) RETURN g LIMIT 101"
    )
    # A nested limit must survive untouched while the outer one is bumped.
    query = "CALL { MATCH (g) RETURN g LIMIT 10 } RETURN g LIMIT 5"
    assert replace_trailing_limit(query, 6) == (
        "CALL { MATCH (g) RETURN g LIMIT 10 } RETURN g LIMIT 6"
    )


# ---------------------------------------------------------------------------
# Saturation probe
# ---------------------------------------------------------------------------

class StubDriver:
    """Returns ``rows_available`` rows capped at each query's trailing LIMIT."""

    def __init__(self, rows_available: int, fail: bool = False):
        self.rows_available = rows_available
        self.fail = fail
        self.queries: list[str] = []

    def execute_query(self, query, parameters=None):
        self.queries.append(query)
        if self.fail:
            return None, [{"description": "boom"}]
        limit = parse_trailing_limit(query)
        n = self.rows_available if limit is None else min(limit, self.rows_available)
        return [{"i": i} for i in range(n)], []


@pytest.fixture
def guard_on(monkeypatch):
    monkeypatch.setattr(params, "NEXA_COMPLETENESS_GUARD", True)


def test_guard_off_never_assesses(monkeypatch):
    monkeypatch.setattr(params, "NEXA_COMPLETENESS_GUARD", False)
    driver = StubDriver(rows_available=5000)
    assert _assess_completeness(driver, "MATCH (g) RETURN g LIMIT 100", 100) is None
    assert driver.queries == []  # and costs nothing


def test_under_the_limit_is_complete_without_a_probe(guard_on):
    driver = StubDriver(rows_available=42)
    verdict = _assess_completeness(driver, "MATCH (g) RETURN g LIMIT 100", 42)
    assert verdict == {"complete": True, "limit": 100, "rows_returned": 42}
    assert driver.queries == []  # the limit never bound the result


def test_saturated_with_more_rows_available_is_incomplete(guard_on):
    driver = StubDriver(rows_available=1656)
    verdict = _assess_completeness(driver, "MATCH (g) RETURN g LIMIT 100", 100)
    assert verdict["complete"] is False
    assert verdict["limit"] == 100
    assert verdict["rows_returned"] == 100
    assert verdict["reason"] == "trailing_limit_saturated"
    assert driver.queries == ["MATCH (g) RETURN g LIMIT 101"]


def test_saturated_exactly_at_the_boundary_is_complete(guard_on):
    """Exactly `limit` rows available is complete, since the probe asks for n+1."""
    driver = StubDriver(rows_available=100)
    verdict = _assess_completeness(driver, "MATCH (g) RETURN g LIMIT 100", 100)
    assert verdict == {"complete": True, "limit": 100, "rows_returned": 100}
    assert len(driver.queries) == 1


def test_failed_probe_stays_silent(guard_on):
    """A failed probe gives no verdict."""
    driver = StubDriver(rows_available=1656, fail=True)
    assert _assess_completeness(driver, "MATCH (g) RETURN g LIMIT 100", 100) is None


def test_uninterpretable_limit_is_left_unassessed(guard_on):
    driver = StubDriver(rows_available=1656)
    assert _assess_completeness(driver, "MATCH (g) RETURN g LIMIT $k", 100) is None
    assert driver.queries == []


# ---------------------------------------------------------------------------
# Lineage: a subset of an incomplete set is incomplete
# ---------------------------------------------------------------------------

def test_incomplete_parent_taints_the_derived_result():
    parent = {
        "result_id": "abc",
        "completeness": {"complete": False, "limit": 100, "rows_returned": 100},
    }
    child = _derive_completeness(parent)
    assert child == {
        "complete": False,
        "reason": "derived_from_incomplete",
        "inherited_from": "abc",
    }


def test_complete_parent_yields_complete_child():
    parent = {"result_id": "abc", "completeness": {"complete": True, "limit": 100}}
    assert _derive_completeness(parent) == {"complete": True, "inherited_from": "abc"}


def test_unassessed_parent_yields_unassessed_child():
    """Absence of a verdict must not be upgraded into a completeness claim."""
    assert _derive_completeness({"result_id": "abc"}) is None
    assert _derive_completeness({"result_id": "abc", "completeness": None}) is None


# ---------------------------------------------------------------------------
# Payload surfacing
# ---------------------------------------------------------------------------

def test_unassessed_payload_is_untouched():
    payload = {"result_id": "abc", "row_count": 3}
    assert _attach_completeness(dict(payload), None) == payload


def test_incomplete_payload_carries_a_directive_the_model_cannot_miss():
    payload = _attach_completeness(
        {"result_id": "abc", "row_count": 100},
        {"complete": False, "limit": 100, "reason": "trailing_limit_saturated"},
    )
    assert payload["complete"] is False
    warning = payload["warnings"][0]
    assert warning["code"] == "Nexa.IncompleteResult"
    assert "LIMIT of 100" in warning["description"]
    # The warning must forbid concluding an absence.
    assert "absence" in warning["description"].lower()


def test_derived_incomplete_payload_names_its_parent():
    payload = _attach_completeness(
        {"result_id": "child", "row_count": 0},
        {"complete": False, "reason": "derived_from_incomplete", "inherited_from": "parent-id"},
    )
    assert "parent-id" in payload["warnings"][0]["description"]


def test_complete_payload_gets_no_warning():
    payload = _attach_completeness(
        {"result_id": "abc", "row_count": 42},
        {"complete": True, "limit": 100, "rows_returned": 42},
    )
    assert payload["complete"] is True
    assert "warnings" not in payload
