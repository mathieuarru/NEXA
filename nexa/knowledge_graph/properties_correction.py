"""Property-literal reconciliation for the query-correction step.

Extracts property literals from a generated Cypher query, retrieves candidate
values stored in the graph, and reranks them so a misspelled or reworded
literal can be replaced by the canonical stored value.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List

from nexa.config import params
from nexa.utils.utils import parse_key, tokenize_for_matching
from nexa.utils.logger import get_logger

logger = get_logger(__name__)


############################
# Cypher parser
############################

_NODE_PATTERN_BRACE = re.compile(r'\(\s*(\w*)\s*:\s*(\w+)\s*\{([^}]*)\}\s*\)')
_REL_PATTERN_BRACE = re.compile(r'\[\s*(\w*)\s*:\s*(\w+)\s*\{([^}]*)\}\s*\]')

# Alias patterns without inline properties, used to resolve `alias.prop` in WHERE clauses.
_NODE_ALIAS_PATTERN = re.compile(r'\(\s*(\w+)\s*:\s*(\w+)\b')
_REL_ALIAS_PATTERN = re.compile(r'\[\s*(\w+)\s*:\s*(\w+)\b')

# WHERE alias.prop = "value" (single or double quotes)
_WHERE_EQ_PATTERN = re.compile(
    r'(\w+)\.(\w+)\s*=\s*(?P<q>["\'])(?P<val>.*?)(?P=q)'
)

# WHERE alias.prop IN ["v1", "v2"]
_WHERE_IN_PATTERN = re.compile(
    r'(\w+)\.(\w+)\s+IN\s*\[(?P<items>[^\]]*)\]',
    re.IGNORECASE,
)
_WHERE_IN_ITEM_PATTERN = re.compile(r'(?P<q>["\'])(?P<val>.*?)(?P=q)')


def parse_properties(properties_str):
    """Parse an inline ``key: value, ...`` property string into a dict of unquoted values."""
    properties = {}
    if properties_str:
        properties_pairs = [p.strip() for p in properties_str.split(",")]
        for pair in properties_pairs:
            if ':' in pair:
                key, value = pair.split(":", 1)
                properties[key.strip()] = value.strip().replace('"', "").replace("'", "")
    return properties


def _build_alias_map(query: str) -> Dict[str, tuple]:
    """Map aliases in `(alias:Label)` and `[alias:REL]` patterns to ``(element_type, label_or_rel)``.

    Later occurrences overwrite earlier ones.
    """
    alias_map: Dict[str, tuple] = {}
    for alias, label in _NODE_ALIAS_PATTERN.findall(query):
        if alias:
            alias_map[alias] = ("nodes", label)
    for alias, rel_type in _REL_ALIAS_PATTERN.findall(query):
        if alias:
            alias_map[alias] = ("relationships", rel_type)
    return alias_map


def _add_property_entry(
    suggested_properties: dict, element_type: str, label_or_rel: str,
    prop_name: str, prop_value: str,
) -> None:
    """Add a literal under its ``element_type|label|prop|value`` key, keeping the first occurrence."""
    key = f"{element_type}|{label_or_rel}|{prop_name}|{prop_value}"
    if key not in suggested_properties:
        suggested_properties[key] = {"suggested_values": [{prop_value: []}]}


def parse_cypher_query(query: str) -> dict:
    """Extract property literals from a Cypher query for the correction step.

    Returns a dict keyed by ``"<element_type>|<label_or_rel>|<prop>|<value>"``.
    Recognized forms: inline `{prop: "value"}` maps on nodes and relationships,
    `WHERE alias.prop = "value"`, and `WHERE alias.prop IN [...]` (one entry per
    value). `STARTS WITH`, `ENDS WITH` and `CONTAINS` are skipped on purpose:
    they express a fuzzy match that should not be corrected.
    """
    suggested_properties: dict = {}

    # Inline {prop: value} maps
    for alias, label, properties_str in _NODE_PATTERN_BRACE.findall(query):
        for prop_name, prop_value in parse_properties(properties_str).items():
            _add_property_entry(suggested_properties, "nodes", label, prop_name, prop_value)

    for alias, rel_type, properties_str in _REL_PATTERN_BRACE.findall(query):
        for prop_name, prop_value in parse_properties(properties_str).items():
            _add_property_entry(suggested_properties, "relationships", rel_type, prop_name, prop_value)

    alias_map = _build_alias_map(query)

    # WHERE alias.prop = "value"
    for match in _WHERE_EQ_PATTERN.finditer(query):
        alias, prop_name, _quote, value = match.group(1), match.group(2), match.group("q"), match.group("val")
        if alias in alias_map and value:
            element_type, label_or_rel = alias_map[alias]
            _add_property_entry(suggested_properties, element_type, label_or_rel, prop_name, value)

    # WHERE alias.prop IN [...]
    for match in _WHERE_IN_PATTERN.finditer(query):
        alias, prop_name, items_str = match.group(1), match.group(2), match.group("items")
        if alias not in alias_map:
            continue
        element_type, label_or_rel = alias_map[alias]
        for item_match in _WHERE_IN_ITEM_PATTERN.finditer(items_str):
            value = item_match.group("val")
            if value:
                _add_property_entry(suggested_properties, element_type, label_or_rel, prop_name, value)

    return suggested_properties


############################
# Reranking
############################

def find_top_k_similar_values_legacy(
    used_value: str,
    existing_values: List[str],
    k: int = 10,
) -> List[str]:
    """Return the top ``k`` candidates by SequenceMatcher ratio.

    Used when ``NEXA_PROPERTY_MATCHING`` is ``"legacy"``.
    """
    scored = []
    for candidate in existing_values:
        similarity = SequenceMatcher(None, used_value, candidate).ratio()
        scored.append((candidate, similarity))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored[:k]]


def find_top_k_similar_values(
    used_value: str,
    existing_values: List[str],
    k: int = 10,
) -> List[str]:
    """Return the top ``k`` candidates by a composite case-insensitive score.

    The score is the sum of: +1.0 for an exact match, the SequenceMatcher ratio,
    +0.3 when one string contains the other (if not exact), and 0.2 times the
    token Jaccard similarity. Ties go to the shorter candidate.
    """
    if not existing_values:
        return []

    lower_used = str(used_value).lower()
    used_tokens = set(tokenize_for_matching(used_value))

    scored = []
    for candidate in existing_values:
        cand_str = str(candidate)
        lower_cand = cand_str.lower()

        exact = 1.0 if lower_used == lower_cand else 0.0
        base = SequenceMatcher(None, lower_used, lower_cand).ratio()

        contain = 0.0
        if lower_used and lower_cand and exact == 0.0:
            if lower_used in lower_cand or lower_cand in lower_used:
                contain = 0.30

        tokens_bonus = 0.0
        if used_tokens:
            cand_tokens = set(tokenize_for_matching(cand_str))
            if cand_tokens:
                inter = used_tokens & cand_tokens
                union = used_tokens | cand_tokens
                jaccard = len(inter) / len(union) if union else 0.0
                tokens_bonus = 0.20 * jaccard

        total = exact + base + contain + tokens_bonus
        scored.append((cand_str, total, len(cand_str)))

    scored.sort(key=lambda x: (-x[1], x[2]))
    return [item[0] for item in scored[:k]]


############################
# Orchestration (called by the correction subgraph)
############################

def _retrieve_candidates(driver, element_type: str, label_or_rel: str,
                        property_name: str, query_value: str,
                        recall_limit: int = 50) -> tuple:
    """Fetch candidate values for a property; return ``(candidates, warning)``.

    In ``legacy`` mode every distinct value is fetched. Otherwise the database
    filters candidates first, falling back to every distinct value if none match.
    """
    if params.NEXA_PROPERTY_MATCHING == "legacy":
        if element_type == "nodes":
            return driver.get_property_values_for_node(label_or_rel, property_name)
        return driver.get_property_values_for_relationship(label_or_rel, property_name)

    tokens = tokenize_for_matching(query_value)
    candidates, warning = driver.find_property_value_candidates(
        element_type=element_type,
        label_or_rel=label_or_rel,
        property_name=property_name,
        query_value=query_value,
        tokens=tokens,
        limit=recall_limit,
    )
    if candidates:
        return candidates, warning

    logger.info(
        f"[property_matching] DB-side recall returned no candidates for "
        f"{element_type}|{label_or_rel}|{property_name}|{query_value!r}; "
        f"falling back to pull-all-distinct."
    )
    if element_type == "nodes":
        return driver.get_property_values_for_node(label_or_rel, property_name)
    return driver.get_property_values_for_relationship(label_or_rel, property_name)


def _rerank(used_value: str, existing_values: List[str], k: int) -> List[str]:
    """Rerank candidates with the strategy set by ``NEXA_PROPERTY_MATCHING``."""
    if params.NEXA_PROPERTY_MATCHING == "legacy":
        return find_top_k_similar_values_legacy(used_value, existing_values, k)
    return find_top_k_similar_values(used_value, existing_values, k)


def find_top_properties_for_single_key(
    driver,
    parsed: Dict[tuple, dict],
    k: int = 10
) -> Dict[tuple, dict]:
    """Attach the top ``k`` graph values to the single property in ``parsed``.

    The last suggested value of that property is matched against the graph, and
    the ranked candidates are stored under it. Returns ``parsed``, updated in place.
    """
    if not parsed:
        logger.warning("'parsed' is empty.")
        return parsed

    (triplet_key, content) = next(iter(parsed.items()))
    element_type, label_or_rel, property_name, original_value = parse_key(triplet_key)

    suggested_values_list = content.get("suggested_values", [])
    if not suggested_values_list:
        logger.warning("'suggested_values' is empty, nothing to update.")
        return parsed

    last_suggestion_dict = suggested_values_list[-1]
    if not last_suggestion_dict:
        logger.warning("Last suggestion is empty.")
        return parsed

    suggested_value = list(last_suggestion_dict.keys())[0]

    existing_values, warning = _retrieve_candidates(
        driver=driver,
        element_type=element_type,
        label_or_rel=label_or_rel,
        property_name=property_name,
        query_value=str(suggested_value),
    )

    if warning:
        logger.warning(f"{warning}")
    if not existing_values:
        top_k_list: List[str] = []
    else:
        top_k_list = _rerank(str(suggested_value), existing_values, k)
        logger.info(
            f"[property_matching mode={params.NEXA_PROPERTY_MATCHING}] "
            f"Top {k} for {suggested_value!r}: {top_k_list}"
        )

    last_suggestion_dict[suggested_value] = top_k_list
    return parsed
