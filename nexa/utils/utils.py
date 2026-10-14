"""Shared helpers: tokenization for entity matching, key parsing, and LangGraph state reducers."""
from nexa.models.enums import Status
from typing import Dict, List
import re

_MATCHING_STOPWORDS = frozenset({"and", "or", "of", "the", "a", "an", "in", "on", "to", "for", "by", "with"})
_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9']+")


def tokenize_for_matching(value: str, min_len: int = 3) -> List[str]:
    """Tokenize a string for fuzzy matching against KG entity names.

    Lowercases, splits on non-alphanumerics (keeping inner apostrophes), drops
    tokens shorter than ``min_len`` and stopwords, and deduplicates in order.
    """
    if not value:
        return []
    tokens = _TOKEN_SPLIT_RE.split(str(value).lower())
    seen = set()
    out = []
    for tok in tokens:
        tok = tok.strip("'")
        if len(tok) < min_len or tok in _MATCHING_STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def reduce_list(left: list | None, right: list | None) -> list:
    """Concatenate two lists, treating None as empty."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right

def reduce_status(status1: Status, status2: Status) -> Status:
    """Return the more advanced of two statuses (the later one in Status order)."""

    status_order = list(Status)

    if status1 == status2:
        return status1

    return max(status1, status2, key=status_order.index)

def reduce_parsed_properties(current: Dict, update: Dict) -> Dict:
    """Merge ``update`` into ``current``, or return ``update["value"]`` when ``update["reset"]`` is set."""
    if update.get("reset"):
        return update["value"]
    else:
        return current | update

def parse_key(key: str) -> tuple:
    """Split a ``|``-delimited property key into a tuple."""
    return tuple(key.split("|"))

def reduce_thinking(current: str, update: str) -> str:
    """Join two thinking strings with a newline, skipping empty ones."""
    if not current:
        return update
    if not update:
        return current
    return f"{current}\n{update}"

def reduce_error(current: str | None, update: str | None) -> str | None:
    """Keep the most recent non-None error."""
    return update if update is not None else current

def reduce_dict(current: dict, update: dict) -> dict:
    """Merge two dicts; values from ``update`` win."""
    if not current:
        return update
    if not update:
        return current
    return current | update

def reduce_attempts_info(current: dict, update: dict) -> dict:
    """Merge attempts_info dicts, merging nested dicts one level deep."""
    if not current:
        return update
    if not update:
        return current

    result = current.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = result[key] | value
        else:
            result[key] = value
    return result
