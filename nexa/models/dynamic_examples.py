"""Schema-grounded few-shot examples for the Direct pipeline prompts.

Builds, for a KG profile, few-shot examples whose labels, relationships,
properties and literal values come from the connected graph, instead of the
generic examples in prompts.py. The modes share the same Cypher and differ only
in how questions are phrased:
    - "template": deterministic, no LLM call.
    - "llm": questions written by an LLM, with template fallback.
The "hardcoded" mode stores only the identifying-property mapping.

Cache layout (one JSON file per profile):

    api_data/kg_examples/<profile>.json
    {
        "schema_fingerprint": "<sha256hex>",
        "mode": "hardcoded" | "template" | "llm",
        "identifying_properties_resolved": {"Gene": "geneSymbol", ...},
        "examples_template": {
            "labels_selection": "...",
            "relationships_selection": "...",
            "query_generation": "...",
            "query_correction": "..."
        },
        "examples_llm": { ... same shape ... }   # only present if mode=="llm"
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from nexa.config import params
from nexa.knowledge_graph.neo4j_handler import Neo4jHandler

logger = logging.getLogger(__name__)

KG_EXAMPLES_DIR = os.path.join(params.KG_PROFILES_DIR, "kg_examples")

IDENTIFYING_PROPERTY_FALLBACKS = [
    "name", "commonName", "common_name","symbol", "geneSymbol", "title",
    "label",
]

MAX_SAMPLED_VALUES_PER_LABEL = 3
RANDOM_SEED = 42

# Examples are emitted in shape order (chain, 3-chain, single hops,
# intersections), so this cut drops the last shapes first.
MAX_QUERY_GENERATION_EXAMPLES = 6


############################
# Public entry points
############################

def build_and_store_examples(
    profile_name: str,
    handler: Neo4jHandler,
    nodes_descriptions: Optional[Dict[str, str]] = None,
    relationships_descriptions: Optional[Dict[str, str]] = None,
    identifying_properties_user: Optional[Dict[str, str]] = None,
    llm: Optional[BaseChatModel] = None,
    mode: str = "template",
) -> Optional[Dict[str, Any]]:
    """Build a profile's examples payload and save it to disk.

    Returns the payload, or None on failure (the prompts then use their default examples).
    """
    try:
        builder = DynamicExampleBuilder(
            handler=handler,
            nodes_descriptions=nodes_descriptions,
            relationships_descriptions=relationships_descriptions,
            identifying_properties_user=identifying_properties_user,
            llm=llm,
            mode=mode,
        )
        payload = builder.build()
        save_examples_to_disk(profile_name, payload)
        logger.info(f"Dynamic examples built for profile '{profile_name}' (mode={mode}).")
        return payload
    except Exception as e:
        logger.error(f"Failed to build dynamic examples for profile '{profile_name}': {e}", exc_info=True)
        return None


def load_examples_for_profile(profile_name: str, mode: str) -> Optional[Dict[str, str]]:
    """Return a profile's examples for ``mode``, keyed by prompt family, or None.

    Families: labels_selection, relationships_selection, query_generation and
    query_correction. None in hardcoded mode or when the file or block is missing.
    """
    if mode == "hardcoded":
        return None

    payload = load_examples_from_disk(profile_name)
    if not payload:
        return None

    key = f"examples_{mode}"
    examples = payload.get(key)
    if not examples:
        logger.info(
            f"Profile '{profile_name}' examples file has no '{key}' entry; "
            "caller will fall back to hardcoded defaults."
        )
        return None
    return examples


def load_identifying_properties_for_profile(profile_name: str) -> Optional[Dict[str, str]]:
    """Return a profile's resolved identifying properties, or None if unavailable.

    Works in every examples mode. Labels without an identifying property are left out.
    """
    payload = load_examples_from_disk(profile_name)
    if not payload:
        return None
    mapping = payload.get("identifying_properties_resolved")
    if not isinstance(mapping, dict):
        return None
    return {label: prop for label, prop in mapping.items() if prop}


def save_examples_to_disk(profile_name: str, payload: Dict[str, Any]) -> None:
    """Write a profile's built few-shot examples payload to its JSON file."""
    os.makedirs(KG_EXAMPLES_DIR, exist_ok=True)
    safe_name = _safe_filename(profile_name)
    path = os.path.join(KG_EXAMPLES_DIR, f"{safe_name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_examples_from_disk(profile_name: str) -> Optional[Dict[str, Any]]:
    """Load a profile's stored few-shot examples payload, or None if absent/unreadable."""
    safe_name = _safe_filename(profile_name)
    path = os.path.join(KG_EXAMPLES_DIR, f"{safe_name}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read examples for profile '{profile_name}': {e}")
        return None


def delete_examples_from_disk(profile_name: str) -> None:
    """Delete a profile's stored few-shot examples file, if it exists."""
    safe_name = _safe_filename(profile_name)
    path = os.path.join(KG_EXAMPLES_DIR, f"{safe_name}.json")
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError as e:
            logger.warning(f"Could not delete examples file for profile '{profile_name}': {e}")


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


############################
# Pydantic helpers (LLM outputs)
############################

class _IdentifyingPropertyItem(BaseModel):
    label: str = Field(description="The node label, exactly as provided.")
    property: Optional[str] = Field(
        description=(
            "The single property name a human would naturally use to refer "
            "to an entity of this label (e.g. 'name', 'geneSymbol'). "
            "Use null when no property fits."
        )
    )


class _IdentifyingPropertiesOutput(BaseModel):
    items: List[_IdentifyingPropertyItem] = Field(
        description="One entry per input label, preserving the input order."
    )


class _NLQuestionOutput(BaseModel):
    question: str = Field(
        description="A short natural-language biomedical question that the given Cypher query would answer."
    )


############################
# Builder
############################

class DynamicExampleBuilder:
    """Build few-shot question/Cypher examples from the connected graph.

    Modes: ``hardcoded`` (identifying properties only), ``template`` (templated
    questions) and ``llm`` (LLM-written questions; requires ``llm``).
    """

    def __init__(
        self,
        handler: Neo4jHandler,
        nodes_descriptions: Optional[Dict[str, str]] = None,
        relationships_descriptions: Optional[Dict[str, str]] = None,
        identifying_properties_user: Optional[Dict[str, str]] = None,
        llm: Optional[BaseChatModel] = None,
        mode: str = "template",
    ):
        if mode not in ("hardcoded", "template", "llm"):
            raise ValueError(f"Unsupported mode: {mode}")
        if mode == "llm" and llm is None:
            raise ValueError("mode='llm' requires a non-null llm argument.")

        self.handler = handler
        self.nodes_descriptions = nodes_descriptions or {}
        self.relationships_descriptions = relationships_descriptions or {}
        self.identifying_properties_user = identifying_properties_user or {}
        self.llm = llm
        self.mode = mode
        self._rng = random.Random(RANDOM_SEED)
        self._llm_question_cache: Dict[str, str] = {}

    def build(self) -> Dict[str, Any]:
        """Build the examples payload from the connected graph.

        Raises RuntimeError if the graph has no nodes or relationships.
        """
        nodes_raw = self.handler.get_nodes() or []
        relationships = self.handler.get_relationships() or []

        if not nodes_raw or not relationships:
            raise RuntimeError("Cannot build examples: the connected graph is empty.")

        # label -> list of property names
        label_properties = self._index_properties_by_label(nodes_raw)

        identifying = self._resolve_identifying_properties(label_properties)

        # Hardcoded mode keeps only the identifying-property mapping, which the
        # generation and correction prompts still use.
        if self.mode == "hardcoded":
            return {
                "schema_fingerprint": self._schema_fingerprint(nodes_raw, relationships),
                "mode": self.mode,
                "identifying_properties_resolved": identifying,
            }

        # Sampled once and shared by every prompt family.
        sampled_values: Dict[str, List[str]] = {}
        for label, prop in identifying.items():
            if prop is None:
                sampled_values[label] = []
                continue
            sampled_values[label] = self._sample_values(label, prop)

        triples = self._diverse_triples(relationships)

        # Each family tracks its own used rels so every block stays internally diverse.
        examples_scaffold = {
            "labels_selection": self._build_labels_selection_examples(triples, identifying, sampled_values, set()),
            "relationships_selection": self._build_relationships_selection_examples(triples, identifying, sampled_values, set()),
            "query_generation": self._build_query_generation_examples(triples, identifying, sampled_values, set()),
            "query_correction": self._build_query_correction_examples(triples, identifying, sampled_values, set()),
        }

        examples_template = {
            family: self._render_block(family, entries, use_llm=False)
            for family, entries in examples_scaffold.items()
        }

        payload: Dict[str, Any] = {
            "schema_fingerprint": self._schema_fingerprint(nodes_raw, relationships),
            "mode": self.mode,
            "identifying_properties_resolved": identifying,
            "examples_template": examples_template,
        }

        if self.mode == "llm":
            payload["examples_llm"] = {
                family: self._render_block(family, entries, use_llm=True)
                for family, entries in examples_scaffold.items()
            }

        return payload

    ############################
    # Schema sampling
    ############################

    def _index_properties_by_label(self, nodes_raw: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Map each node label to its property names."""
        out: Dict[str, List[str]] = {}
        for entry in nodes_raw:
            node = entry.get("node") if isinstance(entry, dict) else None
            if not node:
                continue
            label = node.get("labels")
            if isinstance(label, list):
                label = label[0] if label else None
            if not label:
                continue
            props = node.get("properties") or []
            out.setdefault(label, [])
            for p in props:
                if p not in out[label]:
                    out[label].append(p)
        return out

    def _sample_values(self, label: str, prop: str) -> List[str]:
        """Return up to MAX_SAMPLED_VALUES_PER_LABEL distinct values of `label.prop`."""
        try:
            values, _ = self.handler.get_property_values_for_node(label, prop)
        except Exception as e:
            logger.warning(f"Could not sample values for {label}.{prop}: {e}")
            return []

        clean = []
        for v in values or []:
            if v is None:
                continue
            s = str(v).strip()
            if s and s not in clean:
                clean.append(s)
            if len(clean) >= MAX_SAMPLED_VALUES_PER_LABEL:
                break
        return clean

    ############################
    # Instance grounding
    ############################
    #
    # A schema triple only says that two types can connect. Literals are taken
    # from a real instance of the pattern, so that each example returns rows.

    def _ground(self, pattern: str, returns: str) -> Optional[List[Any]]:
        """Run `MATCH <pattern> RETURN <returns> LIMIT 1`; None if it matches nothing."""
        query = f"MATCH {pattern} RETURN {returns} LIMIT 1"
        try:
            rows, _ = self.handler.execute_query(query)
        except Exception as e:
            logger.warning(f"Could not ground an example pattern: {e}")
            return None
        if not rows:
            return None
        row = rows[0]
        return [row[k] for k in row]

    def _ground_single_hop(self, triple: Dict[str, str], prop: str) -> Optional[str]:
        """Source literal of an entity that has this edge."""
        src, rel, tgt = triple["source"], triple["relationship"], triple["target"]
        got = self._ground(
            f"(s:{src})-[:{rel}]->(:{tgt}) WHERE s.`{prop}` IS NOT NULL",
            f"s.`{prop}` AS v")
        return str(got[0]) if got else None

    def _ground_chain(self, chain: List[Dict[str, str]], prop: str) -> Optional[str]:
        """Start-entity literal of an entity that has the whole chain."""
        head = f"(n0:{chain[0]['source']})"
        parts = [head]
        for i, t in enumerate(chain):
            parts.append(f"-[:{t['relationship']}]->(n{i + 1}:{t['target']})")
        got = self._ground(
            "".join(parts) + f" WHERE n0.`{prop}` IS NOT NULL", f"n0.`{prop}` AS v")
        return str(got[0]) if got else None

    def _ground_intersection(self, triple: Dict[str, str], prop: str) -> Optional[Tuple[str, str]]:
        """Two target literals that share at least one source node."""
        src, rel, tgt = triple["source"], triple["relationship"], triple["target"]
        got = self._ground(
            f"(s:{src})-[:{rel}]->(t1:{tgt}), (s)-[:{rel}]->(t2:{tgt}) "
            f"WHERE t1 <> t2 AND t1.`{prop}` IS NOT NULL AND t2.`{prop}` IS NOT NULL",
            f"t1.`{prop}` AS v1, t2.`{prop}` AS v2")
        return (str(got[0]), str(got[1])) if got else None

    def _returns_rows(self, cypher: str) -> bool:
        """Return True if `cypher` returns at least one row."""
        try:
            rows, _ = self.handler.execute_query(cypher)
        except Exception as e:
            logger.warning(f"Could not verify a generated example: {e}")
            return False
        return bool(rows)

    def _diverse_triples(self, relationships: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Pick one triple per relationship type, in a seeded random order."""
        by_type: Dict[str, List[Dict[str, str]]] = {}
        for rel in relationships:
            by_type.setdefault(rel["relationship"], []).append(rel)

        shuffled_types = sorted(by_type.keys())
        self._rng.shuffle(shuffled_types)

        picks: List[Dict[str, str]] = []
        for rtype in shuffled_types:
            candidates = by_type[rtype]
            picks.append(self._rng.choice(candidates))
        return picks

    ############################
    # Identifying property resolution
    ############################

    def _resolve_identifying_properties(
        self, label_properties: Dict[str, List[str]]
    ) -> Dict[str, Optional[str]]:
        """Pick the identifying property of each label.

        Order: user mapping, single fallback-list match, LLM (if available),
        first fallback match, then the first property that has values.
        """
        resolved: Dict[str, Optional[str]] = {}
        unresolved_for_llm: Dict[str, List[str]] = {}

        for label, props in label_properties.items():
            # 1. User-provided mapping wins outright.
            user_choice = self.identifying_properties_user.get(label)
            if user_choice and user_choice in props:
                resolved[label] = user_choice
                continue

            # 2. A single fallback match is unambiguous. Otherwise ask the LLM if
            #    there is one (e.g. geneSymbol over name for Gene), else take the first match.
            matches = self._match_fallback_all(props)
            if len(matches) == 1:
                resolved[label] = matches[0]
                continue
            if self.llm is not None:
                unresolved_for_llm[label] = props
                continue
            if matches:
                resolved[label] = matches[0]
                continue

            unresolved_for_llm[label] = props

        # 3. One batched LLM call for the ambiguous labels.
        if unresolved_for_llm and self.llm is not None:
            try:
                llm_mapping = self._infer_identifying_properties_llm(unresolved_for_llm)
            except Exception as e:
                logger.warning(f"LLM inference for identifying properties failed: {e}")
                llm_mapping = {}
            for label, prop in llm_mapping.items():
                if prop and prop in unresolved_for_llm.get(label, []):
                    resolved[label] = prop

        # 4. Last resort: first fallback match, else first property with a real value.
        for label, props in label_properties.items():
            if resolved.get(label):
                continue
            matches = self._match_fallback_all(props)
            if matches:
                resolved[label] = matches[0]
                continue
            resolved[label] = self._first_non_empty_property(label, props)

        return resolved

    def _match_fallback_all(self, props: List[str]) -> List[str]:
        """Return every property in `props` that matches the fallback list, in priority order."""
        lower = {p.lower(): p for p in props}
        return [
            lower[candidate.lower()]
            for candidate in IDENTIFYING_PROPERTY_FALLBACKS
            if candidate.lower() in lower
        ]

    def _first_non_empty_property(self, label: str, props: List[str]) -> Optional[str]:
        """Return the first property of `label` that has a stored value."""
        for p in props:
            values = self._sample_values(label, p)
            if values:
                return p
        return None

    def _infer_identifying_properties_llm(
        self, label_properties: Dict[str, List[str]]
    ) -> Dict[str, Optional[str]]:
        """Ask the LLM for the identifying property of each label."""
        system = (
            "You help pick the property name a biomedical researcher would naturally "
            "use to refer to an entity of a given node label in a knowledge graph "
            "(e.g. a human-readable name, symbol, or identifier). Prefer short, "
            "human-readable properties over internal identifiers (URIs, hashes). "
            "Return the property verbatim from the provided list, or null if none fits."
        )
        lines = []
        for label, props in label_properties.items():
            desc = self.nodes_descriptions.get(label)
            if desc:
                lines.append(f"- {label} ({desc}): {props}")
            else:
                lines.append(f"- {label}: {props}")
        listing = "\n".join(lines)
        user = (
            "For each of the following node labels, pick the single property name a "
            "human would use to designate an entity of that type. When a label has a "
            "description, use it to infer which property best matches the natural name:\n\n"
            f"{listing}"
        )
        structured = self.llm.with_structured_output(_IdentifyingPropertiesOutput)
        result = structured.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return {item.label: item.property for item in result.items}

    ############################
    # Scaffold builders (Cypher + triple metadata, mode-agnostic)
    ############################

    def _build_labels_selection_examples(
        self,
        triples: List[Dict[str, str]],
        identifying: Dict[str, Optional[str]],
        sampled_values: Dict[str, List[str]],
        used_rels: Set[str],
    ) -> List[Dict[str, Any]]:
        """Return up to 4 label-selection examples, involving 1 to 4 labels."""
        entries: List[Dict[str, Any]] = []

        # Ex1: one label (zero-hop lookup).
        if triples:
            zero = self._cypher_zero_hop(triples[0], identifying, sampled_values)
            entries.append({
                "kind": "labels",
                "triple": zero["triple"],
                "cypher": zero["cypher"],
                "question": zero["question"],
                "selected_labels": [zero["triple"]["source"]],
            })

        # Ex2: two labels (single hop on an unused relationship).
        second = self._first_unused_rel_triple(triples, used_rels) or (triples[1] if len(triples) >= 2 else None)
        if second:
            single = self._cypher_single_hop(second, identifying, sampled_values)
            entries.append({
                "kind": "labels",
                "triple": single["triple"],
                "cypher": single["cypher"],
                "question": single["question"],
                "selected_labels": sorted({
                    single["triple"]["source"], single["triple"]["target"]
                }),
            })
            used_rels.add(second["relationship"])

        # Ex3: three labels (two-hop chain).
        multi = self._pick_multi_hop(triples, avoid_rels=used_rels)
        if multi:
            mh = self._cypher_multi_hop(multi[0], multi[1], identifying, sampled_values)
            entries.append({
                "kind": "labels",
                "triple": multi[0],
                "cypher": mh["cypher"],
                "question": mh["question"],
                "selected_labels": sorted({
                    multi[0]["source"], multi[0]["target"], multi[1]["target"]
                }),
            })
            used_rels.add(multi[0]["relationship"])
            used_rels.add(multi[1]["relationship"])

        # Ex4: four labels (three-hop chain).
        chain = self._pick_chain_of_three(triples, avoid_rels=used_rels)
        if chain:
            ch = self._cypher_chain_of_three(chain[0], chain[1], chain[2], identifying, sampled_values)
            entries.append({
                "kind": "labels",
                "triple": chain[0],
                "cypher": ch["cypher"],
                "question": ch["question"],
                "selected_labels": sorted({
                    chain[0]["source"], chain[0]["target"],
                    chain[1]["target"], chain[2]["target"],
                }),
            })
            for t in chain:
                used_rels.add(t["relationship"])

        return entries

    def _first_unused_rel_triple(
        self,
        triples: List[Dict[str, str]],
        used_rels: Set[str],
    ) -> Optional[Dict[str, str]]:
        """Return the first triple whose relationship is not in `used_rels`."""
        for t in triples:
            if t["relationship"] not in used_rels:
                return t
        return None

    def _build_relationships_selection_examples(
        self,
        triples: List[Dict[str, str]],
        identifying: Dict[str, Optional[str]],
        sampled_values: Dict[str, List[str]],
        used_rels: Set[str],
    ) -> List[Dict[str, Any]]:
        """Return up to 3 relationship-selection examples, involving 1 to 3 relationships."""
        entries: List[Dict[str, Any]] = []

        # Ex1: one relationship (single hop on an unused relationship).
        first = self._first_unused_rel_triple(triples, used_rels) or (triples[0] if triples else None)
        if first:
            single = self._cypher_single_hop(first, identifying, sampled_values)
            entries.append({
                "kind": "rels",
                "triple": first,
                "cypher": single["cypher"],
                "question": single["question"],
                "selected_relationships": [first["relationship"]],
            })
            used_rels.add(first["relationship"])

        # Ex2: two relationships (two-hop chain; deduplicated if both hops share one).
        multi = self._pick_multi_hop(triples, avoid_rels=used_rels)
        if multi:
            mh = self._cypher_multi_hop(multi[0], multi[1], identifying, sampled_values)
            rels = list(dict.fromkeys([multi[0]["relationship"], multi[1]["relationship"]]))
            entries.append({
                "kind": "rels",
                "triple": multi[0],
                "cypher": mh["cypher"],
                "question": mh["question"],
                "selected_relationships": rels,
            })
            used_rels.add(multi[0]["relationship"])
            used_rels.add(multi[1]["relationship"])

        # Ex3: three relationships (three-hop chain).
        chain = self._pick_chain_of_three(triples, avoid_rels=used_rels)
        if chain:
            ch = self._cypher_chain_of_three(chain[0], chain[1], chain[2], identifying, sampled_values)
            rels = list(dict.fromkeys([
                chain[0]["relationship"], chain[1]["relationship"], chain[2]["relationship"]
            ]))
            entries.append({
                "kind": "rels",
                "triple": chain[0],
                "cypher": ch["cypher"],
                "question": ch["question"],
                "selected_relationships": rels,
            })
            for t in chain:
                used_rels.add(t["relationship"])

        return entries

    def _build_query_generation_examples(
        self,
        triples: List[Dict[str, str]],
        identifying: Dict[str, Optional[str]],
        sampled_values: Dict[str, List[str]],
        used_rels: Set[str],
    ) -> List[Dict[str, Any]]:
        """Return up to MAX_QUERY_GENERATION_EXAMPLES examples of varied shapes, each returning rows."""
        entries: List[Dict[str, Any]] = []
        seen_cyphers: set = set()

        def _add(entry: Dict[str, Any], touched: List[Dict[str, str]]):
            if not entry or not entry.get("cypher") or entry["cypher"] in seen_cyphers:
                return
            # Never show an example that returns no rows.
            if not self._returns_rows(entry["cypher"]):
                logger.warning(
                    f"[dynamic_examples] dropping a {entry.get('kind')} example that "
                    f"returns no rows: {entry['cypher']}"
                )
                return
            seen_cyphers.add(entry["cypher"])
            entries.append(entry)
            for t in touched:
                used_rels.add(t["relationship"])

        # Chains first: multi-hop questions need them most, and the final cut
        # drops the last entries.
        multi = self._pick_multi_hop(triples, avoid_rels=used_rels)
        if multi:
            _add(self._cypher_multi_hop(multi[0], multi[1], identifying, sampled_values), list(multi))

        chain3 = self._pick_chain_of_three(triples, avoid_rels=used_rels)
        if chain3:
            _add(self._cypher_chain_of_three(*chain3, identifying, sampled_values), list(chain3))

        # Then two single hops on relationships not yet shown.
        for _ in range(2):
            pick = self._first_unused_rel_triple(triples, used_rels)
            if not pick:
                fallback_index = len(entries)
                if fallback_index < len(triples):
                    pick = triples[fallback_index]
                else:
                    break
            _add(self._cypher_single_hop(pick, identifying, sampled_values), [pick])

        # Two intersections on different relationships, with different question
        # phrasings for the same Cypher shape.
        inter = self._pick_intersection(triples, sampled_values, avoid_rels=used_rels)
        if inter:
            _add(self._cypher_intersection(inter, identifying, sampled_values, variant=0), [inter])
            inter2 = self._pick_intersection(triples, sampled_values, avoid_rels=used_rels)
            if inter2:
                _add(self._cypher_intersection(inter2, identifying, sampled_values, variant=1), [inter2])

        # Top up with single hops when the schema is shallow or examples were dropped.
        i = 2
        while len(entries) < MAX_QUERY_GENERATION_EXAMPLES and i < len(triples):
            _add(self._cypher_single_hop(triples[i], identifying, sampled_values), [triples[i]])
            i += 1

        return entries[:MAX_QUERY_GENERATION_EXAMPLES]

    def _build_query_correction_examples(
        self,
        triples: List[Dict[str, str]],
        identifying: Dict[str, Optional[str]],
        sampled_values: Dict[str, List[str]],
        used_rels: Set[str],
    ) -> List[Dict[str, Any]]:
        """Return up to 2 examples matching what the query-correction prompt receives.

          1. correction_value: wrong literal, fixed from a **Suggested Properties** block.
          2. correction_intersection: flattened IN list rewritten as an intersection.

        Unknown labels or relationship types are handled by the Neo4j
        error-correction prompt, which has no few-shot block.
        """
        entries: List[Dict[str, Any]] = []

        # Ex1: value correction, on a source with an identifying property and sampled values.
        value_triple = None
        for t in triples:
            if t["relationship"] in used_rels:
                continue
            prop = identifying.get(t["source"])
            vals = sampled_values.get(t["source"], [])
            if prop and vals:
                value_triple = t
                break
        if value_triple is None:
            for t in triples:
                prop = identifying.get(t["source"])
                vals = sampled_values.get(t["source"], [])
                if prop and vals:
                    value_triple = t
                    break

        if value_triple is not None:
            base = self._cypher_single_hop(value_triple, identifying, sampled_values)
            value = base.get("value")
            prop = base.get("property")
            wrong_value = self._deterministic_wrong_token(value) if value else None
            if value and prop and wrong_value:
                cypher_broken = self._replace_value(base["cypher"], value, wrong_value)
                if cypher_broken != base["cypher"]:
                    entries.append({
                        "kind": "correction_value",
                        "triple": base["triple"],
                        "source_label": value_triple["source"],
                        "property": prop,
                        "value": value,
                        "wrong_value": wrong_value,
                        "cypher_broken": cypher_broken,
                        "cypher_fixed": base["cypher"],
                        "question": base["question"],
                    })
                    used_rels.add(value_triple["relationship"])

        # Ex2: intersection correction (IN list rewritten as two patterns).
        inter = self._pick_intersection(triples, sampled_values, avoid_rels=used_rels)
        if inter is not None:
            ok = self._cypher_intersection(inter, identifying, sampled_values)
            tgt_prop = identifying.get(inter["target"])
            # Keep the grounded entity pair: this example corrects the operator,
            # not the identity of the entities in the user's question.
            tgt_vals = ok.get("values", [])
            if tgt_prop and len(tgt_vals) >= 2:
                broken = (
                    f"MATCH (s:{inter['source']})-[r:{inter['relationship']}]->(t:{inter['target']}) "
                    f"WHERE t.{tgt_prop} IN "
                    f"[\"{self._escape(tgt_vals[0])}\", \"{self._escape(tgt_vals[1])}\"] "
                    f"RETURN s, r, t"
                )
                entries.append({
                    "kind": "correction_intersection",
                    "triple": inter,
                    "cypher_broken": broken,
                    "cypher_fixed": ok["cypher"],
                    "question": ok["question"],
                })
                used_rels.add(inter["relationship"])

        return entries

    ############################
    # Cypher construction (templates)
    ############################

    def _cypher_single_hop(
        self,
        triple: Dict[str, str],
        identifying: Dict[str, Optional[str]],
        sampled_values: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        src, rel, tgt = triple["source"], triple["relationship"], triple["target"]
        prop = identifying.get(src)
        # Prefer an entity the graph confirms is on this edge; fall back to a
        # per-label sample only if the grounding query finds nothing.
        value = self._ground_single_hop(triple, prop) if prop else None
        if value is None:
            value = sampled_values.get(src, [None])[0]

        if prop and value:
            cypher = (
                f'MATCH (s:{src} {{{prop}: "{self._escape(value)}"}})'
                f'-[r:{rel}]->(t:{tgt}) RETURN s, r, t'
            )
        else:
            cypher = f"MATCH (s:{src})-[r:{rel}]->(t:{tgt}) RETURN s, r, t"

        return {
            "kind": "single_hop",
            "triple": triple,
            "value": value,
            "property": prop,
            "cypher": cypher,
            "question": self._template_question_single_hop(triple, prop, value),
        }

    def _cypher_multi_hop(
        self,
        first: Dict[str, str],
        second: Dict[str, str],
        identifying: Dict[str, Optional[str]],
        sampled_values: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Two-hop chain example; `first.target` must equal `second.source`."""
        s1, r1, t1 = first["source"], first["relationship"], first["target"]
        r2, t2 = second["relationship"], second["target"]
        prop = identifying.get(s1)
        # The start entity must have both hops, so ground it on the whole chain.
        value = self._ground_chain([first, second], prop) if prop else None
        if value is None:
            value = sampled_values.get(s1, [None])[0]

        pattern_head = (
            f'(a:{s1} {{{prop}: "{self._escape(value)}"}})' if (prop and value) else f"(a:{s1})"
        )
        cypher = (
            f"MATCH {pattern_head}-[r1:{r1}]->(b:{t1}), "
            f"(b)-[r2:{r2}]->(c:{t2}) RETURN a, r1, b, r2, c"
        )
        return {
            "kind": "multi_hop",
            "triple_1": first,
            "triple_2": second,
            "value": value,
            "property": prop,
            "cypher": cypher,
            "question": self._template_question_multi_hop(first, second, prop, value),
        }

    def _cypher_intersection(
        self,
        triple: Dict[str, str],
        identifying: Dict[str, Optional[str]],
        sampled_values: Dict[str, List[str]],
        variant: int = 0,
    ) -> Dict[str, Any]:
        src, rel, tgt = triple["source"], triple["relationship"], triple["target"]
        prop = identifying.get(tgt)
        # Two independently sampled targets rarely share a source; ask the graph for a pair.
        grounded = self._ground_intersection(triple, prop) if prop else None
        vals = list(grounded) if grounded else sampled_values.get(tgt, [])

        if prop and len(vals) >= 2:
            v1, v2 = vals[0], vals[1]
            cypher = (
                f'MATCH (s:{src})-[r1:{rel}]->(t1:{tgt} {{{prop}: "{self._escape(v1)}"}}), '
                f'(s)-[r2:{rel}]->(t2:{tgt} {{{prop}: "{self._escape(v2)}"}}) '
                f"RETURN s, r1, t1, r2, t2"
            )
            question = self._template_question_intersection(triple, prop, v1, v2, variant=variant)
        else:
            cypher = (
                f"MATCH (s:{src})-[r1:{rel}]->(t1:{tgt}), "
                f"(s)-[r2:{rel}]->(t2:{tgt}) WHERE t1 <> t2 "
                f"RETURN s, r1, t1, r2, t2"
            )
            question = self._template_question_intersection(triple, prop, None, None, variant=variant)

        return {
            "kind": "intersection",
            "triple": triple,
            "values": vals[:2],
            "property": prop,
            "cypher": cypher,
            "question": question,
        }

    def _cypher_zero_hop(
        self,
        triple: Dict[str, str],
        identifying: Dict[str, Optional[str]],
        sampled_values: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Single-label lookup example."""
        label = triple["source"]
        prop = identifying.get(label)
        value = (sampled_values.get(label) or [None])[0]
        if prop and value:
            cypher = f'MATCH (n:{label} {{{prop}: "{self._escape(value)}"}}) RETURN n'
        else:
            cypher = f"MATCH (n:{label}) RETURN n LIMIT 10"
        return {
            "kind": "zero_hop",
            "triple": triple,
            "value": value,
            "property": prop,
            "cypher": cypher,
            "question": self._template_question_zero_hop(triple, prop, value),
        }

    def _cypher_chain_of_three(
        self,
        t1: Dict[str, str],
        t2: Dict[str, str],
        t3: Dict[str, str],
        identifying: Dict[str, Optional[str]],
        sampled_values: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Three-hop chain example; `t1.target == t2.source` and `t2.target == t3.source`."""
        L1, L2, L3, L4 = t1["source"], t1["target"], t2["target"], t3["target"]
        r1, r2, r3 = t1["relationship"], t2["relationship"], t3["relationship"]
        prop = identifying.get(L1)
        value = self._ground_chain([t1, t2, t3], prop) if prop else None
        if value is None:
            value = (sampled_values.get(L1) or [None])[0]

        head = (
            f'(a:{L1} {{{prop}: "{self._escape(value)}"}})'
            if (prop and value) else f"(a:{L1})"
        )
        cypher = (
            f"MATCH {head}-[r1:{r1}]->(b:{L2}), "
            f"(b)-[r2:{r2}]->(c:{L3}), "
            f"(c)-[r3:{r3}]->(d:{L4}) RETURN a, r1, b, r2, c, r3, d"
        )
        return {
            "kind": "chain_of_three",
            "triple_1": t1,
            "triple_2": t2,
            "triple_3": t3,
            "value": value,
            "property": prop,
            "cypher": cypher,
            "question": self._template_question_chain_of_three(t1, t2, t3, prop, value),
        }

    def _pick_multi_hop(
        self,
        triples: List[Dict[str, str]],
        avoid_rels: Optional[Set[str]] = None,
    ) -> Optional[Tuple[Dict[str, str], Dict[str, str]]]:
        """Pick a two-hop chain, preferring unused relationships, then distinct ones."""
        avoid_rels = avoid_rels or set()
        by_source: Dict[str, List[Dict[str, str]]] = {}
        for t in triples:
            by_source.setdefault(t["source"], []).append(t)

        best: Optional[Tuple[Dict[str, str], Dict[str, str]]] = None
        best_score = (999, 999)
        for t1 in triples:
            for t2 in by_source.get(t1["target"], []):
                reused = int(t1["relationship"] in avoid_rels) + int(t2["relationship"] in avoid_rels)
                dup_rels = 1 if t1["relationship"] == t2["relationship"] else 0
                score = (reused, dup_rels)
                if score < best_score:
                    best_score = score
                    best = (t1, t2)
        return best

    def _pick_chain_of_three(
        self,
        triples: List[Dict[str, str]],
        avoid_rels: Optional[Set[str]] = None,
    ) -> Optional[Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]]:
        """Pick a three-hop chain, preferring unused relationships, then distinct ones."""
        avoid_rels = avoid_rels or set()
        by_source: Dict[str, List[Dict[str, str]]] = {}
        for t in triples:
            by_source.setdefault(t["source"], []).append(t)

        best: Optional[Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]] = None
        best_score = (999, 999)
        for t1 in triples:
            for t2 in by_source.get(t1["target"], []):
                for t3 in by_source.get(t2["target"], []):
                    reused = (
                        int(t1["relationship"] in avoid_rels)
                        + int(t2["relationship"] in avoid_rels)
                        + int(t3["relationship"] in avoid_rels)
                    )
                    rels = {t1["relationship"], t2["relationship"], t3["relationship"]}
                    dup_rels = 3 - len(rels)
                    score = (reused, dup_rels)
                    if score < best_score:
                        best_score = score
                        best = (t1, t2, t3)
        return best

    def _pick_intersection(
        self,
        triples: List[Dict[str, str]],
        sampled_values: Optional[Dict[str, List[str]]] = None,
        avoid_rels: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, str]]:
        """Pick a triple for an intersection: unused relationship first, then a target with 2+ sampled values."""
        avoid_rels = avoid_rels or set()
        sampled_values = sampled_values or {}

        best: Optional[Dict[str, str]] = None
        best_score = (999, 999)  # (reused, no_values)
        for t in triples:
            reused = int(t["relationship"] in avoid_rels)
            has_two = 1 if len(sampled_values.get(t["target"], [])) >= 2 else 0
            no_values = 1 - has_two
            score = (reused, no_values)
            if score < best_score:
                best_score = score
                best = t
        return best

    @staticmethod
    def _escape(value: str) -> str:
        """Escape backslashes and double quotes for a Cypher string literal."""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _deterministic_wrong_token(token: Optional[str]) -> Optional[str]:
        """Return a deterministic variant of `token` that differs from it, for correction examples.

        Tries swapcase, upper, then lower; None if none of them changes the token.
        """
        if not token:
            return None
        t = token.strip()
        if not t:
            return None
        for candidate in (t.swapcase(), t.upper(), t.lower()):
            if candidate and candidate != t:
                return candidate
        return None

    @staticmethod
    def _replace_value(cypher: str, old: Optional[str], new: str) -> str:
        if not old:
            return cypher
        return cypher.replace(f'"{old}"', f'"{new}"')

    ############################
    # NL question templates
    ############################

    def _template_question_zero_hop(
        self, triple: Dict[str, str], prop: Optional[str], value: Optional[str]
    ) -> str:
        src_desc = self._describe(triple["source"], self.nodes_descriptions, fallback=triple["source"])
        if prop and value:
            return f"What information do we have about the {src_desc} whose {prop} is \"{value}\"?"
        return f"List entities of type {src_desc}."

    def _template_question_chain_of_three(
        self,
        t1: Dict[str, str],
        t2: Dict[str, str],
        t3: Dict[str, str],
        prop: Optional[str],
        value: Optional[str],
    ) -> str:
        src_desc = self._describe(t1["source"], self.nodes_descriptions, fallback=t1["source"])
        mid1_desc = self._describe(t1["target"], self.nodes_descriptions, fallback=t1["target"])
        mid2_desc = self._describe(t2["target"], self.nodes_descriptions, fallback=t2["target"])
        end_desc = self._describe(t3["target"], self.nodes_descriptions, fallback=t3["target"])
        r1_desc = self._describe(t1["relationship"], self.relationships_descriptions, fallback=t1["relationship"])
        r2_desc = self._describe(t2["relationship"], self.relationships_descriptions, fallback=t2["relationship"])
        r3_desc = self._describe(t3["relationship"], self.relationships_descriptions, fallback=t3["relationship"])

        prefix = (
            f"Starting from the {src_desc} whose {prop} is \"{value}\""
            if (prop and value) else f"Starting from any {src_desc}"
        )
        return (
            f"{prefix}, which {end_desc} are reachable through a {mid1_desc} "
            f"({r1_desc}), then a {mid2_desc} ({r2_desc}), then {r3_desc}?"
        )

    def _template_question_single_hop(
        self, triple: Dict[str, str], prop: Optional[str], value: Optional[str]
    ) -> str:
        src_desc = self._describe(triple["source"], self.nodes_descriptions, fallback=triple["source"])
        rel_desc = self._describe(triple["relationship"], self.relationships_descriptions, fallback=triple["relationship"])
        tgt_desc = self._describe(triple["target"], self.nodes_descriptions, fallback=triple["target"])

        if prop and value:
            return (
                f"Which {tgt_desc} are connected via {rel_desc} to the {src_desc} "
                f"whose {prop} is \"{value}\"?"
            )
        return f"Which {tgt_desc} are connected via {rel_desc} to any {src_desc}?"

    def _template_question_multi_hop(
        self, t1: Dict[str, str], t2: Dict[str, str], prop: Optional[str], value: Optional[str]
    ) -> str:
        src_desc = self._describe(t1["source"], self.nodes_descriptions, fallback=t1["source"])
        mid_desc = self._describe(t1["target"], self.nodes_descriptions, fallback=t1["target"])
        end_desc = self._describe(t2["target"], self.nodes_descriptions, fallback=t2["target"])
        r1_desc = self._describe(t1["relationship"], self.relationships_descriptions, fallback=t1["relationship"])
        r2_desc = self._describe(t2["relationship"], self.relationships_descriptions, fallback=t2["relationship"])

        prefix = (
            f"Starting from the {src_desc} whose {prop} is \"{value}\""
            if (prop and value)
            else f"Starting from any {src_desc}"
        )
        return (
            f"{prefix}, which {end_desc} are reachable via a {mid_desc} "
            f"that is {r1_desc} then {r2_desc}?"
        )

    def _template_question_intersection(
        self,
        triple: Dict[str, str],
        prop: Optional[str],
        v1: Optional[str],
        v2: Optional[str],
        variant: int = 0,
    ) -> str:
        """Phrase an intersection question; `variant` selects one of several paraphrases.

        The Cypher is the same for every variant, so the model learns the pattern
        rather than one wording.
        """
        src_desc = self._describe(triple["source"], self.nodes_descriptions, fallback=triple["source"])
        tgt_desc = self._describe(triple["target"], self.nodes_descriptions, fallback=triple["target"])
        rel_desc = self._describe(triple["relationship"], self.relationships_descriptions, fallback=triple["relationship"])

        if prop and v1 and v2:
            templates = [
                # V0: "BOTH ... AND ..."
                f'Which {src_desc} are {rel_desc} to BOTH the {tgt_desc} with {prop} "{v1}" AND the {tgt_desc} with {prop} "{v2}"?',
                # V1: plain "and"
                f'Which {src_desc} commonly {rel_desc} the {tgt_desc} "{v1}" and "{v2}"?',
                # V2: "share ... with"
                f'Find the {src_desc} that share {rel_desc} relationships with the {tgt_desc} "{v1}" and "{v2}".',
                # V3: "common to"
                f'List the {src_desc} common to the {tgt_desc} "{v1}" and "{v2}" through {rel_desc}.',
                # V4: "both" mid-sentence
                f'Identify the {src_desc} that {rel_desc} both "{v1}" and "{v2}" ({tgt_desc}).',
            ]
            return templates[variant % len(templates)]

        # No values available.
        no_value_templates = [
            f"Which {src_desc} share {rel_desc} connections to two distinct {tgt_desc}?",
            f"Find the {src_desc} that are {rel_desc} to multiple {tgt_desc} entities.",
            f"List the {src_desc} common to two {tgt_desc} via {rel_desc}.",
        ]
        return no_value_templates[variant % len(no_value_templates)]

    @staticmethod
    def _describe(key: str, descriptions: Dict[str, str], fallback: str) -> str:
        desc = descriptions.get(key)
        if desc:
            # Label plus the first sentence of its description.
            clipped = desc.split(".")[0]
            return f"{fallback} ({clipped.lower()})"
        return fallback

    ############################
    # Block rendering
    ############################

    def _render_block(
        self, family: str, entries: List[Dict[str, Any]], use_llm: bool
    ) -> str:
        if not entries:
            return ""
        if family == "labels_selection":
            return self._render_labels_block(entries, use_llm=use_llm)
        if family == "relationships_selection":
            return self._render_rels_block(entries, use_llm=use_llm)
        if family == "query_generation":
            return self._render_query_gen_block(entries, use_llm=use_llm)
        if family == "query_correction":
            return self._render_query_corr_block(entries, use_llm=use_llm)
        raise ValueError(f"Unknown family: {family}")

    def _render_labels_block(self, entries: List[Dict[str, Any]], use_llm: bool) -> str:
        lines = ["Examples (grounded in the connected graph schema):"]
        for i, e in enumerate(entries, start=1):
            question = self._maybe_llm_question(e, use_llm)
            lines.append(
                f"\nExample {i}:\n"
                f'User query: "{question}"\n'
                f"→ Output Selected Labels: {e['selected_labels']}"
            )
        return "\n".join(lines)

    def _render_rels_block(self, entries: List[Dict[str, Any]], use_llm: bool) -> str:
        lines = ["Examples (grounded in the connected graph schema):"]
        for i, e in enumerate(entries, start=1):
            question = self._maybe_llm_question(e, use_llm)
            lines.append(
                f"\nExample {i}:\n"
                f'User query: "{question}"\n'
                f"→ Output Selected Relationships: {e['selected_relationships']}"
            )
        return "\n".join(lines)

    def _render_query_gen_block(self, entries: List[Dict[str, Any]], use_llm: bool) -> str:
        lines = ["Examples (grounded in the connected graph schema):"]
        for i, e in enumerate(entries, start=1):
            question = self._maybe_llm_question(e, use_llm)
            lines.append(
                f"\nExample {i}:\n"
                f'User Query: "{question}"\n'
                f"→ Cypher: {e['cypher']}"
            )
        return "\n".join(lines)

    def _render_query_corr_block(self, entries: List[Dict[str, Any]], use_llm: bool) -> str:
        lines = ["Examples (grounded in the connected graph schema):"]
        rendered = 0
        for e in entries:
            kind = e["kind"]

            if kind == "correction_value":
                question = self._maybe_llm_question(e, use_llm)
                prop = e["property"]
                label = e["source_label"]
                wrong_value = e["wrong_value"]
                value = e["value"]
                suggested_props_block = (
                    "**Suggested Properties**:\n"
                    f"1. For the property '{prop}' in the element '{label}':\n"
                    f"   - The user provided the value '{wrong_value}'.\n"
                    f"   - A suggested replacement is '{value}', "
                    "which has been found in the graph schema."
                )
                rendered += 1
                lines.append(
                    f"\nExample {rendered}:\n"
                    f"Initial Failed Query: `{e['cypher_broken']}`\n"
                    f"User Question: \"{question}\"\n"
                    f"{suggested_props_block}\n"
                    f"Output: `{e['cypher_fixed']}`"
                )

            elif kind == "correction_intersection":
                question = self._maybe_llm_question(e, use_llm)
                rendered += 1
                lines.append(
                    f"\nExample {rendered} (INTERSECTION CORRECTION):\n"
                    f"Initial Failed Query: `{e['cypher_broken']}`\n"
                    f"User Question: \"{question}\"\n"
                    f"Output: `{e['cypher_fixed']}`"
                )

        return "\n".join(lines)

    def _maybe_llm_question(self, entry: Dict[str, Any], use_llm: bool) -> str:
        """Return an LLM-written question if `use_llm`, else the template question."""
        base_question = entry.get("question", "")
        if not use_llm or self.llm is None:
            return base_question
        try:
            return self._nl_question_llm(entry.get("cypher") or entry.get("cypher_fixed", ""))
        except Exception as e:
            logger.warning(f"LLM NL-question generation failed, falling back to template: {e}")
            return base_question

    def _nl_question_llm(self, cypher: str) -> str:
        """Ask the LLM for a question that `cypher` answers (cached per query)."""
        if not cypher:
            return ""
        cached = self._llm_question_cache.get(cypher)
        if cached is not None:
            return cached
        system = (
            "You rephrase Cypher queries as short natural-language biomedical "
            "questions that a researcher might ask. Respond with a single question, "
            "no preamble, no explanation."
        )
        context_descriptions = "\n".join(
            f"- {k}: {v}"
            for src in (self.nodes_descriptions, self.relationships_descriptions)
            for k, v in src.items()
            if k in cypher  # only pass descriptions that are relevant
        )
        glossary_block = (
            f"Relevant glossary:\n{context_descriptions}\n\n"
            if context_descriptions
            else ""
        )
        user = (
            f"Cypher query:\n{cypher}\n\n"
            f"{glossary_block}"
            "Produce a concise natural-language question this query would answer."
        )
        structured = self.llm.with_structured_output(_NLQuestionOutput)
        result = structured.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        question = result.question.strip()
        self._llm_question_cache[cypher] = question
        return question

    ############################
    # Fingerprint
    ############################

    @staticmethod
    def _schema_fingerprint(nodes_raw: List[Dict[str, Any]], relationships: List[Dict[str, str]]) -> str:
        """SHA-256 of the sorted labels and relationship triples."""
        labels = sorted(
            (
                (entry["node"]["labels"][0] if isinstance(entry["node"]["labels"], list) else entry["node"]["labels"])
                for entry in nodes_raw if isinstance(entry, dict) and "node" in entry
            )
        )
        rel_keys = sorted(
            f"{r['source']}|{r['relationship']}|{r['target']}" for r in relationships
        )
        blob = json.dumps({"labels": labels, "rels": rel_keys}, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
