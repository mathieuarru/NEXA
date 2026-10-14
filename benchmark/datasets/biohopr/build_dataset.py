#!/usr/bin/env python3
"""Build BioHopR_2hop.json, the BioHopR question set run on the PrimeKG profile.

Reads data/BioHopR.json (fetched by ./download_biohopr.sh) and writes one
record per question, in the shape of the runner's `open_ended_list` type:

    {"question": "...", "answer": ["entity", ...], "meta": {...}}

Two details of the release are resolved here:

- BioHopR drops PrimeKG's " (disease)" suffix but keeps the space before it
  ("hypophosphatemia " is PrimeKG's "hypophosphatemia (disease)"). An answer is
  rewritten only when it matches no node name as is and its stripped form plus
  " (disease)" does: PrimeKG also holds some unsuffixed diseases as separate
  nodes, so the space cannot simply be stripped.
- Answers stay a JSON list, since some entity names contain commas.

The question is the 2-hop one (`hop2_question_multi`), paired with the
release's single gold list (`answer`).

    python benchmark/datasets/biohopr/build_dataset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUFFIX = " (disease)"
LABEL = {
    "drug": "Drug",
    "disease": "Disease",
    "effect/phenotype": "EffectPhenotype",
    "gene/protein": "GeneProtein",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=here / "data" / "BioHopR.json")
    parser.add_argument("--output-dir", type=Path, default=here)
    parser.add_argument("--profile", default="PrimeKG 21",
                        help="KG profile used to resolve names against PrimeKG.")
    parser.add_argument("--no-graph", action="store_true",
                        help="Skip PrimeKG name resolution (offline inspection only).")
    return parser.parse_args()


def load_graph_names(profile_name: str) -> dict[str, set[str]]:
    """Node names of each BioHopR entity type in PrimeKG."""
    from nexa.api.profile_manager import get_profile_with_password
    from nexa.knowledge_graph.neo4j_handler import Neo4jHandler

    profile = get_profile_with_password(profile_name)
    if not profile:
        raise SystemExit(f"KG profile {profile_name!r} not found.")
    handler = Neo4jHandler(
        host=profile["host"], port=str(profile["port"]),
        username=profile["username"], password=profile["password"],
        db_name=profile["db_name"],
    )
    try:
        names: dict[str, set[str]] = {}
        with handler.driver.session(database=profile["db_name"]) as session:
            for entity_type, label in LABEL.items():
                rows = session.run(f"MATCH (n:{label}) RETURN collect(n.name) AS names").single()
                names[entity_type] = set(rows["names"])
        return names
    finally:
        handler.close()


def resolve(answer: str, known: set[str] | None) -> tuple[str, bool]:
    """Return (resolved name, whether the (disease) suffix was restored)."""
    if known is None or answer in known:
        return answer, False
    candidate = answer.strip() + SUFFIX
    if candidate in known:
        return candidate, True
    return answer, False


def size_bucket(n: int) -> str:
    """Answer-set size class used for stratification."""
    if n < 10:
        return "1-9"
    if n < 50:
        return "10-49"
    if n < 200:
        return "50-199"
    return "200+"


def main() -> None:
    args = parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    names = None if args.no_graph else load_graph_names(args.profile)

    items = []
    restored = unresolved = 0
    for index, row in enumerate(rows):
        known = None if names is None else names.get(row["target_type"])
        answers, seen = [], set()
        for raw in row["answer"]:
            value, was_restored = resolve(raw, known)
            restored += was_restored
            if known is not None and value not in known:
                unresolved += 1
            if value not in seen:  # the same entity can appear twice only through the suffix encoding
                seen.add(value)
                answers.append(value)

        items.append({
            "question": row["hop2_question_multi"],
            "answer": answers,
            "meta": {
                "row_index": index,
                "path": row["relation_hop2"],
                "start": row["hop2"],
                "start_type": row["hop2_type"],
                "bridge": row["hop1"],
                "bridge_type": row["hop1_type"],
                "target_type": row["target_type"],
                "n_gold": len(answers),
                "size_bucket": size_bucket(len(answers)),
            },
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full = args.output_dir / "BioHopR_2hop.json"
    full.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {full} ({len(items)} questions)")
    print(f"  '(disease)' suffixes restored: {restored}")
    print(f"  answers matching no node name: {unresolved}"
          + ("  (graph resolution skipped)" if names is None else ""))


if __name__ == "__main__":
    main()
