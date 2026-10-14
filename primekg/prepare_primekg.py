#!/usr/bin/env python3
"""Convert the official PrimeKG edge list (kg.csv) to neo4j-admin import files.

Writes nodes.csv, relationships.csv, the expected label and relationship-type
counts, manifest.json and counts.env to --output-dir. Run by Dockerfile.primekg.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = (
    "relation",
    "display_relation",
    "x_index",
    "x_id",
    "x_type",
    "x_name",
    "x_source",
    "y_index",
    "y_id",
    "y_type",
    "y_name",
    "y_source",
)

LABEL_OVERRIDES = {
    "gene/protein": "GeneProtein",
    "effect/phenotype": "EffectPhenotype",
}


@dataclass(frozen=True)
class Node:
    index: int
    identifier: str
    entity_type: str
    name: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-file-id", required=True)
    parser.add_argument("--source-md5", required=True)
    return parser.parse_args()


def neo4j_token(value: str, *, uppercase: bool = False) -> str:
    """Neo4j label (CamelCase) or, with `uppercase`, relationship type (UPPER_SNAKE)."""
    words = re.findall(r"[A-Za-z0-9]+", value)
    if not words:
        raise ValueError(f"Cannot derive a Neo4j token from {value!r}")
    token = "_".join(words).upper() if uppercase else "".join(word.title() for word in words)
    if token[0].isdigit():
        token = f"T_{token}" if uppercase else f"Type{token}"
    return token


def label_for(entity_type: str) -> str:
    return LABEL_OVERRIDES.get(entity_type, neo4j_token(entity_type))


def relationship_type_for(relation: str) -> str:
    return neo4j_token(relation, uppercase=True)


def node_from_row(row: dict[str, str], prefix: str, row_number: int) -> Node:
    raw_index = row[f"{prefix}_index"].strip()
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: invalid {prefix}_index {raw_index!r}") from exc

    entity_type = row[f"{prefix}_type"].strip()
    identifier = row[f"{prefix}_id"].strip()
    if not entity_type or not identifier:
        raise ValueError(f"Row {row_number}: {prefix} node has an empty type or identifier")

    return Node(
        index=index,
        identifier=identifier,
        entity_type=entity_type,
        name=row[f"{prefix}_name"].strip(),
        source=row[f"{prefix}_source"].strip(),
    )


def register_node(nodes: dict[int, Node], node: Node, row_number: int) -> None:
    """Record `node`; an index seen before with different metadata is an error."""
    previous = nodes.setdefault(node.index, node)
    if previous != node:
        raise ValueError(
            f"Row {row_number}: PrimeKG index {node.index} has conflicting metadata: "
            f"{previous!r} versus {node!r}"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    relationships_path = args.output_dir / "relationships.csv"
    nodes: dict[int, Node] = {}
    relationship_counts: Counter[str] = Counter()
    relation_token_sources: dict[str, str] = {}
    relationship_count = 0

    with (
        args.input.open("r", encoding="utf-8-sig", newline="") as source,
        relationships_path.open("w", encoding="utf-8", newline="") as destination,
    ):
        reader = csv.DictReader(source)
        if reader.fieldnames != list(REQUIRED_COLUMNS):
            raise ValueError(
                "Unexpected PrimeKG header. "
                f"Expected {list(REQUIRED_COLUMNS)!r}, got {reader.fieldnames!r}"
            )

        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(
            (
                ":START_ID(PrimeKG)",
                ":END_ID(PrimeKG)",
                "primekg_row:long",
                "relation",
                "display_relation",
                ":TYPE",
            )
        )

        for row_number, row in enumerate(reader, start=2):
            if row is None or not any(row.values()):
                continue
            if None in row:
                raise ValueError(f"Row {row_number}: too many CSV fields")

            x_node = node_from_row(row, "x", row_number)
            y_node = node_from_row(row, "y", row_number)
            register_node(nodes, x_node, row_number)
            register_node(nodes, y_node, row_number)

            relation = row["relation"].strip()
            if not relation:
                raise ValueError(f"Row {row_number}: empty relation")
            relationship_type = relationship_type_for(relation)
            previous_relation = relation_token_sources.setdefault(relationship_type, relation)
            if previous_relation != relation:
                raise ValueError(
                    f"Neo4j relationship type collision: {previous_relation!r} and {relation!r} "
                    f"both map to {relationship_type!r}"
                )

            relationship_count += 1
            relationship_counts[relationship_type] += 1
            writer.writerow(
                (
                    x_node.index,
                    y_node.index,
                    relationship_count,
                    relation,
                    row["display_relation"].strip(),
                    relationship_type,
                )
            )

            if relationship_count % 1_000_000 == 0:
                print(f"Prepared {relationship_count:,} relationships", flush=True)

    label_counts: Counter[str] = Counter()
    label_token_sources: dict[str, str] = {}
    with (args.output_dir / "nodes.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(
            (
                ":ID(PrimeKG)",
                "primekg_index:long",
                "id",
                "name",
                "entity_type",
                "source",
                ":LABEL",
            )
        )
        for index in sorted(nodes):
            node = nodes[index]
            label = label_for(node.entity_type)
            previous_type = label_token_sources.setdefault(label, node.entity_type)
            if previous_type != node.entity_type:
                raise ValueError(
                    f"Neo4j label collision: {previous_type!r} and {node.entity_type!r} "
                    f"both map to {label!r}"
                )
            label_counts[label] += 1
            writer.writerow(
                (
                    node.index,
                    node.index,
                    node.identifier,
                    node.name,
                    node.entity_type,
                    node.source,
                    label,
                )
            )

    with (args.output_dir / "expected-label-counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.writer(destination, lineterminator="\n")
        for label, count in sorted(label_counts.items()):
            writer.writerow((label, count))

    with (args.output_dir / "expected-relationship-type-counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.writer(destination, lineterminator="\n")
        for relationship_type, count in sorted(relationship_counts.items()):
            writer.writerow((relationship_type, count))

    manifest = {
        "source": {
            "dataset": "PrimeKG",
            "version": args.source_version,
            "dataverse_file_id": args.source_file_id,
            "md5": args.source_md5,
        },
        "node_count": len(nodes),
        "relationship_count": relationship_count,
        "labels": dict(sorted(label_counts.items())),
        "relationship_types": dict(sorted(relationship_counts.items())),
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")

    with (args.output_dir / "counts.env").open("w", encoding="ascii") as destination:
        destination.write(f"EXPECTED_NODES={len(nodes)}\n")
        destination.write(f"EXPECTED_RELATIONSHIPS={relationship_count}\n")
        destination.write(f"SOURCE_MD5={args.source_md5}\n")

    print(
        f"Prepared {len(nodes):,} nodes and {relationship_count:,} relationships "
        f"across {len(label_counts)} labels and {len(relationship_counts)} relationship types.",
        flush=True,
    )


if __name__ == "__main__":
    main()
