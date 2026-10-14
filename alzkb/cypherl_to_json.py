"""Convert the AlzKB v1.2.1 release (a Memgraph CYPHERL export) into the JSON the Neo4j loader reads.

The release file `alzkb-v1-2-1_memgraph` has one statement per line:

    CREATE (:__mg_vertex__:`Drug` {__mg_id__: 696781, `nodeID`: "263", ...});
    MATCH (u:__mg_vertex__), (v:__mg_vertex__) WHERE u.__mg_id__ = 1 AND v.__mg_id__ = 2
    CREATE (u)-[:`CHEMICALBINDSGENE`]->(v);

plus Memgraph index and clean-up statements, which are skipped. The output is
the list `load_alzkb121.sh` imports: {"type": "node", "id", "labels",
"properties"} and {"type": "relationship", "start", "end", "label",
"properties"}, without the `__mg_vertex__` label and `__mg_id__` property.
The result matches the graph used in the paper (251,704 nodes, 1,636,739
relationships, identical labels and properties).

Usage:
    python alzkb/cypherl_to_json.py alzkb-v1-2-1_memgraph alzkb.json.gz
"""

import gzip
import json
import re
import sys

NODE = re.compile(r"^CREATE \((?P<labels>(?::`?[^:`{ ]+`?)+) (?P<props>\{.*\})\);$")
REL = re.compile(r"^MATCH \(u:__mg_vertex__\), \(v:__mg_vertex__\) WHERE u\.__mg_id__ = (?P<a>\d+) "
                 r"AND v\.__mg_id__ = (?P<b>\d+) CREATE \(u\)-\[:`?(?P<type>[^`\] {]+)`?(?: (?P<props>\{.*\}))?\]->\(v\);$")
KEY = re.compile(r'([{,]\s*)(`[^`]+`|__mg_id__)(\s*:)')
# Statements that set up or tear down Memgraph's own import scaffolding.
SCAFFOLDING = ("CREATE INDEX ON ", "CREATE CONSTRAINT ON ", "DROP INDEX ON ",
               "MATCH (u) REMOVE u:__mg_vertex__, u.__mg_id__;")


def cypher_map(text: str) -> dict:
    """Parse a CYPHERL property map as JSON, after quoting its keys and undoing the \\' escape."""
    text = text.replace("\\'", "'")
    return json.loads(KEY.sub(lambda m: f'{m.group(1)}"{m.group(2).strip("`")}"{m.group(3)}', text))


def convert(src: str, dst: str) -> None:
    """Convert the CYPHERL file `src` into gzipped JSON at `dst`; fail on any unexpected statement."""
    items, n_nodes, n_rels = [], 0, 0
    with open(src, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            s = line.rstrip("\n")
            m = NODE.match(s)
            if m:
                labels = [label.strip("`") for label in m.group("labels").split(":")
                          if label and label != "__mg_vertex__"]
                props = cypher_map(m.group("props"))
                items.append({"type": "node", "id": props.pop("__mg_id__"), "labels": labels, "properties": props})
                n_nodes += 1
                continue
            m = REL.match(s)
            if m:
                items.append({"type": "relationship", "start": int(m.group("a")), "end": int(m.group("b")),
                              "label": m.group("type"),
                              "properties": cypher_map(m.group("props")) if m.group("props") else {}})
                n_rels += 1
                continue
            if s and not s.startswith(SCAFFOLDING):
                raise SystemExit(f"{src}:{number}: unexpected statement: {s[:120]}")
    with gzip.open(dst, "wt", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False)
    print(f"{n_nodes} nodes and {n_rels} relationships written to {dst}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
