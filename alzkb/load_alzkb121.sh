#!/usr/bin/env bash
set -euo pipefail
source /opt/nexa/neo4j_read_only.sh

import_alzkb() {
  echo "Creating temporary nodes from JSON..."
  nexa_shell <<'CYPHER'
CALL apoc.load.json('file:///alzkb.json.gz') YIELD value
UNWIND value AS item
WITH item WHERE item.type = 'node'
CALL apoc.create.node(item.labels + ['_JsonNode'], item.properties) YIELD node
SET node.jsonId = item.id;
CYPHER

  echo "Creating temporary index on jsonId..."
  nexa_shell "CREATE INDEX jsonIdIndex IF NOT EXISTS FOR (n:_JsonNode) ON (n.jsonId)"

  echo "Creating relationships in batches..."
  nexa_shell <<'CYPHER'
CALL apoc.periodic.iterate(
  "CALL apoc.load.json('file:///alzkb.json.gz') YIELD value WITH value WHERE value.type = 'relationship' RETURN value",
  "MATCH (s:_JsonNode {jsonId: value.start}), (e:_JsonNode {jsonId: value.end})
   CALL apoc.create.relationship(s, value.label, value.properties, e) YIELD rel
   RETURN count(rel)",
  {batchSize:1000, parallel:false}
) YIELD batches, total, failedBatches
WITH batches, total, failedBatches
CALL apoc.util.validate(failedBatches > 0, "AlzKB relationship import failed", [])
RETURN batches, total, failedBatches;
CYPHER

  echo "Cleaning temporary labels and properties..."
  nexa_shell "MATCH (n:_JsonNode) REMOVE n.jsonId, n:_JsonNode"

  echo "Dropping temporary index..."
  nexa_shell "DROP INDEX jsonIdIndex IF EXISTS"

}

nexa_start_read_only /data/.import_done_alzkb121 import_alzkb
