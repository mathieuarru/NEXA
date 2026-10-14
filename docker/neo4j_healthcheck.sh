#!/usr/bin/env bash
# Healthy once the read-only configuration is applied and Neo4j reports it.
set -euo pipefail
[[ -f /tmp/nexa-read-only-ready ]] || exit 1
auth="${NEO4J_AUTH:-neo4j/password}"
/var/lib/neo4j/bin/cypher-shell --format plain -a bolt://127.0.0.1:7687 \
  -u neo4j -p "${auth#*/}" -d neo4j \
  "SHOW DATABASES YIELD name, access WHERE name = 'neo4j' RETURN access" 2>/dev/null \
  | grep -qx '"read-only"'
