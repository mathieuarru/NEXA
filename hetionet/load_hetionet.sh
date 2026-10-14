#!/usr/bin/env bash
set -euo pipefail
source /opt/nexa/neo4j_read_only.sh

import_hetionet() {
  echo "Loading node Cypher scripts..."
  shopt -s nullglob
  for f in /opt/cypher/node_*.cypher; do
    echo "  - $f"
    nexa_shell < "$f"
  done

  echo "Loading edge Cypher scripts..."
  for f in /opt/cypher/edge_*.cypher; do
    echo "  - $f"
    nexa_shell < "$f"
  done
  shopt -u nullglob

}

nexa_start_read_only /data/.import_done import_hetionet
