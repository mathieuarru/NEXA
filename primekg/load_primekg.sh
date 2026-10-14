#!/usr/bin/env bash
# Offline neo4j-admin import of PrimeKG on first start, then a check of every
# count and a read-only start.
set -euo pipefail

export PATH="/var/lib/neo4j/bin:${PATH}"

DB="neo4j"
IMPORT_DIR="/var/lib/neo4j/import/primekg"
OWNER_FLAG="/data/.managed_by_primekg_loader"
OFFLINE_IMPORT_FLAG="/data/.offline_import_done_primekg"
VERIFIED_FLAG="/data/.import_done_primekg"

# Generated from the source CSV during the image build.
# shellcheck disable=SC1091
source "${IMPORT_DIR}/counts.env"

if [[ ! -f "${OFFLINE_IMPORT_FLAG}" ]]; then
  if [[ -d "/data/databases/${DB}" && ! -f "${OWNER_FLAG}" ]]; then
    echo "Refusing to overwrite an existing ${DB} database not created by the PrimeKG loader." >&2
    exit 1
  fi

  touch "${OWNER_FLAG}"
  echo "Importing ${EXPECTED_NODES} PrimeKG nodes and ${EXPECTED_RELATIONSHIPS} relationships..."
  neo4j-admin database import full \
    --nodes="${IMPORT_DIR}/nodes.csv" \
    --relationships="${IMPORT_DIR}/relationships.csv" \
    --overwrite-destination=true \
    --skip-bad-relationships=false \
    --skip-duplicate-nodes=false \
    "${DB}"
  touch "${OFFLINE_IMPORT_FLAG}"
  echo "Offline PrimeKG import completed."
else
  echo "Offline PrimeKG import already completed; keeping the existing database."
fi

source /opt/nexa/neo4j_read_only.sh

verify_primekg() {
  query_scalar() {
    local query="$1"
    nexa_shell --format plain "${query}" \
      | awk 'NF { value=$0 } END { gsub(/^"|"$/, "", value); print value }'
  }

  echo "Creating the PrimeKG node identifier constraints..."
  while IFS=, read -r label _expected; do
    nexa_shell \
      "CREATE CONSTRAINT \`primekg_${label}_index_unique\` IF NOT EXISTS FOR (n:\`${label}\`) REQUIRE n.primekg_index IS UNIQUE"
  done < "${IMPORT_DIR}/expected-label-counts.csv"

  echo "Validating imported totals..."
  actual_nodes="$(query_scalar 'MATCH (n) RETURN count(n) AS count')"
  actual_relationships="$(query_scalar 'MATCH ()-[r]->() RETURN count(r) AS count')"
  unique_node_indexes="$(query_scalar 'MATCH (n) RETURN count(DISTINCT n.primekg_index) AS count')"
  unique_relationship_rows="$(query_scalar 'MATCH ()-[r]->() RETURN count(DISTINCT r.primekg_row) AS count')"

  [[ "${actual_nodes}" == "${EXPECTED_NODES}" ]] || {
    echo "Node-count mismatch: expected ${EXPECTED_NODES}, got ${actual_nodes}" >&2
    exit 1
  }
  [[ "${actual_relationships}" == "${EXPECTED_RELATIONSHIPS}" ]] || {
    echo "Relationship-count mismatch: expected ${EXPECTED_RELATIONSHIPS}, got ${actual_relationships}" >&2
    exit 1
  }
  [[ "${unique_node_indexes}" == "${EXPECTED_NODES}" ]] || {
    echo "Node identity mismatch: expected ${EXPECTED_NODES} unique indexes, got ${unique_node_indexes}" >&2
    exit 1
  }
  [[ "${unique_relationship_rows}" == "${EXPECTED_RELATIONSHIPS}" ]] || {
    echo "Relationship identity mismatch: expected ${EXPECTED_RELATIONSHIPS} unique rows, got ${unique_relationship_rows}" >&2
    exit 1
  }

  echo "Validating every PrimeKG node label..."
  while IFS=, read -r label expected; do
    actual="$(query_scalar "MATCH (n:\`${label}\`) RETURN count(n) AS count")"
    [[ "${actual}" == "${expected}" ]] || {
      echo "Label ${label} mismatch: expected ${expected}, got ${actual}" >&2
      exit 1
    }
  done < "${IMPORT_DIR}/expected-label-counts.csv"

  echo "Validating every PrimeKG relationship type..."
  while IFS=, read -r relationship_type expected; do
    actual="$(query_scalar "MATCH ()-[r:\`${relationship_type}\`]->() RETURN count(r) AS count")"
    [[ "${actual}" == "${expected}" ]] || {
      echo "Relationship type ${relationship_type} mismatch: expected ${expected}, got ${actual}" >&2
      exit 1
    }
  done < "${IMPORT_DIR}/expected-relationship-type-counts.csv"

}

nexa_start_read_only "${VERIFIED_FLAG}" verify_primekg
