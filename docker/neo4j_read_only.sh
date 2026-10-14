#!/usr/bin/env bash
# Shared start-up for the bundled Neo4j 5.26 Community images: import once,
# then serve read-only. Sourced by the loaders after the official entrypoint.
set -euo pipefail
export PATH="/var/lib/neo4j/bin:${PATH}"

# Set key=value in neo4j.conf, replacing any existing line for the key.
nexa_config_set() {
  local key="$1" value="$2" config="${NEO4J_HOME:-/var/lib/neo4j}/conf/neo4j.conf"
  local tmp
  tmp="$(mktemp)"
  awk -v key="$key" 'index($0, key "=") != 1 { print }' "$config" > "$tmp"
  printf '%s=%s\n' "$key" "$value" >> "$tmp"
  cat "$tmp" > "$config"
  rm -f "$tmp"
}

nexa_shell() {
  cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p "$NEXA_LOADER_PASSWORD" -d neo4j "$@"
}

# nexa_start_read_only IMPORT_FLAG IMPORT_FUNCTION
nexa_start_read_only() {
  local import_flag="$1" import_function="$2"
  local config_dir="${NEO4J_HOME:-/var/lib/neo4j}/conf"
  local config_backup apoc_backup boot_command
  local auth="${NEO4J_AUTH:-neo4j/password}"
  if [[ "$auth" != neo4j/* || -z "${auth#*/}" ]]; then
    echo "Bundled NEXA loaders require NEO4J_AUTH=neo4j/<password>." >&2
    exit 1
  fi
  NEXA_LOADER_PASSWORD="${auth#*/}"
  # The official entrypoint sets the password only for the 'neo4j' command.
  # This sets the initial password; an existing user is left unchanged.
  neo4j-admin dbms set-initial-password "$NEXA_LOADER_PASSWORD"
  # An import flag persists in /data. Readiness never persists across starts.
  rm -f /tmp/nexa-read-only-ready

  if [[ ! -f "$import_flag" ]]; then
    config_backup="$(mktemp)"
    apoc_backup="$(mktemp)"
    cp "$config_dir/neo4j.conf" "$config_backup"
    if [[ -f "$config_dir/apoc.conf" ]]; then
      cp "$config_dir/apoc.conf" "$apoc_backup"
    fi
    # The temporary writable server listens on localhost only, without HTTP.
    nexa_config_set server.default_listen_address 127.0.0.1
    nexa_config_set server.bolt.listen_address 127.0.0.1:7687
    nexa_config_set server.bolt.enabled true
    nexa_config_set server.bolt.tls_level DISABLED
    nexa_config_set server.http.enabled false
    nexa_config_set server.https.enabled false
    nexa_config_set server.databases.default_to_read_only false
    nexa_config_set server.databases.read_only ""
    nexa_config_set dbms.security.procedures.allowlist 'apoc.*,db.*,dbms.*'
    nexa_config_set dbms.security.procedures.unrestricted 'apoc.*'
    printf '%s\n' 'apoc.import.file.enabled=true' 'apoc.export.file.enabled=false' > "$config_dir/apoc.conf"

    boot_command="$(neo4j console --dry-run)"
    bash -c "exec $boot_command" &
    NEXA_BOOT_PID=$!
    trap 'kill -TERM "$NEXA_BOOT_PID" 2>/dev/null || true; wait "$NEXA_BOOT_PID" 2>/dev/null || true; cat "$config_backup" > "$config_dir/neo4j.conf"; cat "$apoc_backup" > "$config_dir/apoc.conf"' EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    local ready=false
    for ((attempt=0; attempt<150; attempt++)); do
      kill -0 "$NEXA_BOOT_PID" 2>/dev/null || { echo "Import server exited." >&2; exit 1; }
      if nexa_shell 'RETURN 1' >/dev/null 2>&1; then ready=true; break; fi
      sleep 2
    done
    [[ "$ready" == true ]] || { echo "Import server did not become ready." >&2; exit 1; }
    "$import_function"
    # Mark success only after every import/verification command has succeeded.
    touch "$import_flag"
    kill -TERM "$NEXA_BOOT_PID"
    wait "$NEXA_BOOT_PID" || true
    trap - EXIT TERM INT
    cat "$config_backup" > "$config_dir/neo4j.conf"
    cat "$apoc_backup" > "$config_dir/apoc.conf"
    rm -f "$config_backup" "$apoc_backup"
  fi

  # Reapplied on every start, so the public server always starts read-only.
  nexa_config_set server.databases.read_only neo4j
  nexa_config_set dbms.security.procedures.allowlist 'apoc.meta.stats,apoc.meta.data,apoc.text.replace,db.labels,db.relationshipTypes,db.propertyKeys,db.schema.*,db.index.fulltext.queryNodes,db.index.fulltext.queryRelationships,db.index.vector.queryNodes,db.index.vector.queryRelationships,db.ping,dbms.components'
  nexa_config_set dbms.security.procedures.unrestricted 'apoc.meta.stats,apoc.meta.data'
  printf '%s\n' 'apoc.import.file.enabled=false' 'apoc.export.file.enabled=false' > "$config_dir/apoc.conf"
  echo "Starting NEXA graph service in read-only mode."
  # The marker means the read-only configuration is in place; the healthcheck
  # also queries Bolt. Launching directly keeps the container environment from
  # re-applying permissive settings.
  touch /tmp/nexa-read-only-ready
  boot_command="$(neo4j console --dry-run)"
  # The boot command holds a Java classpath with '*': disable shell globbing.
  set -f
  eval "exec $boot_command"
}
