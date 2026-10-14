#!/bin/sh
# preflight PORT COMMAND...: refuse to start COMMAND when PORT is already taken.
# With host networking (Linux) the service would otherwise crash-loop while a
# health check is answered by the process already holding the port.
set -e

port="$1"
shift

if python -c "import socket, sys; sys.exit(0 if socket.socket().connect_ex(('127.0.0.1', $port)) == 0 else 1)"; then
    echo "ERROR: port $port is already in use on this host." >&2
    echo "Stop the process holding it, then start NEXA again." >&2
    exit 1
fi

exec "$@"
