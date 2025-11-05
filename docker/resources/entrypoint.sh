#!/usr/bin/env bash

set -o pipefail

if [ -n "$RETINA_PORTS" ]; then
  # In this mode, we expect to receive data over UDP, telling websocket ip/port of the server.
  export WS_URL=$(socat -u UDP-RECVFROM:"${RETINA_PORTS}",reuseaddr STDOUT)
fi

# Start Telegraf in background
telegraf --config /etc/srs/telegraf.conf $TELEGRAF_CLI_EXTRA_ARGS &
child=$!

# Trap termination signals
_term() {
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') I! Received SIGTERM/SIGINT, stopping Telegraf..."
    kill -TERM "$child" 2>/dev/null
}
trap _term SIGTERM SIGINT

# Wait for Telegraf to exit
wait "$child"
exit $?
