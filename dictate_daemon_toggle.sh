#!/bin/bash

PIDFILE="/tmp/whisper_daemon.pid"
SCRIPT_DIR="$(dirname "$0")"

if [ -f "$PIDFILE" ]; then
    # Recording is active, stop it
    PID=$(cat "$PIDFILE")
    if ps -p $PID > /dev/null 2>&1; then
        # Send stop command to daemon
        python "$SCRIPT_DIR/whisper_client.py" STOP
        rm -f "$PIDFILE"
    else
        rm -f "$PIDFILE"
    fi
else
    # Start recording by sending command to daemon
    if python "$SCRIPT_DIR/whisper_client.py" STATUS | grep -q "READY"; then
        python "$SCRIPT_DIR/whisper_client.py" DICTATE &
        echo $! > "$PIDFILE"
    else
        echo "Daemon not ready"
    fi
fi