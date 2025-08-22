#!/bin/bash

# Double-tap detection for streaming transcription
TEMP_FILE="/tmp/ctrl_tap_streaming"
TIMEOUT=0.5  # seconds between taps
CLIENT_SCRIPT="$HOME/whisper-dictation-daemon/whisper_streaming_client.py"

# Check if temp file exists and is recent
if [ -f "$TEMP_FILE" ]; then
    # Get file modification time
    FILE_TIME=$(stat -c %Y "$TEMP_FILE")
    CURRENT_TIME=$(date +%s)
    TIME_DIFF=$((CURRENT_TIME - FILE_TIME))
    
    # Check if within timeout window
    if [ $TIME_DIFF -le 1 ]; then
        # Second tap detected - toggle streaming
        rm -f "$TEMP_FILE"
        python3 "$CLIENT_SCRIPT" toggle
        exit 0
    fi
fi

# First tap - create temp file
touch "$TEMP_FILE"

# Schedule cleanup after timeout
(sleep $TIMEOUT && rm -f "$TEMP_FILE") &