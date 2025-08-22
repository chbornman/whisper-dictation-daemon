#!/bin/bash

# Streaming transcription control
# Double-tap to start, single tap to stop
TEMP_FILE="/tmp/ctrl_tap_stream"
RECORDING_FILE="/tmp/whisper_stream_recording"
TIMEOUT=0.5  # seconds between taps
SCRIPT_DIR="$HOME/whisper-dictation-daemon"

# Check if currently streaming
if [ -f "$RECORDING_FILE" ]; then
    # Streaming active - single tap stops it
    rm -f "$RECORDING_FILE"
    python3 "$SCRIPT_DIR/whisper_streaming.py" -c stop
    exit 0
fi

# Not streaming - check for double tap to start
if [ -f "$TEMP_FILE" ]; then
    # Get file modification time
    FILE_TIME=$(stat -c %Y "$TEMP_FILE")
    CURRENT_TIME=$(date +%s)
    TIME_DIFF=$((CURRENT_TIME - FILE_TIME))
    
    # Check if within timeout window
    if [ $TIME_DIFF -le 1 ]; then
        # Second tap detected - start streaming
        rm -f "$TEMP_FILE"
        touch "$RECORDING_FILE"
        python3 "$SCRIPT_DIR/whisper_streaming.py" -c start
        exit 0
    fi
fi

# First tap - create temp file
touch "$TEMP_FILE"

# Schedule cleanup after timeout
(sleep $TIMEOUT && rm -f "$TEMP_FILE") &