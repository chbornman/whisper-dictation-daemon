#!/bin/bash

# Non-streaming transcription control
# Double-tap to start, single tap to stop
TEMP_FILE="/tmp/ctrl_tap_nostream"
RECORDING_FILE="/tmp/whisper_nostream_recording"
TIMEOUT=0.5  # seconds between taps
SCRIPT_DIR="$HOME/whisper-dictation-daemon"

# Check if currently recording
if [ -f "$RECORDING_FILE" ]; then
    # Recording active - single tap stops it
    rm -f "$RECORDING_FILE"
    python3 "$SCRIPT_DIR/whisper_streaming.py" -n -c stop
    exit 0
fi

# Not recording - check for double tap to start
if [ -f "$TEMP_FILE" ]; then
    # Get file modification time
    FILE_TIME=$(stat -c %Y "$TEMP_FILE")
    CURRENT_TIME=$(date +%s)
    TIME_DIFF=$((CURRENT_TIME - FILE_TIME))
    
    # Check if within timeout window
    if [ $TIME_DIFF -le 1 ]; then
        # Second tap detected - start recording
        rm -f "$TEMP_FILE"
        touch "$RECORDING_FILE"
        python3 "$SCRIPT_DIR/whisper_streaming.py" -n -c start
        exit 0
    fi
fi

# First tap - create temp file
touch "$TEMP_FILE"

# Schedule cleanup after timeout
(sleep $TIMEOUT && rm -f "$TEMP_FILE") &