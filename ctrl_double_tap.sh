#!/bin/bash

# Script to handle double-tap detection for Whisper daemon
LAST_PRESS_FILE="/tmp/whisper_daemon_last_ctrl_press"
RECORDING_FILE="/tmp/whisper_daemon_recording.active"
DOUBLE_TAP_TIMEOUT=0.5  # 500ms for double tap
LOG_FILE="/tmp/whisper_daemon_ctrl_tap.log"

current_time=$(date +%s.%N)
echo "[$(date)] Ctrl pressed at $current_time" >> $LOG_FILE

if [ -f "$RECORDING_FILE" ]; then
    # Recording is active, single tap stops it
    echo "[$(date)] Stopping recording (single tap)" >> $LOG_FILE
    ~/whisper-dictation-daemon/dictate_daemon_toggle.sh
    rm -f "$RECORDING_FILE"
    rm -f "$LAST_PRESS_FILE"
else
    # Check for double tap
    if [ -f "$LAST_PRESS_FILE" ]; then
        last_press=$(cat "$LAST_PRESS_FILE")
        time_diff=$(echo "$current_time - $last_press" | bc)
        
        # Check if within double tap window
        if (( $(echo "$time_diff < $DOUBLE_TAP_TIMEOUT" | bc -l) )); then
            # Double tap detected - start recording
            echo "[$(date)] Double tap detected! Time diff: $time_diff" >> $LOG_FILE
            ~/whisper-dictation-daemon/dictate_daemon_toggle.sh
            touch "$RECORDING_FILE"
            rm -f "$LAST_PRESS_FILE"
        else
            # Too slow, update timestamp
            echo "[$(date)] Single tap (too slow). Time diff: $time_diff" >> $LOG_FILE
            echo "$current_time" > "$LAST_PRESS_FILE"
        fi
    else
        # First press
        echo "[$(date)] First tap detected" >> $LOG_FILE
        echo "$current_time" > "$LAST_PRESS_FILE"
    fi
fi