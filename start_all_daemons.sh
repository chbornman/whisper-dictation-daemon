#!/bin/bash

# Start both streaming and non-streaming daemons

echo "Starting non-streaming daemon..."
python3 /home/caleb/whisper-dictation-daemon/whisper_streaming.py -n &
NOSTREAM_PID=$!

sleep 2

echo "Starting streaming daemon..."
python3 /home/caleb/whisper-dictation-daemon/whisper_streaming.py &
STREAM_PID=$!

echo "Non-streaming daemon PID: $NOSTREAM_PID"
echo "Streaming daemon PID: $STREAM_PID"

echo "Both daemons started!"
echo "  Left Ctrl (double-tap): Non-streaming mode (complete recording)"
echo "  Right Ctrl (double-tap): Streaming mode (real-time)"

# Wait for either daemon to exit
wait