# Whisper Dictation Daemon

A high-performance voice dictation system using faster-whisper with persistent model loading for instant transcription.

## Features

- **Two Modes**: Streaming (real-time) or Non-streaming (complete recording)
- **Persistent Model**: Whisper large-v3 stays loaded in RAM (~3GB)
- **Fast IPC**: C-based client for instant response
- **Local Agreement Algorithm**: Parallel transcription with sliding windows
- **System Integration**: Systemd user services + hotkey support
- **Audio Feedback**: Start/stop sounds for clear recording state

## Architecture

- **Daemon**: `whisper_streaming.py` - Keeps model loaded, processes audio
- **Client**: `fast_client` - C binary for instant IPC via Unix socket
- **Backend**: faster-whisper with CTranslate2 (int8 quantized)
- **CPU**: Optimized for multi-core (e.g., 24-core Ryzen 9 9900X)
- **Hotkeys**: Double-tap Ctrl to start, single tap to stop

## Files

### Core System
- `whisper_streaming.py` - Main daemon with streaming/non-streaming modes
- `fast_client` / `fast_client.c` - Fast C client for IPC
- `ctrl_double_tap_stream.sh` - Streaming mode control script
- `ctrl_double_tap_nostream.sh` - Non-streaming mode control script

### Service Files
- `whisper-stream.service` - Systemd user service (streaming mode)
- `whisper-nostream.service` - Systemd user service (non-streaming mode)

### Audio Files
- `snare.wav` / `snare.mp3` - Start recording sound
- `hihat.wav` / `hihat.mp3` - Stop recording sound

### Other
- `requirements.txt` - Python dependencies
- `test-locally.sh` - Test both daemons locally without systemd

## Installation (Arch + Hyprland)

### 1. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# System dependencies (Arch)
sudo pacman -S python python-pip gcc wtype

# Compile the C client (if needed)
gcc -O2 -o fast_client fast_client.c
```

### 2. Install Systemd User Services

```bash
# Copy service files
cp whisper-nostream.service ~/.config/systemd/user/
cp whisper-stream.service ~/.config/systemd/user/

# Enable and start the non-streaming service (recommended)
systemctl --user enable whisper-nostream.service
systemctl --user start whisper-nostream.service

# Optional: Enable streaming service if you want both modes
# systemctl --user enable whisper-stream.service
# systemctl --user start whisper-stream.service
```

### 3. Configure Hyprland Keybindings

Add to `~/.config/hypr/bindings.conf` (or `hyprland.conf`):

```conf
# Non-streaming mode (waits until done talking)
bind = , Control_L, exec, /home/caleb/projects/whisper-dictation-daemon/ctrl_double_tap_nostream.sh

# Optional: Streaming mode (real-time transcription)
# bind = , Control_R, exec, /home/caleb/projects/whisper-dictation-daemon/ctrl_double_tap_stream.sh
```

**Note**: Update the path to match your installation directory.

### 4. Reload Hyprland

```bash
hyprctl reload
```

## Usage

### Non-Streaming Mode (Recommended)
- **Double-tap Left Ctrl**: Start recording (hear snare sound)
- **Single tap Left Ctrl**: Stop recording (hear hihat sound)
- Waits until you're done talking, then processes entire recording
- More accurate for longer dictation

### Streaming Mode (Optional)
- **Double-tap Right Ctrl**: Start streaming (hear snare sound)
- **Single tap Right Ctrl**: Stop streaming (hear hihat sound)
- Transcribes in real-time as you speak
- Lower latency but may have accuracy trade-offs

Text is automatically typed at cursor position using `wtype`.

## Configuration

### Model Selection

Edit the service file to change models:

```bash
# In whisper-nostream.service or whisper-stream.service
ExecStart=/usr/bin/python3 /path/to/whisper_streaming.py -n --model large-v3
```

Available models: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`

### Algorithm Options

The daemon uses a Local Agreement algorithm by default. You can configure:

```bash
# Window duration (default: 5.0 seconds)
python3 whisper_streaming.py --window-duration 5.0

# Stride duration (default: 2.0 seconds)
python3 whisper_streaming.py --stride-duration 2.0

# Other algorithms
python3 whisper_streaming.py --algorithm sliding-window
python3 whisper_streaming.py --algorithm simple
```

## Performance

- **Model**: Whisper large-v3 (1550M parameters)
- **Backend**: faster-whisper with CTranslate2 optimization
- **Quantization**: int8 (reduces memory, maintains accuracy)
- **Memory**: ~3GB RAM for persistent model
- **Workers**: 4 parallel workers
- **Speed**: Near-instant (model pre-loaded)

## Monitoring

```bash
# Check service status
systemctl --user status whisper-nostream.service

# View real-time logs
journalctl --user -u whisper-nostream.service -f

# Check transcription logs
tail -f /tmp/whisper_streaming.log

# Test daemon directly
./fast_client nostream status
./fast_client nostream start
./fast_client nostream stop
```

## Troubleshooting

### Service not starting

```bash
# Check service logs
journalctl --user -u whisper-nostream.service -n 50

# Test daemon manually
python3 whisper_streaming.py -n

# Check if socket exists
ls -la /tmp/whisper_streaming_*
```

### Hotkeys not working

```bash
# Check Hyprland keybindings
hyprctl binds | grep Control

# Test control script directly
./ctrl_double_tap_nostream.sh

# Check if recording flag file is created
ls -la /tmp/whisper_nostream_recording
```

### Audio issues

```bash
# Test microphone
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Test audio playback
paplay snare.wav
paplay hihat.wav

# Check PulseAudio/PipeWire
pactl info
```

### Model not loading

```bash
# Check if model is downloaded
ls -la ~/.cache/huggingface/hub/

# Download manually
python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"
```

## Development

### Test Locally Without Systemd

```bash
# Run both daemons for testing
./test-locally.sh

# Or run individually
python3 whisper_streaming.py -n  # Non-streaming
python3 whisper_streaming.py     # Streaming
```

### Process Audio File

```bash
# Test with audio file
python3 whisper_streaming.py --file audio.wav
python3 whisper_streaming.py --file audio.mp3 --algorithm local-agreement
```

## Requirements

- Python 3.8+
- `faster-whisper`
- `sounddevice`
- `numpy`
- `scipy`
- `wtype` (for typing text on Wayland)
- `paplay` or `aplay` (for audio feedback)

## License

MIT