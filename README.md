# Whisper Dictation Daemon

A high-performance voice dictation system using OpenAI Whisper with persistent model loading for instant transcription.

## Architecture

- **Daemon Service**: Keeps Whisper large-v3 model loaded in RAM (3GB)
- **Client Interface**: Instant dictation requests via Unix socket
- **CPU Backend**: Uses 24-core Ryzen 9 9900X with faster-whisper optimization
- **System Hotkeys**: Double-tap Ctrl to start, single tap to stop

## Files

### Core System
- `whisper_daemon.py` - Main daemon with persistent model
- `whisper_client.py` - Client for sending commands to daemon
- `dictate_daemon_toggle.sh` - Toggle script for hotkeys
- `ctrl_double_tap.sh` - Double-tap detection logic

### Service Management  
- `whisper-daemon.service` - Systemd service file
- `install-daemon.sh` - Service installation script

### Audio Files
- `tr707-snare-drum-241412.mp3` - Start recording sound
- `echoed-hi-hats-89491.mp3` - Stop recording sound

## Installation

1. **Install the daemon service**:
   ```bash
   ./install-daemon.sh
   ```

2. **Start the daemon**:
   ```bash
   sudo systemctl start whisper-daemon
   ```

3. **Update Hyprland config** (`~/.config/hypr/hyprland.conf`):
   ```
   bind = CTRL, CTRL_L, exec, ~/whisper-dictation-daemon/ctrl_double_tap.sh
   bind = CTRL, CTRL_R, exec, ~/whisper-dictation-daemon/ctrl_double_tap.sh
   ```

4. **Reload Hyprland**:
   ```bash
   hyprctl reload
   ```

## Usage

- **Double-tap Left/Right Ctrl**: Start recording (hear drum sound)
- **Single tap Ctrl**: Stop recording and transcribe (hear hi-hat sound)
- Text appears at cursor position automatically

## Performance

- **Model**: Whisper large-v3 (1550M parameters)
- **Backend**: faster-whisper with CTranslate2 optimization
- **CPU**: 24-core Ryzen 9 9900X with int8 quantization
- **Memory**: ~3GB RAM for persistent model
- **Speed**: Instant transcription (model pre-loaded)

## Monitoring

```bash
# Check daemon status
sudo systemctl status whisper-daemon

# View real-time logs  
journalctl -u whisper-daemon -f

# Check hotkey logs
tail -f /tmp/whisper_daemon_ctrl_tap.log

# Test manual dictation
python whisper_client.py DICTATE
```

## Troubleshooting

### Daemon not starting
```bash
# Check service logs
journalctl -u whisper-daemon -n 50

# Test daemon manually
python whisper_daemon.py
```

### Hotkeys not working
```bash
# Check Hyprland config
hyprctl binds | grep CTRL

# Test toggle script
./dictate_daemon_toggle.sh
```

### Audio issues
```bash
# Test microphone
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test audio playback
ffplay -nodisp -autoexit tr707-snare-drum-241412.mp3
```