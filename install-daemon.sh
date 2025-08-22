#!/bin/bash

echo "Installing Whisper daemon service..."

# Copy service file
sudo cp /home/caleb/whisper-dictation-daemon/whisper-daemon.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable the service
sudo systemctl enable whisper-daemon.service

echo "Daemon service installed and enabled!"
echo ""
echo "Usage:"
echo "  Start daemon: sudo systemctl start whisper-daemon"
echo "  Stop daemon:  sudo systemctl stop whisper-daemon"  
echo "  Check status: sudo systemctl status whisper-daemon"
echo "  View logs:    journalctl -u whisper-daemon -f"
echo ""
echo "Update Hyprland config to use:"
echo "  bind = CTRL, CTRL_L, exec, ~/whisper-dictation-daemon/ctrl_double_tap.sh"
echo "  bind = CTRL, CTRL_R, exec, ~/whisper-dictation-daemon/ctrl_double_tap.sh"