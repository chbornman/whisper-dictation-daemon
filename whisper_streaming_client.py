#!/usr/bin/env python3
import socket
import sys
import time

def send_command(command):
    """Send command to the streaming daemon"""
    socket_path = "/tmp/whisper_streaming_daemon.sock"
    
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(socket_path)
        client.send(command.encode())
        response = client.recv(1024).decode()
        client.close()
        return response
    except FileNotFoundError:
        print("Streaming daemon not running!")
        return None
    except Exception as e:
        print(f"Error communicating with daemon: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: whisper_streaming_client.py [start|stop|status|toggle]")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    if action == "start":
        response = send_command("STREAM_START")
        if response == "STREAMING":
            print("Started streaming transcription")
        elif response == "ALREADY_STREAMING":
            print("Already streaming")
        else:
            print(f"Unexpected response: {response}")
    
    elif action == "stop":
        response = send_command("STREAM_STOP")
        if response == "STOPPED":
            print("Stopped streaming")
        else:
            print(f"Unexpected response: {response}")
    
    elif action == "status":
        response = send_command("STATUS")
        if response:
            print(f"Daemon status: {response}")
        
    elif action == "toggle":
        # Check current status
        status = send_command("STATUS")
        if status == "STREAMING":
            # Currently streaming, stop it
            response = send_command("STREAM_STOP")
            print("Stopped streaming")
        else:
            # Not streaming, start it
            response = send_command("STREAM_START")
            if response == "STREAMING":
                print("Started streaming transcription")

    else:
        print(f"Unknown action: {action}")
        print("Valid actions: start, stop, status, toggle")
        sys.exit(1)

if __name__ == "__main__":
    main()