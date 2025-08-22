#!/usr/bin/env python3
import socket
import sys

def send_command(command):
    """Send command to whisper daemon"""
    try:
        client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client_socket.connect("/tmp/whisper_daemon.sock")
        client_socket.send(command.encode())
        response = client_socket.recv(1024).decode()
        client_socket.close()
        return response
    except Exception as e:
        print(f"Error communicating with daemon: {e}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python whisper_client.py [DICTATE|STOP|STATUS]")
        sys.exit(1)
    
    command = sys.argv[1].upper()
    response = send_command(command)
    
    if response:
        print(f"Response: {response}")
    else:
        print("Failed to communicate with daemon")

if __name__ == "__main__":
    main()