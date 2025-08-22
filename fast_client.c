#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s [nostream|stream] [start|stop|status|toggle]\n", argv[0]);
        return 1;
    }
    
    int sock;
    struct sockaddr_un addr;
    char socket_path[108];
    char *command;
    
    // Build socket path
    if (strcmp(argv[1], "nostream") == 0) {
        strcpy(socket_path, "/tmp/whisper_streaming_local-agreement_nostream_daemon.sock");
    } else {
        strcpy(socket_path, "/tmp/whisper_streaming_local-agreement_daemon.sock");
    }
    
    // Map command
    if (strcmp(argv[2], "start") == 0) {
        command = "STREAM_START";
    } else if (strcmp(argv[2], "stop") == 0) {
        command = "STREAM_STOP";
    } else if (strcmp(argv[2], "status") == 0) {
        command = "STATUS";
    } else {
        fprintf(stderr, "Unknown command: %s\n", argv[2]);
        return 1;
    }
    
    // Create socket
    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock == -1) {
        perror("socket");
        return 1;
    }
    
    // Connect
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
    
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        close(sock);
        return 1;
    }
    
    // Send command
    send(sock, command, strlen(command), 0);
    
    // Get response
    char buffer[1024] = {0};
    recv(sock, buffer, sizeof(buffer) - 1, 0);
    
    close(sock);
    return 0;
}