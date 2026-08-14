#include <windows.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    char endpoint[1024];
    char reply[64] = {};
    DWORD written = 0, received = 0;
    HANDLE pipe;

    if (argc != 3) {
        fprintf(stderr, "usage: %s <token> <pid>\n", argv[0]);
        return 2;
    }
    if (snprintf(endpoint, sizeof(endpoint), "\\\\.\\pipe\\vkd3d-fg-latency-%s-%s",
            argv[1], argv[2]) < 0 || strlen(endpoint) >= sizeof(endpoint))
        return 2;

    if (!WaitNamedPipeA(endpoint, 5000)) {
        fprintf(stderr, "telemetry control pipe is unavailable\n");
        return 3;
    }
    pipe = CreateFileA(endpoint, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
    if (pipe == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "could not connect to telemetry control pipe\n");
        return 3;
    }
    if (!WriteFile(pipe, "FINALIZE\n", 9, &written, NULL) || written != 9 ||
            !ReadFile(pipe, reply, sizeof(reply) - 1, &received, NULL)) {
        CloseHandle(pipe);
        fprintf(stderr, "telemetry control request failed\n");
        return 4;
    }
    CloseHandle(pipe);
    reply[received] = '\0';
    fputs(reply, stdout);
    return !strcmp(reply, "SUCCESS\n") ? 0 : 1;
}
