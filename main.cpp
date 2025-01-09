#include "commbench.h"
#include <string>
#include <vector>

using namespace CommBench;

// Function to check if a given flag is present in the command-line arguments
bool flagPresent(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}

int main(int argc, char* argv[]) {

    size_t *sendbuf;
    size_t *recvbuf;
    size_t numbytes = 1e9;

    init();
    allocate(sendbuf, numbytes);
    allocate(recvbuf, numbytes);

    // Always execute MPI test
    Comm<size_t> test1(MPI);
    test1.add(sendbuf, recvbuf, numbytes, 0, 1);
    test1.measure(5, 10);

    // Check if IPC should be used
    if (flagPresent(argc, argv, "--use-ipc")) {
        Comm<size_t> test2(IPC);
        test2.add(sendbuf, recvbuf, numbytes, 0, 1);
        test2.measure(5, 10);
    }

    // Check if NCCL should be used
    if (flagPresent(argc, argv, "--use-nccl")) {
        Comm<size_t> test3(NCCL);
        test3.add(sendbuf, recvbuf, numbytes, 0, 1);
        test3.measure(5, 10);
    }

    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();

    return 0;
}

