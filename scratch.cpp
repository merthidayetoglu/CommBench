#include "commbench.h"
#include <cstdlib>

using namespace CommBench;

int main(int argc, char* argv[]) {
    init();

    int *sendbuf, *recvbuf;
    size_t num_bytes = 1e8;
    const unsigned N = 2, G = 4, K = 1;

    allocate(sendbuf, num_bytes * numproc);
    allocate(recvbuf, num_bytes * numproc);

    Comm<int> test(library::MPI);

    // Rail


    MPI_Finalize();
}