#include <stdio.h> // for printf
#include <stdlib.h> // for atoi
#include <cstring> // for memcpy
#include <algorithm> // for sort
#include <mpi.h>
#include <omp.h>
#define ROOT 0
// HEADERS
#define PORT_CUDA
// #define PORT_HIP
// #define PORT_SYCL
#include "comm.h"
// UTILITIES
#include "util.h"
using namespace CommBench;
using namespace std;
int main(int argc, char *argv[]) {
    // INITIALIZE
    int myid;
    int numproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    setup_gpu();

    //allocate GPU memory buffer
    int count = 268435456;
    int *sendbuf_d;
    int *recvbuf_d;
    cudaMalloc(&sendbuf_d, count * sizeof(int));//1gb
    cudaMalloc(&recvbuf_d, count * sizeof(int));

    // register communication pattern
    CommBench::printid = 0;
    Comm<int> partition(IPC);
    Comm<int> translate(NCCL);
    Comm<int> assemble(IPC);
    partition.add(sendbuf_d, count/4, sendbuf_d, 0, count/4, 0, 1);
    partition.add(sendbuf_d, (2*count)/4, sendbuf_d, 0, count/4, 0, 2);
    partition.add(sendbuf_d, (3*count)/4, sendbuf_d, 0, count/4, 0, 3);
    translate.add(sendbuf_d, 0, recvbuf_d, 0, count/4, 0, 4);
    translate.add(sendbuf_d, 0, recvbuf_d, 0, count/4, 1, 5);
    translate.add(sendbuf_d, 0, recvbuf_d, 0, count/4, 2, 6);
    translate.add(sendbuf_d, 0, recvbuf_d, 0, count/4, 3, 7);
    assemble.add(recvbuf_d, 0, recvbuf_d, count/4, count/4, 5, 4);
    assemble.add(recvbuf_d, 0, recvbuf_d, (2*count)/4, count/4, 6, 4);
    assemble.add(recvbuf_d, 0, recvbuf_d, (3*count)/4, count/4, 7, 4);

    // create sequence
    std::vector<Comm<int>> striping = {partition, translate, assemble};

    // measure end-to-end
    CommBench::measure(striping, 5, 10, count);

    cudaFree(sendbuf_d);
    cudaFree(recvbuf_d);

    MPI_Finalize();
}
