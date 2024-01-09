
// HEADERS
#define PORT_CUDA
// #define PORT_HIP
// #define PORT_SYCL
#include "../../comm.h"

// UTILITIES
#define ROOT 0
#include "../../util.h"
using namespace CommBench;
using namespace std;

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    setup_gpu();

    //allocate GPU memory buffer
    int count = 268435456;
    int *sendbuf_d;
    int *recvbuf_d;
    allocate(sendbuf_d, count);//1gb
    allocate(recvbuf_d, count);

    // register communication pattern
    CommBench::printid = 0;
    Comm<int> partition(library::IPC);
    Comm<int> translate(library::NCCL);
    Comm<int> assemble(library::IPC);

    // allocate staging buffer
    int groupsize = 4;
    int *temp_d;
    allocate(temp_d, count / groupsize);

    // compose steps
    for(int i = 1; i < groupsize; i++)
      partition.add(sendbuf_d, i * count / groupsize, temp_d, 0, count / groupsize, 0, i);
    translate.add(sendbuf_d, 0, recvbuf_d, 0, count / groupsize, 0, groupsize);
    for(int i = 1; i < groupsize; i++)
      translate.add(temp_d, 0, temp_d, 0, count / groupsize, i, groupsize + i);
    for(int i = 1; i < groupsize; i++)
      assemble.add(temp_d, 0, recvbuf_d, i * count / groupsize, count / groupsize, groupsize + i, groupsize);

    // create sequence
    vector<Comm<int>> striping = {partition, translate, assemble};

    // steps in isolation
    for(auto comm : striping)
      comm.measure(5, 10, count);

    // measure end-to-end
    measure(striping, 5, 10, count);

    free(sendbuf_d);
    free(recvbuf_d);
    free(temp_d);

    MPI_Finalize();
}
