
// HEADERS
// #define PORT_CUDA
// #define PORT_HIP
 #define PORT_SYCL
#include "comm.h"

// UTILITIES
#define ROOT 0
#include "util.h"
using namespace CommBench;
using namespace std;

int main() {

    //allocate GPU memory buffer
    int count = 268435456;
    int *sendbuf_d;
    int *recvbuf_d;
    allocate(sendbuf_d, count);//1gb
    allocate(recvbuf_d, count);

    // register communication pattern
    Comm<int> split(library::MPI);
    Comm<int> translate(library::MPI);
    Comm<int> assemble(library::MPI);

    // allocate staging buffer
    int groupsize = 12;
    int *temp_d;
    allocate(temp_d, count / groupsize);

    // compose steps
    for(int i = 1; i < groupsize; i++)
      split.add(sendbuf_d, i * count / groupsize, temp_d, 0, count / groupsize, 0, i);
    translate.add(sendbuf_d, 0, recvbuf_d, 0, count / groupsize, 0, groupsize);
    for(int i = 1; i < groupsize; i++)
      translate.add(temp_d, 0, temp_d, 0, count / groupsize, i, groupsize + i);
    for(int i = 1; i < groupsize; i++)
      assemble.add(temp_d, 0, recvbuf_d, i * count / groupsize, count / groupsize, groupsize + i, groupsize);

    // steps in isolation
    split.measure(5, 10);
    translate.measure(5, 10);
    assemble.measure(5, 10);

    // create sequence
    vector<Comm<int>> striping = {split, translate, assemble};

    // measure end-to-end
    measure_async(striping, 5, 10, count);

    free(sendbuf_d);
    free(recvbuf_d);
    free(temp_d);

    return 0;
}
