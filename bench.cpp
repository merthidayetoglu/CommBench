#include "commbench.h"
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace CommBench;

int main(int argc, char *argv[]) {
  init();

  int *sendbuf, *recvbuf;
  int max_num_bytes = 17;
  library libs[] = {library::MPI, library::IPC_get, library::IPC,
                    library::NCCL};
  std::string libs_str[] = {"mpi", "ipc_get", "ipc_put", "xccl"};
  char file_name[100];
  int gpus = numproc;

  allocate(sendbuf, 1ull << max_num_bytes);
  allocate(recvbuf, 1ull << max_num_bytes);

  for (int i = 0; i < 4; i++) {
    if (myid == 0)
      fprintf(stderr, "Testing library %s\n", libs_str[i].c_str());
    for (int source = 0; source < gpus; source++) {
      for (int dest = 0; dest < gpus; dest++) {
        // fprintf(stderr, "\033[2K\rP2P Comm: %d->%d", source, dest);
        if (myid == 0)
          fprintf(stderr, "P2P Comm: %d->%d\n", source, dest);
        for (int size = 4; size < max_num_bytes; size += 1) {
          Comm<int> bench(libs[i]);
          // if (myid == 0) {
          //   sprintf(file_name, "data_CPX_all_prelim/%s/%d_%d_%llu.out",
          //           libs_str[i].c_str(), source, dest, 1ull << size);
          //   freopen(file_name, "w", stdout);
          // }
          bench.add(sendbuf, recvbuf, 1ull << size, source, dest);
          bench.measure(5, 10);
          // bench.clear();
        }
      }
    }

    // fprintf(stderr, "\033[2K\rCompleted testing library %s\n",
    // libs_str[i].c_str());
    if (myid == 0)
      fprintf(stderr, "Completed testing library %s\n", libs_str[i].c_str());
  }
  free(sendbuf);
  free(recvbuf);
  MPI_Finalize();
}
