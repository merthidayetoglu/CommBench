#include "commbench.h"
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace CommBench;

int main(int argc, char *argv[]) {
  init();

  if (argc < 3) {
    if (myid == 0)
      fprintf(stderr, "Invalid arguments, usage is ./bench <folder> <libs>\n");
    return 1;
  }
  char* folder = argv[1];

  int *sendbuf, *recvbuf;
  int max_num_bytes = 27;
  char file_name[100];
  int gpus = numproc;

  allocate(sendbuf, 1ull << max_num_bytes);
  allocate(recvbuf, (1ull << max_num_bytes) * numproc);

  for (int i = 2; i < argc; i++) {
    char* lib_str = argv[i];
    library lib;
    if (strcmp(lib_str, "mpi") == 0)
      lib = library::MPI;
    else if (strcmp(lib_str, "ipc_put") == 0)
      lib = library::IPC;
    else if (strcmp(lib_str, "ipc_get") == 0)
      lib = library::IPC_get;
    else if (strcmp(lib_str, "xccl") == 0)
      lib = library::NCCL;
    else {
      if (myid == 0)
        fprintf(stderr, "Invalid library %s\n", lib_str);
      return 1;
    }
    if (myid == 0)
      fprintf(stderr, "Testing library %s\n", lib_str);
      for (int dest = 0; dest < gpus; dest++) {
          // fprintf(stderr, "P2P Comm: %d->%d\n", source, dest);
        for (int size = 2; size < max_num_bytes; size+=2) {
          if (myid == 0)
            fprintf(stderr, "\033[2K\rGatherer: %d, size: %llu", dest, 1ull << size);
          Comm<int> bench(lib);
          if (myid == 0) {
            sprintf(file_name, "%s/%s/%d_%llu.out", folder,
                    lib_str, dest, 1ull << size);
            freopen(file_name, "w", stdout);
          }
          for (int p = 0; p < numproc; p++)
            bench.add(sendbuf, 0, recvbuf, p * (1ull << size), 1ull << size, p, dest);
          bench.measure(5, 10);
        }
    }

    if (myid == 0)
      fprintf(stderr, "\033[2K\rCompleted testing library %s\n", lib_str);
  }
  free(sendbuf);
  free(recvbuf);
  MPI_Finalize();
}
