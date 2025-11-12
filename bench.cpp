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
  int max_num_bytes = 31;
  char file_name[100];
  int gpus = numproc;

  allocate(sendbuf, 1ull << max_num_bytes);
  allocate(recvbuf, 1ull << max_num_bytes);

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
    for (int source = 0; source < gpus; source++) {
      for (int dest = 0; dest < gpus; dest++) {
        if (myid == 0)
          fprintf(stderr, "\033[2K\rP2P Comm: %d->%d", source, dest);
          // fprintf(stderr, "P2P Comm: %d->%d\n", source, dest);
        for (int size = 2; size < max_num_bytes; size+=2) {
          Comm<int> bench(lib);
          if (myid == 0) {
            sprintf(file_name, "%s/%s/%d_%d_%llu.out", folder,
                    lib_str, source, dest, 1ull << size);
            freopen(file_name, "w", stdout);
          }
          bench.add(sendbuf, recvbuf, 1ull << size, source, dest);
          bench.measure(5, 10);
          // bench.clear();
        }
      }
    }

    if (myid == 0)
      fprintf(stderr, "\033[2K\rCompleted testing library %s\n", lib_str);
      // fprintf(stderr, "Completed testing library %s\n", libs_str[i].c_str());
  }
  free(sendbuf);
  free(recvbuf);
  MPI_Finalize();
}
