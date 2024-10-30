#include "commbench.h"

#define ROOT 0
#include "validate.h"

using namespace CommBench;

int main() {

  int *sendbuf;
  int *recvbuf;
  size_t numbytes = 1e8;

  init();
  int numproc = CommBench::numproc;
 
  allocate(sendbuf, numbytes);
  allocate(recvbuf, numbytes);

  Comm<int> test1(NCCL);
  test1.add(sendbuf, recvbuf, numbytes, 0, 1);

  test1.measure(5, 10, numbytes * numproc);

  validate(sendbuf, recvbuf, numbytes, 0 /*P2P*/, test1);

  free(sendbuf);
  free(recvbuf);

  MPI_Finalize();

  return 0;
}
