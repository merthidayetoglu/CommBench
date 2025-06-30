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
  int myid = CommBench::myid;

  allocate(sendbuf, numbytes * numproc);
  allocate(recvbuf, numbytes * numproc);

  Comm<int> test1(MPI);
  for(int p = 0; p < numproc; p++){
  	test1.add(sendbuf, 0, recvbuf, 0, numbytes, ROOT, p);
  }

  //test1.measure(5, 10, numbytes * numproc);

  validate(sendbuf, recvbuf, numbytes, 3 /*BCAST*/, test1);

  free(sendbuf);
  free(recvbuf);

  MPI_Finalize();

  return 0;
}
