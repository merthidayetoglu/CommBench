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

  Comm<int> test1(IPC_get);
  for(int sender = 0; sender < numproc; sender++){
     for(int recver = 0; recver < numproc; recver++){
  	test1.add(sendbuf, 0, recvbuf, sender * numbytes, numbytes, sender, recver);
     }
  }

  //test1.measure(5, 10, numbytes * numproc);

  validate(sendbuf, recvbuf, numbytes, 6 /*ALLGATHER*/, test1);

  free(sendbuf);
  free(recvbuf);

  MPI_Finalize();

  return 0;
}
