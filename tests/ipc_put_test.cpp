#include "commbench.h"

using namespace CommBench;

int main() {

  char *sendbuf;
  char *recvbuf;
  size_t numbytes = 1e9;

  init();
  allocate(sendbuf, numbytes);
  allocate(recvbuf, numbytes);

  Comm<char> test1(IPC);
  test1.add(sendbuf, recvbuf, numbytes, 0, 1);

  test1.measure(5, 10);

  free(sendbuf);
  free(recvbuf);

  MPI_Finalize();

  return 0;
}
