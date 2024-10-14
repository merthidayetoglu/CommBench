#include "commbench.h"

using namespace CommBench;

int main() {

  char *sendbuf;
  char *recvbuf;
  size_t numbytes = 1e9;

  init();
  allocate(sendbuf, numbytes);
  allocate(recvbuf, numbytes);

  Comm<char> test1(MPI);
  test1.add(sendbuf, recvbuf, numbytes, 0, 0);

  test1.measure(5, 10);

  Comm<char> test2(IPC);
  test2.add(sendbuf, recvbuf, numbytes, 0, 0);

  test2.measure(5, 10);


  Comm<char> test3(NCCL);
  test3.add(sendbuf, recvbuf, numbytes, 0, 0);

  test3.measure(5, 10);

  free(sendbuf);
  free(recvbuf);

  return 0;
}
