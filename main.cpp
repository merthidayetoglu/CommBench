#include "commbench.h"

using namespace CommBench;

int main() {

  char *sendbuf;
  char *recvbuf;
  size_t numbytes = 1e9;

  init();
  allocate(sendbuf, numbytes);
  allocate(recvbuf, numbytes);

  Comm<char> test(NCCL);
  test.add(sendbuf, recvbuf, numbytes, 0, 0);

  test.measure(5, 10);

  free(sendbuf);
  free(recvbuf);

  return 0;
}
