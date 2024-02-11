#define PORT_SYCL
#import "comm.h"

using namespace CommBench;

int main() {

  char *sendbuf;
  char *recvbuf;
  size_t numbytes = 128e6;
  allocate(sendbuf, numbytes);
  allocate(recvbuf, numbytes);

  Comm<char> test(CommBench::IPC);
  for(int i = 0; i < CommBench::numproc; i++)
    test.add(sendbuf, recvbuf, numbytes, i, 0);
  // test.add(sendbuf, recvbuf, numbytes, 0, 1);
  test.measure(5, 10);

  free(sendbuf);
  free(recvbuf);

  return 0;
}
