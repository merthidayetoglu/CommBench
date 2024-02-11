#define PORT_SYCL
#import "comm.h"

using namespace CommBench;

int main() {

  char *sendbuf;
  char *recvbuf;
  size_t numbytes = 128e6;
  allocate(sendbuf, numbytes);
  allocate(recvbuf, numbytes);

  Comm<char> test(IPC);
  for(int i = 1; i < numproc; i++)
    test.add(sendbuf, recvbuf, numbytes, 0, i);

  test.measure(5, 10);

  free(sendbuf);
  free(recvbuf);

  return 0;
}
