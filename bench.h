
#include "comm.h"

namespace CommBench
{
  enum workload {P2P, P2G, G2G_rail, G2G_full, A2A};

  template <typename T>
  class Bench
  {
    const workload work;
    const int groupsize;
    const size_t count;
    Comm<T> *comm;

    T *sendbuf;
    T *recvbuf;

    public:

    void start() { comm->init(); };
    void wait() { comm->wait(); };

    Bench(const MPI_Comm &comm_mpi, const int groupsize, const CommBench::workload work, CommBench::capability cap, const size_t count) : work(work), groupsize(groupsize), count(count) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      int numgroup = numproc / groupsize;

      comm = new Comm<T>(comm_mpi, cap);

      if(myid == ROOT)
        printf("CommBench: Create bench with %d processes, %d groups of %d\n", numproc, numgroup, groupsize);

      switch(work) {
        case P2P:
	  if(myid == ROOT) {
            printf("CommBench: P2P Communication\n");
            printf("CommBench: allocate %.2f MB comm buffer per GPU\n", (count + count * (numgroup - 1)) * sizeof(T) / 1.e6);
          }
#ifdef PORT_CUDA
          cudaMalloc(&sendbuf, count * sizeof(T));
          cudaMalloc(&recvbuf, count * (numgroup - 1) * sizeof(T));
#elif defined PORT_HIP
          hipMalloc(&sendbuf, count * sizeof(T));
          hipMalloc(&recvbuf, count * (numgroup - 1) * sizeof(T));
#else
          sendbuf = new T[count];
          recvbuf = new T[count * (numgroup - 1) * sizeof(T)];
#endif
          for(int recver = 0; recver < numproc; recver++) {
            int mygroup = recver / groupsize;
            int mylocalid = recver % groupsize;
            int numrecv = 0;
            if(mylocalid == 0)
              for (int group = 0; group < numgroup; group++)
                if(group != mygroup) {
                  int sender = group * groupsize + mylocalid;
                  comm->add(sendbuf, 0, recvbuf, numrecv * count, count, sender, recver);
                  numrecv++;
                }
          }
          break;
        case P2G:
          if(myid == ROOT) {
            printf("CommBench: P2G Communication\n");
            printf("CommBench: allocate %.2f MB comm buffer per GPU\n", (count + count * (numgroup - 1) * groupsize) * sizeof(T) / 1.e6);
          }
#ifdef PORT_CUDA
          cudaMalloc(&sendbuf, count * sizeof(T));
          cudaMalloc(&recvbuf, count * (numgroup - 1) * groupsize * sizeof(T));
#elif defined PORT_HIP
          hipMalloc(&sendbuf, count * sizeof(T));
          hipMalloc(&recvbuf, count * (numgroup - 1) * groupsize * sizeof(T));
#else       
          sendbuf = new T[count];
          recvbuf = new T[count * (numgroup - 1) * groupsize * sizeof(T)];
#endif
          for(int recver = 0; recver < numproc; recver++) {
            int mygroup = recver / groupsize;
            int mylocalid = recver % groupsize;
            int numrecv = 0;
            if(mylocalid == 0)
              for (int group = 0; group < numgroup; group++)
                if(group != mygroup)
                  for(int p = 0; p < groupsize; p++)  {
                    int sender = group * groupsize + p;
                    comm->add(sendbuf, 0, recvbuf, numrecv * count, count, sender, recver);
                    numrecv++;
                  }
          }
          break;
        case G2G_rail:
          if(myid == ROOT) {
            printf("CommBench: G2G (Rail) Communication\n");
            printf("CommBench: allocate %.2f MB comm buffer per GPU\n", (count + count * (numgroup - 1)) * sizeof(T) / 1.e6);
          }
#ifdef PORT_CUDA
          cudaMalloc(&sendbuf, count * sizeof(T));
          cudaMalloc(&recvbuf, count * (numgroup - 1) * sizeof(T));
#elif defined PORT_HIP
          hipMalloc(&sendbuf, count * sizeof(T));
          hipMalloc(&recvbuf, count * (numgroup - 1) * sizeof(T));
#else
          sendbuf = new T[count];
          recvbuf = new T[count * (numgroup - 1) * sizeof(T)];
#endif
         for(int recver = 0; recver < numproc; recver++) {
            int mygroup = recver / groupsize;
            int mylocalid = recver % groupsize;
            int numrecv = 0;
            for (int group = 0; group < numgroup; group++)
              if(group != mygroup) {
                int sender = group * groupsize + mylocalid;
                comm->add(sendbuf, 0, recvbuf, numrecv * count, count, sender, recver);
                numrecv++;
              }
          }
          break;
        case G2G_full:
          if(myid == ROOT) {
            printf("CommBench: G2G (Full) Communication\n");
            printf("CommBench: allocate %.2f MB comm buffer per GPU\n", (count + count * (numgroup - 1) * groupsize) * sizeof(T) / 1.e6);
          }
#ifdef PORT_CUDA
          cudaMalloc(&sendbuf, count * sizeof(T));
          cudaMalloc(&recvbuf, count * (numgroup - 1) * groupsize * sizeof(T));
#elif defined PORT_HIP
          hipMalloc(&sendbuf, count * sizeof(T));
          hipMalloc(&recvbuf, count * (numgroup - 1) * groupsize * sizeof(T));
#else
          sendbuf = new T[count];
          recvbuf = new T[count * (numgroup - 1) * groupsize * sizeof(T)];
#endif
         for(int recver = 0; recver < numproc; recver++) {
            int mygroup = recver / groupsize;
            int mylocalid = recver % groupsize;
            int numrecv = 0;
            for (int group = 0; group < numgroup; group++)
              if(group != mygroup)
                for(int p = 0; p < groupsize; p++) {
                  int sender = group * groupsize + p;
                  comm->add(sendbuf, 0, recvbuf, numrecv * count, count, sender, recver);
                  numrecv++;
                }
          }
          break;
        case A2A:
          if(myid == ROOT) {
            printf("CommBench: A2A Communication\n");
            printf("CommBench: allocate %.2f MB comm buffer per GPU\n", (count + count * groupsize) * sizeof(T) / 1.e6);
          }
#ifdef PORT_CUDA
          cudaMalloc(&sendbuf, count * sizeof(T));
          cudaMalloc(&recvbuf, count * groupsize * sizeof(T));
#elif defined PORT_HIP
          hipMalloc(&sendbuf, count * sizeof(T));
          hipMalloc(&recvbuf, count * groupsize * sizeof(T));
#else
          sendbuf = new T[count];
          recvbuf = new T[count * groupsize * sizeof(T)];
#endif
          for(int recver = 0; recver < numproc; recver++) {
            int mygroup = recver / groupsize;
            int mylocalid = recver % groupsize;
            int numrecv = 0;
            for(int p = 0; p < groupsize; p++) {
              int sender = mygroup * groupsize + p;
              comm->add(sendbuf, 0, recvbuf, numrecv * count, count, sender, recver);
              numrecv++;
            }
          }
          break;
      }
      comm->report();
    }
  }; // class Bench


} // namespace CommBench*/
