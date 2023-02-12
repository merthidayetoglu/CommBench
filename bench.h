
#include "comm.h"

namespace CommBench
{

  class Arch {

    public:

    const int numlevel;

    MPI_Comm *comm_within = new MPI_Comm[numlevel + 1];
    MPI_Comm *comm_across = new MPI_Comm[numlevel + 1];

    Arch(const int numlevel, int groupsize[], const MPI_Comm &comm_world) : numlevel(numlevel) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_world, &myid);
      MPI_Comm_size(comm_world, &numproc);

      MPI_Comm_split(comm_world, myid / numproc, myid % numproc, comm_within);
      MPI_Comm_split(comm_world, myid % numproc, myid / numproc, comm_across);
      for(int level = 1; level < numlevel; level++) {
        int myid_within;
        MPI_Comm_rank(comm_within[level - 1], &myid_within);
        MPI_Comm_split(comm_within[level - 1], myid_within / groupsize[level - 1], myid_within % groupsize[level - 1], comm_within + level);
        MPI_Comm_split(comm_within[level - 1], myid_within % groupsize[level - 1], myid_within / groupsize[level - 1], comm_across + level);
      }
      int myid_within;
      MPI_Comm_rank(comm_within[numlevel - 1], &myid_within);
      MPI_Comm_split(comm_within[numlevel - 1], myid_within / 1, myid_within % 1, comm_within + numlevel);
      MPI_Comm_split(comm_within[numlevel - 1], myid_within % 1, myid_within / 1, comm_across + numlevel);

      for(int level = 0; level < numlevel + 1; level++) {
        int numproc_within;
        int numproc_across;
        MPI_Comm_size(comm_within[level], &numproc_within);
        MPI_Comm_size(comm_across[level], &numproc_across);
        if(myid == ROOT)
          printf("level %d numproc_within %d numproc_across %d\n", level, numproc_within, numproc_across);
      }
    } 

  };

  enum heuristic {across, within};
  enum direction {unidirect, bidirect};

  template <typename T>
  class Bench
  {

    MPI_Comm commgroup;
    const capability cap;
    Comm<T> *transport;

    const heuristic mode;
    const size_t count;

    T *sendbuf;
    T *recvbuf;

    public:

    void start() { transport->init(); };
    void wait() { transport->wait(); };
    void test();

    ~Bench() {
      //delete transport;
#ifdef PORT_CUDA
     // cudaFree(sendbuf);
     // cudaFree(recvbuf);
#elif defined PORT_HIP
     // hipFree(sendbuf);
     // hipFree(recvbuf);
#else
     // delete[] sendbuf;
     // delete[] recvbuf;
#endif
    };

    Bench(const MPI_Comm &comm_world, const int groupsize, const heuristic mode, CommBench::capability cap, const size_t count) : mode(mode), count(count), cap(cap) {

      int myid_root;
      MPI_Comm_rank(MPI_COMM_WORLD, &myid_root);
      if(myid_root == ROOT)
        printf("CommBench: Creating a Bench object requires global synchronization\n");

      int myid;
      int numproc;
      MPI_Comm_rank(comm_world, &myid);
      MPI_Comm_size(comm_world, &numproc);

      if(myid_root == ROOT)
        printf("CommBench: Create bench with %d processes\n", numproc);

      switch(mode) {
        case across:
          MPI_Comm_split(comm_world, myid % groupsize, myid / groupsize, &commgroup);
          if(myid_root == ROOT)
            printf("CommBench: Split comm across groups of %d\n", groupsize);
          break;
        case within:
          MPI_Comm_split(comm_world, myid / groupsize, myid % groupsize, &commgroup);
          if(myid_root == ROOT)
            printf("CommBench: Split comm within groups of %d\n", groupsize);
          break;
      }

      int mygroup;
      int numgroup;
      MPI_Comm_rank(commgroup, &mygroup);
      MPI_Comm_size(commgroup, &numgroup);

      transport = new CommBench::Comm<T>(commgroup, cap);

      switch(mode) {
        case across:
          {
	    if(myid_root == ROOT) {
              printf("CommBench: There are %d groups to comm. across\n", numgroup);
              printf("CommBench: allocate %.2f MB comm buffer per GPU\n", count * numgroup * sizeof(T) / 1.e6);
	    }
#ifdef PORT_CUDA
            if(myid_root == ROOT) {
              printf("CUDA memory management\n");
            }
            cudaMalloc(&sendbuf, count * sizeof(T));
            cudaMalloc(&recvbuf, count * (numgroup - 1) * sizeof(T));
#elif defined PORT_HIP
            if(myid_root == ROOT)
              printf("HIP memory management\n");
            hipMalloc(&sendbuf, count * sizeof(T));
            hipMalloc(&recvbuf, count * (numgroup - 1) * sizeof(T));
#else
            if(myid_root == ROOT)
              printf("CPU memory management\n");
            sendbuf = new T[count];
            recvbuf = new T[count * (numgroup - 1)];
#endif
            for(int sender = 0; sender < numgroup; sender++) {
              int numrecv = 0;
              for(int recver = 0; recver < numgroup; recver++)
                if(sender != recver) {
                  transport->add(sendbuf, 0, recvbuf, numrecv * count, count, sender, recver);
                  numrecv++;
                }
            }
          }
          break;
        case within:
          {
            if(myid_root == ROOT) {
              printf("CommBench: There are %d processes to comm within\n", numgroup);
              printf("CommBench: allocate %.2f MB comm buffer per GPU\n", count * (numgroup + 1) * sizeof(T) / 1.e6);
            }
#ifdef PORT_CUDA
            if(myid_root == ROOT)
              printf("CUDA memory management\n");
            cudaMalloc(&sendbuf, count * sizeof(T));
            cudaMalloc(&recvbuf, count * numgroup * sizeof(T));
#elif defined PORT_HIP
            if(myid_root == ROOT)
              printf("HIP memory management\n");
            hipMalloc(&sendbuf, count * sizeof(T));
            hipMalloc(&recvbuf, count * numgroup * sizeof(T));
#else
            if(myid_root == ROOT)
              printf("CPU memory management\n");
            sendbuf = new T[count];
            recvbuf = new T[count * numgroup];
#endif
            for(int sender = 0; sender < numgroup; sender++)
              for(int recver = 0; recver < numgroup; recver++)
                  transport->add(sendbuf, 0, recvbuf, sender * count, count, sender, recver);
          }
          break;
      }
    }

  }; // class Bench


  template <typename T>
  void Bench<T>::test() {

    int myid;
    int numproc;
    MPI_Comm_rank(commgroup, &myid);
    MPI_Comm_size(commgroup, &numproc);

    int recvproc;
    switch(mode) {case across: recvproc = numproc - 1; break; case within: recvproc = numproc; break;}

    T *sendbuf = new T[count];
    T *recvbuf = new T[count * recvproc];

    for(size_t i = 0; i < count; i++)
      sendbuf[i].data[0] = myid;
    memset(recvbuf, -1, count * recvproc * sizeof(T));

#ifdef PORT_CUDA
    cudaMemcpy(this->sendbuf, sendbuf, count * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(this->recvbuf, -1, count * recvproc * sizeof(T));
#elif defined PORT_HIP
    hipMemcpy(this->sendbuf, sendbuf, count * sizeof(T), hipMemcpyHostToDevice);
    hipMemset(this->recvbuf, -1, count * recvproc * sizeof(T));
#else
    memcpy(this->sendbuf, sendbuf, count * sizeof(T));
    memset(this->recvbuf, -1, count * recvproc * sizeof(T));
#endif

    this->start();
    this->wait();

#ifdef PORT_CUDA
    cudaMemcpy(recvbuf, this->recvbuf, count * recvproc * sizeof(T), cudaMemcpyDeviceToHost);
#elif defined PORT_HIP
    hipMemcpy(recvbuf, this->recvbuf, count * recvproc * sizeof(T), hipMemcpyDeviceToHost);
#else
    memcpy(recvbuf, this->recvbuf, count * recvproc * sizeof(T));
#endif

    bool pass = true;
    switch(mode) {
      case across:
        recvproc = 0;
        for(int p = 0; p < numproc; p++)
          if(p != myid) {
            for(size_t i = 0; i < count; i++)
              if(recvbuf[recvproc * count + i].data[0] != p)
                pass = false;
            recvproc++;
          }
        break;
      case within:
        for(int p = 0; p < numproc; p++)
          for(size_t i = 0; i < count; i++)
            if(recvbuf[p * count + i].data[0] != p)
              pass = false;
        break;
    }
    MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if(pass && myid == ROOT)
      printf("PASS!\n");
    else
      if(myid == ROOT)
        printf("ERROR!!!!\n");

    delete[] sendbuf;
    delete[] recvbuf;
  }

  /*template <typename T>
  class Allgather {

    const int numlevel;
    const int commDim = 5;

    Comm<T> *transport;
    Comm<T> ****transports = new Comm<T>***[numlevel];

    public:

    void waitall() {
      for(int i = 0; i < 1; i++) {
        for(int level = 0; level < numlevel+1; level++)
          for(int dim = 0; dim < commDim; dim++) {
             Comm<T> *ptr = transports[i][level][dim];
             if(ptr) ptr->start();
          }
        for(int level = 0; level < numlevel+1; level++)
          for(int dim = 0; dim < commDim; dim++) {
             Comm<T> *ptr = transports[i][level][dim];
             if(ptr) ptr->wait();
          }
      }
    }

    Allgather(T *sendbuf, size_t count, T *recvbuf, Arch &arch, capability *cap) : numlevel(arch.numlevel) {

      int myid;
      int numproc;
      MPI_Comm_rank(arch.comm_within[0], &myid);
      MPI_Comm_size(arch.comm_within[0], &numproc);


      bool state[numproc][numproc];
      for(int p1 = 0; p1 < numproc; p1++)
        for(int p2 = 0; p2 < numproc; p2++)
          state[p1][p2] = 0;

      state[myid][myid] = 1;

      MPI_Allreduce(MPI_IN_PLACE, state, numproc * numproc, MPI_C_BOOL, MPI_LOR, arch.comm_within[0]);

      if(myid == ROOT) {
        printf("state matrix\n");
        for(int p1 = 0; p1 < numproc; p1++) {
          for(int p2 = 0; p2 < numproc; p2++)
            printf("%d ", state[p1][p2]);
          printf("\n");
        }
      }

      bool send_tot[numproc] = {false};
      for(int i = 0; i < 2; i++) {

        transports[i] = new Comm<T>**[numlevel + 1];

        for(int level = 0; level < numlevel + 1; level++) {

          int myid_within;
          int myid_across;
          int numproc_within;
          int numproc_across;
          MPI_Comm_size(arch.comm_across[level], &numproc_across);
          MPI_Comm_rank(arch.comm_across[level], &myid_across);
          MPI_Comm_size(arch.comm_within[level], &numproc_within);
          MPI_Comm_rank(arch.comm_within[level], &myid_within);


          transports[i][level] = new Comm<T>*[commDim]();
          int commDim = 0;
          {
            bool found = false;

            bool send[numproc] = {0};

            size_t sendcount[numproc_across] = {0};
            size_t recvcount[numproc_across] = {0};
            size_t sendoffset[numproc_across];
            size_t recvoffset[numproc_across];
  
            for(int proc = 0; proc < numproc_across; proc++) {
              int recvid;
              MPI_Sendrecv(&myid, 1, MPI_INT, proc, 0, &recvid, 1, MPI_INT, proc, 0, arch.comm_across[level], MPI_STATUS_IGNORE);
              if(myid == ROOT)
                printf("myid %d level %d proc %d recvid %d cap %d\n", myid, level, proc, recvid, cap[level]);
              for(int p = 0; p < numproc; p++) {
                if(state[recvid][p] && !send_tot[p]) {
                  if(i == 0) {
                    send[recvid] = true; 
                    sendcount[proc] = count;
                    recvcount[proc] = count;
                    sendoffset[proc] = 0;
                    recvoffset[proc] = recvid * count;
                    found = true;
                  }
                  else {
                    send[recvid] = true;
                    sendcount[proc] = count;
                    recvcount[proc] = count;
                    sendoffset[proc] = ;
                    recvoffset[proc] = recvid * count;
                  }
                  found = true;
                  break;
                }
              }
            }
            for(int p = 0; p < numproc; p++)
              send_tot[p] = send_tot[p] || send[p];

            if(myid == ROOT) {
              printf("level %d ****************************************\n", level);
              printf("myid_within %d numproc_within %d\n", myid_within, numproc_within);
              printf("myid_across %d numproc_across %d\n", myid_across, numproc_across);
              printf("send     ");
              for(int p = 0; p < numproc; p++)
                printf("%d ", send[p]);
              printf("\n");
              printf("send_tot ");
              for(int p = 0; p < numproc; p++)
                printf("%d ", send_tot[p]);
              printf("\n");
            }

            if(i == 0)
              transports[i][level][commDim] = new Comm<T>(sendbuf, sendcount, sendoffset, recvbuf, recvcount, recvoffset, arch.comm_across[level], cap[level]);
            else
              transports[i][level][commDim] = new Comm<T>(recvbuf, sendcount, sendoffset, recvbuf, recvcount, recvoffset, arch.comm_across[level], cap[level]);
            commDim++;

            // DONE
            bool send_all[numproc][numproc];
            MPI_Allgather(send, numproc, MPI_C_BOOL, send_all, numproc, MPI_C_BOOL, arch.comm_within[0]);

            if(myid == ROOT) {
              printf("commDim: %d\n", commDim);
              printf("send_all\n");
              for(int p1 = 0; p1 < numproc; p1++) {
                for(int p2 = 0; p2 < numproc; p2++)
                  printf("%d ", send_all[p1][p2]);
                printf("\n");
              }
            }
          }

        }

        for(int p = 0; p < numproc; p++)
          state[myid][p] = state[myid][p] || send_tot[p];

        MPI_Allreduce(MPI_IN_PLACE, state, numproc * numproc, MPI_C_BOOL, MPI_LOR, arch.comm_within[0]);

        if(myid == ROOT) {
          printf("state matrix\n");
          for(int p1 = 0; p1 < numproc; p1++) {
            for(int p2 = 0; p2 < numproc; p2++)
              printf("%d ", state[p1][p2]);
            printf("\n");
          }
        }


      }
    }

    Allgather(T *sendbuf, size_t count, T *recvbuf, const MPI_Comm &comm, capability cap) : numlevel(1) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm, &myid);
      MPI_Comm_size(comm, &numproc);

      if(myid == ROOT)
        printf("CommBench: Creating Allgather object\n");

      size_t sendcount[numproc];
      size_t recvcount[numproc];
      size_t sendoffset[numproc];
      size_t recvoffset[numproc];
      for(int p = 0; p < numproc; p++) {
        sendcount[p] = count;
        sendoffset[p] = 0;
        recvcount[p] = count;
        recvoffset[p] = p * count;
      }

      transport = new Comm<T>(sendbuf, sendcount, sendoffset, recvbuf, recvcount, recvoffset, comm, cap);
    }

    //~Allgather() {delete transport;};
    void wait() {transport->start(); transport->wait();};
  };*/

} // namespace CommBench*/
