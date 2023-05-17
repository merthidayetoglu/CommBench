
template <typename T>
class Gather {

  int myid;
  int numproc;
  Comm::Comm<T> *comm;

  public:

  Gather(T *sendbuf, T *recvbuf, size_t count, int root, const MPI_Comm &comm_mpi) {
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    comm = new Comm::Comm<T>(comm_mpi, Comm::NCCL);

    // BUILD GATHER
    for(int p = 0; p < numproc; p++)
      comm->add(sendbuf, 0, recvbuf, p * count, count, p, root);
  };

  void start() { comm->start(); }
  void wait() { comm->wait(); }
};

template <typename T>
class Gather_mixed {

  int myid;
  int numproc;
  Comm::Comm<T> *comm_self;
  Comm::Comm<T> *comm_intra;
  Comm::Comm<T> *comm_inter;

  public:

  Gather_mixed(T *sendbuf, T *recvbuf, size_t count, int root, int groupsize) {
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    comm_self = new Comm::Comm<T>(MPI_COMM_WORLD, Comm::IPC);
    comm_intra = new Comm::Comm<T>(MPI_COMM_WORLD, Comm::IPC);
    comm_inter = new Comm::Comm<T>(MPI_COMM_WORLD, Comm::NCCL);

    // BUILD GATHER
    for(int p = 0; p < numproc; p++) {
      if(p == root)
        comm_self->add(sendbuf, 0, recvbuf, p * count, count, p, root);
      else if(p / groupsize == root / groupsize)
        comm_intra->add(sendbuf, 0, recvbuf, p * count, count, p, root);
      else
        comm_inter->add(sendbuf, 0, recvbuf, p * count, count, p, root);
    } 
  };

  void start() {
    comm_inter->start();
    comm_intra->start();
    comm_self->start();
  }

  void wait() {
    comm_self->wait();
    comm_intra->wait();
    comm_inter->wait();
  }
};
