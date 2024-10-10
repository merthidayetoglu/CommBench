#include "commbench.h"

// MEMORY MANAGEMENT
void CommBench::barrier() {
#ifdef USE_MPI
    MPI_Barrier(comm_mpi);
#endif
#ifdef CAP_GASNET
    gex_Event_Wait(gex_Event_WaitAll(myteam, NULL));
#endif
}

template <typename T>
void CommBench::allocate(T*& buffer, size_t n) {
    buffer = new T[n];
}

template <typename T>
void CommBench::allocateHost(T*& buffer, size_t n) {
    buffer = static_cast<T*>(malloc(n * sizeof(T)));
}

template <typename T>
void CommBench::memcpyD2H(T* host, T* device, size_t n) {
#if defined PORT_CUDA
    cudaMemcpy(host, device, n * sizeof(T), cudaMemcpyDeviceToHost);
#elif defined PORT_HIP
    hipMemcpy(host, device, n * sizeof(T), hipMemcpyDeviceToHost);
#elif defined PORT_ONEAPI
    q.memcpy(host, device, n * sizeof(T)).wait();
#endif
}

template <typename T>
void CommBench::memcpyH2D(T* device, T* host, size_t n) {
#if defined PORT_CUDA
    cudaMemcpy(device, host, n * sizeof(T), cudaMemcpyHostToDevice);
#elif defined PORT_HIP
    hipMemcpy(device, host, n * sizeof(T), hipMemcpyHostToDevice);
#elif defined PORT_ONEAPI
    q.memcpy(device, host, n * sizeof(T)).wait();
#endif
}

template <typename T>
void CommBench::free(T* buffer) {
    delete[] buffer;
}

template <typename T>
void CommBench::freeHost(T* buffer) {
    free(buffer);
}

// PAIR COMMUNICATION
template <typename T>
void CommBench::send(T* sendbuf, int recvid) {
#ifdef USE_GASNET
    GASNET_BLOCKUNTIL(am_ready[recvid]);
    am_ready[recvid] = false;
    gex_AM_RequestMedium0(myteam, recvid, am_send_index, sendbuf, sizeof(T), GEX_EVENT_NOW, 0);
#else
    MPI_Ssend(sendbuf, sizeof(T), MPI_BYTE, recvid, 0, comm_mpi);
#endif
}

template <typename T>
void CommBench::recv(T* recvbuf, int sendid) {
#ifdef USE_GASNET
    am_ptr = recvbuf;
    am_busy = true;
    gex_AM_RequestShort1(myteam, sendid, am_recv_index, 0, myid);
    GASNET_BLOCKUNTIL(!am_busy);
#else
    MPI_Recv(recvbuf, sizeof(T), MPI_BYTE, sendid, 0, comm_mpi, MPI_STATUS_IGNORE);
#endif
}

template <typename T>
void CommBench::pair(T* sendbuf, T* recvbuf, int sendid, int recvid) {
    if (sendid == recvid) {
        if (myid == sendid)
            memcpy(recvbuf, sendbuf, sizeof(T));
        return;
    }
    if (myid == sendid)
        send(sendbuf, recvid);
    if (myid == recvid)
        recv(recvbuf, sendid);
}

template <typename T>
void CommBench::broadcast(T* sendbuf, T* recvbuf, int root) {
    T temp;
    for (int i = 0; i < numproc; i++)
        pair(sendbuf, &temp, root, i);
    *recvbuf = temp;
}

template <typename T>
void CommBench::broadcast(T* sendbuf) {
    broadcast(sendbuf, sendbuf, 0);
}

template <typename T>
void CommBench::allgather(T* sendval, T* recvbuf) {
    for (int root = 0; root < numproc; root++)
        broadcast(sendval, recvbuf + root, root);
}

template <typename T>
void CommBench::allreduce_sum(T* sendbuf, T* recvbuf) {
    std::vector<T> temp(numproc);
    allgather(sendbuf, temp.data());
    T sum = 0;
    for (int i = 0; i < numproc; i++)
        sum += temp[i];
    *recvbuf = sum;
}

template <typename T>
void CommBench::allreduce_sum(T* sendbuf) {
    allreduce_sum(sendbuf, sendbuf);
}

template <typename T>
void CommBench::allreduce_max(T* sendbuf, T* recvbuf) {
    std::vector<T> temp(numproc);
    allgather(sendbuf, temp.data());
    T max = *sendbuf;
    for (int i = 0; i < numproc; i++)
        if (temp[i] > max)
            max = temp[i];
    *recvbuf = max;
}

template <typename T>
void CommBench::allreduce_max(T* sendbuf) {
    allreduce_max(sendbuf, sendbuf);
}

char CommBench::allreduce_land(char logic) {
    std::vector<char> temp(numproc);
    allgather(&logic, temp.data());
    for (int i = 0; i < numproc; i++)
        if (temp[i] == 0)
            return 0;
    return 1;
}

// MEASUREMENT
template <typename C>
void CommBench::measure(int warmup, int numiter, double& minTime, double& medTime, double& maxTime, double& avgTime, C& comm) {
    std::vector<double> times;
    for (int iter = -warmup; iter < numiter; iter++) {
        barrier();
        double time = omp_get_wtime();
        comm.start();
        comm.wait();
        time = omp_get_wtime() - time;
        times.push_back(time);
    }
    std::sort(times.begin(), times.end());
    minTime = times[0];
    medTime = times[times.size() / 2];
    maxTime = times.back();
    avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
}

template <typename T>
void CommBench::measure_async(std::vector<Comm<T>> commlist, int warmup, int numiter, size_t count) {
    std::vector<double> t;
    for (int iter = -warmup; iter < numiter; iter++) {
        barrier();
        double time = omp_get_wtime();
        for (auto& i : commlist) {
            i.start();
            i.wait();
        }
        time = omp_get_wtime() - time;
        allreduce_max(&time);
        if (iter >= 0)
            t.push_back(time);
    }
    print_stats(t, count * sizeof(T));
}

template <typename T>
void CommBench::measure_concur(std::vector<Comm<T>> commlist, int warmup, int numiter, size_t count) {
    std::vector<double> t;
    for (int iter = -warmup; iter < numiter; iter++) {
        barrier();
        double time = omp_get_wtime();
        for (auto& i : commlist) {
            i.start();
        }
        for (auto& i : commlist) {
            i.wait();
        }
        time = omp_get_wtime() - time;
        allreduce_max(&time);
        if (iter >= 0)
            t.push_back(time);
    }
    print_stats(t, count * sizeof(T));
}

#ifdef USE_MPI
template <typename T>
void CommBench::measure_MPI_Alltoallv(std::vector<std::vector<int>> pattern, int warmup, int numiter) {
    std::vector<int> sendcount;
    std::vector<int> recvcount;
    for (int i = 0; i < numproc; i++) {
        sendcount.push_back(pattern[myid][i]);
        recvcount.push_back(pattern[i][myid]);
    }
    std::vector<int> senddispl(numproc + 1, 0);
    std::vector<int> recvdispl(numproc + 1, 0);
    for (int i = 1; i < numproc + 1; i++) {
        senddispl[i] = senddispl[i - 1] + sendcount[i - 1];
        recvdispl[i] = recvdispl[i - 1] + recvcount[i - 1];
    }

    T* sendbuf;
    T* recvbuf;
    allocate(sendbuf, senddispl[numproc]);
    allocate(recvbuf, recvdispl[numproc]);

    for (int p = 0; p < numproc; p++) {
        sendcount[p] *= sizeof(T);
        recvcount[p] *= sizeof(T);
        senddispl[p] *= sizeof(T);
        recvdispl[p] *= sizeof(T);
    }

    std::vector<double> t;
    for (int iter = -warmup; iter < numiter; iter++) {
        barrier();
        double time = omp_get_wtime();
        MPI_Alltoallv(sendbuf, &sendcount[0], &senddispl[0], MPI_BYTE, recvbuf, &recvcount[0], &recvdispl[0], MPI_BYTE, comm_mpi);
        time = omp_get_wtime() - time;
        allreduce_max(&time);
        if (iter >= 0)
            t.push_back(time);
    }
    print_stats(t, 0);
    free(sendbuf);
    free(recvbuf);
}
#endif

void CommBench::print_data(size_t data) {
    if (data < 1e3)
        printf("%d bytes", (int)data);
    else if (data < 1e6)
        printf("%.4f KB", data / 1e3);
    else if (data < 1e9)
        printf("%.4f MB", data / 1e6);
    else if (data < 1e12)
        printf("%.4f GB", data / 1e9);
    else
        printf("%.4f TB", data / 1e12);
}

void CommBench::print_lib(library lib) {
    switch (lib) {
    case dummy:
        printf("dummy");
        break;
    case IPC:
        printf("IPC (PUT)");
        break;
    case IPC_get:
        printf("IPC (GET)");
        break;
    case MPI:
        printf("MPI");
        break;
    case NCCL:
        printf("NCCL");
        break;
    case GEX:
        printf("GASNET (PUT)");
        break;
    case GEX_get:
        printf("GASNET (GET)");
        break;
    case numlib:
        printf("numlib");
        break;
    }
}

void CommBench::init() {
    static bool initialized = false;
    if (initialized)
        return;
    initialized = true;

#ifdef USE_MPI
    int init_mpi;
    MPI_Initialized(&init_mpi);
    if (!init_mpi) {
        MPI_Init(NULL, NULL);
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_mpi);
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);
#endif
}

void CommBench::finalize() {
    static bool finalized = false;
    if (finalized)
        return;
    finalized = true;

#ifdef USE_MPI
    MPI_Finalize();
#endif
}

void CommBench::print_stats(std::vector<double> times, size_t data) {
    std::sort(times.begin(), times.end());
    double minTime = times[0];
    double medTime = times[times.size() / 2];
    double maxTime = times.back();
    double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    printf("Min: %.6f s, Median: %.6f s, Max: %.6f s, Average: %.6f s, Data: %lu bytes\n", minTime, medTime, maxTime, avgTime, data);
}


