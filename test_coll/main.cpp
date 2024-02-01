/* Copyright 2023 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h> // for printf
#include <stdlib.h> // for atoi
#include <cstring> // for memcpy
#include <algorithm> // for sort
#include <mpi.h>
#include <omp.h>

#define ROOT 0

// HEADERS

// PORTS - define here or in Makefile
// #define PORT_CUDA
// #define PORT_HIP
// #define PORT_SYCL

// CONTROL NCCL CAPABILITY
#if defined(PORT_CUDA)
#define CAP_NCCL
#include <nccl.h>
#endif
#if defined(PORT_HIP)
#define CAP_NCCL
#include <rccl.h>
#endif
#if defined(PORT_SYCL)
#include <sycl.hpp>
#endif

// UTILITIES
#include "../util.h"
void print_args();

namespace test {
enum library {MPI, NCCL};
}
enum pattern {pt2pt, gather, scatter, broadcast, reduce, alltoall, allgather, reducescatter, allreduce};

int main(int argc, char *argv[])
{
  // INITIALIZE
  int myid;
  int numproc;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  int numthread;
  #pragma omp parallel
  if(omp_get_thread_num() == 0)
    numthread = omp_get_num_threads();
  //char machine_name[MPI_MAX_PROCESSOR_NAME];
  //int name_len = 0;
  //MPI_Get_processor_name(machine_name, &name_len);
  //printf("myid %d %s\n",myid, machine_name);

  if(argc != 6) {print_args(); MPI_Finalize(); return 0;}
  // INPUT PARAMETERS
  int library = atoi(argv[1]);
  int pattern = atoi(argv[2]);
  size_t count = atol(argv[3]);
  int warmup = atoi(argv[4]);
  int numiter = atoi(argv[5]);


  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of threads per proc: %d\n", numthread);
    printf("Number of warmup %d\n", warmup);
    printf("Number of iterations %d\n", numiter);

    printf("Library: ");
    switch(library) {
      case test::NCCL : printf("NCCL\n"); break;
      case test::MPI  : printf("MPI\n");  break;
    }
    printf("Pattern: ");
    switch(pattern) {
      case gather        : printf("Gather\n");         break;
      case scatter       : printf("Scatter\n");        break;
      case broadcast     : printf("Broadcast\n");      break;
      case reduce        : printf("Reduce\n");         break;
      case alltoall      : printf("All-to-All\n");     break;
      case allgather     : printf("All-Gather\n");     break;
      case reducescatter : printf("Reduce-Scatter\n"); break;
      case allreduce     : printf("All-Reduce\n");     break;
      default : return 0;
    }
    printf("Peer-to-peer count %ld (%ld Bytes)\n", count, count * sizeof(float));
    printf("\n");
  }

  setup_gpu();

  // SETUP NCCL
#ifdef CAP_NCCL
    ncclComm_t comm_nccl;
    ncclUniqueId id;
    if(myid == 0)
      ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm_nccl, numproc, id, myid);
#endif

  float *sendbuf_d;
  float *recvbuf_d;
#ifdef PORT_CUDA
  cudaMalloc(&sendbuf_d, count * sizeof(float) * numproc);
  cudaMalloc(&recvbuf_d, count * sizeof(float) * numproc);
#elif defined PORT_HIP
  hipMalloc(&sendbuf_d, count * sizeof(float) * numproc);
  hipMalloc(&recvbuf_d, count * sizeof(float) * numproc);
#elif defined PORT_SYCL
  sycl::queue q(sycl::gpu_selector_v);
  sendbuf_d = sycl::malloc_device<float>(count * numproc, q);
  recvbuf_d = sycl::malloc_device<float>(count * numproc, q);
#else
  sendbuf_d = new float[count * numproc];
  recvbuf_d = new float[count * numproc];
#endif

  int recvcounts[numproc];
  for(int p = 0; p < numproc; p++)
    recvcounts[p] = count;

  double times[numiter];
  if(myid == ROOT)
    printf("%d warmup iterations (in order):\n", warmup);

  for (int iter = -warmup; iter < numiter; iter++) {
    //INITIALIZE
#ifdef PORT_CUDA
    cudaMemset(sendbuf_d, -1, count * numproc * sizeof(float));
    cudaMemset(recvbuf_d, -1, count * numproc * sizeof(float));
    cudaDeviceSynchronize();
#elif defined PORT_HIP
    hipMemset(sendbuf_d, -1, count * numproc * sizeof(float));
    hipMemset(recvbuf_d, -1, count * numproc * sizeof(float));
    hipDeviceSynchronize();
#endif
    // MEASURE
    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();
    switch(library) {
      case test::MPI :
        switch(pattern) {
          case gather        : MPI_Gather(sendbuf_d, count, MPI_FLOAT, recvbuf_d, count, MPI_FLOAT, ROOT, MPI_COMM_WORLD);  break;
          case scatter       : MPI_Scatter(sendbuf_d, count, MPI_FLOAT, recvbuf_d, count, MPI_FLOAT, ROOT, MPI_COMM_WORLD); break;
          case broadcast     : MPI_Bcast(sendbuf_d, count, MPI_FLOAT, ROOT, MPI_COMM_WORLD);                      break;
          case reduce        : MPI_Reduce(sendbuf_d, recvbuf_d, count, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD); break;
          case alltoall      : MPI_Alltoall(sendbuf_d, count, MPI_FLOAT, recvbuf_d, count, MPI_FLOAT, MPI_COMM_WORLD);      break;
          case allgather     : MPI_Allgather(sendbuf_d, count, MPI_FLOAT, recvbuf_d, count, MPI_FLOAT, MPI_COMM_WORLD);     break;
          case reducescatter : MPI_Reduce_scatter(sendbuf_d, recvbuf_d, recvcounts, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);    break;
          case allreduce     : MPI_Allreduce(sendbuf_d, recvbuf_d, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);    break;
          default            : return 0;
        }
        break;
#ifdef CAP_NCCL
      // note reducescatter takes scalar for count, while MPI takes array of counts
      case test::NCCL :
        switch(pattern) {
          case broadcast     : ncclBcast(sendbuf_d, count, ncclFloat32, ROOT, comm_nccl, 0);                      break;
          case reduce        : ncclReduce(sendbuf_d, recvbuf_d, count, ncclFloat32, ncclSum, ROOT, comm_nccl, 0); break;
          case allgather     : ncclAllGather(sendbuf_d, recvbuf_d, count, ncclFloat32, comm_nccl, 0);                       break;
          case reducescatter : ncclReduceScatter(sendbuf_d, recvbuf_d, count, ncclFloat32, ncclSum, comm_nccl, 0);          break;
          case allreduce     : ncclAllReduce(sendbuf_d, recvbuf_d, count, ncclFloat32, ncclSum, comm_nccl, 0);    break;
          default            : return 0;
        }
#ifdef PORT_CUDA
        cudaStreamSynchronize(0);
#elif defined PORT_HIP
        hipStreamSynchronize(0);
#endif
        break;
#endif
      default:
        return 0;
    }
    // MPI_Barrier(MPI_COMM_WORLD); // eliminate barrier
    time = MPI_Wtime() - time;
    // TAKE MAXIMUM ELAPSED TIME ON ALL PROCESSES
    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(iter < 0) {
      if(myid == ROOT)
        printf("warmup: %e\n", time);
    }
    else
      times[iter] = time;
  }

  std::sort(times, times + numiter,  [](const double & a, const double & b) -> bool {return a < b;});

  if(myid == ROOT) {
    printf("%d measurement iterations (sorted):\n", numiter);
    for(int iter = 0; iter < numiter; iter++) {
      printf("time: %.4e", times[iter]);
      if(iter == 0)
        printf(" -> min\n");
      else if(iter == numiter / 2)
        printf(" -> median\n");
      else if(iter == numiter - 1)
        printf(" -> max\n");
      else
        printf("\n");
    }
    printf("\n");
    double minTime = times[0];
    double medTime = times[numiter / 2];
    double maxTime = times[numiter - 1];
    double avgTime = 0;
    for(int iter = 0; iter < numiter; iter++)
      avgTime += times[iter];
    avgTime /= numiter;
    size_t data = count * sizeof(float);
    switch(library) {
      case test::MPI :
        switch(pattern) {
          case gather        : printf("MPI_Gather\n"); data *= (numproc-1); break;
          case scatter       : printf("MPI_Scatter\n"); data *= (numproc-1); break;
          case broadcast     : printf("MPI_Bcast\n"); data *= (numproc-1); break;
          case reduce        : printf("MPI_Reduce\n"); data *= (numproc-1); break;
          case alltoall      : printf("MPI_Alltoall\n"); data *= numproc*(numproc-1); break;
          case allgather     : printf("MPI_Allgather\n"); data *= numproc*(numproc-1); break;
          case reducescatter : printf("MPI_Reduce_scatter\n"); data *= (numproc+1)*(numproc-1); break;
          case allreduce     : printf("MPI_Allreduce\n"); data *= 2*(numproc-1); break;
        } break;
#ifdef CAP_NCCL
      case test::NCCL :
        switch(pattern) {
          case broadcast     : printf("ncclBcast\n"); data *= (numproc-1); break;
          case reduce        : printf("ncclReduce\n"); data *= (numproc-1); break;
          case allgather     : printf("ncclAllGather\n"); data *= numproc*(numproc-1); break;
          case reducescatter : printf("ncclReduceScatter\n"); data *= (numproc+1)*(numproc-1); break;
          case allreduce     : printf("ncclAllReduce\n"); data *= 2*(numproc-1); break;
        } break;
#endif
    }
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
    printf("\n");
    printf("minTime: %.4e us, %.4e s/GB, %.4e GB/s\n", minTime * 1e6, minTime / data * 1e9, data / minTime / 1e9);
    printf("medTime: %.4e us, %.4e s/GB, %.4e GB/s\n", medTime * 1e6, medTime / data * 1e9, data / medTime / 1e9);
    printf("maxTime: %.4e us, %.4e s/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data * 1e9, data / maxTime / 1e9);
    printf("avgTime: %.4e us, %.4e s/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data * 1e9, data / avgTime / 1e9);
    printf("\n");
  }

#ifdef PORT_CUDA
  cudaFree(sendbuf_d);
  cudaFree(recvbuf_d);
#elif defined PORT_HIP
  hipFree(sendbuf_d);
  hipFree(recvbuf_d);
#elif defined PORT_SYCL
  sycl::free(sendbuf_d, q);
  sycl::free(recvbuf_d, q);
#else
  delete[] sendbuf_d;
  delete[] recvbuf_d;
#endif

#ifdef CAP_NCCL
  // ncclCommDestroy(comm_nccl); // crashes on frontier
#endif

  // FINALIZE
  MPI_Finalize();

  return 0;
} // main()

void print_args() {

  int myid;
  int numproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  if(myid == ROOT) {
    printf("\n");
    printf("Collective tests requires five arguments:\n");
    printf("1. library:\n");
    printf("      %d for NCCL or RCCL\n", test::NCCL);
    printf("      %d for MPI\n", test::MPI);
    printf("2. pattern:\n");
    printf("      %d for Gather\n",        gather);
    printf("      %d for Scatter\n",       scatter);
    printf("      %d for Reduce\n",        reduce);
    printf("      %d for Broadcast\n",     broadcast);
    printf("      %d for Alltoall\n",      alltoall);
    printf("      %d for Allgather\n",     allgather);
    printf("      %d for ReduceScatter\n", reducescatter);
    printf("      %d for Allreduce\n",     allreduce);
    printf("3. count: number of 4-byte elements\n");
    printf("4. warmup: number of warmup rounds\n");
    printf("5. numiter: number of measurement rounds\n");
    printf("where on can run CollBench as\n");
    printf("mpirun ./CommBench library pattern count warmup numiter\n");
    printf("\n");
  }
}

