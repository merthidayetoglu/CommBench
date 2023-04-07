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

#include <cassert>

#define ROOT 0

// HEADERS
// #include <nccl.h>
// #include <rccl.h>
// #include <sycl.hpp>

// PORTS
// #define PORT_CUDA
// #define PORT_HIP
// #define PORT_SYCL

#include "comm.h"

void setup_gpu();

// USER DEFINED TYPE
struct Type
{
  // int tag;
  int data[1];
  // complex<double> x, y, z;
};

int main(int argc, char *argv[])
{
  // INITIALIZE MPI+OPENMP
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

  int cap = atoi(argv[1]);
  int pattern = atoi(argv[2]);
  int direction = atoi(argv[3]);
  size_t count = atoi(argv[4]);
  int warmup = atoi(argv[5]);
  int numiter = atoi(argv[6]);
  int numgpu = atoi(argv[7]);
  int groupsize = atoi(argv[8]);
  int subgroupsize = atoi(argv[9]);

  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of threads per proc: %d\n", numthread);
    printf("Number of warmup %d\n", warmup);
    printf("Number of iterations %d\n", numiter);
    printf("Number of GPUs %d\n", numgpu);
    printf("Group Size: %d\n", groupsize);
    printf("Subgroup Size: %d\n", subgroupsize);

    printf("Library: %d\n", cap);
    printf("Pattern: %d\n", pattern);
    printf("Direction: %d\n", direction);

    printf("Bytes per Type %lu\n", sizeof(Type));
    printf("Peer-to-peer count %ld ( %ld Bytes)\n", count, count * sizeof(Type));
    printf("send buffer: %lu (%.2f GB) recv buffer: %lu (%.2f GB)\n", count, count * sizeof(Type) / 1.e9, count * numproc, count * numproc * sizeof(Type) / 1.e9);
    printf("\n");
  }

  setup_gpu();


    if(pattern == 0)
#include "test_P2P.h"
    if(pattern == 1)
#include "test_RAIL.h"
    if(pattern == 2)
#include "test_FULL.h"
    if(pattern == 3)
#include "test_FAN.h"

  /*{
    Type *sendbuf_d;
    Type *recvbuf_d;
    cudaMalloc(&sendbuf_d, count * sizeof(Type));
    cudaMalloc(&recvbuf_d, count * sizeof(Type));

    CommBench::Comm<Type> comm(MPI_COMM_WORLD, (CommBench::capability)cap);

    comm.add(sendbuf_d, 0, recvbuf_d, 0, count, 0, 1);

    comm.report();

    double data = count * sizeof(Type) / 1.e9;
    double minTime, medTime, maxTime, avgTime;
    comm.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    if(myid == ROOT) {
      printf("TEST_P2P (%d)\n", subgroupsize);
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime * 1e6, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime * 1e6, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data, data / avgTime);
    }

    cudaFree(sendbuf_d);
    cudaFree(recvbuf_d);
  }*/

  /*{
    sycl::queue q(sycl::gpu_selector_v);
    int *sendbuf_d = sycl::malloc_device<int>(count, q);
    int *recvbuf_d = sycl::malloc_device<int>(count, q);

    for(int iter = -warmup; iter < numiter; iter++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();
      if(myid == 0)
        MPI_Send(sendbuf_d, count, MPI_INT, 11, 0, MPI_COMM_WORLD);
      if(myid == 11)
        MPI_Recv(recvbuf_d, count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(myid == 0) printf("time %e bandwidth %e\n", time, count * sizeof(int) / 1.e9 / time);
    }

    sycl::free(sendbuf_d, q);
    sycl::free(recvbuf_d, q);
  }*/

  // FINALIZE
  MPI_Finalize();

  return 0;
} // main()

void setup_gpu() {

  int myid;
  int numproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

#ifdef PORT_CUDA
  if(myid == ROOT)
    printf("CUDA PORT\n");
  // SET DEVICE
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device = myid % deviceCount;
  cudaSetDevice(device);
  // DONE
  // REPORT
  if(myid == ROOT){
    system("nvidia-smi");
    int deviceCount;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceCount(&deviceCount);
    printf("Device %d Count: %d\n", device, deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,0);
    printf("Device %d name: %s\n",0,deviceProp.name);
    printf("Clock Frequency: %f GHz\n",deviceProp.clockRate/1.e9);
    printf("Computational Capabilities: %d, %d\n",deviceProp.major,deviceProp.minor);
    printf("Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
    printf("Maximum constant memory size: %lu\n",deviceProp.totalConstMem);
    printf("Maximum shared memory size per block: %lu\n",deviceProp.sharedMemPerBlock);
    printf("Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf("Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    printf("Warp size: %d\n",deviceProp.warpSize);
    printf("32-bit Reg. per block: %d\n",deviceProp.regsPerBlock);
    printf("\n");
  }
#elif defined PORT_HIP
  if(myid == ROOT)
    printf("HIP PORT\n");
  //DEVICE MANAGEMENT
  int deviceCount;
  hipGetDeviceCount(&deviceCount);
  int device = myid % deviceCount;
  if(myid == ROOT)
    printf("deviceCount: %d\n", deviceCount);
  hipSetDevice(device);
  // DONE
  // REPORT
  if(myid == ROOT)
    system("rocm-smi");
#elif defined PORT_SYCL
  if(myid == ROOT)
    printf("SYCL PORT\n");
  // set affinity through ZE_AFFINITY_MASK
  // REPORT
  //char *test = getenv("ZE_AFFINITY_MASK");
  //printf("myid %d ZE_AFFINITY_MASK %s\n", myid, test);
#else
  if(myid == ROOT)
    printf("CPU VERSION\n");
  // DONE
#endif
}
