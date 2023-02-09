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
#include <mpi.h>
#include <omp.h>

#include <cassert>

#define ROOT 1

// HEADERS
 #include <nccl.h>
// #include <rccl.h>

// PORTS AND CAPS
 #define PORT_CUDA
// #define PORT_HIP
// #define PORT_SYCL
 #define CAP_NCCL

//#include "comm.h"
#include "bench.h"

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

  size_t count = atoi(argv[1]);
  int warmup = atoi(argv[2]);
  int numiter = atoi(argv[3]);
  int groupsize = atoi(argv[4]);
  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of threads per proc: %d\n", numthread);
    printf("Number of warmup %d\n", warmup);
    printf("Number of iterations %d\n", numiter);
    printf("Group Size: %d\n", groupsize);

    printf("Bytes per Type %lu\n", sizeof(Type));
    printf("Peer-to-peer count %ld ( %ld Bytes)\n", count, count * sizeof(Type));
    printf("send buffer: %lu (%.2f GB) recv buffer: %lu (%.2f GB)\n", count, count * sizeof(Type) / 1.e9, count * numproc, count * numproc * sizeof(Type) / 1.e9);
    printf("\n");
  }

  setup_gpu();

#include "test_saturation.h"

  //CommBench::Bench<Type>(MPI_COMM_WORLD, groupsize, CommBench::across, CommBench::MPI, count);
  //CommBench::Bench<Type>(MPI_COMM_WORLD, 1, CommBench::across, CommBench::MPI, count);

  return 0;

  Type *sendbuf_d;
  Type *recvbuf_d;
  Type *recvbuf_2;
  cudaMalloc(&sendbuf_d, count * sizeof(Type));
  cudaMalloc(&recvbuf_d, count * 2 * numproc *sizeof(Type));
  cudaMalloc(&recvbuf_2, count * 2 * numproc *sizeof(Type));

  CommBench::Comm<Type> test(MPI_COMM_WORLD, CommBench::MPI);


  /*test.add(sendbuf_d, 0, recvbuf_d, 0, count, 1, 2);
  test.add(sendbuf_d, 0, recvbuf_d, 0, count, 2, 1);
  test.add(sendbuf_d, 0, recvbuf_d, 0, 20, 2, 1);
  test.add(sendbuf_d, 0, recvbuf_2, 0, count, 2, 1);
  test.add(sendbuf_d, 0, recvbuf_2, 0, count, 2, 1);
  test.add(sendbuf_d, 0, recvbuf_d, 0, count, 1, 2);*/

  for(int sender = 0; sender < numproc; sender++)
    for(int recver = 0; recver < numproc; recver++)
      if(sender != recver)
        test.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);

  /*for(int p = 0; p < numproc; p++)
    if(p != 0)
      test.add(sendbuf_d, 0, recvbuf_2, p * count, count, p, 0);

  for(int p = 0; p < numproc; p++)
    if(p != 0)
      test.add(sendbuf_d, 0, recvbuf_d, p * count, count, 0, p);*/

  test.report();

  test.init();
  test.wait();

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
#else
  if(myid == ROOT)
    printf("CPU VERSION\n");
  // DONE
#endif
}
