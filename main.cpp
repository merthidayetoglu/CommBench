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
// #include <nccl.h>
// #include <rccl.h>
// #include <sycl.hpp>
// #include <ze_api.h>

// PORTS
// #define PORT_CUDA
// #define PORT_HIP
// #define PORT_SYCL

#include "comm.h"

// UTILITIES
#include "util.h"
void print_args();

// USER DEFINED TYPE
struct Type
{
  // int tag;
  int data[1];
  // complex<double> x, y, z;
};

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

  if(argc != 10) {print_args(); MPI_Finalize(); return 0;}
  // INPUT PARAMETERS
  int library = atoi(argv[1]);
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

    printf("Library: %d\n", library);
    printf("Pattern: %d\n", pattern);
    printf("Direction: %d\n", direction);

    printf("Bytes per Type %lu\n", sizeof(Type));
    printf("Point-to-point (P2P) count %ld ( %ld Bytes)\n", count, count * sizeof(Type));
    printf("\n");
  }

  setup_gpu();

  int numgroup = numgpu / groupsize;

  // ALLOCATE
  Type *sendbuf_d;
  Type *recvbuf_d;
#ifdef PORT_CUDA
  cudaMalloc(&sendbuf_d, count * sizeof(Type));
  cudaMalloc(&recvbuf_d, count * sizeof(Type));
#elif defined PORT_HIP
  hipMalloc(&sendbuf_d, count * sizeof(Type));
  hipMalloc(&recvbuf_d, count * sizeof(Type));
#elif defined PORT_SYCL
  sycl::queue q(sycl::gpu_selector_v);
  sendbuf_d = sycl::malloc_device<Type>(count, q);
  recvbuf_d = sycl::malloc_device<Type>(count, q);
#else
  sendbuf_d = new Type[count];
  recvbuf_d = new Type[count];
#endif

  {
    CommBench::Comm<Type> bench(MPI_COMM_WORLD, (CommBench::library) library);

    switch(pattern) {
      case 1: // RAIL PATTERN
        switch(direction) {
          case 1: // UNI-DIRECTIONAL
            for(int send = 0; send < subgroupsize; send++)
              for(int recvgroup = 1; recvgroup < numgroup; recvgroup++) {
                int sender = send;
                int recver = recvgroup * groupsize + send;
                bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
              }
            break;
          case 2: // BI-DIRECTIONAL
            for(int send = 0; send < subgroupsize; send++)
              for(int recvgroup = 1; recvgroup < numgroup; recvgroup++) {
                int sender = send;
                int recver = recvgroup * groupsize + send;
                bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
                bench.add(sendbuf_d, 0, recvbuf_d, 0, count, recver, sender);
              }
            break;
          case 3: // OMNI-DIRECTIONAL
            for(int sendgroup = 0; sendgroup < numgroup; sendgroup++)
              for(int recvgroup = 0; recvgroup < numgroup; recvgroup++)
                if(sendgroup != recvgroup)
                  for(int send = 0; send < subgroupsize; send++) {
                    int sender = sendgroup * groupsize + send;
                    int recver = recvgroup * groupsize + send;
                    bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
                  }
            break;
        }
        break;
      case 2: // DENSE PATTERN
        switch(direction) {
          case 1: // UNI-DIRECTIONAL
            for(int send = 0; send < subgroupsize; send++)
              for(int recvgroup = 1; recvgroup < numgroup; recvgroup++)
                for(int recv = 0; recv < subgroupsize; recv++) {
                  int sender = send;
                  int recver = recvgroup * groupsize + recv;
                  bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
                }
            break;
          case 2: // BI-DIRECTIONAL
            for(int send = 0; send < subgroupsize; send++)
              for(int recvgroup = 1; recvgroup < numgroup; recvgroup++)
                for(int recv = 0; recv < subgroupsize; recv++) {
                  int sender = send;
                  int recver = recvgroup * groupsize + recv;
                bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
                bench.add(sendbuf_d, 0, recvbuf_d, 0, count, recver, sender);
              }
            break;
          case 3: // OMNI-DIRECTIONAL
            for(int sendgroup = 0; sendgroup < numgroup; sendgroup++)
              for(int recvgroup = 0; recvgroup < numgroup; recvgroup++)
                if(sendgroup != recvgroup)
                  for(int send = 0; send < subgroupsize; send++)
                    for(int recv = 0; recv < subgroupsize; recv++) {
                      int sender = sendgroup * groupsize + send;
                      int recver = recvgroup * groupsize + recv;
                      bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
                    }
            break;
        }
        break;
      case 3: // FAN PATTERN
        switch(direction) {
          case 1: // UNI-DIRECTIONAL
            for(int send = 0; send < subgroupsize; send++)
              for(int recvgroup = 1; recvgroup < numgroup; recvgroup++)
                for(int recv = 0; recv < groupsize; recv++) {
                  int sender = send;
                  int recver = recvgroup * groupsize + recv;
                  bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
                }
            break;
          case 2: // OMNI-DIRECTIONAL
            for(int send = 0; send < subgroupsize; send++)
              for(int recvgroup = 1; recvgroup < numgroup; recvgroup++)
                for(int recv = 0; recv < groupsize; recv++) {
                  int sender = send;
                  int recver = recvgroup * groupsize + recv;
                  bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
                  bench.add(sendbuf_d, 0, recvbuf_d, 0, count, recver, sender);
                }
          break;
        }
        break;
      default:
	; // DO NOTHING
    }

    // bench.measure(warmup, numiter); // SIMPLIFIED VIEW

    bench.report(); // SEE COMMUNICATION PATTERN

    // MEASURE
    double minTime, medTime, maxTime, avgTime;
    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    
    // REPORT
    if(myid == ROOT) {
      double data = 0;
      switch(pattern) {
        case 1:
          switch(direction) {
            case 1: printf("UNIDIRECTIONAL");  data =     count * sizeof(Type) / 1.e9 * subgroupsize * (numgroup - 1); break;
            case 2: printf("BIDIRECTIONAL");   data = 2 * count * sizeof(Type) / 1.e9 * subgroupsize * (numgroup - 1); break;
            case 3: printf("OMNIDIRECTIONAL"); data = 2 * count * sizeof(Type) / 1.e9 * subgroupsize * (numgroup - 1); break;
          }
          printf(" RAIL (%d, %d, %d) PATTERN\n", numgpu, groupsize, subgroupsize); break;
        case 2:
          switch(direction) {
            case 1: printf("UNIDIRECTIONAL");  data =     count * sizeof(Type) / 1.e9 * subgroupsize * (numgroup - 1) * subgroupsize; break;
	    case 2: printf("BIDIRECTIONAL");   data = 2 * count * sizeof(Type) / 1.e9 * subgroupsize * (numgroup - 1) * subgroupsize; break;
	    case 3: printf("OMNIDIRECTIONAL"); data = 2 * count * sizeof(Type) / 1.e9 * subgroupsize * (numgroup - 1) * subgroupsize; break;
          }
          printf(" DENSE (%d, %d, %d) PATTERN\n", numgpu, groupsize, subgroupsize); break;
        case 3:
          switch(direction) {
	    case 1: printf("UNIDIRECTIONAL");  data =     count * sizeof(Type) / 1.e9 * subgroupsize * (numgroup - 1) * groupsize; break;
	    case 2: printf("BIDIRECTIONAL") ;  data = 2 * count * sizeof(Type) / 1.e9 * subgroupsize * (numgroup - 1) * groupsize; break;
          }
          printf(" FAN (%d, %d, %d) PATTERN\n", numgpu, groupsize, subgroupsize); break;
        default:
          ; // DO NOTHING
      }
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e us, %.4e s/GB, %.4e GB/s\n", minTime * 1e6, minTime / data, data / minTime);
      printf("medTime: %.4e us, %.4e s/GB, %.4e GB/s\n", medTime * 1e6, medTime / data, data / medTime);
      printf("maxTime: %.4e us, %.4e s/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data, data / maxTime);
      printf("avgTime: %.4e us, %.4e s/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data, data / avgTime);
    }
  }

// DEALLOCATE
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
    printf("CommBench requires nine arguments:\n");
    printf("1. library: 0 for IPC, 1 for MPI, 2 for NCCL or RCCL\n");
    printf("2. pattern: 1 for Rail, 2 for Dense, 3 for Fan\n");
    printf("3. direction: 1 for unidirectional, 2 for bidirectional, 3 for omnidirectional\n");
    printf("4. count: number of 4-byte elements\n");
    printf("5. warmup: number of warmup rounds\n");
    printf("6. numiter: number of measurement rounds\n");
    printf("7. p: number of processors\n");
    printf("8. g: group size\n");
    printf("9. k: subgroup size\n");
    printf("where on can run CommBench as\n");
    printf("mpirun ./CommBench library pattern direction count warmup numiter p g k\n");
    printf("\n");
  }
}

