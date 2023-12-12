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

// GPU PORTS
// #define PORT_CUDA
// #define PORT_HIP
#define PORT_SYCL

#include "comm.h"

// UTILITIES
#include "util.h"
void print_args(int, char**, int&, int&, int&, size_t&, int&, int&, int&, int&, int&);

// USER DEFINED TYPE
struct Type
{
  // int tag;
  int data[1];
  // complex<double> x, y, z;
};

// GROUP-TO-GROUP PATTERNS
enum Pattern {rail, fan, dense, numpattern};
enum Direction {outbound, inbound, bidirect, omnidirect, numdirect};

int main(int argc, char *argv[])
{

  // INPUT PARAMETERS
  int library;
  int pattern;
  int direction;
  size_t count;
  int warmup;
  int numiter;
  int numgpu;
  int groupsize;
  int subgroupsize;
  print_args(argc, argv, library, pattern, direction, count, warmup, numiter, numgpu, groupsize, subgroupsize);

  int numgroup = numgpu / groupsize;

  // ALLOCATE
  Type *sendbuf_d;
  Type *recvbuf_d;
  CommBench::allocate(sendbuf_d, count);
  CommBench::allocate(recvbuf_d, count);

  CommBench::printid = ROOT;
  CommBench::Comm<Type> bench((CommBench::library) library);

  double data = 0;
  switch(pattern) {
    case Pattern::rail: // RAIL PATTERN
      switch(direction) {
        case Direction::outbound: // UNI-DIRECTIONAL (OUTBOUND)
         for(int sender = 0; sender < subgroupsize; sender++)
            for(int recvgroup = 1; recvgroup < numgroup; recvgroup++) {
              int recver = recvgroup * groupsize + sender;
              bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
            }
          data = count * sizeof(Type) * subgroupsize * (numgroup - 1);
          break;
        case Direction::inbound: // UNI-DIRECTIONAL (INBOUND)
          for(int recver = 0; recver < subgroupsize; recver++)
            for(int sendgroup = 1; sendgroup < numgroup; sendgroup++) {
              int sender = sendgroup * groupsize + recver;
              bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
            }
          data = count * sizeof(Type) * subgroupsize * (numgroup - 1);
          break;
        case Direction::bidirect: // BI-DIRECTIONAL
          for(int sender = 0; sender < subgroupsize; sender++)
            for(int recvgroup = 1; recvgroup < numgroup; recvgroup++) {
              int recver = recvgroup * groupsize + sender;
              bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
              bench.add(sendbuf_d, 0, recvbuf_d, 0, count, recver, sender);
            }
          data = 2 * count * sizeof(Type) * subgroupsize * (numgroup - 1);
          break;
        case Direction::omnidirect: // OMNI-DIRECTIONAL
          for(int sendgroup = 0; sendgroup < numgroup; sendgroup++)
            for(int recvgroup = 0; recvgroup < numgroup; recvgroup++)
              if(sendgroup != recvgroup)
                for(int send = 0; send < subgroupsize; send++) {
                  int sender = sendgroup * groupsize + send;
                  int recver = recvgroup * groupsize + send;
                  bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
                }
          data = 2 * count * sizeof(Type) * subgroupsize * (numgroup - 1);
          break;
      }
      break;
    case Pattern::fan: // FAN PATTERN
      switch(direction) {
        case Direction::outbound: // UNI-DIRECTIONAL (OUTBOUND)
          for(int sender = 0; sender < subgroupsize; sender++)
            for(int recvgroup = 1; recvgroup < numgroup; recvgroup++)
              for(int recv = 0; recv < groupsize; recv++) {
                int recver = recvgroup * groupsize + recv;
                bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
              }
          data = count * sizeof(Type) * subgroupsize * (numgroup - 1) * groupsize;
          break;
        case Direction::inbound: // UNI-DIRECTIONAL (INBOUND)
          for(int recver = 0; recver < subgroupsize; recver++)
            for(int sendgroup = 1; sendgroup < numgroup; sendgroup++)
              for(int send = 0; send < groupsize; send++) {
                int sender = sendgroup * groupsize + send;
                bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
              }
          data = count * sizeof(Type) * subgroupsize * (numgroup - 1) * groupsize;
          break;
        case Direction::bidirect: // BI-DIRECTIONAL
          for(int sender = 0; sender < subgroupsize; sender++)
            for(int recvgroup = 1; recvgroup < numgroup; recvgroup++)
              for(int recv = 0; recv < groupsize; recv++) {
                int recver = recvgroup * groupsize + recv;
                bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
                bench.add(sendbuf_d, 0, recvbuf_d, 0, count, recver, sender);
              }
          data = 2 * count * sizeof(Type) * subgroupsize * (numgroup - 1) * groupsize;
          break;
      }
      break;
    case Pattern::dense: // DENSE PATTERN
      switch(direction) {
        case Direction::outbound: // UNI-DIRECTIONAL (OUTBOUND)
          for(int sender = 0; sender < subgroupsize; sender++)
            for(int recvgroup = 1; recvgroup < numgroup; recvgroup++)
              for(int recv = 0; recv < subgroupsize; recv++) {
                int recver = recvgroup * groupsize + recv;
                bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
              }
          data = count * sizeof(Type) * subgroupsize * (numgroup - 1) * subgroupsize;
          break;
        case Direction::inbound: // UNI-DIRECTIONAL (INBOUND)
          for(int recver = 0; recver < subgroupsize; recver++)
            for(int sendgroup = 1; sendgroup < numgroup; sendgroup++)
              for(int send = 0; send < subgroupsize; send++) {
                int sender = sendgroup * groupsize + send;
                bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
              }
          data = count * sizeof(Type) * subgroupsize * (numgroup - 1) * subgroupsize;
          break;
        case Direction::bidirect: // BI-DIRECTIONAL
          for(int sender = 0; sender < subgroupsize; sender++)
            for(int recvgroup = 1; recvgroup < numgroup; recvgroup++)
              for(int recv = 0; recv < subgroupsize; recv++) {
                int recver = recvgroup * groupsize + recv;
              bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
              bench.add(sendbuf_d, 0, recvbuf_d, 0, count, recver, sender);
            }
          data = 2 * count * sizeof(Type) * subgroupsize * (numgroup - 1) * subgroupsize;
          break;
        case Direction::omnidirect: // OMNI-DIRECTIONAL
          for(int sendgroup = 0; sendgroup < numgroup; sendgroup++)
            for(int recvgroup = 0; recvgroup < numgroup; recvgroup++)
              if(sendgroup != recvgroup)
                for(int send = 0; send < subgroupsize; send++)
                  for(int recv = 0; recv < subgroupsize; recv++) {
                    int sender = sendgroup * groupsize + send;
                    int recver = recvgroup * groupsize + recv;
                    bench.add(sendbuf_d, 0, recvbuf_d, 0, count, sender, recver);
                  }
          data = 2 * count * sizeof(Type) * subgroupsize * (numgroup - 1) * subgroupsize;
          break;
      }
      break;
    default:
      break; // DO NOTHING
  }

  bench.report();
  bench.measure(warmup, numiter, data);
    
  // DEALLOCATE
  CommBench::free(sendbuf_d);
  CommBench::free(recvbuf_d);

} // main()

void print_args(int argc, char *argv[],
		int &library, 
		int &pattern, 
		int &direction, 
		size_t &count, 
		int &warmup, 
		int &numiter, 
		int &numgpu, 
		int &groupsize,
		int &subgroupsize) {
  int myid;
  int numproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  int numthread;
  #pragma omp parallel
  if(omp_get_thread_num() == 0)
  numthread = omp_get_num_threads();

  if(argc == 10) {
    // INPUT PARAMETERS
    library = atoi(argv[1]);
    pattern = atoi(argv[2]);
    direction = atoi(argv[3]);
    count = atol(argv[4]);
    warmup = atoi(argv[5]);
    numiter = atoi(argv[6]);
    numgpu = atoi(argv[7]);
    groupsize = atoi(argv[8]);
    subgroupsize = atoi(argv[9]);
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
  }
  else {
    if(myid == ROOT) {
      printf("\n");
      printf("CommBench requires ten arguments:\n");
      printf("1. library:\n");
      for(int lib = 0; lib < CommBench::numlib; lib++)
        switch(lib) {
          case CommBench::null : printf("      %d for null\n", CommBench::null); break;
          case CommBench::MPI  : printf("      %d for MPI\n", CommBench::MPI); break;
          case CommBench::NCCL : printf("      %d for NCCL or RCCL\n", CommBench::NCCL); break;
          case CommBench::IPC  : printf("      %d for IPC\n", CommBench::IPC);
        }
      printf("2. pattern:\n");
      for(int pat = 0; pat < numpattern; pat++)
        switch(pat) {
          case Pattern::rail  : printf("      %d for rail\n", Pattern::rail); break;
          case Pattern::fan   : printf("      %d for fan\n", Pattern::fan); break;
          case Pattern::dense : printf("      %d for dense\n", Pattern::dense); break;
        }
      printf("3. direction:\n");
      for(int dir = 0; dir < numdirect; dir++)
        switch(dir) {
          case Direction::outbound   : printf("      %d for outbound\n", Direction::outbound); break;
          case Direction::inbound    : printf("      %d for inbound\n", Direction::inbound); break;
          case Direction::bidirect   : printf("      %d for bidirect\n", Direction::bidirect); break;
          case Direction::omnidirect : printf("      %d for omnidirect\n", Direction::omnidirect); break;
        }
      printf("4. count: number of elements per message\n");
      printf("5. warmup: number of warmup rounds\n");
      printf("6. numiter: number of measurement rounds\n");
      printf("7. p: number of processors\n");
      printf("8. g: group size\n");
      printf("9. k: subgroup size\n");
      printf("where on can run CommBench as\n");
      printf("mpirun ./CommBench library pattern direction count warmup numiter p g k\n");
      printf("\n");
    }
    abort();
  }
}

