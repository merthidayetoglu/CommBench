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

// PORTS
// #define PORT_CUDA
#define PORT_HIP
// #define PORT_SYCL

#include "../commbench.h"

#define ROOT 0
#include "validate.h"

void print_args();

// USER DEFINED TYPE
#define Type int
/*struct Type
{
  // int tag;
  int data[1];
  // complex<double> x, y, z;
};*/

int main(int argc, char *argv[])
{
  // INITIALIZE
  int myid;
  int numproc;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  // INPUT PARAMETERS
  if(argc != 6) {
    printf("arcgc: %d\n", argc);
    print_args();
    MPI_Finalize();
    return 0;
  }
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
    printf("Number of warmup %d\n", warmup);
    printf("Number of iterations %d\n", numiter);

    printf("Pattern: %d\n", pattern);

    printf("Bytes per Type %lu\n", sizeof(Type));
    printf("Point-to-point (P2P) count %ld ( %ld Bytes)\n", count, count * sizeof(Type));
    printf("\n");
  }

  // COMMBENCH SCOPE
  {
    using namespace CommBench;

    // INITIALIZE
    init();

    // ALLOCATE
    Type *sendbuf_d;
    Type *recvbuf_d;
    allocate(sendbuf_d, count * numproc);
    allocate(recvbuf_d, count * numproc);

    // CREATE COMMUNICATOR
    Comm<Type> coll((CommBench::library) library);

    // REGISTER PATTERN
    switch(pattern) {
      case 0:
        if(myid == 0)
          printf("TEST P2P\n");
        coll.add(sendbuf_d, 0, recvbuf_d, 0, count, 0, 1);
        break;
      case 1:
        if(myid == ROOT)
          printf("TEST GATHER\n");
        for(int p = 0; p < numproc; p++)
          coll.add(sendbuf_d, 0, recvbuf_d, p * count, count, p, ROOT);
        break;
      case 2:
        if(myid == ROOT)
          printf("TEST SCATTER\n");
        for(int p = 0; p < numproc; p++)
          coll.add(sendbuf_d, p * count, recvbuf_d, 0, count, ROOT, p);
        break;
      case 3:
        if(myid == ROOT)
          printf("TEST BROADCAST\n");
        for(int p = 0; p < numproc; p++)
          coll.add(sendbuf_d, 0, recvbuf_d, 0, count, ROOT, p);
        break;
      case 4:
        if(myid == ROOT)
          printf("TEST REDUCE\n");
        // CommBench does not offer computational kernels.
        break;
      case 5:
        if(myid == ROOT)
          printf("TEST ALL-TO-ALL\n");
        for(int sender = 0; sender < numproc; sender++)
          for(int recver = 0; recver < numproc; recver++)
            coll.add(sendbuf_d, recver * count, recvbuf_d, sender * count, count, sender, recver);
        break;
      case 6:
        if(myid == ROOT)
          printf("TEST ALL-GATHER\n");
        for(int sender = 0; sender < numproc; sender++)
          for(int recver = 0; recver < numproc; recver++)
            coll.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);
        break;
      case 7:
        if(myid == ROOT)
          printf("TEST REDUCE-SCATTER\n");
        // CommBench does not offer computational kernels.
        break;
      case 8:
        if(myid == ROOT)
          printf("TEST ALL-REDUCE\n");
        // CommBench does not offer computational kernels.
        break;
    }

    // MEASURE 
    coll.measure(warmup, numiter, count * numproc);

    // VALIDATE
    for(int iter = 0; iter < numiter; iter++)
      validate(sendbuf_d, recvbuf_d, count, pattern, coll);

    // DEALLOCATE
    free(sendbuf_d);
    free(recvbuf_d);
  }

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
    printf("CollBench requires five arguments:\n");
    printf("1. library:\n");
    printf("      1 for MPI\n");
    printf("      2 for NCCL/RCCL/OneCCL\n");
    printf("      3 for IPC (PUT)\n");
    printf("      4 for IPC (GET)\n");
    printf("2. pattern:\n");
    printf("      1 for Gather\n");
    printf("      2 for Scatter\n");
    printf("      3 for Broadcast\n");
    printf("      4 for Reduce\n");
    printf("      5 for Alltoall\n");
    printf("      6 for Allgather\n");
    printf("      7 for ReduceScatter\n");
    printf("      8 for Allreduce\n");
    printf("3. count: number of 4-byte elements\n");
    printf("4. warmup: number of warmup rounds\n");
    printf("5. numiter: number of measurement rounds\n");
    printf("where on can run CollBench as\n");
    printf("mpirun ./CollBench $library $pattern $count $warmup $numiter\n");
    printf("\n");
  }
}

