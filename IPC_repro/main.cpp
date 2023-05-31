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

#include <stdio.h>
#include <mpi.h>

#define ROOT 0

// HEADERS
#include <nccl.h>

// UTILITIES
#define PORT_CUDA
#include "../util.h"

#define Type int

int main(int argc, char *argv[])
{
  // INITIALIZE MPI
  int myid;
  int numproc;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  size_t count = atol(argv[1]);

  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Message size %lu\n", count * sizeof(int));
    printf("\n");
  }

  setup_gpu();

  // REMOTE MEMORY POINTER
  Type **recvbuf_ipc;
  if(myid == ROOT)
    recvbuf_ipc = new Type*[numproc];

  // IPC STREAMS
  cudaStream_t stream_ipc[numproc];
  if(myid == ROOT)
    for(int p = 0; p < numproc; p++)
      cudaStreamCreate(stream_ipc + p);

  // ALLOCATE DEVICE
  Type *sendbuf_d;
  Type *recvbuf_d;
  if(myid == ROOT)
    cudaMalloc(&sendbuf_d, count * sizeof(Type) * numproc);
  cudaMalloc(&recvbuf_d, count * sizeof(Type));
  // ALLOCATE HOST
  Type *sendbuf;
  Type *recvbuf;
  if(myid == ROOT)
    cudaMallocHost(&sendbuf, count * sizeof(Type) * numproc);
  cudaMallocHost(&recvbuf, count * sizeof(Type));

  // EXCHANGE MEMORY HANDLES
  if(myid == ROOT) {
    for(int p = 0; p < numproc; p++) {
      if(p == myid) // SELF COMMUNICATION
        recvbuf_ipc[p] = recvbuf_d;
      else {
        cudaIpcMemHandle_t memhandle;
        MPI_Recv(&memhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, p, p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int error = cudaIpcOpenMemHandle((void**) recvbuf_ipc + p, memhandle, cudaIpcMemLazyEnablePeerAccess);
	printf("myid %d cudaIpcOpenMemHandle error %d\n", myid, error);
      }
    }
  }
  else {
    cudaIpcMemHandle_t memhandle;
    int error = cudaIpcGetMemHandle(&memhandle, recvbuf_d);
    printf("myid %d cudaIpcGetMemHandle error %d\n", myid, error);
    MPI_Send(&memhandle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, ROOT, myid, MPI_COMM_WORLD);
  }

  // INITIALIZE VERIFICATION
  for(int p = 0; p < numproc; p++)
    for(size_t i = 0; i < count; i++)
      sendbuf[p * count + i] = p * count * i;
  cudaMemcpy(sendbuf_d, sendbuf, count * sizeof(Type) * numproc, cudaMemcpyHostToDevice);
  cudaMemset(recvbuf_d, -1, count * sizeof(Type));
  cudaStream_t stream_verify;
  cudaStreamCreate(&stream_verify);
  bool test = true;

  MPI_Barrier(MPI_COMM_WORLD);

  if(myid == ROOT)
    for(int p = 0; p < numproc; p++)
      cudaMemcpyAsync(sendbuf_d + count * p, recvbuf_ipc[p], count * sizeof(Type), cudaMemcpyDeviceToDevice, stream_ipc[p]);

  // SYNCHRONIZATION
  MPI_Barrier(MPI_COMM_WORLD);

  // VERIFY
  cudaMemcpyAsync(recvbuf, recvbuf_d, count * sizeof(Type), cudaMemcpyDeviceToHost, stream_verify);
  cudaStreamSynchronize(stream_verify);
  for(size_t i = 0; i < count; i++) {
    printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
    if(recvbuf[i] != myid * count + i)
      test = false;
  }
  MPI_Allreduce(&test, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
  if(myid == ROOT) {
    printf("SCATTER VERIFICATION: ");
    if(test)
      printf("PASS!\n");
    else
      printf("ERROR!\n");
  }

// FINALIZE IPC
  if(myid == ROOT)
    for(int p = 0; p < numproc; p++)
      if(p != myid)
        cudaIpcCloseMemHandle(recvbuf_ipc[p]);

// DEALLOCATE
  cudaFree(sendbuf_d);
  cudaFree(recvbuf_d);

  // FINALIZE
  MPI_Finalize();

  return 0;
} // main()

