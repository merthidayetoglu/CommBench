template <class Coll>
void validate(int *sendbuf_d, int *recvbuf_d, size_t count, int pattern, Coll &coll) {

  int myid;
  int numproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  int *recvbuf;
  int *sendbuf;
#ifdef PORT_CUDA
  cudaMallocHost(&sendbuf, count * numproc * sizeof(int));
  cudaMallocHost(&recvbuf, count * numproc * sizeof(int));
#elif defined PORT_HIP
  hipHostMalloc(&sendbuf, count * numproc * sizeof(int));
  hipHostMalloc(&recvbuf, count * numproc * sizeof(int));
#endif
  

  for(int p = 0; p < numproc; p++)
    for(size_t i = p * count; i < (p + 1) * count; i++)
      sendbuf[i] = i;
#ifdef PORT_CUDA
  cudaMemcpy(sendbuf_d, sendbuf, count * sizeof(int) * numproc, cudaMemcpyHostToDevice);
  cudaMemset(recvbuf_d, -1, count * numproc * sizeof(int));
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaDeviceSynchronize();
#elif defined PORT_HIP
  hipMemcpy(sendbuf_d, sendbuf, count * sizeof(int) * numproc, hipMemcpyHostToDevice);
  hipMemset(recvbuf_d, -1, count * numproc * sizeof(int));
  hipStream_t stream;
  hipStreamCreate(&stream);
  hipDeviceSynchronize();
#endif
  memset(recvbuf, -1, count * numproc * sizeof(int));

  barrier();

  coll.start();
  coll.wait();

#ifdef PORT_CUDA
  cudaMemcpyAsync(recvbuf, recvbuf_d, count * sizeof(int) * numproc, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
#elif defined PORT_HIP
  hipMemcpyAsync(recvbuf, recvbuf_d, count * sizeof(int) * numproc, hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);
#endif

  bool pass = true;
  switch(pattern) {
    case 0:
      {
        if(myid == 0) printf("VERIFY P2P\n");
        if(myid == 1) {
          for(size_t i = 0; i < count; i++) {
            // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
            if(recvbuf[i] != i)
              pass = false;
          }
        }
      }
      break;
    case 1:
      {
        if(myid == ROOT) printf("VERIFY GATHER\n");
        if(myid == ROOT) {
          for(int p = 0; p < numproc; p++)
            for(size_t i = 0; i < count; i++) {
              // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
              if(recvbuf[p * count + i] != i)
                pass = false;
            }
        }
      }
      break;
    case 2:
      {
        if(myid == ROOT) printf("VERIFY SCATTER\n");
        for(size_t i = 0; i < count; i++) {
          // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
          if(recvbuf[i] != myid * count + i)
            pass = false;
        }
      }
      break;
    case 3:
      {
        if(myid == ROOT) printf("VERIFY BCAST\n");
        for(size_t i = 0; i < count; i++) {
          // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
          if(recvbuf[i] != i)
            pass = false;
        }
      }
      break;
    case 4:
      {
        if(myid == ROOT) printf("REDUCE IS NOT TESTED\n");
          pass = false;
      }
      break;
    case 5:
      {
        if(myid == ROOT) printf("VERIFY ALL-TO-ALL\n");
        for(int p = 0; p < numproc; p++)
          for(size_t i = 0; i < count; i++) {
            // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
            if(recvbuf[p * count + i] != myid * count + i)
              pass = false;
          }
      }
      break;
    case 6:
      {
        if(myid == ROOT) printf("VERIFY ALL-GATHER\n");
        for(int p = 0; p < numproc; p++)
          for(size_t i = 0; i < count; i++) {
            // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
            if(recvbuf[p * count + i] != i)
              pass = false;
          }
      }
      break;
    case 7:
      {
        if(myid == ROOT) printf("REDUCE-SCATTER IS NOT TESTED\n");
          pass = false;
      }
      break;
    case 8: 
      { 
        if(myid == ROOT) printf("ALL-REDUCE IS NOT TESTED\n");
          pass = false;
      }
      break;
  }

  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
  if(myid == ROOT) {
    if(pass) 
      printf("PASSED!\n");
    else 
      printf("FAILED!!!\n");
  }

#ifdef PORT_CUDA
  cudaFreeHost(sendbuf);
  cudaFreeHost(recvbuf);
#elif defined PORT_HIP
  hipHostFree(sendbuf);
  hipHostFree(recvbuf);
#endif
};
