{

  Type *sendbuf_d;
  Type *recvbuf_d;

#ifdef PORT_CUDA
  cudaMalloc(&sendbuf_d, count * sizeof(Type) * numproc);
  cudaMalloc(&recvbuf_d, count * sizeof(Type) * numproc);
#elif defined PORT_HIP
  hipMalloc(&sendbuf_d, count * sizeof(Type) * numproc);
  hipMalloc(&recvbuf_d, count * sizeof(Type) * numproc);
#else
    sendbuf_d = new Type[count * numproc];
    recvbuf_d = new Type[count * numproc];
#endif

  {
    double times[numiter];
    for (int iter = -warmup; iter < numiter; iter++) {
#if defined PORT_CUDA
      cudaMemset(sendbuf_d, -1, count * sizeof(Type) * numproc);
#elif defined PORT_HIP
      hipMemset(sendbuf_d, -1, count * sizeof(Type) * numproc);
#else
      memset(sendbuf_d, -1, count * sizeof(Type) * numproc);
#endif

      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();

      //MPI_Allgather(sendbuf_d, count, MPI_FLOAT, recvbuf_d, count, MPI_FLOAT, MPI_COMM_WORLD);
      MPI_Alltoall(sendbuf_d, count, MPI_FLOAT, recvbuf_d, count, MPI_FLOAT, MPI_COMM_WORLD);
      //MPI_Scatter(sendbuf_d, count, MPI_FLOAT, recvbuf_d, count, MPI_FLOAT, 0, MPI_COMM_WORLD);
      //MPI_Bcast(sendbuf_d, count, MPI_FLOAT, 0, MPI_COMM_WORLD);

      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if(iter < 0) {
        if(myid == ROOT)
          printf("warmup: %.2e\n", time);
      }
      else {
        times[iter] = time;
      }
    }

    std::sort(times, times + numiter,  [](const double & a, const double & b) -> bool {return a < b;});

    if(myid == ROOT)
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

    double minTime, medTime, maxTime, avgTime;
    minTime = times[0];
    medTime = times[numiter / 2];
    maxTime = times[numiter - 1];
    avgTime = 0;
    for(int iter = 0; iter < numiter; iter++)
      avgTime += times[iter];
    avgTime /= numiter;

    double data = count * sizeof(Type) / 1.e9;
    if(myid == ROOT) {
      printf("TEST_G2G_rail (%d)\n", subgroupsize);
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime, avgTime / data, data / avgTime);
    }
  }

#ifdef PORT_CUDA
  cudaFree(sendbuf_d);
  cudaFree(recvbuf_d);
#elif defined PORT_HIP
  hipFree(sendbuf_d);
  hipFree(recvbuf_d);
#else
  delete[] sendbuf_d;
  delete[] recvbuf_d;
#endif

}
