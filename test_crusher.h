{

  int numgroup = numproc / groupsize;

  Type *sendbuf_d;
  Type *recvbuf_d;
#ifdef PORT_CUDA
  cudaMalloc(&sendbuf_d, count * sizeof(Type));
  cudaMalloc(&recvbuf_d, count * numproc * sizeof(Type));
#elif defined PORT_HIP
  hipMalloc(&sendbuf_d, count * sizeof(Type));
  hipMalloc(&recvbuf_d, count * numproc * sizeof(Type));
#else
  sendbuf_d = new Type[count];
  recvbuf_d = new Type[count * numproc];
#endif

#ifdef TEST_P2P
  {
    CommBench::Comm<Type> bench(MPI_COMM_WORLD, CommBench::TEST_CAPABILITY);
    for(int send = 0; send < 1; send++) {
        int sender = send;
        int recver = groupsize + send;
        bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);
        bench.add(sendbuf_d, 0, recvbuf_d, recver * count, count, recver, sender);
      }
    bench.report();
    double data = 2 * count * sizeof(Type) / 1.e9;
    double minTime, medTime, maxTime, avgTime;
    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    if(myid == ROOT) {
     printf("TEST_P2P\n");
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime, avgTime / data, data / avgTime);
    }
  }
#endif

#ifdef TEST_G2G_rail_scaling
  for(int numsend = 2; numsend < groupsize; numsend += 2)
  {
    CommBench::Comm<Type> bench(MPI_COMM_WORLD, CommBench::TEST_CAPABILITY);
    for(int send = 0; send < numsend; send++) {
      int sender = send;
      int recver = groupsize + send;
      bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);
      bench.add(sendbuf_d, 0, recvbuf_d, recver * count, count, recver, sender);
    }
    bench.report();
    double data = 2 * count * sizeof(Type) / 1.e9 * numsend;
    double minTime, medTime, maxTime, avgTime;
    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    if(myid == ROOT) {
      printf("TEST_G2G_rail_scaling\n");
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime, avgTime / data, data / avgTime);
    }
  }
#endif

#ifdef TEST_G2G_rail
  {
    CommBench::Comm<Type> bench(MPI_COMM_WORLD, CommBench::TEST_CAPABILITY);
    for(int send = 0; send < groupsize; send++) {
        int sender = send;
        int recver = groupsize + send;
        bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);
        bench.add(sendbuf_d, 0, recvbuf_d, recver * count, count, recver, sender);
      }
    bench.report();
    double data = 2 * count * sizeof(Type) / 1.e9 * groupsize;
    double minTime, medTime, maxTime, avgTime;
    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    if(myid == ROOT) {
      printf("TEST_G2G_rail\n");
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime, avgTime / data, data / avgTime);
    }
  }
#endif

#ifdef TEST_P2G
  {
    CommBench::Comm<Type> bench(MPI_COMM_WORLD, CommBench::TEST_CAPABILITY);
    for(int send = 0; send < 1; send++)
      for(int recv = 0; recv < groupsize; recv++) {
        int sender = send;
        int recver = groupsize + recv;
        bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);
        bench.add(sendbuf_d, 0, recvbuf_d, recver * count, count, recver, sender);
      }
    bench.report();
    double data = 2 * count * sizeof(Type) / 1.e9 * groupsize;
    double minTime, medTime, maxTime, avgTime;
    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    if(myid == ROOT) {
      printf("TEST_P2G\n");
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime, avgTime / data, data / avgTime);
    }
  }
#endif

#ifdef TEST_G2G_full_scaling
  for(int numsend = 2; numsend < groupsize; numsend += 2)
  {
    CommBench::Comm<Type> bench(MPI_COMM_WORLD, CommBench::TEST_CAPABILITY);
    for(int send = 0; send < numsend; send++)
      for(int recv = 0; recv < groupsize; recv++) {
        int sender = send;
        int recver = groupsize + recv;
        bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);
        bench.add(sendbuf_d, 0, recvbuf_d, recver * count, count, recver, sender);
      }
    bench.report();
    double data = 2 * count * sizeof(Type) / 1.e9 * numsend * groupsize;
    double minTime, medTime, maxTime, avgTime;
    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    if(myid == ROOT) {
      printf("TEST_G2G_full_scaling\n");
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime, avgTime / data, data / avgTime);
    }
  }
#endif

#ifdef TEST_G2G_full
  {
    CommBench::Comm<Type> bench(MPI_COMM_WORLD, CommBench::TEST_CAPABILITY);
    for(int send = 0; send < groupsize; send++)
      for(int recv = 0; recv < groupsize; recv++) {
        int sender = send;
        int recver = groupsize + recv;
        bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);
        bench.add(sendbuf_d, 0, recvbuf_d, recver * count, count, recver, sender);
      }
    bench.report();
    double data = 2 * count * sizeof(Type) / 1.e9 * groupsize * groupsize;
    double minTime, medTime, maxTime, avgTime;
    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    if(myid == ROOT) {
      printf("TEST_G2G_full\n");
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime, avgTime / data, data / avgTime);
    }
  }
#endif

#ifdef TEST_P2A
  {
    CommBench::Comm<Type> bench(MPI_COMM_WORLD, CommBench::TEST_CAPABILITY);
    for(int send = 0; send < 1; send++)
      for(int recv = 0; recv < numproc; recv++) {
        int sender = send;
        int recver = recv;
        bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);
      }
    bench.report();
    double data = count * sizeof(Type) / 1.e9 * groupsize;
    double minTime, medTime, maxTime, avgTime;
    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    if(myid == ROOT) {
      printf("TEST_P2A\n");
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime, avgTime / data, data / avgTime);
    }
  }
#endif

#ifdef TEST_A2A
  {
    CommBench::Comm<Type> bench(MPI_COMM_WORLD, CommBench::TEST_CAPABILITY);
    for(int send = 0; send < numproc; send++)
      for(int recv = 0; recv < numproc; recv++) {
        int sender = send;
        int recver = recv; 
        bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);
      }
    bench.report();
    double data = 2 * count * sizeof(Type) / 1.e9 * groupsize * groupsize;
    double minTime, medTime, maxTime, avgTime;
    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    if(myid == ROOT) {
      printf("TEST_A2A\n");
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime, avgTime / data, data / avgTime);
    }
  }
#endif


  /*for(int sender = 0; sender < groupsize; sender += 2)
    for(int recver = 0; recver < groupsize; recver += 2) {
      if(myid == ROOT)printf("sender %d recver %d\n", sender, recver);
      if(((sender == 0) && (recver == 0)) ||
         ((sender == 0) && (recver == 4)) ||
         ((sender == 2) && (recver == 4)) ||
         ((sender == 2) && (recver == 6)) ||
         ((sender == 4) && (recver == 0)) ||
         ((sender == 4) && (recver == 2)) ||
         ((sender == 6) && (recver == 2)) ||
         ((sender == 6) && (recver == 6)) ) {
        bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, groupsize + recver);
        bench.add(sendbuf_d, 0, recvbuf_d, (groupsize + recver) * count, count, groupsize + recver, sender);
      }
    }
    bench.add(sendbuf_d, 0, recvbuf_d, 0, count, 0, groupsize + 6);*/


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
