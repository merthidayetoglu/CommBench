{
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

  for(int p = 0; p < numproc; p++)
  {
    CommBench::Comm<Type> bench(MPI_COMM_WORLD, (CommBench::capability) cap);

    double data = 0;
    if(direction == 1) {
      bench.add(sendbuf_d, 0, recvbuf_d, 0, count, 0, p);
      data = count * sizeof(Type) / 1.e9;
    }

    if(direction == 2) {
      bench.add(sendbuf_d, 0, recvbuf_d, 0, count, 0, p);
      bench.add(sendbuf_d, 0, recvbuf_d, 0, count, p, 0);
      data = 2 * count * sizeof(Type) / 1.e9;
    }

    bench.report();

    double minTime, medTime, maxTime, avgTime;
    bench.measure(warmup, numiter, minTime, medTime, maxTime, avgTime);
    if(myid == ROOT) {
      printf("TEST_P2P (%d)\n", subgroupsize);
      printf("data: %.4e MB\n", data * 1e3);
      printf("minTime: %.4e s, %.4e s/GB, %.4e GB/s\n", minTime * 1e6, minTime / data, data / minTime);
      printf("medTime: %.4e s, %.4e s/GB, %.4e GB/s\n", medTime * 1e6, medTime / data, data / medTime);
      printf("maxTime: %.4e s, %.4e s/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data, data / maxTime);
      printf("avgTime: %.4e s, %.4e s/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data, data / avgTime);
    }
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

}
