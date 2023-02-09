{
  int numgroup = numproc / groupsize;

  CommBench::Bench<Type> bench(MPI_COMM_WORLD, groupsize, CommBench::across, CommBench::MPI, count);

  bench.test();

  return 0;

  double totalData = 0;
  double totalTime = 0;
  for (int iter = -warmup; iter < numiter; iter++) {
    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();
    bench.start();
    double start = MPI_Wtime() - time;
    bench.wait();
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime() - time;
    if(iter < 0) {
      if(myid == ROOT)
        printf("start %.2e warmup: %.2e\n", start, time);
    }
    else {
      if(myid == ROOT)
        printf("start %.2e time: %.2e\n", start, time);
     totalTime += time;
     totalData += 2 * count * (numgroup - 1) * sizeof(Type) / 1.e9;
    }
  }
  if(myid == ROOT)
    printf("totalTime %.2e s totalData %.2e GB B/W %.2e GB/s --- MPI\n", totalTime, totalData, totalData / totalTime * groupsize);

}
