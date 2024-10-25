template <class Coll>
void validate(int *sendbuf_d, int *recvbuf_d, size_t count, int pattern, Coll &coll) {

  int myid = CommBench::myid;
  int numproc = CommBench::numproc;

  int *recvbuf;
  int *sendbuf;
  CommBench::allocateHost(sendbuf, count * numproc);
  CommBench::allocateHost(recvbuf, count * numproc);

  for(int p = 0; p < numproc; p++)
    for(size_t i = p * count; i < (p + 1) * count; i++)
      sendbuf[i] = i;

  CommBench::memcpyH2D(sendbuf_d, sendbuf, count * numproc);

  CommBench::barrier();

  coll.start();
  coll.wait();

  CommBench::memcpyD2H(recvbuf, recvbuf_d, count * numproc);

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
  pass = CommBench::allreduce_land(pass);
  if(myid == ROOT) {
    if(pass) 
      printf("PASSED!\n");
    else 
      printf("FAILED!!!\n");
  }

  CommBench::freeHost(sendbuf);
  CommBench::freeHost(recvbuf);
};
