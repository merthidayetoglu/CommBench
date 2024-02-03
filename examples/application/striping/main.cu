#define PORT_CUDA
#include "spcomm.h"

#define ROOT 0
#define Type long
#define Index int
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>

using namespace std;

void parsefile(int gpus, string fn, vector<vector<int>> &pat) {
  ifstream fp;
  string s;
  int temp_val;
  vector<int> temp_v;
  fp.open(fn, ifstream::in);
  for(int i = 0; i < gpus; i++) {
    getline(fp, s);
    stringstream ss(s);
    for(int j = 0 ; j < gpus; j++) {
      ss >> temp_val;
      temp_v.push_back(temp_val);
    }
    pat.push_back(temp_v);
    temp_v.clear();
  }
  fp.close();
}

int main(int argc, char* argv[]) {

  //arg list:# of GPUs per node, # of nodes, filename(data)
  // int myid;
  // int numproc;
  vector<vector<int>> patterns;
  int nodesize = atoi(argv[1]);
  int numnodes = atoi(argv[2]);
  int numstripe = atoi(argv[3]);
  string filename = argv[4];
  int numrhs = atoi(argv[5]);

  int numgpus = numnodes * nodesize;

  // MPI_Init(&argc, &argv);
  // MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  // MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  parsefile(numgpus, filename, patterns); //in kb
  //check parsefile
  //if (myid == ROOT) {
  //      for(int i = 0 ; i < numgpus ; i++){
  //              for(int j = 0 ; j < numgpus ; j++) {
  //                      printf("%d ", patterns[i][j]);
  //              }
  //              printf("\n");
  //      }
  //}
  /*for(int i = 0; i < numgpus; i++)
    for(int j = 0; j < numgpus; j++)
      if(i / nodesize == j / nodesize)
        patterns[i][j] = 0;*/

  // DIRECT PATTERN
  CommBench::SpComm<Type, Index> comm(CommBench::MPI, ROOT);
  std::vector<size_t> sendoffset(numgpus + 1);
  std::vector<size_t> recvoffset(numgpus + 1);
  {
    std::vector<size_t> sendcount(numgpus);
    std::vector<size_t> recvcount(numgpus);
    for (int recver = 0; recver < numgpus; recver++)
      sendcount[recver] = patterns[CommBench::myid][recver];
    MPI_Alltoall(sendcount.data(), sizeof(size_t), MPI_BYTE, recvcount.data(), sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);
    sendoffset[0] = 0;
    recvoffset[0] = 0;
    for (int i = 1; i < numgpus + 1; i++) {
      sendoffset[i] = sendoffset[i - 1] + sendcount[i - 1];
      recvoffset[i] = recvoffset[i - 1] + recvcount[i - 1];
    }
  }
  Type *sendbuf;
  Type *recvbuf;
  CommBench::allocate(sendbuf, sendoffset[numgpus] * numrhs);
  CommBench::allocate(recvbuf, recvoffset[numgpus] * numrhs);

  // REGISTER PACKED COMMUNICATION
  std::vector<Index> sendindex(sendoffset[numgpus] * numrhs);
  std::vector<Index> recvindex(recvoffset[numgpus] * numrhs);
  {
    for(int k = 0; k < numrhs; k++)
      for(int p = 0; p < numgpus; p++) {
        size_t sendcount = sendoffset[p + 1] - sendoffset[p];
        for(size_t i = 0; i < sendcount; i++)
          sendindex[sendoffset[p] * numrhs + k * sendcount + i] = k * sendoffset[numgpus] + sendoffset[p] + i;
        size_t recvcount = recvoffset[p + 1] - recvoffset[p];
        for(size_t i = 0; i < recvcount; i++)
          recvindex[recvoffset[p] * numrhs + k * recvcount + i] = k * recvoffset[numgpus] + recvoffset[p] + i;
      }
    Type *sendbuf_temp;
    Type *recvbuf_temp;
    comm.allocate(sendbuf_temp, sendoffset[numgpus] * numrhs);
    comm.add_precomp_gather(sendbuf, sendbuf_temp, sendoffset[numgpus] * numrhs, sendindex.data());
    comm.allocate(recvbuf_temp, recvoffset[numgpus] * numrhs);
    for(int i = 0; i < numgpus; i++)
      for(int j = 0; j < numgpus; j++)
        comm.add(sendbuf_temp, sendoffset[j] * numrhs, 
                                 sendoffset[j + 1] * numrhs,
                   recvbuf_temp, recvoffset[i] * numrhs,
                                 recvoffset[i + 1] * numrhs,
                   i, j);
    comm.add_postcomp_scatter(recvbuf_temp, recvbuf, recvoffset[numgpus] * numrhs, recvindex.data());
  }
  comm.measure(5, 10);

  // REGISTER SPLIT
  CommBench::SpComm<Type, Index> split(CommBench::IPC);
  std::vector<size_t> sendoffset_split(nodesize * numgpus + 1);
  std::vector<size_t> recvoffset_split(nodesize * numgpus + 1);
  Type *sendbuf_split;
  Type *recvbuf_split;
  {
    std::vector<size_t> sendcount(nodesize * numgpus, 0);
    std::vector<size_t> recvcount(nodesize * numgpus, 0);
    for(int p = 0; p < numgpus; p++)
      for(int stripe = 0; stripe < numstripe; stripe++) {
        size_t splitcount = (sendoffset[p + 1] - sendoffset[p]) * numrhs;
        sendcount[stripe * numgpus + p] = splitcount / numstripe + (stripe < splitcount % numstripe ? 1 : 0);
      }
    MPI_Comm comm_intra;
    MPI_Comm_split(MPI_COMM_WORLD, CommBench::myid / nodesize, CommBench::myid % nodesize, &comm_intra);
    MPI_Alltoall(sendcount.data(), numgpus * sizeof(size_t), MPI_BYTE, recvcount.data(), numgpus * sizeof(size_t), MPI_BYTE, comm_intra);
    sendoffset_split[0] = 0;
    recvoffset_split[0] = 0;
    for(int i = 1; i < nodesize * numgpus + 1; i++) {
      sendoffset_split[i] = sendoffset_split[i - 1] + sendcount[i - 1];
      recvoffset_split[i] = recvoffset_split[i - 1] + recvcount[i - 1];
    }
    std::vector<Index> sendindex_split(sendoffset_split[nodesize * numgpus]);
    for(int j = 0; j < numgpus; j++) {
      size_t count = 0;
      for(int i = 0; i < nodesize; i++)
        for(size_t k = 0; k < sendcount[i * numgpus + j]; k++) {
          sendindex_split[sendoffset_split[i * numgpus + j] + k] = sendindex[sendoffset[j] * numrhs + count];
          count++;
        }
    }
    split.allocate(sendbuf_split, sendoffset_split[nodesize * numgpus]);
    split.add_precomp_gather(sendbuf, sendbuf_split, sendoffset_split[nodesize * numgpus], sendindex_split.data());
    split.allocate(recvbuf_split, recvoffset_split[nodesize * numgpus]);
    for(int node = 0; node < numnodes; node++)
      for(int i = 0; i < nodesize; i++)
        for(int j = 0; j < nodesize; j++)
          split.add(sendbuf_split, sendoffset_split[j * numgpus],
                                   sendoffset_split[(j + 1) * numgpus],
                    recvbuf_split, recvoffset_split[i * numgpus],
                                   recvoffset_split[(i + 1) * numgpus],
                    node * nodesize + i, node * nodesize + j);
  }
  split.measure(5, 10);

  // REGISTER TRANSLATE
  CommBench::SpComm<Type, Index> translate(CommBench::MPI);
  std::vector<size_t> sendoffset_translate(numnodes * nodesize * nodesize + 1);
  std::vector<size_t> recvoffset_translate(numnodes * nodesize * nodesize + 1);
  Type *sendbuf_translate;
  Type *recvbuf_translate;
  {
    std::vector<size_t> sendcount(numnodes * nodesize * nodesize);
    std::vector<size_t> recvcount(numnodes * nodesize * nodesize);
    for(int n = 0; n < numnodes; n++)
      for(int i = 0; i < nodesize; i++)
        for(int j = 0; j < nodesize; j++)
          sendcount[n * nodesize * nodesize + i * nodesize + j] = recvoffset_split[i * numgpus + n * nodesize + j + 1] - recvoffset_split[i * numgpus + n * nodesize + j];
    MPI_Comm comm_inter;
    MPI_Comm_split(MPI_COMM_WORLD, CommBench::myid % nodesize, CommBench::myid / nodesize, &comm_inter);
    MPI_Alltoall(sendcount.data(), nodesize * nodesize * sizeof(size_t), MPI_BYTE, recvcount.data(), nodesize * nodesize * sizeof(size_t), MPI_BYTE, comm_inter);
    sendoffset_translate[0] = 0;
    recvoffset_translate[0] = 0;
    for(int i = 1; i < numnodes * nodesize * nodesize + 1; i++) {
      sendoffset_translate[i] = sendoffset_translate[i - 1] + sendcount[i - 1];
      recvoffset_translate[i] = recvoffset_translate[i - 1] + recvcount[i - 1];
    }
    translate.allocate(sendbuf_translate, sendoffset_translate[numnodes * nodesize * nodesize]);
    std::vector<Index> sendindex_translate(sendoffset_translate[numnodes * nodesize * nodesize]);
    for(int n = 0; n < numnodes; n++)
      for(int i = 0; i < nodesize; i++)
         for(int j = 0; j < nodesize; j++)
           for(size_t k = 0; k < sendcount[n * nodesize * nodesize + i * nodesize + j]; k++)
             sendindex_translate[sendoffset_translate[n * nodesize * nodesize + i * nodesize + j] + k] = recvoffset_split[i * numgpus + n * nodesize + j] + k;
    translate.add_precomp_gather(recvbuf_split, sendbuf_translate, sendoffset_translate[numnodes * nodesize * nodesize], sendindex_translate.data());
    translate.allocate(recvbuf_translate, recvoffset_translate[numnodes * nodesize * nodesize]);
    for(int n = 0; n < numnodes; n++)
      for(int i = 0; i < nodesize; i++)
        for(int m = 0; m < numnodes; m++)
          translate.add(sendbuf_translate, sendoffset_translate[m * nodesize * nodesize],
                                           sendoffset_translate[(m + 1) * nodesize * nodesize],
                        recvbuf_translate, recvoffset_translate[n * nodesize * nodesize],
                                           recvoffset_translate[(n + 1) * nodesize * nodesize],
                        n * nodesize + i, m * nodesize + i);
  }
  translate.measure(5, 10);

  // REGISTER ASSEMBLE
  CommBench::SpComm<Type, Index> assemble(CommBench::IPC);
  std::vector<size_t> sendoffset_assemble(nodesize * numgpus + 1);
  std::vector<size_t> recvoffset_assemble(nodesize * numgpus + 1);
  Type *sendbuf_assemble;
  Type *recvbuf_assemble;
  {
    std::vector<size_t> sendcount(nodesize * numgpus);
    std::vector<size_t> recvcount(nodesize * numgpus);
    for(int n = 0; n < numnodes; n++)
      for(int i = 0; i < nodesize; i++)
        for(int j = 0; j < nodesize; j++)
          sendcount[j * numgpus + n * nodesize + i] = recvoffset_translate[n * nodesize * nodesize + i * nodesize + j + 1] - recvoffset_translate[n * nodesize * nodesize + i * nodesize + j];
    MPI_Comm comm_intra;
    MPI_Comm_split(MPI_COMM_WORLD, CommBench::myid / nodesize, CommBench::myid % nodesize, &comm_intra);
    MPI_Alltoall(sendcount.data(), numgpus * sizeof(size_t), MPI_BYTE, recvcount.data(), numgpus * sizeof(size_t), MPI_BYTE, comm_intra);
    sendoffset_assemble[0] = 0;
    recvoffset_assemble[0] = 0;
    for(int i = 1; i < nodesize * numgpus + 1; i++) {
      sendoffset_assemble[i] = sendoffset_assemble[i - 1] + sendcount[i - 1];
      recvoffset_assemble[i] = recvoffset_assemble[i - 1] + recvcount[i - 1];
    }
    assemble.allocate(sendbuf_assemble, sendoffset_assemble[numnodes * numgpus]);
    std::vector<Index> sendindex_assemble(sendoffset_assemble[numnodes * numgpus]);
    for(int n = 0; n < numnodes; n++)
      for(int i = 0; i < nodesize; i++)
        for(int j = 0; j < nodesize; j++)
          for(size_t k = 0; k < sendcount[j * numgpus + n * nodesize + i]; k++)
            sendindex_assemble[sendoffset_assemble[j * numgpus + n * nodesize + i] + k] = recvoffset_translate[n * nodesize * nodesize + i * nodesize + j] + k;
    assemble.add_precomp_gather(recvbuf_translate, sendbuf_assemble, sendoffset_assemble[nodesize * numgpus], sendindex_assemble.data());
    assemble.allocate(recvbuf_assemble, recvoffset_assemble[numnodes * numgpus]);
    for(int n = 0; n < numnodes; n++)
      for(int i = 0; i < nodesize; i++)
        for(int j = 0; j < nodesize; j++)
          assemble.add(sendbuf_assemble, sendoffset_assemble[j * numgpus],
                                         sendoffset_assemble[(j + 1) * numgpus],
                       recvbuf_assemble, recvoffset_assemble[i * numgpus],
                                         recvoffset_assemble[(i + 1) * numgpus],
                       n * nodesize + i, n * nodesize + j);

    std::vector<Index> recvindex_assemble(recvoffset_assemble[nodesize * numgpus]);
    for(int j = 0; j < numgpus; j++) {
      size_t count = 0;
      for(int i = 0; i < nodesize; i++)
        for(size_t k = 0; k < recvcount[i * numgpus + j]; k++) {
          recvindex_assemble[recvoffset_assemble[i * numgpus + j] + k] =  recvindex[recvoffset[j] * numrhs + count];
          count++;
        }
    }
    assemble.add_postcomp_scatter(recvbuf_assemble, recvbuf, recvoffset_assemble[nodesize * numgpus], recvindex_assemble.data());
  }
  assemble.measure(5, 10);
  
  if(CommBench::myid == 0)
    printf("sendoffset %ld \nrecvoffset %ld \nsendoffset_split %ld \nrecvoffset_split %ld \nsendoffset_translate %ld \nrecvoffset_translate %ld\n sendoffset_assemble %ld \nrecvoffset_assemble %ld \n", sendoffset[numgpus], recvoffset[numgpus], sendoffset_split[nodesize * numgpus], recvoffset_split[nodesize * numgpus], sendoffset_translate[numnodes * nodesize * nodesize], recvoffset_translate[numnodes * nodesize * nodesize], sendoffset_assemble[numnodes * numgpus], recvoffset_assemble[numnodes * numgpus]);

  // VERIFY
  Type *sendbuf_h;
  Type *recvbuf_h;
  CommBench::allocateHost(sendbuf_h, sendoffset[numgpus] * numrhs);
  CommBench::allocateHost(recvbuf_h, recvoffset[numgpus] * numrhs);
  for (size_t i = 0; i < sendoffset[numgpus] * numrhs; i++)
    sendbuf_h[i] = i;
  CommBench::memcpyH2D(sendbuf, sendbuf_h, sendoffset[numgpus] * numrhs);
  // run CommBench
  {
    split.start();
    split.wait();
    translate.start();
    translate.wait();
    assemble.start();
    assemble.wait();
    //comm.start();
    //comm.wait();
  }
  CommBench::memcpyD2H(recvbuf_h, recvbuf, recvoffset[numgpus] * numrhs);
  // MPI_Alltoallv
  {
    Type *recvbuf_temp = new Type[recvoffset[numgpus] * numrhs];
    int *sendcount = new int[numgpus];
    int *recvcount = new int[numgpus];
    int *senddispl = new int[numgpus];
    int *recvdispl = new int[numgpus];
    for(int p = 0; p < numgpus; p++) {
      sendcount[p] = (sendoffset[p + 1] - sendoffset[p]) * sizeof(Type);
      senddispl[p] = sendoffset[p] * sizeof(Type);
      recvcount[p] = (recvoffset[p + 1] - recvoffset[p]) * sizeof(Type);
      recvdispl[p] = recvoffset[p] * sizeof(Type);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();
    for (int k = 0; k < numrhs; k++)
      MPI_Alltoallv(sendbuf + k * sendoffset[numgpus], sendcount, senddispl, MPI_BYTE, recvbuf + k * recvoffset[numgpus], recvcount, recvdispl, MPI_BYTE, MPI_COMM_WORLD);
    time = MPI_Wtime() - time;
    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(CommBench::myid == ROOT)
      printf("MPI_Alltoallv time %e s\n", time);
    CommBench::memcpyD2H(recvbuf_temp, recvbuf, recvoffset[numgpus] * numrhs);
    bool found = false;
    for (size_t i = 0; i < recvoffset[numgpus] * numrhs; i++)
      if (recvbuf_h[i] != recvbuf_temp[i]) {
        printf("myid %d recvbuf_h[%d] = %d, recvbuf_temp[%d] = %d\n", CommBench::myid, i, recvbuf_h[i], i, recvbuf_temp[i]);
        found = true;
        break;
      }
    MPI_Allreduce(MPI_IN_PLACE, &found, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    if(CommBench::myid == ROOT)
      if(!found)
        printf("PASSED!\n");
      else
        printf("FAILED!\n");
    delete[] sendcount;
    delete[] recvcount;
    delete[] senddispl;
    delete[] recvdispl;
    delete[] recvbuf_temp;
  }

  // MPI_Finalize();
}
