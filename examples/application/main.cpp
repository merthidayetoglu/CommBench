#define PORT_CUDA
#include "comm.h"

#define ROOT 0
#define Type int
#include "util.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>

using namespace std;

template <typename T, typename I>
__device__ void memory_kernel(T *output, T *input, size_t count, I *index) {
  size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if(tid < count)
    output[tid] = input[index[tid]];
}

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
  int myid;
  int numproc;
  vector<vector<int>> patterns;
  int nodesize = atoi(argv[1]);
  int numnodes = atoi(argv[2]);
  int numstripe = atoi(argv[3]);
  string filename = argv[4];

  int numgpus = numnodes * nodesize;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  setup_gpu();

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

#define CB_STRIPE
#ifdef CB_STRIPE

  CommBench::printid = ROOT;

  /*CommBench::Comm<Type> dummy(CommBench::dummy);
  dummy.add_lazy(10, 0, 4);
  std::vector<size_t> matrix;
  dummy.getMatrix(matrix);
  for (int sender = 0; sender < numgpus; sender++)
    for (int recver = 0; recver < numgpus; recver++)
      patterns[sender][recver] = matrix[sender * numgpus + recver]; */

  // DIRECT PATTERN
  Type *sendbuf;
  Type *recvbuf;
  std::vector<size_t> sendoffset(numgpus + 1);
  std::vector<size_t> recvoffset(numgpus + 1);
  {
    std::vector<size_t> sendcount(numgpus);
    std::vector<size_t> recvcount(numgpus);
    for (int recver = 0; recver < numgpus; recver++)
      sendcount[recver] = patterns[myid][recver];
    MPI_Alltoall(sendcount.data(), sizeof(size_t), MPI_BYTE, recvcount.data(), sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);
    sendoffset[0] = 0;
    recvoffset[0] = 0;
    for (int i = 1; i < numgpus + 1; i++) {
      sendoffset[i] = sendoffset[i - 1] + sendcount[i - 1];
      recvoffset[i] = recvoffset[i - 1] + recvcount[i - 1];
    }
  }
  // PROCESS SPLIT
  Type *sendbuf_split;
  Type *recvbuf_split;
  std::vector<size_t> sendoffset_split(nodesize * numgpus + 1);
  std::vector<size_t> recvoffset_split(nodesize * numgpus + 1);
  {
    std::vector<size_t> sendcount_split(nodesize * numgpus, 0);
    std::vector<size_t> recvcount_split(nodesize * numgpus, 0);
    for (int i = 0; i < numgpus; i++)
      if (myid / nodesize != i / nodesize)
        for (int j = 0; j < numstripe; j++) {
          size_t count = sendoffset[i + 1] - sendoffset[i];
          size_t count_stripe = count / numstripe + (j < count % numstripe ? 1 : 0);
          sendcount_split[j * numgpus + i] = count_stripe;
        }
    MPI_Comm comm_within;
    MPI_Comm_split(MPI_COMM_WORLD, myid / nodesize, myid % nodesize, &comm_within);
    MPI_Alltoall(sendcount_split.data(), numgpus * sizeof(size_t), MPI_BYTE, recvcount_split.data(), numgpus * sizeof(size_t), MPI_BYTE, comm_within);
    sendoffset_split[0] = 0;
    recvoffset_split[0] = 0;
    for (int i = 1; i < nodesize * numgpus + 1; i++) {
      sendoffset_split[i] = sendoffset_split[i - 1] + sendcount_split[i - 1];
      recvoffset_split[i] = recvoffset_split[i - 1] + recvcount_split[i - 1];
    }
  }
  // PROCESS TRANSLATE
  Type *sendbuf_translate;
  Type *recvbuf_translate;
  std::vector<size_t> sendoffset_translate(numnodes * nodesize * nodesize + 1);
  std::vector<size_t> recvoffset_translate(numnodes * nodesize * nodesize + 1);
  {
    std::vector<size_t> sendcount_translate(numnodes * nodesize * nodesize);
    std::vector<size_t> recvcount_translate(numnodes * nodesize * nodesize);
    for (int n = 0; n < numnodes; n++)
      for (int i = 0; i < nodesize; i++)
        for (int j = 0; j < nodesize; j++) {
          int outindex = n * nodesize * nodesize + i * nodesize + j;
          int inindex = i * numgpus + n * nodesize + j;
          sendcount_translate[outindex] = recvoffset_split[inindex + 1] - recvoffset_split[inindex];
        }
    MPI_Comm comm_across;
    MPI_Comm_split(MPI_COMM_WORLD, myid % nodesize, myid / nodesize, &comm_across);
    MPI_Alltoall(sendcount_translate.data(), nodesize * nodesize * sizeof(size_t), MPI_BYTE, recvcount_translate.data(), nodesize * nodesize * sizeof(size_t), MPI_BYTE, comm_across);
    sendoffset_translate[0] = 0;
    recvoffset_translate[0] = 0;
    for (int i = 1; i < numnodes * nodesize * nodesize + 1; i++) {
      sendoffset_translate[i] = sendoffset_translate[i - 1] + sendcount_translate[i - 1];
      recvoffset_translate[i] = recvoffset_translate[i - 1] + recvcount_translate[i - 1];
    }
  }
  // PROCESS ASSEMBLE
  Type *sendbuf_assemble;
  Type *recvbuf_assemble;
  std::vector<size_t> sendoffset_assemble(nodesize * numgpus + 1);
  std::vector<size_t> recvoffset_assemble(nodesize * numgpus + 1);
  {
    std::vector<size_t> sendcount_assemble(nodesize * numgpus, 0);
    std::vector<size_t> recvcount_assemble(nodesize * numgpus, 0);
    for (int n = 0; n < numnodes; n++)
      for (int i = 0; i < nodesize; i++)
        for (int j = 0; j < nodesize; j++) {
          int inindex = n * nodesize * nodesize + i * nodesize + j;
          int outindex = j * numgpus + n * nodesize + i;
          sendcount_assemble[outindex] = recvoffset_translate[inindex + 1] - recvoffset_translate[inindex];
        }
    MPI_Comm comm_within;
    MPI_Comm_split(MPI_COMM_WORLD, myid / nodesize, myid % nodesize, &comm_within);
    MPI_Alltoall(sendcount_assemble.data(), numgpus * sizeof(size_t), MPI_BYTE, recvcount_assemble.data(), numgpus * sizeof(size_t), MPI_BYTE, comm_within);
    sendoffset_assemble[0] = 0;
    recvoffset_assemble[0] = 0;
    for (int i = 1; i < nodesize * numgpus + 1; i++) {
      sendoffset_assemble[i] = sendoffset_assemble[i - 1] + sendcount_assemble[i - 1];
      recvoffset_assemble[i] = recvoffset_assemble[i - 1] + recvcount_assemble[i - 1];
    }
  }

  CommBench::Comm<Type> direct(CommBench::IPC);
  CommBench::Comm<Type> split(CommBench::IPC);
  CommBench::Comm<Type> translate(CommBench::MPI);
  CommBench::Comm<Type> assemble(CommBench::IPC);
  // REGISTER DIRECT (INTRA)
  {
    direct.allocate(sendbuf, sendoffset[numgpus]);
    direct.allocate(recvbuf, recvoffset[numgpus]);
    for (int i = 0; i < numgpus; i++)
      for (int j = 0; j < numgpus; j++)
        if (i / nodesize == j / nodesize)
          direct.add(sendbuf, sendoffset[j], sendoffset[j + 1], recvbuf, recvoffset[i], recvoffset[i + 1], i, j);
    direct.measure(5, 10);
  }
  // REGISTER SPLIT (INTRA)
  {
    split.allocate(sendbuf_split, sendoffset_split[nodesize * numgpus]);
    split.allocate(recvbuf_split, recvoffset_split[nodesize * numgpus]);
    for (int n = 0; n < numnodes; n++)
      for (int i = 0; i < nodesize; i++)
        for (int j = 0; j < nodesize; j++)
          split.add(sendbuf_split, sendoffset_split[j * numgpus], sendoffset_split[(j + 1) * numgpus],
                    recvbuf_split, recvoffset_split[i * numgpus], recvoffset_split[(i + 1) * numgpus],
                    n * nodesize + i, n * nodesize + j);
    split.measure(5, 10);
  }
  // REGISTER TRANSLATE (INTER)
  {
    translate.allocate(sendbuf_translate, sendoffset_translate[numnodes * nodesize * nodesize]);
    translate.allocate(recvbuf_translate, recvoffset_translate[numnodes * nodesize * nodesize]);
    for (int m = 0; m < numnodes; m++)
      for (int n = 0; n < numnodes; n++)
        for (int k = 0; k < nodesize; k++)
          translate.add(sendbuf_translate, sendoffset_translate[n * nodesize * nodesize], sendoffset_translate[(n + 1) * nodesize * nodesize],
                        recvbuf_translate, recvoffset_translate[m * nodesize * nodesize], recvoffset_translate[(m + 1) * nodesize * nodesize],
                        m * nodesize + k, n * nodesize + k);
    translate.measure(5, 10);
  }
  // REGISTER ASSEMBLE (INTER)
  {
    assemble.allocate(sendbuf_assemble, sendoffset_assemble[nodesize * numgpus]);
    assemble.allocate(recvbuf_assemble, recvoffset_assemble[nodesize * numgpus]);
    for (int n = 0; n < numnodes; n++)
      for (int i = 0; i < nodesize; i++)
        for (int j = 0; j < nodesize; j++)
          assemble.add(sendbuf_assemble, sendoffset_assemble[j * numgpus], sendoffset_assemble[(j + 1) * numgpus],
                       recvbuf_assemble, recvoffset_assemble[i * numgpus], recvoffset_assemble[(i + 1) * numgpus],
                       n * nodesize + i, n * nodesize + j);
    assemble.measure(5, 10);
  }

#endif

// #define CB_STRIPE_EASY
#ifdef CB_STRIPE_EASY

  CommBench::printid = 0;
  CommBench::Comm<Type> memory(CommBench::dummy);

  CommBench::library intra_lib = CommBench::IPC;
  CommBench::library inter_lib = CommBench::MPI;

  std::vector<std::vector<size_t>> inter_table(numgpus);
  for (int i = 0; i < numgpus; i++) {
    for (int j = 0; j < numnodes; j++) {
      inter_table[i].push_back(0);
    }
  }
  // INTRA
  std::vector<Type*> sendbuf(numgpus);
  std::vector<Type*> recvbuf(numgpus);
  CommBench::Comm<Type> intra(intra_lib);
  for(int i = 0; i < numgpus; i++) {
    for(int j = 0; j < numgpus; j++) {
      size_t count = patterns[j][i];
      memory.allocate(sendbuf[j], count, i);
      memory.allocate(recvbuf[i], count, j);
      int sendnode = i / nodesize;
      int recvnode = j / nodesize;
      if(sendnode == recvnode) 
        intra.add(sendbuf[j], 0, recvbuf[i], 0, count, i, j);
      if(sendnode != recvnode) {
        for(int k = 0; k < nodesize; k++) { // REPEAT TABLE
	  size_t count_part = count / nodesize + (k < count % nodesize ? 1 : 0);
          inter_table[sendnode * nodesize + k][recvnode] += count_part;
        }
      }
    }
  }
  // TRANSLATE
  CommBench::Comm<Type> translate(inter_lib);
  std::vector<Type*> sendbuf_temp(numnodes);
  std::vector<Type*> recvbuf_temp(numnodes);
  for(int i = 0; i < numgpus; i++) {
    for (int j = 0; j < numnodes; j++) {
      int sendid = i;
      int recvid = j * nodesize + i % nodesize;
      memory.allocate(sendbuf_temp[j], inter_table[i][j], sendid);
      memory.allocate(recvbuf_temp[j], inter_table[i][j], recvid);
      translate.add(sendbuf_temp[j], 0, recvbuf_temp[j], 0, inter_table[i][j], sendid, recvid);
      inter_table[i][j] = 0;
    }
  }
  // SPLIT & ASSEMBLE
  CommBench::Comm<Type> split(intra_lib);
  CommBench::Comm<Type> assemble(intra_lib);
  for(int i = 0; i < numgpus; i++) {
    for(int j = 0; j < numgpus; j++) {
      size_t count = patterns[j][i];
      int sendnode = i / nodesize;
      int recvnode = j / nodesize;
      if(sendnode != recvnode) {
        for(int k = 0; k < nodesize; k++) {
	  size_t count_part = count / nodesize + (k < count % nodesize ? 1 : 0);
          int sender_temp = sendnode * nodesize + k;
          int recver_temp = recvnode * nodesize + k;
          size_t offset = inter_table[sender_temp][recvnode];
          split.add(sendbuf[j], count_part * k, sendbuf_temp[recvnode], offset, count_part, i, sender_temp);
          assemble.add(recvbuf_temp[sendnode], offset, recvbuf[i], count_part * k, count_part, recver_temp, j);
          inter_table[sender_temp][recvnode] += count_part;
        }
      }
    }
  }

  intra.measure(5, 10);
  split.measure(5, 10);
  translate.measure(5, 10);
  assemble.measure(5, 10);
  memory.report();
#endif

// #define CB_STRIPE_DIRECT
#ifdef CB_STRIPE_DIRECT
  int inter_count = 0;
  int intra_count = 0;
  CommBench::printid = 0;
  CommBench::Comm<Type> inter(CommBench::NCCL);
  CommBench::Comm<Type> intra(CommBench::IPC);
  CommBench::Comm<Type> comb(CommBench::NCCL);

  for (int i = 0 ; i < numgpus ; i++) {//sendnode
    for (int j = 0 ; j < numgpus ; j++) {//recvnode
      comb.add_lazy(patterns[i][j], i, j);
      if (i/nodesize == j/nodesize) {//intra
        intra.add_lazy(patterns[i][j], i, j);
        intra_count += patterns[i][j];
      }else{//inter
        inter.add_lazy(patterns[i][j], i, j);
        inter_count += patterns[i][j];
      }
    }
  }

  comb.measure(5, 10, inter_count+intra_count);
  intra.measure(5, 10, intra_count);
  inter.measure(5, 10, inter_count);

  vector<CommBench::Comm<Type>> vec = {inter, intra};
  CommBench::measure_concur(vec, 5, 10, inter_count+intra_count);
#endif
  // measure_MPI_Alltoallv<int>(patterns, 5, 10);

  MPI_Finalize();
}
