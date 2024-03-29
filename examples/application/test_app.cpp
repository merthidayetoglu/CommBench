#define PORT_CUDA

#include "../../comm.h"

#define ROOT 0
#define Type int
#include "../../util.h"
#include <vector>
#include <fstream>
#include <sstream>

using namespace CommBench;
using namespace std;

int parsefile(int gpus, string fn, vector<vector<int>> &pat) {
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
	return 0;
}

int main(int argc, char* argv[]) {
	//arg list:# of GPUs per node, # of nodes, filename(data)
	int myid;
        int numproc;	
	int inter_count = 0;
        int intra_count = 0;
	vector<vector<int>> patterns;
        int nodesize = atoi(argv[1]);
        int numnodes = atoi(argv[2]);
        string filename = argv[3];
	int i, j;

        int numgpus = numnodes * nodesize;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	setup_gpu();

	parsefile(numgpus, filename, patterns); //in kb
	//check parsefile
	//if (myid == ROOT) {
	//	for(int i = 0 ; i < numgpus ; i++){
	//		for(int j = 0 ; j < numgpus ; j++) {
	//			printf("%d ", patterns[i][j]);
	//		}
	//		printf("\n");
	//	}
	//}
	CommBench::printid = 0;
	Comm<Type> inter(library::MPI);
	Comm<Type> intra(library::MPI);
	Comm<Type> comb(library::MPI);

	for(i = 0 ; i < numgpus ; i++) {//sendnode
          for(j = 0 ; j < numgpus ; j++) {//recvnode
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
        
	vector<Comm<Type>*> vec = {&inter, &intra};
	measure_concur(vec, 5, 10, inter_count+intra_count);
	measure_MPI_Alltoallv<int>(patterns, 5, 10);

	MPI_Finalize();
}
