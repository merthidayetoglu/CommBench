#include "commbench.h"
#define ROOT 0
#include <string>
#include <vector>
#include <json/json.h>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <unordered_map>
#include <algorithm>
#include "validate.h"

using namespace CommBench;

enum pattern { p2p, gather, scatter, broadcast, reduce, alltoall, allgather };
// Function to check if a given flag is present in the command-line arguments
bool flagPresent(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}



int main(int argc, char* argv[]) {
    init();
    std::ifstream file("config_doc.json", std::ifstream::binary);
    if(!file.is_open()) {
	    std::cerr << "Could not open file!" << std::endl;
    }

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errsg
    int step = 1;

    if(!Json::parseFromStream(builder, file, &root, &errs)) {
	    std::cerr << "Failed to parse Json" << std::endl;
    }

    library platform = library::MPI;
    std::vector<pattern> patterns;

    if (root.isMember("steps") && root["steps"].isArray()) {
            Json::Value& stepsArray = root["steps"];
            for (const auto& stepNumber : stepsArray) {
		std::cout << "STEP " << step << std::endl;
	        step = step + 1;	
	    	std::string library = stepNumber["library"].asString();
		if (library == "mpi") {
		   platform = library::MPI;
		} else if (library == "ipc_put") {
			  platform = library::IPC;
		} else if (library == "ipc_get") {
			  platform = library::IPC_get;
		} else if (library == "xccl") {
			  platform = library::NCCL;
		} else {
		  std::cerr << "Unknown library" << std::endl;
		}
		Json::Value patternsArray = stepNumber["patterns"];
                for (const auto& testType : patternsArray) {
		    Comm<int> test(platform);
                    if (testType.asString() == "gather") {
                       std::cout << "adding gather_test" << std::endl;
		       patterns.push_back(pattern::gather);
                    } else if (testType.asString() == "p2p") {
			      std::cout << "adding p2p_test" << std::endl;
			      patterns.push_back(pattern::p2p);
                    } else if (testType.asString() == "scatter") {
			      std::cout << "adding scatter_test" << std::endl;
                              patterns.push_back(pattern::scatter);
                    } else if (testType.asString() == "allgather") {
			      std::cout << "adding allgather test" << std::endl;
			      patterns.push_back(pattern::allgather);
                    } else if (testType.asString() == "alltoall") {
                              std::cout << "adding alltoall_test" << std::endl;
		              patterns.push_back(pattern::alltoall);
                    } else if (testType.asString() == "bcast") {
			      std::cout << "adding bcast_test" << std::endl;
			      patterns.push_back(pattern::broadcast);
                    }
                }
            }
    }
    return 0;
}
