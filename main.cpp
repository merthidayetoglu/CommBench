#include "commbench.h"
#include <string>
#include <vector>
#include <json/json.h>
#include <fstream>
#include <iostream>

using namespace CommBench;

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

    std::ifstream file("config_doc.json", std::ifstream::binary);
    if(!file.is_open()) {
	    std::cerr << "Could not open file!" << std::endl;
    }

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;
    std::string library = "";
    std::vector<std::string> patterns;
    int step = 1;

    if(!Json::parseFromStream(builder, file, &root, &errs)) {
	    std::cerr << "Failed to parse Json" << std::endl;
    }
    if (root.isMember("tests") && root["tests"].isArray()) {
            Json::Value& testsArray = root["tests"];
            for (const auto& test : testsArray) {
		std::cout << "STEP " << step << std::endl;
	        step = step + 1;	
	    	std::string library = test["library"].asString();
		Json::Value patternsArray = test["patterns"];
               	if (library == "mpi" || library == "xccl" || library == "ipc_put" || library == "ipc_get") {
                   std::cout << "entered mpi for loop" << std::endl;
                   for (const auto& pattern : patternsArray) {
                       if (pattern.asString() == "gather") {
                          std::cout << "running gather_test" << std::endl;
                          std::string command = "mpirun -n 2 unit_tests/" + library + "_gather_test";
                          int result = system(command.c_str());
                       } else if (pattern.asString() == "p2p") {
                                 std::cout << "running p2p_test" << std::endl;
                                 std::string command = "mpirun -n 2 unit_tests/" + library + "_p2p_test";
                                 int result = system(command.c_str());
                       } else if (pattern.asString() == "scatter") {
                                 std::string command = "mpirun -n 2 unit_tests/" + library + "_scatter_test";
                                 std::cout << "running scatter_test" << std::endl;
                                 int result = system(command.c_str());
                       } else if (pattern.asString() == "allgather") {
                                 std::string command = "mpirun -n 2 unit_tests/" + library + "_allgather_test";
                                 std::cout << "running allgather_test" << std::endl;
                                 int result = system(command.c_str());
                       } else if (pattern.asString() == "alltoall") {
                                 std::cout << "running alltoall_test" << std::endl;
                                 std::string command = "mpirun -n 2 unit_tests/" + library + "_alltoall_test";
                                 int result = system(command.c_str());
                       } else if (pattern.asString() == "bcast") {
                                 std::cout << "running bcast_test" << std::endl;
                                 std::string command = "mpirun -n 2 unit_tests/" + library + "_bcast_test";
                                 int result = system(command.c_str());
                       } else if (pattern.asString() == "xccl" || pattern.asString() == "ipc_put" || pattern.asString() == "ipc_get") {
                                 std::cout << "running default test" << std::endl;
				 std::string command = "mpirun -n 2 unit_tests/" + library + "_test";
                                 int result = system(command.c_str());
                       }
                   }
		}
            }
    }
}
