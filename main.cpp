#include "commbench.h"
#define ROOT 0
#include "validate.h"
#include <array>
#include <cstdio>
#ifdef CONFIG_FILE
#include <json/json.h>
#endif
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

using namespace CommBench;

enum pattern { p2p, gather, scatter, broadcast, reduce, alltoall, allgather };
int myid_loc;

struct step {
  std::vector<pattern> patterns;
  library lib;
};

/*template <typename... Args> void FATAL_ERROR(const char *fmt, Args... args) {
  if (myid_loc == printid) {
    fprintf(stderr, "FATAL ERROR: ");
    fprintf(stderr, fmt, args...);
    std::abort();
  }
}

template <typename... Args> void ERROR(const char *fmt, Args... args) {
  if (myid_loc == printid) {
    fprintf(stderr, "ERROR: ");
    fprintf(stderr, fmt, args...);
  }
}

template <typename... Args> void WARNING(const char *fmt, Args... args) {
  if (myid_loc == printid) {
    fprintf(stderr, "WARNING: ");
    fprintf(stderr, fmt, args...);
  }
}*/

std::unordered_map<std::string, std::vector<std::string>>
parseArgs(int argc, char *argv[]) {
  static const std::array<std::string, 5> valid_args = {"use", "pattern", "validate",
                                           "nbytes", "file"};
  int i = 1;
  std::unordered_map<std::string, std::vector<std::string>> args;
  std::string prev = "";
  while (i < argc) {
    std::string cur(argv[i]);
    // potentially expand aliases for args
    if (cur.substr(0, 2) == "--") {
      std::string arg = cur.substr(2);
      // check for valid args or maybe do that elsewhere
      bool valid = false;
      for (int j = 0; j < valid_args.size(); j++)
        if (valid_args[j] == arg) {
          valid = true;
          break;
        }
      if (!valid) {
//FATAL_ERROR("unknown argument \"%s\"\n", argv[i]);
	std::cout << "unknown argument \"%s\"\n" << std::endl;
	std::exit(EXIT_FAILURE);
      }
      if (args.find(arg) == args.end())
        args.insert({arg, {}});
      prev = arg;
    } else if (prev != "") {
      // validate input

      args[prev].push_back(cur);
    } else {
//FATAL_ERROR("unknown argument", argv[i]);
      std::cout << "unknown argument" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    i++;
  }
  return args;
}

library parseLib(const std::string &libStr) {
  if (libStr == "mpi")
    return library::MPI;
  else if (libStr == "ipc_put") {
#if !(defined(PORT_CUDA) || defined(PORT_HIP) || defined(PORT_ONEAPI))
    //FATAL_ERROR("Cannot use IPC without compiling for CUDA, "
      //          "ROCm, or OneAPI\n");
    std::cout << "Cannot use IPC without compiling for CUDA, ROCm, or OneAPI" << std::endl;
    std::exit(EXIT_FAILURE);
#endif
    return library::IPC;
  } else if (libStr == "ipc_get") {
#if !(defined(PORT_CUDA) || defined(PORT_HIP) || defined(PORT_ONEAPI))
    //FATAL_ERROR("Cannot use IPC without compiling for CUDA, "
      //          "ROCm, or OneAPI\n");
    std::cout << "Cannot use IPC without compiling for CUDA, ROCm, or OneAPI" << std::endl;
    std::exit(EXIT_FAILURE);
#endif
    return library::IPC_get;
  } else if (libStr == "xccl") {
#if !(defined(PORT_CUDA) || defined(PORT_HIP) || defined(PORT_ONEAPI))
    //FATAL_ERROR("Cannot use IPC without compiling for CUDA, "
    //            "ROCm, or OneAPI\n");
    std::cout << "Cannot use IPC without compiling for CUDA, ROCm, or OneAPI" << std::endl;
    std::exit(EXIT_FAILURE);
#endif
#ifndef CAP_NCCL
    //FATAL_ERROR("Not compiled for using XCCL\n");
    std::cout << "Not compiled for using XCCL" << std::endl;
    std::exit(EXIT_FAILURE);
#endif
    return library::NCCL;
  } else {
    //FATAL_ERROR("Unknown communication library option \"%s\". Please "
    //            "specify one of: mpi, ipc_put, ipc_get, or xccl.\n",
    //            libStr.c_str());
    std::cout << "Unknown communication library option. Please specify one of: mpi, ipc_get, or xccl." << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

pattern parsePattern(const std::string &patStr) {
  if (patStr == "p2p")
    return pattern::p2p;
  else if (patStr == "broadcast")
    return pattern::broadcast;
  else if (patStr == "gather")
    return pattern::gather;
  else if (patStr == "scatter")
    return pattern::scatter;
  else if (patStr == "alltoall")
    return pattern::alltoall;
  else if (patStr == "allgather")
    return pattern::allgather;
  else {
    std::cout << "Unknown communication pattern. Please use one of: p2p, broadcast, gather, scatter, alltoall, or allgather." << std::endl;
    std::exit(EXIT_FAILURE);
    //FATAL_ERROR(
    //    "Unknown communication pattern \"%s\". Please use one "
    //    "of: p2p, broadcast, gather, scatter, alltoall, or allgather.\n",
    //    patStr.c_str());
  }
}

int main(int argc, char *argv[]) {
  init();

  int numproc = CommBench::numproc;
  myid_loc = CommBench::myid;

  std::unordered_map<std::string, std::vector<std::string>> args =
      parseArgs(argc, argv);

  if (args.find("pattern") != args.end() && args.find("file") != args.end()) {
    std::cout << "Cannot use both the --file and --pattern options." << std::endl;
    std::exit(EXIT_FAILURE);
  }
    //FATAL_ERROR("Cannot use both the --file and --pattern options.\n");

  library lib_def = library::MPI;
  if (args["use"].size() != 0) {
    lib_def = parseLib(args["use"][0]);
  } else {
    std::cout << "No communication library specified, using MPI by default" << std::endl;
    //WARNING("No communication library specified, using MPI by default\n");
  }

  std::vector<step> steps;
  if (args.find("file") != args.end()) {
    #ifndef CONFIG_FILE 
      std::cout << "Cannot use the --file flag unless compiled with jsoncpp support." << std::endl;
      std::exit(EXIT_FAILURE);
      //FATAL_ERROR("Cannot use the --file flag unless compiled with jsoncpp support.\n");
    #else
    std::ifstream file(args["file"][0], std::ifstream::binary);
    if (!file.is_open())
      std::cout << "Could not open file." << std::endl;
      std::exit(EXIT_FAILURE);
      //FATAL_ERROR("Could not open file \"%s\"\n", args["file"][0]);

      Json::Value root;
      Json::CharReaderBuilder builder;
      std::string errs;
      int step = 1;

      if (!Json::parseFromStream(builder, file, &root, &errs))
	std::cout << "Failed to parse JSON" << std::endl;
	std::exit(EXIT_FAILURE);
        //FATAL_ERROR("Failed to parse JSON\n");

      if (root.isMember("steps") && root["steps"].isArray()) {
        Json::Value &stepsArray = root["steps"];
        for (const auto &stepNumber : stepsArray) {
          std::cout << "STEP " << step << std::endl;
          step = step + 1;
          library lib = lib_def;
          if (stepNumber.isMember("library"))
            lib = parseLib(stepNumber["library"].asString());
          std::vector<pattern> patterns;
          Json::Value patternsArray = stepNumber["patterns"];
          for (const auto &testType : patternsArray)
            patterns.push_back(parsePattern(testType.asString()));
          steps.push_back({patterns, lib});
        }
      }
      #endif
    } else {
      std::vector<pattern> patterns;
      for (int i = 0; i < args["pattern"].size(); i++) {
        patterns.push_back(parsePattern(args["pattern"][i]));
      }

      if (patterns.size() == 0) {
	std::cout << "No communication pattern specified, using P2P by default." << std::endl;
        //WARNING("No communication pattern specified, using P2P by default\n");
        patterns.push_back(p2p);
      }
      steps.push_back({patterns, lib_def});
    }

    bool run_validate = false;
    if (args.find("validate") != args.end())
      run_validate = true;

    int *sendbuf;
    int *recvbuf;
    size_t numbytes = 1e8;
    if (args.find("nbytes") != args.end()) {
      if (args["nbytes"].size() == 0)
	std::cout << "Missing number of bytes argument for --nbytes" << std::endl;
        std::exit(EXIT_FAILURE);
      	//FATAL_ERROR("Missing number of bytes argument for --nbytes");
      try {
        numbytes = std::stoull(args["nbytes"][0]);
      } catch (...) {
	std::cout << "Invalid input to --nbytes" << std::endl;
	std::exit(EXIT_FAILURE);
        //FATAL_ERROR("Invalid input \"%s\" to --nbytes.", args["nbytes"][0]);
      }
    }

    allocate(sendbuf, numbytes * numproc);
    allocate(recvbuf, numbytes * numproc);
    for (int j = 0; j < steps.size(); j++) {
      const auto &patterns = steps[j].patterns;
      for (int i = 0; i < patterns.size(); i++) {
        Comm<int> test(steps[j].lib);
        switch (patterns[i]) {
        case pattern::p2p:
          test.add(sendbuf, recvbuf, numbytes, 0, 1);
          break;
        case pattern::broadcast:
          for (int p = 0; p < numproc; p++)
            test.add(sendbuf, 0, recvbuf, 0, numbytes, ROOT, p);
          break;
        case pattern::gather:
          for (int p = 0; p < numproc; p++)
            test.add(sendbuf, 0, recvbuf, p * numbytes, numbytes, p, ROOT);
          break;
        case pattern::scatter:
          for (int p = 0; p < numproc; p++)
            test.add(sendbuf, p * numbytes, recvbuf, 0, numbytes, ROOT, p);
          break;
        case pattern::alltoall:
          for (int sender = 0; sender < numproc; sender++)
            for (int recver = 0; recver < numproc; recver++)
              test.add(sendbuf, recver * numbytes, recvbuf, sender * numbytes,
                       numbytes, sender, recver);
          break;
        case pattern::allgather:
          for (int sender = 0; sender < numproc; sender++)
            for (int recver = 0; recver < numproc; recver++)
              test.add(sendbuf, 0, recvbuf, sender * numbytes, numbytes, sender,
                       recver);
          break;
        default:; // error?
        }
        if (run_validate)
          validate(sendbuf, recvbuf, numbytes, patterns[i], test);
        else {
#ifndef BENCH_CALIPER
          test.measure(5, 10, numbytes * numproc);
#else
          test.measure_caliper(5, 10);
#endif
        }
      }
    }
    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();

    return 0;
  }
