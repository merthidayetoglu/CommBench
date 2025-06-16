#include "commbench.h"
#include <algorithm>
#define ROOT 0
#include "validate.h"
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

using namespace CommBench;

enum pattern { p2p, gather, scatter, broadcast, reduce, alltoall, allgather };
int myid_loc;

template <typename... Args> void FATAL_ERROR(const char *fmt, Args... args) {
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
}

std::unordered_map<std::string, std::vector<std::string>>
parseArgs(int argc, char *argv[]) {
  static const std::string valid_args[] = {"use", "pattern", "validate", "nbytes"};
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
      for (int j = 0; j < 4; j++)
        if (valid_args[j] == arg) {
          valid = true;
          break;
        }
      if (!valid) {
        FATAL_ERROR("unknown argument \"%s\"\n", argv[i]);
      }
      if (args.find(arg) == args.end())
        args.insert({arg, {}});
      prev = arg;
    } else if (prev != "") {
      // validate input

      args[prev].push_back(cur);
    } else {
      FATAL_ERROR("unknown argument \"%s\"\n", argv[i]);
    }
    i++;
  }
  return args;
}

int main(int argc, char *argv[]) {
  init();
  int numproc = CommBench::numproc;
  myid_loc = CommBench::myid;

  std::unordered_map<std::string, std::vector<std::string>> args =
      parseArgs(argc, argv);

  library lib = library::MPI;
  if (args["use"].size() != 0) {
    std::string &libStr = args["use"][0];
    if (libStr == "mpi")
      lib = library::MPI;
    else if (libStr == "ipc_put") {
#if !(defined(PORT_CUDA) || defined(PORT_HIP) || defined(PORT_ONEAPI))
      FATAL_ERROR("Cannot use IPC without compiling for CUDA, "
                  "ROCm, or OneAPI\n");
#endif
      lib = library::IPC;
    } else if (libStr == "ipc_get") {
#if !(defined(PORT_CUDA) || defined(PORT_HIP) || defined(PORT_ONEAPI))
      FATAL_ERROR("Cannot use IPC without compiling for CUDA, "
                  "ROCm, or OneAPI\n");
#endif
      lib = library::IPC_get;
    } else if (libStr == "xccl") {
#if !(defined(PORT_CUDA) || defined(PORT_HIP) || defined(PORT_ONEAPI))
      FATAL_ERROR("Cannot use IPC without compiling for CUDA, "
                  "ROCm, or OneAPI\n");
#endif
#ifndef CAP_NCCL
      FATAL_ERROR("Not compiled for using XCCL\n");
#endif
      lib = library::NCCL;
    } else {
      FATAL_ERROR("Unknown communication library option \"%s\". Please "
                  "specify one of: mpi, ipc_put, ipc_get, or xccl.\n",
                  libStr.c_str());
    }
  } else {
    WARNING("No communication library specified, using MPI by default\n");
  }

  std::vector<pattern> patterns;
  for (int i = 0; i < args["pattern"].size(); i++) {
    std::string &patStr = args["pattern"][i];
    if (patStr == "p2p")
      patterns.push_back(pattern::p2p);
    else if (patStr == "broadcast")
      patterns.push_back(pattern::broadcast);
    else if (patStr == "gather")
      patterns.push_back(pattern::gather);
    else if (patStr == "scatter")
      patterns.push_back(pattern::scatter);
    else if (patStr == "alltoall")
      patterns.push_back(pattern::alltoall);
    else if (patStr == "allgather")
      patterns.push_back(pattern::allgather);
    else {
      FATAL_ERROR(
          "Unknown communication pattern \"%s\". Please use one "
          "of: p2p, broadcast, gather, scatter, alltoall, or allgather.\n",
          patStr.c_str());
    }
  }

  if (patterns.size() == 0) {
    WARNING("No communication pattern specified, using P2P by default\n");
    patterns.push_back(p2p);
  }

  bool run_validate = false;
  if (args.find("validate") != args.end())
    run_validate = true;

  int *sendbuf;
  int *recvbuf;
  size_t numbytes = 1e8;
  if (args.find("nbytes") != args.end()) {
    if (args["nbytes"].size() == 0)
        FATAL_ERROR("Missing number of bytes argument for --nbytes");
    try {
        numbytes = std::stoull(args["nbytes"][0]);
    } catch (...) {
        FATAL_ERROR("Invalid input \"%s\" to --nbytes.", args["nbytes"][0]);
    }
  }

  allocate(sendbuf, numbytes * numproc);
  allocate(recvbuf, numbytes * numproc);

  for (int i = 0; i < patterns.size(); i++) {
    Comm<int> test(lib);
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
    else
      test.measure(5, 10, numbytes * numproc);
  }

  free(sendbuf);
  free(recvbuf);

  MPI_Finalize();

  return 0;
}
