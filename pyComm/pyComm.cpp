#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
//  #include "../comm.h"

namespace py = pybind11;

#include <mpi.h>
#include <stdio.h> // for printf
#include <string.h> // for memcpy
#include <algorithm> // for std::sort
#include <vector> // for std::vector

namespace CommBench {
    static int printid = 0;
    enum library {null, MPI, NCCL, IPC, STAGE, numlib};

    template <typename T>
    class Comm {
        public:
            const library lib;
            Comm(library lib);
    }
}

template <typename T>
Comm<T>::Comm(library lib) : lib(lib) {
    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);
    if(myid == printid) {
        printf("success.\n");
    }
}


PYBIND11_MODULE(pyComm, m) {
    py::enum_<CommBench::library>(m, "library")
        .value("null", CommBench::library::null)
        .value("MPI", CommBench::library::MPI)
        .value("NCCL", CommBench::library::NCCL)
        .value("IPC", CommBench::library::IPC)
        .value("STAGE", CommBench::library::STAGE)
        .value("numlib", CommBench::library::numlib);
    py::class_<CommBench::Comm<int>>(m, "Comm")
        .def(py::init<CommBench::library>())
        // .def("add", &CommBench::Comm<int>::add)
        // .def("start", &CommBench::Comm<int>::start)
        // .def("wait", &CommBench::Comm<int>::wait);
}
