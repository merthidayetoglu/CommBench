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
    static MPI_Comm comm_mpi;
    static bool initialized_MPI = false;

    void mpi_init();
    void mpi_fin();

    template <typename T>
    class Comm {
        public:
            const library lib;
            Comm(library lib);
    };
<<<<<<< HEAD
=======
};

void CommBench::mpi_init() {
    MPI_Init(NULL, NULL);
}

void CommBench::mpi_fin() {
    MPI_Finalize();
>>>>>>> 2550bf06872e0318b5ac157dc56b1faa89896fbb
}

template <typename T>
CommBench::Comm<T>::Comm(CommBench::library lib) : lib(lib) {
    if(!CommBench::initialized_MPI)
	MPI_Comm_dup(MPI_COMM_WORLD, &CommBench::comm_mpi);
    int myid;
    int numproc;
    MPI_Comm_rank(CommBench::comm_mpi, &myid);
    MPI_Comm_size(CommBench::comm_mpi, &numproc);
    if(myid == CommBench::printid) {
        printf("success.\n");
    }
};


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
        .def("mpi_init", &CommBench::mpi_init)
	.def("mpi_fin", &CommBench::mpi_fin);
	// .def("add", &CommBench::Comm<int>::add)
        // .def("start", &CommBench::Comm<int>::start)
        // .def("wait", &CommBench::Comm<int>::wait);
}
