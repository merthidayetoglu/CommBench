#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../comm.h"

namespace py = pybind11;

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
        .def("add", &CommBench::Comm<int>::add)
        .def("start", &CommBench::Comm<int>::start)
        .def("wait", &CommBench::Comm<int>::wait);
}
