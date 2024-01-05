#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../comm.h"

namespace py = pybind11;

#define PORT_CUDA

template <typename T>
void CommBench::Comm<T>::pyadd(CommBench::pyalloc<T> sendbuf, size_t sendoffset, CommBench::pyalloc<T> recvbuf, size_t recvoffset, size_t count, int sendid, int recvid){
    CommBench::Comm<T>::add(sendbuf.ptr, sendoffset, recvbuf.ptr, recvoffset, count, sendid, recvid);
}

PYBIND11_MODULE(pyComm, m) {
    py::enum_<CommBench::library>(m, "library")
        .value("null", CommBench::library::null)
        .value("MPI", CommBench::library::MPI)
        .value("NCCL", CommBench::library::NCCL)
        .value("IPC", CommBench::library::IPC)
        .value("STAGE", CommBench::library::STAGE)
        .value("numlib", CommBench::library::numlib);
    py::class_<CommBench::pyalloc<int>>(m, "pyalloc")
	.def(py::init<size_t>())
        .def("free", &CommBench::pyalloc<int>::pyfree);
    py::class_<CommBench::Comm<int>>(m, "Comm")
        .def(py::init<CommBench::library>())
        .def("add", &CommBench::Comm<int>::pyadd)
        .def("finalize", &CommBench::Comm<int>::finalize)
        .def("add_lazy", &CommBench::Comm<int>::add_lazy)
        .def("setprintid", &CommBench::setprintid)
        .def("measure", static_cast<void (CommBench::Comm<int>::*)(int, int)>(&CommBench::Comm<int>::measure), "measure the latency")
        // .def("add", &CommBench::Comm<int>::add)
        .def("start", &CommBench::Comm<int>::start)
        .def("wait", &CommBench::Comm<int>::wait);
}
