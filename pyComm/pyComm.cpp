
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define PORT_CUDA
#include "../commbench.h"

namespace py = pybind11;

template <typename T>
void CommBench::Comm<T>::pyadd(CommBench::pyalloc<T> sendbuf, size_t sendoffset, CommBench::pyalloc<T> recvbuf, size_t recvoffset, size_t count, int sendid, int recvid){
    CommBench::Comm<T>::add(sendbuf.ptr, sendoffset, recvbuf.ptr, recvoffset, count, sendid, recvid);
}

/*void init() {
	MPI_Init(NULL, NULL);
}

void fin() {
	MPI_Finalize();
}*/

PYBIND11_MODULE(pyComm, m) {
    //m.def("init", &init);
    //m.def("fin", &fin);
    //m.def("setup_gpu", &setup_gpu);
    py::enum_<CommBench::library>(m, "library")
        .value("dummy", CommBench::library::dummy)
        .value("MPI", CommBench::library::MPI)
        .value("XCCL", CommBench::library::XCCL)
        .value("IPC", CommBench::library::IPC)
        .value("numlib", CommBench::library::numlib);
    py::class_<CommBench::pyalloc<int>>(m, "pyalloc")
	.def(py::init<size_t>())
        .def("free", &CommBench::pyalloc<int>::pyfree);
    py::class_<CommBench::Comm<int>>(m, "Comm")
        .def(py::init<CommBench::library>())
        .def("add", &CommBench::Comm<int>::pyadd)
        .def("add_lazy", &CommBench::Comm<int>::add_lazy)
        // .def("setprintid", &CommBench::setprintid)
        .def("measure", static_cast<void (CommBench::Comm<int>::*)(int, int)>(&CommBench::Comm<int>::measure), "measure the latency")
        .def("start", &CommBench::Comm<int>::start)
        .def("wait", &CommBench::Comm<int>::wait);
}
