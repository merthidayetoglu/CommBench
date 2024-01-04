import pyComm
#from mpi4py import MPI
import numpy as np

#mpi_rank = MPI.COMM_WORLD.Get_rank()
#mpi_size = MPI.COMM_WORLD.Get_size()
#print(mpi_rank, mpi_size)
#if(MPI.Is_initialized()):
#    mpi_rank = MPI.COMM_WORLD.Get_rank()
#    mpi_size = MPI.COMM_WORLD.Get_size()
#    print(mpi_rank, mpi_size)
pyComm.Comm.setprintid(0)
buf = pyComm.pyalloc(1024)
pyComm.pyalloc.free(buf)
c = pyComm.Comm(pyComm.library.MPI)
c.add_lazy(1024, 0, 1)
c.measure(5, 10)
c.finalize()

#c.add(a, 0, b, 0, a.size, 0, 1)
#c.start()
#c.wait()
