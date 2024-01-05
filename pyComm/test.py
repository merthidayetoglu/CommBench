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
sendbuf = pyComm.pyalloc(1024)
recvbuf = pyComm.pyalloc(1024)
c = pyComm.Comm(pyComm.library.MPI)
# c.add_lazy(1024, 0, 1)
c.add(sendbuf, 0, recvbuf, 0, 1024, 0, 1)
c.measure(5, 10)
pyComm.pyalloc.free(sendbuf)
pyComm.pyalloc.free(recvbuf)
c.finalize()

#c.add(a, 0, b, 0, a.size, 0, 1)
#c.start()
#c.wait()
