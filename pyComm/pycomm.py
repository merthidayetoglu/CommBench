#!/usr/bin/env python
import pyComm
from mpi4py import MPI
import numpy as np


#MPI.Init()
#mpi_rank = MPI.COMM_WORLD.Get_rank()
#mpi_size = MPI.COMM_WORLD.Get_size()
#print(mpi_rank, mpi_size)
a = np.array([1,2,3,4])
b = np.array([0,0,0,0])
c = pyComm.Comm(pyComm.library.MPI)
#c.add(a, 0, b, 0, a.size, 0, 1)
#c.start()
#c.wait()
#MPI.Finalize()
