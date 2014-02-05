from mpi4py import MPI
comm = MPI.COMM_WORLD
print comm.rank, comm.size

jj = comm.rank

data = jj**4
print data
