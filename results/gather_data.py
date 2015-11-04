import os
import time

n= range(500,5100,500)
threads_mpi = range(2,25,2) + [1]
threads_omp = [1]

def create_pbs(name, threads, n, mpi=False):
    fname = "{}.pbs".format(name)
    out = open(fname,'w')
    s = '''#!/bin/sh -l
#PBS -l nodes=1:ppn=24
#PBS -l walltime=5:00:00
#PBS -N {}
#PBS -j oe
module load cs5220
cd $PBS_O_WORKDIR
'''.format(name)
    out.write(s)
    for t in threads:
        for ni in n:
            if mpi:
                line = "mpirun -n {} ../{}.x -n {}\n".format(t, name, ni)
            else:
                line = "../{}.x -n {}\n".format(name, ni)
            out.write(line)
    out.close()
    os.system("qsub {}".format(fname))

create_pbs("hybrid", mpi=True, n=n, threads=threads_mpi)
create_pbs("omp", mpi=False, n=n, threads=threads_mpi)
create_pbs("mpi", mpi=True, threads=threads_omp, n=n)
