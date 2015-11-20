#!/bin/bash

for i in "0.01" "0.02" "0.05" "0.10" "0.20" "0.25" "0.30" "0.40" "0.50" # "0.60" "0.75" "0.90"
do
  for j in 100 200 250 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 # 7500 10000
  do
  for k in {1..24}
  do 
    echo "#!/bin/sh -l
#PBS -l nodes=1:ppn=24
#PBS -l walltime=0:30:00
#PBS -N path
#PBS -j oe

module load cs5220
cd \$PBS_O_WORKDIR
mpirun -n ${k} ./path.x -n ${j} -p ${i} -o path_${k}_${i}_${j}.out -s path_${k}_${i}_${j}.stats" > path_${k}_${i}_${j}.pbs
   done
   done
done
