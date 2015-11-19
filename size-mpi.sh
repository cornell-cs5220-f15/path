#!/bin/bash

touch indices.txt

./submit-mpi 1000 | cut -d '.' -f 1 >> indices.txt
./submit-mpi 2000 | cut -d '.' -f 1 >> indices.txt
./submit-mpi 3000 | cut -d '.' -f 1 >> indices.txt
./submit-mpi 4000 | cut -d '.' -f 1 >> indices.txt
./submit-mpi 5000 | cut -d '.' -f 1 >> indices.txt
./submit-mpi 6000 | cut -d '.' -f 1 >> indices.txt
# ./submit 10000 | cut -d '.' -f 1 >> indices.txt

# ./submit wave 1000 40 | cut -d '.' -f 1 >> indices.txt