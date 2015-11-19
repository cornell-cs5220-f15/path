#!/bin/bash

touch indices.txt

# ./submit-mpi 2000 1 | cut -d '.' -f 1 >> indices.txt
./submit-mpi 2000 2 | cut -d '.' -f 1 >> indices.txt
./submit-mpi 2000 4 | cut -d '.' -f 1 >> indices.txt
./submit-mpi 2000 8 | cut -d '.' -f 1 >> indices.txt
./submit-mpi 2000 16 | cut -d '.' -f 1 >> indices.txt
./submit-mpi 2000 24 | cut -d '.' -f 1 >> indices.txt


# ./submit 5000 10| cut -d '.' -f 1 >> indices.txt
# ./submit 5000 20| cut -d '.' -f 1 >> indices.txt


# ./submit 5000 40| cut -d '.' -f 1 >> indices.txt
# ./submit 5000 80| cut -d '.' -f 1 >> indices.txt
# ./submit 5000 160| cut -d '.' -f 1 >> indices.txt
# ./submit 5000 200| cut -d '.' -f 1 >> indices.txt
# ./submit 5000 236| cut -d '.' -f 1 >> indices.txt
# ./submit 5000 | cut -d '.' -f 1 >> indices.txt
# ./submit 10000 | cut -d '.' -f 1 >> indices.txt

# ./submit wave 1000 40 | cut -d '.' -f 1 >> indices.txt