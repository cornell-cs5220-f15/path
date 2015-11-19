#!/bin/bash

touch indices.txt

./submit 5000 1| cut -d '.' -f 1 >> indices.txt
./submit 5000 10| cut -d '.' -f 1 >> indices.txt
./submit 5000 20| cut -d '.' -f 1 >> indices.txt
./submit 5000 40| cut -d '.' -f 1 >> indices.txt
./submit 5000 80| cut -d '.' -f 1 >> indices.txt
./submit 5000 160| cut -d '.' -f 1 >> indices.txt
./submit 5000 200| cut -d '.' -f 1 >> indices.txt
./submit 5000 236| cut -d '.' -f 1 >> indices.txt
# ./submit 5000 | cut -d '.' -f 1 >> indices.txt
# ./submit 10000 | cut -d '.' -f 1 >> indices.txt

# ./submit wave 1000 40 | cut -d '.' -f 1 >> indices.txt