#!/bin/bash

touch indices.txt

./submit 1000 | cut -d '.' -f 1 >> indices.txt
./submit 2000 | cut -d '.' -f 1 >> indices.txt
./submit 3000 | cut -d '.' -f 1 >> indices.txt
./submit 4000 | cut -d '.' -f 1 >> indices.txt
./submit 5000 | cut -d '.' -f 1 >> indices.txt
./submit 6000 | cut -d '.' -f 1 >> indices.txt
# ./submit 5000 | cut -d '.' -f 1 >> indices.txt
# ./submit 10000 | cut -d '.' -f 1 >> indices.txt

# ./submit wave 1000 40 | cut -d '.' -f 1 >> indices.txt