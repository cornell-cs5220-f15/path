#!/bin/bash

touch indices.txt

NT="1 10 20 40 80 160 200 236"
# NT="10 20"
for p in $NT; do
    ./submit $(python weak.py $p) $p | cut -d '.' -f 1 >> indices.txt
    echo $p "processors with domain size " $(python weak.py $p)
done

# ./submit wave 400 1 | cut -d '.' -f 1 >> indices.txt
# ./submit wave 640 4 | cut -d '.' -f 1 >> indices.txt
# ./submit wave 800 8 | cut -d '.' -f 1 >> indices.txt
# ./submit wave 920 12 | cut -d '.' -f 1 >> indices.txt