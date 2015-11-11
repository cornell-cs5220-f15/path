#!/bin/bash
# Usage: ./make_csv.sh time-path.oXXXXX > timing.csv

if [ -z $1 ]; then
	echo "Error: You must specify a file of Path output"
	exit	
fi

output=$1
echo "n,runtime"
awk '/n:/ {printf "%s,", $2} 
     /Time:/ {printf "%s\n", $2}' $output
