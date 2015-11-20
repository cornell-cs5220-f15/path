set terminal cairolatex color
set output 'group003-gnuplottex-fig3.tex'
#set key at 15.8,41
set size 1.0,0.75
unset log
unset label
#set xtic auto offset 0,-0.5 font ",10"
#set ytic 25 font ",10"
#Set Info
set xlabel "Number of Threads"
#set xlabel offset 0,-0.5
set ylabel "Scaling Efficiency"
#set ylabel offset -1.25,0
#set xr [0.0:16.0]
set yr [0.0:1.0]
plot "./benchmarking/mpi/strong/p025strong.txt" u 1:(490.5691/$2/$1) t 'p = 0.025' w linespoints, \
"./benchmarking/mpi/strong/p05strong.txt" u 1:(538.9763960838318/$2/$1) t 'p = 0.050' w lp, \
"./benchmarking/mpi/strong/p10strong.txt" u 1:(325.8776059150/$2/$1) t 'p = 0.100' w lp, \
"./benchmarking/mpi/strong/p30strong.txt" u 1:(325.54634/$2/$1) t 'p = 0.300' w lp
