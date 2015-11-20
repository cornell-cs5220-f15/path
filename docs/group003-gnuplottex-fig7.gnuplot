set terminal cairolatex color
set output 'group003-gnuplottex-fig7.tex'
#set key at 15.8,41
set size 1.0,0.75
unset log
unset label
#set xtic auto offset 0,-0.5 font ",10"
#set ytic 25 font ",10"
#Set Info
set xlabel "Number of Threads"
#set xlabel offset 0,-0.5
set ylabel "Weak Scaling"
#set ylabel offset -1.25,0
#set xr [0.0:16.0]
set yr [0.0:1.2]
plot "./benchmarking/blocked/weak/p025weak.txt" u 1:(1.01929498/$2) t 'p = 0.025' w linespoints, \
"./benchmarking/blocked/weak/p05weak.txt" u 1:(1.18449306/$2) t 'p = 0.050' w lp, \
"./benchmarking/blocked/weak/p10weak.txt" u 1:(1.25237/$2) t 'p = 0.100' w lp, \
"./benchmarking/blocked/weak/p30weak.txt" u 1:(0.84293103/$2) t 'p = 0.300' w lp
