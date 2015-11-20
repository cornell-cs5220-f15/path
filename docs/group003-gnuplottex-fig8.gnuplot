set terminal cairolatex color
set output 'group003-gnuplottex-fig8.tex'
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
set yr [0.0:1.5]
plot "./benchmarking/blocked_phi/strong/p025strong.txt" u 1:(1350.99216/$2/$1*8) t 'p = 0.025' w linespoints, \
"./benchmarking/blocked_phi/strong/p05strong.txt" u 1:(1790.88339304/$2/$1*4) t 'p = 0.050' w lp, \
"./benchmarking/blocked_phi/strong/p10strong.txt" u 1:(1807.047016/$2/$1*4) t 'p = 0.100' w lp, \
"./benchmarking/blocked_phi/strong/p30strong.txt" u 1:(1767.469844/$2/$1*4) t 'p = 0.300' w lp
