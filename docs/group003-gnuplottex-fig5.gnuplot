set terminal cairolatex color
set output 'group003-gnuplottex-fig5.tex'
set logscale y
set yr[1.0:100]
plot "./benchmarking/blocked/strong/p025strong.txt" u 1:2 t 'Blocked p = 0.025' w lp, \
"./benchmarking/blocked/strong/p05strong.txt" u 1:2 t 'Blocked p = 0.050' w lp, \
"./benchmarking/vectorized/strong/p025strong.txt" u 1:2 t 'Initial p = 0.025' w lp, \
"./benchmarking/vectorized/strong/p05strong.txt" u 1:2 t 'Initial p = 0.050' w lp
