n = [500, 1000, 2000, 4000, 8000];
square = [0.357, 4.760, 39.880, 726.863, 4282.106];
kmp_barrier = [1.176, 1.251, 3.427, 17.314, 11.025];
kmpc_reduce_nowait = [0.348, 0.240, 1.700, 8.456, 8.328];
kmp_fork_barrier = [1.659, 2.530, 2.529, 5.731, 7.711];

barriers = kmp_barrier + kmp_fork_barrier;

% plot(n, square);
% legend('square');
% xlabel('Number of nodes');
% ylabel('CPU time (sec)');
% title('Profiling of OpenMP version (24 threads)');

%%% Copy memory becomes a time issue for large N
%
plot(n, barriers, n, kmpc_reduce_nowait);
legend('barriers','reduce');
xlabel('Number of nodes');
ylabel('CPU time (sec)');
title('Profiling of OpenMP version (24 threads)');
%
% nthreads = [1, 2, 3, 4, 5, 8, 24];
% cpu_time = [32.346, 29.649, 28.859, 25.663, 31.019, 28.594, 47.625];
% plot(nthreads, cpu_time);
% xlabel('Number of threads');
% ylabel('CPU time (sec)');
% title('Profiling of OpenMP version (2000 nodes)');
% % n = 2000 nodes