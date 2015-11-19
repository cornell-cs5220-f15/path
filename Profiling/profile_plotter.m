% n = [500, 1000, 2000, 4000, 8000];
% square = [0.357, 4.760, 39.880, 726.863, 4282.106];
% kmp_barrier = [1.176, 1.251, 3.427, 17.314, 11.025];
% kmpc_reduce_nowait = [0.348, 0.240, 1.700, 8.456, 8.328];
% kmp_fork_barrier = [1.659, 2.530, 2.529, 5.731, 7.711];
% 
% barriers = kmp_barrier + kmp_fork_barrier;

% plot(n, square);
% legend('square');
% xlabel('Number of nodes');
% ylabel('CPU time (sec)');
% title('Profiling of OpenMP version (24 threads)');

%%% Copy memory becomes a time issue for large N
%
% plot(n, barriers, n, kmpc_reduce_nowait);
% legend('barriers','reduce');
% xlabel('Number of nodes');
% ylabel('CPU time (sec)');
% title('Profiling of OpenMP version (24 threads)');
%
% nthreads = [1, 2, 3, 4, 5, 8, 24];
% cpu_time = [32.346, 29.649, 28.859, 25.663, 31.019, 28.594, 47.625];
% plot(nthreads, cpu_time);
% xlabel('Number of threads');
% ylabel('CPU time (sec)');
% title('Profiling of OpenMP version (2000 nodes)');
% % n = 2000 nodes

range = [2, 4, 8, 16, 24];
lgrange = [1, 2, 3, 4, log(24)/log(2)];
serial_2000 = 27.2595;
serial_2048 = 128.367;
omp_2000 = [14.0572, 7.48329, 3.6316, 2.4283, 2.82006];
omp_2048 = [59.6255, 24.1244, 27.7544, 12.3269, 17.6007];
MPI_loop_2000 = [4.30714, 4.09089, 6.69964, 19.4977, 19.7719];
MPI_loop_2048 = [3.08672, 3.60793, 5.60671, 16.0228, 16.2416];
MPI_trsp_2000 = [3.75525, 4.3389, 3.73869, 5.77921, 4.455];
MPI_trsp_2048 = [4.6726, 3.64148, 3.49973, 5.88343, 5.32963];

serial_512 = 0.739196;
omp_512 = [0.377014, 0.203083, 0.110632, 0.089843, 0.128389];
MPI_loop_512 = [0.162072, 0.144989, 0.114844, 0.065762, 0.068635];
MPI_trsp_512 = [0.173914, 0.22332, 0.212149, 0.0765021, 0.072273];

plot(lgrange, log(omp_512), lgrange, log(MPI_loop_512), lgrange, log(MPI_trsp_512), [1,5], log([serial_512,serial_512]), 'k--')
legend('OpenMP orig','MPI loop reorder','MPI transpose');

%plot(lgrange, log(omp_2000), lgrange, log(MPI_loop_2000), lgrange, log(MPI_trsp_2000), [1,5], log([serial_2000,serial_2000]), 'k--')
%xlabel('log_2(# of processors)');
%ylabel('log(Wall clock time)');
%title('Strong Scaling (n = 2000 nodes)');
%legend('OpenMP orig','MPI loop reorder','MPI transpose');

%plot(lgrange, log(omp_2048), lgrange, log(MPI_loop_2048), lgrange, log(MPI_trsp_2048), [1,5], log([serial_2048,serial_2048]), 'k--')
% xlabel('log_2(# of processors)');
% ylabel('log(Wall clock time)');
% title('Strong Scaling (n = 2048 nodes)');
%legend('OpenMP orig','MPI loop reorder','MPI transpose');