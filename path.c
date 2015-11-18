#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include "mt19937p.h"
#include <mpi.h>

static inline void infinitize(int n, int* l)
{
    for (int i = 0; i < n*n; ++i)
        if (l[i] == 0)
            l[i] = n+1;
}

static inline void deinfinitize(int n, int* l)
{
    for (int i = 0; i < n*n; ++i)
        if (l[i] == n+1)
            l[i] = 0;
}

int square(int n,
           int n1,              // starting row
           int n2,              // (one after) ending row
           int* restrict l,     // Partial distance at step s
           int* restrict lnew)  // Partial distance at step s+1
{
    int done = 1;
    for (int j = n1; j < n2; ++j) {
        for (int i = 0; i < n; ++i) {
            int lij = lnew[(j-n1)*n+i];
            for (int k = 0; k < n; ++k) {
                int lik = l[k*n+i];
                int lkj = l[j*n+k];
                if (lik + lkj < lij) {
                    lij = lik+lkj;
                    done = 0;
                }
            }
            lnew[(j-n1)*n+i] = lij;
        }
    }
    return done;
}
void shortest_paths(int n, int* restrict l, int rank, int nthreads)
{
    // Generate l_{ij}^0 from adjacency matrix representation
    infinitize(n, l);
    for (int i = 0; i < n*n; i += n+1)
        l[i] = 0;

    // NOTE: We assume that problem size is divisible by number of processors.
    int numRows = n / nthreads; // # rows per processor
    int n1 = rank * numRows; // starting position
    // handle the case where # rows in l is not divisble by # threads
    int n2 = (rank < nthreads - 1) ? n1 + numRows : n; 

    int* restrict lnew = (int*) calloc(n*(n2-n1), sizeof(int));
    memcpy(lnew, l+n*n1, n*(n2 - n1) * sizeof(int));
    // Repeated squaring until nothing changes
    // Everyone calculate one step of their local nodes (idx based off rank)
    for (int globaldone = 0; !globaldone;) {
        int localdone = square(n, n1, n2, l, lnew);
        MPI_Allreduce(&localdone, &globaldone, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD); // is every process done?
        MPI_Allgather(lnew, n*(n2 - n1), MPI_INT, l, n*(n2 - n1), MPI_INT, MPI_COMM_WORLD); // copy to l matrix
    }

    free(lnew);
    deinfinitize(n, l);
}

int* gen_graph(int n, double p)
{
    int* l = calloc(n*n, sizeof(int));
    struct mt19937p state;
    sgenrand(10302011UL, &state);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i)
            l[j*n+i] = (genrand(&state) < p);
        l[j*n+j] = 0;
    }
    return l;
}

int fletcher16(int* data, int count)
{
    int sum1 = 0;
    int sum2 = 0;
    for(int index = 0; index < count; ++index) {
          sum1 = (sum1 + data[index]) % 255;
          sum2 = (sum2 + sum1) % 255;
    }
    return (sum2 << 8) | sum1;
}

void write_matrix(const char* fname, int n, int* a)
{
    FILE* fp = fopen(fname, "w+");
    if (fp == NULL) {
        fprintf(stderr, "Could not open output file: %s\n", fname);
        exit(-1);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) 
            fprintf(fp, "%d ", a[j*n+i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}


const char* usage =
    "path.x -- Parallel all-pairs shortest path on a random graph\n"
    "Flags:\n"
    "  - n -- number of nodes (200)\n"
    "  - p -- probability of including edges (0.05)\n"
    "  - i -- file name where adjacency matrix should be stored (none)\n"
    "  - o -- file name where output matrix should be stored (none)\n";

int main(int argc, char** argv)
{
    int n    = 200;            // Number of nodes
    double p = 0.05;           // Edge probability
    const char* ifname = NULL; // Adjacency matrix file name
    const char* ofname = NULL; // Distance matrix file name

    // Option processing
    extern char* optarg;
    const char* optstring = "hn:d:p:o:i:";
    int c;
    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
        case 'h':
            fprintf(stderr, "%s", usage);
            return -1;
        case 'n': n = atoi(optarg); break;
        case 'p': p = atof(optarg); break;
        case 'o': ofname = optarg;  break;
        case 'i': ifname = optarg;  break;
        }
    }

    int rank, nthreads;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nthreads);

    // gen graph for all matrices, inefficient but not bottleneck (squaring step is)
    int *l = gen_graph(n, p);

    if (ifname && rank == 0)
        write_matrix(ifname, n, l);

    // Time the shortest paths code    
    double t0 = MPI_Wtime();
    shortest_paths(n, l, rank, nthreads);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("== MPI with %d threads\n", nthreads);
        printf("n:     %d\n", n);
        printf("p:     %g\n", p);
        printf("Time:  %g\n", t1-t0);
        printf("Check: %X\n", fletcher16(l, n*n));
    }

    // Generate output file
    if (ofname && rank == 0)
        write_matrix(ofname, n, l);

    // Clean up
    free(l);
    MPI_Finalize();
    return 0;
}

