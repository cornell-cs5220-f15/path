#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include "mt19937p.h"
#include <mpi.h>

struct Result
{
    int n;
    int p;
    double t0;
    double t1;
    char* ofname;
};

//ldoc on

/**
 * # The basic recurrence
 *
 * At the heart of the method is the following basic recurrence.
 * If $l_{ij}^s$ represents the length of the shortest path from
 * $i$ to $j$ that can be attained in at most $2^s$ steps, then
 * $$
 *   l_{ij}^{s+1} = \min_k \{ l_{ik}^s + l_{kj}^s \}.
 * $$
 * That is, the shortest path of at most $2^{s+1}$ hops that connects
 * $i$ to $j$ consists of two segments of length at most $2^s$, one
 * from $i$ to $k$ and one from $k$ to $j$.  Compare this with the
 * following formula to compute the entries of the square of a
 * matrix $A$:
 * $$
 *   a_{ij}^2 = \sum_k a_{ik} a_{kj}.
 * $$
 * These two formulas are identical, save for the niggling detail that
 * the latter has addition and multiplication where the former has min
 * and addition.  But the basic pattern is the same, and all the
 * tricks we learned when discussing matrix multiplication apply -- or
 * at least, they apply in principle.  I'm actually going to be lazy
 * in the implementation of `square`, which computes one step of
 * this basic recurrence.  I'm not trying to do any clever blocking.
 * You may choose to be more clever in your assignment, but it is not
 * required.
 *
 * The return value for `square` is true if `l` and `lnew` are
 * identical, and false otherwise.
 */

int square_column(const int n, 
                 int* restrict l,
                 //int* restrict l_t,
                 int* restrict lnew,
                 const int j)
{
    int done = 1;
    for (int i = 0; i < n; ++i) {
        int lij = l[j*n+i];
        for (int k = 0; k < n; ++k) {
            int lik = l[k*n+i];
            int lkj = l[j*n+k];
            if (lik + lkj < lij) {
                lij = lik+lkj;
                done = 0;
            }
        }
        lnew[j*n+i] = lij;
    }
    return done;
}

int square_columns(int n,               // Number of nodes
                   int j_min,
                   int j_max,
                   int* restrict l,     // Partial distance at step s
                   int* restrict lnew)  // Partial distance at step s+1                          
{
    // Precompute the transpose for more efficient memory access
    // int l_t[n*n];
    // for (int j = 0; j < n; ++j) {
    //     for (int i = 0; i < n; ++i) {
    //         l_t[i*n + j] = l[j*n + i];
    //     }
    // }

    int done = 1;
    for (int j = 0; j < j_max - j_min; ++j) {
        int entry_done = square_column(n, l, lnew, j + j_min);
        done = done && entry_done;
    }
    return done;
}

/**
 *
 * The value $l_{ij}^0$ is almost the same as the $(i,j)$ entry of
 * the adjacency matrix, except for one thing: by convention, the
 * $(i,j)$ entry of the adjacency matrix is zero when there is no
 * edge between $i$ and $j$; but in this case, we want $l_{ij}^0$
 * to be "infinite".  It turns out that it is adequate to make
 * $l_{ij}^0$ longer than the longest possible shortest path; if
 * edges are unweighted, $n+1$ is a fine proxy for "infinite."
 * The functions `infinitize` and `deinfinitize` convert back 
 * and forth between the zero-for-no-edge and $n+1$-for-no-edge
 * conventions.
 */

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

/**
 *
 * Of course, any loop-free path in a graph with $n$ nodes can
 * at most pass through every node in the graph.  Therefore,
 * once $2^s \geq n$, the quantity $l_{ij}^s$ is actually
 * the length of the shortest path of any number of hops.  This means
 * we can compute the shortest path lengths for all pairs of nodes
 * in the graph by $\lceil \lg n \rceil$ repeated squaring operations.
 *
 * The `shortest_path` routine attempts to save a little bit of work
 * by only repeatedly squaring until two successive matrices are the
 * same (as indicated by the return value of the `square` routine).
 */

void shortest_paths(int num_p, int rank, int n, int* restrict l)
{
    if (rank == 0) {
        infinitize(n, l);
        for (int i = 0; i < n*n; i += n+1)
            l[i] = 0;
    }

    // Each processor gets the initial data
    MPI_Bcast(l, n*n, MPI_INT, 0, MPI_COMM_WORLD);

    // TODO: Fix this to handle sizes that aren't evenly divisible
    int num_columns = n / num_p;
    int j_min = num_columns*rank;
    int j_max = num_columns*(rank+1);

    int* restrict lnew = (int*) calloc(n*n, sizeof(int));
    
    int global_done = 1;
    do {
        int local_done = 1;

        local_done = square_columns(n, j_min, j_max, l, lnew);

        MPI_Allgather(lnew + j_min*n, n*num_columns, MPI_INT, l, n*num_columns, MPI_INT, MPI_COMM_WORLD);

        // Must reach consensus on completion
        MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    
    } while (!global_done);
    
    free(lnew);
    if (rank == 0) {
        deinfinitize(n, l);
    }
}

/**
 * # The random graph model
 *
 * Of course, we need to run the shortest path algorithm on something!
 * For the sake of keeping things interesting, let's use a simple random graph
 * model to generate the input data.  The $G(n,p)$ model simply includes each
 * possible edge with probability $p$, drops it otherwise -- doesn't get much
 * simpler than that.  We use a thread-safe version of the Mersenne twister
 * random number generator in lieu of coin flips.
 */

void gen_graph(int n, double p, int* l)
{
    struct mt19937p state;
    sgenrand(10302011UL, &state);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i)
            l[j*n+i] = (genrand(&state) < p);
        l[j*n+j] = 0;
    }
}

/**
 * # Result checks
 *
 * Simple tests are always useful when tuning code, so I have included
 * two of them.  Since this computation doesn't involve floating point
 * arithmetic, we should get bitwise identical results from run to
 * run, even if we do optimizations that change the associativity of
 * our computations.  The function `fletcher16` computes a simple
 * [simple checksum][wiki-fletcher] over the output of the
 * `shortest_paths` routine, which we can then use to quickly tell
 * whether something has gone wrong.  The `write_matrix` routine
 * actually writes out a text representation of the matrix, in case we
 * want to load it into MATLAB to compare results.
 *
 * [wiki-fletcher]: http://en.wikipedia.org/wiki/Fletcher's_checksum
 */

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

/**
 * # The `main` event
 */

const char* usage =
    "path.x -- Parallel all-pairs shortest path on a random graph\n"
    "Flags:\n"
    "  - n -- number of nodes (200)\n"
    "  - p -- probability of including edges (0.05)\n"
    "  - i -- file name where adjacency matrix should be stored (none)\n"
    "  - o -- file name where output matrix should be stored (none)\n";

int main(int argc, char** argv)
{
    int rank, num_p;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int n;              // Number of nodes
    double p;           // Edge probability
    const char* ifname; // Adjacency matrix file name
    const char* ofname; // Distance matrix file name

    if (rank == 0) {
        n    = 200;
        p = 0.05;
        ifname = NULL;
        ofname = NULL; 
    
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
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int* l = calloc(n*n, sizeof(int));
    double t0;
    if (rank == 0) {
        // Graph generation + output
        gen_graph(n, p, l);
        if (ifname)
            write_matrix(ifname, n, l);

        t0 = omp_get_wtime();
    }

    shortest_paths(num_p, rank, n, l);

    if (rank == 0) {
        double t1 = omp_get_wtime();

        // printf("== OpenMP with %d threads\n", omp_get_max_threads());
        printf("n:     %d\n", n);
        printf("p:     %g\n", p);
        printf("Time:  %g\n", t1-t0);
        printf("Check: %X\n", fletcher16(l, n*n));
    
        // Generate output file
        if (ofname)
            write_matrix(ofname, n, l);
    }

    // Clean up
    free(l);
    MPI_Finalize();
    return 0;
}
