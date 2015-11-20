#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include "mt19937p.h"

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

#define NUM_THREADS 64
#define SQRT_THREADS 8
#define BLOCK_SIZE 64

int __attribute__((target(mic))) square(int n,               // Number of nodes
           int* restrict l,     // Partial distance at step s
           int* restrict lnew)  // Partial distance at step s+1
{
    int tid;

    int done = 1;
    #pragma omp parallel private(tid) shared(l, lnew, n) reduction(&& : done)
    {
        int nrows = n / SQRT_THREADS;
        int nblocks = nrows / BLOCK_SIZE;

        tid = omp_get_thread_num();
        int col = tid % SQRT_THREADS;
        int row = tid / SQRT_THREADS;
        int col_offset = col * nrows;
        int row_offset = row * nrows;

        __assume_aligned(l, 32);
        __assume_aligned(lnew, 32);
        for(int T = 0; T < SQRT_THREADS; T++) {
            for(int I = 0; I < nblocks; ++I) { // block row
                for(int J = 0; J < nblocks; ++J) { // block column
                    int C_offset = col_offset + J * BLOCK_SIZE +
                                   (row_offset + I * BLOCK_SIZE) * n;
                    for(int K = 0; K < nblocks; ++K) {
                        int A_offset = T * nrows + K * BLOCK_SIZE + (row_offset + I * BLOCK_SIZE) * n;
                        int B_offset = col_offset + J * BLOCK_SIZE + (T * nrows + K * BLOCK_SIZE) * n;

                        #pragma unroll
                        for (int i = 0; i < BLOCK_SIZE; ++i) {
                            #pragma unroll
                            for (int j = 0; j < BLOCK_SIZE; ++j) {
                                int a = l[A_offset + j + i * n];
                                #pragma unroll
                                for (int k = 0; k < BLOCK_SIZE; ++k) {
                                    int result = a + l[B_offset + k + j * n];

                                    int result_idx = k + C_offset + i * n;
                                    int c = lnew[result_idx];
                                    if(result < c){
                                        done = 0;
                                        lnew[result_idx] = result;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        int end_row = (row + 1) * nrows;
        int end_col = (col + 1) * nrows;

        #pragma omp barrier

        for (int j = row_offset; j < end_row; ++j) {
            for (int i = col_offset; i < end_col; ++i) {
                l[j*n+i] = lnew[j*n+i];
            }
        }
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

static inline void __attribute__((target(mic))) infinitize(int n, int* l)
{
    for (int i = 0; i < n*n; ++i)
        if (l[i] == 0)
            l[i] = n+1;
}

static inline void __attribute__((target(mic))) deinfinitize(int n, int* l)
{
    for (int i = 0; i < n*n; ++i)
        if (l[i] == n+1)
            l[i] = 0;
}

/**
 *
 * Of course, any loop-free path in a graph with $n$ nodes can
 * at most pass theough every node in the graph.  Therefore,
 * once $2^s \geq n$, the quantity $l_{ij}^s$ is actually
 * the length of the shortest path of any number of hops.  This means
 * we can compute the shortest path lengths for all pairs of nodes
 * in the graph by $\lceil \lg n \rceil$ repeated squaring operations.
 *
 * The `shortest_path` routine attempts to save a little bit of work
 * by only repeatedly squaring until two successive matrices are the
 * same (as indicated by the return value of the `square` routine).
 */

void __attribute__((target(mic))) shortest_paths(int n, int* restrict l)
{
    omp_set_num_threads(NUM_THREADS);
    // Generate l_{ij}^0 from adjacency matrix representation
    infinitize(n, l);
    for (int i = 0; i < n*n; i += n+1)
        l[i] = 0;

    // Repeated squaring until nothing changes
    int* restrict lnew = (int*) _mm_malloc(n*n*sizeof(int), 32);
    memcpy(lnew, l, n*n * sizeof(int));
    for (int done = 0; !done; ) {
        done = square(n, l, lnew);
    }
    _mm_free(lnew);
    deinfinitize(n, l);
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

int* gen_graph(int n, double p)
{
    int* l = (int*) _mm_malloc(n*n*sizeof(int), 32);
    struct mt19937p state;
    sgenrand(10302011UL, &state);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            l[j*n+i] = (genrand(&state) < p);
        }
        l[j*n+j] = 0;
    }
    return l;
}

int* gen_graphCopy(int copySize, int n, double p)
{
    int* l = (int*) _mm_malloc(copySize*copySize*sizeof(int), 32);
    struct mt19937p state;
    sgenrand(10302011UL, &state);
    for (int j = 0; j < copySize; ++j) {
        for (int i = 0; i < copySize; ++i) {
            if(i < n && j < n)
                l[j*copySize+i] = (genrand(&state) < p);
            else
                l[j*copySize+i] = n+1;
        }
        l[j*copySize+j] = 0;
    }
    return l;
}

/**
 * # Result checks
 *
 * Simple tests are always useful when tuning code, so I have included
 * two of them.  Since this computation doesn't involve floating point
 * arithmetic, we should get bitwise identical results from run to
 * run, even if we do optimizations that change the associativity of
 * our computations.  The function `fletcher16` computes a simple
 * [simple checksum][wiki-fletcher].  over the output of the
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
    int n    = 512;            // Number of nodes
    double p = 0.05;           // Edge probability
    const char* ifname = NULL; // Adjacency matrix file name
    const char* ofname = NULL; // Distance matrix file name
    int mic = 0;

    // omp_set_num_threads(NUM_THREADS);

    // Option processing
    extern char* optarg;
    const char* optstring = "hn:d:p:o:i:m:";
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
        case 'm': mic = atoi(optarg); break;
        }
    }

    int nBlocks  = (int) ceil((float) n / (float) SQRT_THREADS / (float) BLOCK_SIZE);
    int copySize = nBlocks * SQRT_THREADS * BLOCK_SIZE;

    // Graph generation + output
    int* lCopy = gen_graphCopy(copySize, n, p);
    int* l     = gen_graph(n, p);
    if (ifname)
        write_matrix(ifname,  n, l);

    // Time the shortest paths code
    double t0 = omp_get_wtime();
    if(mic) {
        #pragma offload target(mic) inout( lCopy : length(copySize*copySize)) \
        in(copySize)
        {
            shortest_paths(copySize, lCopy);
        }
    } else {
        shortest_paths(copySize, lCopy);
    }
    double t1 = omp_get_wtime();
    double t0_copy = omp_get_wtime();
    int i, j;
    if(copySize != n) {
        for (j = 0; j < n; ++j) {
            for (i = 0; i < n; ++i) {
                l[j*n+i] = lCopy[j * copySize + i];
            }
        }
    }
    else {
        l = lCopy;
    }
    double t1_copy = omp_get_wtime();

    printf("== OpenMP with %d threads\n", NUM_THREADS);
    printf("n:         %d\n", n);
    printf("p:         %g\n", p);
    printf("Time:      %g\n", t1-t0);
    printf("Copy time: %g\n", t1_copy-t0_copy);
    printf("Check:     %X\n", fletcher16(l, n*n));

    // Generate output file
    if (ofname)
        write_matrix(ofname, copySize, lCopy);

    // Clean up
    _mm_free(l);
    _mm_free(lCopy);
    return 0;
}
