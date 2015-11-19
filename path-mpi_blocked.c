#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include "mt19937p.h"

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
#endif
#ifndef ALIGNED_SIZE
#define ALIGNED_SIZE ((int) 64)
#endif

int square_dgemm(const int M, const int N, const int *A, const int *B, int *C);
void copy_optimize(const int M, const int m_blocks, const int N, const int n_blocks, int* A, int* cp);
void copy_back(const int M, const int m_blocks, const int N, const int n_blocks, int* A, int* cp);
int do_block(const int* restrict A_block, const int* restrict B_block, int* C_block);

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

int square_stripe(int n,               // Number of nodes
           int* restrict l,     // Partial distance at step s
           int* restrict lnew,  // Partial distance at step s+1
           int myid,            // The index of the processor
           int ncolumns)           // Number of columns to compute
{
    int done = 1;
    int end = min(n, (myid + 1) * ncolumns);
    int lij, lik, lkj;

    int* restrict lst = (int*) calloc(n*ncolumns, sizeof(int));
    memcpy(lst, l+n*myid*ncolumns,n*ncolumns*sizeof(int));
    done = square_dgemm(n, ncolumns, l, lst, lnew);

  return done;
}

int square_dgemm(const int M, const int N, const int *A, const int *B, int *C)
{
    int done = 1, done_part;
    const int m_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    const int n_blocks = N / BLOCK_SIZE + (N%BLOCK_SIZE? 1 : 0);
    const int Mc = BLOCK_SIZE * m_blocks;
    const int Nc = BLOCK_SIZE * n_blocks;
    int* A_cp = (int*) _mm_malloc(Mc*Mc*sizeof(int), ALIGNED_SIZE);
    int* B_cp = (int*) _mm_malloc(Mc*Nc*sizeof(int), ALIGNED_SIZE);
    int* C_cp = (int*) _mm_malloc(Mc*Nc*sizeof(int), ALIGNED_SIZE);
    copy_optimize(M, m_blocks, M, m_blocks, A, A_cp);
    copy_optimize(M, m_blocks, N, n_blocks, B, B_cp);
    memcpy(C_cp, B_cp, Mc*Nc*sizeof(int));
    int *A_block, *B_block, *C_block;
    int bi, bj, bk;
    for (bi = 0; bi < m_blocks; ++bi) {
        for (bj = 0; bj < n_blocks; ++bj) {
            for (bk = 0; bk < m_blocks; ++bk) {
                A_block = A_cp + BLOCK_SIZE * BLOCK_SIZE * (bi * m_blocks + bk);
                B_block = B_cp + BLOCK_SIZE * BLOCK_SIZE * (bk * n_blocks + bj);
                C_block = C_cp + BLOCK_SIZE * BLOCK_SIZE * (bi * n_blocks + bj);
                done_part = do_block(A_block, B_block, C_block);
                done = min(done, done_part);
            }
        }
    }
    copy_back(M, m_blocks, N, n_blocks, C_cp, C);
    _mm_free(A_cp);
    _mm_free(B_cp);
    _mm_free(C_cp);
    return done;
}

void copy_optimize(const int M, const int m_blocks, const int N, const int n_blocks, int* A, int* cp )
{
    int Mc = BLOCK_SIZE * m_blocks;
    int Nc = BLOCK_SIZE * n_blocks;
    int i, j, I, J, ii, jj, id;
    memset(cp, 0x3f3f3f3f, Mc * Nc * sizeof( int));
    for (j = 0; j < N; ++j)
    {
        J = j / BLOCK_SIZE;
        jj = j % BLOCK_SIZE;
        for (i = 0; i < M; ++i)
        {
            I = i / BLOCK_SIZE;
            ii = i % BLOCK_SIZE;
            id = (I * n_blocks + J) * BLOCK_SIZE * BLOCK_SIZE + ii * BLOCK_SIZE + jj;
            cp[id] = *(A++);
        }
    }
    return;
}

void copy_back(const int M, const int m_blocks, const int N, const int n_blocks, int* A, int* cp)
{
    int i, j, I, J, ii, jj, id;
    for (j = 0; j < N; ++j)
    {
        J = j / BLOCK_SIZE;
        jj = j % BLOCK_SIZE;
        for (i = 0; i < M; ++i)
        {
            I = i / BLOCK_SIZE;
            ii = i % BLOCK_SIZE;
            id = (I * n_blocks + J) * BLOCK_SIZE * BLOCK_SIZE + ii * BLOCK_SIZE + jj;
            *(cp++) = A[id];
        }
    }
}

int do_block(const int* restrict A_block, const int* restrict B_block, int* C_block)
{
    int i, j, k;
    int *Ci, *Bk;
    int Aik, Bkj;
    int done = 1;
    __assume_aligned( A_block, ALIGNED_SIZE );
    __assume_aligned( B_block, ALIGNED_SIZE );
    __assume_aligned( C_block, ALIGNED_SIZE );

    for (i = 0; i < BLOCK_SIZE; ++i)
    {
        Ci = C_block + i * BLOCK_SIZE;
        __assume_aligned(Ci, ALIGNED_SIZE);
        for (k = 0; k < BLOCK_SIZE; ++k)
        {
            Aik = A_block[i * BLOCK_SIZE + k];
            Bk = B_block + k * BLOCK_SIZE;
            __assume_aligned(Bk, ALIGNED_SIZE);
            for (j = 0; j < BLOCK_SIZE; ++j)
            {
                //#pragma vector always
                Bkj = Bk[j];
                if (Aik + Bkj < Ci[j])
                {
                    done = 0;
                    Ci[j] = Aik + Bkj;
                }
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

static inline void infinitize(int n, int* l)
{
    for (int i = 0; i < n*n; ++i)
        if (l[i] == 0 && (i % (n + 1)) != 0)
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

/*void shortest_paths(int n, int* restrict l)
{
    // Generate l_{ij}^0 from adjacency matrix representation
    // infinitize(n, l);
    for (int i = 0; i < n*n; i += n+1)
        l[i] = 0;

    // Repeated squaring until nothing changes
    int* restrict lnew = (int*) calloc(n*n, sizeof(int));
    memcpy(lnew, l, n*n * sizeof(int));
    for (int done = 0; !done; ) {
        done = square(n, l, lnew);
        memcpy(l, lnew, n*n * sizeof(int));
    }
    free(lnew);
    deinfinitize(n, l);
}*/

void shortest_paths_mpi(int n, int* restrict l, int myid, int nproc)
{
    int count = 0;
    int done = 0, done_part;
    int ncolumns = n / nproc + (n % nproc? 1 : 0 );
    // Generate l_{ij}^0 from adjacency matrix representation
    if (myid == 0)
    {
    infinitize(n, l);
    /*for (int i = 0; i < n*n; i += n+1)
        l[i] = 0;*/
    }

    // Repeated squaring until nothing changes
    int* restrict lnew = (int*) calloc(n*ncolumns, sizeof(int));
    while (!done) {
        MPI_Bcast(l, n*n, MPI_INT, 0, MPI_COMM_WORLD);
        done_part = square_stripe(n, l, lnew, myid, ncolumns);
        MPI_Gather(lnew, ncolumns*n, MPI_INT, l, ncolumns*n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Reduce(&done_part, &done, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
        MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    free(lnew);
    if (myid == 0)
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
    int n    = 200;            // Number of nodes
    double p = 0.05;           // Edge probability
    const char* ifname = NULL; // Adjacency matrix file name
    const char* ofname = NULL; // Distance matrix file name

    // Option processing
    extern char* optarg;
    const char* optstring = "hn:d:p:o:i:";
    int c;
    int* l;
    double t0, t1;
    int nproc, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
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

    l = calloc(n*n, sizeof(int));
    if (myid == 0)
    {
    // Graph generation + output
    l = gen_graph(n, p);
    if (ifname)
        write_matrix(ifname,  n, l);

    // Time the shortest paths code
    t0 = MPI_Wtime();
    }

    shortest_paths_mpi(n, l, myid, nproc);

    if (myid == 0)
    {
    t1 = MPI_Wtime();

    printf("== MPI with %d threads with blocked graph\n", nproc);
    printf("n:     %d\n", n);
    printf("p:     %g\n", p);
    printf("Time:  %g\n", t1-t0);
    printf("Check: %X\n", fletcher16(l, n*n));

    // Generate output file
    if (ofname)
        write_matrix(ofname, n, l);

    // Clean up
    free(l);
    }

    MPI_Finalize();
    return 0;
}
