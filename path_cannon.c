#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <omp.h>
#include "mt19937p.h"
#include <mpi.h>

#define BLOCK_SIZE 64
#define TAG_DIST_A 11
#define TAG_DIST_B 12
#define TAG_DIST_C 13
#define TAG_RESULT 20
#define TAG_DONE 31
#define TAG_ALL_DONE 32

//ldoc on

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
 * # The basic recurrence
 *
 * At the heart of the method is the following basic recurrence.
 * If $l_{ij}^s$ represents the length of the shortest path from
 * $i$ to $j$ that can be attained in at most $2^s$ steps, then
 * $$
 *   l_{ij}^{s+1} = \min_k \{ l_{ik}^s + l_{kj}^2 \}.
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

int basic_square(int n, const int * restrict A, const int * restrict B, int * restrict C) {
    //__assume_aligned(A, 64);
    //__assume_aligned(B, 64);
    //__assume_aligned(C, 64);
    
    int oi, oj, ok;
    int ta, tb, tc;
    int done = 1;
    
    for (int j = 0; j < n; ++j) {
        oj = j * n;
        for (int k = 0; k < n; ++k) {
            ok = k * n;
            tb = B[oj+k];
            for (int i = 0; i < n; ++i) {
                if (A[ok+i] + tb < C[oj+i]) {
                    C[oj+i] = A[ok+i] + tb;
                    done = 0;
                }
            }
        }
    }
    
    return done;
}

int square( int worldsize, int procs,
            int n,               // Number of nodes
            int * restrict l,     // Partial distance at step s
            int * restrict lnew)  // Partial distance at step s+1
{
    printf("f16%X\n", fletcher16(l, n*n));
    // Size logistics
    int done = 1;
    int blocks = procs;
    int blocksize = n / procs + (n % procs ? 1 : 0);
    int totalblocks = blocks * blocks;
    int totalblocksize = blocksize * blocksize;
    
    // Copied l matrix
    int * CL __attribute__((aligned(64))) =
        (int *) malloc(totalblocks * totalblocksize * sizeof(int));
    
    // Copy Blocking
    int copyoffset = 0;
    for (int bi = 0; bi < blocks; ++bi) {
        for (int bj = 0; bj < blocks; ++bj) {
            int oi = bi * blocksize;
            int oj = bj * blocksize;
            copyoffset = (bi + bj * blocks) * totalblocksize;
            for (int j = 0; j < blocksize; ++j) {
                for (int i = 0; i < blocksize; ++i) {
                    int offset = (oi + i) + (oj + j) * n;
                    // Check bounds
                    if (oi + i < n && oj + j < n) {
                        CL[copyoffset] = l[offset];
                    }
                    else {
                        CL[copyoffset] = n + 1;
                    }
                    copyoffset++;
                }
            }
        }
    }
    
    // Send A and B blocks in appropriate order
    printf("r0send\n");
    for (int b = 0; b < totalblocks; ++b) {
        int target = b + 1;
        
        int i = b / blocks;
        int j = b % blocks;
        int bi, bj;
        
        // Block C
        bi = i;
        bj = j;
        
        printf("r0sendc%d\n", target);
        MPI_Send(CL + (bi + bj * blocks) * totalblocksize,
            totalblocksize, MPI_INT, target, TAG_DIST_C, MPI_COMM_WORLD);
        
        // Block A
        bi = i;
        bj = (j - i + blocks) % blocks;
        
        printf("r0senda%d\n", target);
        MPI_Send(CL + (bi + bj * blocks) * totalblocksize,
            totalblocksize, MPI_INT, target, TAG_DIST_A, MPI_COMM_WORLD);
        
        // Block B
        bi = (i - j + blocks) % blocks;
        bj = j;
        
        printf("r0sendb%d\n", target);
        MPI_Send(CL + (bi + bj * blocks) * totalblocksize,
            totalblocksize, MPI_INT, target, TAG_DIST_B, MPI_COMM_WORLD);
    }
    
    // Receive results
    int * res __attribute__((aligned(64))) =
        (int *) malloc(totalblocksize * sizeof(int));
    int * isdone = (int *) malloc(sizeof(int));
    for (int b = 0; b < totalblocks; ++b) {
        int target = b + 1;
        MPI_Recv(res, totalblocksize, MPI_INT, target,
            TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(isdone, 1, MPI_INT, target,
            TAG_DONE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        done = done && isdone[0];
        
        // Copy back
        int bi = b / blocks;
        int bj = b % blocks;
        
        int oi = bi * blocksize;
        int oj = bj * blocksize;
        copyoffset = 0;
        for (int j = 0; j < blocksize; ++j) {
            for (int i = 0; i < blocksize; ++i) {
                int offset = (oi + i) + (oj + j) * n;
                if (oi + i < n && oj + j < n) {
                    lnew[offset] = res[copyoffset];
                }
                copyoffset++;
            }
        }
    }
    free(res);
    
    // Broadcast done
    printf("r0bc%d\n", done);
    //done = 1; // ===== DEBUG =====
    isdone[0] = done;
    MPI_Bcast(isdone, 1, MPI_INT, 0, MPI_COMM_WORLD);
    free(isdone);
    printf("f16%X\n", fletcher16(lnew, n*n));
    
    return done;
}

// Wait for cannon's algorithm until done
void cannon(int worldsize, int rank, int procs, int n)
{
    printf("\nr%d\n", rank);
    // Size logistics
    int done = 1;
    int blocks = procs;
    int blocksize = n / procs + (n % procs ? 1 : 0);
    int totalblocks = blocks * blocks;
    int totalblocksize = blocksize * blocksize;
    
    // Where to send blocks
    int current = rank - 1;
    int bi = current % blocks;
    int bj = current / blocks;
    int atarget = ((bi - 1 + blocks) % blocks) + bj * blocks + 1;
    int btarget = bi + ((bj - 1 + blocks) % blocks) * blocks + 1;
    printf("r%dat%d\n", rank, atarget);
    printf("r%dbt%d\n", rank, btarget);
    
    int * A __attribute__((aligned(64))) =
        (int *) malloc(totalblocksize * sizeof(int));
    int * B __attribute__((aligned(64))) =
        (int *) malloc(totalblocksize * sizeof(int));
    int * C __attribute__((aligned(64))) =
        (int *) malloc(totalblocksize * sizeof(int));
    int * isdone = (int *) malloc(sizeof(int));
    isdone[0] = 0;
    
    while (!isdone[0]) {
        int done = 1;
        
        // Receive C
        printf("r%dwc\n", rank);
        MPI_Recv(C, totalblocksize, MPI_INT, MPI_ANY_SOURCE,
            TAG_DIST_C, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("r%drc\n", rank);
        
        // The total number of swaps is constant
        for (int s = 0; s < blocks; ++s) {
            // Receive new A and Bs
            // Block A
            printf("r%dwa\n", rank);
            MPI_Recv(A, totalblocksize, MPI_INT, MPI_ANY_SOURCE,
                TAG_DIST_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("r%dra\n", rank);
            printf("r%dfa16%X\n", rank, fletcher16(A, totalblocksize));
            // Block B
            printf("r%dwb\n", rank);
            MPI_Recv(B, totalblocksize, MPI_INT, MPI_ANY_SOURCE,
                TAG_DIST_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("r%drb\n", rank);
            printf("r%dfb16%X\n", rank, fletcher16(B, totalblocksize));
            
            // For now just do the basic square
            int ndone = basic_square(blocksize, A, B, C);
            done = ndone && done;
            
            printf("r%dfc16%X\n", rank, fletcher16(C, totalblocksize));
            
            printf("r%ddone\n", rank);
            
            if (s != blocks - 1) {
                // Send A and B to next processors
                // Block A
                MPI_Send(A, totalblocksize, MPI_INT, atarget,
                    TAG_DIST_A, MPI_COMM_WORLD);
                // Block B
                MPI_Send(B, totalblocksize, MPI_INT, btarget,
                    TAG_DIST_B, MPI_COMM_WORLD);
                printf("r%dsend\n", rank);
            }
        }
        
        // Send result
        MPI_Send(C, totalblocksize, MPI_INT, 0,
            TAG_RESULT, MPI_COMM_WORLD);
        isdone[0] = done;
        MPI_Send(isdone, 1, MPI_INT, 0,
            TAG_DONE, MPI_COMM_WORLD);
        printf("r%dres\n", rank);
        
        // Check if done
        MPI_Bcast(isdone, 1, MPI_INT, 0, MPI_COMM_WORLD);
        printf("r%dalldone\n", rank);
        //isdone[0] = 1; // ===== DEBUG =====
    }
    
    // Completed
    printf("r%dcomplete\n", rank);
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
void shortest_paths(int worldsize, int rank, int procs, int n, int* restrict l)
{
    printf("\nr%d\n", rank);
    // Generate l_{ij}^0 from adjacency matrix representation
    infinitize(n, l);
    for (int i = 0; i < n*n; i += n+1)
        l[i] = 0;

    // Repeated squaring until nothing changes
    int * restrict lnew = (int*) calloc(n*n, sizeof(int));
    memcpy(lnew, l, n*n * sizeof(int));
    int count = 0;
    int done = 0;
    while (!done) {
        done = square(worldsize, procs, n, l, lnew);
        memcpy(l, lnew, n*n * sizeof(int));
    }
    printf("%d\n", count);
    free(lnew);
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

int* gen_graph(int n, double p, unsigned long int s)
{
    int* l = calloc(n*n, sizeof(int));
    struct mt19937p state;
    sgenrand(s, &state);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i)
            l[j*n+i] = (genrand(&state) < p);
        l[j*n+j] = 0;
    }
    return l;
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
    "  - s -- seed to generate matrix entries\n"
    "  - i -- file name where adjacency matrix should be stored (none)\n"
    "  - o -- file name where output matrix should be stored (none)\n";

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int worldsize = 0;
    int procs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    procs = sqrt(worldsize);
    if (procs * procs + 1 != worldsize) {
        if (rank == 0) {
            printf("Error: 1 more than a square number of processors is required.");
        }
        MPI_Finalize();
        return 0;
    }
    
    int n    = 200;            // Number of nodes
    double p = 0.05;           // Edge probability
    unsigned long int s = 10302011UL;        // Random number generator seed
    const char* ifname = NULL; // Adjacency matrix file name
    const char* ofname = NULL; // Distance matrix file name

    // Option processing done by all processors
    extern char* optarg;
    const char* optstring = "hn:d:p:s:o:i:";
    int c;
    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
        case 'h':
            fprintf(stderr, "%s", usage);
            return -1;
        case 'n': n = atoi(optarg); break;
        case 'p': p = atof(optarg); break;
        case 's': s = strtoul(optarg, NULL, 10); break;
        case 'o': ofname = optarg;  break;
        case 'i': ifname = optarg;  break;
        }
    }
    
    // Only master node generates the full graph
    if (rank == 0) {
        // Graph generation + output
        int* l = gen_graph(n, p, s);
        if (ifname)
            write_matrix(ifname,  n, l);

        // Time the shortest paths code
        double t0 = omp_get_wtime();
        shortest_paths(worldsize, rank, procs, n, l);
        double t1 = omp_get_wtime();

        printf("== OpenMP with %d threads\n", omp_get_max_threads());
        printf("n:     %d\n", n);
        printf("p:     %g\n", p);
        printf("Time:  %g\n", t1-t0);
        printf("Check: %X\n", fletcher16(l, n*n));

        // Generate output file
        if (ofname) {
            write_matrix(ofname, n, l);
        }
        
        // Clean up
        free(l);
    }
    else { // Wait for cannon's algorithm to start
        cannon(worldsize, rank, procs, n);
    }
    
    MPI_Finalize();
    return 0;
}
