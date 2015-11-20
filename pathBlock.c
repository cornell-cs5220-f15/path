#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include "mt19937p.h"

#ifndef SMALL_BLOCK_SIZE
#define SMALL_BLOCK_SIZE ((int) 128)
#endif
#ifndef BYTE_ALIGNMENT
#define BYTE_ALIGNMENT ((int) 64)
#endif


int basic_square(const int *restrict l1, const int *restrict l2, int *restrict l3){
  int done=1;
  for (int j = 0; j < SMALL_BLOCK_SIZE; ++j) {
    for (int i = 0; i < SMALL_BLOCK_SIZE; ++i) {
      int lij = l3[j*SMALL_BLOCK_SIZE+i];
      for (int k = 0; k < SMALL_BLOCK_SIZE; ++k) {
        int lik = l1[k*SMALL_BLOCK_SIZE+i];
        int lkj = l2[j*SMALL_BLOCK_SIZE+k];
        if (lik + lkj < lij) {
          lij = lik+lkj;
          done = 0;
        }
      }
      l3[j*SMALL_BLOCK_SIZE+i] = lij;
    }
  }
  return done;
}


int do_block(const int n, 
             const int *restrict l, int *restrict lnew, 
             const int *restrict l1, const int *restrict l2, int *restrict l3,
             int i, int j, int k){
  for(int kk=0; kk<SMALL_BLOCK_SIZE; ++kk){
    memcpy((void *) (l1 + (kk * SMALL_BLOCK_SIZE)), (const void *) (l + i + (k + kk)*n), SMALL_BLOCK_SIZE * sizeof(int));
  }
  for(int jj=0; jj<SMALL_BLOCK_SIZE; ++jj){
    memcpy((void *) (l2 + (jj * SMALL_BLOCK_SIZE)), (const void *) (l + k + (j + jj)*n), SMALL_BLOCK_SIZE * sizeof(int));
    memcpy((void *) (l3 + (jj * SMALL_BLOCK_SIZE)), (const void *) (lnew + i + (j + jj)*n), SMALL_BLOCK_SIZE * sizeof(int));
  }
  
  int done = basic_square(l1, l2, l3);
  
  if(!done)
    for(int jj=0; jj<SMALL_BLOCK_SIZE; ++jj){
      memcpy((void *) (lnew + i + (j + jj)*n), (const void *) (l3 + (jj * SMALL_BLOCK_SIZE)), SMALL_BLOCK_SIZE * sizeof(int));
    }
    
  return done;
}



int square(int n,               // Number of nodes
           int* restrict l,     // Partial distance at step s
           int* restrict lnew)  // Partial distance at step s+1
{
  int done = 1;
  const int *l1 = (int *) _mm_malloc(SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE * sizeof(int), BYTE_ALIGNMENT);
  const int *l2 = (int *) _mm_malloc(SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE * sizeof(int), BYTE_ALIGNMENT);
  int *l3 = (int *) _mm_malloc(SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE * sizeof(int), BYTE_ALIGNMENT);
  const int n_small_blocks = n / SMALL_BLOCK_SIZE;
  
  //#pragma omp parallel for shared(l, lnew) reduction(&& : done)
  for (int bj = 0; bj < n_small_blocks; ++bj){
    const int j = bj * SMALL_BLOCK_SIZE;
    for (int bi = 0; bi < n_small_blocks; ++bi){
      const int i = bi * SMALL_BLOCK_SIZE;
        for (int bk = 0; bk < n_small_blocks; ++bk){
          const int k = bk * SMALL_BLOCK_SIZE;
          if(!do_block(n, l, lnew, l1, l2, l3, i, j, k)) done = 0;
        }
    }
  }
  
  _mm_free((void *) l1);
  _mm_free((void *) l2);
  _mm_free((void *) l3);   
  
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

void shortest_paths(int n, int* restrict l)
{
    // Generate l_{ij}^0 from adjacency matrix representation
    infinitize(n, l);
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

    // Graph generation + output
    int* l = gen_graph(n, p);
    if (ifname)
        write_matrix(ifname,  n, l);

    // Time the shortest paths code
    double t0 = omp_get_wtime();
    shortest_paths(n, l);
    double t1 = omp_get_wtime();

    printf("== OpenMP with %d threads\n", omp_get_max_threads());
    printf("n:     %d\n", n);
    printf("p:     %g\n", p);
    printf("Time:  %g\n", t1-t0);
    printf("Check: %X\n", fletcher16(l, n*n));

    // Generate output file
    if (ofname)
        write_matrix(ofname, n, l);

    // Clean up
    free(l);
    return 0;
}
