#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include "mt19937p.h"

#include <xmmintrin.h>// _mm_malloc
#include <string.h>   // memset

#ifdef __MIC__
    #define BYTE_ALIGN 64
    #define width_size ((int) 512)
    #define height_size ((int) 128)
#else
    #define BYTE_ALIGN 32
    #define width_size ((int) 1024)
    #define height_size ((int) 64)
#endif

#ifdef __INTEL_COMPILER
    #define DEF_ALIGN(x) __declspec(align((x)))
    #define USE_ALIGN(var, align) __assume_aligned((var), (align));

    // // frankly, this article is very unclear as to what needs to happen, lets just see what happens?
    // //     https://software.intel.com/en-us/articles/memcpy-memset-optimization-and-control
    // #define memcpy _intel_fast_memcpy
    // #define memset _intel_fast_memset
#else // GCC
    #define DEF_ALIGN(x) __attribute__ ((aligned((x))))
    #define USE_ALIGN(var, align) ((void)0) /* __builtin_assume_align is unreliabale... */
#endif

#ifndef CHAR_BIT
    #define CHAR_BIT 8
#endif

// how many threads?
int n_threads = 1;

// global timing constants
double square_avg = 0.0;
long   num_square = 0;

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

int square(int n,                 // Number of nodes
           int * restrict l,      // Partial distance at step s
           int * restrict lnew,    // Partial distance at step s+1
           int n_width,            // Width (x direction) of block
           int n_height) {       // Height (y direction) of block

    USE_ALIGN(l,    BYTE_ALIGN);
    USE_ALIGN(lnew, BYTE_ALIGN);

    int done = 1;
    #pragma omp parallel for       \
            num_threads(n_threads) \
            shared(l, lnew)        \
            reduction(&& : done)   \
            // Major Blocks
            for(int J=0; J<n_height; ++J) {
              for(int K=0; K<n_height; ++K) {
                for(int I=0; I<n_width; ++I){
                  // Calculate ending indices for the set of blocks
                  int j_end = ((J+1)*height_size < n? height_size : (n-(J*height_size)));
                  int k_end = ((K+1)*height_size < n? height_size : (n-(K*height_size)));
                  int i_end = ((I+1)*width_size  < n? width_size  : (n-(I*width_size)));
                  int j_init = J*height_size*n;
                  int kn_init = K*height_size*n;
                  int k_init = K*height_size;
                  int i_init = I*width_size;
                  
                  // Minor Blocks
                  for (int j = 0; j < j_end; ++j) {
                    int jn = j_init+j*n;
                    for (int k = 0; k < k_end; ++k) {
                      int kn = kn_init+k*n;
                      int lkj = l[jn+k_init+k];
                      for (int i = 0; i < i_end; ++i) {
                        int lij_ind = jn+i_init+i;
                        int lij = lnew[lij_ind];
                        int lik = l[kn+i_init+i];
                        
                        if (lik + lkj < lij) {
                          lij = lik+lkj;
                          lnew[lij_ind] = lij;
                          done = 0;
                        }
                      }
                    }
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

static inline void infinitize(int n, int * restrict l) {
    USE_ALIGN(l, BYTE_ALIGN);

    for (int i = 0; i < n*n; ++i)
        if (l[i] == 0)
            l[i] = n+1;
}

static inline void deinfinitize(int n, int * restrict l) {
    USE_ALIGN(l, BYTE_ALIGN);

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

void shortest_paths(int n, int * restrict l)
{
    USE_ALIGN(l, BYTE_ALIGN);

    printf("-------------------------------------------\n");
    printf("Individual Squares:\n");

    // Generate l_{ij}^0 from adjacency matrix representation
    double to_inf_start = omp_get_wtime();
    infinitize(n, l);
    double to_inf_stop  = omp_get_wtime();

    for (int i = 0; i < n*n; i += n+1)
        l[i] = 0;

    // Repeated squaring until nothing changes
    // int* restrict lnew = (int*) calloc(n*n, sizeof(int));
    size_t num_bytes = n*n*sizeof(int);
    DEF_ALIGN(BYTE_ALIGN) int * restrict lnew = (int *)_mm_malloc(num_bytes, BYTE_ALIGN);
    // memset(lnew, 0, num_bytes);
    USE_ALIGN(lnew, BYTE_ALIGN);
    memcpy(lnew, l, n*n * sizeof(int));
  
    const int n_width = n/width_size + (n%width_size? 1 : 0);
    const int n_height = n/ height_size + (n%height_size? 1 : 0);
    for (int done = 0; !done; ) {

        double square_start = omp_get_wtime();
        done = square(n, l, lnew, n_width, n_height);
        double square_stop  = omp_get_wtime();

        //
        // tmp just to make sure avg is legit
        /////////////////////////////////////////
        printf(" -- %.16g\n", square_stop - square_start);
        /////////////////////////////////////////
        //
        //

        // don't want to printf in here.
        square_avg += square_stop-square_start;
        num_square++;

        memcpy(l, lnew, n*n * sizeof(int));
    }
    _mm_free(lnew);

    double de_inf_start = omp_get_wtime();
    deinfinitize(n, l);
    double de_inf_stop  = omp_get_wtime();

    printf("To Inf: %.16g\n", to_inf_stop - to_inf_start);
    printf("De Inf: %.16g\n", de_inf_stop - de_inf_start);
    printf("-------------------------------------------\n");
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

int * gen_graph(int n, double p)
{
    // int* l = calloc(n*n, sizeof(int));
    // 'calloc'
    size_t num_bytes = n*n*sizeof(int);
    DEF_ALIGN(BYTE_ALIGN) int *l = (int *)_mm_malloc(num_bytes, BYTE_ALIGN);
    memset(l, 0, num_bytes);

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
    "  - o -- file name where output matrix should be stored (none)\n"
    "  - t -- number of threads (default := omp_max_threads)\n";

int main(int argc, char** argv)
{
    double overall_start = omp_get_wtime();

    int n     = 200;                   // Number of nodes
    double p  = 0.05;                  // Edge probability
    n_threads = omp_get_max_threads(); // Number of threads to use with OMP
    const char* ifname = NULL;         // Adjacency matrix file name
    const char* ofname = NULL;         // Distance matrix file name

    // Option processing
    extern char* optarg;
    const char* optstring = "hn:d:p:o:i:t:";
    int c;
    // int num_threads <-- defined at top for global scope
    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
        case 'h':
            fprintf(stderr, "%s", usage);
            return -1;
        case 'n': n         = atoi(optarg); break;
        case 'p': p         = atof(optarg); break;
        case 'o': ofname    = optarg;       break;
        case 'i': ifname    = optarg;       break;
        case 't': n_threads = atoi(optarg); break;
        }
    }

    // Graph generation + output
    double gen_start = omp_get_wtime();
    int* l = gen_graph(n, p);
    double gen_stop  = omp_get_wtime();

    if (ifname)
        write_matrix(ifname,  n, l);

    // Time the shortest paths code
    double t0 = omp_get_wtime();
    shortest_paths(n, l);
    double t1 = omp_get_wtime();


    // Generate output file
    if (ofname)
        write_matrix(ofname, n, l);

    // check solution
    int check = fletcher16(l, n*n);

    // Clean up
    _mm_free(l);

    double overall_stop = omp_get_wtime();
    
    printf("== OpenMP with %d threads\n", n_threads);
    printf("n:     %d\n", n);
    printf("p:     %g\n", p);
    printf("Check: %X\n", check);
    printf("-------------------------------------------\n");
    printf("Timings:\n");
    printf("Shortest Paths:  %.16g\n", t1 - t0);
    printf("Square Average:  %.16g\n", square_avg / ((double)num_square));
    printf("Overall:         %.16g\n", overall_stop - overall_start);

    return 0;
}
