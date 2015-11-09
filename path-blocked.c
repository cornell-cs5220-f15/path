#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdbool.h>
#include <omp.h>
#include <getopt.h>
#include <immintrin.h>
#include "mt19937p.h"

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

static inline void infinitize(int n, int * l)
{
    for (int i = 0; i < n*n; ++i) if (l[i] == 0) l[i] = n+1;
}

static inline void deinfinitize(int n, int * l)
{
    for (int i = 0; i < n*n; ++i) if (l[i] == n+1) l[i] = 0;
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

void shortest_paths(int n, int * restrict l)
{
    // Generate l_{ij}^0 from adjacency matrix representation
    infinitize(n, l);

    for (int i = 0; i < n * n; ++i) {
      if (i % (n + 1) == 0) l[i] = 0;
    }

    // Repeated squaring until nothing changes
    int * restrict lnew = (int *) _mm_malloc(n*n*sizeof(int), 64);
    memcpy(lnew, l, n*n*sizeof(int));

    bool done = false;
    while (!done) {
      done = true;

      #pragma omp parallel for shared(l, lnew) reduction(&& : done)
      for (int j = 0; j < n; ++j) {
        const int jn = j * n;
        for (int k = 0; k < n; ++k) {
          const int lkj = l[jn+k];

          #pragma vector aligned
          for (int i = 0; i < n; ++i) {
            const int lijOriginal = lnew[jn+i];
            const int lik = l[k*n+i];
            const int lijTest = lik + lkj;

            if (lijTest < lijOriginal) {
              lnew[jn+i] = lijTest;
              done = false;
            }
          }
        }
      }

      memcpy(l, lnew, n*n*sizeof(int));
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

int* gen_graph(const int n, const double p)
{
  int * l = _mm_malloc(n*n*sizeof(int), 64);
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
 * # The `main` event
 */

const char* usage =
    "path.x -- Parallel all-pairs shortest path on a random graph\n"
    "Flags:\n"
    "  - n -- number of nodes (200)\n"
    "  - p -- probability of including edges (0.05)\n";

int main(int argc, char** argv)
{
    int n    = 200;            // Number of nodes
    double p = 0.05;           // Edge probability

    // Option processing
    extern char* optarg;
    const char* optstring = "hn:d:p:o:i:";
    int c;

    while ((c = getopt(argc, argv, optstring)) != -1) {
      switch (c) {
        case 'h': fprintf(stderr, "%s", usage); return -1;
        case 'n': n = atoi(optarg); break;
        case 'p': p = atof(optarg); break;
      }
    }

    // Graph generation + output
    int * l = gen_graph(n, p);

    // Time the shortest paths code
    double t0 = omp_get_wtime();
    shortest_paths(n, l);
    double t1 = omp_get_wtime();

    printf("== OpenMP with %d threads\n", omp_get_max_threads());
    printf("n:     %d\n", n);
    printf("p:     %g\n", p);
    printf("Time:  %g\n", t1-t0);
    printf("Check: %X\n", fletcher16(l, n*n));

    // Clean up
    _mm_free(l);

    return 0;
}
