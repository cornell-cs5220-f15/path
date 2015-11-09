#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "mt19937p.h"
#include <mpi.h>
#include <immintrin.h>

//ldoc on

/**
 *  Copying a column using global (n*n) indexing. This is done so as to
 *  remove data dependencies from the k-loop and making it the outermost
 *  loop.
 */

inline __attribute__((always_inline))
void col_copy(int *col, int *l, int index, int n)
{
  for (int m = 0; m < n; ++m)
    col[m] = l[n * index + m];
}

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

inline __attribute__((always_inline))
int square(int nproc, int rank, int vector_nwords,
           int n, int nlocal,
           int* restrict lproc, int* restrict col_k)
{
  int done      = 1;
  int col_shift = n / nproc;

  // Moving across columns of the global matrix
  for (int k = 0; k < n; ++k) {

    // Based on k value calculate which processor owns the column,
    // And broadcast it to every processor.
    int root = k / col_shift;
    if (root >= nproc)
      root--;

    if (rank == root) {
      int kproc = k % (col_shift);
      col_copy(col_k, lproc, kproc, n);
    }

    MPI_Bcast(col_k, n, MPI_INT, root, MPI_COMM_WORLD);

    // Moving across local columns in lproc

    int num_wide_ops = (n + vector_nwords - 1) / vector_nwords;

    for (int j = 0; j < nlocal; ++j) {

      // Broadcast column element to be added to row elements to compare
      // against current shortest path.
      int     lkj     = lproc[j*n+k];
      __m256i lkj_vec = _mm256_set1_epi32(lkj);

      // Vectorize inner loop across row elements of both output elements
      // (i.e., current shortest path) and input elements in k-th
      // iteration.
      for (int i = 0; i < num_wide_ops; ++i) {

        int *lij_vec_addr = lproc + (j * n) + (i * vector_nwords);
        int *lik_vec_addr = col_k + (i * vector_nwords);

        __m256i lij_vec = _mm256_load_si256((__m256i*)lij_vec_addr);
        __m256i lik_vec = _mm256_load_si256((__m256i*)lik_vec_addr);

        // Calculate sum of shortest paths between (i,k) and (k,j)
        __m256i sum_vec = _mm256_add_epi32(lik_vec, lkj_vec);

        // Create a mask based on a vector-wide comparison between the
        // current shortest path and the sum of the shortest paths
        // between (i,k) and (k,j). For each element in the vector, this
        // comparison will return a mask of all 1s (0xffffffff) if the
        // sum is the new shortest path, otherwise will return a mask of
        // all 0s (0x00000000).
        __m256i mask_vec = _mm256_cmpgt_epi32(lij_vec, sum_vec);
        __m256i zero_vec = _mm256_setzero_si256();

        // If the comparison for any of the elements was true (i.e., a
        // new shortest path was found), then we are not done yet. This
        // vector operation ANDs the mask and the NOT of a zero vector,
        // so a vector of all 1s, and returns 1 if the result is all 0s,
        // otherwise returns 0.
        if (!_mm256_testc_si256(mask_vec, zero_vec))
          done = 0;

        // Use the mask to zero out the elements in both the sum and
        // current shortest path vectors that are not the shortest path,
        // then add them together to get the new shortest path vector
        // which is stored back to memory.
        sum_vec = _mm256_and_si256(mask_vec, sum_vec);
        lij_vec = _mm256_andnot_si256(mask_vec, lij_vec);
        lij_vec = _mm256_add_epi32(sum_vec, lij_vec);

        _mm256_store_si256((__m256i*)lij_vec_addr, lij_vec);
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

void shortest_paths(int nproc, int rank, int vector_nwords,
                    int n, int* restrict l)
{
  // Calculate number of columns to process for this rank and the offset
  // to the block from the global grid this rank is responsible for. We
  // use a 1D domain decomposition where each rank computes the output
  // for a non-overlapping "stripe" in the global grid.
  int nlocal = n / nproc;
  int offset = rank * nlocal * n;

  if (rank == (nproc - 1))
    nlocal += n % nproc;
  int nelements = n * nlocal;

  // Number of elements to send and displacement from global grid array
  // from which the master rank should send data to each rank.
  int *scounts = (int*) calloc(nproc, sizeof(int));
  int *displs  = (int*) calloc(nproc, sizeof(int));

  for (int i = 0; i < nproc; ++i)
    displs[i]  = offset;

  MPI_Gather(&nelements, 1, MPI_INT, scounts, 1,
             MPI_INT, 0, MPI_COMM_WORLD);

  // Generate l_{ij}^0 from adjacency matrix representation
  if (rank == 0)  {
    infinitize(n, l);
    for (int i = 0; i < n*n; i += n+1)
      l[i] = 0;
  }

  // Allocate per-rank local buffers to hold blocks and columns
  // transferred from other ranks. Align local buffers to cache lines
  // (64B) to allow vector loads/stores. Always allocate memory at the
  // granularity of vector accesses to avoid masked vector loads/stores.

  int lproc_nwords = ((n * nlocal) + vector_nwords - 1) / vector_nwords;
  int col_k_nwords = (n + vector_nwords - 1) / vector_nwords;

  int* restrict lproc = _mm_malloc(lproc_nwords * sizeof(int), 32);
  int* restrict col_k = _mm_malloc(col_k_nwords * sizeof(int), 32);

  // Master rank sends blocks of global grid to corresponding ranks
  MPI_Scatterv(l, scounts, displs, MPI_INT, lproc,
               nelements, MPI_INT, 0, MPI_COMM_WORLD);

  // Repeated squaring until nothing changes
  for (int done = 0; !done; )
    done = square(nproc, rank, vector_nwords, n, nlocal, lproc, col_k);

  // Master rank receives blocks from each rank to pack final results
  MPI_Gatherv(lproc, nelements, MPI_INT, l, scounts, displs,
              MPI_INT, 0, MPI_COMM_WORLD);

  // Clean up local buffers
  free(scounts);
  free(displs);
  free(lproc);
  free(col_k);

  if (rank == 0)
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
  // MPI parallelization
  MPI_Init(&argc, &argv);

  int    n = 200;            // Number of nodes
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

  // Set number of threads in system and retrieve thread id (rank)

  int nproc, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Graph generation + output. Only the master rank generates the graph
  // for now. Look into parallelizing graph generation across ranks.
  int* l;
  if (rank == 0) {
    l = gen_graph(n, p);
    if (ifname)
      write_matrix(ifname, n, l);
  }

  // Set the number of 32b words in the vector width
  int vector_nwords = 8;

  // Time the shortest paths code
  double t0 = MPI_Wtime();
  shortest_paths(nproc, rank, vector_nwords, n, l);
  double t1 = MPI_Wtime();

  // Execution statistics. Only master rank prints this out.
  if (rank == 0)  {
    printf("== MPI with %d processors\n", nproc);
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
