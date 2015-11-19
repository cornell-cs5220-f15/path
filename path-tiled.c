#include <getopt.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "mt19937p.h"

#ifndef ALIGN_BOUND
  #define ALIGN_BOUND 64
#endif

#ifndef UNROLL_ROW
  #define UNROLL_ROW 4
#endif

#ifndef UNROLL_COL
  #define UNROLL_COL 4
#endif

#ifndef UNROLL_STEP
  #define UNROLL_STEP 1
#endif

#ifndef BLOCK_SIZE
  #define BLOCK_SIZE 64
#endif

#define min(a, b) ((a) < (b) ? (a) : (b))

static inline
void * _mm_calloc(int size, int alignment) {
  void * memory = _mm_malloc(size, alignment);
  if(memory != NULL) memset(memory, 0, size);
  return memory;
}

static inline
void dump_array(int n, int ** restrict l, const char * fname) {
  FILE * fp = fopen(fname, "w+");

  if (fp == NULL) {
    fprintf(stderr, "Could not open output file: %s\n", fname);
    exit(-1);
  }

  __assume_aligned(l, ALIGN_BOUND);
  for (int row = 0; row < n; ++row) {
    fprintf(fp, "%d ", l[row][0:n]);
    fprintf(fp, "\n");
  }

  fclose(fp);
}

static inline
void infinitize(const int n, int ** restrict l) {
  const int PATH_VAL = n + 1;

  __assume_aligned(l, ALIGN_BOUND);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      if (l[row][col] == 0) l[row][col] = PATH_VAL;
    }

    l[row][row] = 0;
  }
}

static inline
void deinfinitize(const int n, int ** restrict l) {
  const int PATH_VAL = n + 1;

  __assume_aligned(l, ALIGN_BOUND);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (l[i][j] == PATH_VAL) l[i][j] = 0;
    }
  }
}

static inline
int ** alloc_array(const int n) {
  int ** l = _mm_calloc(n * sizeof(int *), ALIGN_BOUND);

  __assume_aligned(l, ALIGN_BOUND);
  for (int i = 0; i < n; ++i) l[i] = _mm_calloc(n * sizeof(int), ALIGN_BOUND);

  return l;
}

static inline
void destroy_array(const int n, int ** restrict l) {
  __assume_aligned(l, ALIGN_BOUND);
  for (int i = 0; i < n; ++i) _mm_free(l[i]);
  _mm_free(l);
}

static inline
void flatten_array(const int n, int ** restrict l, int * restrict flat_l) {
  __assume_aligned(l, ALIGN_BOUND);
  __assume_aligned(flat_l, ALIGN_BOUND);

  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      flat_l[col * n + row] = l[row][col];
    }
  }
}

static inline
void copy_from_array(const int n, int ** restrict src, int ** restrict dest) {
  __assume_aligned(src, ALIGN_BOUND);
  __assume_aligned(dest, ALIGN_BOUND);

  for (int row = 0; row < n; ++row) memcpy(dest[row], src[row], n * sizeof(int));
}

void fwi_abc(const int n, int ** restrict a, int ** restrict b, int ** restrict c) {
  __assume_aligned(a, ALIGN_BOUND);
  __assume_aligned(b, ALIGN_BOUND);
  __assume_aligned(c, ALIGN_BOUND);

  for (int row = 0; row < n; row += UNROLL_ROW) {
    for (int col = 0; col < n; col += UNROLL_COL) {
      for (int step = 0; step < n; step += UNROLL_STEP) {
        for (int step_p = step; step_p < step + UNROLL_STEP; ++step_p) {
          for (int row_p = row; row_p < row + UNROLL_ROW; ++row_p) {
            for (int col_p = col; col_p < col + UNROLL_COL; ++col_p) {
              c[row_p][col_p] = min(c[row_p][col_p], a[row_p][step_p] + b[step_p][col_p]);
            }
          }
        }
      }
    }
  }
}

void fwi(const int n, int ** restrict a, int ** restrict b, int ** restrict c) {
  __assume_aligned(a, ALIGN_BOUND);
  __assume_aligned(b, ALIGN_BOUND);
  __assume_aligned(c, ALIGN_BOUND);

  for (int step = 0; step < n; ++step) {
    for (int row = 0; row < n; row += UNROLL_ROW) {
      for (int col = 0; col < n; col += UNROLL_COL) {
        for (int row_p = row; row_p < row + UNROLL_ROW; ++row_p) {
          for (int col_p = col; col_p < col + UNROLL_COL; ++col_p) {
            c[row_p][col_p] = min(c[row_p][col_p], a[row_p][step] + b[step][col_p]);
          }
        }
      }
    }
  }
}

void fwt(const int n, int ** restrict a, int ** restrict b, int ** restrict c) {
  const int num_blocks = n / BLOCK_SIZE;

  for (int b = 0; b < num_blocks; ++b) {
    // TODO(Kenneth): Malloc blocks
    fwi(BLOCK_SIZE, C_kk, C_kk, C_kk);

    for (int j = 0; j < num_blocks; ++j) {
      if (j != k) fwi(BLOCK_SIZE, C_kk, C_kj, C_kj);
    }

    for (int i = 0; i < num_blocks; ++i) {
      if (i != k) fwi(BLOCK_SIZE, C_ik, C_kk, C_ik);
    }

    for (int i = 0; i < num_blocks; ++i) {
      for (int j = 0; j < num_blocks; ++j) {
        if (i != k && j != k) fwi_abc(BLOCK_SIZE, C_ik, C_kj, C_ij);
      }
    }
  }
}

void shortest_paths(const int n, int ** restrict l) {
  fwi(n, l, l, l);
}

int ** gen_graph(const int n, const double p) {
  struct mt19937p state;
  sgenrand(10302011UL, &state);

  int ** l = alloc_array(n);

  // Seed with coin flips
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) l[i][j] = (genrand(&state) < p);
    l[j][j] = 0;
  }

  return l;
}

int fletcher16(const int n, int ** restrict l) {
  int sum1 = 0;
  int sum2 = 0;
  int count = n * n;

  // Flatten array
  int * data = _mm_malloc(count * sizeof(int), ALIGN_BOUND);
  flatten_array(n, l, data);

  __assume_aligned(l, ALIGN_BOUND);
  for(int index = 0; index < count; ++index) {
    sum1 = (sum1 + data[index]) % 255;
    sum2 = (sum2 + sum1) % 255;
  }

  // Cleanup array as early as possible
  _mm_free(data);

  return (sum2 << 8) | sum1;
}

int main(int argc, char ** argv) {
  const char * USAGE =
    "path.x -- Parallel all-pairs shortest path on a random graph\n"
    "Flags:\n"
    "  - n -- number of nodes (200)\n"
    "  - p -- probability of including edges (0.05)\n"
    "  - i -- file name where adjacency matrix should be stored (none)\n"
    "  - o -- file name where output matrix should be stored (none)\n";;

  int n = 200;                 // Number of nodes
  double p = 0.05;             // Edge Probability
  const char * ifname = NULL;  // Adjacency matrix file name
  const char * ofname = NULL;  // Distance matrix file name

  // Option processing
  extern char * optarg;
  const char * optstring = "hn:d:p:o:i:";
  int c;

  while ((c = getopt(argc, argv, optstring)) != -1) {
    switch (c) {
      case 'h':
        fprintf(stderr, "%s", USAGE);
        return -1;
      case 'n': n = atoi(optarg); break;
      case 'p': p = atof(optarg); break;
      case 'o': ofname = optarg;  break;
      case 'i': ifname = optarg;  break;
    }
  }

  // Graph generation + output
  int ** l = gen_graph(n, p);

  if (ifname) dump_array(n, l, ifname);

  // Populate adjacency matrix with path lengths
  infinitize(n, l);

  // Time the shortest paths code
  const double t0 = omp_get_wtime();
  shortest_paths(n, l);
  const double t1 = omp_get_wtime();

  // Clamp 'infinite' values in the output to zero.
  deinfinitize(n, l);

  printf("== OpenMP with %d threads\n", omp_get_max_threads());
  printf("n:     %d\n", n);
  printf("p:     %g\n", p);
  printf("Time:  %g\n", t1-t0);
  printf("Check: %X\n", fletcher16(n, l));

  if (ofname) dump_array(n, l, ofname);

  // Clean up
  destroy_array(n, l);

  return 0;
}
