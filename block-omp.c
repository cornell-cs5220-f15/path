#include <getopt.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "clear.h"
#include "copy.h"
#include "indexing.h"
#include "mem.h"
#include "mt19937p.h"
#include "transpose.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 128)
#endif

/*
 * A is M-by-K
 * B is K-by-N
 * C is M-by-N
 *
 * lda is the leading dimension of the matrix (the M of square_dgemm).
 */
void basic_dgemm(const int lda,
                 const int M,
                 const int N,
                 const int K,
                 const int* restrict A,
                 const int* restrict B,
                 int* restrict C,
                 int* restrict done) {
    int i, j, k;

    int A_[BLOCK_SIZE * BLOCK_SIZE];
    int B_[BLOCK_SIZE * BLOCK_SIZE];

    // transpose A into A_
    cm_transpose_into(A, lda, lda, M, K, A_, BLOCK_SIZE, BLOCK_SIZE);

    // copy B into B_
    cm_copy_into(B, lda, lda, K, N, B_, BLOCK_SIZE, BLOCK_SIZE);

    // Most of the time, M = N = K = BLOCK_SIZE. By replacing all bounds with
    // BLOCK_SIZE, the compiler can optimize these loops better.
    if (M == BLOCK_SIZE && N == BLOCK_SIZE && K == BLOCK_SIZE) {
        for (j = 0; j < BLOCK_SIZE; ++j) {
            for (i = 0; i < BLOCK_SIZE; ++i) {
                int cij = C[cm(lda, lda, i, j)];
                for (k = 0; k < BLOCK_SIZE; ++k) {
                    int sum = A_[rm(BLOCK_SIZE, BLOCK_SIZE, i, k)] +
                              B_[cm(BLOCK_SIZE, BLOCK_SIZE, k, j)];
                    if (sum < cij) {
                        cij = sum;
                        *done = 0;
                    }
                }
                C[cm(lda, lda, i, j)] = cij;
            }
        }
    } else {
        for (j = 0; j < N; ++j) {
            for (i = 0; i < M; ++i) {
                int cij = C[cm(lda, lda, i, j)];
                for (k = 0; k < K; ++k) {
                    int sum = A_[rm(BLOCK_SIZE, BLOCK_SIZE, i, k)] +
                              B_[cm(BLOCK_SIZE, BLOCK_SIZE, k, j)];
                    if (sum < cij) {
                        cij = sum;
                        *done = 0;
                    }
                }
                C[cm(lda, lda, i, j)] = cij;
            }
        }
    }
}

void do_block(const int lda,
              const int i,
              const int j,
              const int k,
              const int* restrict A,
              const int* restrict B,
              int* restrict C,
              int* restrict done) {
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                &A[cm(lda, lda, i, k)],
                &B[cm(lda, lda, k, j)],
                &C[cm(lda, lda, i, j)],
                done);
}

int square(int n,              // Number of nodes
           int* restrict l,    // Partial distance at step s
           int* restrict lnew) // Partial distance at step s+1
{
    const int n_blocks = n / BLOCK_SIZE + (n % BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    int done = 1;
    #pragma omp parallel for shared(l, lnew) reduction(&& : done)
    for (bi = 0; bi < n_blocks; ++bi) {
        for (bj = 0; bj < n_blocks; ++bj) {
            for (bk = 0; bk < n_blocks; ++bk) {
                const int i = bi * BLOCK_SIZE;
                const int j = bj * BLOCK_SIZE;
                const int k = bk * BLOCK_SIZE;
                do_block(n, i, j, k, l, l, lnew, &done);
            }
        }
    }
    return done;
}

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

void shortest_paths(int n, int* restrict l)
{
    // Generate l_{ij}^0 from adjacency matrix representation
    infinitize(n, l);
    for (int i = 0; i < n*n; i += n+1)
        l[i] = 0;

    // Repeated squaring until nothing changes
    int* restrict lnew = (int*) _mm_calloc(n*n, sizeof(int), 64);
    memcpy(lnew, l, n*n * sizeof(int));
    for (int done = 0; !done; ) {
        done = square(n, l, lnew);
        memcpy(l, lnew, n*n * sizeof(int));
    }
    _mm_free(lnew);
    deinfinitize(n, l);
}

int* gen_graph(int n, double p)
{
    int* l = _mm_calloc(n*n, sizeof(int), 64);
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

    // Graph generation + output
    int* l = gen_graph(n, p);
    if (ifname)
        write_matrix(ifname,  n, l);

    // Time the shortest paths code
    double t0 = omp_get_wtime();
    shortest_paths(n, l);
    double t1 = omp_get_wtime();

    // n, p, time, check, omp threads, mpi threads
    printf("%d, %g, %g, %X, %d, %d\n",
           n, p, t1-t0, fletcher16(l, n*n), omp_get_max_threads(), -1);

    // Generate output file
    if (ofname)
        write_matrix(ofname, n, l);

    // Clean up
    _mm_free(l);
    return 0;
}
