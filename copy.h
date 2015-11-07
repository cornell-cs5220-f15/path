#ifndef __COPY_H__
#define __COPY_H__

#include <stdlib.h>
#include "indexing.h"

inline int *cm_copy(const int *A,
                    const int lN, const int lM,
                    const int N,  const int M) {
    int *transposed = (int *)malloc(N * M * sizeof(int));
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            transposed[cm(N, M, n, m)] = A[cm(lN, lM, n, m)];
        }
    }
    return transposed;
}

inline int *rm_copy(const int *A,
                    const int lN, const int lM,
                    const int N,  const int M) {
    int *transposed = (int *)malloc(N * M * sizeof(int));
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            transposed[rm(N, M, n, m)] = A[rm(lN, lM, n, m)];
        }
    }
    return transposed;
}

inline void cm_copy_into(const int *A,
                         const int lN, const int lM,
                         const int N,  const int M,
                         int *A_,
                         const int A_N, const int A_M) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            A_[cm(A_N, A_M, n, m)] = A[cm(lN, lM, n, m)];
        }
    }
}

#endif // __COPY_H__
