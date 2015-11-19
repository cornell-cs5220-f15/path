#ifndef __CLEAR_H__
#define __CLEAR_H__

#include <string.h>
#include "indexing.h"

inline void cm_clear_but(double *A,
                         const int lN, const int lM,
                         const int N,  const int M) {
    for (int n = 0; n < M; ++n) {
        memset(&A[rm(lM, lN, n, N)], 0, (lN - N) * sizeof(double));
    }

    memset(&A[rm(lM, lN, M, 0)], 0, (lM - M) * lN * sizeof(double));
}

inline void rm_clear_but(double *A,
                         const int lN, const int lM,
                         const int N,  const int M) {
    for (int n = 0; n < N; ++n) {
        memset(&A[rm(lN, lM, n, M)], 0, (lM - M) * sizeof(double));
    }

    memset(&A[rm(lN, lM, N, 0)], 0, (lN - N) * lM * sizeof(double));
}

#endif // __CLEAR_H__
