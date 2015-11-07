#ifndef __INDEXING_H__
#define __INDEXING_H__

// column-major
inline int cm(const int N, const int M, const int i, const int j) {
    return i + N*j;
}

// row-major
inline int rm(const int N, const int M, const int i, const int j) {
    return i*M + j;
}

#endif // __INDEXING_H__
