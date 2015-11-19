#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include "mt19937p.h"

//
#include <immintrin.h>
#include <x86intrin.h>

#ifndef L3_BLOCK_SIZE
#define L3_BLOCK_SIZE ((int) 720)
#define L3 L3_BLOCK_SIZE
#endif

#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE ((int) 180)
#define L2 L2_BLOCK_SIZE
#endif

//L1 fits into registers, we have a fast kernel for that part
//right now the same as L1 blocks. This many blocking levels tend to slow
//the computation (don't know exactly why.)
#ifndef L1_BLOCK_SIZE
#define L1_BLOCK_SIZE ((int) 36)
#define L1 L1_BLOCK_SIZE
#endif


#define N2 L3/L2
#define N1 L2/L1

//Fast kernel

inline int MMult4by4VRegAC(const int* restrict B, const int* restrict A, int* restrict C)
{
    int a, b, c;
    int done = 1;
    for (int j = 0; j < L1; j++) {
        for (int i = 0; i < L1; i++) {
            c = C[j*L1 + i];
            for (int k = 0; k < L1; k++) {
                b = B[i*L1 + k];
                a = A[j*L1 + k];
                if(a + b < c) {
                    c = a + b;
                    done = 0;
                }
            }
            C[j*L1 + i] = c;
        }
    }
    return done;
}


// read (L1 x L1) matrix from A(i,j) into block_A (assumes col major A)
// block_A is row major (historical reasons)
void read_to_contiguous(const int M, const int* restrict A, int* restrict block_A,
                        const int i, const int j) {
    // guard against matrix edge case
    const int mBound = (j+L1 > M? M-j : L1);
    const int nBound = (i+L1 > M? M-i : L1);
    
    // offset is index of upper left corner of desired block within A
    const int offset = i + M*j;
    int m, n;
    for (n = 0; n < nBound; ++n) {
        for (m = 0; m < mBound; ++m) {
            block_A[m + L1*n] = A[offset + m*M + n];
        }
        while (m < L1){
            block_A[m + L1*n] = 0.0;
            m++;
        }
    }
    while (n < L1) {
        for (m = 0; m < L1; m++)
            block_A[m + L1*n] = 0.0;
        n++;
    }
}

// write block_C into C(i,j)
void write_from_contiguousC(const int M, int* restrict C,
                            const int* restrict block_C,
                            const int i, const int j) {
    // guard against matrix edge case
    // printf("%d %d\n", i, j);
    const int mBound = (i+L1 > M? M-i : L1); // rows
    const int nBound = (j+L1 > M? M-j : L1); // cols
    
    int m, n;
    const int offset = i + M * j;
    for (n = 0; n < nBound; ++n) {
        for (m = 0; m < mBound; ++m) {
            C[offset + m + M * n] = block_C[m * L1 + n];
        }
    }
}

//Assumes L3 is integer mutliple of L2 and L2 is integer multiple of L1
void to_contiguous3lvlBlock(const int M,
                            const int* restrict A,
                            int* restrict Ak,
                            const int* restrict B,
                            int* restrict Bk,
                            const int* restrict C,
                            int* restrict Ck) {
    int ind_Ak = 0, ind_Bk = 0, ind_Ck = 0;
    const int n3 = M / L3_BLOCK_SIZE + (M%L3_BLOCK_SIZE? 1 : 0);
    for(int i = 0; i < n3; i++){
        const int row_i = i * L3;
        const int n2_i = (i == n3-1 && (M % L3) ? (M % L3) / L2 + (M%L2? 1 : 0) : N2);
        for (int j = 0; j < n3; j++){
            const int col_j = j * L3;
            const int n2_j = (j == n3-1 && (M % L3) ? (M % L3) / L2 + (M%L2? 1 : 0) : N2);
            for(int q = 0; q < n2_i; q++){
                const int row_q = row_i + q * L2;
                const int n1_q = (i == n3-1 && q == n2_i-1 && (M%L2) ? (M % L2) / L1 + (M%L1? 1 : 0) : N1);
                for (int s = 0; s < n2_j; s++){
                    const int col_s = col_j + s * L2;
                    const int n1_s = (j == n3-1 && s == n2_j-1 && (M%L2)? (M % L2) / L1 + (M%L1? 1 : 0) : N1);
                    for (int m = 0; m < n1_q; m++){
                        for (int n = 0; n < n1_s; n++){
                            read_to_contiguous(M, A, Ak + ind_Ak, row_q + m * L1, col_s + n * L1);
                            ind_Ak += L1*L1;
                            read_to_contiguous(M, B, Bk + ind_Bk, col_s + n * L1, row_q + m * L1);
                            ind_Bk += L1*L1;
                            read_to_contiguous(M, C, Ck + ind_Ck, row_q + m * L1, col_s + n * L1);
                            ind_Ck += L1*L1;
                        }
                    }
                }
            }
        }
        
    }
}

void from_contiguous3lvlBlock(const int M,
                              int* restrict C,
                              const int* restrict Ck){
    int ind_Ck = 0;
    const int n3 = M / L3 + (M%L3? 1 : 0);
    for(int i = 0; i < n3; i++){
        const int row_i = i * L3;
        const int n2_i = (i == n3-1  && (M % L3) ? (M % L3) / L2 + (M%L2? 1 : 0) : N2);
        for (int j = 0; j < n3; j++){
            const int col_j = j * L3;
            const int n2_j = (j == n3-1  && (M % L3) ? (M % L3) / L2 + (M%L2? 1 : 0) : N2);
            for(int q = 0; q < n2_i; q++){
                const int row_q = row_i + q * L2;
                const int n1_q = (i == n3-1 && q == n2_i-1  && (M % L2) ? (M % L2) / L1 + (M%L1? 1 : 0) : N1);
                for (int s = 0; s < n2_j; s++){
                    const int col_s = col_j + s * L2;
                    const int n1_s = (j == n3-1 && s == n2_j - 1  && (M % L2) ? (M % L2) / L1 + (M%L1? 1 : 0) : N1);
                    for (int m = 0; m < n1_q; m++){
                        for (int n = 0; n < n1_s; n++){
                            write_from_contiguousC(M, C, Ck + ind_Ck, row_q + m * L1, col_s + n * L1);
                            ind_Ck += L1*L1;
                        }
                    }
                }
            }
        }
        
    }
}

int do_block_L2(const int* restrict Ak, const int* restrict Bk, int* restrict Ck,
                 const int M, const int N, const int K) {
    int done = 1;
    
    int bi, bj, bk;
    const int ni = M / L1 + (M%L1? 1 : 0); // number of blocks in M rows
    const int nj = N / L1 + (N%L1? 1 : 0); // number of blocks in N cols
    const int nk = K / L1 + (K%L1? 1 : 0); // number of blocks in K rows
    
    int iA, iC, j, jB, k;
    int MM, NN, KK;
    
    int sizeOfBlock_C = L1 * L1;
    int sizeOfBlock_B = L1 * L1;
    for (bi = 0; bi < ni-1; bi++) {
        iA = bi * K * L1;
        iC = bi * N * L1;
        for (bj = 0; bj < nj-1; bj++) {
            j = bj * sizeOfBlock_C;
            jB = bj * K * L1;
            const int ind_Ck = iC+j;
            for (bk = 0; bk < nk; bk++) {
                const int ind_Ak = iA + sizeOfBlock_C * bk;
                const int ind_Bk = jB + sizeOfBlock_B * bk;
                done = done * MMult4by4VRegAC(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck);
            }
        }
        j = bj * sizeOfBlock_C;
        jB = bj * K * L1;
        sizeOfBlock_B = ( N%L1 ? L1 * (N%L1) : L1 * L1);
        const int ind_Ck = iC+j;
        for (bk = 0; bk < nk; bk++) {
            const int ind_Ak = iA + sizeOfBlock_C * bk;
            const int ind_Bk = jB + sizeOfBlock_B * bk;
            done = done * MMult4by4VRegAC(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck);
        }
    }
    sizeOfBlock_C = M%L1 ? L1 * (M%L1) : L1 * L1;
    iA = bi * K * L1;
    iC = bi * N * L1;
    for (bj = 0; bj < nj-1; bj++) {
        j = bj * sizeOfBlock_C;
        jB = bj * K * L1;
        const int ind_Ck = iC+j;
        for (bk = 0; bk < nk; bk++) {
            const int ind_Ak = iA + sizeOfBlock_C * bk;
            const int ind_Bk = jB + sizeOfBlock_B * bk;
            done = done * MMult4by4VRegAC(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck);
        }
    }
    j = bj * sizeOfBlock_C;
    jB = bj * K * L1;
    sizeOfBlock_B = ( N%L1 ? L1 * (N%L1) : L1 * L1);
    const int ind_Ck = iC+j;
    for (bk = 0; bk < nk; bk++) {
        const int ind_Ak = iA + sizeOfBlock_C * bk;
        const int ind_Bk = jB + sizeOfBlock_B * bk;
        done = done * MMult4by4VRegAC(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck);
    }
    
    return done;
}

int do_block_L3(const int* restrict Ak, const int* restrict Bk, int* restrict Ck,
                 const int M, const int N, const int K) {
    int done = 1;
    
    int bi, bj, bk;
    const int ni = M / L2 + (M%L2? 1 : 0); // number of blocks in M rows
    const int nj = N / L2 + (N%L2? 1 : 0); // number of blocks in N cols
    const int nk = K / L2 + (K%L2? 1 : 0); // number of blocks in K rows
    
    int ind_Ak, ind_Bk, ind_Ck;
    int iA, iC, j, jB, k;
    int MM, NN, KK;
    
    for (bi = 0; bi < ni; bi++) {
        iA = bi * K * L2;
        iC = bi * N * L2;
        int sizeOfBlock_C = ( bi == ni-1 && M%L2 ? L2 * (M%L2) : L2 * L2);
        MM = ( bi == ni-1 && M%L2) ? M%L2 : L2;
        for (bj = 0; bj < nj; bj++) {
            NN = ( bj == nj-1 && N%L2) ? N%L2 : L2;
            j = bj * sizeOfBlock_C;
            jB = bj * K * L2;
            int sizeOfBlock_B = ( bj == nj-1 && N%L2 ? L2 * (N%L2) : L2 * L2);
            const int ind_Ck = iC+j;
            for (bk = 0; bk < nk; bk++) {
                const int ind_Ak = iA + sizeOfBlock_C * bk;
                const int ind_Bk = jB + sizeOfBlock_B * bk;
                KK = (bk == nk-1 && K%L2) ? K%L2 : L2;
                
                //A is MM by KK
                //B is KK by NN
                //C is MM by NN
                // printf("\t%d %d %d\n", ind_Ak, ind_Bk, ind_Ck);
                // printf("\t%d %d %d\n",ni, nj, nk);
                
                done = done * do_block_L2(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck, MM, NN, KK);
            }
        }
    }
    
    return done;
}

int square_dgemm(const int M, const int* restrict A, const int* restrict B, int* restrict C) {
    int done = 1;
    
    const int N = (M / L1 + (M%L1? 1 : 0)) * L1; // new size after padding with zeros
    
    //Begin copying to new layout
    int* Ak = _mm_malloc(N*N*sizeof(int), 32);
    int* Bk = _mm_malloc(N*N*sizeof(int), 32);
    int* Ck = _mm_malloc(N*N*sizeof(int), 32);
    
    to_contiguous3lvlBlock(M, A, Ak, B, Bk, C, Ck);
    //End of copying
    
    const int n3 = N / L3 + (N%L3? 1 : 0); //Number of L3 block in one dimension
    
    int i, j, k;
    int MM, NN, KK; //The sizes of the blocks (at the edges they may be rectangular)
    
    for (int bi = 0; bi < n3; bi++){
        i = bi * N * L3;
        int sizeOfBlock_C = ( bi == n3-1 && N%L3 ? L3 * (N%L3) : L3 * L3);
        MM = ( bi == n3-1 && N%L3) ? N%L3 : L3;
        for (int bj = 0; bj < n3; bj++){
            NN = ( bj == n3-1 && N%L3) ? N%L3 : L3;
            j = bj * sizeOfBlock_C;
            int jB = bj * N * L3;
            int sizeOfBlock_B = ( bj == n3-1 && N%L3 ? L3 * (N%L3) : L3 * L3);
            const int ind_Ck = i+j;
            for (int bk = 0; bk < n3; bk++){
                int ind_Ak = i + sizeOfBlock_C * bk;
                int ind_Bk = jB + sizeOfBlock_B * bk;
                KK = (bk == n3-1 && N%L3) ? N%L3 : L3;
                
                done = done * do_block_L3(Ak + ind_Ak, Bk + ind_Bk, Ck + ind_Ck, MM, NN, KK);
            }
        }
    }
    _mm_free(Ak);
    _mm_free(Bk);
    from_contiguous3lvlBlock(M, C, Ck);
    _mm_free(Ck);
    
    return done;
}
//

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

int square(int n,               // Number of nodes
           int* restrict l,     // Partial distance at step s
           int* restrict lnew)  // Partial distance at step s+1
{
    int done = 1;

    /**
	// copy optimization
	int* restrict temp = malloc(n * n * sizeof(int));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			temp[i * n + j] = l[j * n + i];
		}
	}
    
    for (int j = 0; j < n; ++j) {
    	for (int i = 0; i < n; ++i) {
            int lij = lnew[j*n+i];
            for (int k = 0; k < n; ++k) {
                //int lik = l[k*n+i];
				int lik = temp[i*n+k];
                int lkj = l[j*n+k];
                if (lik + lkj < lij) {
                    lij = lik+lkj;
                    done = 0;
                }
            }
            lnew[j*n+i] = lij;
        }
    }
    
    return done;
    **/
    
    //
    // copy optimzation embedded in square_dgemm()
    return square_dgemm(n,l,l,lnew);
    //
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
    //sgenrand(omp_get_wtime(), &state);
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

    printf("== Serial implementation\n");
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
