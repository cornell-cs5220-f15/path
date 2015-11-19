#define _GNU_SOURCE
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "mt19937p.h"
#include <mpi.h>

#define SETUP_STAGE 55
#define INIT_STAGE 66
#define PARTITION_STAGE 77
#define ROTATION_STAGE 88
#define END_STAGE 99

// Print matrix assuming row-major format
void write_matrix_process(const char* fname, int cols, int rows, int* a, int rank)
{
    int namelen = strlen(fname);
    char rank_str[3];
    sprintf(rank_str, "%d", rank);
    char* proc_fname;
    asprintf(&proc_fname,"%s%s", fname, rank_str);

    FILE* fp = fopen(proc_fname, "w+");
    if (fp == NULL) {
        fprintf(stderr, "Could not open output file: %s\n", fname);
        exit(-1);
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            fprintf(fp, "%d ", a[i*cols+j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

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
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int lij = lnew[j*n+i];
            for (int k = 0; k < n; ++k) {
                int lik = l[k*n+i];
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
}

// Multiply the i^th block of rows and the j^th block of columns and store in the 
// (i,j) square block in the column
//
// Input:
//  n        -- Length of a row or a column
//  side     -- Number of rows or columns in the rectangular block
//  block_no -- Row block number (i)
//  lr       -- Row block
//  lc       -- Column block
//
// Output:
//  Done flag if (i,j) block does not get updated
int mult(int n, int side, int block_no, int* restrict lr, int* restrict lc, int* restrict lc_new) {
    int done = 1;
    for (int j = 0; j < side; ++j) {
        for (int i = 0; i < side; ++i) {
            int lij = lc_new[j*n + i + (block_no*side)];
            for (int k = 0; k < n; ++k) {
                int lik = lr[i*n+k];
                int lkj = lc[j*n+k];
                if (lik + lkj < lij) {
                    lij = lik+lkj;
                    done = 0;
                }
            }
            lc_new[j*n+i+(block_no*side)] = lij;
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

static inline void infinitize_rect(int n, int side, int* l)
{
    for (int i = 0; i < n*side; ++i)
        if (l[i] == 0)
            l[i] = n+1;
}

static inline void deinfinitize(int n, int* l)
{
    for (int i = 0; i < n*n; ++i)
        if (l[i] == n+1)
            l[i] = 0;
}

static inline void deinfinitize_rect(int n, int side, int* l)
{
    for (int i = 0; i < n*side; ++i)
        if (l[i] == n+1)
            l[i] = 0;
}

void row_unjumble(int n, int side, int* restrict row, int* restrict row_recv) {
    int recv_ind, col_slot;
    for (int i=0; i<side; i++) {
        for (int j=0; j<n; j++) {
            col_slot = j/side;
            recv_ind = col_slot*(side*side) + ((i*side) + (j - (col_slot * side)));
            row[i*n+j] = row_recv[recv_ind];
        }
    }
}

void shortest_paths(int n, int side, int rank, int size, int* restrict col, int* restrict row)
{
    int* row_chunk = (int*) malloc (sizeof(int) * side * side);
    int* row_recv = (int*) malloc (sizeof(int) * n * side);
    int proc_done = 0;
    int row_col_flag = 0;
    int done = 0;
    int* restrict col_new = (int*) calloc(n*side, sizeof(int));
    memcpy(col_new, col, n*side*sizeof(int));

    // Repeated squaring until nothing changes
    for (int done = 0; !done; ) {

        proc_done = 0;
        //For all rows
        for (int r=0; r<(n/side); r++) {
            //Make local row chunk in row major
            for (int i=0; i<side; i++)
                for (int j=0; j<side; j++)
                    row_chunk[j*side+i] = col[(i*n)+(r*side)+j];

            //Send your chunk of row and get row (Allgather)
            MPI_Allgather(row_chunk, (side*side), MPI_INT, row_recv, (side*side), MPI_INT, MPI_COMM_WORLD);
            row_unjumble (n, side, row, row_recv);

            //Complete row-col operation
            row_col_flag = mult(n, side, r, row, col, col_new);
            proc_done += row_col_flag;
        }
        proc_done = (proc_done == (n/side)); // Column chunk locally done if all rows squares are done
        memcpy(col, col_new, n*side*sizeof(int));

        //Reduce done flag across processors
        MPI_Allreduce (&proc_done, &done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        done = (done == size); // Stop if all processors are done.
    }

    free (col_new);
    free (row_chunk);
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

int* gen_graph_rect(int n, int m, int rank, double p)
{
    int* l = calloc(n*m, sizeof(int));
    struct mt19937p state;
    sgenrand(10302011UL, &state);
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i)
            l[j*n+i] = (genrand(&state) < p);
        l[j*n+j+(rank*m)] = 0;
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
    "  -n -- number of nodes (200)\n"
    "  -p -- probability of including edges (0.05)\n"
    "  -i -- file name where adjacency matrix should be stored (none)\n"
    "  -o -- file name where output matrix should be stored (none)\n";

int main(int argc, char** argv)
{

    const char* ofname = NULL; // Distance matrix file name
    int rank, size, block_side;
    MPI_Init(&argc, &argv);

    // Assume size is a square number
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 200;            // Number of nodes
    double p = 0.05;           // Edge probability
    int *l, *lc;
    if (rank == 0) {
        const char* ifname = NULL; // Adjacency matrix file name

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

        l = gen_graph(n,p);

        if (ifname)
            write_matrix(ifname,  n, l);

        infinitize(n,l);
        for (int i = 0; i < n*n; i += n+1)
            l[i] = 0;

        block_side = n / size;

        for (int r=1; r<size; r++) {
            MPI_Send(&n, 1, MPI_INT, r, SETUP_STAGE, MPI_COMM_WORLD);     
        }
        // Send column chunks to processors
        for (int r=1; r<size; r++) {
            MPI_Send(l + (r*block_side*n), (n*block_side), MPI_INT, r, INIT_STAGE, MPI_COMM_WORLD);     
        } 

        lc = (int*) malloc( n * block_side * sizeof(int));
        memcpy (lc, l, (n * block_side * sizeof(int)));

    } else {
        MPI_Recv(&n, 1, MPI_INT, 0, SETUP_STAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Assume n is divisible by processors per side
        block_side = n / size;

        lc = (int*) malloc( n * block_side * sizeof(int));
        MPI_Recv(lc, (n * block_side), MPI_INT, 0, INIT_STAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    }

    int* curr_row = (int*) malloc (sizeof(int) * n * block_side);

    // Time the shortest paths code
    double t0 = MPI_Wtime();
    shortest_paths(n, block_side, rank, size, lc, curr_row);
    double t1 = MPI_Wtime();

    printf("\n== MPI with %d processors ==\n", size);
    printf("n:     %d\n", n);
    printf("Time:  %g\n", t1-t0);

    if (rank == 0) {
        for (int r=1; r<size; r++) {
            MPI_Recv(l + (r*block_side*n), (n*block_side), MPI_INT, r, END_STAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } 
        memcpy (l, lc, (n * block_side * sizeof(int)));
    } else {
        MPI_Send(lc, (n * block_side), MPI_INT, 0, END_STAGE, MPI_COMM_WORLD); 
    }

    if (rank == 0) {
        deinfinitize(n, l);
        // Generate output file
        if (ofname)
            write_matrix(ofname, n, l);
            
        printf("\np:     %g\n", p);
        printf("Check: %X\n", fletcher16(l, n*n));
        free(l);
    }

    free(curr_row);
    free(lc);

    MPI_Finalize();
    return 0;
}
