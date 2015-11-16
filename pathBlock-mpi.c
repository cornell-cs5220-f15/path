//
//  pathBlock-mpi.c
//  
//
//  Created by Marc Gilles on 11/16/15.
//
//

#include "pathBlock-mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>
#include "mt19937p.h"
#include <assert.h>
#include <omp.h>
#include <math.h>

#ifndef SMALL_BLOCK_SIZE
#define SMALL_BLOCK_SIZE ((int) 128)
#endif
#ifndef BYTE_ALIGNMENT
#define BYTE_ALIGNMENT ((int) 64)
#endif


//ldoc on
/**
 * # The basic recurrence
 *
 * At the heart of the method is the following basic recurrence.
 * If $l_{ij}^s$ represents the length of the shortest path from
 * $i$ to $j$ that can be attained in at most $2^s$ steps, then
 * $$
 *   l_{ij}^{s+1} = \min_k \{ l_{ik}^s + l_{kj}^2 \}.
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

// new blocked implementation



int basic_square(const int *restrict l1, const int *restrict l2, int *restrict l3){
    int done=1;
    for (int j = 0; j < SMALL_BLOCK_SIZE; ++j) {
        for (int i = 0; i < SMALL_BLOCK_SIZE; ++i) {
            int lij = l3[j*SMALL_BLOCK_SIZE+i];
            for (int k = 0; k < SMALL_BLOCK_SIZE; ++k) {
                int lik = l1[k*SMALL_BLOCK_SIZE+i];
                int lkj = l2[j*SMALL_BLOCK_SIZE+k];
                if (lik + lkj < lij) {
                    lij = lik+lkj;
                    done = 0;
                }
            }
            l3[j*SMALL_BLOCK_SIZE+i] = lij;
        }
    }
    return done;
}


int do_block(const int n,
             const int block_size,
             const int bigb_address,
             const int *restrict row_buf,
             const int *restrict col_buf,
             int *restrict myblock,
             const int *restrict row_smallblock, // sub block array
             const int *restrict col_smallblock,
             int *restrict mysmallblock,
             int i, int j, int k){
    for(int kk=0; kk<SMALL_BLOCK_SIZE; ++kk){
        // copies column by column ?
        memcpy((void *) (row_smallblock + (kk * SMALL_BLOCK_SIZE)), (const void *) (row_buf + bigb_address+ i + (k + kk)*block_size), SMALL_BLOCK_SIZE * sizeof(int));
    }
    for(int jj=0; jj<SMALL_BLOCK_SIZE; ++jj){
        memcpy((void *) (col_smallblock + (jj * SMALL_BLOCK_SIZE)), (const void *) (col_buf + bigb_address + k + (j + jj)*block_size), SMALL_BLOCK_SIZE * sizeof(int));
        memcpy((void *) (mysmallblock + (jj * SMALL_BLOCK_SIZE)), (const void *) (myblock + i + (j + jj)*block_size), SMALL_BLOCK_SIZE * sizeof(int));
    }
    
    int done = basic_square(row_smallblock, col_smallblock, mysmallblock);
    /* not needed in this implementation?:
    if(!done)
        for(int jj=0; jj<SMALL_BLOCK_SIZE; ++jj){
            //memcpy((void *) (lnew + i + (j + jj)*n), (const void *) (l3 + (jj * SMALL_BLOCK_SIZE)), SMALL_BLOCK_SIZE * sizeof(int));
        }
    */
    return done;
}



int square(const int n,   // Number of nodes
           const int block_size, // big block size
           int* restrict row_buf,     // col buffer
           int* restrict col_buf,     // row buffer
           int* restrict myblock)  // block buffer
{
    int done = 1;
    // block from row array, dimension is blocksize*n
    const int *row_smallblock = (int *) _mm_malloc(SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE * sizeof(int), BYTE_ALIGNMENT);
    // block from column array, dimension is n*blocksize
    const int *col_smallblock = (int *) _mm_malloc(SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE * sizeof(int), BYTE_ALIGNMENT);
    // new block
    int * mysmallblock= (int *) _mm_malloc(SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE * sizeof(int), BYTE_ALIGNMENT);
    const int n_small_blocks = block_size / SMALL_BLOCK_SIZE;
    
    for (int bigb=0; b< n/(block_size*SMALL_BLOCK_SIZE); ++b){ // big block iteration
        bigb_address = bigb*block_size*block_size;
        for (int bj = 0; bj < n_small_blocks; ++bj){
            const int j = bj * SMALL_BLOCK_SIZE;
            for (int bi = 0; bi < n_small_blocks; ++bi){
                const int i = bi * SMALL_BLOCK_SIZE;
                for (int bk = 0; bk < n_small_blocks; ++bk){
                    const int k = bk * SMALL_BLOCK_SIZE;
                    if(!do_block(n, block_size, bigb_address, row_buf, col_buf, myblock, row_smallblock, col_smallblock, mysmallblock, i, j, k)) done = 0; // need to change this
                }
            }
        }
    }
    
    _mm_free((void *) l1);
    _mm_free((void *) l2);
    _mm_free((void *) l3);   
    
    return done;        
}


// old implementation:

int rectangle(int n, int block_size,               // Number of nodes
              int* restrict myblock,     // Partial distance at step s+1
              int*  mycol,		// row matrix
              int*  myrow)  // column matrix
{
    int done = 1;
    for (int b=0; b< n/block_size; ++b){
        int BA = b*block_size*block_size; // Sub Block address
        for (int j = 0; j < block_size; ++j) {
            for (int i = 0; i < block_size; ++i) {
                int lij = myblock[j*block_size+i];
                for (int k = 0; k < block_size; ++k) {
                    int lik = myrow[BA+k*block_size +i];
                    int lkj = mycol[BA+j*block_size +k];
                    if (lik + lkj < lij) {
                        lij = lik+lkj;
                        done = 0;
                    }
                }
                myblock[j*block_size+i] = lij;
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


/*
 * Reindexes matrix from column major to bock column major with column major blocks
 * n is the dimension of the square matrix
 * block_size is the size of the blocks
 */
void column_to_block(int* old, int* blocked, int n, int block_size){
    int nblock = n/block_size; // number of blocks
    int bi,bj,i,j;
    
    for(bi=0; bi < nblock; ++bi){
        for(bj=0; bj < nblock; ++bj){
            for(i=0; i < block_size; ++i){
                for(j=0; j < block_size; ++j){
                    blocked[(bj * nblock + bi) * block_size * block_size + j*block_size+i]= //
                    old[(j + bj * block_size) * n + bi * block_size + i];
                    
                }
            }
        }
    }
}

/*
 * Reindexes matrix from ock column major with column major blocks to column major
 * n is the dimension of the square matrix
 * block_size is the size of the blocks
 */
void block_to_column(int* old, int* blocked, int n, int block_size){
    int nblock = n/block_size; // number of blocks
    int bi,bj,i,j;
    
    for(bi=0; bi < nblock; ++bi){
        for(bj=0; bj < nblock; ++bj){
            for(i=0; i < block_size; ++i){
                for(j=0; j < block_size; ++j){
                    old[(j + bj * block_size) * n + bi * block_size + i]= //;
                    blocked[(bj * nblock + bi) * block_size * block_size + j*block_size+i];
                }
            }
        }
    }
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

/*
 * Assumes the blocked is in this format: (column major with column major blocks)
 * ------ -------
 * | 1 3 |  9 10 |
 * | 2 4 | 11 12 |
 * ------  ------
 * | 5 7 | 13 15 |
 * | 6 8 | 14 16 |
 * ------ -------
 */
void shortest_paths(int n, int* restrict l, int argc, char** argv)
{
    
    // original initialization
    // Generate l_{ij}^0 from adjacency matrix representation
    infinitize(n, l);
    for (int i = 0; i < n*n; i += n+1){
        l[i] = 0;
    }
    
    // Some serial setup
    
    int nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);  // assume this is a square for now
    int n_block = sqrt(nproc); // ratio of big to small, that is there is ratio^2 sub_grids
    assert(n%n_block == 0 ); // throws error is n is not divisible by ratio, need to change this later
    int block_size = n/n_block;
    int* restrict bl = (int*) _mm_malloc(n*n*sizeof(int),64); // blocked version of l where blocks are n/4 * n/4 and
    column_to_block(l,bl,n, block_size);
    
    
    
    //MPI Setup
    
    MPI_Group* col_group= (MPI_Group*) _mm_malloc(n_block*sizeof(MPI_Group),64);// group ID of column groups
    MPI_Group* row_group= (MPI_Group*) _mm_malloc(n_block*sizeof(MPI_Group),64);// group ID of row groups
    MPI_Group World_Group; // group handle of World (main group)
    
    MPI_Comm* col_comm = (MPI_Comm*) _mm_malloc(n_block*sizeof(MPI_Comm),64);// communicator of column groups
    MPI_Comm* row_comm = (MPI_Comm*) _mm_malloc(n_block*sizeof(MPI_Comm),64);// communicator  of row groups
    
    
    int mycolrank, myrowrank;
    int ranks[nproc]; // Ranks in main group
    int indices[nproc] ; // indices array used to include group
    int* myblock = (int*) _mm_malloc(block_size*block_size*sizeof(int),64); // block owned by specific processor
    int* col_buf = (int*) _mm_malloc(n_block*block_size*block_size*sizeof(int),64); // buffer for the column (size n x block_size
    int* row_buf = (int*) _mm_malloc(n_block*block_size*block_size*sizeof(int),64); // buffer for the row (size n x block_size)
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    MPI_Comm_group(MPI_COMM_WORLD, &World_Group);
    
    mycolrank = myrank/n_block;
    myrowrank = myrank%n_block;
    
    // Include processes into their respective column group
    for(int i=0; i < n_block; ++i){
        for(int j=0; j < n_block; ++j){
            indices[j]= i*n_block+j; // n_block consecutive blocks
        }
        MPI_Group_incl(World_Group, n_block, indices, &col_group[i]);
        MPI_Comm_create(MPI_COMM_WORLD, col_group[i], &col_comm[i]);
    }
    
    // Include processes into their respective row group
    for(int i=0; i < n_block; ++i){
        for(int j=0; j < n_block; ++j){
            indices[j]= j*n_block+i; // every n_block blocks
        }
        MPI_Group_incl(World_Group, n_block, indices, &row_group[i]);
        MPI_Comm_create(MPI_COMM_WORLD, row_group[i], &row_comm[i]);
        
    }
    
    
    int BA = (mycolrank*n_block+myrowrank)*block_size*block_size; // Block address
    //Copies personal block from blocked matrix
    for (int i=0; i<block_size*block_size; ++i){
        assert(bl[BA+i] >=0 );
        myblock[i]=bl[BA+i];
    }
    int colid, rowid;
    MPI_Comm_rank( row_comm[myrowrank], &rowid);
    MPI_Comm_rank( col_comm[mycolrank], &colid);
    
    //printf("processor %d, row group: %d, row ID: %d, rank group: %d, rank ID: %d \n", myrank, myrowrank, rowid, mycolrank, colid);
    
    
    // gather data from column group
    MPI_Allgather( myblock, block_size*block_size, MPI_INT, col_buf, block_size*block_size, MPI_INT, col_comm[mycolrank]);
    // gather data from row group
    MPI_Allgather( myblock, block_size*block_size, MPI_INT, row_buf, block_size*block_size, MPI_INT, row_comm[myrowrank]);
    
    int local_done;
    int term = 2*log2(n);
    int done = 0;
    int iter = 0;
    for (int done = 0; !done ;  ) {
        local_done = square(n, block_size, myblock, row_buf, col_buf );
        
        // gather data from column group
        MPI_Allgather( myblock, block_size*block_size, MPI_INT, col_buf, block_size*block_size, MPI_INT, col_comm[mycolrank]);
        // gather data from row group
        MPI_Allgather( myblock, block_size*block_size, MPI_INT, row_buf, block_size*block_size, MPI_INT, row_comm[myrowrank]);
        
        // need to get rid of this barrier: might not be needed
        MPI_Barrier(MPI_COMM_WORLD);
        // checks if anybody did work this iteration
        MPI_Allreduce( &local_done, &done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        
        iter = iter+1;
    }
    for (int i=0; i<block_size*block_size; ++i){
        bl[BA+i]=myblock[i];
    }
    MPI_Gather(myblock, block_size*block_size, MPI_INT, bl, block_size*block_size, MPI_INT,0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank==0){
        block_to_column(l,bl,n, block_size);
        //	free(bl); // for some reason this makes the program crash
        deinfinitize(n, l);
    }
    
    
    
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
    const char* ifname = "adj"; // Adjacency matrix file name
    const char* ofname = "dist"; // Distance matrix file name
    
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
    MPI_Init(&argc, &argv);
    double t0 = MPI_Wtime();
    shortest_paths(n, l, argc, argv);
    double t1 = MPI_Wtime();
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if(myrank==0) {
        int threads;
        MPI_Comm_size(MPI_COMM_WORLD, &threads);  
        printf("== MPI with %d threads\n", threads);
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
