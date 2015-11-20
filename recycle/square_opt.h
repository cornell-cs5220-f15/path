#include <stdlib.h>
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32 
#endif


typedef char ddt;
long ddt_upper_range = (1 << (8 * sizeof(ddt) - 1)) - 1;

void show_matrix(int n, ddt* l);

/* Copies the input matrix so that we optimize for cache hits.
*  We assume that the input matrix is arranged in column-major order.
*  arguments:
*       const int num_blocks : number of blocks in one dimension
*       const int M          : length of one column in M. 
*       const double *A      : the matrix in column major order. We assume that it is square.
*
*  output:
*       double *M : a copy of the matrix *A, but blocks are contiguous in memory and each
*                   block is arranged in row-major order. Then blocks are arranged in row major order.
*/
ddt* copy_optimize_rowmajor( const int num_blocks, const int M, const ddt* A )
{
    int out_dim = num_blocks * BLOCK_SIZE;
    ddt* out = ( ddt* ) _mm_malloc( out_dim * out_dim * sizeof( ddt ) , 64);
    int i, j; // specific row, column for the matrix. < M
    int I, J; // specific row, column for the block < num_blocks
    int ii, jj; // specific row, column within the block. < BLOCK_SIZE
    int out_idx;
    int in_idx;
    for( i = 0; i < out_dim; ++i )
    {
        I = i / BLOCK_SIZE;
        ii = i % BLOCK_SIZE;
        for( j = 0; j < out_dim; ++j )
        {
            J = j / BLOCK_SIZE;
            jj = j % BLOCK_SIZE;
            out_idx = ( I * num_blocks + J ) * BLOCK_SIZE * BLOCK_SIZE + ii * BLOCK_SIZE + jj;
            if( i < M && j < M )
            {
                in_idx  = j * M + i;
                out[ out_idx ] = A[ in_idx ];
            }
            else
            {
                out[ out_idx ] = ddt_upper_range;
            }
        }
    }

    return out;
}


ddt* copy_optimize_colmajor( const int num_blocks, const int M, const ddt* A )
{
    int out_dim = num_blocks * BLOCK_SIZE;
    ddt* out = ( ddt* ) _mm_malloc( out_dim * out_dim * sizeof( ddt ) , 64);
    int i, j; // specific row, column for the matrix. < M
    int I, J; // specific row, column for the block < num_blocks
    int ii, jj; // specific row, column within the block. < BLOCK_SIZE
    int out_idx;
    int in_idx;
    for( i = 0; i < out_dim; ++i )
    {
        I = i / BLOCK_SIZE;
        ii = i % BLOCK_SIZE;
        for( j = 0; j < out_dim; ++j )
        {
            J = j / BLOCK_SIZE;
            jj = j % BLOCK_SIZE;
            out_idx = ( J * num_blocks + I ) * BLOCK_SIZE * BLOCK_SIZE + jj * BLOCK_SIZE + ii;
            if( i < M && j < M )
            {
                in_idx  = j * M + i;
                out[ out_idx ] = A[ in_idx ];
            }
            else
            {
                out[ out_idx ] = ddt_upper_range;
            }
        }
    }

    return out;
}


ddt* copy_optimize_mixed( const int num_blocks, const int M, const ddt* A )
{
    int out_dim = num_blocks * BLOCK_SIZE;
    ddt* out = ( ddt* ) _mm_malloc( out_dim * out_dim * sizeof( ddt ) , 64);
    int i, j; // specific row, column for the matrix. < M
    int I, J; // specific row, column for the block < num_blocks
    int ii, jj; // specific row, column within the block. < BLOCK_SIZE
    int out_idx;
    int in_idx;
    for( i = 0; i < out_dim; ++i )
    {
        I = i / BLOCK_SIZE;
        ii = i % BLOCK_SIZE;
        for( j = 0; j < out_dim; ++j )
        {
            J = j / BLOCK_SIZE;
            jj = j % BLOCK_SIZE;
            out_idx = ( I * num_blocks + J ) * BLOCK_SIZE * BLOCK_SIZE + jj * BLOCK_SIZE + ii;
            if( i < M && j < M )
            {
                in_idx  = j * M + i;
                out[ out_idx ] = A[ in_idx ];
            }
            else
            {
                out[ out_idx ] = ddt_upper_range;
            }
        }
    }

    return out;
}
/* Performs a block multiply on block I, J in matrix A and with block K, L in matrix B.
*  A is block row major and B is block column major.
*  C is outputted in block column major order.
*  C = C + AB
*/
void block_multiply_kernel( const int num_blocks, const int M, const int I, const int J, const int K, const ddt* A, const ddt* B, ddt* C, int finished )
{
    int done = 1;

    // index of first element to be multiplied in matrix A
    const int A_idx = ( I * num_blocks + K ) * BLOCK_SIZE * BLOCK_SIZE;

    // index of first element to be multiplied in matrix B
    const int B_idx = ( J * num_blocks + K ) * BLOCK_SIZE * BLOCK_SIZE;

    // index of first element in matrix C.
    const int C_idx = ( I * num_blocks + J ) * BLOCK_SIZE * BLOCK_SIZE;


    const ddt* A_block = A + A_idx;
    const ddt* B_block = B + B_idx;
    ddt* C_block = C + C_idx;
    
    __assume_aligned( A_block, 64 );
    __assume_aligned( B_block, 64 );
    __assume_aligned( C_block, 64 );

    for (int j = 0; j < BLOCK_SIZE; ++j) 
    {
        for (int i = 0; i < BLOCK_SIZE; ++i) 
        {
            ddt lij = C_block[j*BLOCK_SIZE+i];
            for (int k = 0; k < BLOCK_SIZE; ++k) 
            {
                ddt lik = A_block[i*BLOCK_SIZE+k];
                ddt lkj = B_block[j*BLOCK_SIZE+k];
                if (lik + lkj < lij) 
                {
                    lij = lik+lkj;
                    done = 0;
                }
            }
            C_block[j*BLOCK_SIZE+i] = lij;
        }
    }
    
    finished = finished && done;
}



/* Copies matrix A so that it is back to normal column major order. 
*  A is block column major on input.
*/
void copy_normal_mixed( const int num_blocks, const int M, const ddt *A, ddt* out )
{
    int i, j; // specific row, column for the matrix. < M
    int I, J; // specific row, column for the block < num_blocks
    int ii, jj; // specific row, column within the block. < BLOCK_SIZE
    int in_idx;
    int out_idx;
    for( i = 0; i < M; ++i )
    {
        int I = i / BLOCK_SIZE;
        int ii = i % BLOCK_SIZE;
        for( j = 0; j < M; ++j )
        {
            J = j / BLOCK_SIZE;
            jj = j % BLOCK_SIZE;
            out_idx = j * M + i;
            in_idx = ( I * num_blocks + J ) * BLOCK_SIZE * BLOCK_SIZE + jj * BLOCK_SIZE + ii;
            out[ out_idx ] = A[ in_idx ];
        }
    }
}

int square_opt(const int n, ddt* l, ddt* lnew)
{
    int done = 1;
    
    const int num_blocks = n / BLOCK_SIZE + (int) ( n % BLOCK_SIZE != 0 );
    ddt* A_copied = copy_optimize_rowmajor( num_blocks, n, l );
    ddt* B_copied = copy_optimize_colmajor( num_blocks, n, l );
    ddt* C_copied = copy_optimize_mixed( num_blocks, n, lnew );

    int I, J, K;
    #pragma omp parallel for shared(A_copied, B_copied, lnew) reduction(&& : done)
    for( I = 0; I < num_blocks; ++I ) 
    {
        for( J = 0; J < num_blocks; ++J )
        {
            for( K = 0; K < num_blocks; ++K )
            {
                block_multiply_kernel( num_blocks, n, I, J, K, A_copied, B_copied, C_copied, done );
            }
        }
    }

    copy_normal_mixed( num_blocks, n, C_copied, lnew );

    _mm_free(A_copied);
    _mm_free(B_copied);
    _mm_free(C_copied);

    return done;
}

