#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>
#include "mt19937p.h"
#define mpi_ddt MPI_INT

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

typedef int  ddt;
//long ddt_upper_range = (1 << (8 * sizeof(ddt) - 2)) - 1;
long ddt_upper_range = 9; 


// ========================================================================== 
//          Justin: this should be done as matrix multiplication
// ==========================================================================

int square(int nblock, ddt* restrict col, ddt* restrict row, ddt* restrict cpd_new) 
{
    int done = 1;

    ddt* row_T = malloc(nblock*nblock * sizeof(ddt));

    for (int j = 0; j < nblock; j++)
        for(int i = 0; i < nblock; i++)
        {
            row_T[j*nblock + i] = row[i*nblock + j];
        }

    #pragma omp parallel for shared(col, row_T, cpd_new) reduction(&& : done)
    for (int j = 0; j < nblock; ++j) 
    {
        for (int i = 0; i < nblock; ++i) 
        {
            int cpd_ij = cpd_new[j*nblock+i];
            for (int k = 0; k < nblock; ++k) 
            {
                // ===== Justin: addition now vectorized =====
                int cpd_ik = row_T[i*nblock+k];
                // ===========================================
                int cpd_kj = col[j*nblock+k];
                if (cpd_ik + cpd_kj < cpd_ij) 
                {
                    cpd_ij = cpd_ik+cpd_kj;
                    done = 0;
                }
            }
            cpd_new[j*nblock+i] = cpd_ij;
        }
    }

    free(row_T);

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

static inline void infinitize(int n, ddt* l)
{
    for (int i = 0; i < n*n; ++i)
        if (l[i] == 0)
            l[i] = ddt_upper_range; 
}

static inline void deinfinitize(int n, ddt* l)
{
    for (int i = 0; i < n*n; ++i)
        if (l[i] == ddt_upper_range)
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



// ==================================================================================
//      Justin: Since the totient is 2d torus, we need to use cannon's algorithm
// ==================================================================================

int shortest_paths(int nsplit, int nblock, int my_id, ddt* cpd_graph, int* comm_matrix)
{
    int done = 1;

    ddt* colblocks = malloc(nsplit * nblock * nblock * sizeof(ddt));
    ddt* rowblocks = malloc(nsplit * nblock * nblock * sizeof(ddt));
    MPI_Status status;

    for (int j = 0; j < nsplit; j++)
    {
        if (my_id / nsplit == j)
        {
            for (int i = 0; i < nsplit; i++)                                        // sending block_{ij}
            {
                if (my_id % nsplit == i)                                            // I have the block
                {
                    for (int itr = 0; itr < nsplit; itr++)                          // send to everyone in the col 
                    {
                        if (itr == i)                                               // self communication
                        {
                            memcpy(colblocks + i*nblock*nblock, cpd_graph, nblock*nblock*sizeof(ddt));
                        }
                        else
                        {
                            MPI_Send(cpd_graph, nblock*nblock, mpi_ddt, j*nsplit+itr, i, MPI_COMM_WORLD);
                        }
                    }
                }
                else
                {
                    MPI_Recv(colblocks + i*nblock*nblock, nblock*nblock, mpi_ddt, j*nsplit+i, i, MPI_COMM_WORLD, &status);
                }
            }
        } 
    }    

    for (int i = 0; i < nsplit; i++)
    {
        if (my_id % nsplit == i)
        {
            for (int j = 0; j < nsplit; j++)                                        // sending block_{ij}
            {
                if (my_id / nsplit == j)                                            // I have the block
                {
                    for (int itr = 0; itr < nsplit; itr++)                          // send to everyone in the row
                    {
                        if (itr == j)                                               // self communication
                        {
                            memcpy(rowblocks + j*nblock*nblock, cpd_graph, nblock*nblock*sizeof(ddt));
                        }
                        else
                        {
                            MPI_Send(cpd_graph, nblock*nblock, mpi_ddt, itr*nsplit+i, j, MPI_COMM_WORLD);
                        }
                    }
                }
                else
                {
                    MPI_Recv(rowblocks + j*nblock*nblock, nblock*nblock, mpi_ddt, j*nsplit+i, j, MPI_COMM_WORLD, &status);
                }
            }
        } 
    }    

    for (int i = 0; i < nsplit; i++)
    {
        done = done && square(nblock, colblocks + i * nblock * nblock, rowblocks + i * nblock * nblock, cpd_graph); 
    }

    free(rowblocks);
    free(colblocks);

    return done;
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

ddt* gen_graph(int n, double p)
{
    ddt* l = malloc(n*n * sizeof(ddt));
    struct mt19937p state;
    sgenrand(10302011UL, &state);
    for (int j = 0; j < n; ++j) 
    {
        for (int i = 0; i < n; ++i)
            l[j*n+i] = (genrand(&state) < p);
        l[j*n+j] = 0;
    }
    return l;
}



ddt* copy_to_cpd(ddt* ori_graph, int n, int nsplit, int nblock)
{
    int nside = nsplit * nblock; 
    ddt* cpd_graph = malloc(nside*nside * sizeof(ddt));
    
    for (int j = 0; j < nside; j++)
    {
        int  J = j / nblock;
        int jj = j % nblock;
        for (int i = 0; i < nside; i++)
        {
            int  I = i / nblock; 
            int ii = i % nblock;

            int cpd_offset = (J * nsplit + I) * (nblock*nblock) + (jj * nblock + ii);
            
            if ( i >= n || j >= n )
            {
                cpd_graph[cpd_offset] = ddt_upper_range;
            }
            else
            {
                int ori_offset = j * n + i;
                cpd_graph[cpd_offset] = ori_graph[ori_offset];   
            }
        }
    }

    return cpd_graph;
}


void copy_to_ori(ddt* ori_graph, ddt* cpd_graph, int n, int nsplit, int nblock)
{
    int nside = nsplit * nblock; 
    
    for (int i = 0; i < n; i++)
    {
        int  I = i / nblock; 
        int ii = i % nblock;
        for (int j = 0; j < n; j++)
        {
            int  J = j / nblock;
            int jj = j % nblock;

            int ori_offset = j * n + i;
            int cpd_offset = (J * nsplit + I) * (nblock*nblock) + (jj * nblock + ii);

            ori_graph[ori_offset] = cpd_graph[cpd_offset];   
        }
    }
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

int fletcher16(ddt* data, int count)
{
    int sum1 = 0;
    int sum2 = 0;
    for(int index = 0; index < count; ++index) 
    {
          sum1 = (sum1 + data[index]) % 255;
          sum2 = (sum2 + sum1) % 255;
    }
    return (sum2 << 8) | sum1;
}

void write_matrix(const char* fname, int n, ddt* a)
{
    FILE* fp = fopen(fname, "w+");
    if (fp == NULL) 
    {
        fprintf(stderr, "Could not open output file: %s\n", fname);
        exit(-1);
    }
    for (int i = 0; i < n; ++i) 
    {
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
    MPI_Init(&argc, &argv);
    
    int num_ranks;
    int my_id;

    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    
    int n    = 200;            // Number of nodes
    double p = 0.05;           // Edge probability
    const char* ifname = NULL; // Adjacency matrix file name
    const char* ofname = NULL; // Distance matrix file name

    // Option processing
    extern char* optarg;
    const char* optstring = "hn:d:p:o:i:";
    int c;
    while ((c = getopt(argc, argv, optstring)) != -1) 
    {
        switch (c) 
        {
            case 'h':
                fprintf(stderr, "%s", usage);
                return -1;
            case 'n': n = atoi(optarg); break;
            case 'p': p = atof(optarg); break;
            case 'o': ofname = optarg;  break;
            case 'i': ifname = optarg;  break;
        }
    }

    ddt* ori_graph = NULL;
    ddt* cpd_graph = NULL;
    
    int nsplit = (int) ( sqrt(num_ranks) + 0.001 );
    int nblock = (int) ( ceil( (double) n / nsplit ) + 0.001 );

    printf("nblock:%d", nblock);

    if ( my_id == 0 )
    {
        // Graph generation + output
        ori_graph = gen_graph(n, p);
        if (ifname)
            write_matrix(ifname, n, ori_graph);
    
        // Generate ori_graph 
        infinitize(n, ori_graph);
        for (int i = 0; i < n*n; i += n+1)
            ori_graph[i] = 0;
        
//        printf("%d: ", my_id);
//        for (int i = 0; i < n*n; i++)
//            printf("%d ", ori_graph[i]);
//        printf("\n");
        
        cpd_graph = copy_to_cpd(ori_graph, n, nsplit, nblock);

        for (int i = 1; i < nsplit * nsplit; i++)
        {
            MPI_Send(cpd_graph + i * nblock*nblock, nblock*nblock, mpi_ddt, i, i, MPI_COMM_WORLD);
        }

//        printf("%d: ", my_id);
//        for (int i = 0; i < nblock*nblock; i++)
//            printf("%d ", cpd_graph[i]);
//        printf("\n");
    }
    else
    {
        cpd_graph = malloc(nblock * nblock * sizeof(ddt));

        MPI_Status status;

        MPI_Recv(cpd_graph, nblock*nblock, mpi_ddt, 0, my_id, MPI_COMM_WORLD, &status);

//        printf("%d: ", my_id);
//        for (int i = 0; i < nblock*nblock; i++)
//            printf("%d ", cpd_graph[i]);
//        printf("\n");
    }

    // Initial graph received
    
   
   

    int* comm_matrix = malloc(nsplit * nsplit * sizeof(int));

    for (int j = 0; j < nsplit; j++)
    {
        for (int i = 0; i < nsplit; i++)
        {
            comm_matrix[j * nsplit + i] = j * nsplit + i;
        }
    }

    // Time the shortest paths code
    double t0 = MPI_Wtime();

    int global_done = 0;
    while(!global_done)
    {
        int done = shortest_paths(nsplit, nblock, my_id, cpd_graph, comm_matrix);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&done, &global_done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    }

    double t1 = MPI_Wtime();

    if ( my_id != 0 )
    {
        MPI_Send(cpd_graph, nblock*nblock, mpi_ddt, 0, my_id, MPI_COMM_WORLD);

//        printf("%d: ", my_id);
//        for (int i = 0; i < nblock*nblock; i++)
//            printf("%d ", cpd_graph[i]);
//        printf("\n");
    }
    else
    {
        MPI_Status status;

        for (int i = 1; i < nsplit * nsplit; i++)
            MPI_Recv(cpd_graph + i * nblock*nblock, nblock*nblock, mpi_ddt, i, i, MPI_COMM_WORLD, &status);

//        printf("%d: ", my_id);
//        for (int i = 0; i < nblock*nblock * nsplit*nsplit; i++)
//            printf("%d ", cpd_graph[i]);
//        printf("\n");

        copy_to_ori(ori_graph, cpd_graph, n, nsplit, nblock);

        printf("=============================\n");
        printf("n:     %d\n", n);
        printf("p:     %g\n", p);
        printf("Time:  %g\n", t1-t0);
        printf("Check: %X\n", fletcher16(ori_graph, n*n));

        // Generate output file
        if (ofname)
            write_matrix(ofname, n, ori_graph);

        free(ori_graph);
    }

    free(cpd_graph);

    MPI_Finalize();

    return 0;
}
