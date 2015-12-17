#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>
#include "mt19937p.h"
#define mpi_ddt MPI_CHAR 

typedef char ddt;
long ddt_upper_range = (1 << (8 * sizeof(ddt) - 2)) - 1;


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



// ==================================================================================
//      Justin: Boardcast algorithm is used here 
// ==================================================================================

int shortest_paths(int nsplit, int nblock, int my_id, ddt* cpd_graph, MPI_Comm col_comm, MPI_Comm row_comm, int col_id, int row_id)
{
    int done = 1;

    ddt* send = malloc(nblock*nblock * sizeof(ddt));
    memcpy(send, cpd_graph, nblock*nblock*sizeof(ddt));

    ddt* colblocks = malloc(nsplit * nblock * nblock * sizeof(ddt));
    ddt* rowblocks = malloc(nsplit * nblock * nblock * sizeof(ddt));

    MPI_Request* colReq = malloc(nsplit * sizeof(MPI_Request));
    MPI_Request* rowReq = malloc(nsplit * sizeof(MPI_Request));

    MPI_Status status;

//    MPI_Allgather(cpd_graph, nblock*nblock, mpi_ddt, colblocks, nblock*nblock, mpi_ddt, col_comm);
//    MPI_Allgather(cpd_graph, nblock*nblock, mpi_ddt, rowblocks, nblock*nblock, mpi_ddt, row_comm);

    if (col_id == 0)
    {
        MPI_Ibcast(send, nblock*nblock, mpi_ddt, 0, col_comm, &colReq[0]);
    }
    else
    {
        MPI_Ibcast(colblocks, nblock*nblock, mpi_ddt, 0, col_comm, &colReq[0]);
    }
    
    if (row_id == 0)
    {
        MPI_Ibcast(send, nblock*nblock, mpi_ddt, 0, row_comm, &rowReq[0]);
    }
    else
    {
        MPI_Ibcast(rowblocks, nblock*nblock, mpi_ddt, 0, row_comm, &rowReq[0]);
    }

    
    for (int i = 0; i < nsplit; i++)
    {
        if (col_id == i+1)
        {
            MPI_Ibcast(send, nblock*nblock, mpi_ddt, i+1, col_comm, &colReq[i+1]);
        }
        else if (i+1 < nsplit)
        {
            MPI_Ibcast(colblocks + (i+1) * nblock*nblock, nblock*nblock, mpi_ddt, i+1, col_comm, &colReq[i+1]);
        }
        
        if (row_id == i+1)
        {
            MPI_Ibcast(send, nblock*nblock, mpi_ddt, i+1, row_comm, &rowReq[i+1]);
        }
        else if (i+1 < nsplit)
        {
            MPI_Ibcast(rowblocks + (i+1) * nblock*nblock, nblock*nblock, mpi_ddt, i+1, row_comm, &rowReq[i+1]);
        }

        MPI_Wait(&colReq[i], &status);
        MPI_Wait(&rowReq[i], &status);

        done = done && square(nblock, colblocks + i * nblock * nblock, rowblocks + i * nblock * nblock, cpd_graph); 
    }

    free(send);
    free(colblocks);
    free(rowblocks);
    free(colReq);
    free(rowReq);

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
    ddt* ori_cpd   = NULL;
    ddt* cpd_graph = NULL;
    
    int nsplit = (int) ( sqrt(num_ranks) + 0.001 );
    int nblock = (int) ( ceil( (double) n / nsplit ) + 0.001 );

    printf("my_id:%d, nblock:%d\n", my_id, nblock);

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
        
        ori_cpd = copy_to_cpd(ori_graph, n, nsplit, nblock);
    }
    
    cpd_graph = malloc(nblock * nblock * sizeof(ddt));

    MPI_Scatter(ori_cpd, nblock*nblock, mpi_ddt, cpd_graph, nblock*nblock, mpi_ddt, 0, MPI_COMM_WORLD);

    // Initial graph received
    
    int my_col = my_id / nsplit;
    int my_row = my_id % nsplit;
    MPI_Comm col_comm;
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_col, my_id, &col_comm); 
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_id, &row_comm); 

    int col_id;
    int row_id;
    MPI_Comm_rank(col_comm, &col_id);
    MPI_Comm_rank(row_comm, &row_id);
    
    // Time the shortest paths code
    double t0 = MPI_Wtime();

    int global_done = 0;
    while(!global_done)
    {
        int done = shortest_paths(nsplit, nblock, my_id, cpd_graph, col_comm, row_comm, col_id, row_id);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&done, &global_done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    }

    double t1 = MPI_Wtime();

    MPI_Gather(cpd_graph, nblock*nblock, mpi_ddt, ori_cpd, nblock*nblock, mpi_ddt, 0, MPI_COMM_WORLD);

    if ( my_id == 0 )
    {
        copy_to_ori(ori_graph, ori_cpd, n, nsplit, nblock);

        printf("=============================\n");
        printf("OpenMP threads: %d\n", omp_get_num_threads());
        printf("=============================\n");
        printf("n:     %d\n", n);
        printf("p:     %g\n", p);
        printf("Time:  %g\n", t1-t0);
        printf("Check: %X\n", fletcher16(ori_graph, n*n));

        // Generate output file
        if (ofname)
            write_matrix(ofname, n, ori_graph);

        free(ori_graph);
        free(ori_cpd);
    }

    free(cpd_graph);

    MPI_Finalize();

    return 0;
}
