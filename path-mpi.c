#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>
#include "mt19937p.h"
#include <mpi.h>


/*---------------------------------------------------------------------------------------------------
The basic recurrence
----------------------------------------------------------------------------------------------------*/

int square_columns(int n,               // Number of nodes
                   int j_min,
                   int j_max,
                   int* restrict l,     // Partial distance at step s
                   int* restrict lnew)  // Partial distance at step s+1                          
{

	int done = 1;
    	for (int j = j_min; j < j_max; ++j) {

        	for (int i = 0; i < n; ++i) {

            		int lij = l[j*n+i];

            		for (int k = 0; k < n; ++k) {

                		int lik = l[k*n+i];
                		int lkj = l[j*n+k];

                		if (lik + lkj < lij) {
                    			lij = lik+lkj;
                			done = 0;
                		}
            		}

        		lnew[(j-j_min)*n+i] = lij;
        	}
    	}

    	return done;
}

/*--------------------------------------------------------------------------------------------------*/

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


/*--------------------------------------------------------------------------------------------------*/

void shortest_paths(int num_p, int rank, int n, int* restrict l)
{
    	if (rank == 0) {
        	infinitize(n, l);

        	for (int i = 0; i < n*n; i += n+1)
            		l[i] = 0;
    	}

    	// Each processor gets the initial data
    	MPI_Bcast(l, n*n, MPI_INT, 0, MPI_COMM_WORLD);

    	int num_columns = n / num_p;
    	int j_min = num_columns*rank;
    	int j_max = num_columns*(rank+1);

    	int* restrict partial_lnew = (int*) calloc((j_max - j_min)*n, sizeof(int));
    
    	int global_done = 1;
 
	// Here each group of columns is reduced iteratively   
	do {
		// Performing one group of reductions
        	int local_done = square_columns(n, j_min, j_max, l, partial_lnew);
		 		
       		MPI_Allgather(partial_lnew, n*num_columns, MPI_INT, l, n*num_columns, MPI_INT, MPI_COMM_WORLD);

        	// Must reach consensus on completion
        	MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    
    	} while (!global_done);
    
    	free(partial_lnew);

    	if (rank == 0) 
        	deinfinitize(n, l);
    	
}

/*--------------------------------------------------------------------------------------------------*/

void gen_graph(int n, double p, int* l)
{
    	struct mt19937p state;
    	sgenrand(10302011UL, &state);

    	for (int j = 0; j < n; ++j) {

        	for (int i = 0; i < n; ++i)
            		l[j*n+i] = (genrand(&state) < p);
        
		l[j*n+j] = 0;
    	}
}

/*--------------------------------------------------------------------------------------------------*/

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

/*----------------------------------------------------------------------------------------------------
 * # The `main` event
----------------------------------------------------------------------------------------------------*/

const char* usage =
    "path.x -- Parallel all-pairs shortest path on a random graph\n"
    "Flags:\n"
    "  - n -- number of nodes (200)\n"
    "  - p -- probability of including edges (0.05)\n"
    "  - i -- file name where adjacency matrix should be stored (none)\n"
    "  - o -- file name where output matrix should be stored (none)\n";

int main(int argc, char** argv)
{
    	int rank, num_p;

	// Here we initialize the parallel processes    	
	MPI_Init(&argc, &argv);

	// Get the available process in num_p and the rank of each one
    	MPI_Comm_size(MPI_COMM_WORLD, &num_p);
    	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    	int n;              // Number of nodes
    	double p;           // Edge probability
    	const char* ifname; // Adjacency matrix file name
    	const char* ofname; // Distance matrix file name

    	if (rank == 0) {
        	n    = 200;
        	p = 0.05;
        	ifname = NULL;
        	ofname = NULL; 
    
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
    	}

    	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    	int* l = calloc(n*n, sizeof(int));
    	double t0;

	// The only one that is going to generate the graph is process=0    
	if (rank == 0) {
        	// Graph generation + output
        	gen_graph(n, p, l);
        	
		if (ifname)
            		write_matrix(ifname, n, l);
		// The walltime start to be measured when inside process=0 
        	t0 = omp_get_wtime();
    	}
	
	// Each process use this function, and l is broadcasted inside
    	shortest_paths(num_p, rank, n, l);

	// Just the process=0 is going to write the time spent in shortest_paths and the 
	// number of processes
    	if (rank == 0) {
		// The second walltime measurment is after shortest_paths when again in process=0
        	double t1 = omp_get_wtime();

		// Here we modified the printf to get easy access to the output
        	printf("%g           %d\n", num_p, t1-t0);
        	
		// Generate output file
        	if (ofname)
            		write_matrix(ofname, n, l);
    	}

    	// Clean up
    	free(l);
    	
	MPI_Finalize();
    
	return 0;
}
