#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>
#include "mt19937p.h"

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

int rectangle(int n, int sub_size,               // Number of nodes
           int* restrict myblock,     // Partial distance at step s+1
           int* restrict mycol,		// row matrix
		   int* restrict myrow)  // column matrix 
{
    int done = 1;
	for (int b=0; b< n/sub_size; ++b){
		int BA = b*sub_size*sub_size; // Sub Block address
		for (int j = 0; j < sub_size; ++j) {
			for (int i = 0; i < sub_size; ++i) {
				int lij = myblock[j*sub_size+i];
				for (int k = 0; k < n; ++k) {
					int lik = myrow[BA+k*sub_size+i]; //needs to be modified according to how to array is arranged
					int lkj = mycol[BA+j*sub_size+k];
					if (lik + lkj < lij) {
						lij = lik+lkj;
						done = 0;
					}
				}
				lnew[j*sub_size+i] = lij;
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
 * | 7 8 | 14 16 | 
 * ------ -------
 */
void shortest_paths(int n, int* restrict l)
{	// original initialization 
	// Generate l_{ij}^0 from adjacency matrix representation
    infinitize(n, l);
    for (int i = 0; i < n*n; i += n+1)
        l[i] = 0;

	
	// Some serial setup
	nproc = 16; // assume this is a square for now
	n_sub = sqrt(nproc); // ratio of big to small, that is there is ratio^2 sub_grids
	if (n%ratio != 0 ){  \\THROW ERROR}
	int sub_size = n/ratio;
	int* restrict bl = (int*) _mm_malloc(n*n*sizeof(int),64); // blocked version of l where blocks are n/4 * n/4 and
	column_to_block(l,bl,n, sub_size); // implement this function later
	

	
	//MPI Setup
	
	MPI_Init(); // probably wrong
	MPI_Group* col_group= (MPI_Group*) _mm_malloc(n_sub*sizeof(MPI_Group));// group ID of column groups [0; n_sub-1] 
	MPI_Group* row_group= (MPI_Group*) _mm_malloc(n_sub*sizeof(MPI_Group));// group IP of row groups
	MPI_Group World_Group; // group handle of World (main group) 
	
	MPI_Comm* col_comm = (MPI_Group*) _mm_malloc(n_sub*sizeof(MPI_Comm));// communicator of column groups
	MPI_Comm* row_comm = (MPI_Group*) _mm_malloc(n_sub*sizeof(MPI_Comm));// communicator  of row groups
	
	
	int myrank, mycolrank, myrowrank; 
	int ranks[nprocs]; // Ranks in main group
	int* indices[nprocs]; // indices array used to include group
	int* myblock = (int*) _mm_malloc(sub_size*sub_size*sizeof(int),64); // block owned by specific processor
	int* col_buf = (int*) _mm_malloc(n*sub_size*sizeof(int),64); // buffer for the column (size n x sub_size
	int* row_buf = (int*) _mm_malloc(n*sub_size*sizeof(int),64); // buffer for the row (size n x sub_size)

	
	
	MPI_Comm_group(MPI_COMM_WORLD, &World_Group);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	mycolrank = myrank/n_sub;
	myrowrank = myrank%n_sub;
	
	// Include processes into their respective column group
	for(int i=0; i < n_sub; ++i){ 
		for(j=0; j < n*sub; ++j){
			indices[j]= i*n_sub+j; // n_sub consecutive blocks
		}
		MPI_group_incl(World Group, n_sub, indices, col_group[i]); 
		MPI_Comm_create(MPI_COMM_WORLD, col_group[i], col_comm[i]);
	}

	// Include processes into their respective row group
	for(int i=0; i < n_sub; ++i){ 
		for(int j=0; j < n_sub; ++j){
			indices[j]= j*n_sub+i; // every n_sub blocks
		}
		MPI_group_incl(World Group, n_sub, indices, row_group[i]); 
		MPI_Comm_create(MPI_COMM_WORLD, row_group[i], row_comm[i]);
			
	}	
	

	int BA = (mycolrank*n_sub+myrowrank)*sub_size*sub_size; // Block address
	//Copies personal block from blocked matrix
	for (int i=0; i<n_sub; ++i){
		myblock[i]=bl[BA+i];
	}
	 // gather data from column group
	MPI_Allgather( myblock, sub_size*sub_size, MPI_INT, col_buf, sub_size*sub_size*n, MPI_INT, col_comm(mycolrank));
	 // gather data from row group
	MPI_Allgather( myblock, sub_size*sub_size, MPI_INT, row_buf, sub_size*sub_size*n, MPI_INT, row_comm(myrowrank));
	int local_done;
    for (int done = 0; !done; ) {
        local_done = rectangle(n, sub_size, myblock, col_buf, row_buf );
		
		// gather data from column group
		MPI_Allgather( myblock, sub_size*sub_size, MPI_INT, col_buf, sub_size*sub_size*n, MPI_INT, col_comm(mycolrank));
		// gather data from row group
		MPI_Allgather( myblock, sub_size*sub_size, MPI_INT, row_buf, sub_size*sub_size*n, MPI_INT, row_comm(myrowrank));	
		// need to get rid of this barrier: might not be needed
        MPI_Barrier(MPI_COMM_WORLD); 
		// checks if anybody did work this iteration
		MPI_Reduce( &local_done, &done, 1, MPI_INT, MPI_LAND,0, MPI_COMM_WORLD);
    }
	
	block_to_column(l,bl,n, sub_size); // implement this function later
    free(bl);
    deinfinitize(n, l);
}

/*
*
*
*/




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

    printf("== OpenMP with %d threads\n", omp_get_max_threads());
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
