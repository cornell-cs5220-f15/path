#include <getopt.h>
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
           int start,
           int numRows,
           int* restrict l,     // Partial distance at step s
           int* restrict lnew, // Partial distance at step s+1
		   int* lp,
		   int sp,
		   int lplen)  
{
    int done = 1;
    for (int i = 0; i <numRows ; ++i) {
        for (int j = 0; j < lplen; ++j) {
            int lij = lnew[i*n+j+sp];
            for (int k = 0; k < n; ++k) {
                int lik = l[i*n + k];
				int lkj = lp[j*n + k];
                if (lik + lkj < lnew[j*n+i]) {
                    lnew[j*n+i] = lik+lkj;
                    done = 0;
                }
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

void shortest_paths(int n, int* restrict l, int size, int rank)
{
	int sizework = size-1;
	int masterrank =sizework;
    int* restrict intervals = (int*) calloc(size, sizeof(int));
    int* restrict displacements = (int*) calloc(size, sizeof(int));
    int numRows = n/sizework;
    int extraRows = n%sizework;
    MPI_Request request;
	MPI_Status status;
    // divide up the work amonst all processes
    if (rank==masterrank) {
        displacements[0] = 0;
        for (int i = 0; i < sizework-1; i++) {
            if (i < extraRows)
                intervals[i] = (numRows+1)*n;
            else
                intervals[i] = numRows*n;
            displacements[i+1] = displacements[i] + intervals[i];
        }
        intervals[sizework-1] = numRows*n;
    }

    MPI_Bcast(intervals, size, MPI_INT, masterrank, MPI_COMM_WORLD);
    MPI_Bcast(displacements, size, MPI_INT, masterrank, MPI_COMM_WORLD);
	int* restrict lnew ;
	if(rank!=masterrank){
		lnew = (int*) calloc(intervals[rank], sizeof(int));
		//memcpy(lnew, l + displacements[rank], intervals[rank] * sizeof(int));
	}
	else{
		lnew = (int*) calloc(n*n, sizeof(int));
	}
	int* restrict lp = (int*) calloc(n*n, sizeof(int));
    MPI_Scatterv(l, intervals, displacements, MPI_INT, lnew, intervals[rank], MPI_INT, masterrank, MPI_COMM_WORLD);

//    MPI_Bcast(l, n*n, MPI_INT, 0, MPI_COMM_WORLD);
    for (int done = 0; !done; ) {
		int notdoneLocal=0;
		if(rank!=masterrank){
			memcpy(l,lnew,intervals[rank] * sizeof(int));
		}
		else{
			for(int j=0;j<n;++j){
				for(int i=0;i<intervals[0]/n;++i){
					lp[j+i*n]=l[i+j*n];
				}
			}
		}
		MPI_Bcast(lp, intervals[0], MPI_INT, masterrank, MPI_COMM_WORLD);
		for(int p=1;p<sizework;++p){
			if(rank==masterrank){
				for(int j=0;j<n;++j){
					for(int i=displacements[p]/n;i<displacements[p]/n+intervals[p]/n;++i){
						lp[j+i*n]=l[i+j*n];
					}
				}
				MPI_Ibcast(lp+displacements[p], intervals[p], MPI_INT, masterrank, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
			}
			else{
				MPI_Ibcast(lp+displacements[p], intervals[p], MPI_INT, masterrank, MPI_COMM_WORLD, &request);
				
				notdoneLocal += 1-square(n, displacements[rank], intervals[rank]/n, l, lnew,lp+displacements[p-1],displacements[p-1]/n,intervals[p-1]/n);
				MPI_Wait(&request, &status);
			}
		}
		if(rank!=masterrank){
			notdoneLocal += 1-square(n, displacements[rank], intervals[rank]/n, l, lnew,lp+displacements[sizework-1],displacements[sizework-1]/n,intervals[sizework-1]/n);
		}
		MPI_Gatherv(lnew, intervals[rank], MPI_INT, l, intervals, displacements, MPI_INT, masterrank, MPI_COMM_WORLD);
		MPI_Allreduce(&notdoneLocal, &done, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        done= !done;
    }

    free(lnew);
    if (rank == masterrank)
        deinfinitize(n, l);
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
    int rank, size;
    int* l;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    if (rank == size-1) {
        // Graph generation + output
        l = gen_graph(n, p);
        if (ifname)
            write_matrix(ifname,  n, l);
        // Generate l_{ij}^0 from adjacency matrix representation
        infinitize(n, l);
        for (int i = 0; i < n*n; i += n+1)
            l[i] = 0;
    } else {
        l = calloc(n*n, sizeof(int));
    }

    

    // Time the shortest paths code
    double t0 = MPI_Wtime();
    shortest_paths(n, l, size, rank);
    double t1 = MPI_Wtime();

    if (rank == size-1) {
        printf("== MPI with %d processes\n", size);
        printf("n:     %d\n", n);
        printf("p:     %g\n", p);
        printf("Time:  %g\n", t1-t0);
        printf("Check: %X\n", fletcher16(l, n*n));

        // Generate output file
        if (ofname)
            write_matrix(ofname, n, l);
    }

    // Clean up
    free(l);
    MPI_Finalize();
    return 0;
}

